from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from drone_interceptor.dynamics.kalman import kalman_predict, kalman_update
from drone_interceptor.types import Detection, TargetState


class KinematicTargetTracker:
    """Single-target finite-difference tracker for lightweight fallback operation."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config["tracking"]
        self._dt = float(config["mission"]["time_step"])
        self._last_position: np.ndarray | None = None
        self._last_velocity: np.ndarray | None = None

    def update(self, detection: Detection) -> TargetState:
        if self._last_position is None:
            velocity = np.zeros_like(detection.position)
            acceleration = np.zeros_like(detection.position)
        else:
            velocity = (detection.position - self._last_position) / self._dt
            if self._last_velocity is None:
                acceleration = np.zeros_like(detection.position)
            else:
                acceleration = (velocity - self._last_velocity) / self._dt

        self._last_position = detection.position.copy()
        self._last_velocity = velocity.copy()
        covariance = np.eye(3) * float(self._config["measurement_noise"])
        return TargetState(
            position=detection.position.copy(),
            velocity=velocity,
            acceleration=acceleration,
            covariance=covariance,
            timestamp=detection.timestamp,
            track_id=str(detection.metadata.get("track_id", "sim-target")),
            metadata=dict(detection.metadata),
        )


class KalmanTargetTracker:
    """Planar Kalman tracker with explicit Gaussian process and measurement noise."""

    def __init__(self, config: dict[str, Any]) -> None:
        tracking = config["tracking"]
        mission = config["mission"]
        self._default_dt = float(mission["time_step"])
        self._process_noise_std = float(tracking.get("process_noise", 0.15))
        self._measurement_noise_std = float(tracking.get("measurement_noise", 0.5))
        self._initial_velocity_std = float(tracking.get("initial_velocity_std", 6.0))
        self._max_target_acceleration = float(tracking.get("max_target_acceleration_mps2", 4.0))
        self._acceleration_smoothing = float(tracking.get("acceleration_smoothing", 0.7))
        self._velocity_measurement_blend = float(np.clip(tracking.get("velocity_measurement_blend", 0.55), 0.0, 1.0))
        self._max_velocity_residual_mps = float(tracking.get("max_velocity_residual_mps", 3.0))
        self._state = np.zeros(4, dtype=float)
        self._covariance = np.eye(4, dtype=float)
        self._initialized = False
        self._last_timestamp: float | None = None
        self._last_altitude: float | None = None
        self._last_vertical_velocity = 0.0
        self._last_planar_velocity = np.zeros(2, dtype=float)
        self._last_planar_acceleration = np.zeros(2, dtype=float)

    def update(self, detection: Detection) -> TargetState:
        measurement_position = np.asarray(detection.position, dtype=float).reshape(-1)
        if measurement_position.shape not in {(2,), (3,)}:
            raise ValueError("Detection position must be 2D or 3D for Kalman tracking.")

        measurement_xy = measurement_position[:2]
        altitude = float(measurement_position[2]) if measurement_position.shape == (3,) else 0.0
        dt = self._resolve_dt(detection.timestamp)
        measurement_noise = np.eye(2, dtype=float) * max(self._measurement_noise_std**2, 1e-6)

        if not self._initialized:
            self._state[:2] = measurement_xy
            self._state[2:] = np.asarray(detection.metadata.get("measured_velocity_xy", [0.0, 0.0]), dtype=float)
            self._covariance = np.diag(
                [
                    self._measurement_noise_std**2,
                    self._measurement_noise_std**2,
                    self._initial_velocity_std**2,
                    self._initial_velocity_std**2,
                ]
            ).astype(float)
            kalman_gain = np.zeros((4, 2), dtype=float)
            innovation = np.zeros(2, dtype=float)
            mahalanobis_distance = 0.0  # No innovation for first measurement
            innovation_covariance = measurement_noise.copy()
            self._initialized = True
        else:
            control = np.asarray(detection.metadata.get("measured_acceleration_xy", [0.0, 0.0]), dtype=float)
            process_noise = self._build_process_noise(dt)
            predicted_state, predicted_covariance = kalman_predict(
                state=self._state,
                covariance=self._covariance,
                acceleration=control,
                dt=dt,
                process_noise=process_noise,
            )
            innovation = measurement_xy - predicted_state[:2]
            updated_state, updated_covariance, kalman_gain = kalman_update(
                predicted_state=predicted_state,
                predicted_covariance=predicted_covariance,
                measurement=measurement_xy,
                measurement_noise=measurement_noise,
            )
            # Calculate innovation covariance and mahalanobis distance
            innovation_covariance = predicted_covariance[:2, :2] + measurement_noise
            mahalanobis_distance = float(innovation.T @ np.linalg.inv(innovation_covariance) @ innovation)
            measured_velocity_xy = np.asarray(detection.metadata.get("measured_velocity_xy", updated_state[2:]), dtype=float)
            if measured_velocity_xy.shape == (2,):
                residual = _clip_planar_norm(measured_velocity_xy - updated_state[2:], self._max_velocity_residual_mps)
                updated_state[2:] = updated_state[2:] + (self._velocity_measurement_blend * residual)
            self._state = updated_state
            self._covariance = updated_covariance

        planar_velocity = self._state[2:].copy()
        raw_planar_acceleration = (planar_velocity - self._last_planar_velocity) / max(dt, 1e-6)
        clipped_planar_acceleration = _clip_planar_norm(raw_planar_acceleration, self._max_target_acceleration)
        planar_acceleration = (
            self._acceleration_smoothing * self._last_planar_acceleration
            + (1.0 - self._acceleration_smoothing) * clipped_planar_acceleration
        )
        if self._last_altitude is None:
            vertical_velocity = 0.0
            vertical_acceleration = 0.0
        else:
            vertical_velocity = (altitude - self._last_altitude) / max(dt, 1e-6)
            vertical_acceleration = (vertical_velocity - self._last_vertical_velocity) / max(dt, 1e-6)

        self._last_planar_velocity = planar_velocity.copy()
        self._last_planar_acceleration = planar_acceleration.copy()
        self._last_altitude = altitude
        self._last_vertical_velocity = vertical_velocity
        self._last_timestamp = detection.timestamp

        position = np.array([self._state[0], self._state[1], altitude], dtype=float)
        velocity = np.array([planar_velocity[0], planar_velocity[1], vertical_velocity], dtype=float)
        acceleration = np.array(
            [planar_acceleration[0], planar_acceleration[1], vertical_acceleration],
            dtype=float,
        )
        metadata = dict(detection.metadata)
        metadata.update(
            {
                "tracker_backend": "kalman",
                "innovation_xy": innovation.tolist(),
                "kalman_gain": kalman_gain.tolist(),
                "covariance_trace": float(np.trace(self._covariance)),
                "measurement_noise_std": self._measurement_noise_std,
                "process_noise_std": self._process_noise_std,
                "smoothed_acceleration_xy": planar_acceleration.tolist(),
                "mahalanobis_distance": mahalanobis_distance,
                "innovation_covariance": innovation_covariance.tolist(),
            }
        )
        return TargetState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            covariance=self._covariance.copy(),
            timestamp=detection.timestamp,
            track_id=str(detection.metadata.get("track_id", "kalman-target")),
            metadata=metadata,
        )

    def _resolve_dt(self, timestamp: float | None) -> float:
        if timestamp is None or self._last_timestamp is None:
            return self._default_dt
        return max(float(timestamp - self._last_timestamp), 1e-6)

    def _build_process_noise(self, dt: float) -> np.ndarray:
        spectral_density = max(self._process_noise_std**2, 1e-8)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        return spectral_density * np.array(
            [
                [0.25 * dt4, 0.0, 0.5 * dt3, 0.0],
                [0.0, 0.25 * dt4, 0.0, 0.5 * dt3],
                [0.5 * dt3, 0.0, dt2, 0.0],
                [0.0, 0.5 * dt3, 0.0, dt2],
            ],
            dtype=float,
        )


class TargetTracker:
    """Configurable single-target tracker facade."""

    def __init__(self, config: dict[str, Any]) -> None:
        tracking = config.get("tracking", {})
        mode = str(tracking.get("mode", "kalman")).lower()
        self._backend = KinematicTargetTracker(config) if mode == "kinematic" else KalmanTargetTracker(config)

    def update(self, detection: Detection) -> TargetState:
        return self._backend.update(detection)


@dataclass(slots=True)
class TrackedObject:
    track_id: str
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    centroid: tuple[int, int]
    trajectory: list[tuple[int, int]]


class TrajectoryHistory:
    def __init__(self, max_length: int = 30) -> None:
        self._max_length = max_length
        self._paths: dict[str, deque[tuple[int, int]]] = {}

    def append(self, track_id: str, point: tuple[int, int]) -> list[tuple[int, int]]:
        history = self._paths.setdefault(track_id, deque(maxlen=self._max_length))
        history.append(point)
        return list(history)

    def get(self, track_id: str) -> list[tuple[int, int]]:
        history = self._paths.get(track_id)
        return list(history) if history is not None else []

    def prune(self, active_track_ids: set[str]) -> None:
        stale_ids = [track_id for track_id in self._paths if track_id not in active_track_ids]
        for track_id in stale_ids:
            self._paths.pop(track_id, None)


class DeepSortTracker:
    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 1,
        nn_budget: int | None = 100,
        max_cosine_distance: float = 0.2,
        max_iou_distance: float = 0.7,
        history_length: int = 30,
        embedder: str = "mobilenet",
        embedder_gpu: bool | None = None,
    ) -> None:
        deep_sort_cls = _import_deepsort()
        use_gpu = _torch_cuda_available() if embedder_gpu is None else embedder_gpu
        self._tracker = deep_sort_cls(
            max_age=max_age,
            n_init=n_init,
            nn_budget=nn_budget,
            max_cosine_distance=max_cosine_distance,
            max_iou_distance=max_iou_distance,
            embedder=embedder,
            embedder_gpu=use_gpu,
            half=use_gpu,
            bgr=True,
        )
        self._history = TrajectoryHistory(max_length=history_length)

    def update_from_yolo(self, result: Any, frame: np.ndarray) -> list[TrackedObject]:
        raw_detections = yolo_result_to_deepsort_detections(result)
        tracks = self._tracker.update_tracks(raw_detections, frame=frame)

        live_track_ids = {str(track.track_id) for track in tracks}
        self._history.prune(live_track_ids)

        tracked_objects: list[TrackedObject] = []
        frame_height, frame_width = frame.shape[:2]

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            bbox = track.to_ltrb(orig=True)
            if bbox is None:
                bbox = track.to_ltrb()

            x1, y1, x2, y2 = _clip_bbox_to_frame(
                bbox=bbox,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if x2 <= x1 or y2 <= y1:
                continue

            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            track_id = str(track.track_id)
            trajectory = self._history.append(track_id, centroid)
            tracked_objects.append(
                TrackedObject(
                    track_id=track_id,
                    class_name=str(track.get_det_class() or "object"),
                    confidence=float(track.get_det_conf() or 0.0),
                    bbox_xyxy=(x1, y1, x2, y2),
                    centroid=centroid,
                    trajectory=trajectory,
                )
            )

        return tracked_objects


def yolo_result_to_deepsort_detections(result: Any) -> list[tuple[list[float], float, str]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    if xyxy.size == 0:
        return []
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    names = result.names

    detections: list[tuple[list[float], float, str]] = []
    for box, confidence, class_id in zip(xyxy, confidences, class_ids, strict=False):
        x1, y1, x2, y2 = [float(value) for value in box]
        width = max(x2 - x1, 0.0)
        height = max(y2 - y1, 0.0)
        if width <= 0.0 or height <= 0.0:
            continue

        class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(names[class_id])
        detections.append(([x1, y1, width, height], float(confidence), class_name))

    return detections


def draw_tracked_objects(frame: np.ndarray, tracked_objects: list[TrackedObject], fps: float) -> np.ndarray:
    cv2 = _import_cv2()
    annotated = frame.copy()

    for tracked in tracked_objects:
        x1, y1, x2, y2 = tracked.bbox_xyxy
        color = _color_for_track(tracked.track_id)
        label = f"ID {tracked.track_id} {tracked.class_name} {tracked.confidence:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

        if len(tracked.trajectory) >= 2:
            points = np.array(tracked.trajectory, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [points], isClosed=False, color=color, thickness=2)

    cv2.putText(
        annotated,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _clip_bbox_to_frame(
    bbox: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    x1 = min(max(x1, 0), frame_width - 1)
    y1 = min(max(y1, 0), frame_height - 1)
    x2 = min(max(x2, 0), frame_width - 1)
    y2 = min(max(y2, 0), frame_height - 1)
    return x1, y1, x2, y2


def _import_deepsort() -> type:
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
    except ImportError as exc:
        raise ImportError(
            "deep-sort-realtime is required for DeepSORT tracking. "
            "Install it with 'python -m pip install deep-sort-realtime'."
        ) from exc

    return DeepSort


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for DeepSORT visualization. "
            "Install it before running tracking."
        ) from exc

    return cv2


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _color_for_track(track_id: str) -> tuple[int, int, int]:
    palette = (
        (231, 76, 60),
        (46, 204, 113),
        (52, 152, 219),
        (241, 196, 15),
        (155, 89, 182),
        (26, 188, 156),
        (230, 126, 34),
        (149, 165, 166),
        (52, 73, 94),
        (243, 156, 18),
    )
    return palette[hash(track_id) % len(palette)]


def _clip_planar_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm < 1e-6:
        return vector.astype(float)
    return vector.astype(float) / norm * max_norm
