from __future__ import annotations

from pathlib import Path
from typing import Any
from dataclasses import dataclass

import numpy as np

from drone_interceptor.types import Detection


class TargetDetector:
    """YOLO-backed detector with a synthetic fallback for the current simulator."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config["perception"]
        self._model: Any | None = None
        self._class_filter = self._config.get("target_classes")
        self._imgsz = int(self._config.get("inference_imgsz", 640))
        self._confidence_threshold = float(self._config["confidence_threshold"])
        self._model_path = self._resolve_model_path(self._config.get("model_path"))
        seed = int(config.get("system", {}).get("random_seed", 7))
        self._rng = np.random.default_rng(seed + 101)
        self._synthetic_noise_std = float(self._config.get("synthetic_measurement_noise_std_m", 0.0))

    @property
    def model_path(self) -> Path | str:
        return self._model_path

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @property
    def inference_imgsz(self) -> int:
        return self._imgsz

    def detect(self, observation: dict[str, np.ndarray]) -> Detection:
        image = self._extract_image(observation)
        if image is None:
            return self._synthetic_detection(observation)

        result = self._infer(image)
        if result.boxes is None or len(result.boxes) == 0:
            if "target_position" in observation:
                detection = self._synthetic_detection(observation)
                detection.metadata["reason"] = "no_yolo_detections"
                detection.metadata["fallback_from_image"] = True
                return detection
            raise RuntimeError("YOLO detector produced no target candidates for the provided frame.")

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        targets: list[dict[str, Any]] = []
        for index, (bbox_xyxy, confidence, class_id) in enumerate(zip(boxes_xyxy, confidences, classes, strict=False)):
            x1, y1, x2, y2 = bbox_xyxy
            names = result.names
            class_name = names.get(int(class_id), str(class_id)) if isinstance(names, dict) else names[int(class_id)]
            targets.append(
                {
                    "detection_index": int(index),
                    "position": [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0), 0.0],
                    "confidence": float(confidence),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "class_id": int(class_id),
                    "class_name": class_name,
                }
            )
        best_index = int(np.argmax(confidences))

        x1, y1, x2, y2 = boxes_xyxy[best_index]
        x_center = float((x1 + x2) / 2.0)
        y_center = float((y1 + y2) / 2.0)
        position = np.array([x_center, y_center, 0.0], dtype=float)
        class_id = int(classes[best_index])
        names = result.names
        class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]

        return Detection(
            position=position,
            confidence=float(confidences[best_index]),
            metadata={
                "backend": "ultralytics",
                "position_space": "image_pixels",
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "class_id": class_id,
                "class_name": class_name,
                "image_shape": list(image.shape),
                "model_path": str(self._model_path),
                "targets": targets,
            },
        )

    def _infer(self, image: np.ndarray) -> Any:
        model = self._load_model()
        kwargs: dict[str, Any] = {
            "imgsz": self._imgsz,
            "conf": self._confidence_threshold,
            "verbose": False,
        }
        if self._class_filter is not None:
            kwargs["classes"] = self._class_filter
        return model(image, **kwargs)[0]

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required for image-based detection. "
                "Install requirements.txt before using the perception model."
            ) from exc

        self._model = YOLO(str(self._model_path))
        return self._model

    def _resolve_model_path(self, configured_path: str | None) -> Path | str:
        if configured_path:
            candidate = Path(configured_path)
            if candidate.is_absolute() and candidate.exists():
                return candidate

            project_root = Path(__file__).resolve().parents[3]
            for path in (
                project_root / candidate,
                project_root / "models" / candidate,
            ):
                if path.exists():
                    return path.resolve()

        project_root = Path(__file__).resolve().parents[3]
        preferred_candidates = (
            project_root / "models" / "visdrone_yolov10s_brain" / "weights" / "best.pt",
            project_root / "models" / "visdrone_yolov10s_brain.pt",
            project_root / "models" / "visdrone_yolov10n.pt",
        )
        for candidate in preferred_candidates:
            if candidate.exists():
                return candidate.resolve()

        return "yolov10s.pt"

    def _extract_image(self, observation: dict[str, np.ndarray]) -> np.ndarray | None:
        for key in ("image", "frame", "rgb"):
            value = observation.get(key)
            if value is not None:
                return np.asarray(value)
        return None

    def _synthetic_detection(self, observation: dict[str, np.ndarray]) -> Detection:
        if "target_positions" in observation:
            positions = np.asarray(observation["target_positions"], dtype=float)
            if positions.ndim == 1:
                positions = positions.reshape(1, -1)
        else:
            positions = np.asarray([observation["target_position"]], dtype=float)
        velocities = np.asarray(
            observation.get("target_velocities", [observation.get("target_velocity", np.zeros(3, dtype=float))]),
            dtype=float,
        )
        if velocities.ndim == 1:
            velocities = velocities.reshape(1, -1)
        accelerations = np.asarray(
            observation.get("target_accelerations", [observation.get("target_acceleration", np.zeros(3, dtype=float))]),
            dtype=float,
        )
        if accelerations.ndim == 1:
            accelerations = accelerations.reshape(1, -1)
        true_position = positions[0].astype(float)
        noise = self._rng.normal(0.0, self._synthetic_noise_std, size=true_position.shape)
        if noise.shape[0] >= 3:
            noise[2] *= 0.35
        position = true_position + noise
        confidence = min(self._confidence_threshold + 0.5, 0.99)
        targets = []
        for index, candidate in enumerate(positions):
            candidate_noise = self._rng.normal(0.0, self._synthetic_noise_std, size=candidate.shape)
            if candidate_noise.shape[0] >= 3:
                candidate_noise[2] *= 0.35
            measured = candidate.astype(float) + candidate_noise
            targets.append(
                {
                    "detection_index": int(index),
                    "position": measured.tolist(),
                    "confidence": float(max(confidence - 0.02 * index, 0.5)),
                    "track_id": f"sim-target-{index + 1}",
                    "measured_velocity_xy": np.asarray(velocities[min(index, len(velocities) - 1)], dtype=float)[:2].tolist(),
                    "measured_acceleration_xy": np.asarray(accelerations[min(index, len(accelerations) - 1)], dtype=float)[:2].tolist(),
                }
            )
        return Detection(
            position=position,
            confidence=float(confidence),
            metadata={
                "backend": "synthetic",
                "position_space": "world",
                "true_position": true_position.tolist(),
                "true_positions": positions.astype(float).tolist(),
                "measured_velocity_xy": np.asarray(velocities[0], dtype=float)[:2].tolist(),
                "measured_acceleration_xy": np.asarray(accelerations[0], dtype=float)[:2].tolist(),
                "measurement_noise_std_m": self._synthetic_noise_std,
                "targets": targets,
            },
            timestamp=float(np.asarray(observation.get("time", [0.0]), dtype=float)[0]),
        )


@dataclass(frozen=True)
class DetectionBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    class_id: int = 0


@dataclass(frozen=True)
class DetectorBenchmarkSummary:
    sample_count: int
    precision: float
    recall: float
    map50: float
    map50_95: float
    mean_iou: float
    fps: float
    device_label: str


def benchmark_detection_sets(
    ground_truth_sets: list[list[DetectionBox]],
    prediction_sets: list[list[DetectionBox]],
    inference_times_s: list[float] | None = None,
    device_label: str = "workstation_unverified",
) -> DetectorBenchmarkSummary:
    if len(ground_truth_sets) != len(prediction_sets):
        raise ValueError("Ground-truth and prediction sets must have the same length.")
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    iou_scores: list[float] = []
    thresholds = [0.5 + 0.05 * index for index in range(10)]
    ap_hits = {threshold: [] for threshold in thresholds}

    for ground_truth, predictions in zip(ground_truth_sets, prediction_sets, strict=False):
        matched_gt: set[int] = set()
        ordered_predictions = sorted(predictions, key=lambda item: item.confidence, reverse=True)
        for prediction in ordered_predictions:
            best_match = None
            best_iou = 0.0
            for index, target in enumerate(ground_truth):
                if index in matched_gt or prediction.class_id != target.class_id:
                    continue
                iou = _box_iou(prediction, target)
                if iou > best_iou:
                    best_iou = iou
                    best_match = index
            if best_match is not None and best_iou >= 0.5:
                matched_gt.add(best_match)
                total_true_positives += 1
                iou_scores.append(best_iou)
                for threshold in thresholds:
                    ap_hits[threshold].append(1.0 if best_iou >= threshold else 0.0)
            else:
                total_false_positives += 1
        total_false_negatives += max(len(ground_truth) - len(matched_gt), 0)

    precision = _safe_ratio(total_true_positives, total_true_positives + total_false_positives)
    recall = _safe_ratio(total_true_positives, total_true_positives + total_false_negatives)
    map50 = float(np.mean(ap_hits[0.5])) if ap_hits[0.5] else 0.0
    map50_95 = float(np.mean([np.mean(values) if values else 0.0 for values in ap_hits.values()]))
    fps = 0.0
    if inference_times_s:
        mean_inference_time = float(np.mean(np.asarray(inference_times_s, dtype=float)))
        fps = 1.0 / max(mean_inference_time, 1e-6)
    return DetectorBenchmarkSummary(
        sample_count=len(ground_truth_sets),
        precision=float(precision),
        recall=float(recall),
        map50=float(map50),
        map50_95=float(map50_95),
        mean_iou=float(np.mean(iou_scores)) if iou_scores else 0.0,
        fps=float(fps),
        device_label=str(device_label),
    )


def _box_iou(left: DetectionBox, right: DetectionBox) -> float:
    x1 = max(float(left.x1), float(right.x1))
    y1 = max(float(left.y1), float(right.y1))
    x2 = min(float(left.x2), float(right.x2))
    y2 = min(float(left.y2), float(right.y2))
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    left_area = max(float(left.x2 - left.x1), 0.0) * max(float(left.y2 - left.y1), 0.0)
    right_area = max(float(right.x2 - right.x1), 0.0) * max(float(right.y2 - right.y1), 0.0)
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


__all__ = [
    "DetectionBox",
    "DetectorBenchmarkSummary",
    "TargetDetector",
    "benchmark_detection_sets",
]
