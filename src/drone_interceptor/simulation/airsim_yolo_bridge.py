from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from drone_interceptor.config import load_config
from drone_interceptor.constraints import ConstraintEnvelope, load_constraint_envelope
from drone_interceptor.perception.detector import TargetDetector
from drone_interceptor.simulation.airsim_connection import connect_airsim


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True, slots=True)
class AirSimYOLOStatus:
    timestamp: float
    connected: bool
    connect_airsim: bool
    fallback_mode: bool
    connect_error: str
    host: str
    port: int
    vehicle_name: str
    camera_name: str
    image_source: str
    frame_width: int
    frame_height: int
    model_path: str
    model_grade: str
    inference_time_ms: float
    detection_count: int
    best_confidence: float
    tracking_threshold: float
    tracking_constraint_ok: bool
    drift_rate_mps: float
    drift_constraint_min_mps: float
    drift_constraint_max_mps: float
    drift_constraint_ok: bool
    annotated_frame_path: str
    detections: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AirSimYOLOBridge:
    """Capture AirSim frames, run YOLOv10 inference, and expose a frontend-safe status payload."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        output_dir: str | Path | None = None,
    ) -> None:
        self._config_path = Path(config_path)
        self._config = load_config(self._config_path)
        self._envelope: ConstraintEnvelope = load_constraint_envelope(self._config)
        self._detector = TargetDetector(self._config)
        self._output_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "outputs" / "airsim_yolo"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._latest_frame_path = self._output_dir / "airsim_yolov10_interceptor_latest.jpg"
        self._lock = threading.Lock()
        self._last_status: AirSimYOLOStatus | None = None

    def get_status(
        self,
        *,
        refresh: bool = True,
        host: str | None = None,
        port: int | None = None,
        vehicle_name: str | None = None,
        camera_name: str = "0",
        drift_rate_mps: float = 0.3,
        timeout_s: float = 3.0,
        connect_airsim: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            if not refresh and self._last_status is not None:
                return self._last_status.to_dict()

        airsim_config = self._config.get("airsim", {})
        resolved_host = str(host if host is not None else airsim_config.get("host", "127.0.0.1"))
        resolved_port = int(port if port is not None else airsim_config.get("port", 41451))
        resolved_vehicle_name = str(vehicle_name if vehicle_name is not None else airsim_config.get("vehicle_name", ""))
        resolved_camera_name = str(camera_name or "0")
        resolved_timeout_s = max(float(timeout_s), 0.5)
        resolved_drift_rate = float(drift_rate_mps)
        resolved_connect = bool(connect_airsim)

        frame: np.ndarray | None = None
        connected = False
        connect_error = ""
        image_source = "fallback"

        if resolved_connect:
            frame, connected, connect_error, image_source = self._capture_airsim_frame(
                host=resolved_host,
                port=resolved_port,
                vehicle_name=resolved_vehicle_name,
                camera_name=resolved_camera_name,
                timeout_s=resolved_timeout_s,
            )
        else:
            connect_error = "AirSim connection disabled by payload."

        if frame is None:
            frame = self._build_fallback_frame(error_text=connect_error)
            fallback_mode = True
        else:
            fallback_mode = False

        inference_start = time.perf_counter()
        detections, best_confidence, inference_error = self._run_detection(frame)
        inference_time_ms = (time.perf_counter() - inference_start) * 1000.0
        if inference_error:
            if connect_error:
                connect_error = f"{connect_error} | inference_error: {inference_error}"
            else:
                connect_error = f"inference_error: {inference_error}"

        annotated = self._annotate_frame(
            frame=frame,
            detections=detections,
            connected=connected,
            fallback_mode=fallback_mode,
            camera_name=resolved_camera_name,
            vehicle_name=resolved_vehicle_name,
        )
        self._latest_frame_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self._latest_frame_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        tracking_threshold = float(self._detector.confidence_threshold)
        tracking_constraint_ok = bool(best_confidence >= tracking_threshold) if detections else False
        drift_constraint_ok = bool(
            self._envelope.drift_rate_min_mps <= resolved_drift_rate <= self._envelope.drift_rate_max_mps
        )

        status = AirSimYOLOStatus(
            timestamp=time.time(),
            connected=bool(connected),
            connect_airsim=resolved_connect,
            fallback_mode=bool(fallback_mode),
            connect_error=str(connect_error or ""),
            host=resolved_host,
            port=int(resolved_port),
            vehicle_name=resolved_vehicle_name,
            camera_name=resolved_camera_name,
            image_source=image_source,
            frame_width=int(frame.shape[1]),
            frame_height=int(frame.shape[0]),
            model_path=str(self._detector.model_path),
            model_grade=_model_grade(str(self._detector.model_path)),
            inference_time_ms=float(inference_time_ms),
            detection_count=len(detections),
            best_confidence=float(best_confidence),
            tracking_threshold=tracking_threshold,
            tracking_constraint_ok=tracking_constraint_ok,
            drift_rate_mps=resolved_drift_rate,
            drift_constraint_min_mps=float(self._envelope.drift_rate_min_mps),
            drift_constraint_max_mps=float(self._envelope.drift_rate_max_mps),
            drift_constraint_ok=drift_constraint_ok,
            annotated_frame_path=str(self._latest_frame_path.resolve()),
            detections=tuple(detections),
        )
        with self._lock:
            self._last_status = status
        return status.to_dict()

    def _capture_airsim_frame(
        self,
        *,
        host: str,
        port: int,
        vehicle_name: str,
        camera_name: str,
        timeout_s: float,
    ) -> tuple[np.ndarray | None, bool, str, str]:
        try:
            client = connect_airsim(host=host, port=port, timeout_value=max(int(round(timeout_s)), 1), vehicle_name=vehicle_name)
        except Exception as exc:
            return None, False, str(exc), "connect_failed"

        try:
            airsim = __import__("airsim")
        except Exception as exc:
            return None, False, f"airsim import error: {exc}", "airsim_import_error"

        try:
            request = airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
            responses = client.simGetImages([request], vehicle_name=vehicle_name)
            if responses:
                response = responses[0]
                width = int(getattr(response, "width", 0))
                height = int(getattr(response, "height", 0))
                raw = bytes(getattr(response, "image_data_uint8", b""))
                expected = width * height * 3
                if width > 0 and height > 0 and len(raw) >= expected:
                    rgb = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, width, 3)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    return bgr, True, "", "simGetImages"
        except Exception:
            pass

        try:
            png_bytes = client.simGetImage(camera_name, airsim.ImageType.Scene, vehicle_name=vehicle_name)
        except TypeError:
            png_bytes = client.simGetImage(camera_name, airsim.ImageType.Scene)
        except Exception as exc:
            return None, True, f"image capture failed: {exc}", "simGetImage_failed"

        if not png_bytes:
            return None, True, "AirSim returned an empty image payload.", "simGetImage_empty"

        decoded = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            return None, True, "Failed to decode AirSim PNG frame.", "decode_failed"
        return decoded, True, "", "simGetImage"

    def _run_detection(self, frame: np.ndarray) -> tuple[list[dict[str, Any]], float, str]:
        try:
            detection = self._detector.detect({"frame": frame})
        except Exception as exc:
            return [], 0.0, str(exc)

        targets = detection.metadata.get("targets", []) if isinstance(detection.metadata, dict) else []
        if not isinstance(targets, list):
            targets = []

        detections: list[dict[str, Any]] = []
        best_conf = float(detection.confidence)

        for index, target in enumerate(targets):
            if not isinstance(target, dict):
                continue
            bbox = target.get("bbox_xyxy")
            if bbox is None or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(value) for value in bbox]
            confidence = float(target.get("confidence", detection.confidence))
            detections.append(
                {
                    "index": int(index),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": int(target.get("class_id", -1)),
                    "class_name": str(target.get("class_name", "target")),
                }
            )
            if confidence > best_conf:
                best_conf = confidence
        return detections, float(best_conf if detections else detection.confidence), ""

    def _annotate_frame(
        self,
        *,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        connected: bool,
        fallback_mode: bool,
        camera_name: str,
        vehicle_name: str,
    ) -> np.ndarray:
        canvas = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox_xyxy"]]
            x1 = max(0, min(x1, canvas.shape[1] - 1))
            x2 = max(0, min(x2, canvas.shape[1] - 1))
            y1 = max(0, min(y1, canvas.shape[0] - 1))
            y2 = max(0, min(y2, canvas.shape[0] - 1))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (48, 255, 96), 2)
            label = f"{detection['class_name']} {float(detection['confidence']):.2f}"
            cv2.putText(
                canvas,
                label,
                (x1, max(y1 - 8, 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (48, 255, 96),
                2,
                cv2.LINE_AA,
            )

        mode_label = "AirSim LIVE" if connected else "AirSim OFFLINE (Fallback)"
        cv2.putText(canvas, mode_label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 238, 255), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"Model: {Path(str(self._detector.model_path)).name} [{_model_grade(str(self._detector.model_path))}]",
            (14, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Camera: {camera_name} | Vehicle: {vehicle_name or 'default'} | Detections: {len(detections)}",
            (14, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if fallback_mode:
            cv2.putText(
                canvas,
                "Running with synthetic frame because AirSim stream is not available.",
                (14, 106),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (51, 215, 255),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def _build_fallback_frame(self, error_text: str) -> np.ndarray:
        height = 720
        width = 1280
        gradient_x = np.linspace(0, 1, width, dtype=np.float32)
        gradient_y = np.linspace(0, 1, height, dtype=np.float32).reshape(-1, 1)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :, 0] = np.clip(20 + 65 * gradient_x, 0, 255).astype(np.uint8)
        canvas[:, :, 1] = np.clip(18 + 80 * gradient_y, 0, 255).astype(np.uint8)
        canvas[:, :, 2] = 22
        cv2.putText(canvas, "AIRSIM INTERCEPTOR FEED", (34, 82), cv2.FONT_HERSHEY_DUPLEX, 1.22, (72, 248, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Fallback frame active", (34, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (255, 255, 255), 2, cv2.LINE_AA)
        if error_text:
            clipped = error_text[:160]
            cv2.putText(canvas, f"Reason: {clipped}", (34, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas


def _model_grade(model_path: str) -> str:
    lowered = str(model_path).lower()
    if "yolov10x" in lowered or "yolov10l" in lowered:
        return "ultra"
    if "yolov10m" in lowered or "yolov10s" in lowered:
        return "high"
    if "yolov10n" in lowered:
        return "balanced"
    return "custom"


__all__ = ["AirSimYOLOBridge", "AirSimYOLOStatus"]
