from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from drone_interceptor.config import load_config
from drone_interceptor.perception.detector import TargetDetector
from drone_interceptor.ros2.common import load_rclpy, load_string_message, to_json_message
from drone_interceptor.ros2.mavlink_bridge import MavlinkBridge


EARTH_METERS_PER_DEG = 111_320.0


def _clip(value: float, lower: float, upper: float) -> float:
    return float(max(min(value, upper), lower))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    except (TypeError, ValueError):
        pass
    return float(default)


@dataclass(frozen=True, slots=True)
class HardwareMapping:
    compute: str = "Jetson Nano"
    sdr: str = "HackRF One"
    vision: str = "YOLOv10-tiny"


@dataclass(frozen=True, slots=True)
class SafetyInterlockConfig:
    min_safe_distance_m: float = 1.5
    fade_distance_m: float = 8.0
    min_power_dbm: float = -40.0
    max_power_dbm: float = -8.0
    telemetry_guard_bands_hz: tuple[tuple[float, float], ...] = (
        (2.4e9, 120e6),
        (5.8e9, 220e6),
    )
    frequency_pool_hz: tuple[float, ...] = (
        915e6,
        1.2e9,
        1.57542e9,
    )


@dataclass(frozen=True, slots=True)
class SafetyInterlockDecision:
    distance_m: float
    power_limit_dbm: float
    selected_frequency_hz: float
    interference_detected: bool


class SafetyInterlock:
    """Safety-first RF gate. This module computes constraints only and does not transmit."""

    def __init__(self, config: SafetyInterlockConfig | None = None) -> None:
        self._config = config or SafetyInterlockConfig()

    def _in_guard_band(self, frequency_hz: float) -> bool:
        freq = float(frequency_hz)
        for center_hz, bandwidth_hz in self._config.telemetry_guard_bands_hz:
            if abs(freq - float(center_hz)) <= float(bandwidth_hz) * 0.5:
                return True
        return False

    def power_limit_dbm(self, distance_m: float) -> float:
        distance = max(float(distance_m), 0.0)
        min_safe = max(float(self._config.min_safe_distance_m), 1e-3)
        fade = max(float(self._config.fade_distance_m), 1e-3)
        if distance <= min_safe:
            return float(self._config.min_power_dbm)
        ratio = _clip((distance - min_safe) / fade, 0.0, 1.0)
        return float(
            self._config.min_power_dbm
            + ratio * (self._config.max_power_dbm - self._config.min_power_dbm)
        )

    def choose_frequency_hz(
        self,
        desired_frequency_hz: float,
        interference_frequency_hz: float | None = None,
    ) -> tuple[float, bool]:
        desired = float(desired_frequency_hz)
        interference = float(interference_frequency_hz) if interference_frequency_hz is not None else None
        interference_detected = False
        if self._in_guard_band(desired):
            interference_detected = True
        if interference is not None and abs(desired - interference) <= 5e6:
            interference_detected = True
        if not interference_detected:
            return desired, False

        candidates = list(self._config.frequency_pool_hz)
        ranked: list[tuple[float, float]] = []
        for candidate in candidates:
            if self._in_guard_band(candidate):
                continue
            score = 0.0
            if interference is not None:
                score += abs(candidate - interference)
            for center_hz, _ in self._config.telemetry_guard_bands_hz:
                score += abs(candidate - float(center_hz))
            ranked.append((score, float(candidate)))
        if not ranked:
            return desired, True
        ranked.sort(key=lambda item: item[0], reverse=True)
        return float(ranked[0][1]), True

    def evaluate(
        self,
        sdr_to_own_gnss_distance_m: float,
        desired_frequency_hz: float,
        interference_frequency_hz: float | None = None,
    ) -> SafetyInterlockDecision:
        selected_frequency_hz, interference_detected = self.choose_frequency_hz(
            desired_frequency_hz=desired_frequency_hz,
            interference_frequency_hz=interference_frequency_hz,
        )
        return SafetyInterlockDecision(
            distance_m=float(max(sdr_to_own_gnss_distance_m, 0.0)),
            power_limit_dbm=self.power_limit_dbm(sdr_to_own_gnss_distance_m),
            selected_frequency_hz=selected_frequency_hz,
            interference_detected=bool(interference_detected),
        )


@dataclass(frozen=True, slots=True)
class GeoFix:
    lat_deg: float
    lon_deg: float
    alt_m: float
    timestamp_s: float


@dataclass(frozen=True, slots=True)
class DriftPlan:
    north_offset_m: float
    east_offset_m: float
    spoof_lat_deg: float
    spoof_lon_deg: float
    spoof_alt_m: float
    target_distance_m: float


class DefensiveDriftPlanner:
    """Defensive planner for simulation/testing; computes offset plans only."""

    def __init__(self, lead_gain: float = 0.35, max_offset_m: float = 30.0) -> None:
        self._lead_gain = float(max(lead_gain, 0.0))
        self._max_offset_m = float(max(max_offset_m, 1.0))

    @staticmethod
    def _offset_geodetic(lat_deg: float, lon_deg: float, north_m: float, east_m: float) -> tuple[float, float]:
        dlat = float(north_m) / EARTH_METERS_PER_DEG
        cos_lat = max(abs(math.cos(math.radians(lat_deg))), 1e-6)
        dlon = float(east_m) / (EARTH_METERS_PER_DEG * cos_lat)
        return float(lat_deg + dlat), float(lon_deg + dlon)

    def plan(self, fix: GeoFix, relative_xyz_m: np.ndarray) -> DriftPlan:
        relative = np.asarray(relative_xyz_m, dtype=float).reshape(-1)
        x_m = float(relative[0]) if relative.size > 0 else 0.0
        y_m = float(relative[1]) if relative.size > 1 else 0.0
        z_m = float(relative[2]) if relative.size > 2 else 0.0
        horizontal_distance_m = float(np.linalg.norm(np.array([x_m, y_m], dtype=float)))

        east_offset_m = _clip(x_m * self._lead_gain, -self._max_offset_m, self._max_offset_m)
        north_offset_m = _clip(y_m * self._lead_gain, -self._max_offset_m, self._max_offset_m)
        spoof_lat_deg, spoof_lon_deg = self._offset_geodetic(
            lat_deg=fix.lat_deg,
            lon_deg=fix.lon_deg,
            north_m=north_offset_m,
            east_m=east_offset_m,
        )
        spoof_alt_m = float(fix.alt_m + _clip(0.1 * z_m, -15.0, 15.0))
        return DriftPlan(
            north_offset_m=north_offset_m,
            east_offset_m=east_offset_m,
            spoof_lat_deg=spoof_lat_deg,
            spoof_lon_deg=spoof_lon_deg,
            spoof_alt_m=spoof_alt_m,
            target_distance_m=horizontal_distance_m,
        )


@dataclass(frozen=True, slots=True)
class SDRRuntimeStatus:
    gps_sdr_sim_available: bool
    hackrf_info_available: bool
    hackrf_info_ok: bool
    details: str


class SDRDryRunInterface:
    """Simulation-safe SDR adapter. It never starts RF transmission."""

    def __init__(self, gps_sdr_sim_bin: str = "gps-sdr-sim", hackrf_info_bin: str = "hackrf_info") -> None:
        self._gps_sdr_sim_bin = str(gps_sdr_sim_bin)
        self._hackrf_info_bin = str(hackrf_info_bin)

    def inspect_runtime(self, timeout_s: float = 3.0) -> SDRRuntimeStatus:
        gps_sdr_sim_available = shutil.which(self._gps_sdr_sim_bin) is not None
        hackrf_info_available = shutil.which(self._hackrf_info_bin) is not None
        hackrf_ok = False
        details = ""
        if hackrf_info_available:
            try:
                completed = subprocess.run(
                    [self._hackrf_info_bin],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=max(float(timeout_s), 0.5),
                )
                hackrf_ok = completed.returncode == 0
                details = (completed.stdout or completed.stderr or "").strip()[:500]
            except Exception as exc:
                details = str(exc)
        return SDRRuntimeStatus(
            gps_sdr_sim_available=bool(gps_sdr_sim_available),
            hackrf_info_available=bool(hackrf_info_available),
            hackrf_info_ok=bool(hackrf_ok),
            details=details,
        )

    def dry_run_plan(
        self,
        current_fix: GeoFix,
        drift_plan: DriftPlan,
        interlock: SafetyInterlockDecision,
        duration_s: float = 5.0,
    ) -> dict[str, Any]:
        return {
            "mode": "dry_run_only",
            "note": "RF transmission is intentionally disabled in this implementation.",
            "duration_s": float(max(duration_s, 0.1)),
            "from_fix": {
                "lat_deg": current_fix.lat_deg,
                "lon_deg": current_fix.lon_deg,
                "alt_m": current_fix.alt_m,
            },
            "to_fix": {
                "lat_deg": drift_plan.spoof_lat_deg,
                "lon_deg": drift_plan.spoof_lon_deg,
                "alt_m": drift_plan.spoof_alt_m,
            },
            "interlock": {
                "power_limit_dbm": interlock.power_limit_dbm,
                "selected_frequency_hz": interlock.selected_frequency_hz,
                "interference_detected": interlock.interference_detected,
            },
        }


@dataclass(slots=True)
class SpoofManagerConfig:
    hardware: HardwareMapping = field(default_factory=HardwareMapping)
    spoof_enable: bool = False
    desired_frequency_hz: float = 1.57542e9
    sdr_to_own_gnss_distance_m: float = 2.0
    telemetry_interference_hz: float | None = None
    log_path: Path = field(default_factory=lambda: Path("logs") / "spoof_manager_telemetry.jsonl")
    heatmap_bins: int = 8
    heatmap_range_m: float = 200.0


class SpoofManagerCore:
    """Core analytics for a ROS2 spoof-manager style workflow."""

    def __init__(
        self,
        config: SpoofManagerConfig | None = None,
        planner: DefensiveDriftPlanner | None = None,
        interlock: SafetyInterlock | None = None,
        sdr: SDRDryRunInterface | None = None,
    ) -> None:
        self._config = config or SpoofManagerConfig()
        self._planner = planner or DefensiveDriftPlanner()
        self._interlock = interlock or SafetyInterlock()
        self._sdr = sdr or SDRDryRunInterface()
        self._target_history: deque[np.ndarray] = deque(maxlen=240)
        self._confidence_history: deque[float] = deque(maxlen=120)
        self._log_path = Path(self._config.log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_lock = threading.Lock()

    def _spoof_confidence(self, relative_xyz_m: np.ndarray, detection_confidence: float) -> float:
        target = np.asarray(relative_xyz_m, dtype=float).reshape(-1)
        distance_m = float(np.linalg.norm(target[:2])) if target.size >= 2 else 0.0
        distance_score = float(math.exp(-distance_m / 180.0))
        detection_score = _clip(detection_confidence, 0.0, 1.0)
        stability_score = 1.0
        if len(self._target_history) >= 2:
            deltas = np.diff(np.asarray(self._target_history, dtype=float), axis=0)
            jitter = float(np.mean(np.linalg.norm(deltas[:, :2], axis=1)))
            stability_score = float(math.exp(-jitter / 25.0))
        confidence = float(
            np.clip(
                0.45 * detection_score + 0.35 * distance_score + 0.20 * stability_score,
                0.0,
                1.0,
            )
        )
        self._confidence_history.append(confidence)
        return confidence

    def _build_heatmap(self) -> list[list[float]]:
        bins = max(int(self._config.heatmap_bins), 4)
        extent = max(float(self._config.heatmap_range_m), 10.0)
        if not self._target_history:
            return [[0.0 for _ in range(bins)] for _ in range(bins)]
        points = np.asarray(self._target_history, dtype=float)
        x = points[:, 0]
        y = points[:, 1]
        hist, _, _ = np.histogram2d(
            x=x,
            y=y,
            bins=bins,
            range=[[-extent, extent], [-extent, extent]],
        )
        max_value = float(np.max(hist))
        if max_value <= 0.0:
            return [[0.0 for _ in range(bins)] for _ in range(bins)]
        normalized = hist / max_value
        return normalized.round(4).tolist()

    def _write_log(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, separators=(",", ":"))
        with self._log_lock:
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def update(self, fix: GeoFix, relative_xyz_m: np.ndarray, detection_confidence: float) -> dict[str, Any]:
        target = np.asarray(relative_xyz_m, dtype=float).reshape(-1)
        if target.size < 3:
            padded = np.zeros(3, dtype=float)
            padded[: target.size] = target
            target = padded
        self._target_history.append(target.copy())

        drift_plan = self._planner.plan(fix, target)
        interlock_decision = self._interlock.evaluate(
            sdr_to_own_gnss_distance_m=float(self._config.sdr_to_own_gnss_distance_m),
            desired_frequency_hz=float(self._config.desired_frequency_hz),
            interference_frequency_hz=self._config.telemetry_interference_hz,
        )
        runtime_status = self._sdr.inspect_runtime(timeout_s=1.5)
        dry_run = self._sdr.dry_run_plan(
            current_fix=fix,
            drift_plan=drift_plan,
            interlock=interlock_decision,
            duration_s=3.0,
        )
        spoof_confidence = self._spoof_confidence(target, detection_confidence)
        heatmap = self._build_heatmap()

        status = "STANDBY"
        if detection_confidence >= 0.45:
            status = "TARGET ACQUIRED"
        if bool(self._config.spoof_enable) and detection_confidence >= 0.45:
            status = "SPOOFING ACTIVE (DRY-RUN)"

        payload = {
            "timestamp": time.time(),
            "status": status,
            "hardware": {
                "compute": self._config.hardware.compute,
                "sdr": self._config.hardware.sdr,
                "vision": self._config.hardware.vision,
            },
            "detection_confidence": float(_clip(detection_confidence, 0.0, 1.0)),
            "target_relative_m": target.tolist(),
            "spoof_confidence_score": float(round(spoof_confidence, 4)),
            "sdr_heatmap": heatmap,
            "drift_plan": {
                "north_offset_m": drift_plan.north_offset_m,
                "east_offset_m": drift_plan.east_offset_m,
                "target_distance_m": drift_plan.target_distance_m,
                "spoof_fix": {
                    "lat_deg": drift_plan.spoof_lat_deg,
                    "lon_deg": drift_plan.spoof_lon_deg,
                    "alt_m": drift_plan.spoof_alt_m,
                },
            },
            "safety_interlock": {
                "distance_m": interlock_decision.distance_m,
                "power_limit_dbm": interlock_decision.power_limit_dbm,
                "selected_frequency_hz": interlock_decision.selected_frequency_hz,
                "interference_detected": interlock_decision.interference_detected,
            },
            "sdr_runtime": {
                "gps_sdr_sim_available": runtime_status.gps_sdr_sim_available,
                "hackrf_info_available": runtime_status.hackrf_info_available,
                "hackrf_info_ok": runtime_status.hackrf_info_ok,
            },
            "dry_run_plan": dry_run,
        }
        self._write_log(payload)
        return payload


def _load_navsat_msg_type() -> Any:
    from sensor_msgs.msg import NavSatFix  # type: ignore

    return NavSatFix


def _load_image_msg_type() -> Any:
    from sensor_msgs.msg import Image  # type: ignore

    return Image


def _image_msg_to_bgr(image_msg: Any) -> np.ndarray | None:
    try:
        from cv_bridge import CvBridge  # type: ignore
    except Exception:
        CvBridge = None  # type: ignore
    if CvBridge is not None:
        try:
            bridge = CvBridge()
            return np.asarray(bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8"))
        except Exception:
            pass
    try:
        height = int(getattr(image_msg, "height", 0))
        width = int(getattr(image_msg, "width", 0))
        encoding = str(getattr(image_msg, "encoding", "rgb8")).lower()
        raw = bytes(getattr(image_msg, "data", b""))
        if height <= 0 or width <= 0 or not raw:
            return None
        channels = 3
        expected = height * width * channels
        if len(raw) < expected:
            return None
        frame = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, width, channels)
        if encoding == "bgr8":
            return frame.copy()
        return frame[:, :, ::-1].copy()
    except Exception:
        return None


class VisionInferenceWorker:
    def __init__(self, config: dict[str, Any]) -> None:
        local_config = json.loads(json.dumps(config))
        local_config.setdefault("perception", {})
        local_config["perception"]["model_path"] = str(
            local_config["perception"].get("model_path", "yolov10n.pt")
        )
        self._detector = TargetDetector(local_config)

    def infer_relative(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        detection = self._detector.detect({"frame": frame_bgr})
        confidence = float(_clip(detection.confidence, 0.0, 1.0))
        h, w = frame_bgr.shape[:2]
        cx = _safe_float(detection.position[0], w * 0.5) if detection.position.size > 0 else w * 0.5
        cy = _safe_float(detection.position[1], h * 0.5) if detection.position.size > 1 else h * 0.5
        dx_px = cx - (w * 0.5)
        dy_px = cy - (h * 0.5)
        meters_per_px = 0.08
        relative = np.array(
            [
                dx_px * meters_per_px,
                -dy_px * meters_per_px,
                25.0 * max(1.0 - confidence, 0.1),
            ],
            dtype=float,
        )
        return relative, confidence


def main() -> None:
    parser = argparse.ArgumentParser(description="ROS2 Spoof-Manager node (defensive, dry-run only).")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--spoof-enable", action="store_true", help="Enable dry-run planning state machine.")
    parser.add_argument("--log-path", type=Path, default=Path("logs") / "spoof_manager_telemetry.jsonl")
    parser.add_argument("--desired-frequency-hz", type=float, default=1.57542e9)
    parser.add_argument("--sdr-offset-distance-m", type=float, default=2.0)
    parser.add_argument("--mavlink-uri", type=str, default="udpout:127.0.0.1:14550")
    args = parser.parse_args()

    config = load_config(args.config)
    rclpy, node_cls = load_rclpy()
    string_msg = load_string_message()
    navsat_type = _load_navsat_msg_type()
    image_type = _load_image_msg_type()

    class SpoofManagerNode(node_cls):
        def __init__(self) -> None:
            super().__init__("drone_interceptor_spoof_manager")
            self._status_pub = self.create_publisher(string_msg, "/spoof_manager/status", 10)
            self._heartbeat_pub = self.create_publisher(string_msg, "/spoof_manager/heartbeat", 10)
            self._latest_fix: GeoFix | None = None
            self._latest_frame: np.ndarray | None = None
            self._latest_relative = np.zeros(3, dtype=float)
            self._latest_confidence = 0.0
            self._vision_future: Future[tuple[np.ndarray, float]] | None = None
            self._manager_future: Future[dict[str, Any]] | None = None
            self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="spoof_mgr")
            self._vision_worker = VisionInferenceWorker(config)
            self._mavlink = MavlinkBridge(connection_uri=str(args.mavlink_uri))
            self._core = SpoofManagerCore(
                config=SpoofManagerConfig(
                    spoof_enable=bool(args.spoof_enable),
                    desired_frequency_hz=float(args.desired_frequency_hz),
                    sdr_to_own_gnss_distance_m=float(args.sdr_offset_distance_m),
                    log_path=Path(args.log_path),
                )
            )
            self._status_lock = threading.Lock()
            self._last_status: str = "STANDBY"

            self.create_subscription(navsat_type, "/mavros/global_position/raw", self._on_fix, 10)
            self.create_subscription(image_type, "/camera/image_raw", self._on_image, 10)
            self.create_timer(0.10, self._tick)
            self.create_timer(0.50, self._heartbeat)

        def _on_fix(self, msg: Any) -> None:
            self._latest_fix = GeoFix(
                lat_deg=_safe_float(getattr(msg, "latitude", 0.0), 0.0),
                lon_deg=_safe_float(getattr(msg, "longitude", 0.0), 0.0),
                alt_m=_safe_float(getattr(msg, "altitude", 0.0), 0.0),
                timestamp_s=time.time(),
            )

        def _on_image(self, msg: Any) -> None:
            frame = _image_msg_to_bgr(msg)
            if frame is not None:
                self._latest_frame = frame

        def _heartbeat(self) -> None:
            message = string_msg()
            message.data = to_json_message(
                {
                    "timestamp": time.time(),
                    "node": "drone_interceptor_spoof_manager",
                    "status": "alive",
                    "spoof_enable": bool(args.spoof_enable),
                }
            )
            self._heartbeat_pub.publish(message)

        def _tick(self) -> None:
            if self._latest_fix is None:
                return

            if self._latest_frame is not None and (
                self._vision_future is None or self._vision_future.done()
            ):
                frame_copy = self._latest_frame.copy()
                self._vision_future = self._executor.submit(self._vision_worker.infer_relative, frame_copy)

            if self._vision_future is not None and self._vision_future.done():
                try:
                    self._latest_relative, self._latest_confidence = self._vision_future.result()
                except Exception:
                    self._latest_relative = np.zeros(3, dtype=float)
                    self._latest_confidence = 0.0
                finally:
                    self._vision_future = None

            if self._manager_future is None or self._manager_future.done():
                fix = self._latest_fix
                relative = self._latest_relative.copy()
                confidence = float(self._latest_confidence)
                self._manager_future = self._executor.submit(self._core.update, fix, relative, confidence)

            if self._manager_future is not None and self._manager_future.done():
                try:
                    payload = self._manager_future.result()
                except Exception:
                    payload = {
                        "timestamp": time.time(),
                        "status": "ERROR",
                        "spoof_confidence_score": 0.0,
                        "sdr_heatmap": [[0.0] * 8 for _ in range(8)],
                    }
                message = string_msg()
                message.data = to_json_message(payload)
                self._status_pub.publish(message)
                status_text = str(payload.get("status", "STANDBY"))
                with self._status_lock:
                    if status_text != self._last_status:
                        severity = "WARNING" if "ACTIVE" in status_text else "INFO"
                        self._mavlink.send_statustext(status_text, severity=severity)
                        self._last_status = status_text
                self._manager_future = None

        def destroy_node(self) -> bool:
            self._executor.shutdown(wait=False, cancel_futures=True)
            return super().destroy_node()

    rclpy.init()
    node = SpoofManagerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


__all__ = [
    "DefensiveDriftPlanner",
    "GeoFix",
    "HardwareMapping",
    "SDRDryRunInterface",
    "SafetyInterlock",
    "SafetyInterlockConfig",
    "SafetyInterlockDecision",
    "SpoofManagerConfig",
    "SpoofManagerCore",
]


if __name__ == "__main__":
    main()
