from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from drone_interceptor.config import load_config
from drone_interceptor.ros2.common import load_rclpy, load_string_message, to_json_message
from drone_interceptor.ros2.mavlink_bridge import MavlinkBridge
from drone_interceptor.ros2.spoof_manager import GeoFix, SpoofManagerConfig, SpoofManagerCore, _load_navsat_msg_type, _safe_float


def main() -> None:
    parser = argparse.ArgumentParser(description="ROS2 SpoofNode (defensive dry-run planner).")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--spoof-enable", action="store_true")
    parser.add_argument("--target-topic", type=str, default="/spoof/target_relative")
    parser.add_argument("--status-topic", type=str, default="/spoof/status")
    parser.add_argument("--mavlink-uri", type=str, default="udpout:127.0.0.1:14550")
    parser.add_argument("--log-path", type=Path, default=Path("logs") / "spoof_manager_telemetry.jsonl")
    args = parser.parse_args()

    _ = load_config(args.config)
    rclpy, node_cls = load_rclpy()
    string_msg = load_string_message()
    navsat_type = _load_navsat_msg_type()

    class SpoofNode(node_cls):
        def __init__(self) -> None:
            super().__init__("drone_interceptor_spoof_node")
            self._publisher = self.create_publisher(string_msg, str(args.status_topic), 10)
            self._mavlink = MavlinkBridge(connection_uri=str(args.mavlink_uri))
            self._core = SpoofManagerCore(
                config=SpoofManagerConfig(
                    spoof_enable=bool(args.spoof_enable),
                    log_path=Path(args.log_path),
                )
            )
            self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="spoof_node")
            self._future: Future[dict[str, Any]] | None = None
            self._latest_fix: GeoFix | None = None
            self._latest_relative = np.zeros(3, dtype=float)
            self._latest_confidence = 0.0
            self._last_status = "STANDBY"
            self._lock = threading.Lock()
            self.create_subscription(navsat_type, "/mavros/global_position/raw", self._on_fix, 10)
            self.create_subscription(string_msg, str(args.target_topic), self._on_target_relative, 10)
            self.create_timer(0.10, self._tick)

        def _on_fix(self, msg: Any) -> None:
            fix = GeoFix(
                lat_deg=_safe_float(getattr(msg, "latitude", 0.0)),
                lon_deg=_safe_float(getattr(msg, "longitude", 0.0)),
                alt_m=_safe_float(getattr(msg, "altitude", 0.0)),
                timestamp_s=time.time(),
            )
            with self._lock:
                self._latest_fix = fix

        def _on_target_relative(self, msg: Any) -> None:
            try:
                payload = json.loads(str(getattr(msg, "data", "{}")))
            except json.JSONDecodeError:
                payload = {}
            relative = np.asarray(payload.get("relative_xyz_m", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)
            if relative.size < 3:
                padded = np.zeros(3, dtype=float)
                padded[: relative.size] = relative
                relative = padded
            confidence = _safe_float(payload.get("confidence", 0.0), 0.0)
            with self._lock:
                self._latest_relative = relative
                self._latest_confidence = confidence

        def _tick(self) -> None:
            with self._lock:
                fix = self._latest_fix
                relative = self._latest_relative.copy()
                confidence = float(self._latest_confidence)
            if fix is None:
                return

            if self._future is None or self._future.done():
                self._future = self._executor.submit(self._core.update, fix, relative, confidence)
                return
            if not self._future.done():
                return

            try:
                payload = self._future.result()
            except Exception:
                payload = {
                    "timestamp": time.time(),
                    "status": "ERROR",
                    "spoof_confidence_score": 0.0,
                    "sdr_heatmap": [[0.0] * 8 for _ in range(8)],
                }
            message = string_msg()
            message.data = to_json_message(payload)
            self._publisher.publish(message)
            status = str(payload.get("status", "STANDBY"))
            if status != self._last_status:
                severity = "WARNING" if "ACTIVE" in status else "INFO"
                self._mavlink.send_statustext(status, severity=severity)
                self._last_status = status
            self._future = None

        def destroy_node(self) -> bool:
            self._executor.shutdown(wait=False, cancel_futures=True)
            return super().destroy_node()

    rclpy.init()
    node = SpoofNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
