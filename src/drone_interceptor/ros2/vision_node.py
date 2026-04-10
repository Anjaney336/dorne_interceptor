from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from drone_interceptor.config import load_config
from drone_interceptor.ros2.common import load_rclpy, load_string_message, to_json_message
from drone_interceptor.ros2.spoof_manager import VisionInferenceWorker, _image_msg_to_bgr, _load_image_msg_type


def main() -> None:
    parser = argparse.ArgumentParser(description="ROS2 VisionNode for target-relative telemetry.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--publish-topic", type=str, default="/spoof/target_relative")
    args = parser.parse_args()

    config = load_config(args.config)
    rclpy, node_cls = load_rclpy()
    string_msg = load_string_message()
    image_type = _load_image_msg_type()

    class VisionNode(node_cls):
        def __init__(self) -> None:
            super().__init__("drone_interceptor_vision_node")
            self._publisher = self.create_publisher(string_msg, str(args.publish_topic), 10)
            self._worker = VisionInferenceWorker(config)
            self._latest_frame: np.ndarray | None = None
            self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vision_node")
            self._future: Future[tuple[np.ndarray, float]] | None = None
            self._lock = threading.Lock()
            self.create_subscription(image_type, "/camera/image_raw", self._on_image, 10)
            self.create_timer(0.05, self._tick)

        def _on_image(self, msg: Any) -> None:
            frame = _image_msg_to_bgr(msg)
            if frame is None:
                return
            with self._lock:
                self._latest_frame = frame

        def _tick(self) -> None:
            with self._lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
            if frame is not None and (self._future is None or self._future.done()):
                self._future = self._executor.submit(self._worker.infer_relative, frame)
            if self._future is None or not self._future.done():
                return
            try:
                relative, confidence = self._future.result()
            except Exception:
                relative = np.zeros(3, dtype=float)
                confidence = 0.0
            payload = {
                "timestamp": time.time(),
                "relative_xyz_m": np.asarray(relative, dtype=float).tolist(),
                "confidence": float(confidence),
                "target_acquired": bool(confidence >= 0.45),
                "model": "YOLOv10-tiny",
            }
            message = string_msg()
            message.data = to_json_message(payload)
            self._publisher.publish(message)
            self._future = None

        def destroy_node(self) -> bool:
            self._executor.shutdown(wait=False, cancel_futures=True)
            return super().destroy_node()

    rclpy.init()
    node = VisionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
