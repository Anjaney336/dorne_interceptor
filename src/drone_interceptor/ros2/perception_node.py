from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from drone_interceptor.config import load_config
from drone_interceptor.perception.detector import TargetDetector
from drone_interceptor.ros2.common import load_rclpy, load_string_message, to_json_message


def main() -> None:
    parser = argparse.ArgumentParser(description="ROS2 perception node.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--image", type=Path, help="Optional bootstrap image for dry publishing.")
    args = parser.parse_args()

    config = load_config(args.config)
    detector = TargetDetector(config)
    rclpy, node_cls = load_rclpy()
    string_msg = load_string_message()

    class PerceptionNode(node_cls):
        def __init__(self) -> None:
            super().__init__("drone_interceptor_perception")
            self.publisher = self.create_publisher(string_msg, "interceptor/perception/detections", 10)
            self.timer = self.create_timer(0.1, self.publish_detection)

        def publish_detection(self) -> None:
            if args.image is None:
                return
            frame = cv2.imread(str(args.image))
            if frame is None:
                return
            detection = detector.detect({"frame": frame})
            message = string_msg()
            message.data = to_json_message(
                {
                    "position": detection.position.tolist(),
                    "confidence": detection.confidence,
                    "metadata": detection.metadata,
                }
            )
            self.publisher.publish(message)

    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
