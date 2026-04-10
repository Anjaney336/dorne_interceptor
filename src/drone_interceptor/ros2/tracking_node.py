from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from drone_interceptor.config import load_config
from drone_interceptor.ros2.common import load_rclpy, load_string_message, to_json_message
from drone_interceptor.ros2.runtime import build_tracking_message
from drone_interceptor.tracking.tracker import TargetTracker
from drone_interceptor.types import Detection


def main() -> None:
    parser = argparse.ArgumentParser(description="ROS2 tracking node.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    tracker = TargetTracker(config)
    rclpy, node_cls = load_rclpy()
    string_msg = load_string_message()

    class TrackingNode(node_cls):
        def __init__(self) -> None:
            super().__init__("drone_interceptor_tracking")
            self.publisher = self.create_publisher(string_msg, "interceptor/tracking/state", 10)
            self.timer = self.create_timer(0.1, self.publish_track)
            self._time_s = 0.0

        def publish_track(self) -> None:
            detection = Detection(
                position=np.array([20.0 - 0.4 * self._time_s, 5.0 + 0.1 * self._time_s, 100.0]),
                confidence=0.95,
                metadata={"backend": "synthetic_ros2"},
                timestamp=self._time_s,
            )
            track = tracker.update(detection)
            message = string_msg()
            message.data = to_json_message(build_tracking_message(track))
            self.publisher.publish(message)
            self._time_s += 0.1

    rclpy.init()
    node = TrackingNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
