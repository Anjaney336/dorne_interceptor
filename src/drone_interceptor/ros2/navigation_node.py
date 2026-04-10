from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from drone_interceptor.config import load_config
from drone_interceptor.navigation.state_estimator import GPSIMUKalmanFusion
from drone_interceptor.ros2.common import load_rclpy, load_string_message, to_json_message
from drone_interceptor.ros2.runtime import build_navigation_message
from drone_interceptor.types import SensorPacket


def main() -> None:
    parser = argparse.ArgumentParser(description="ROS2 navigation node.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    navigator = GPSIMUKalmanFusion(config)
    rclpy, node_cls = load_rclpy()
    string_msg = load_string_message()

    class NavigationNode(node_cls):
        def __init__(self) -> None:
            super().__init__("drone_interceptor_navigation")
            self.publisher = self.create_publisher(string_msg, "interceptor/navigation/state", 10)
            self.timer = self.create_timer(0.1, self.publish_state)
            self._time_s = 0.0

        def publish_state(self) -> None:
            packet = SensorPacket(
                gps_position=np.array([10.0 + 0.2 * self._time_s, 2.0, -5.0]),
                imu_acceleration=np.array([0.1, 0.0, 0.0]),
                timestamp=self._time_s,
            )
            state = navigator.update(packet)
            message = string_msg()
            message.data = to_json_message(build_navigation_message(state))
            self.publisher.publish(message)
            self._time_s += 0.1

    rclpy.init()
    node = NavigationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
