from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from drone_interceptor.config import load_config
from drone_interceptor.control.controller import InterceptionController
from drone_interceptor.ros2.common import load_rclpy, load_string_message, to_json_message
from drone_interceptor.types import Plan, TargetState


def main() -> None:
    parser = argparse.ArgumentParser(description="ROS2 control node.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    controller = InterceptionController(config)
    rclpy, node_cls = load_rclpy()
    string_msg = load_string_message()

    class ControlNode(node_cls):
        def __init__(self) -> None:
            super().__init__("drone_interceptor_control")
            self.publisher = self.create_publisher(string_msg, "interceptor/control/command", 10)
            self.timer = self.create_timer(0.1, self.publish_command)

        def publish_command(self) -> None:
            interceptor = TargetState(
                position=np.array([0.0, 0.0, 100.0]),
                velocity=np.array([0.0, 0.0, 0.0]),
                acceleration=np.zeros(3),
            )
            plan = Plan(
                intercept_point=np.array([20.0, 5.0, 102.0]),
                desired_velocity=np.array([6.0, 1.0, 0.2]),
                desired_acceleration=np.array([0.5, 0.0, 0.0]),
                metadata={"target_velocity": np.array([3.0, 0.0, 0.0])},
            )
            command = controller.compute_command(interceptor, plan)
            message = string_msg()
            message.data = to_json_message(
                {
                    "velocity_command": command.velocity_command.tolist(),
                    "acceleration_command": (
                        command.acceleration_command.tolist()
                        if command.acceleration_command is not None
                        else [0.0, 0.0, 0.0]
                    ),
                    "mode": command.mode,
                }
            )
            self.publisher.publish(message)

    rclpy.init()
    node = ControlNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
