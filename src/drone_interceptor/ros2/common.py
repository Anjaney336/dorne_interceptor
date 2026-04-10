from __future__ import annotations

import json
from typing import Any


def load_rclpy() -> tuple[Any, Any]:
    try:
        import rclpy
        from rclpy.node import Node
    except ImportError as exc:
        raise ImportError(
            "ROS2 rclpy is not installed in this environment. "
            "Install ROS2 Humble or Iron with rclpy to run these nodes."
        ) from exc
    return rclpy, Node


def load_string_message() -> Any:
    try:
        from std_msgs.msg import String
    except ImportError as exc:
        raise ImportError(
            "std_msgs is required for the lightweight ROS2 JSON bridge."
        ) from exc
    return String


def to_json_message(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))
