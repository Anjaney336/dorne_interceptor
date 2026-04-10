from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Detection:
    position: np.ndarray
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float | None = None


@dataclass(slots=True)
class TargetState:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray | None = None
    covariance: np.ndarray | None = None
    timestamp: float | None = None
    track_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Plan:
    intercept_point: np.ndarray
    desired_velocity: np.ndarray
    desired_acceleration: np.ndarray | None = None
    time_to_intercept: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ControlCommand:
    velocity_command: np.ndarray
    acceleration_command: np.ndarray | None = None
    yaw_rate_command: float = 0.0
    mode: str = "velocity"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SensorPacket:
    gps_position: np.ndarray
    imu_acceleration: np.ndarray
    timestamp: float
    true_position: np.ndarray | None = None
    true_velocity: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NavigationState:
    position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray | None = None
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PredictionResult:
    predicted_states: list[TargetState]
    model_name: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
