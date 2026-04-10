from __future__ import annotations

from typing import Any

import numpy as np

from drone_interceptor.constraints import clamp_drift_rate, load_constraint_envelope
from drone_interceptor.types import NavigationState, SensorPacket


def simulate_gps_with_drift(
    true_position: np.ndarray,
    time_s: float,
    drift_rate_mps: float,
) -> np.ndarray:
    position = np.asarray(true_position, dtype=float)
    drift = np.array([drift_rate_mps * time_s, 0.0, 0.0], dtype=float)
    return position + drift


def simulate_axis_drift(
    x_true: float,
    time_s: float,
    drift_rate_mps: float,
) -> float:
    return float(x_true + drift_rate_mps * time_s)


class GPSIMUKalmanFusion:
    """Constant-velocity Kalman filter with IMU acceleration as control input."""

    def __init__(self, config: dict[str, Any]) -> None:
        mission = config["mission"]
        navigation = config["navigation"]
        self._constraint_envelope = load_constraint_envelope(config)
        self._dt = float(mission["time_step"])
        self._gps_noise_std = float(navigation.get("gps_noise_std_m", 1.5))
        self._imu_noise_std = float(navigation.get("imu_noise_std_mps2", 0.15))
        configured_drift = float(navigation.get("gps_drift_rate_mps", 0.2))
        self._drift_rate = clamp_drift_rate(configured_drift, self._constraint_envelope)
        self._process_noise_scale = float(navigation.get("process_noise_scale", 1.0))
        self._measurement_noise_scale = float(navigation.get("measurement_noise_scale", 1.0))
        self._state = np.zeros(6, dtype=float)
        self._covariance = np.eye(6, dtype=float)
        self._initialized = False

    def update(self, packet: SensorPacket) -> NavigationState:
        dt = self._dt
        state_transition = np.block(
            [
                [np.eye(3), dt * np.eye(3)],
                [np.zeros((3, 3)), np.eye(3)],
            ]
        )
        control_matrix = np.vstack((0.5 * (dt**2) * np.eye(3), dt * np.eye(3)))
        process_noise = np.eye(6, dtype=float) * max((self._imu_noise_std**2) * self._process_noise_scale, 1e-4)
        measurement_matrix = np.hstack((np.eye(3), np.zeros((3, 3))))
        measurement_noise = np.eye(3, dtype=float) * max((self._gps_noise_std**2) * self._measurement_noise_scale, 1e-4)

        if not self._initialized:
            self._state[:3] = np.asarray(packet.gps_position, dtype=float)
            if packet.true_velocity is not None:
                self._state[3:] = np.asarray(packet.true_velocity, dtype=float)
            self._initialized = True

        control = np.asarray(packet.imu_acceleration, dtype=float)
        self._state = state_transition @ self._state + control_matrix @ control
        self._covariance = state_transition @ self._covariance @ state_transition.T + process_noise

        measurement = np.asarray(packet.gps_position, dtype=float)
        innovation = measurement - (measurement_matrix @ self._state)
        innovation_covariance = (
            measurement_matrix @ self._covariance @ measurement_matrix.T + measurement_noise
        )
        kalman_gain = self._covariance @ measurement_matrix.T @ np.linalg.inv(innovation_covariance)
        self._state = self._state + kalman_gain @ innovation
        self._covariance = (np.eye(6) - kalman_gain @ measurement_matrix) @ self._covariance

        return NavigationState(
            position=self._state[:3].copy(),
            velocity=self._state[3:].copy(),
            covariance=self._covariance.copy(),
            timestamp=packet.timestamp,
            metadata={
                "gps_drift_rate_mps": self._drift_rate,
                "drift_rate_in_bounds": self._constraint_envelope.drift_rate_min_mps <= self._drift_rate <= self._constraint_envelope.drift_rate_max_mps,
                "source": "gps_imu_kalman",
                "process_noise_scale": self._process_noise_scale,
                "measurement_noise_scale": self._measurement_noise_scale,
            },
        )
