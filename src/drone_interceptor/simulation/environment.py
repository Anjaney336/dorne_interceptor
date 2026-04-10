from __future__ import annotations

from typing import Any

import numpy as np

from drone_interceptor.constraints import load_constraint_envelope
from drone_interceptor.types import ControlCommand, SensorPacket, TargetState


class DroneInterceptionEnv:
    """Kinematic multi-sensor environment for interceptor research."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._mission = config["mission"]
        self._simulation = config["simulation"]
        self._constraint_envelope = load_constraint_envelope(config)
        self._dt = float(self._mission["time_step"])
        self._intercept_distance = float(config["planning"]["desired_intercept_distance_m"])
        self._rng = np.random.default_rng(int(config.get("system", {}).get("random_seed", 7)))
        self._gps_drift_rate = float(config.get("navigation", {}).get("gps_drift_rate_mps", 0.02))
        self._gps_noise_std = float(config.get("navigation", {}).get("gps_noise_std_m", 1.5))
        self._imu_noise_std = float(config.get("navigation", {}).get("imu_noise_std_mps2", 0.15))
        self._target_accel_limit = float(self._simulation.get("target_max_acceleration_mps2", 4.0))
        self._interceptor_process_noise_std = float(self._simulation.get("interceptor_process_noise_std_mps2", 0.2))
        self._target_process_noise_std = float(self._simulation.get("target_process_noise_std_mps2", 0.35))
        self._wind_disturbance_std = float(self._simulation.get("wind_disturbance_std_mps2", 0.15))
        self.interceptor_state: TargetState
        self.target_state: TargetState
        self._step_count = 0
        self._time_s = 0.0
        self._last_interceptor_acceleration = np.zeros(3, dtype=float)
        self._last_target_acceleration = np.zeros(3, dtype=float)

    def reset(self) -> dict[str, np.ndarray]:
        self.interceptor_state = TargetState(
            position=np.array(self._simulation["interceptor_initial_position"], dtype=float),
            velocity=np.array(self._simulation["interceptor_initial_velocity"], dtype=float),
            acceleration=np.zeros(3, dtype=float),
        )
        self.target_state = TargetState(
            position=np.array(self._simulation["target_initial_position"], dtype=float),
            velocity=np.array(self._simulation["target_initial_velocity"], dtype=float),
            acceleration=np.zeros(3, dtype=float),
        )
        self._step_count = 0
        self._time_s = 0.0
        self._last_interceptor_acceleration = np.zeros(3, dtype=float)
        self._last_target_acceleration = np.zeros(3, dtype=float)
        return self._observation()

    def step(self, command: ControlCommand) -> tuple[dict[str, np.ndarray], bool, dict[str, float]]:
        interceptor_acceleration = self._resolve_interceptor_acceleration(command) + self._sample_process_disturbance(
            std=self._interceptor_process_noise_std
        )
        target_acceleration = self._sample_target_acceleration() + self._sample_process_disturbance(
            std=self._target_process_noise_std
        )
        wind_acceleration = self._sample_process_disturbance(std=self._wind_disturbance_std)
        interceptor_acceleration = interceptor_acceleration + 0.5 * wind_acceleration
        target_acceleration = target_acceleration + wind_acceleration
        interceptor_acceleration = _clip_vector(
            interceptor_acceleration,
            self._constraint_envelope.max_acceleration_mps2,
        )

        self.interceptor_state.position = self._propagate_position(
            self.interceptor_state.position,
            self.interceptor_state.velocity,
            interceptor_acceleration,
        )
        self.target_state.position = self._propagate_position(
            self.target_state.position,
            self.target_state.velocity,
            target_acceleration,
        )
        self.interceptor_state.velocity = self.interceptor_state.velocity + interceptor_acceleration * self._dt
        self.interceptor_state.velocity = _clip_vector(
            self.interceptor_state.velocity,
            self._constraint_envelope.max_velocity_mps,
        )
        self.target_state.velocity = self.target_state.velocity + target_acceleration * self._dt
        self.interceptor_state.acceleration = interceptor_acceleration.copy()
        self.target_state.acceleration = target_acceleration.copy()
        self._step_count += 1
        self._time_s += self._dt
        self._last_interceptor_acceleration = interceptor_acceleration.copy()
        self._last_target_acceleration = target_acceleration.copy()

        distance_to_target = float(
            np.linalg.norm(self.target_state.position - self.interceptor_state.position)
        )
        done = distance_to_target <= self._intercept_distance
        truncated = self._step_count >= int(self._mission["max_steps"])

        info = {
            "distance_to_target": distance_to_target,
            "truncated": float(truncated),
            "wind_acceleration_norm": float(np.linalg.norm(wind_acceleration)),
        }
        return self._observation(), bool(done or truncated), info

    def _observation(self) -> dict[str, np.ndarray]:
        sensor_packet = self._sensor_packet()
        return {
            "target_position": self.target_state.position.copy(),
            "target_velocity": self.target_state.velocity.copy(),
            "target_acceleration": (
                self.target_state.acceleration.copy()
                if self.target_state.acceleration is not None
                else np.zeros(3, dtype=float)
            ),
            "interceptor_position": self.interceptor_state.position.copy(),
            "interceptor_velocity": self.interceptor_state.velocity.copy(),
            "time": np.array([self._time_s], dtype=float),
            "gps_position": sensor_packet.gps_position.copy(),
            "imu_acceleration": sensor_packet.imu_acceleration.copy(),
            "sensor_packet": sensor_packet,
        }

    def _resolve_interceptor_acceleration(self, command: ControlCommand) -> np.ndarray:
        if command.acceleration_command is not None:
            return np.asarray(command.acceleration_command, dtype=float)
        desired_velocity = np.asarray(command.velocity_command, dtype=float)
        return (desired_velocity - self.interceptor_state.velocity) / self._dt

    def _sample_target_acceleration(self) -> np.ndarray:
        direction = self._rng.normal(size=3)
        direction[2] *= 0.2
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.zeros(3, dtype=float)
        scale = self._rng.uniform(0.1, self._target_accel_limit)
        return direction / norm * scale

    def _sample_process_disturbance(self, std: float) -> np.ndarray:
        if std <= 0.0:
            return np.zeros(3, dtype=float)
        disturbance = self._rng.normal(0.0, std, size=3)
        disturbance[2] *= 0.4
        return disturbance.astype(float)

    def _propagate_position(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
    ) -> np.ndarray:
        return position + velocity * self._dt + 0.5 * acceleration * (self._dt**2)

    def _sensor_packet(self) -> SensorPacket:
        gps_bias = np.array([self._gps_drift_rate * self._time_s, 0.0, 0.0], dtype=float)
        gps_noise = self._rng.normal(0.0, self._gps_noise_std, size=3)
        imu_noise = self._rng.normal(0.0, self._imu_noise_std, size=3)
        return SensorPacket(
            gps_position=self.interceptor_state.position + gps_bias + gps_noise,
            imu_acceleration=self._last_interceptor_acceleration + imu_noise,
            timestamp=self._time_s,
            true_position=self.interceptor_state.position.copy(),
            true_velocity=self.interceptor_state.velocity.copy(),
            metadata={
                "gps_bias": gps_bias.tolist(),
                "gps_noise_std_m": self._gps_noise_std,
                "imu_noise_std_mps2": self._imu_noise_std,
                "interceptor_process_noise_std_mps2": self._interceptor_process_noise_std,
                "target_process_noise_std_mps2": self._target_process_noise_std,
                "wind_disturbance_std_mps2": self._wind_disturbance_std,
            },
        )


def _clip_vector(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm < 1e-6:
        return vector.astype(float)
    return vector.astype(float) / norm * max_norm
