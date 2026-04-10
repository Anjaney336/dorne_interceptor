from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_interceptor.constraints import ConstraintModel
from drone_interceptor.types import ControlCommand, TargetState


@dataclass(slots=True)
class GuidanceSolution:
    acceleration_command: np.ndarray
    velocity_command: np.ndarray
    time_to_go: float
    closing_speed: float
    zero_effort_miss: np.ndarray
    relative_position: np.ndarray
    relative_velocity: np.ndarray
    line_of_sight_rate: np.ndarray
    navigation_constant: float


class ProportionalNavigationGuidance:
    """Uncertainty-aware augmented proportional navigation guidance."""

    def __init__(
        self,
        dt: float,
        navigation_constant: float = 3.0,
        max_acceleration: float = 18.0,
        max_speed: float = 55.0,
        min_closing_speed: float = 0.5,
        time_to_go_gain: float = 1.0,
        target_acceleration_gain: float = 0.5,
        uncertainty_gain: float = 0.25,
        constraint_model: ConstraintModel | None = None,
    ) -> None:
        self._dt = float(dt)
        self._navigation_constant = float(navigation_constant)
        self._max_acceleration = float(max_acceleration)
        self._max_speed = float(max_speed)
        self._min_closing_speed = float(min_closing_speed)
        self._time_to_go_gain = float(time_to_go_gain)
        self._target_acceleration_gain = float(target_acceleration_gain)
        self._uncertainty_gain = float(uncertainty_gain)
        self._constraint_model = constraint_model

    def compute_command(
        self,
        interceptor_state: TargetState,
        target_state: TargetState,
        tracking_error_m: float = 0.0,
    ) -> ControlCommand:
        solution = self.solve(interceptor_state=interceptor_state, target_state=target_state)
        if self._constraint_model is not None:
            constrained_command, status = self._constraint_model.enforce_guidance_command(
                interceptor_state=interceptor_state,
                target_state=target_state,
                raw_acceleration=solution.acceleration_command,
                dt=self._dt,
                tracking_error_m=tracking_error_m,
            )
            constrained_command.metadata.update(
                {
                    "controller": "proportional_navigation",
                    "time_to_go": solution.time_to_go,
                    "closing_speed": solution.closing_speed,
                    "zero_effort_miss": solution.zero_effort_miss.tolist(),
                    "relative_position": solution.relative_position.tolist(),
                    "relative_velocity": solution.relative_velocity.tolist(),
                    "line_of_sight_rate": solution.line_of_sight_rate.tolist(),
                    "line_of_sight_rate_norm": float(np.linalg.norm(solution.line_of_sight_rate)),
                    "navigation_constant": solution.navigation_constant,
                    "tracking_ok": status.tracking_ok,
                    "safety_override": status.safety_override,
                    "velocity_clipped": status.velocity_clipped,
                    "acceleration_clipped": status.acceleration_clipped,
                    "distance_to_target_m": status.distance_to_target_m,
                }
            )
            return constrained_command
        return ControlCommand(
            velocity_command=solution.velocity_command,
            acceleration_command=solution.acceleration_command,
            mode="pn",
            metadata={
                "controller": "proportional_navigation",
                "time_to_go": solution.time_to_go,
                "closing_speed": solution.closing_speed,
                "zero_effort_miss": solution.zero_effort_miss.tolist(),
                "relative_position": solution.relative_position.tolist(),
                "relative_velocity": solution.relative_velocity.tolist(),
                "line_of_sight_rate": solution.line_of_sight_rate.tolist(),
                "line_of_sight_rate_norm": float(np.linalg.norm(solution.line_of_sight_rate)),
                "navigation_constant": solution.navigation_constant,
            },
        )

    def solve(
        self,
        interceptor_state: TargetState,
        target_state: TargetState,
    ) -> GuidanceSolution:
        interceptor_position = self._as_3d(interceptor_state.position)
        interceptor_velocity = self._as_3d(interceptor_state.velocity)
        target_position = self._as_3d(target_state.position)
        target_velocity = self._as_3d(target_state.velocity)
        target_acceleration = (
            self._as_3d(target_state.acceleration)
            if target_state.acceleration is not None
            else np.zeros(3, dtype=float)
        )

        relative_position = target_position - interceptor_position
        relative_velocity = target_velocity - interceptor_velocity
        distance = max(float(np.linalg.norm(relative_position)), 1e-6)
        line_of_sight = relative_position / distance
        projected_relative_velocity = relative_velocity + 0.5 * target_acceleration * self._dt
        closing_speed = max(-float(np.dot(projected_relative_velocity, line_of_sight)), self._min_closing_speed)
        projected_range_rate = max(-float(np.dot(relative_velocity, line_of_sight)), self._min_closing_speed)
        blended_closing_speed = max(0.5 * (closing_speed + projected_range_rate), self._min_closing_speed)
        time_to_go = max(distance / blended_closing_speed, self._dt)
        covariance_trace = _planar_covariance_trace(target_state.covariance)
        adaptive_navigation_constant = self._navigation_constant * (1.0 + self._uncertainty_gain * min(covariance_trace, 4.0))

        line_of_sight_rate = np.cross(relative_position, relative_velocity) / (distance**2)
        lateral_acceleration = (
            adaptive_navigation_constant
            * closing_speed
            * np.cross(line_of_sight_rate, line_of_sight)
        )
        target_lateral_acceleration = target_acceleration - np.dot(target_acceleration, line_of_sight) * line_of_sight

        zero_effort_miss = relative_position + relative_velocity * time_to_go
        pursuit_acceleration = (
            self._time_to_go_gain
            * zero_effort_miss
            / max(time_to_go**2, self._dt**2)
        )

        commanded_acceleration = (
            lateral_acceleration
            + pursuit_acceleration
            + self._target_acceleration_gain * target_lateral_acceleration
        )
        commanded_acceleration = _clip_vector(commanded_acceleration, self._max_acceleration)
        velocity_command = interceptor_velocity + commanded_acceleration * self._dt
        velocity_command = _clip_vector(velocity_command, self._max_speed)

        return GuidanceSolution(
            acceleration_command=commanded_acceleration,
            velocity_command=velocity_command,
            time_to_go=time_to_go,
            closing_speed=blended_closing_speed,
            zero_effort_miss=zero_effort_miss,
            relative_position=relative_position,
            relative_velocity=relative_velocity,
            line_of_sight_rate=line_of_sight_rate,
            navigation_constant=adaptive_navigation_constant,
        )

    def _as_3d(self, vector: np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=float).reshape(-1)
        if array.shape == (3,):
            return array
        if array.shape == (2,):
            return np.array([array[0], array[1], 0.0], dtype=float)
        raise ValueError("Guidance states must be 2D or 3D vectors.")


def _clip_vector(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm < 1e-6:
        return vector.astype(float)
    return vector.astype(float) / norm * max_norm


def _planar_covariance_trace(covariance: np.ndarray | None) -> float:
    if covariance is None:
        return 0.0
    covariance_matrix = np.asarray(covariance, dtype=float)
    if covariance_matrix.shape == (4, 4):
        return float(np.trace(covariance_matrix[:2, :2]))
    if covariance_matrix.shape == (3, 3):
        return float(np.trace(covariance_matrix[:2, :2]))
    return float(np.trace(covariance_matrix))
