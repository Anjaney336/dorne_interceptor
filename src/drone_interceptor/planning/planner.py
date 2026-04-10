from __future__ import annotations

from typing import Any

import numpy as np

from drone_interceptor.types import Plan, TargetState


class InterceptPlanner:
    """Chooses a feasible intercept point from the prediction horizon."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config["planning"]
        self._mission = config["mission"]
        self._dt = float(self._mission["time_step"])
        optimization = config.get("optimization", {})
        self._uncertainty_weight = float(
            self._config.get("uncertainty_weight", optimization.get("gamma", optimization.get("uncertainty_weight", 1.0)))
        )

    def plan(
        self,
        interceptor_state: TargetState,
        predicted_target_states: list[TargetState],
    ) -> Plan:
        max_speed = float(self._config["max_speed_mps"])
        best_target_state = predicted_target_states[0]
        best_score = float("inf")

        for index, target_state in enumerate(predicted_target_states, start=1):
            offset = target_state.position - interceptor_state.position
            distance = np.linalg.norm(offset)
            time_to_go = max(index * self._dt, self._dt)
            required_speed = distance / time_to_go
            feasibility_penalty = 0.0 if required_speed <= max_speed else (required_speed - max_speed) * 5.0
            covariance_trace = _covariance_trace(target_state.covariance)
            score = distance + feasibility_penalty + self._uncertainty_weight * covariance_trace
            if score < best_score:
                best_score = score
                best_target_state = target_state

        offset = best_target_state.position - interceptor_state.position
        distance = np.linalg.norm(offset)
        time_to_intercept = max(float(distance / max(max_speed, 1e-6)), self._dt)
        if distance < 1e-6:
            desired_velocity = np.zeros_like(offset)
            desired_acceleration = np.zeros_like(offset)
        else:
            desired_velocity = (offset / distance) * min(distance / time_to_intercept, max_speed)
            target_acceleration = (
                np.asarray(best_target_state.acceleration, dtype=float)
                if best_target_state.acceleration is not None
                else np.zeros_like(offset)
            )
            desired_acceleration = 0.5 * target_acceleration + 0.3 * (desired_velocity - interceptor_state.velocity)
        return Plan(
            intercept_point=best_target_state.position.copy(),
            desired_velocity=desired_velocity,
            desired_acceleration=desired_acceleration,
            time_to_intercept=time_to_intercept,
            metadata={
                "planner": "intercept_planner",
                "target_velocity": best_target_state.velocity.copy(),
                "target_acceleration": (
                    best_target_state.acceleration.copy()
                    if best_target_state.acceleration is not None
                    else np.zeros_like(offset)
                ),
                "target_covariance": (
                    None
                    if best_target_state.covariance is None
                    else np.asarray(best_target_state.covariance, dtype=float).copy()
                ),
                "uncertainty_trace": _covariance_trace(best_target_state.covariance),
                "prediction_track_id": best_target_state.track_id,
            },
        )


def _covariance_trace(covariance: np.ndarray | None) -> float:
    if covariance is None:
        return 0.0
    covariance_matrix = np.asarray(covariance, dtype=float)
    return float(np.trace(covariance_matrix))
