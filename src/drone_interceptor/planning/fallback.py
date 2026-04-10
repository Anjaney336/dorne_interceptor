from __future__ import annotations

from typing import Any

import numpy as np

from drone_interceptor.types import Plan, TargetState


class FallbackWaypointPlanner:
    """Simple replanner that retargets the interceptor through an aggressive waypoint."""

    def __init__(self, config: dict[str, Any]) -> None:
        planning = config.get("planning", {})
        mission = config.get("mission", {})
        self._dt = float(mission.get("time_step", 0.1))
        self._max_speed = float(planning.get("max_speed_mps", 20.0))
        self._lead_weight = float(planning.get("fallback_lead_weight", 0.7))
        self._altitude_bias_m = float(planning.get("fallback_altitude_bias_m", 0.0))

    def plan(
        self,
        interceptor_state: TargetState,
        current_target_state: TargetState,
        predicted_target_states: list[TargetState],
    ) -> Plan:
        anchor_state = predicted_target_states[min(len(predicted_target_states) - 1, 2)] if predicted_target_states else current_target_state
        waypoint = (
            (1.0 - self._lead_weight) * np.asarray(current_target_state.position, dtype=float)
            + self._lead_weight * np.asarray(anchor_state.position, dtype=float)
        )
        waypoint = waypoint.astype(float)
        waypoint[2] += self._altitude_bias_m

        offset = waypoint - np.asarray(interceptor_state.position, dtype=float)
        distance = float(np.linalg.norm(offset))
        if distance < 1e-6:
            desired_velocity = np.zeros(3, dtype=float)
        else:
            desired_velocity = offset / distance * self._max_speed
        desired_acceleration = 0.5 * (desired_velocity - np.asarray(interceptor_state.velocity, dtype=float)) / max(self._dt, 1e-6)

        return Plan(
            intercept_point=waypoint,
            desired_velocity=desired_velocity.astype(float),
            desired_acceleration=desired_acceleration.astype(float),
            time_to_intercept=max(distance / max(self._max_speed, 1e-6), self._dt),
            metadata={
                "planner": "fallback_waypoint_planner",
                "target_velocity": np.asarray(current_target_state.velocity, dtype=float).copy(),
                "target_acceleration": (
                    np.asarray(current_target_state.acceleration, dtype=float).copy()
                    if current_target_state.acceleration is not None
                    else np.zeros(3, dtype=float)
                ),
                "target_covariance": (
                    np.asarray(current_target_state.covariance, dtype=float).copy()
                    if current_target_state.covariance is not None
                    else None
                ),
                "replan_reason": "interception_progress_stalled",
            },
        )


__all__ = ["FallbackWaypointPlanner"]
