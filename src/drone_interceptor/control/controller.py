from __future__ import annotations

from typing import Any

import numpy as np

from drone_interceptor.constraints import ConstraintModel
from drone_interceptor.control.guidance import ProportionalNavigationGuidance
from drone_interceptor.optimization.trajectory_optimizer import InterceptionTrajectoryOptimizer
from drone_interceptor.types import ControlCommand, Plan, TargetState


class ProportionalNavigationController:
    def __init__(self, config: dict[str, Any]) -> None:
        control = config["control"]
        planning = config["planning"]
        mission = config["mission"]
        self._dt = float(mission["time_step"])
        self._constraints = ConstraintModel(config)
        self._max_speed = float(self._constraints.envelope.max_velocity_mps)
        self._max_acceleration = float(self._constraints.envelope.max_acceleration_mps2)
        self._terminal_gain = float(control.get("terminal_gain", 1.2))
        self._guidance = ProportionalNavigationGuidance(
            dt=self._dt,
            navigation_constant=float(control.get("navigation_constant", 4.5)),
            max_acceleration=self._max_acceleration,
            max_speed=self._max_speed,
            min_closing_speed=float(control.get("min_closing_speed_mps", 0.5)),
            time_to_go_gain=float(control.get("time_to_go_gain", 1.4)),
            target_acceleration_gain=float(control.get("target_acceleration_gain", 0.5)),
            uncertainty_gain=float(control.get("uncertainty_gain", 0.25)),
            constraint_model=self._constraints,
        )

    def compute_command(
        self,
        interceptor_state: TargetState,
        plan: Plan,
    ) -> ControlCommand:
        target_state = TargetState(
            position=np.asarray(plan.intercept_point, dtype=float).copy(),
            velocity=np.asarray(
                plan.metadata.get("target_velocity", np.zeros(3, dtype=float)),
                dtype=float,
            ),
            acceleration=np.asarray(
                plan.metadata.get("target_acceleration", np.zeros(3, dtype=float)),
                dtype=float,
            ),
            track_id=plan.metadata.get("prediction_track_id"),
            metadata={"source": "planner_prediction"},
        )
        tracking_error_m = float(plan.metadata.get("tracking_error_m", 0.0))
        guidance_command = self._guidance.compute_command(
            interceptor_state=interceptor_state,
            target_state=target_state,
            tracking_error_m=tracking_error_m,
        )
        desired_acceleration = np.asarray(
            plan.desired_acceleration if plan.desired_acceleration is not None else np.zeros(3, dtype=float),
            dtype=float,
        )
        velocity_error = plan.desired_velocity - interceptor_state.velocity
        commanded_acceleration = np.asarray(guidance_command.acceleration_command, dtype=float) + self._terminal_gain * velocity_error + 0.2 * desired_acceleration
        commanded_acceleration = _clip_vector(commanded_acceleration, self._max_acceleration)
        constraint_target_state = _constraint_target_state(plan=plan, fallback_target_state=target_state)
        constrained_command, status = self._constraints.enforce_guidance_command(
            interceptor_state=interceptor_state,
            target_state=constraint_target_state,
            raw_acceleration=commanded_acceleration,
            dt=self._dt,
            tracking_error_m=tracking_error_m,
        )
        return ControlCommand(
            velocity_command=constrained_command.velocity_command,
            acceleration_command=constrained_command.acceleration_command,
            mode="pn",
            metadata={
                "controller": "proportional_navigation",
                "time_to_go": guidance_command.metadata["time_to_go"],
                "closing_speed": guidance_command.metadata["closing_speed"],
                "zero_effort_miss": guidance_command.metadata["zero_effort_miss"],
                "relative_position": guidance_command.metadata.get("relative_position", [0.0, 0.0, 0.0]),
                "relative_velocity": guidance_command.metadata.get("relative_velocity", [0.0, 0.0, 0.0]),
                "line_of_sight_rate": guidance_command.metadata.get("line_of_sight_rate", [0.0, 0.0, 0.0]),
                "line_of_sight_rate_norm": guidance_command.metadata.get("line_of_sight_rate_norm", 0.0),
                "navigation_constant": guidance_command.metadata.get("navigation_constant", 0.0),
                "tracking_ok": status.tracking_ok,
                "safety_override": status.safety_override,
                "velocity_clipped": status.velocity_clipped,
                "acceleration_clipped": status.acceleration_clipped,
                "objective": "minimize_interception_time",
            },
        )


class MPCController:
    def __init__(self, config: dict[str, Any]) -> None:
        control = config["control"]
        planning = config["planning"]
        mission = config["mission"]
        self._dt = float(mission["time_step"])
        self._constraints = ConstraintModel(config)
        self._horizon_steps = int(control.get("mpc_horizon_steps", 12))
        self._candidate_levels = int(control.get("mpc_candidate_acceleration_levels", 7))
        self._max_acceleration = float(self._constraints.envelope.max_acceleration_mps2)
        self._max_speed = float(self._constraints.envelope.max_velocity_mps)
        self._position_weight = float(control.get("mpc_position_weight", 1.0))
        self._velocity_weight = float(control.get("mpc_velocity_weight", 0.3))
        self._control_weight = float(control.get("mpc_control_weight", 0.05))
        self._guidance_blend = float(control.get("mpc_guidance_blend", 0.45))
        self._optimizer = InterceptionTrajectoryOptimizer(
            config=config,
            horizon_steps=self._horizon_steps,
            num_trajectories=int(control.get("mpc_num_trajectories", config.get("optimization", {}).get("num_trajectories", 25))),
        )
        self._guidance = ProportionalNavigationGuidance(
            dt=self._dt,
            navigation_constant=float(control.get("navigation_constant", 4.5)),
            max_acceleration=self._max_acceleration,
            max_speed=self._max_speed,
            min_closing_speed=float(control.get("min_closing_speed_mps", 0.5)),
            time_to_go_gain=float(control.get("time_to_go_gain", 1.4)),
            target_acceleration_gain=float(control.get("target_acceleration_gain", 0.5)),
            uncertainty_gain=float(control.get("uncertainty_gain", 0.25)),
            constraint_model=self._constraints,
        )

    def compute_command(
        self,
        interceptor_state: TargetState,
        plan: Plan,
    ) -> ControlCommand:
        target_velocity = np.asarray(plan.metadata.get("target_velocity", np.zeros(3)), dtype=float)
        target_acceleration = np.asarray(plan.metadata.get("target_acceleration", np.zeros(3)), dtype=float)
        target_state = TargetState(
            position=np.asarray(plan.intercept_point, dtype=float).copy(),
            velocity=target_velocity.copy(),
            acceleration=target_acceleration.copy(),
            covariance=(
                np.asarray(plan.metadata["target_covariance"], dtype=float).copy()
                if plan.metadata.get("target_covariance") is not None
                else None
            ),
            metadata={"source": "planner_prediction"},
        )
        guidance_command = self._guidance.compute_command(
            interceptor_state=interceptor_state,
            target_state=target_state,
            tracking_error_m=float(plan.metadata.get("tracking_error_m", 0.0)),
        )
        target_covariance = plan.metadata.get("target_covariance")
        optimization_result = self._optimizer.optimize(
            interceptor_state=np.array(
                [
                    interceptor_state.position[0],
                    interceptor_state.position[1],
                    interceptor_state.velocity[0],
                    interceptor_state.velocity[1],
                ],
                dtype=float,
            ),
            target_state=np.array(
                [
                    plan.intercept_point[0],
                    plan.intercept_point[1],
                    float(target_velocity[0]),
                    float(target_velocity[1]),
                ],
                dtype=float,
            ),
            target_acceleration=target_acceleration[:2],
            target_covariance=(
                np.asarray(target_covariance, dtype=float)
                if target_covariance is not None
                else None
            ),
        )
        best_acceleration_xy = optimization_result.optimal_controls[0]
        desired_acceleration = np.asarray(
            plan.desired_acceleration if plan.desired_acceleration is not None else np.zeros(3, dtype=float),
            dtype=float,
        )
        blended_xy = (
            (1.0 - self._guidance_blend) * best_acceleration_xy
            + self._guidance_blend * np.asarray(guidance_command.acceleration_command[:2], dtype=float)
        )
        best_acceleration = np.array(
            [
                blended_xy[0],
                blended_xy[1],
                0.35 * desired_acceleration[2],
            ],
            dtype=float,
        )
        best_acceleration = _clip_vector(best_acceleration + 0.15 * desired_acceleration, self._max_acceleration)
        constraint_target_state = _constraint_target_state(plan=plan, fallback_target_state=target_state)
        constrained_command, status = self._constraints.enforce_guidance_command(
            interceptor_state=interceptor_state,
            target_state=constraint_target_state,
            raw_acceleration=best_acceleration,
            dt=self._dt,
            tracking_error_m=float(plan.metadata.get("tracking_error_m", 0.0)),
        )
        return ControlCommand(
            velocity_command=constrained_command.velocity_command,
            acceleration_command=constrained_command.acceleration_command,
            mode="mpc",
            metadata={
                "controller": "mpc",
                "cost": optimization_result.optimal_cost,
                "evaluated_trajectories": optimization_result.evaluated_trajectories,
                "best_candidate_index": optimization_result.best_candidate_index,
                "uncertainty_trace": float(plan.metadata.get("uncertainty_trace", 0.0)),
                "time_to_go": guidance_command.metadata["time_to_go"],
                "closing_speed": guidance_command.metadata["closing_speed"],
                "relative_position": guidance_command.metadata.get("relative_position", [0.0, 0.0, 0.0]),
                "relative_velocity": guidance_command.metadata.get("relative_velocity", [0.0, 0.0, 0.0]),
                "line_of_sight_rate": guidance_command.metadata.get("line_of_sight_rate", [0.0, 0.0, 0.0]),
                "line_of_sight_rate_norm": guidance_command.metadata.get("line_of_sight_rate_norm", 0.0),
                "navigation_constant": guidance_command.metadata.get("navigation_constant", 0.0),
                "tracking_ok": status.tracking_ok,
                "safety_override": status.safety_override,
                "velocity_clipped": status.velocity_clipped,
                "acceleration_clipped": status.acceleration_clipped,
            },
        )


class InterceptionController:
    """Runtime-selectable controller facade for PN and lightweight MPC."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config["control"]
        mode = str(self._config.get("mode", "pn")).lower()
        self._backend = MPCController(config) if mode == "mpc" else ProportionalNavigationController(config)

    def compute_command(
        self,
        interceptor_state: TargetState,
        plan: Plan,
    ) -> ControlCommand:
        return self._backend.compute_command(interceptor_state=interceptor_state, plan=plan)


def _clip_vector(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm < 1e-6:
        return vector.astype(float)
    return vector.astype(float) / norm * max_norm


def _constraint_target_state(plan: Plan, fallback_target_state: TargetState) -> TargetState:
    current_position = plan.metadata.get("current_target_position")
    if current_position is None:
        return fallback_target_state

    fallback_acceleration = (
        fallback_target_state.acceleration
        if fallback_target_state.acceleration is not None
        else np.zeros(3, dtype=float)
    )
    current_covariance = plan.metadata.get("current_target_covariance")
    return TargetState(
        position=np.asarray(current_position, dtype=float).copy(),
        velocity=np.asarray(plan.metadata.get("current_target_velocity", fallback_target_state.velocity), dtype=float).copy(),
        acceleration=np.asarray(plan.metadata.get("current_target_acceleration", fallback_acceleration), dtype=float).copy(),
        covariance=(
            np.asarray(current_covariance, dtype=float).copy()
            if current_covariance is not None
            else fallback_target_state.covariance
        ),
        track_id=plan.metadata.get("prediction_track_id"),
        metadata={"source": "current_target_estimate"},
    )
