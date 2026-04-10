from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from drone_interceptor.constraints import ConstraintStatus
from drone_interceptor.control.controller import InterceptionController
from drone_interceptor.navigation.state_estimator import GPSIMUKalmanFusion
from drone_interceptor.optimization.cost import InterceptionCostModel
from drone_interceptor.perception.detector import TargetDetector
from drone_interceptor.planning.planner import InterceptPlanner
from drone_interceptor.prediction.predictor import TargetPredictor
from drone_interceptor.simulation.environment import DroneInterceptionEnv
from drone_interceptor.tracking.tracker import TargetTracker
from drone_interceptor.types import TargetState
from drone_interceptor.visualization.dashboard import (
    plot_mission_dashboard,
    plot_trajectory_3d,
)


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SystemRunResult:
    steps_executed: int
    intercepted: bool
    mean_loop_fps: float
    output_paths: list[Path]
    final_distance_m: float
    total_cost: float


class AutonomousInterceptorSystem:
    """End-to-end autonomy stack for perception, prediction, control, and visualization."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._env = DroneInterceptionEnv(config)
        self._detector = TargetDetector(config)
        self._tracker = TargetTracker(config)
        self._predictor = TargetPredictor(config)
        self._planner = InterceptPlanner(config)
        self._controller = InterceptionController(config)
        self._navigator = GPSIMUKalmanFusion(config)
        self._cost_model = InterceptionCostModel.from_config(config)
        self._visualization = config.get("visualization", {})
        self._output_dir = Path(self._visualization.get("output_dir", "outputs"))

    def run(self) -> SystemRunResult:
        observation = self._env.reset()
        max_steps = int(self._config["mission"]["max_steps"])
        interceptor_positions: list[np.ndarray] = []
        target_positions: list[np.ndarray] = []
        measured_target_positions: list[np.ndarray] = []
        filtered_target_positions: list[np.ndarray] = []
        times: list[float] = []
        distances: list[float] = []
        speed_commands: list[float] = []
        output_paths: list[Path] = []
        start_time = time.perf_counter()
        done = False
        final_distance = float("inf")
        completed_steps = 0
        total_cost = 0.0

        for step in range(max_steps):
            navigation_state = self._navigator.update(observation["sensor_packet"])
            interceptor_estimate = TargetState(
                position=navigation_state.position.copy(),
                velocity=navigation_state.velocity.copy(),
                covariance=navigation_state.covariance,
                timestamp=navigation_state.timestamp,
                metadata=dict(navigation_state.metadata),
            )
            detection = self._detector.detect(observation)
            track = self._tracker.update(detection)
            prediction = self._predictor.predict(track)
            plan = self._planner.plan(interceptor_estimate, prediction)
            plan.metadata["current_target_position"] = track.position.copy()
            plan.metadata["current_target_velocity"] = track.velocity.copy()
            plan.metadata["current_target_acceleration"] = (
                track.acceleration.copy() if track.acceleration is not None else np.zeros(3, dtype=float)
            )
            plan.metadata["current_target_covariance"] = (
                None if track.covariance is None else np.asarray(track.covariance, dtype=float).copy()
            )
            plan.metadata["tracking_error_m"] = float(
                np.linalg.norm(track.position - self._env.target_state.position)
            )
            command = self._controller.compute_command(interceptor_estimate, plan)
            observation, done, info = self._env.step(command)
            constraint_status = ConstraintStatus(
                velocity_clipped=bool(command.metadata.get("velocity_clipped", False)),
                acceleration_clipped=bool(command.metadata.get("acceleration_clipped", False)),
                tracking_ok=bool(command.metadata.get("tracking_ok", True)),
                drift_rate_in_bounds=bool(navigation_state.metadata.get("drift_rate_in_bounds", True)),
                safety_override=bool(command.metadata.get("safety_override", False)),
                distance_to_target_m=float(info["distance_to_target"]),
            )
            stage_cost = self._cost_model.stage_cost(
                interceptor_position=self._env.interceptor_state.position,
                target_position=self._env.target_state.position,
                control_input=(
                    command.acceleration_command
                    if command.acceleration_command is not None
                    else command.velocity_command
                ),
                constraint_status=constraint_status,
                uncertainty_term=float(plan.metadata.get("uncertainty_trace", 0.0)),
            )
            total_cost += stage_cost

            interceptor_positions.append(self._env.interceptor_state.position.copy())
            target_positions.append(self._env.target_state.position.copy())
            measured_target_positions.append(detection.position.copy())
            filtered_target_positions.append(track.position.copy())
            times.append(float(observation["time"][0]))
            distances.append(float(info["distance_to_target"]))
            speed_commands.append(float(np.linalg.norm(command.velocity_command)))
            final_distance = float(info["distance_to_target"])
            completed_steps = step + 1

            LOGGER.info(
                "step=%s distance=%.2f cost=%.2f uncertainty=%.4f controller=%s predictor=%s",
                step,
                final_distance,
                stage_cost,
                float(plan.metadata.get("uncertainty_trace", 0.0)),
                command.metadata.get("controller", command.mode),
                self._predictor.model_name,
            )

            if done:
                break

        elapsed = max(time.perf_counter() - start_time, 1e-6)
        mean_loop_fps = completed_steps / elapsed

        if interceptor_positions and self._visualization.get("save_outputs", True):
            target_array = np.asarray(target_positions, dtype=float)
            interceptor_array = np.asarray(interceptor_positions, dtype=float)
            time_array = np.asarray(times, dtype=float)
            distance_array = np.asarray(distances, dtype=float)
            speed_array = np.asarray(speed_commands, dtype=float)
            measured_target_array = np.asarray(measured_target_positions, dtype=float)
            filtered_target_array = np.asarray(filtered_target_positions, dtype=float)
            intercept_point = interceptor_array[-1] if done else None
            trajectory_path = plot_trajectory_3d(
                target_positions=target_array,
                interceptor_positions=interceptor_array,
                output_path=self._output_dir / "autonomous_trajectory_3d.png",
                intercept_point=intercept_point,
                measured_target_positions=measured_target_array,
                filtered_target_positions=filtered_target_array,
            )
            dashboard_path = plot_mission_dashboard(
                times=time_array,
                target_positions=target_array,
                interceptor_positions=interceptor_array,
                distances=distance_array,
                speed_commands=speed_array,
                output_path=self._output_dir / "autonomous_dashboard.png",
            )
            output_paths.extend([trajectory_path, dashboard_path])

        LOGGER.info(
            "run_complete steps=%s intercepted=%s mean_loop_fps=%.2f final_distance_m=%.2f total_cost=%.2f",
            completed_steps,
            done and final_distance <= float(self._config["planning"]["desired_intercept_distance_m"]),
            mean_loop_fps,
            final_distance,
            total_cost,
        )
        return SystemRunResult(
            steps_executed=completed_steps,
            intercepted=done and final_distance <= float(self._config["planning"]["desired_intercept_distance_m"]),
            mean_loop_fps=mean_loop_fps,
            output_paths=output_paths,
            final_distance_m=final_distance,
            total_cost=total_cost,
        )
