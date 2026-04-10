from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[3]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from drone_interceptor.config import load_config
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
from drone_interceptor.visualization.dashboard import plot_trajectory_3d


LOGGER = logging.getLogger("day3_validation")

NOISE_LEVELS = {
    "low": 0.6,
    "medium": 1.0,
    "high": 1.4,
}


@dataclass(frozen=True)
class ValidationStatus:
    name: str
    passed: bool
    details: str


@dataclass(frozen=True)
class Day3Metrics:
    rmse_before: float
    rmse_after: float
    monte_carlo_rmse_mean: float
    monte_carlo_rmse_std: float
    tracking_stability_before: float
    tracking_stability_after: float
    prediction_rmse_before: float
    prediction_rmse_after: float
    interception_time_s: float
    success_rate: float
    mean_loop_fps: float
    monotonic_decrease_ratio: float
    optimization_improvement: float
    noise_tracking_rmse: dict[str, float]
    constraint_violations: dict[str, int]


@dataclass(frozen=True)
class Day3Artifacts:
    trajectory_3d_plot: Path
    metrics_plot: Path
    log_file: Path


@dataclass(slots=True)
class ScenarioResult:
    tracker_mode: str
    noise_level: str
    true_target_positions: list[np.ndarray]
    measured_target_positions: list[np.ndarray]
    estimated_target_positions: list[np.ndarray]
    interceptor_positions: list[np.ndarray]
    distances_m: list[float]
    uncertainty_traces: list[float]
    stage_costs: list[float]
    prediction_snapshots: list[np.ndarray]
    commanded_speeds: list[float]
    commanded_accelerations: list[float]
    intercepted: bool
    interception_time_s: float
    loop_fps: float
    velocity_violations: int
    acceleration_violations: int
    safety_violations: int
    drift_violations: int


def run_day3_validation(
    project_root: str | Path,
    monte_carlo_runs: int = 20,
    noise_runs: int = 3,
    control_runs: int = 5,
    random_seed: int = 7,
) -> tuple[list[ValidationStatus], Day3Metrics, Day3Artifacts]:
    root = Path(project_root).resolve()
    outputs_dir = root / "outputs"
    logs_dir = root / "logs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_config(root / "configs" / "default.yaml")
    estimation_steps = min(int(base_config["mission"]["max_steps"]), 80)
    kalman_results = _run_scenario_batch(
        base_config=base_config,
        runs=monte_carlo_runs,
        random_seed=random_seed,
        tracker_mode="kalman",
        noise_level="medium",
        max_steps_override=estimation_steps,
        profile="estimation",
    )
    baseline_results = _run_scenario_batch(
        base_config=base_config,
        runs=monte_carlo_runs,
        random_seed=random_seed,
        tracker_mode="kinematic",
        noise_level="medium",
        max_steps_override=estimation_steps,
        profile="estimation",
    )
    control_results = _run_scenario_batch(
        base_config=base_config,
        runs=control_runs,
        random_seed=random_seed + 50,
        tracker_mode="kalman",
        noise_level="medium",
        profile="control",
    )
    noise_results = {
        level: _run_scenario_batch(
            base_config=base_config,
            runs=noise_runs,
            random_seed=random_seed + 100,
            tracker_mode="kalman",
            noise_level=level,
            max_steps_override=estimation_steps,
            profile="estimation",
        )
        for level in NOISE_LEVELS
    }

    primary_scenario = control_results[0]
    rmse_before = _mean([_tracking_rmse(result.measured_target_positions, result.true_target_positions) for result in kalman_results])
    rmse_after = _mean([_tracking_rmse(result.estimated_target_positions, result.true_target_positions) for result in kalman_results])
    monte_carlo_rmse_std = float(
        np.std(
            [_tracking_rmse(result.estimated_target_positions, result.true_target_positions) for result in kalman_results],
            ddof=0,
        )
    )
    tracking_stability_before = _mean([_trajectory_stability(result.measured_target_positions) for result in kalman_results])
    tracking_stability_after = _mean([_trajectory_stability(result.estimated_target_positions) for result in kalman_results])
    prediction_rmse_before = _mean([_prediction_rmse(result, positions_attr="measured_target_positions") for result in baseline_results])
    prediction_rmse_after = _mean([_prediction_rmse(result, positions_attr="estimated_target_positions") for result in kalman_results])
    mean_interception_time = _mean([result.interception_time_s for result in control_results])
    success_rate = _mean([1.0 if result.intercepted else 0.0 for result in control_results])
    mean_loop_fps = _mean([result.loop_fps for result in control_results])
    monotonic_decrease_ratio = _mean([_distance_monotonic_ratio(result.distances_m) for result in control_results])
    optimization_improvement = _mean([_optimization_improvement(result.stage_costs) for result in control_results])
    noise_tracking_rmse = {
        level: _mean([_tracking_rmse(result.estimated_target_positions, result.true_target_positions) for result in results])
        for level, results in noise_results.items()
    }
    constraint_violations = _aggregate_constraint_violations(control_results)

    trajectory_plot = plot_trajectory_3d(
        target_positions=np.asarray(primary_scenario.true_target_positions, dtype=float),
        interceptor_positions=np.asarray(primary_scenario.interceptor_positions, dtype=float),
        output_path=outputs_dir / "day3_3d_plot.png",
        intercept_point=(
            np.asarray(primary_scenario.interceptor_positions[-1], dtype=float)
            if primary_scenario.intercepted and primary_scenario.interceptor_positions
            else None
        ),
        measured_target_positions=np.asarray(primary_scenario.measured_target_positions, dtype=float),
        filtered_target_positions=np.asarray(primary_scenario.estimated_target_positions, dtype=float),
    )
    metrics_plot = _plot_day3_metrics(
        primary_scenario=primary_scenario,
        kalman_results=kalman_results,
        noise_tracking_rmse=noise_tracking_rmse,
        rmse_before=rmse_before,
        rmse_after=rmse_after,
        output_path=outputs_dir / "day3_metrics.png",
    )

    statuses = [
        ValidationStatus(
            "Kalman Filter Improvement",
            rmse_after < rmse_before,
            (
                f"rmse_before={rmse_before:.3f} rmse_after={rmse_after:.3f} "
                f"monte_carlo_mean={rmse_after:.3f} monte_carlo_std={monte_carlo_rmse_std:.3f}"
            ),
        ),
        ValidationStatus(
            "Tracking Stability",
            tracking_stability_after > tracking_stability_before,
            f"stability_before={tracking_stability_before:.4f} stability_after={tracking_stability_after:.4f}",
        ),
        ValidationStatus(
            "Prediction Accuracy",
            prediction_rmse_after < prediction_rmse_before,
            f"prediction_rmse_before={prediction_rmse_before:.3f} prediction_rmse_after={prediction_rmse_after:.3f}",
        ),
        ValidationStatus(
            "Control Performance",
            monotonic_decrease_ratio >= 0.8 and success_rate >= 0.6,
            (
                f"monotonic_ratio={monotonic_decrease_ratio:.3f} "
                f"mean_interception_time={mean_interception_time:.2f}s success_rate={success_rate:.2%}"
            ),
        ),
        ValidationStatus(
            "Optimization",
            optimization_improvement > 0.0,
            f"mean_cost_improvement={optimization_improvement:.3f}",
        ),
        ValidationStatus(
            "Simulation Robustness",
            _noise_robustness_ok(noise_tracking_rmse),
            (
                "noise_rmse="
                + ", ".join(f"{level}:{value:.3f}" for level, value in noise_tracking_rmse.items())
            ),
        ),
    ]
    all_passed = all(status.passed for status in statuses)
    statuses.append(
        ValidationStatus(
            "Integration",
            all_passed and trajectory_plot.exists() and metrics_plot.exists(),
            "full Day 3 validation pipeline completed without crash" if all_passed else "one or more Day 3 validation stages failed",
        )
    )

    metrics = Day3Metrics(
        rmse_before=rmse_before,
        rmse_after=rmse_after,
        monte_carlo_rmse_mean=rmse_after,
        monte_carlo_rmse_std=monte_carlo_rmse_std,
        tracking_stability_before=tracking_stability_before,
        tracking_stability_after=tracking_stability_after,
        prediction_rmse_before=prediction_rmse_before,
        prediction_rmse_after=prediction_rmse_after,
        interception_time_s=mean_interception_time,
        success_rate=success_rate,
        mean_loop_fps=mean_loop_fps,
        monotonic_decrease_ratio=monotonic_decrease_ratio,
        optimization_improvement=optimization_improvement,
        noise_tracking_rmse=noise_tracking_rmse,
        constraint_violations=constraint_violations,
    )
    artifacts = Day3Artifacts(
        trajectory_3d_plot=trajectory_plot,
        metrics_plot=metrics_plot,
        log_file=logs_dir / "day3_validation.log",
    )
    return statuses, metrics, artifacts


def _run_scenario_batch(
    base_config: dict[str, Any],
    runs: int,
    random_seed: int,
    tracker_mode: str,
    noise_level: str,
    max_steps_override: int | None = None,
    profile: str = "control",
) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    for run_index in range(runs):
        config = _build_config_variant(
            base_config=base_config,
            seed=random_seed + run_index,
            tracker_mode=tracker_mode,
            noise_level=noise_level,
            max_steps_override=max_steps_override,
            profile=profile,
        )
        results.append(_run_single_scenario(config=config, tracker_mode=tracker_mode, noise_level=noise_level))
    return results


def _build_config_variant(
    base_config: dict[str, Any],
    seed: int,
    tracker_mode: str,
    noise_level: str,
    max_steps_override: int | None = None,
    profile: str = "control",
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config.setdefault("system", {})["random_seed"] = int(seed)
    config.setdefault("tracking", {})["mode"] = tracker_mode
    if max_steps_override is not None:
        config.setdefault("mission", {})["max_steps"] = int(max_steps_override)
    if profile == "estimation":
        config.setdefault("control", {})["mode"] = "pn"
    _apply_noise_level(config, noise_level)
    return config


def _apply_noise_level(config: dict[str, Any], noise_level: str) -> None:
    scale = NOISE_LEVELS[noise_level]
    perception = config.setdefault("perception", {})
    tracking = config.setdefault("tracking", {})
    navigation = config.setdefault("navigation", {})
    simulation = config.setdefault("simulation", {})

    perception["synthetic_measurement_noise_std_m"] = float(perception.get("synthetic_measurement_noise_std_m", 1.25)) * scale
    tracking["measurement_noise"] = float(tracking.get("measurement_noise", 0.5)) * scale
    tracking["process_noise"] = float(tracking.get("process_noise", 0.15)) * max(0.8, scale)
    navigation["gps_noise_std_m"] = float(navigation.get("gps_noise_std_m", 1.5)) * scale
    navigation["imu_noise_std_mps2"] = float(navigation.get("imu_noise_std_mps2", 0.15)) * max(0.8, scale)
    simulation["interceptor_process_noise_std_mps2"] = float(simulation.get("interceptor_process_noise_std_mps2", 0.2)) * scale
    simulation["target_process_noise_std_mps2"] = float(simulation.get("target_process_noise_std_mps2", 0.35)) * scale
    simulation["wind_disturbance_std_mps2"] = float(simulation.get("wind_disturbance_std_mps2", 0.15)) * scale


def _run_single_scenario(
    config: dict[str, Any],
    tracker_mode: str,
    noise_level: str,
) -> ScenarioResult:
    env = DroneInterceptionEnv(config)
    detector = TargetDetector(config)
    tracker = TargetTracker(config)
    predictor = TargetPredictor(config)
    planner = InterceptPlanner(config)
    controller = InterceptionController(config)
    navigator = GPSIMUKalmanFusion(config)
    cost_model = InterceptionCostModel.from_config(config)

    observation = env.reset()
    max_steps = int(config["mission"]["max_steps"])
    true_target_positions: list[np.ndarray] = []
    measured_target_positions: list[np.ndarray] = []
    estimated_target_positions: list[np.ndarray] = []
    interceptor_positions: list[np.ndarray] = []
    distances_m: list[float] = []
    uncertainty_traces: list[float] = []
    stage_costs: list[float] = []
    prediction_snapshots: list[np.ndarray] = []
    commanded_speeds: list[float] = []
    commanded_accelerations: list[float] = []
    velocity_violations = 0
    acceleration_violations = 0
    safety_violations = 0
    drift_violations = 0

    start_time = time.perf_counter()
    intercepted = False
    interception_time_s = float(max_steps * float(config["mission"]["time_step"]))

    for _step in range(max_steps):
        navigation_state = navigator.update(observation["sensor_packet"])
        interceptor_estimate = TargetState(
            position=navigation_state.position.copy(),
            velocity=navigation_state.velocity.copy(),
            covariance=(None if navigation_state.covariance is None else navigation_state.covariance.copy()),
            timestamp=navigation_state.timestamp,
            metadata=dict(navigation_state.metadata),
        )
        detection = detector.detect(observation)
        track = tracker.update(detection)
        prediction = predictor.predict(track)
        plan = planner.plan(interceptor_estimate, prediction)
        plan.metadata["current_target_position"] = track.position.copy()
        plan.metadata["current_target_velocity"] = track.velocity.copy()
        plan.metadata["current_target_acceleration"] = (
            track.acceleration.copy() if track.acceleration is not None else np.zeros(3, dtype=float)
        )
        plan.metadata["current_target_covariance"] = (
            None if track.covariance is None else np.asarray(track.covariance, dtype=float).copy()
        )
        plan.metadata["tracking_error_m"] = float(np.linalg.norm(track.position - env.target_state.position))
        command = controller.compute_command(interceptor_estimate, plan)
        observation, done, info = env.step(command)

        constraint_status = ConstraintStatus(
            velocity_clipped=bool(command.metadata.get("velocity_clipped", False)),
            acceleration_clipped=bool(command.metadata.get("acceleration_clipped", False)),
            tracking_ok=bool(command.metadata.get("tracking_ok", True)),
            drift_rate_in_bounds=bool(navigation_state.metadata.get("drift_rate_in_bounds", True)),
            safety_override=bool(command.metadata.get("safety_override", False)),
            distance_to_target_m=float(info["distance_to_target"]),
        )
        stage_costs.append(
            cost_model.stage_cost(
                interceptor_position=env.interceptor_state.position,
                target_position=env.target_state.position,
                control_input=command.acceleration_command if command.acceleration_command is not None else command.velocity_command,
                constraint_status=constraint_status,
                uncertainty_term=float(plan.metadata.get("uncertainty_trace", 0.0)),
            )
        )

        true_target_positions.append(env.target_state.position.copy())
        measured_target_positions.append(detection.position.copy())
        estimated_target_positions.append(track.position.copy())
        interceptor_positions.append(env.interceptor_state.position.copy())
        distances_m.append(float(info["distance_to_target"]))
        uncertainty_traces.append(float(plan.metadata.get("uncertainty_trace", 0.0)))
        prediction_snapshots.append(np.asarray([state.position.copy() for state in prediction], dtype=float))
        commanded_speeds.append(float(np.linalg.norm(command.velocity_command)))
        commanded_accelerations.append(
            float(
                np.linalg.norm(
                    command.acceleration_command if command.acceleration_command is not None else np.zeros(3, dtype=float)
                )
            )
        )
        velocity_violations += int(constraint_status.velocity_clipped)
        acceleration_violations += int(constraint_status.acceleration_clipped)
        safety_violations += int(constraint_status.safety_override)
        drift_violations += int(not constraint_status.drift_rate_in_bounds)

        if done:
            intercepted = float(info["distance_to_target"]) <= float(config["planning"]["desired_intercept_distance_m"])
            interception_time_s = float(observation["time"][0])
            break

    elapsed = max(time.perf_counter() - start_time, 1e-6)
    loop_fps = len(true_target_positions) / elapsed if true_target_positions else 0.0
    return ScenarioResult(
        tracker_mode=tracker_mode,
        noise_level=noise_level,
        true_target_positions=true_target_positions,
        measured_target_positions=measured_target_positions,
        estimated_target_positions=estimated_target_positions,
        interceptor_positions=interceptor_positions,
        distances_m=distances_m,
        uncertainty_traces=uncertainty_traces,
        stage_costs=stage_costs,
        prediction_snapshots=prediction_snapshots,
        commanded_speeds=commanded_speeds,
        commanded_accelerations=commanded_accelerations,
        intercepted=intercepted,
        interception_time_s=interception_time_s,
        loop_fps=loop_fps,
        velocity_violations=velocity_violations,
        acceleration_violations=acceleration_violations,
        safety_violations=safety_violations,
        drift_violations=drift_violations,
    )


def _tracking_rmse(estimate_positions: list[np.ndarray], truth_positions: list[np.ndarray]) -> float:
    estimate = np.asarray(estimate_positions, dtype=float)
    truth = np.asarray(truth_positions, dtype=float)
    if len(estimate) == 0 or len(truth) == 0:
        return float("inf")
    aligned = min(len(estimate), len(truth))
    return float(np.sqrt(np.mean((estimate[:aligned] - truth[:aligned]) ** 2)))


def _trajectory_stability(positions: list[np.ndarray]) -> float:
    trajectory = np.asarray(positions, dtype=float)
    if len(trajectory) < 3:
        return 0.0
    jerk = np.diff(trajectory, n=2, axis=0)
    jerk_norm = float(np.mean(np.linalg.norm(jerk, axis=1)))
    return float(1.0 / (1.0 + jerk_norm))


def _prediction_rmse(result: ScenarioResult, positions_attr: str) -> float:
    prediction_snapshots = result.prediction_snapshots
    if not prediction_snapshots or not result.true_target_positions:
        return float("inf")

    truth = np.asarray(result.true_target_positions, dtype=float)
    anchor_positions = np.asarray(getattr(result, positions_attr), dtype=float)
    squared_errors: list[float] = []
    for step_index, prediction in enumerate(prediction_snapshots):
        if step_index >= len(anchor_positions):
            break
        for horizon_index in range(prediction.shape[0]):
            truth_index = step_index + horizon_index + 1
            if truth_index >= len(truth):
                break
            squared_errors.append(float(np.mean((prediction[horizon_index] - truth[truth_index]) ** 2)))
    if not squared_errors:
        return float("inf")
    return float(np.sqrt(np.mean(squared_errors)))


def _distance_monotonic_ratio(distances: list[float]) -> float:
    distance_array = np.asarray(distances, dtype=float)
    if len(distance_array) < 3:
        return 0.0
    smoothed = _moving_average(distance_array, window=min(5, len(distance_array)))
    differences = np.diff(smoothed)
    return float(np.mean(differences <= 1e-6))


def _optimization_improvement(stage_costs: list[float]) -> float:
    cost_array = np.asarray(stage_costs, dtype=float)
    if len(cost_array) < 4:
        return 0.0
    window = min(5, len(cost_array) // 2)
    initial_mean = float(np.mean(cost_array[:window]))
    final_mean = float(np.mean(cost_array[-window:]))
    return initial_mean - final_mean


def _aggregate_constraint_violations(results: list[ScenarioResult]) -> dict[str, int]:
    return {
        "velocity": int(sum(result.velocity_violations for result in results)),
        "acceleration": int(sum(result.acceleration_violations for result in results)),
        "safety": int(sum(result.safety_violations for result in results)),
        "drift": int(sum(result.drift_violations for result in results)),
    }


def _noise_robustness_ok(noise_tracking_rmse: dict[str, float]) -> bool:
    low = float(noise_tracking_rmse["low"])
    medium = float(noise_tracking_rmse["medium"])
    high = float(noise_tracking_rmse["high"])
    finite = all(np.isfinite(value) for value in (low, medium, high))
    ordered = low <= medium + 1e-6 <= high + 1e-6
    bounded = high <= max(3.0 * low, low + 1.0)
    return finite and ordered and bounded


def _plot_day3_metrics(
    primary_scenario: ScenarioResult,
    kalman_results: list[ScenarioResult],
    noise_tracking_rmse: dict[str, float],
    rmse_before: float,
    rmse_after: float,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    times = np.arange(len(primary_scenario.distances_m), dtype=float)
    measured = np.asarray(primary_scenario.measured_target_positions, dtype=float)
    estimated = np.asarray(primary_scenario.estimated_target_positions, dtype=float)
    truth = np.asarray(primary_scenario.true_target_positions, dtype=float)

    measured_error = np.linalg.norm(measured - truth, axis=1) if len(measured) else np.array([], dtype=float)
    estimated_error = np.linalg.norm(estimated - truth, axis=1) if len(estimated) else np.array([], dtype=float)
    monte_carlo_rmses = np.asarray(
        [_tracking_rmse(result.estimated_target_positions, result.true_target_positions) for result in kalman_results],
        dtype=float,
    )

    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].plot(times, primary_scenario.distances_m, color="#1f77b4")
    axes[0, 0].set_title("Distance to Target")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Distance [m]")

    axes[0, 1].plot(times, primary_scenario.stage_costs, color="#9467bd")
    axes[0, 1].set_title("Cost Over Time")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Stage Cost")

    axes[0, 2].plot(times, measured_error, label="Raw", color="#ff9896")
    axes[0, 2].plot(times, estimated_error, label="KF", color="#2ca02c")
    axes[0, 2].set_title("Tracking Error vs Truth")
    axes[0, 2].set_xlabel("Step")
    axes[0, 2].set_ylabel("Error [m]")
    axes[0, 2].legend()

    axes[1, 0].bar(["Before KF", "After KF"], [rmse_before, rmse_after], color=["#ff9896", "#2ca02c"])
    axes[1, 0].set_title("RMSE Improvement")
    axes[1, 0].set_ylabel("RMSE [m]")

    axes[1, 1].bar(list(noise_tracking_rmse.keys()), list(noise_tracking_rmse.values()), color=["#8dd3c7", "#80b1d3", "#fb8072"])
    axes[1, 1].set_title("Noise Robustness")
    axes[1, 1].set_ylabel("Tracking RMSE [m]")

    axes[1, 2].hist(monte_carlo_rmses, bins=min(8, len(monte_carlo_rmses)), color="#80b1d3", edgecolor="black")
    axes[1, 2].set_title("Monte Carlo RMSE Distribution")
    axes[1, 2].set_xlabel("Tracking RMSE [m]")
    axes[1, 2].set_ylabel("Count")

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="valid")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def setup_logging(log_path: Path) -> None:
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def print_report(statuses: list[ValidationStatus], metrics: Day3Metrics) -> None:
    print("DAY 3 VALIDATION REPORT:")
    for status in statuses:
        outcome = "PASS" if status.passed else "FAIL"
        print(f"- {status.name}: {outcome}")
        LOGGER.info("%s %s %s", status.name, outcome, status.details)

    print("PERFORMANCE METRICS:")
    print(f"- Tracking RMSE Before KF: {metrics.rmse_before:.3f}")
    print(f"- Tracking RMSE After KF: {metrics.rmse_after:.3f}")
    print(f"- Monte Carlo RMSE Mean: {metrics.monte_carlo_rmse_mean:.3f}")
    print(f"- Monte Carlo RMSE Std: {metrics.monte_carlo_rmse_std:.3f}")
    print(f"- Tracking Stability Before KF: {metrics.tracking_stability_before:.4f}")
    print(f"- Tracking Stability After KF: {metrics.tracking_stability_after:.4f}")
    print(f"- Prediction RMSE Before KF: {metrics.prediction_rmse_before:.3f}")
    print(f"- Prediction RMSE After KF: {metrics.prediction_rmse_after:.3f}")
    print(f"- Interception Time: {metrics.interception_time_s:.2f}s")
    print(f"- Success Rate: {metrics.success_rate:.2%}")
    print(f"- Mean Loop FPS: {metrics.mean_loop_fps:.2f}")
    print(f"- Monotonic Distance Ratio: {metrics.monotonic_decrease_ratio:.3f}")
    print(f"- Optimization Improvement: {metrics.optimization_improvement:.3f}")
    print(
        "- Noise Tracking RMSE: "
        + ", ".join(f"{level}={value:.3f}" for level, value in metrics.noise_tracking_rmse.items())
    )
    print(
        "- Constraint Violations: "
        + ", ".join(f"{name}={value}" for name, value in metrics.constraint_violations.items())
    )
    LOGGER.info(
        "metrics rmse_before=%.4f rmse_after=%.4f monte_carlo_rmse_mean=%.4f monte_carlo_rmse_std=%.4f "
        "tracking_stability_before=%.4f tracking_stability_after=%.4f prediction_rmse_before=%.4f "
        "prediction_rmse_after=%.4f interception_time_s=%.2f success_rate=%.4f mean_loop_fps=%.2f "
        "monotonic_decrease_ratio=%.4f optimization_improvement=%.4f noise_tracking_rmse=%s constraint_violations=%s",
        metrics.rmse_before,
        metrics.rmse_after,
        metrics.monte_carlo_rmse_mean,
        metrics.monte_carlo_rmse_std,
        metrics.tracking_stability_before,
        metrics.tracking_stability_after,
        metrics.prediction_rmse_before,
        metrics.prediction_rmse_after,
        metrics.interception_time_s,
        metrics.success_rate,
        metrics.mean_loop_fps,
        metrics.monotonic_decrease_ratio,
        metrics.optimization_improvement,
        metrics.noise_tracking_rmse,
        metrics.constraint_violations,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run rigorous Day 3 UAV interception validation.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root directory.")
    parser.add_argument("--monte-carlo-runs", type=int, default=20, help="Number of Monte Carlo runs for consistency validation.")
    parser.add_argument("--control-runs", type=int, default=5, help="Number of full interception runs for control validation.")
    parser.add_argument("--noise-runs", type=int, default=3, help="Runs per noise level for robustness validation.")
    parser.add_argument("--seed", type=int, default=7, help="Validation random seed.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    log_path = project_root / "logs" / "day3_validation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_path)

    statuses, metrics, artifacts = run_day3_validation(
        project_root=project_root,
        monte_carlo_runs=args.monte_carlo_runs,
        noise_runs=args.noise_runs,
        control_runs=args.control_runs,
        random_seed=args.seed,
    )
    LOGGER.info(
        "artifacts trajectory_3d_plot=%s metrics_plot=%s",
        artifacts.trajectory_3d_plot,
        artifacts.metrics_plot,
    )
    print_report(statuses, metrics)


__all__ = ["run_day3_validation"]


if __name__ == "__main__":
    main()
