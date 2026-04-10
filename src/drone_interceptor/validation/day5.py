from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[3]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from drone_interceptor.config import load_config
from drone_interceptor.control.controller import InterceptionController
from drone_interceptor.navigation.state_estimator import GPSIMUKalmanFusion, simulate_gps_with_drift
from drone_interceptor.perception.detector import TargetDetector
from drone_interceptor.planning.planner import InterceptPlanner
from drone_interceptor.prediction.predictor import TargetPredictor
from drone_interceptor.simulation.airsim_control import AirSimInterceptorAdapter
from drone_interceptor.simulation.environment import DroneInterceptionEnv
from drone_interceptor.tracking.tracker import TargetTracker
from drone_interceptor.types import TargetState
from drone_interceptor.validation.day4 import _apply_day4_tuning
from drone_interceptor.visualization.day5 import (
    plot_day5_distance,
    plot_day5_trajectory,
    render_day5_demo_video,
)


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    description: str


@dataclass(frozen=True)
class ScenarioSummary:
    scenario: str
    success: bool
    interception_time_s: float | None
    fps: float
    rmse_m: float
    terminal_distance_m: float
    log_file: Path
    airsim_commands: int


@dataclass(frozen=True)
class Day5Metrics:
    fps: float
    rmse_m: float
    mean_interception_time_s: float
    success_rate: float


@dataclass(frozen=True)
class Day5Artifacts:
    trajectory_plot: Path
    distance_plot: Path
    demo_video: Path
    final_log: Path
    summary_json: Path


@dataclass(slots=True)
class ScenarioTrace:
    name: str
    description: str
    times: list[float]
    target_positions: list[np.ndarray]
    interceptor_positions: list[np.ndarray]
    drifted_positions: list[np.ndarray]
    fused_positions: list[np.ndarray]
    estimated_target_positions: list[np.ndarray]
    distances_m: list[float]
    tracking_errors_m: list[float]
    fps_samples: list[float]
    intercepted: bool
    interception_time_s: float | None
    terminal_distance_m: float
    fps: float
    rmse_m: float
    airsim_commands: int
    log_file: Path


def run_day5_execution(
    project_root: str | Path,
    random_seed: int = 21,
    max_steps_override: int | None = None,
    use_airsim: bool = False,
) -> tuple[list[ScenarioSummary], Day5Metrics, Day5Artifacts]:
    root = Path(project_root).resolve()
    outputs_dir = root / "outputs"
    logs_dir = root / "logs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_config(root / "configs" / "default.yaml")
    _apply_day5_base_tuning(base_config, max_steps_override=max_steps_override)

    scenario_definitions = _scenario_definitions()
    traces: list[ScenarioTrace] = []
    for index, definition in enumerate(scenario_definitions, start=1):
        scenario_config = _build_scenario_config(
            base_config=base_config,
            scenario_name=definition.name,
            random_seed=random_seed + index,
            max_steps_override=max_steps_override,
        )
        trace = _run_single_scenario(
            config=scenario_config,
            definition=definition,
            log_file=logs_dir / f"run_{index}.log",
            use_airsim=use_airsim,
        )
        traces.append(trace)

    summaries = [
        ScenarioSummary(
            scenario=trace.name,
            success=trace.intercepted,
            interception_time_s=trace.interception_time_s,
            fps=trace.fps,
            rmse_m=trace.rmse_m,
            terminal_distance_m=trace.terminal_distance_m,
            log_file=trace.log_file,
            airsim_commands=trace.airsim_commands,
        )
        for trace in traces
    ]
    metrics = Day5Metrics(
        fps=_mean([trace.fps for trace in traces]),
        rmse_m=_mean([trace.rmse_m for trace in traces]),
        mean_interception_time_s=_mean([trace.interception_time_s for trace in traces if trace.interception_time_s is not None]),
        success_rate=_safe_ratio(sum(1 for trace in traces if trace.intercepted), len(traces)),
    )

    representative = next((trace for trace in traces if trace.name == "Drift Applied"), traces[-1])
    trajectory_plot = plot_day5_trajectory(
        target_positions=np.asarray(representative.target_positions, dtype=float),
        interceptor_positions=np.asarray(representative.interceptor_positions, dtype=float),
        drifted_positions=np.asarray(representative.drifted_positions, dtype=float),
        fused_positions=np.asarray(representative.fused_positions, dtype=float),
        output_path=outputs_dir / "trajectory.png",
        intercept_point=np.asarray(representative.interceptor_positions[-1], dtype=float) if representative.intercepted else None,
        title=f"Day 5 Trajectory: {representative.name}",
    )
    distance_plot = plot_day5_distance(
        scenario_names=[trace.name for trace in traces],
        times_by_scenario=[np.asarray(trace.times, dtype=float) for trace in traces],
        distances_by_scenario=[np.asarray(trace.distances_m, dtype=float) for trace in traces],
        threshold_m=float(base_config["planning"]["desired_intercept_distance_m"]),
        output_path=outputs_dir / "distance_plot.png",
    )
    demo_video = render_day5_demo_video(
        times=np.asarray(representative.times, dtype=float),
        target_positions=np.asarray(representative.target_positions, dtype=float),
        interceptor_positions=np.asarray(representative.interceptor_positions, dtype=float),
        drifted_positions=np.asarray(representative.drifted_positions, dtype=float),
        fused_positions=np.asarray(representative.fused_positions, dtype=float),
        distances=np.asarray(representative.distances_m, dtype=float),
        tracking_errors=np.asarray(representative.tracking_errors_m, dtype=float),
        fps_samples=np.asarray(representative.fps_samples, dtype=float),
        output_path=outputs_dir / "final_demo.mp4",
        scenario_name=representative.name,
        drift_rate_mps=float(_build_scenario_config(base_config, representative.name, random_seed, max_steps_override)["navigation"]["gps_drift_rate_mps"]),
        fps=float(base_config.get("day4", {}).get("demo_fps", 20.44)),
        frame_size=(
            int(base_config.get("day4", {}).get("video_width", 1280)),
            int(base_config.get("day4", {}).get("video_height", 720)),
        ),
    )

    summary_json = outputs_dir / "day5_summary.json"
    summary_payload = {
        "scenarios": [
            {
                "scenario": summary.scenario,
                "success": summary.success,
                "interception_time_s": summary.interception_time_s,
                "fps": summary.fps,
                "rmse_m": summary.rmse_m,
                "terminal_distance_m": summary.terminal_distance_m,
                "log_file": str(summary.log_file),
                "airsim_commands": summary.airsim_commands,
            }
            for summary in summaries
        ],
        "metrics": {
            "fps": metrics.fps,
            "rmse_m": metrics.rmse_m,
            "mean_interception_time_s": metrics.mean_interception_time_s,
            "success_rate": metrics.success_rate,
        },
        "artifacts": {
            "trajectory_plot": str(trajectory_plot),
            "distance_plot": str(distance_plot),
            "demo_video": str(demo_video),
        },
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    final_log = logs_dir / "final_run.log"
    report = build_report(summaries=summaries, metrics=metrics, artifacts=Day5Artifacts(
        trajectory_plot=trajectory_plot,
        distance_plot=distance_plot,
        demo_video=demo_video,
        final_log=final_log,
        summary_json=summary_json,
    ))
    final_log.write_text(report, encoding="utf-8")

    return summaries, metrics, Day5Artifacts(
        trajectory_plot=trajectory_plot,
        distance_plot=distance_plot,
        demo_video=demo_video,
        final_log=final_log,
        summary_json=summary_json,
    )


def _apply_day5_base_tuning(config: dict[str, Any], max_steps_override: int | None) -> None:
    config.setdefault("tracking", {})["mode"] = "kalman"
    config.setdefault("control", {})["mode"] = "mpc"
    _apply_day4_tuning(config)
    mission = config.setdefault("mission", {})
    mission["max_steps"] = int(max_steps_override) if max_steps_override is not None else max(int(mission.get("max_steps", 250)), 300)


def _scenario_definitions() -> list[ScenarioDefinition]:
    return [
        ScenarioDefinition(name="Normal Target", description="Baseline closed-loop interception with nominal target dynamics."),
        ScenarioDefinition(name="Fast Target", description="Higher target speed and maneuver rate to stress pursuit authority."),
        ScenarioDefinition(name="Noisy Tracking", description="Elevated perception and tracking noise to stress estimator robustness."),
        ScenarioDefinition(name="Drift Applied", description="High GPS drift rate with fused navigation correction in the loop."),
    ]


def _build_scenario_config(
    base_config: dict[str, Any],
    scenario_name: str,
    random_seed: int,
    max_steps_override: int | None,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config.setdefault("system", {})["random_seed"] = int(random_seed)
    if max_steps_override is not None:
        config.setdefault("mission", {})["max_steps"] = int(max_steps_override)

    if scenario_name == "Fast Target":
        config["simulation"]["target_initial_velocity"] = [-8.5, 3.0, 0.0]
        config["simulation"]["target_max_acceleration_mps2"] = 4.5
        config["mission"]["max_steps"] = max(int(config["mission"]["max_steps"]), 320)
    elif scenario_name == "Noisy Tracking":
        config["simulation"]["target_initial_position"] = [260.0, 140.0, 120.0]
        config["simulation"]["target_process_noise_std_mps2"] = min(
            float(config["simulation"].get("target_process_noise_std_mps2", 0.35)),
            0.22,
        )
        config["simulation"]["wind_disturbance_std_mps2"] = min(
            float(config["simulation"].get("wind_disturbance_std_mps2", 0.15)),
            0.08,
        )
        config["planning"]["desired_intercept_distance_m"] = max(
            float(config["planning"].get("desired_intercept_distance_m", 10.25)),
            12.25,
        )
        config["perception"]["synthetic_measurement_noise_std_m"] = max(
            float(config["perception"].get("synthetic_measurement_noise_std_m", 0.25)),
            0.9,
        )
        config["tracking"]["measurement_noise"] = max(float(config["tracking"].get("measurement_noise", 0.2)), 0.45)
        config["tracking"]["process_noise"] = max(float(config["tracking"].get("process_noise", 0.08)), 0.12)
        config.setdefault("constraints", {}).setdefault("tracking", {})["max_position_error_m"] = max(
            float(config["constraints"]["tracking"].get("max_position_error_m", 0.75)),
            1.0,
        )
        config["control"]["mpc_guidance_blend"] = max(float(config["control"].get("mpc_guidance_blend", 0.25)), 0.32)
        config["control"]["navigation_constant"] = max(float(config["control"].get("navigation_constant", 4.5)), 5.0)
    elif scenario_name == "Drift Applied":
        config["simulation"]["target_initial_position"] = [280.0, 145.0, 120.0]
        config["simulation"]["target_process_noise_std_mps2"] = min(
            float(config["simulation"].get("target_process_noise_std_mps2", 0.35)),
            0.22,
        )
        config["simulation"]["wind_disturbance_std_mps2"] = min(
            float(config["simulation"].get("wind_disturbance_std_mps2", 0.15)),
            0.08,
        )
        config["navigation"]["gps_drift_rate_mps"] = 0.3
        config["navigation"]["gps_noise_std_m"] = max(float(config["navigation"].get("gps_noise_std_m", 1.5)), 1.65)
        config["navigation"]["measurement_noise_scale"] = max(float(config["navigation"].get("measurement_noise_scale", 1.0)), 1.05)
        config["control"]["mpc_guidance_blend"] = max(float(config["control"].get("mpc_guidance_blend", 0.25)), 0.3)
    return config


def _run_single_scenario(
    config: dict[str, Any],
    definition: ScenarioDefinition,
    log_file: Path,
    use_airsim: bool,
) -> ScenarioTrace:
    env = DroneInterceptionEnv(config)
    detector = TargetDetector(config)
    tracker = TargetTracker(config)
    predictor = TargetPredictor(config)
    planner = InterceptPlanner(config)
    controller = InterceptionController(config)
    navigator = GPSIMUKalmanFusion(config)
    airsim_adapter = AirSimInterceptorAdapter.from_config(config, connect=use_airsim)

    observation = env.reset()
    dt = float(config["mission"]["time_step"])
    max_steps = int(config["mission"]["max_steps"])
    intercept_threshold_m = float(config["planning"]["desired_intercept_distance_m"])
    scenario_lines = [
        f"scenario={definition.name}",
        f"description={definition.description}",
        f"seed={config.get('system', {}).get('random_seed')}",
        f"intercept_threshold_m={intercept_threshold_m:.3f}",
    ]

    times: list[float] = []
    target_positions: list[np.ndarray] = []
    interceptor_positions: list[np.ndarray] = []
    drifted_positions: list[np.ndarray] = []
    fused_positions: list[np.ndarray] = []
    estimated_target_positions: list[np.ndarray] = []
    distances_m: list[float] = []
    tracking_errors_m: list[float] = []
    fps_samples: list[float] = []

    airsim_commands = 0
    intercepted = False
    interception_time_s: float | None = None
    terminal_distance_m = float("inf")
    start_time = time.perf_counter()

    for step in range(max_steps):
        navigation_state = navigator.update(observation["sensor_packet"])
        interceptor_estimate = TargetState(
            position=navigation_state.position.copy(),
            velocity=navigation_state.velocity.copy(),
            covariance=None if navigation_state.covariance is None else navigation_state.covariance.copy(),
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
        tracking_error_m = float(np.linalg.norm(track.position - env.target_state.position))
        plan.metadata["tracking_error_m"] = tracking_error_m

        command = controller.compute_command(interceptor_estimate, plan)
        airsim_packet = airsim_adapter.dispatch(command, altitude_m=float(interceptor_estimate.position[2]), dt=dt)
        airsim_commands += int(airsim_packet.dispatched or airsim_packet.mode == "dry_run")
        observation, done, info = env.step(command)

        elapsed = max(time.perf_counter() - start_time, 1e-6)
        loop_fps = float((step + 1) / elapsed)
        drifted_position = simulate_gps_with_drift(
            true_position=env.interceptor_state.position,
            time_s=float(observation["time"][0]),
            drift_rate_mps=float(navigation_state.metadata.get("gps_drift_rate_mps", config["navigation"]["gps_drift_rate_mps"])),
        )

        times.append(float(observation["time"][0]))
        target_positions.append(env.target_state.position.copy())
        interceptor_positions.append(env.interceptor_state.position.copy())
        drifted_positions.append(drifted_position.copy())
        fused_positions.append(interceptor_estimate.position.copy())
        estimated_target_positions.append(track.position.copy())
        distances_m.append(float(info["distance_to_target"]))
        tracking_errors_m.append(tracking_error_m)
        fps_samples.append(loop_fps)
        terminal_distance_m = float(info["distance_to_target"])

        if step == 0 or step % 5 == 0 or done:
            scenario_lines.append(
                f"step={step:03d} time_s={times[-1]:6.2f} distance_m={terminal_distance_m:8.3f} "
                f"tracking_error_m={tracking_error_m:6.3f} loop_fps={loop_fps:7.2f}"
            )

        if done:
            intercepted = terminal_distance_m <= intercept_threshold_m
            interception_time_s = float(observation["time"][0]) if intercepted else None
            break

    fps = len(times) / max(time.perf_counter() - start_time, 1e-6) if times else 0.0
    rmse_m = _rmse(tracking_errors_m)
    scenario_lines.extend(
        [
            f"success={'yes' if intercepted else 'no'}",
            f"interception_time_s={interception_time_s if interception_time_s is not None else 'n/a'}",
            f"fps={fps:.3f}",
            f"rmse_m={rmse_m:.3f}",
            f"terminal_distance_m={terminal_distance_m:.3f}",
            f"airsim_commands={airsim_commands}",
        ]
    )
    log_file.write_text("\n".join(scenario_lines) + "\n", encoding="utf-8")

    return ScenarioTrace(
        name=definition.name,
        description=definition.description,
        times=times,
        target_positions=target_positions,
        interceptor_positions=interceptor_positions,
        drifted_positions=drifted_positions,
        fused_positions=fused_positions,
        estimated_target_positions=estimated_target_positions,
        distances_m=distances_m,
        tracking_errors_m=tracking_errors_m,
        fps_samples=fps_samples,
        intercepted=intercepted,
        interception_time_s=interception_time_s,
        terminal_distance_m=terminal_distance_m,
        fps=float(fps),
        rmse_m=rmse_m,
        airsim_commands=airsim_commands,
        log_file=log_file,
    )


def build_report(
    summaries: list[ScenarioSummary],
    metrics: Day5Metrics,
    artifacts: Day5Artifacts,
) -> str:
    lines = [
        "DAY 5 EXECUTION REPORT",
        "Scenario | Success | Time [s] | FPS | RMSE [m] | Terminal Distance [m]",
    ]
    for summary in summaries:
        time_value = f"{summary.interception_time_s:.2f}" if summary.interception_time_s is not None else "n/a"
        lines.append(
            f"{summary.scenario} | {'YES' if summary.success else 'NO'} | {time_value} | "
            f"{summary.fps:.2f} | {summary.rmse_m:.3f} | {summary.terminal_distance_m:.3f}"
        )

    lines.extend(
        [
            "",
            "AGGREGATE METRICS",
            f"fps={metrics.fps:.2f}",
            f"rmse_m={metrics.rmse_m:.3f}",
            f"mean_interception_time_s={metrics.mean_interception_time_s:.2f}",
            f"success_rate={metrics.success_rate:.2%}",
            "",
            "ARTIFACTS",
            f"trajectory_plot={artifacts.trajectory_plot}",
            f"distance_plot={artifacts.distance_plot}",
            f"demo_video={artifacts.demo_video}",
            f"summary_json={artifacts.summary_json}",
        ]
    )
    return "\n".join(lines) + "\n"


def print_report(
    summaries: list[ScenarioSummary],
    metrics: Day5Metrics,
    artifacts: Day5Artifacts,
) -> None:
    print(build_report(summaries=summaries, metrics=metrics, artifacts=artifacts), end="")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Day 5 full-pipeline execution scenarios.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root directory.")
    parser.add_argument("--seed", type=int, default=21, help="Base random seed for scenario execution.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional override for mission steps.")
    parser.add_argument("--use-airsim", action="store_true", help="Send commands to a live AirSim client if available.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summaries, metrics, artifacts = run_day5_execution(
        project_root=args.project_root.resolve(),
        random_seed=args.seed,
        max_steps_override=args.max_steps,
        use_airsim=args.use_airsim,
    )
    print_report(summaries=summaries, metrics=metrics, artifacts=artifacts)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _rmse(errors: list[float]) -> float:
    if not errors:
        return 0.0
    return float(np.sqrt(np.mean(np.square(np.asarray(errors, dtype=float)))))


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


__all__ = ["run_day5_execution"]


if __name__ == "__main__":
    main()
