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
from drone_interceptor.ros2.runtime import (
    EdgeProfile,
    LocalControlNode,
    LocalNavigationNode,
    LocalPerceptionNode,
    LocalTopicBus,
    LocalTrackingNode,
)
from drone_interceptor.simulation.environment import DroneInterceptionEnv
from drone_interceptor.validation.day4 import _apply_day4_tuning
from drone_interceptor.visualization.day6 import (
    plot_day6_architecture,
    render_day6_demo_video,
)


@dataclass(frozen=True)
class Day6Metrics:
    success: bool
    interception_time_s: float | None
    mean_loop_fps: float
    edge_mode_fps: float
    px4_setpoints_sent: int
    fallback_activations: int
    node_fps: dict[str, float]
    airsim_mode: str


@dataclass(frozen=True)
class Day6SweepMetrics:
    success_rate: float
    runs: int
    seeds: tuple[int, ...]
    mean_interception_time_s: float


@dataclass(frozen=True)
class Day6Artifacts:
    trajectory_plot: Path
    demo_video: Path
    compatibility_video: Path | None
    log_file: Path
    summary_json: Path


@dataclass(slots=True)
class _ArchitectureTrace:
    times: list[float]
    target_positions: list[np.ndarray]
    interceptor_positions: list[np.ndarray]
    drifted_positions: list[np.ndarray]
    fused_positions: list[np.ndarray]
    distances_m: list[float]
    fallback_flags: list[bool]
    fallback_points: list[np.ndarray]
    node_fps: dict[str, float]
    success: bool
    interception_time_s: float | None
    mean_loop_fps: float
    px4_setpoints_sent: int
    fallback_activations: int
    airsim_mode: str
    sitl_command: str
    log_lines: list[str]


def run_day6_execution(
    project_root: str | Path,
    random_seed: int = 41,
    max_steps_override: int | None = None,
    use_airsim: bool = False,
) -> tuple[Day6Metrics, Day6Artifacts]:
    root = Path(project_root).resolve()
    outputs_dir = root / "outputs"
    logs_dir = root / "logs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_config(root / "configs" / "default.yaml")
    _apply_day6_tuning(base_config, random_seed=random_seed, max_steps_override=max_steps_override)

    trace = _run_architecture(
        config=base_config,
        edge_profile=EdgeProfile(enabled=False),
        use_airsim=use_airsim,
    )
    edge_trace = _run_architecture(
        config=base_config,
        edge_profile=EdgeProfile(enabled=True, detection_stride=3, injected_latency_s=0.012, inference_imgsz=320),
        use_airsim=False,
        max_steps=min(90, int(base_config["mission"]["max_steps"])),
    )

    trajectory_plot = plot_day6_architecture(
        target_positions=np.asarray(trace.target_positions, dtype=float),
        interceptor_positions=np.asarray(trace.interceptor_positions, dtype=float),
        drifted_positions=np.asarray(trace.drifted_positions, dtype=float),
        fused_positions=np.asarray(trace.fused_positions, dtype=float),
        fallback_points=None if not trace.fallback_points else np.asarray(trace.fallback_points, dtype=float),
        output_path=outputs_dir / "day6_trajectory.png",
    )
    demo_video = render_day6_demo_video(
        times=np.asarray(trace.times, dtype=float),
        target_positions=np.asarray(trace.target_positions, dtype=float),
        interceptor_positions=np.asarray(trace.interceptor_positions, dtype=float),
        drifted_positions=np.asarray(trace.drifted_positions, dtype=float),
        fused_positions=np.asarray(trace.fused_positions, dtype=float),
        distances=np.asarray(trace.distances_m, dtype=float),
        edge_fps=edge_trace.mean_loop_fps,
        node_fps=trace.node_fps,
        fallback_flags=np.asarray(trace.fallback_flags, dtype=bool),
        output_path=outputs_dir / "day6_demo.mp4",
        fps=float(base_config.get("day4", {}).get("demo_fps", 20.44)),
        frame_size=(
            int(base_config.get("day4", {}).get("video_width", 1280)),
            int(base_config.get("day4", {}).get("video_height", 720)),
        ),
    )
    compatibility_video = demo_video.with_suffix(".avi")

    metrics = Day6Metrics(
        success=trace.success,
        interception_time_s=trace.interception_time_s,
        mean_loop_fps=trace.mean_loop_fps,
        edge_mode_fps=edge_trace.mean_loop_fps,
        px4_setpoints_sent=trace.px4_setpoints_sent,
        fallback_activations=trace.fallback_activations,
        node_fps=trace.node_fps,
        airsim_mode=trace.airsim_mode,
    )
    summary_payload = {
        "metrics": {
            "success": metrics.success,
            "interception_time_s": metrics.interception_time_s,
            "mean_loop_fps": metrics.mean_loop_fps,
            "edge_mode_fps": metrics.edge_mode_fps,
            "px4_setpoints_sent": metrics.px4_setpoints_sent,
            "fallback_activations": metrics.fallback_activations,
            "node_fps": metrics.node_fps,
            "airsim_mode": metrics.airsim_mode,
        },
        "artifacts": {
            "trajectory_plot": str(trajectory_plot),
            "demo_video": str(demo_video),
            "compatibility_video": str(compatibility_video) if compatibility_video.exists() else None,
        },
        "px4_sitl_command": trace.sitl_command,
    }
    summary_json = outputs_dir / "day6_summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    log_file = logs_dir / "day6.log"
    log_file.write_text(build_report(metrics=metrics, artifacts=Day6Artifacts(
        trajectory_plot=trajectory_plot,
        demo_video=demo_video,
        compatibility_video=compatibility_video if compatibility_video.exists() else None,
        log_file=log_file,
        summary_json=summary_json,
    ), trace=trace, edge_trace=edge_trace), encoding="utf-8")

    return metrics, Day6Artifacts(
        trajectory_plot=trajectory_plot,
        demo_video=demo_video,
        compatibility_video=compatibility_video if compatibility_video.exists() else None,
        log_file=log_file,
        summary_json=summary_json,
    )


def run_day6_validation_sweep(
    project_root: str | Path,
    seeds: Sequence[int] = (41, 42, 43, 44),
    max_steps_override: int | None = 160,
) -> Day6SweepMetrics:
    root = Path(project_root).resolve()
    successes = 0
    interception_times: list[float] = []

    for seed in seeds:
        config = load_config(root / "configs" / "default.yaml")
        _apply_day6_tuning(config, random_seed=int(seed), max_steps_override=max_steps_override)
        trace = _run_architecture(
            config=config,
            edge_profile=EdgeProfile(enabled=False),
            use_airsim=False,
        )
        successes += int(trace.success)
        if trace.interception_time_s is not None:
            interception_times.append(float(trace.interception_time_s))

    return Day6SweepMetrics(
        success_rate=float(successes / max(len(seeds), 1)),
        runs=len(seeds),
        seeds=tuple(int(seed) for seed in seeds),
        mean_interception_time_s=float(np.mean(np.asarray(interception_times, dtype=float))) if interception_times else 0.0,
    )


def _apply_day6_tuning(config: dict[str, Any], random_seed: int, max_steps_override: int | None) -> None:
    config.setdefault("system", {})["random_seed"] = int(random_seed)
    config.setdefault("tracking", {})["mode"] = "kalman"
    config.setdefault("control", {})["mode"] = "mpc"
    _apply_day4_tuning(config)

    mission = config.setdefault("mission", {})
    mission["max_steps"] = int(max_steps_override) if max_steps_override is not None else max(int(mission.get("max_steps", 250)), 300)

    planning = config.setdefault("planning", {})
    planning["fallback_replan_step"] = min(int(mission["max_steps"] * 0.45), 130)
    planning["replan_distance_margin_m"] = max(float(planning.get("replan_distance_margin_m", 18.0)), 18.0)
    planning["fallback_lead_weight"] = max(float(planning.get("fallback_lead_weight", 0.82)), 0.82)
    planning["fallback_altitude_bias_m"] = float(planning.get("fallback_altitude_bias_m", 0.0))

    simulation = config.setdefault("simulation", {})
    simulation["target_initial_position"] = [285.0, 145.0, 120.0]
    simulation["target_initial_velocity"] = [-6.5, 2.1, 0.0]
    simulation["target_process_noise_std_mps2"] = min(float(simulation.get("target_process_noise_std_mps2", 0.35)), 0.22)
    simulation["wind_disturbance_std_mps2"] = min(float(simulation.get("wind_disturbance_std_mps2", 0.15)), 0.08)

    navigation = config.setdefault("navigation", {})
    navigation["gps_drift_rate_mps"] = min(max(float(navigation.get("gps_drift_rate_mps", 0.2)), 0.28), 0.32)
    navigation["gps_noise_std_m"] = max(float(navigation.get("gps_noise_std_m", 1.5)), 1.7)
    navigation["measurement_noise_scale"] = max(float(navigation.get("measurement_noise_scale", 1.0)), 1.08)

    control = config.setdefault("control", {})
    control["mpc_guidance_blend"] = max(float(control.get("mpc_guidance_blend", 0.25)), 0.32)

    airsim = config.setdefault("airsim", {})
    airsim.setdefault("px4_root", "PX4-Autopilot")
    airsim.setdefault("vehicle_model", "iris")
    airsim.setdefault("target_system", "px4_sitl")


def _run_architecture(
    config: dict[str, Any],
    edge_profile: EdgeProfile,
    use_airsim: bool,
    max_steps: int | None = None,
) -> _ArchitectureTrace:
    run_config = copy.deepcopy(config)
    if max_steps is not None:
        run_config.setdefault("mission", {})["max_steps"] = int(max_steps)
    env = DroneInterceptionEnv(run_config)
    bus = LocalTopicBus()
    perception_node = LocalPerceptionNode(run_config, bus=bus, edge_profile=edge_profile)
    tracking_node = LocalTrackingNode(run_config, bus=bus)
    navigation_node = LocalNavigationNode(run_config, bus=bus)
    control_node = LocalControlNode(run_config, bus=bus, use_airsim=use_airsim)

    observation = env.reset()
    dt = float(run_config["mission"]["time_step"])
    max_steps = int(run_config["mission"]["max_steps"])
    intercept_threshold = float(run_config["planning"]["desired_intercept_distance_m"])
    started_at = time.perf_counter()

    times: list[float] = []
    target_positions: list[np.ndarray] = []
    interceptor_positions: list[np.ndarray] = []
    drifted_positions: list[np.ndarray] = []
    fused_positions: list[np.ndarray] = []
    distances_m: list[float] = []
    fallback_flags: list[bool] = []
    fallback_points: list[np.ndarray] = []
    log_lines = [
        f"edge_mode={edge_profile.enabled}",
        "pipeline=camera -> perception_node -> tracking_node -> navigation_node -> control_node -> px4_sitl -> airsim",
    ]

    success = False
    interception_time_s: float | None = None
    px4_setpoints_sent = 0
    fallback_activations = 0
    airsim_mode = "dry_run"
    sitl_command = ""

    for step in range(max_steps):
        navigation_payload = navigation_node.process(observation["sensor_packet"])
        detection_payload = perception_node.process(observation, step=step)
        tracking_payload = tracking_node.process(detection_payload)
        true_distance_m = float(np.linalg.norm(env.target_state.position - env.interceptor_state.position))
        control_cycle = control_node.process(
            navigation_payload=navigation_payload,
            tracking_payload=tracking_payload,
            step=step,
            dt=dt,
            true_distance_m=true_distance_m,
        )
        observation, done, info = env.step(control_cycle.command)

        px4_setpoints_sent += 1
        fallback_activations += int(control_cycle.fallback_used)
        airsim_mode = control_cycle.airsim_mode
        sitl_command = str(control_cycle.command_payload.get("px4_sitl_command", sitl_command))

        times.append(float(observation["time"][0]))
        target_positions.append(env.target_state.position.copy())
        interceptor_positions.append(env.interceptor_state.position.copy())
        drifted_positions.append(np.asarray(navigation_payload["drifted_position"], dtype=float))
        fused_positions.append(np.asarray(navigation_payload["position"], dtype=float))
        distances_m.append(float(info["distance_to_target"]))
        fallback_flags.append(bool(control_cycle.fallback_used))
        if control_cycle.fallback_used:
            fallback_points.append(np.asarray(control_cycle.plan.intercept_point, dtype=float).copy())

        if step == 0 or step % 10 == 0 or control_cycle.fallback_used or done:
            log_lines.append(
                f"step={step:03d} distance_m={distances_m[-1]:8.3f} "
                f"fallback={'yes' if control_cycle.fallback_used else 'no'} "
                f"px4_mode={control_cycle.command_payload.get('px4_mode', 'unknown')} "
                f"airsim_mode={airsim_mode}"
            )

        if done:
            success = distances_m[-1] <= intercept_threshold
            interception_time_s = float(observation["time"][0]) if success else None
            break

    elapsed = max(time.perf_counter() - started_at, 1e-6)
    mean_loop_fps = float(len(times) / elapsed) if times else 0.0
    node_fps = {
        perception_node.stats.name: float(perception_node.stats.publishes / elapsed) if elapsed > 0.0 else 0.0,
        tracking_node.stats.name: float(tracking_node.stats.publishes / elapsed) if elapsed > 0.0 else 0.0,
        navigation_node.stats.name: float(navigation_node.stats.publishes / elapsed) if elapsed > 0.0 else 0.0,
        control_node.stats.name: float(control_node.stats.publishes / elapsed) if elapsed > 0.0 else 0.0,
    }
    log_lines.extend(
        [
            f"success={'yes' if success else 'no'}",
            f"interception_time_s={interception_time_s if interception_time_s is not None else 'n/a'}",
            f"mean_loop_fps={mean_loop_fps:.3f}",
            f"fallback_activations={fallback_activations}",
            f"px4_setpoints_sent={px4_setpoints_sent}",
            f"sitl_command={sitl_command}",
        ]
    )

    return _ArchitectureTrace(
        times=times,
        target_positions=target_positions,
        interceptor_positions=interceptor_positions,
        drifted_positions=drifted_positions,
        fused_positions=fused_positions,
        distances_m=distances_m,
        fallback_flags=fallback_flags,
        fallback_points=fallback_points,
        node_fps=node_fps,
        success=success,
        interception_time_s=interception_time_s,
        mean_loop_fps=mean_loop_fps,
        px4_setpoints_sent=px4_setpoints_sent,
        fallback_activations=fallback_activations,
        airsim_mode=airsim_mode,
        sitl_command=sitl_command,
        log_lines=log_lines,
    )


def build_report(metrics: Day6Metrics, artifacts: Day6Artifacts, trace: _ArchitectureTrace, edge_trace: _ArchitectureTrace) -> str:
    node_lines = [f"{name}={fps:.2f}" for name, fps in metrics.node_fps.items()]
    lines = [
        "DAY 6 FLIGHT-READY ARCHITECTURE REPORT",
        f"success={'YES' if metrics.success else 'NO'}",
        f"interception_time_s={metrics.interception_time_s if metrics.interception_time_s is not None else 'n/a'}",
        f"mean_loop_fps={metrics.mean_loop_fps:.2f}",
        f"edge_mode_fps={metrics.edge_mode_fps:.2f}",
        f"px4_setpoints_sent={metrics.px4_setpoints_sent}",
        f"fallback_activations={metrics.fallback_activations}",
        f"node_fps={', '.join(node_lines)}",
        f"airsim_mode={metrics.airsim_mode}",
        f"px4_sitl_command={trace.sitl_command}",
        "",
        "ARTIFACTS",
        f"trajectory_plot={artifacts.trajectory_plot}",
        f"demo_video={artifacts.demo_video}",
        f"compatibility_video={artifacts.compatibility_video}" if artifacts.compatibility_video is not None else "compatibility_video=n/a",
        f"summary_json={artifacts.summary_json}",
        "",
        "MAIN TRACE",
        *trace.log_lines,
        "",
        "EDGE TRACE",
        *edge_trace.log_lines,
    ]
    return "\n".join(lines) + "\n"


def print_report(metrics: Day6Metrics, artifacts: Day6Artifacts) -> None:
    print("DAY 6 FLIGHT-READY ARCHITECTURE REPORT")
    print(f"- Success: {'YES' if metrics.success else 'NO'}")
    print(f"- Interception Time: {metrics.interception_time_s:.2f}s" if metrics.interception_time_s is not None else "- Interception Time: n/a")
    print(f"- Mean Loop FPS: {metrics.mean_loop_fps:.2f}")
    print(f"- Edge Mode FPS: {metrics.edge_mode_fps:.2f}")
    print(f"- Fallback Activations: {metrics.fallback_activations}")
    print(f"- PX4 Setpoints Sent: {metrics.px4_setpoints_sent}")
    print(f"- Node FPS: " + ", ".join(f"{name}={fps:.2f}" for name, fps in metrics.node_fps.items()))
    print(f"- AirSim Mode: {metrics.airsim_mode}")
    print(f"- Trajectory Plot: {artifacts.trajectory_plot}")
    print(f"- Demo Video: {artifacts.demo_video}")
    if artifacts.compatibility_video is not None:
        print(f"- Compatibility Video: {artifacts.compatibility_video}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Day 6 flight-ready architecture demo.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root directory.")
    parser.add_argument("--seed", type=int, default=41, help="Base random seed for the Day 6 architecture run.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional override for mission steps.")
    parser.add_argument("--use-airsim", action="store_true", help="Dispatch to a live AirSim client if available.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    metrics, artifacts = run_day6_execution(
        project_root=args.project_root.resolve(),
        random_seed=args.seed,
        max_steps_override=args.max_steps,
        use_airsim=args.use_airsim,
    )
    print_report(metrics=metrics, artifacts=artifacts)


__all__ = ["run_day6_execution", "run_day6_validation_sweep"]


if __name__ == "__main__":
    main()
