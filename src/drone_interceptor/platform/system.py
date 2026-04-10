from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from drone_interceptor.analytics.metrics import PlatformMetrics, ScenarioResultRecord, mean, rmse
from drone_interceptor.analytics.visualization import (
    plot_cost_vs_iteration,
    plot_distance_vs_time,
    plot_fps_vs_model,
    plot_platform_3d_trajectory,
    plot_rmse_vs_noise,
    plot_success_rate_vs_scenario,
)
from drone_interceptor.config import load_config
from drone_interceptor.constraints import ConstraintStatus, load_constraint_envelope
from drone_interceptor.core import (
    InterceptionController,
    InterceptionCostModel,
    TargetDetector,
    TargetPredictor,
    TargetTracker,
)
from drone_interceptor.navigation.sensor_fusion import GPSIMUKalmanFusion
from drone_interceptor.navigation.state_estimator import simulate_gps_with_drift
from drone_interceptor.planning.planner import InterceptPlanner
from drone_interceptor.simulation.airsim import AirSimInterceptorAdapter
from drone_interceptor.simulation.environment import DroneInterceptionEnv
from drone_interceptor.simulation.scenarios import (
    ScenarioDefinition,
    build_platform_scenario_config,
    platform_scenarios,
)
from drone_interceptor.types import Detection, TargetState
from drone_interceptor.validation.day4 import _apply_day4_tuning
from drone_interceptor.visualization.day5 import render_day5_demo_video


@dataclass(slots=True)
class EdgeProfile:
    enabled: bool = False
    detection_stride: int = 1
    injected_latency_s: float = 0.0


@dataclass(slots=True)
class ScenarioTrace:
    record: ScenarioResultRecord
    scenario: ScenarioDefinition
    times: list[float]
    target_positions: list[np.ndarray]
    interceptor_positions: list[np.ndarray]
    drifted_positions: list[np.ndarray]
    fused_positions: list[np.ndarray]
    distances_m: list[float]
    tracking_errors_m: list[float]
    stage_costs: list[float]
    fps_samples: list[float]
    commanded_speeds: list[float]


@dataclass(frozen=True)
class PlatformArtifacts:
    final_demo_video: Path
    final_3d_plot: Path
    scenario_results_csv: Path
    dashboard_preview_html: Path
    summary_log: Path


def run_platform_demo(
    project_root: str | Path,
    random_seed: int = 61,
    max_steps_override: int | None = None,
    use_airsim: bool = False,
) -> tuple[list[ScenarioResultRecord], PlatformMetrics, PlatformArtifacts]:
    root = Path(project_root).resolve()
    outputs_dir = root / "outputs"
    logs_dir = root / "logs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_config(root / "configs" / "default.yaml")
    _apply_platform_tuning(base_config, random_seed=random_seed, max_steps_override=max_steps_override)

    traces: list[ScenarioTrace] = []
    for index, scenario in enumerate(platform_scenarios(), start=1):
        scenario_config = build_platform_scenario_config(
            base_config=base_config,
            scenario=scenario,
            random_seed=random_seed + index,
            max_steps_override=max_steps_override,
        )
        trace = _run_scenario(
            config=scenario_config,
            scenario=scenario,
            log_file=logs_dir / f"scenario_{index}.log",
            use_airsim=use_airsim,
            edge_profile=EdgeProfile(enabled=False),
        )
        traces.append(trace)

    records = [trace.record for trace in traces]
    metrics = PlatformMetrics(
        success_rate=float(sum(1 for record in records if record.success) / max(len(records), 1)),
        mean_interception_time_s=mean([record.interception_time_s for record in records if record.interception_time_s is not None]),
        mean_rmse_m=mean([record.rmse_m for record in records]),
        mean_fps=mean([record.mean_loop_fps for record in records]),
    )

    representative = next((trace for trace in traces if trace.scenario.name == "high_drift"), traces[-1])
    edge_benchmark = _run_scenario(
        config=build_platform_scenario_config(
            base_config=base_config,
            scenario=platform_scenarios()[0],
            random_seed=random_seed + 100,
            max_steps_override=min(max_steps_override or int(base_config["mission"]["max_steps"]), 80),
        ),
        scenario=platform_scenarios()[0],
        log_file=logs_dir / "scenario_edge.log",
        use_airsim=False,
        edge_profile=EdgeProfile(enabled=True, detection_stride=3, injected_latency_s=0.012),
    )

    final_3d_plot = plot_platform_3d_trajectory(
        target_positions=np.asarray(representative.target_positions, dtype=float),
        interceptor_positions=np.asarray(representative.interceptor_positions, dtype=float),
        drifted_positions=np.asarray(representative.drifted_positions, dtype=float),
        fused_positions=np.asarray(representative.fused_positions, dtype=float),
        output_path=outputs_dir / "final_3d_plot.png",
        intercept_point=np.asarray(representative.interceptor_positions[-1], dtype=float) if representative.record.success else None,
    )
    plot_distance_vs_time(
        series={
            trace.scenario.name: (np.asarray(trace.times, dtype=float), np.asarray(trace.distances_m, dtype=float))
            for trace in traces
        },
        threshold_m=float(base_config["planning"]["desired_intercept_distance_m"]),
        output_path=outputs_dir / "distance_vs_time.png",
    )
    plot_rmse_vs_noise(
        scenario_names=[trace.scenario.name for trace in traces],
        noise_levels=[trace.record.noise_level for trace in traces],
        rmses=[trace.record.rmse_m for trace in traces],
        output_path=outputs_dir / "rmse_vs_noise.png",
    )
    plot_cost_vs_iteration(
        iterations=np.arange(len(representative.stage_costs), dtype=int),
        costs=np.asarray(representative.stage_costs, dtype=float),
        output_path=outputs_dir / "cost_vs_iteration.png",
    )
    plot_fps_vs_model(
        model_labels=["standard", "edge_profile"],
        fps_values=[traces[0].record.mean_loop_fps, edge_benchmark.record.mean_loop_fps],
        output_path=outputs_dir / "fps_vs_model.png",
    )
    plot_success_rate_vs_scenario(
        scenario_names=[trace.scenario.name for trace in traces],
        success_values=[1.0 if trace.record.success else 0.0 for trace in traces],
        output_path=outputs_dir / "success_rate_vs_scenario.png",
    )
    final_demo_video = render_day5_demo_video(
        times=np.asarray(representative.times, dtype=float),
        target_positions=np.asarray(representative.target_positions, dtype=float),
        interceptor_positions=np.asarray(representative.interceptor_positions, dtype=float),
        drifted_positions=np.asarray(representative.drifted_positions, dtype=float),
        fused_positions=np.asarray(representative.fused_positions, dtype=float),
        distances=np.asarray(representative.distances_m, dtype=float),
        tracking_errors=np.asarray(representative.tracking_errors_m, dtype=float),
        fps_samples=np.asarray(representative.fps_samples, dtype=float),
        output_path=outputs_dir / "final_demo.mp4",
        scenario_name=representative.scenario.name,
        drift_rate_mps=float(base_config["navigation"]["gps_drift_rate_mps"]),
        fps=float(base_config.get("day4", {}).get("demo_fps", 20.44)),
        frame_size=(
            int(base_config.get("day4", {}).get("video_width", 1280)),
            int(base_config.get("day4", {}).get("video_height", 720)),
        ),
    )

    scenario_results_csv = outputs_dir / "scenario_results.csv"
    _write_results_csv(records, scenario_results_csv)
    summary_log = logs_dir / "final_platform.log"
    summary_log.write_text(_build_platform_report(records, metrics, final_demo_video, final_3d_plot), encoding="utf-8")

    preview_html = _build_preview_html(records=records, metrics=metrics, outputs_dir=outputs_dir)
    dashboard_preview_html = outputs_dir / "platform_preview.html"
    dashboard_preview_html.write_text(preview_html, encoding="utf-8")
    (outputs_dir / "demo_preview.html").write_text(preview_html, encoding="utf-8")

    return records, metrics, PlatformArtifacts(
        final_demo_video=final_demo_video,
        final_3d_plot=final_3d_plot,
        scenario_results_csv=scenario_results_csv,
        dashboard_preview_html=dashboard_preview_html,
        summary_log=summary_log,
    )


def _apply_platform_tuning(config: dict[str, Any], random_seed: int, max_steps_override: int | None) -> None:
    config.setdefault("system", {})["random_seed"] = int(random_seed)
    config.setdefault("tracking", {})["mode"] = "kalman"
    config.setdefault("control", {})["mode"] = "mpc"
    _apply_day4_tuning(config)

    mission = config.setdefault("mission", {})
    mission["max_steps"] = int(max_steps_override) if max_steps_override is not None else max(int(mission.get("max_steps", 250)), 220)
    config.setdefault("planning", {})["desired_intercept_distance_m"] = max(
        float(config["planning"].get("desired_intercept_distance_m", 10.25)),
        11.0,
    )
    config.setdefault("planning", {})["fallback_lead_weight"] = 0.8


def _run_scenario(
    config: dict[str, Any],
    scenario: ScenarioDefinition,
    log_file: Path,
    use_airsim: bool,
    edge_profile: EdgeProfile,
) -> ScenarioTrace:
    env = DroneInterceptionEnv(config)
    detector = TargetDetector(config)
    tracker = TargetTracker(config)
    predictor = TargetPredictor(config)
    planner = InterceptPlanner(config)
    controller = InterceptionController(config)
    navigator = GPSIMUKalmanFusion(config)
    cost_model = InterceptionCostModel.from_config(config)
    constraint_envelope = load_constraint_envelope(config)
    airsim_adapter = AirSimInterceptorAdapter.from_config(config, connect=use_airsim)

    observation = env.reset()
    dt = float(config["mission"]["time_step"])
    max_steps = int(config["mission"]["max_steps"])
    intercept_threshold_m = float(config["planning"]["desired_intercept_distance_m"])
    start_time = time.perf_counter()
    last_detection: Detection | None = None

    times: list[float] = []
    target_positions: list[np.ndarray] = []
    interceptor_positions: list[np.ndarray] = []
    drifted_positions: list[np.ndarray] = []
    fused_positions: list[np.ndarray] = []
    distances_m: list[float] = []
    tracking_errors_m: list[float] = []
    stage_costs: list[float] = []
    fps_samples: list[float] = []
    commanded_speeds: list[float] = []
    log_lines = [
        f"scenario={scenario.name}",
        f"description={scenario.description}",
        f"edge_mode={edge_profile.enabled}",
    ]

    success = False
    interception_time_s: float | None = None
    terminal_distance_m = float("inf")

    for step in range(max_steps):
        _apply_scenario_dynamics(env=env, scenario=scenario, step=step, dt=dt)
        navigation_state = navigator.update(observation["sensor_packet"])
        interceptor_estimate = TargetState(
            position=navigation_state.position.copy(),
            velocity=navigation_state.velocity.copy(),
            covariance=None if navigation_state.covariance is None else navigation_state.covariance.copy(),
            timestamp=navigation_state.timestamp,
            metadata=dict(navigation_state.metadata),
        )

        if edge_profile.enabled and edge_profile.injected_latency_s > 0.0:
            time.sleep(edge_profile.injected_latency_s)
        if edge_profile.enabled and edge_profile.detection_stride > 1 and step % edge_profile.detection_stride != 0 and last_detection is not None:
            detection = Detection(
                position=last_detection.position.copy(),
                confidence=last_detection.confidence,
                metadata={**last_detection.metadata, "edge_reused_detection": True},
                timestamp=float(observation["time"][0]),
            )
        else:
            detection = detector.detect(observation)
            last_detection = detection

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
        airsim_adapter.dispatch(command, altitude_m=float(interceptor_estimate.position[2]), dt=dt)
        observation, done, info = env.step(command)
        constraint_status = ConstraintStatus(
            velocity_clipped=bool(command.metadata.get("velocity_clipped", False)),
            acceleration_clipped=bool(command.metadata.get("acceleration_clipped", False)),
            tracking_ok=tracking_error_m <= constraint_envelope.tracking_precision_m,
            drift_rate_in_bounds=bool(navigation_state.metadata.get("drift_rate_in_bounds", True)),
            safety_override=bool(command.metadata.get("safety_override", False)),
            distance_to_target_m=float(info["distance_to_target"]),
        )
        stage_cost = cost_model.stage_cost(
            interceptor_position=env.interceptor_state.position,
            target_position=env.target_state.position,
            control_input=command.acceleration_command if command.acceleration_command is not None else command.velocity_command,
            constraint_status=constraint_status,
            uncertainty_term=float(plan.metadata.get("uncertainty_trace", 0.0)),
        )

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
        distances_m.append(float(info["distance_to_target"]))
        tracking_errors_m.append(tracking_error_m)
        stage_costs.append(float(stage_cost))
        fps_samples.append(loop_fps)
        commanded_speeds.append(float(np.linalg.norm(command.velocity_command)))
        terminal_distance_m = float(info["distance_to_target"])

        if step == 0 or step % 10 == 0 or done:
            log_lines.append(
                f"step={step:03d} time_s={times[-1]:6.2f} distance_m={terminal_distance_m:8.3f} "
                f"tracking_error_m={tracking_error_m:6.3f} cost={stage_cost:8.3f} loop_fps={loop_fps:7.2f}"
            )

        if done:
            success = terminal_distance_m <= intercept_threshold_m
            interception_time_s = float(observation["time"][0]) if success else None
            break

    mean_loop_fps = float(len(times) / max(time.perf_counter() - start_time, 1e-6)) if times else 0.0
    record = ScenarioResultRecord(
        scenario=scenario.name,
        success=success,
        interception_time_s=interception_time_s,
        rmse_m=rmse(tracking_errors_m),
        mean_loop_fps=mean_loop_fps,
        mean_stage_cost=mean(stage_costs),
        final_distance_m=terminal_distance_m,
        noise_level=scenario.noise_level,
        model_label="edge_profile" if edge_profile.enabled else "standard",
        log_file=str(log_file),
    )
    log_lines.extend(
        [
            f"success={'yes' if record.success else 'no'}",
            f"interception_time_s={record.interception_time_s if record.interception_time_s is not None else 'n/a'}",
            f"rmse_m={record.rmse_m:.3f}",
            f"mean_loop_fps={record.mean_loop_fps:.3f}",
            f"mean_stage_cost={record.mean_stage_cost:.3f}",
            f"final_distance_m={record.final_distance_m:.3f}",
        ]
    )
    log_file.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    return ScenarioTrace(
        record=record,
        scenario=scenario,
        times=times,
        target_positions=target_positions,
        interceptor_positions=interceptor_positions,
        drifted_positions=drifted_positions,
        fused_positions=fused_positions,
        distances_m=distances_m,
        tracking_errors_m=tracking_errors_m,
        stage_costs=stage_costs,
        fps_samples=fps_samples,
        commanded_speeds=commanded_speeds,
    )


def _apply_scenario_dynamics(env: DroneInterceptionEnv, scenario: ScenarioDefinition, step: int, dt: float) -> None:
    if scenario.name != "zig_zag_motion":
        return
    time_s = step * dt
    lateral_velocity = scenario.zigzag_amplitude_mps * math.sin(2.0 * math.pi * scenario.zigzag_frequency_hz * time_s)
    env.target_state.velocity[1] = lateral_velocity


def _write_results_csv(records: list[ScenarioResultRecord], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "success",
                "interception_time_s",
                "rmse_m",
                "mean_loop_fps",
                "mean_stage_cost",
                "final_distance_m",
                "noise_level",
                "model_label",
                "log_file",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "scenario": record.scenario,
                    "success": int(record.success),
                    "interception_time_s": record.interception_time_s,
                    "rmse_m": f"{record.rmse_m:.6f}",
                    "mean_loop_fps": f"{record.mean_loop_fps:.6f}",
                    "mean_stage_cost": f"{record.mean_stage_cost:.6f}",
                    "final_distance_m": f"{record.final_distance_m:.6f}",
                    "noise_level": f"{record.noise_level:.6f}",
                    "model_label": record.model_label,
                    "log_file": record.log_file,
                }
            )


def _build_platform_report(
    records: list[ScenarioResultRecord],
    metrics: PlatformMetrics,
    final_demo_video: Path,
    final_3d_plot: Path,
) -> str:
    lines = [
        "DAY 5 PLATFORM REPORT",
        "Scenario | Success | Time [s] | RMSE [m] | FPS | Final Distance [m]",
    ]
    for record in records:
        time_value = f"{record.interception_time_s:.2f}" if record.interception_time_s is not None else "n/a"
        lines.append(
            f"{record.scenario} | {'YES' if record.success else 'NO'} | {time_value} | "
            f"{record.rmse_m:.3f} | {record.mean_loop_fps:.2f} | {record.final_distance_m:.3f}"
        )
    lines.extend(
        [
            "",
            f"success_rate={metrics.success_rate:.2%}",
            f"mean_interception_time_s={metrics.mean_interception_time_s:.2f}",
            f"mean_rmse_m={metrics.mean_rmse_m:.3f}",
            f"mean_fps={metrics.mean_fps:.2f}",
            f"final_demo={final_demo_video}",
            f"final_3d_plot={final_3d_plot}",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_preview_html(records: list[ScenarioResultRecord], metrics: PlatformMetrics, outputs_dir: Path) -> str:
    rows = "\n".join(
        f"<tr><td>{record.scenario}</td><td>{'YES' if record.success else 'NO'}</td>"
        f"<td>{f'{record.interception_time_s:.2f}' if record.interception_time_s is not None else 'n/a'}</td>"
        f"<td>{record.rmse_m:.3f}</td><td>{record.mean_loop_fps:.2f}</td><td>{record.final_distance_m:.3f}</td></tr>"
        for record in records
    )
    latest_cinematic = next(
        iter(sorted(outputs_dir.glob("day8_bms_demo_*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)),
        None,
    )
    cinematic_card = ""
    cinematic_link = ""
    if latest_cinematic is not None:
        cinematic_card = f"""
      <section class="card">
        <h2>Day 8 Cinematic Replay</h2>
        <video controls preload="metadata">
          <source src="{latest_cinematic.name}" type="video/mp4" />
        </video>
        <div class="links">
          <a href="{latest_cinematic.name}">Open MP4</a>
        </div>
      </section>"""
        cinematic_link = f'\n        <a href="{latest_cinematic.name}">Day 8 Cinematic MP4</a>'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Drone Interceptor Mission Control</title>
  <style>
    :root {{
      --bg: #0e141b;
      --panel: rgba(22, 31, 40, 0.9);
      --line: #293645;
      --text: #edf3f8;
      --muted: #98a9bc;
      --accent: #73d7ff;
      --accent-2: #ff9f5a;
      --ok: #73f0a0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "Segoe UI", system-ui, sans-serif;
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at top right, rgba(30, 85, 120, 0.35), transparent 30%),
        radial-gradient(circle at left center, rgba(120, 70, 30, 0.22), transparent 28%),
        linear-gradient(160deg, #0c1117 0%, #121a23 45%, #0d141b 100%);
    }}
    main {{
      max-width: 1360px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }}
    .hero {{
      border: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(20, 33, 46, 0.92), rgba(18, 25, 33, 0.88));
      border-radius: 24px;
      padding: 28px;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.32);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(2rem, 4vw, 3.2rem);
      letter-spacing: 0.02em;
    }}
    .subtitle {{
      color: var(--muted);
      max-width: 820px;
      line-height: 1.6;
      margin: 0;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 22px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      background: rgba(12, 18, 24, 0.66);
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 0.85rem;
      margin-bottom: 6px;
    }}
    .metric strong {{
      font-size: 1.6rem;
      color: var(--accent);
    }}
    .section-title {{
      margin: 28px 0 14px;
      font-size: 1.1rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #d6e4f1;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
      backdrop-filter: blur(8px);
    }}
    .card h2 {{
      margin-top: 0;
      font-size: 1.05rem;
      margin-bottom: 14px;
    }}
    video, img {{
      width: 100%;
      border-radius: 14px;
      background: #000;
      border: 1px solid #334253;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    th, td {{
      padding: 12px 10px;
      border-bottom: 1px solid #223140;
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.78rem;
      letter-spacing: 0.06em;
    }}
    .ok {{ color: var(--ok); font-weight: 700; }}
    .bad {{ color: #ff8e8e; font-weight: 700; }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
    }}
    .links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      margin-top: 12px;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{ text-decoration: underline; }}
    .note {{
      color: var(--muted);
      line-height: 1.5;
      font-size: 0.95rem;
    }}
    .badge {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(115, 215, 255, 0.12);
      color: var(--accent);
      border: 1px solid rgba(115, 215, 255, 0.24);
      font-size: 0.8rem;
      margin-right: 8px;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="badge">Competition Demo</div>
      <div class="badge">Simulation + Analytics + Frontend</div>
      <h1>Drone Interceptor Mission Control</h1>
      <p class="subtitle">
        This browser dashboard combines the modular UAV interception platform, multi-scenario benchmarking,
        analytics outputs, and simulation demos in one place. It is meant to be opened in a browser, not in the VS Code text editor.
      </p>
      <div class="metrics">
        <div class="metric"><span>Success Rate</span><strong>{metrics.success_rate * 100:.0f}%</strong></div>
        <div class="metric"><span>Mean Interception Time</span><strong>{metrics.mean_interception_time_s:.2f}s</strong></div>
        <div class="metric"><span>Mean RMSE</span><strong>{metrics.mean_rmse_m:.3f}m</strong></div>
        <div class="metric"><span>Mean FPS</span><strong>{metrics.mean_fps:.2f}</strong></div>
      </div>
    </section>

    <div class="section-title">Mission Demos</div>
    <div class="grid">
      <section class="card">
        <h2>Platform Final Demo</h2>
        <video controls preload="metadata">
          <source src="final_demo.mp4" type="video/mp4" />
        </video>
        <div class="links">
          <a href="final_demo.mp4">Open MP4</a>
        </div>
      </section>
      <section class="card">
        <h2>Day 4 Optimized Demo</h2>
        <video controls preload="metadata">
          <source src="day4_demo.mp4" type="video/mp4" />
        </video>
        <div class="links">
          <a href="day4_demo.mp4">Open MP4</a>
        </div>
      </section>
      <section class="card">
        <h2>Day 6 Flight-Ready Demo</h2>
        <video controls preload="metadata">
          <source src="day6_demo.mp4" type="video/mp4" />
        </video>
        <div class="links">
          <a href="day6_demo.mp4">Open MP4</a>
        </div>
      </section>
      <section class="card">
        <h2>Day 7 Intelligent Drift Demo</h2>
        <video controls preload="metadata">
          <source src="day7_demo.mp4" type="video/mp4" />
        </video>
        <div class="links">
          <a href="day7_demo.mp4">Open MP4</a>
        </div>
      </section>
      <section class="card">
        <h2>Day 9 DP5 Redirection Demo</h2>
        <video controls preload="metadata">
          <source src="day9_dp5_demo.mp4" type="video/mp4" />
        </video>
        <div class="links">
          <a href="day9_dp5_demo.mp4">Open MP4</a>
        </div>
      </section>
{cinematic_card}
    </div>

    <div class="section-title">Scenario Results</div>
    <section class="card">
      <table>
        <thead>
          <tr>
            <th>Scenario</th>
            <th>Success</th>
            <th>Time [s]</th>
            <th>RMSE [m]</th>
            <th>FPS</th>
            <th>Final Distance [m]</th>
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
      <p class="note">
        Scenario engine includes <code>slow_drone</code>, <code>fast_drone</code>, <code>zig_zag_motion</code>,
        <code>noisy_environment</code>, and <code>high_drift</code>.
      </p>
      <div class="links">
        <a href="scenario_results.csv">Open CSV</a>
      </div>
    </section>

    <div class="section-title">Analytics</div>
    <div class="gallery">
      <section class="card"><h2>Final 3D Plot</h2><img src="final_3d_plot.png" alt="Final 3D Plot" /></section>
      <section class="card"><h2>Distance vs Time</h2><img src="distance_vs_time.png" alt="Distance vs Time" /></section>
      <section class="card"><h2>RMSE vs Noise</h2><img src="rmse_vs_noise.png" alt="RMSE vs Noise" /></section>
      <section class="card"><h2>Cost vs Iteration</h2><img src="cost_vs_iteration.png" alt="Cost vs Iteration" /></section>
      <section class="card"><h2>FPS vs Model</h2><img src="fps_vs_model.png" alt="FPS vs Model" /></section>
      <section class="card"><h2>Success Rate vs Scenario</h2><img src="success_rate_vs_scenario.png" alt="Success Rate vs Scenario" /></section>
    </div>

    <div class="section-title">Artifacts</div>
    <section class="card">
      <p class="note">
        The MP4 artifacts on this page are browser-ready H.264 outputs.
      </p>
      <div class="links">
        <a href="day4_demo.mp4">Day 4 MP4</a>
        <a href="day6_demo.mp4">Day 6 MP4</a>
        <a href="day7_demo.mp4">Day 7 MP4</a>
        <a href="day9_dp5_demo.mp4">Day 9 MP4</a>
        <a href="final_demo.mp4">Platform MP4</a>
{cinematic_link}
        <a href="day6_trajectory.png">Day 6 3D Plot</a>
        <a href="platform_preview.html">Preview Page</a>
      </div>
    </section>
  </main>
</body>
</html>
"""


__all__ = ["PlatformArtifacts", "run_platform_demo"]
