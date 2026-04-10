from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]

if __package__ in (None, ""):
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from drone_interceptor.config import load_config
from drone_interceptor.constraints import load_constraint_envelope
from drone_interceptor.control.controller import InterceptionController
from drone_interceptor.navigation.drift_model import IntelligentDriftEngine
from drone_interceptor.navigation.state_estimator import GPSIMUKalmanFusion
from drone_interceptor.perception.detector import TargetDetector
from drone_interceptor.planning.planner import InterceptPlanner
from drone_interceptor.prediction.predictor import TargetPredictor
from drone_interceptor.tracking.tracker import TargetTracker
from drone_interceptor.types import ControlCommand, SensorPacket, TargetState
from drone_interceptor.validation.day4 import _apply_day4_tuning
from drone_interceptor.visualization.day7 import plot_day7_spoofing, render_day7_demo_video


@dataclass(frozen=True)
class Day7ModeSummary:
    mode: str
    intercepted: bool
    redirection_success: bool
    interception_time_s: float | None
    deviation_from_baseline_m: float
    final_safe_zone_distance_m: float
    mean_adaptive_rate_mps: float
    log_file: Path


@dataclass(frozen=True)
class Day7Metrics:
    success_rate: float
    redirection_success_rate: float
    mean_interception_time_s: float
    mean_deviation_from_baseline_m: float


@dataclass(frozen=True)
class Day7Artifacts:
    trajectory_plot: Path
    demo_video: Path
    compatibility_video: Path | None
    log_file: Path
    summary_json: Path


@dataclass(slots=True)
class _Day7Trace:
    mode: str
    times: list[float]
    target_positions: list[np.ndarray]
    drifted_positions: list[np.ndarray]
    target_estimated_positions: list[np.ndarray]
    tracker_positions: list[np.ndarray]
    interceptor_positions: list[np.ndarray]
    baseline_positions: list[np.ndarray]
    distances_m: list[float]
    safe_zone_distances_m: list[float]
    adaptive_rates_mps: list[float]
    spoofing_errors_m: list[float]
    tracker_errors_m: list[float]
    intercepted: bool
    interception_time_s: float | None
    deviation_from_baseline_m: float
    redirection_success: bool
    final_safe_zone_distance_m: float
    mean_loop_fps: float
    safe_zone: np.ndarray
    log_lines: list[str]


def run_day7_execution(
    project_root: str | Path,
    random_seed: int = 61,
    max_steps_override: int | None = None,
) -> tuple[list[Day7ModeSummary], Day7Metrics, Day7Artifacts]:
    root = Path(project_root).resolve()
    outputs_dir = root / "outputs"
    logs_dir = root / "logs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_config(root / "configs" / "default.yaml")
    _apply_day7_tuning(base_config, random_seed=random_seed, max_steps_override=max_steps_override)

    baseline_trace = _run_day7_case(
        config=base_config,
        drift_mode="directed",
        enable_spoofing=False,
    )
    traces: list[_Day7Trace] = []
    mode_summaries: list[Day7ModeSummary] = []
    for index, mode in enumerate(("linear", "circular", "directed"), start=1):
        trace = _run_day7_case(
            config=base_config,
            drift_mode=mode,
            enable_spoofing=True,
            baseline_positions=np.asarray(baseline_trace.target_positions, dtype=float),
        )
        traces.append(trace)
        log_file = logs_dir / f"day7_{mode}.log"
        log_file.write_text("\n".join(trace.log_lines), encoding="utf-8")
        mode_summaries.append(
            Day7ModeSummary(
                mode=mode,
                intercepted=trace.intercepted,
                redirection_success=trace.redirection_success,
                interception_time_s=trace.interception_time_s,
                deviation_from_baseline_m=trace.deviation_from_baseline_m,
                final_safe_zone_distance_m=trace.final_safe_zone_distance_m,
                mean_adaptive_rate_mps=float(np.mean(np.asarray(trace.adaptive_rates_mps, dtype=float))),
                log_file=log_file,
            )
        )

    representative = next(trace for trace in traces if trace.mode == "directed")
    trajectory_plot = plot_day7_spoofing(
        target_positions=np.asarray(representative.target_positions, dtype=float),
        drifted_positions=np.asarray(representative.drifted_positions, dtype=float),
        target_estimated_positions=np.asarray(representative.target_estimated_positions, dtype=float),
        interceptor_positions=np.asarray(representative.interceptor_positions, dtype=float),
        baseline_positions=np.asarray(representative.baseline_positions, dtype=float),
        safe_zone=representative.safe_zone,
        output_path=outputs_dir / "day7_spoofing_demo.png",
    )
    demo_video = render_day7_demo_video(
        times=np.asarray(representative.times, dtype=float),
        target_positions=np.asarray(representative.target_positions, dtype=float),
        drifted_positions=np.asarray(representative.drifted_positions, dtype=float),
        target_estimated_positions=np.asarray(representative.target_estimated_positions, dtype=float),
        interceptor_positions=np.asarray(representative.interceptor_positions, dtype=float),
        baseline_positions=np.asarray(representative.baseline_positions, dtype=float),
        distances=np.asarray(representative.distances_m, dtype=float),
        safe_zone_distances=np.asarray(representative.safe_zone_distances_m, dtype=float),
        adaptive_rates=np.asarray(representative.adaptive_rates_mps, dtype=float),
        spoofing_errors=np.asarray(representative.spoofing_errors_m, dtype=float),
        safe_zone=representative.safe_zone,
        output_path=outputs_dir / "day7_demo.mp4",
        fps=float(base_config.get("day4", {}).get("demo_fps", 20.44)),
        frame_size=(
            int(base_config.get("day4", {}).get("video_width", 1280)),
            int(base_config.get("day4", {}).get("video_height", 720)),
        ),
    )
    compatibility_video = demo_video.with_suffix(".avi")

    metrics = Day7Metrics(
        success_rate=_safe_ratio(sum(1 for trace in traces if trace.intercepted), len(traces)),
        redirection_success_rate=_safe_ratio(sum(1 for trace in traces if trace.redirection_success), len(traces)),
        mean_interception_time_s=_mean([trace.interception_time_s for trace in traces if trace.interception_time_s is not None]),
        mean_deviation_from_baseline_m=_mean([trace.deviation_from_baseline_m for trace in traces]),
    )

    summary_json = outputs_dir / "day7_summary.json"
    summary_payload = {
        "modes": [
            {
                "mode": summary.mode,
                "intercepted": summary.intercepted,
                "redirection_success": summary.redirection_success,
                "interception_time_s": summary.interception_time_s,
                "deviation_from_baseline_m": summary.deviation_from_baseline_m,
                "final_safe_zone_distance_m": summary.final_safe_zone_distance_m,
                "mean_adaptive_rate_mps": summary.mean_adaptive_rate_mps,
                "log_file": str(summary.log_file),
            }
            for summary in mode_summaries
        ],
        "metrics": {
            "success_rate": metrics.success_rate,
            "redirection_success_rate": metrics.redirection_success_rate,
            "mean_interception_time_s": metrics.mean_interception_time_s,
            "mean_deviation_from_baseline_m": metrics.mean_deviation_from_baseline_m,
        },
        "artifacts": {
            "trajectory_plot": str(trajectory_plot),
            "demo_video": str(demo_video),
            "compatibility_video": str(compatibility_video) if compatibility_video.exists() else None,
        },
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    final_log = logs_dir / "day7.log"
    final_log.write_text(_build_day7_report(mode_summaries=mode_summaries, metrics=metrics, artifacts=Day7Artifacts(
        trajectory_plot=trajectory_plot,
        demo_video=demo_video,
        compatibility_video=compatibility_video if compatibility_video.exists() else None,
        log_file=final_log,
        summary_json=summary_json,
    )), encoding="utf-8")

    return mode_summaries, metrics, Day7Artifacts(
        trajectory_plot=trajectory_plot,
        demo_video=demo_video,
        compatibility_video=compatibility_video if compatibility_video.exists() else None,
        log_file=final_log,
        summary_json=summary_json,
    )


def _apply_day7_tuning(config: dict[str, Any], random_seed: int, max_steps_override: int | None) -> None:
    config.setdefault("system", {})["random_seed"] = int(random_seed)
    config.setdefault("tracking", {})["mode"] = "kalman"
    config.setdefault("control", {})["mode"] = "mpc"
    _apply_day4_tuning(config)

    mission = config.setdefault("mission", {})
    mission["max_steps"] = int(max_steps_override) if max_steps_override is not None else 180

    simulation = config.setdefault("simulation", {})
    simulation["target_initial_position"] = [270.0, 135.0, 120.0]
    simulation["target_initial_velocity"] = [-5.4, 1.9, 0.0]
    simulation["target_max_acceleration_mps2"] = 3.8
    simulation["target_process_noise_std_mps2"] = 0.16
    simulation["wind_disturbance_std_mps2"] = 0.05
    simulation["interceptor_process_noise_std_mps2"] = 0.10

    planning = config.setdefault("planning", {})
    planning["desired_intercept_distance_m"] = max(float(planning.get("desired_intercept_distance_m", 10.25)), 10.5)

    navigation = config.setdefault("navigation", {})
    navigation["gps_noise_std_m"] = 0.8
    navigation["imu_noise_std_mps2"] = 0.10
    navigation["process_noise_scale"] = 0.75
    navigation["measurement_noise_scale"] = 0.9
    navigation["gps_drift_rate_mps"] = 0.2
    navigation.setdefault("day7_safe_zone_position_m", [360.0, 178.0, 120.0])
    navigation.setdefault("day7_drift_near_distance_m", 125.0)
    navigation.setdefault("day7_spoofing_noise_std_m", 0.08)
    navigation.setdefault("day7_target_response_gain", 0.22)
    navigation.setdefault("day7_target_velocity_gain", 0.55)
    navigation.setdefault("day7_safe_zone_pull_gain", 0.35)
    navigation.setdefault("day7_safe_zone_progress_gain", 0.60)
    navigation.setdefault("day7_circular_frequency_hz", 0.12)


def _run_day7_case(
    config: dict[str, Any],
    drift_mode: str,
    enable_spoofing: bool,
    baseline_positions: np.ndarray | None = None,
) -> _Day7Trace:
    run_config = copy.deepcopy(config)
    dt = float(run_config["mission"]["time_step"])
    max_steps = int(run_config["mission"]["max_steps"])
    intercept_distance_m = float(run_config["planning"]["desired_intercept_distance_m"])
    constraints = load_constraint_envelope(run_config)
    rng = np.random.default_rng(int(run_config.get("system", {}).get("random_seed", 7)))

    detector = TargetDetector(run_config)
    tracker = TargetTracker(run_config)
    predictor = TargetPredictor(run_config)
    planner = InterceptPlanner(run_config)
    controller = InterceptionController(run_config)
    interceptor_navigation = GPSIMUKalmanFusion(run_config)
    target_navigation = GPSIMUKalmanFusion(run_config)

    simulation = run_config["simulation"]
    interceptor_state = TargetState(
        position=np.asarray(simulation["interceptor_initial_position"], dtype=float),
        velocity=np.asarray(simulation["interceptor_initial_velocity"], dtype=float),
        acceleration=np.zeros(3, dtype=float),
    )
    target_state = TargetState(
        position=np.asarray(simulation["target_initial_position"], dtype=float),
        velocity=np.asarray(simulation["target_initial_velocity"], dtype=float),
        acceleration=np.zeros(3, dtype=float),
    )
    target_nominal_velocity = np.asarray(simulation["target_initial_velocity"], dtype=float).copy()
    if float(np.linalg.norm(target_nominal_velocity)) < 1e-6:
        target_nominal_velocity = np.array([-4.0, 1.0, 0.0], dtype=float)
    safe_zone = np.asarray(run_config["navigation"]["day7_safe_zone_position_m"], dtype=float)
    drift_engine = IntelligentDriftEngine(
        min_rate_mps=float(run_config["constraints"]["drift"]["min_rate_mps"]),
        max_rate_mps=float(run_config["constraints"]["drift"]["max_rate_mps"]),
        near_distance_m=float(run_config["navigation"]["day7_drift_near_distance_m"]),
        noise_std_m=float(run_config["navigation"]["day7_spoofing_noise_std_m"]) if enable_spoofing else 0.0,
        safe_zone_position=safe_zone,
        circular_frequency_hz=float(run_config["navigation"]["day7_circular_frequency_hz"]),
        random_seed=int(run_config["system"]["random_seed"]) + (0 if enable_spoofing else 500),
    )

    times: list[float] = []
    target_positions: list[np.ndarray] = []
    drifted_positions: list[np.ndarray] = []
    target_estimated_positions: list[np.ndarray] = []
    tracker_positions: list[np.ndarray] = []
    interceptor_positions: list[np.ndarray] = []
    distances_m: list[float] = []
    safe_zone_distances_m: list[float] = []
    adaptive_rates_mps: list[float] = []
    spoofing_errors_m: list[float] = []
    tracker_errors_m: list[float] = []
    log_lines = [
        f"day7_mode={drift_mode}",
        f"spoofing_enabled={enable_spoofing}",
        f"safe_zone={safe_zone.tolist()}",
        "pipeline=detection -> tracking -> prediction -> control -> adaptive_drift -> simulation",
    ]

    start_time = time.perf_counter()
    interception_time_s: float | None = None
    intercepted = False
    time_s = 0.0

    for step in range(max_steps):
        drift_sample = drift_engine.sample(
            true_position=target_state.position,
            interceptor_position=interceptor_state.position,
            time_s=time_s,
            mode=drift_mode,  # type: ignore[arg-type]
        )
        fake_target_position = drift_sample.fake_position if enable_spoofing else target_state.position.copy()
        target_packet = SensorPacket(
            gps_position=fake_target_position,
            imu_acceleration=target_state.acceleration if target_state.acceleration is not None else np.zeros(3, dtype=float),
            timestamp=time_s,
            true_position=target_state.position.copy(),
            true_velocity=target_state.velocity.copy(),
        )
        target_estimate = target_navigation.update(target_packet)
        target_acceleration = _compute_target_acceleration(
            state=target_state,
            estimated_state=target_estimate.position,
            target_nominal_velocity=target_nominal_velocity,
            safe_zone=safe_zone,
            spoofing_active=enable_spoofing,
            response_gain=float(run_config["navigation"]["day7_target_response_gain"]),
            velocity_gain=float(run_config["navigation"]["day7_target_velocity_gain"]),
            safe_zone_pull_gain=float(run_config["navigation"]["day7_safe_zone_pull_gain"]),
            safe_zone_progress_gain=float(run_config["navigation"]["day7_safe_zone_progress_gain"]),
            max_acceleration=float(simulation["target_max_acceleration_mps2"]),
            rng=rng,
            disturbance_std=float(simulation["target_process_noise_std_mps2"]),
        )

        observation = {
            "target_position": target_state.position.copy(),
            "target_velocity": target_state.velocity.copy(),
            "target_acceleration": target_acceleration.copy(),
            "interceptor_position": interceptor_state.position.copy(),
            "interceptor_velocity": interceptor_state.velocity.copy(),
            "time": np.array([time_s], dtype=float),
        }
        detection = detector.detect(observation)
        track = tracker.update(detection)
        prediction = predictor.predict(track)

        interceptor_packet = _build_interceptor_packet(
            position=interceptor_state.position,
            velocity=interceptor_state.velocity,
            acceleration=interceptor_state.acceleration if interceptor_state.acceleration is not None else np.zeros(3, dtype=float),
            time_s=time_s,
            gps_noise_std=float(run_config["navigation"]["gps_noise_std_m"]),
            imu_noise_std=float(run_config["navigation"]["imu_noise_std_mps2"]),
            rng=rng,
        )
        interceptor_estimate = interceptor_navigation.update(interceptor_packet)
        interceptor_estimated_state = TargetState(
            position=interceptor_estimate.position.copy(),
            velocity=interceptor_estimate.velocity.copy(),
            acceleration=np.asarray(interceptor_state.acceleration if interceptor_state.acceleration is not None else np.zeros(3, dtype=float), dtype=float),
            covariance=None if interceptor_estimate.covariance is None else interceptor_estimate.covariance.copy(),
            timestamp=time_s,
            metadata=dict(interceptor_estimate.metadata),
        )
        plan = planner.plan(interceptor_estimated_state, prediction)
        plan.metadata["current_target_position"] = track.position.copy()
        plan.metadata["current_target_velocity"] = track.velocity.copy()
        plan.metadata["current_target_acceleration"] = (
            track.acceleration.copy() if track.acceleration is not None else np.zeros(3, dtype=float)
        )
        plan.metadata["current_target_covariance"] = (
            None if track.covariance is None else np.asarray(track.covariance, dtype=float).copy()
        )
        plan.metadata["tracking_error_m"] = float(np.linalg.norm(track.position - target_state.position))
        command = controller.compute_command(interceptor_estimated_state, plan)

        interceptor_acceleration = _resolve_command_acceleration(
            command=command,
            current_velocity=interceptor_state.velocity,
            dt=dt,
        )
        interceptor_acceleration += _sample_process_noise(rng=rng, std=float(simulation["interceptor_process_noise_std_mps2"]))
        wind_disturbance = _sample_process_noise(rng=rng, std=float(simulation["wind_disturbance_std_mps2"]))
        interceptor_acceleration = _clip_vector(interceptor_acceleration + 0.4 * wind_disturbance, constraints.max_acceleration_mps2)
        target_acceleration = _clip_vector(target_acceleration + wind_disturbance, float(simulation["target_max_acceleration_mps2"]))

        interceptor_state = _propagate_state(interceptor_state, interceptor_acceleration, dt, constraints.max_velocity_mps)
        target_state = _propagate_state(target_state, target_acceleration, dt, constraints.max_velocity_mps)
        time_s += dt

        distance_m = float(np.linalg.norm(target_state.position - interceptor_state.position))
        safe_zone_distance_m = float(np.linalg.norm(target_state.position - safe_zone))

        times.append(time_s)
        target_positions.append(target_state.position.copy())
        drifted_positions.append(fake_target_position.copy())
        target_estimated_positions.append(target_estimate.position.copy())
        tracker_positions.append(track.position.copy())
        interceptor_positions.append(interceptor_state.position.copy())
        distances_m.append(distance_m)
        safe_zone_distances_m.append(safe_zone_distance_m)
        adaptive_rates_mps.append(float(drift_sample.adaptive_rate_mps) if enable_spoofing else 0.0)
        spoofing_errors_m.append(float(np.linalg.norm(fake_target_position - target_state.position)))
        tracker_errors_m.append(float(np.linalg.norm(track.position - target_state.position)))

        if distance_m <= intercept_distance_m:
            intercepted = True
            interception_time_s = time_s
            break

    mean_loop_fps = float(len(times) / max(time.perf_counter() - start_time, 1e-6)) if times else 0.0
    baseline_reference = (
        baseline_positions[: len(target_positions)]
        if baseline_positions is not None and len(baseline_positions) >= len(target_positions)
        else np.asarray(target_positions, dtype=float)
    )
    target_array = np.asarray(target_positions, dtype=float)
    deviation_from_baseline_m = float(np.mean(np.linalg.norm(target_array - baseline_reference, axis=1))) if len(target_array) > 0 else 0.0
    final_safe_zone_distance_m = float(safe_zone_distances_m[-1]) if safe_zone_distances_m else float(np.linalg.norm(target_state.position - safe_zone))
    baseline_final_distance = (
        float(np.linalg.norm(np.asarray(baseline_positions[-1], dtype=float) - safe_zone))
        if baseline_positions is not None and len(baseline_positions) > 0
        else final_safe_zone_distance_m
    )
    redirection_success = bool(enable_spoofing and final_safe_zone_distance_m + 5.0 < baseline_final_distance)

    log_lines.extend(
        [
            f"intercepted={intercepted}",
            f"interception_time_s={interception_time_s}",
            f"deviation_from_baseline_m={deviation_from_baseline_m:.3f}",
            f"final_safe_zone_distance_m={final_safe_zone_distance_m:.3f}",
            f"redirection_success={redirection_success}",
            f"mean_adaptive_rate_mps={float(np.mean(np.asarray(adaptive_rates_mps, dtype=float))) if adaptive_rates_mps else 0.0:.3f}",
            f"mean_loop_fps={mean_loop_fps:.2f}",
        ]
    )

    return _Day7Trace(
        mode=drift_mode,
        times=times,
        target_positions=target_positions,
        drifted_positions=drifted_positions,
        target_estimated_positions=target_estimated_positions,
        tracker_positions=tracker_positions,
        interceptor_positions=interceptor_positions,
        baseline_positions=[] if baseline_positions is None else [np.asarray(point, dtype=float).copy() for point in baseline_positions[: len(target_positions)]],
        distances_m=distances_m,
        safe_zone_distances_m=safe_zone_distances_m,
        adaptive_rates_mps=adaptive_rates_mps,
        spoofing_errors_m=spoofing_errors_m,
        tracker_errors_m=tracker_errors_m,
        intercepted=intercepted,
        interception_time_s=interception_time_s,
        deviation_from_baseline_m=deviation_from_baseline_m,
        redirection_success=redirection_success,
        final_safe_zone_distance_m=final_safe_zone_distance_m,
        mean_loop_fps=mean_loop_fps,
        safe_zone=safe_zone.copy(),
        log_lines=log_lines,
    )


def _compute_target_acceleration(
    state: TargetState,
    estimated_state: np.ndarray,
    target_nominal_velocity: np.ndarray,
    safe_zone: np.ndarray,
    spoofing_active: bool,
    response_gain: float,
    velocity_gain: float,
    safe_zone_pull_gain: float,
    safe_zone_progress_gain: float,
    max_acceleration: float,
    rng: np.random.Generator,
    disturbance_std: float,
) -> np.ndarray:
    navigation_error = np.asarray(estimated_state, dtype=float) - np.asarray(state.position, dtype=float)
    safe_zone_direction = _normalize(np.asarray(safe_zone, dtype=float) - np.asarray(state.position, dtype=float))
    distance_to_safe_zone = float(np.linalg.norm(np.asarray(safe_zone, dtype=float) - np.asarray(state.position, dtype=float)))
    progress_weight = 1.0 + safe_zone_progress_gain * min(distance_to_safe_zone / 150.0, 1.5)
    nominal_acceleration = velocity_gain * (np.asarray(target_nominal_velocity, dtype=float) - np.asarray(state.velocity, dtype=float))
    safe_zone_pull = (safe_zone_pull_gain * progress_weight) * safe_zone_direction if spoofing_active else np.zeros(3, dtype=float)
    spoofing_acceleration = (response_gain * navigation_error if spoofing_active else np.zeros(3, dtype=float)) + safe_zone_pull
    disturbance = _sample_process_noise(rng=rng, std=disturbance_std)
    return _clip_vector(nominal_acceleration + spoofing_acceleration + disturbance, max_acceleration)


def _build_interceptor_packet(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    time_s: float,
    gps_noise_std: float,
    imu_noise_std: float,
    rng: np.random.Generator,
) -> SensorPacket:
    gps_noise = rng.normal(0.0, gps_noise_std, size=3)
    imu_noise = rng.normal(0.0, imu_noise_std, size=3)
    gps_noise[2] *= 0.35
    imu_noise[2] *= 0.35
    return SensorPacket(
        gps_position=np.asarray(position, dtype=float) + gps_noise,
        imu_acceleration=np.asarray(acceleration, dtype=float) + imu_noise,
        timestamp=float(time_s),
        true_position=np.asarray(position, dtype=float).copy(),
        true_velocity=np.asarray(velocity, dtype=float).copy(),
    )


def _resolve_command_acceleration(command: ControlCommand, current_velocity: np.ndarray, dt: float) -> np.ndarray:
    if command.acceleration_command is not None:
        return np.asarray(command.acceleration_command, dtype=float)
    return (np.asarray(command.velocity_command, dtype=float) - np.asarray(current_velocity, dtype=float)) / max(dt, 1e-6)


def _propagate_state(state: TargetState, acceleration: np.ndarray, dt: float, max_velocity_mps: float) -> TargetState:
    position = np.asarray(state.position, dtype=float) + np.asarray(state.velocity, dtype=float) * dt + 0.5 * np.asarray(acceleration, dtype=float) * (dt**2)
    velocity = np.asarray(state.velocity, dtype=float) + np.asarray(acceleration, dtype=float) * dt
    velocity = _clip_vector(velocity, max_velocity_mps)
    return TargetState(
        position=position,
        velocity=velocity,
        acceleration=np.asarray(acceleration, dtype=float).copy(),
    )


def _clip_vector(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6 or norm <= max_norm:
        return np.asarray(vector, dtype=float)
    return np.asarray(vector, dtype=float) / norm * float(max_norm)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.zeros_like(vector, dtype=float)
    return np.asarray(vector, dtype=float) / norm


def _sample_process_noise(rng: np.random.Generator, std: float) -> np.ndarray:
    if std <= 0.0:
        return np.zeros(3, dtype=float)
    disturbance = rng.normal(0.0, std, size=3)
    disturbance[2] *= 0.35
    return np.asarray(disturbance, dtype=float)


def _mean(values: list[float | None]) -> float:
    finite = [float(value) for value in values if value is not None]
    return float(np.mean(np.asarray(finite, dtype=float))) if finite else 0.0


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _build_day7_report(mode_summaries: list[Day7ModeSummary], metrics: Day7Metrics, artifacts: Day7Artifacts) -> str:
    lines = [
        "Day 7 Intelligent Drift Report",
        "==============================",
        "",
        "Mode Results:",
    ]
    for summary in mode_summaries:
        lines.append(
            (
                f"- {summary.mode}: intercepted={'YES' if summary.intercepted else 'NO'}, "
                f"redirected={'YES' if summary.redirection_success else 'NO'}, "
                f"time={summary.interception_time_s if summary.interception_time_s is not None else 'n/a'}, "
                f"deviation={summary.deviation_from_baseline_m:.3f} m, "
                f"safe_zone_distance={summary.final_safe_zone_distance_m:.3f} m"
            )
        )
    lines.extend(
        [
            "",
            "Aggregate Metrics:",
            f"- success_rate={metrics.success_rate:.2%}",
            f"- redirection_success_rate={metrics.redirection_success_rate:.2%}",
            f"- mean_interception_time_s={metrics.mean_interception_time_s:.3f}",
            f"- mean_deviation_from_baseline_m={metrics.mean_deviation_from_baseline_m:.3f}",
            "",
            "Artifacts:",
            f"- trajectory_plot={artifacts.trajectory_plot}",
            f"- demo_video={artifacts.demo_video}",
            f"- compatibility_video={artifacts.compatibility_video}",
            f"- summary_json={artifacts.summary_json}",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Day 7 intelligent spoofing execution.")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--random-seed", type=int, default=61)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    summaries, metrics, artifacts = run_day7_execution(
        project_root=args.project_root,
        random_seed=args.random_seed,
        max_steps_override=args.max_steps,
    )
    print(_build_day7_report(summaries, metrics, artifacts))
    return 0


__all__ = ["run_day7_execution", "Day7ModeSummary", "Day7Metrics", "Day7Artifacts"]


if __name__ == "__main__":
    raise SystemExit(main())
