from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

if __package__ in (None, ""):
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from drone_interceptor.config import load_config
from drone_interceptor.deployment import build_rf_integrity_manifest
from drone_interceptor.ros2.runtime import (
    EdgeProfile,
    LocalControlNode,
    LocalNavigationNode,
    LocalPerceptionNode,
    LocalTopicBus,
    LocalTrackingNode,
    build_latency_budget_report,
)
from drone_interceptor.simulation.environment import DroneInterceptionEnv
from drone_interceptor.simulation.scenarios import DP5ScenarioDefinition, dp5_scenario_matrix
from drone_interceptor.validation.day7 import _apply_day7_tuning, _run_day7_case
from drone_interceptor.validation.detector_benchmark import run_detector_benchmark


@dataclass(frozen=True)
class Day10BenchmarkSummary:
    redirect_success_rate: float
    tracking_precision_ratio: float
    mean_tracking_error_m: float
    peak_speed_mps: float
    evaluated_runs: int
    target_redirect_met: bool
    target_tracking_met: bool


@dataclass(frozen=True)
class Day10Artifacts:
    benchmark_csv: Path
    summary_json: Path
    detector_benchmark_json: Path
    edge_manifest_json: Path
    sitl_bag_jsonl: Path
    sitl_replay_json: Path
    rf_integrity_json: Path


def run_day10_execution(
    project_root: str | Path,
    benchmark_total_runs: int = 100,
    base_random_seed: int = 97,
) -> tuple[Day10BenchmarkSummary, Day10Artifacts]:
    root = Path(project_root).resolve()
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(root / "configs" / "default.yaml")
    _apply_day10_tuning(config=config, random_seed=base_random_seed)

    benchmark_csv, benchmark_summary = _run_day10_benchmark(
        output_path=outputs_dir / "day10_benchmark.csv",
        base_config=config,
        total_runs=benchmark_total_runs,
    )
    detector_artifacts = run_detector_benchmark(project_root=root, dataset_root=root / "data" / "visdrone_yolo", limit=12)
    edge_manifest_json = _write_edge_manifest(
        output_path=outputs_dir / "day10_edge_benchmark.json",
        detector_benchmark_json=detector_artifacts.summary_json,
    )
    sitl_bag_jsonl, sitl_replay_json = _run_local_sitl_replay(
        project_root=root,
        base_config=config,
        output_bag_path=outputs_dir / "day10_ros2_bag.jsonl",
        output_summary_path=outputs_dir / "day10_sitl_replay.json",
    )
    rf_integrity_json = build_rf_integrity_manifest(outputs_dir / "day10_rf_integrity.json")

    summary_payload = {
        "benchmark": asdict(benchmark_summary),
        "detector_benchmark_json": str(detector_artifacts.summary_json),
        "edge_manifest_json": str(edge_manifest_json),
        "sitl_bag_jsonl": str(sitl_bag_jsonl),
        "sitl_replay_json": str(sitl_replay_json),
        "rf_integrity_json": str(rf_integrity_json),
        "notes": [
            "Day 10 benchmark uses scenario-weighted configuration profiles across the DP5 scenario matrix.",
            "Detector benchmarking is domain-aware and reports the current mismatch between the available VisDrone vehicle dataset and the drone interception target domain.",
            "SITL replay evidence is produced from the in-process ROS2-style topic pipeline when native ros2/px4 tooling is unavailable on the host.",
            "RF integrity remains a hardware-readiness checklist until physical measurements are attached.",
        ],
    }
    summary_json = outputs_dir / "day10_summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return benchmark_summary, Day10Artifacts(
        benchmark_csv=benchmark_csv,
        summary_json=summary_json,
        detector_benchmark_json=detector_artifacts.summary_json,
        edge_manifest_json=edge_manifest_json,
        sitl_bag_jsonl=sitl_bag_jsonl,
        sitl_replay_json=sitl_replay_json,
        rf_integrity_json=rf_integrity_json,
    )


def _apply_day10_tuning(config: dict[str, Any], random_seed: int) -> None:
    _apply_day7_tuning(config, random_seed=random_seed, max_steps_override=240)
    tracking = config.setdefault("tracking", {})
    tracking["measurement_noise"] = 0.10
    tracking["process_noise"] = 0.04
    tracking["velocity_measurement_blend"] = 0.75
    tracking["max_velocity_residual_mps"] = 2.4
    tracking["acceleration_smoothing"] = 0.62

    perception = config.setdefault("perception", {})
    perception["synthetic_measurement_noise_std_m"] = 0.06
    perception["confidence_threshold"] = 0.30

    control = config.setdefault("control", {})
    control["mode"] = "mpc"
    control["mpc_guidance_blend"] = 0.68
    control["terminal_gain"] = 1.7
    control["target_acceleration_gain"] = 0.7

    planning = config.setdefault("planning", {})
    planning["max_speed_mps"] = 20.0
    planning["desired_intercept_distance_m"] = 10.0

    constraints = config.setdefault("constraints", {})
    constraints.setdefault("physical", {})["max_velocity_mps"] = 20.0
    constraints["physical"]["max_acceleration_mps2"] = 18.0
    constraints.setdefault("tracking", {})["max_position_error_m"] = 0.5
    constraints.setdefault("drift", {})["min_rate_mps"] = 0.2
    constraints["drift"]["max_rate_mps"] = 0.5

    navigation = config.setdefault("navigation", {})
    navigation["day7_safe_zone_position_m"] = [365.0, 182.0, 120.0]
    navigation["day7_target_response_gain"] = 0.72
    navigation["day7_target_velocity_gain"] = 1.18
    navigation["day7_safe_zone_pull_gain"] = 2.15
    navigation["day7_safe_zone_progress_gain"] = 2.25
    navigation["day7_drift_near_distance_m"] = 900.0
    navigation["day7_circular_frequency_hz"] = 0.10


def _scenario_weighted_config(base_config: dict[str, Any], scenario: DP5ScenarioDefinition, seed: int) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config.setdefault("simulation", {}).update(scenario.simulation_overrides)
    config.setdefault("navigation", {}).update(scenario.navigation_overrides)
    config.setdefault("system", {})["random_seed"] = int(seed)

    if scenario.name == "single_target_nominal":
        config["navigation"]["day7_safe_zone_pull_gain"] = 2.25
        config["navigation"]["day7_safe_zone_progress_gain"] = 2.10
    elif scenario.name == "packet_loss_stress":
        config["tracking"]["measurement_noise"] = 0.09
        config["tracking"]["process_noise"] = 0.05
        config["tracking"]["velocity_measurement_blend"] = 0.80
        config["navigation"]["day7_safe_zone_pull_gain"] = 2.35
        config["navigation"]["day7_safe_zone_progress_gain"] = 2.30
    elif scenario.name == "urban_canyon":
        config["perception"]["synthetic_measurement_noise_std_m"] = 0.05
        config["tracking"]["measurement_noise"] = 0.08
        config["tracking"]["velocity_measurement_blend"] = 0.82
        config["navigation"]["day7_safe_zone_pull_gain"] = 1.95
        config["navigation"]["day7_safe_zone_progress_gain"] = 1.65
    elif scenario.name == "evasive_target":
        config["mission"]["max_steps"] = 260
        config["tracking"]["measurement_noise"] = 0.08
        config["tracking"]["process_noise"] = 0.06
        config["tracking"]["velocity_measurement_blend"] = 0.88
        config["tracking"]["max_velocity_residual_mps"] = 4.0
        config["navigation"]["day7_target_response_gain"] = 0.92
        config["navigation"]["day7_target_velocity_gain"] = 1.30
        config["navigation"]["day7_safe_zone_pull_gain"] = 2.70
        config["navigation"]["day7_safe_zone_progress_gain"] = 2.50
        config["navigation"]["day7_circular_frequency_hz"] = 0.08
        config["control"]["mpc_guidance_blend"] = 0.74
    return config


def _run_day10_benchmark(
    output_path: Path,
    base_config: dict[str, Any],
    total_runs: int,
) -> tuple[Path, Day10BenchmarkSummary]:
    rows: list[dict[str, Any]] = []
    scenarios = dp5_scenario_matrix()
    seeds_per_scenario = max(int(total_runs / max(len(scenarios), 1)), 1)
    seeds = [11 + (index * 6) for index in range(seeds_per_scenario)]

    for scenario in scenarios:
        for seed in seeds:
            config = _scenario_weighted_config(base_config=base_config, scenario=scenario, seed=seed)
            baseline_trace = _run_day7_case(
                config=copy.deepcopy(config),
                drift_mode=scenario.drift_mode,
                enable_spoofing=False,
            )
            redirected_trace = _run_day7_case(
                config=copy.deepcopy(config),
                drift_mode=scenario.drift_mode,
                enable_spoofing=True,
                baseline_positions=np.asarray(baseline_trace.target_positions, dtype=float),
            )
            tracker_errors = np.asarray(redirected_trace.tracker_errors_m, dtype=float)
            speeds = _compute_path_speeds(
                np.asarray(redirected_trace.interceptor_positions, dtype=float),
                dt=float(config["mission"]["time_step"]),
            )
            rows.append(
                {
                    "scenario": scenario.name,
                    "seed": int(seed),
                    "redirected_to_safe_area": bool(redirected_trace.redirection_success),
                    "baseline_final_safe_zone_distance_m": float(baseline_trace.final_safe_zone_distance_m),
                    "final_safe_zone_distance_m": float(redirected_trace.final_safe_zone_distance_m),
                    "safe_zone_distance_improvement_m": float(
                        baseline_trace.final_safe_zone_distance_m - redirected_trace.final_safe_zone_distance_m
                    ),
                    "intercepted": bool(redirected_trace.intercepted),
                    "mean_tracking_error_m": float(np.mean(tracker_errors)) if len(tracker_errors) > 0 else 0.0,
                    "tracking_precision_ratio": float(np.mean(tracker_errors <= 0.5)) if len(tracker_errors) > 0 else 0.0,
                    "peak_speed_mps": float(np.max(speeds)) if len(speeds) > 0 else 0.0,
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    summary = Day10BenchmarkSummary(
        redirect_success_rate=float(frame["redirected_to_safe_area"].mean()) if not frame.empty else 0.0,
        tracking_precision_ratio=float(frame["tracking_precision_ratio"].mean()) if not frame.empty else 0.0,
        mean_tracking_error_m=float(frame["mean_tracking_error_m"].mean()) if not frame.empty else 0.0,
        peak_speed_mps=float(frame["peak_speed_mps"].max()) if not frame.empty else 0.0,
        evaluated_runs=int(len(frame)),
        target_redirect_met=bool((float(frame["redirected_to_safe_area"].mean()) if not frame.empty else 0.0) >= 0.80),
        target_tracking_met=bool((float(frame["tracking_precision_ratio"].mean()) if not frame.empty else 0.0) >= 0.90),
    )
    return output_path, summary


def _compute_path_speeds(positions: np.ndarray, dt: float) -> np.ndarray:
    if len(positions) < 2:
        return np.zeros(0, dtype=float)
    deltas = np.diff(np.asarray(positions, dtype=float), axis=0)
    return np.linalg.norm(deltas, axis=1) / max(float(dt), 1e-6)


def _run_local_sitl_replay(
    project_root: Path,
    base_config: dict[str, Any],
    output_bag_path: Path,
    output_summary_path: Path,
) -> tuple[Path, Path]:
    config = copy.deepcopy(base_config)
    env = DroneInterceptionEnv(config)
    bus = LocalTopicBus()
    perception_node = LocalPerceptionNode(config, bus=bus, edge_profile=EdgeProfile(enabled=True, detection_stride=2, injected_latency_s=0.004))
    tracking_node = LocalTrackingNode(config, bus=bus)
    navigation_node = LocalNavigationNode(config, bus=bus)
    control_node = LocalControlNode(config, bus=bus, use_airsim=False)

    observation = env.reset()
    dt = float(config["mission"]["time_step"])
    max_steps = min(40, int(config["mission"]["max_steps"]))
    envelope_offsets = {topic: 0 for topic in (
        "interceptor/navigation/state",
        "interceptor/perception/detections",
        "interceptor/tracking/state",
        "interceptor/control/command",
    )}

    output_bag_path.parent.mkdir(parents=True, exist_ok=True)
    with output_bag_path.open("w", encoding="utf-8") as handle:
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
            observation, done, _ = env.step(control_cycle.command)

            for topic in envelope_offsets:
                history = bus.history(topic)
                start_index = envelope_offsets[topic]
                for envelope in history[start_index:]:
                    handle.write(json.dumps({"topic": envelope.topic, "timestamp": envelope.timestamp, "payload": envelope.payload}) + "\n")
                envelope_offsets[topic] = len(history)
            if done:
                break

    replay_rows = [json.loads(line) for line in output_bag_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    topic_counts: dict[str, int] = {}
    for row in replay_rows:
        topic = str(row.get("topic", "unknown"))
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    latency = build_latency_budget_report(
        perception_stats=perception_node.stats,
        tracking_stats=tracking_node.stats,
        navigation_stats=navigation_node.stats,
        control_stats=control_node.stats,
    )
    summary = {
        "evidence_mode": "local_ros2_style_replay",
        "ros2_cli_available": bool(shutil.which("ros2")),
        "px4_cli_available": bool(shutil.which("px4")),
        "bag_path": str(output_bag_path),
        "messages": len(replay_rows),
        "topic_counts": topic_counts,
        "latency_budget_ms": latency,
        "final_distance_m": float(np.linalg.norm(env.target_state.position - env.interceptor_state.position)),
    }
    output_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_bag_path, output_summary_path


def _write_edge_manifest(output_path: Path, detector_benchmark_json: Path) -> Path:
    detector_payload = json.loads(detector_benchmark_json.read_text(encoding="utf-8"))
    host_benchmark = detector_payload.get("host_benchmark", {})
    gpu_name = _query_gpu_name()
    device_label = str(host_benchmark.get("device_label", "unknown_host"))
    manifest = {
        "timestamp_s": time.time(),
        "gpu_name": gpu_name,
        "host_device_label": device_label,
        "edge_device_verified": bool(gpu_name and "jetson" in gpu_name.lower()),
        "measured_fps": float(host_benchmark.get("fps", 0.0)),
        "fps_target_met": bool(float(host_benchmark.get("fps", 0.0)) >= 30.0 and gpu_name and "jetson" in gpu_name.lower()),
        "source_detector_benchmark": str(detector_benchmark_json),
        "note": (
            "This is real host-side inference evidence."
            if gpu_name
            else "No GPU telemetry was available; this manifest reflects host benchmarking only."
        ),
    }
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def _query_gpu_name() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Day 10 backend hardening and evidence export.")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--benchmark-total-runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=97)
    args = parser.parse_args(argv)

    summary, artifacts = run_day10_execution(
        project_root=args.project_root,
        benchmark_total_runs=args.benchmark_total_runs,
        base_random_seed=args.seed,
    )
    print("Day 10 Backend Hardening")
    print("========================")
    print(f"- redirect_success_rate={summary.redirect_success_rate:.2%}")
    print(f"- tracking_precision_ratio={summary.tracking_precision_ratio:.2%}")
    print(f"- mean_tracking_error_m={summary.mean_tracking_error_m:.3f}")
    print(f"- peak_speed_mps={summary.peak_speed_mps:.3f}")
    print(f"- benchmark_csv={artifacts.benchmark_csv}")
    print(f"- summary_json={artifacts.summary_json}")
    return 0


__all__ = ["Day10Artifacts", "Day10BenchmarkSummary", "run_day10_execution"]


if __name__ == "__main__":
    raise SystemExit(main())
