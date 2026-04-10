from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
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
from drone_interceptor.navigation.drift_model import DP5CoordinateSpoofingToolkit
from drone_interceptor.simulation.scenarios import dp5_scenario_matrix
from drone_interceptor.validation.day7 import _apply_day7_tuning, _build_day7_report, _run_day7_case
from drone_interceptor.visualization.day9 import render_day9_demo_video, render_day9_keyframe


@dataclass(frozen=True)
class Day9Metrics:
    success: bool
    redirected_to_safe_area: bool
    interceptor_peak_speed_mps: float
    interceptor_mean_speed_mps: float
    tracking_precision_ratio: float
    mean_tracking_error_m: float
    mean_drift_rate_mps: float
    min_drift_rate_mps: float
    max_drift_rate_mps: float
    final_safe_zone_distance_m: float
    final_interceptor_distance_m: float
    mean_loop_fps: float


@dataclass(frozen=True)
class Day9Compliance:
    pursuit_speed_spec_met: bool
    tracking_precision_spec_met: bool
    spoofing_gradient_spec_met: bool
    rf_integrity_ready: bool
    vision_model_ready: bool
    simulation_ready: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class Day9Artifacts:
    demo_video: Path
    compatibility_video: Path | None
    hero_image: Path
    spoofing_profile_csv: Path
    benchmark_csv: Path
    summary_json: Path
    log_file: Path


@dataclass(frozen=True)
class Day9BenchmarkSummary:
    redirect_success_rate: float
    mean_tracking_error_m: float
    tracking_precision_ratio: float
    peak_speed_mps: float
    evaluated_runs: int


def run_day9_execution(
    project_root: str | Path,
    random_seed: int = 61,
    max_steps_override: int | None = None,
) -> tuple[Day9Metrics, Day9Compliance, Day9Artifacts]:
    root = Path(project_root).resolve()
    outputs_dir = root / "outputs"
    logs_dir = root / "logs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(root / "configs" / "default.yaml")
    _apply_day9_tuning(config, random_seed=random_seed, max_steps_override=max_steps_override)

    baseline_trace = _run_day7_case(
        config=copy.deepcopy(config),
        drift_mode="directed",
        enable_spoofing=False,
    )
    redirected_trace = _run_day7_case(
        config=copy.deepcopy(config),
        drift_mode="directed",
        enable_spoofing=True,
        baseline_positions=np.asarray(baseline_trace.target_positions, dtype=float),
    )

    true_positions = np.asarray(redirected_trace.target_positions, dtype=float)
    spoofed_positions = np.asarray(redirected_trace.drifted_positions, dtype=float)
    estimated_positions = np.asarray(redirected_trace.target_estimated_positions, dtype=float)
    tracker_positions = np.asarray(redirected_trace.tracker_positions, dtype=float)
    interceptor_positions = np.asarray(redirected_trace.interceptor_positions, dtype=float)
    times = np.asarray(redirected_trace.times, dtype=float)
    drift_rates = np.asarray(redirected_trace.adaptive_rates_mps, dtype=float)
    tracking_errors = (
        np.asarray(redirected_trace.tracker_errors_m, dtype=float)
        if redirected_trace.tracker_errors_m
        else (
            np.linalg.norm(tracker_positions - true_positions, axis=1)
            if len(true_positions) > 0
            else np.zeros(0, dtype=float)
        )
    )

    interceptor_speeds = _compute_path_speeds(interceptor_positions, dt=float(config["mission"]["time_step"]))
    metrics = Day9Metrics(
        success=bool(redirected_trace.intercepted),
        redirected_to_safe_area=bool(redirected_trace.redirection_success),
        interceptor_peak_speed_mps=float(np.max(interceptor_speeds)) if len(interceptor_speeds) > 0 else 0.0,
        interceptor_mean_speed_mps=float(np.mean(interceptor_speeds)) if len(interceptor_speeds) > 0 else 0.0,
        tracking_precision_ratio=float(np.mean(tracking_errors <= 0.5)) if len(tracking_errors) > 0 else 0.0,
        mean_tracking_error_m=float(np.mean(tracking_errors)) if len(tracking_errors) > 0 else 0.0,
        mean_drift_rate_mps=float(np.mean(drift_rates)) if len(drift_rates) > 0 else 0.0,
        min_drift_rate_mps=float(np.min(drift_rates)) if len(drift_rates) > 0 else 0.0,
        max_drift_rate_mps=float(np.max(drift_rates)) if len(drift_rates) > 0 else 0.0,
        final_safe_zone_distance_m=float(redirected_trace.final_safe_zone_distance_m),
        final_interceptor_distance_m=float(redirected_trace.distances_m[-1]) if redirected_trace.distances_m else 0.0,
        mean_loop_fps=float(redirected_trace.mean_loop_fps),
    )
    benchmark_csv, benchmark_summary = _run_day9_benchmark(
        output_path=outputs_dir / "day9_benchmark.csv",
        base_config=config,
    )

    compliance = _build_day9_compliance(config=config, metrics=metrics)
    spoofing_profile_csv = _export_day9_spoofing_profile(
        output_path=outputs_dir / "day9_spoofing_profile.csv",
        true_positions=true_positions,
        interceptor_positions=interceptor_positions,
        dt=float(config["mission"]["time_step"]),
        safe_zone=np.asarray(config["navigation"]["day7_safe_zone_position_m"], dtype=float),
        random_seed=random_seed,
    )
    demo_video = render_day9_demo_video(
        times=times,
        true_positions=true_positions,
        spoofed_positions=spoofed_positions,
        estimated_positions=tracker_positions if len(tracker_positions) == len(true_positions) else estimated_positions,
        interceptor_positions=interceptor_positions,
        safe_zone=np.asarray(config["navigation"]["day7_safe_zone_position_m"], dtype=float),
        drift_rates=drift_rates if len(drift_rates) > 0 else np.zeros(max(len(times), 1), dtype=float),
        tracking_errors=tracking_errors if len(tracking_errors) > 0 else np.zeros(max(len(times), 1), dtype=float),
        output_path=outputs_dir / "day9_dp5_demo.mp4",
        fps=float(config.get("day4", {}).get("demo_fps", 20.44)),
        frame_size=(
            int(config.get("day4", {}).get("video_width", 1280)),
            int(config.get("day4", {}).get("video_height", 720)),
        ),
    )
    hero_image = render_day9_keyframe(
        true_positions=true_positions,
        spoofed_positions=spoofed_positions,
        estimated_positions=tracker_positions if len(tracker_positions) == len(true_positions) else estimated_positions,
        interceptor_positions=interceptor_positions,
        safe_zone=np.asarray(config["navigation"]["day7_safe_zone_position_m"], dtype=float),
        output_path=outputs_dir / "day9_dp5_blueprint.png",
        frame_size=(
            int(config.get("day4", {}).get("video_width", 1280)),
            int(config.get("day4", {}).get("video_height", 720)),
        ),
    )
    compatibility_video = demo_video.with_suffix(".avi")

    summary_payload = {
        "dp5_prompt_alignment": {
            "goal": "Interceptor UAV detects a rogue drone and redirects it toward a safe area through simulated coordinate drifting.",
            "rf_mode": "simulation_only",
            "airsim_enabled": False,
        },
        "metrics": metrics.__dict__,
        "benchmark": benchmark_summary.__dict__,
        "compliance": {
            "pursuit_speed_spec_met": compliance.pursuit_speed_spec_met,
            "tracking_precision_spec_met": compliance.tracking_precision_spec_met,
            "spoofing_gradient_spec_met": compliance.spoofing_gradient_spec_met,
            "rf_integrity_ready": compliance.rf_integrity_ready,
            "vision_model_ready": compliance.vision_model_ready,
            "simulation_ready": compliance.simulation_ready,
            "notes": list(compliance.notes),
        },
        "artifacts": {
            "demo_video": str(demo_video),
            "compatibility_video": str(compatibility_video) if compatibility_video.exists() else None,
            "hero_image": str(hero_image),
            "spoofing_profile_csv": str(spoofing_profile_csv),
            "benchmark_csv": str(benchmark_csv),
        },
    }
    summary_json = outputs_dir / "day9_summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    log_file = logs_dir / "day9.log"
    log_file.write_text(
        "\n".join(
            [
                "Day 9 DP5 Execution Report",
                "==========================",
                "",
                _build_day7_report(
                    mode_summaries=[],
                    metrics=type("Day7MetricsProxy", (), {
                        "success_rate": float(metrics.success),
                        "redirection_success_rate": float(metrics.redirected_to_safe_area),
                        "mean_interception_time_s": 0.0,
                        "mean_deviation_from_baseline_m": float(redirected_trace.deviation_from_baseline_m),
                    })(),
                    artifacts=type("Day7ArtifactsProxy", (), {
                        "trajectory_plot": hero_image,
                        "demo_video": demo_video,
                        "compatibility_video": compatibility_video if compatibility_video.exists() else None,
                        "summary_json": summary_json,
                    })(),
                ),
                "",
                "Compliance Notes:",
                *[f"- {note}" for note in compliance.notes],
            ]
        ),
        encoding="utf-8",
    )

    return metrics, compliance, Day9Artifacts(
        demo_video=demo_video,
        compatibility_video=compatibility_video if compatibility_video.exists() else None,
        hero_image=hero_image,
        spoofing_profile_csv=spoofing_profile_csv,
        benchmark_csv=benchmark_csv,
        summary_json=summary_json,
        log_file=log_file,
    )


def _apply_day9_tuning(config: dict[str, Any], random_seed: int, max_steps_override: int | None) -> None:
    _apply_day7_tuning(config, random_seed=random_seed, max_steps_override=max_steps_override)
    config.setdefault("tracking", {})["measurement_noise"] = 0.12
    config.setdefault("tracking", {})["process_noise"] = 0.05
    config.setdefault("perception", {})["synthetic_measurement_noise_std_m"] = 0.08
    config.setdefault("planning", {})["max_speed_mps"] = 20.0
    config.setdefault("constraints", {}).setdefault("physical", {})["max_velocity_mps"] = 20.0
    config.setdefault("constraints", {}).setdefault("tracking", {})["max_position_error_m"] = 0.5
    config.setdefault("constraints", {}).setdefault("drift", {})["min_rate_mps"] = 0.2
    config.setdefault("constraints", {}).setdefault("drift", {})["max_rate_mps"] = 0.5
    config.setdefault("control", {})["mode"] = "mpc"
    config["control"]["mpc_guidance_blend"] = 0.65
    config.setdefault("perception", {})["confidence_threshold"] = 0.35
    config.setdefault("navigation", {})["day7_safe_zone_position_m"] = [365.0, 182.0, 120.0]
    config["navigation"]["day7_target_response_gain"] = 0.55
    config["navigation"]["day7_target_velocity_gain"] = 1.05
    config["navigation"]["day7_safe_zone_pull_gain"] = 1.8
    config["navigation"]["day7_safe_zone_progress_gain"] = 1.8
    config["navigation"]["day7_drift_near_distance_m"] = 800.0


def _compute_path_speeds(positions: np.ndarray, dt: float) -> np.ndarray:
    if len(positions) < 2:
        return np.zeros(0, dtype=float)
    deltas = np.diff(np.asarray(positions, dtype=float), axis=0)
    return np.linalg.norm(deltas, axis=1) / max(float(dt), 1e-6)


def _build_day9_compliance(config: dict[str, Any], metrics: Day9Metrics) -> Day9Compliance:
    notes: list[str] = []
    pursuit_speed_spec_met = bool(metrics.interceptor_peak_speed_mps <= 20.5)
    if not pursuit_speed_spec_met:
        notes.append("Interceptor exceeded the 20 m/s dash-speed target during the simulated run.")
    tracking_precision_spec_met = bool(metrics.tracking_precision_ratio >= 0.90)
    if not tracking_precision_spec_met:
        notes.append("Tracking stayed outside the +/-0.5 m envelope too often for competition-grade confidence.")
    spoofing_gradient_spec_met = bool(metrics.min_drift_rate_mps >= 0.2 and metrics.max_drift_rate_mps <= 0.5)
    if not spoofing_gradient_spec_met:
        notes.append("Coordinate drift moved outside the required 0.2-0.5 m/s gradient band.")
    notes.append("RF integrity remains a hardware integration item. This repo now stays in simulation-only spoofing mode.")
    notes.append(
        f"YOLO model configured: {config.get('perception', {}).get('model_path', 'n/a')}. Edge-hardware >30 FPS is not validated in this workstation-only run."
    )
    notes.append("SITL-style deliverable is ready: closed-loop redirect simulation, video artifact, spoofing profile CSV, and summary JSON.")
    return Day9Compliance(
        pursuit_speed_spec_met=pursuit_speed_spec_met,
        tracking_precision_spec_met=tracking_precision_spec_met,
        spoofing_gradient_spec_met=spoofing_gradient_spec_met,
        rf_integrity_ready=False,
        vision_model_ready=False,
        simulation_ready=True,
        notes=tuple(notes),
    )


def _export_day9_spoofing_profile(
    output_path: Path,
    true_positions: np.ndarray,
    interceptor_positions: np.ndarray,
    dt: float,
    safe_zone: np.ndarray,
    random_seed: int,
) -> Path:
    toolkit = DP5CoordinateSpoofingToolkit(
        safe_zone_position=np.asarray(safe_zone, dtype=float),
        min_rate_mps=0.2,
        max_rate_mps=0.5,
        noise_std_m=0.0,
        random_seed=random_seed,
    )
    rows = toolkit.export_profile_rows(
        true_positions=true_positions,
        interceptor_positions=interceptor_positions,
        dt=dt,
        mode="directed",
    )
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def _run_day9_benchmark(output_path: Path, base_config: dict[str, Any]) -> tuple[Path, Day9BenchmarkSummary]:
    rows: list[dict[str, Any]] = []
    for scenario in dp5_scenario_matrix():
        for seed in (11, 17, 23):
            config = copy.deepcopy(base_config)
            config.setdefault("simulation", {}).update(scenario.simulation_overrides)
            config.setdefault("navigation", {}).update(scenario.navigation_overrides)
            config.setdefault("system", {})["random_seed"] = int(seed)
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
                    "intercepted": bool(redirected_trace.intercepted),
                    "mean_tracking_error_m": float(np.mean(tracker_errors)) if len(tracker_errors) > 0 else 0.0,
                    "tracking_precision_ratio": float(np.mean(tracker_errors <= 0.5)) if len(tracker_errors) > 0 else 0.0,
                    "peak_speed_mps": float(np.max(speeds)) if len(speeds) > 0 else 0.0,
                    "final_safe_zone_distance_m": float(redirected_trace.final_safe_zone_distance_m),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    summary = Day9BenchmarkSummary(
        redirect_success_rate=float(frame["redirected_to_safe_area"].mean()) if not frame.empty else 0.0,
        mean_tracking_error_m=float(frame["mean_tracking_error_m"].mean()) if not frame.empty else 0.0,
        tracking_precision_ratio=float(frame["tracking_precision_ratio"].mean()) if not frame.empty else 0.0,
        peak_speed_mps=float(frame["peak_speed_mps"].max()) if not frame.empty else 0.0,
        evaluated_runs=int(len(frame)),
    )
    return output_path, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Day 9 DP5 safe simulation execution.")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--random-seed", type=int, default=61)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    metrics, compliance, artifacts = run_day9_execution(
        project_root=args.project_root,
        random_seed=args.random_seed,
        max_steps_override=args.max_steps,
    )
    print("Day 9 DP5 Execution")
    print("===================")
    print(f"- redirected_to_safe_area={metrics.redirected_to_safe_area}")
    print(f"- interceptor_peak_speed_mps={metrics.interceptor_peak_speed_mps:.3f}")
    print(f"- tracking_precision_ratio={metrics.tracking_precision_ratio:.2%}")
    print(f"- mean_drift_rate_mps={metrics.mean_drift_rate_mps:.3f}")
    print(f"- demo_video={artifacts.demo_video}")
    for note in compliance.notes:
        print(f"- note: {note}")
    return 0


__all__ = ["run_day9_execution", "Day9Artifacts", "Day9BenchmarkSummary", "Day9Compliance", "Day9Metrics"]


if __name__ == "__main__":
    raise SystemExit(main())
