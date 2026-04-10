from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"
REPORT_OUTPUT_DIR = DEFAULT_OUTPUTS / "sovereign_report"
RUN_REGISTRY_DIR = DEFAULT_OUTPUTS / "run_registry"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


@dataclass(frozen=True)
class DatasetBundle:
    run_frame: pd.DataFrame
    per_target_frame: pd.DataFrame
    risk_frame: pd.DataFrame


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _adaptive_ekf_success_threshold(
    *,
    noise_std_m: float,
    drift_rate_mps: float,
    packet_loss_rate: float,
    kill_radius_m: float,
) -> float:
    threshold = (
        0.30
        + 0.85 * max(float(noise_std_m), 0.0)
        + 1.10 * max(float(drift_rate_mps), 0.0)
        + 1.50 * float(np.clip(packet_loss_rate, 0.0, 0.5))
        + 0.15 * max(float(kill_radius_m), 0.1)
    )
    return float(np.clip(threshold, 0.45, 3.5))


def _extract_datasets(run_registry_dir: Path) -> DatasetBundle:
    run_rows: list[dict[str, Any]] = []
    per_target_rows: list[dict[str, Any]] = []

    for run_path in sorted(run_registry_dir.glob("run_*.json")):
        payload = _read_json(run_path)
        if not payload:
            continue
        if str(payload.get("kind", "")) != "run_mission":
            continue
        if str(payload.get("status", "")) != "complete":
            continue

        config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
        metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
        validation = payload.get("validation", {}) if isinstance(payload.get("validation"), dict) else {}

        num_targets = int(config.get("num_targets", metrics.get("target_count", 0)) or 0)
        drift_rate_mps = float(config.get("drift_rate_mps", 0.0) or 0.0)
        noise_std_m = float(config.get("noise_std_m", config.get("noise_level_m", 0.0)) or 0.0)
        packet_loss_rate = float(config.get("packet_loss_rate", validation.get("packet_loss_rate", 0.0)) or 0.0)
        kill_radius_m = float(config.get("kill_radius_m", 1.0) or 1.0)
        threshold_m = _adaptive_ekf_success_threshold(
            noise_std_m=noise_std_m,
            drift_rate_mps=drift_rate_mps,
            packet_loss_rate=packet_loss_rate,
            kill_radius_m=kill_radius_m,
        )

        run_row = {
            "run_id": str(payload.get("run_id", run_path.stem)),
            "created_at": payload.get("created_at"),
            "num_targets": num_targets,
            "enable_spoofing": bool(config.get("enable_spoofing", False)),
            "use_ekf": bool(config.get("use_ekf", True)),
            "use_ekf_anti_spoofing": bool(config.get("use_ekf_anti_spoofing", True)),
            "drift_rate_mps": drift_rate_mps,
            "noise_std_m": noise_std_m,
            "packet_loss_rate_configured": packet_loss_rate,
            "packet_loss_rate_observed": float(validation.get("packet_loss_rate_observed", 0.0) or 0.0),
            "packet_loss_rate_effective_mean": float(validation.get("packet_loss_rate_effective_mean", 0.0) or 0.0),
            "ekf_success_rate": float(metrics.get("ekf_success_rate", validation.get("ekf_success_rate", 0.0)) or 0.0),
            "rmse_m": float(metrics.get("rmse_m", validation.get("ekf_mean_miss_distance_m", 0.0)) or 0.0),
            "mission_success": bool(metrics.get("mission_success", False)),
            "mission_duration_s": float(metrics.get("mission_duration_s", 0.0) or 0.0),
            "safe_intercepts": int(metrics.get("safe_intercepts", 0) or 0),
            "adaptive_ekf_success_threshold_m": threshold_m,
        }
        run_rows.append(run_row)

        per_target_summary = validation.get("per_target_summary", [])
        if isinstance(per_target_summary, list):
            for row in per_target_summary:
                if not isinstance(row, dict):
                    continue
                rmse = float(row.get("rmse", row.get("ekf_mean_miss_distance_m", 0.0)) or 0.0)
                mission_success_probability = float(row.get("mission_success_probability", 0.0) or 0.0)
                compute_latency_ms = float(row.get("compute_latency_ms", 0.0) or 0.0)
                spoofing_variance = float(row.get("spoofing_variance", 0.0) or 0.0)
                energy_consumption_j = float(row.get("energy_consumption_j", 0.0) or 0.0)
                rmse_risk = float(np.clip(rmse / 8.0, 0.0, 1.0))
                latency_risk = float(np.clip(compute_latency_ms / 320.0, 0.0, 1.0))
                spoof_risk = float(np.clip(spoofing_variance / 4.0, 0.0, 1.0))
                energy_risk = float(np.clip(energy_consumption_j / 2000.0, 0.0, 1.0))
                success_risk = float(np.clip(1.0 - mission_success_probability, 0.0, 1.0))
                risk_index = float(
                    np.clip(
                        0.35 * success_risk
                        + 0.25 * rmse_risk
                        + 0.15 * latency_risk
                        + 0.15 * spoof_risk
                        + 0.10 * energy_risk,
                        0.0,
                        1.0,
                    )
                )

                per_target_rows.append(
                    {
                        "run_id": run_row["run_id"],
                        "target_id": str(row.get("target", "")),
                        "enable_spoofing": run_row["enable_spoofing"],
                        "num_targets": num_targets,
                        "noise_std_m": noise_std_m,
                        "drift_rate_mps": drift_rate_mps,
                        "packet_loss_rate_configured": packet_loss_rate,
                        "adaptive_ekf_success_threshold_m": threshold_m,
                        "ekf_success_rate": float(row.get("ekf_success_rate", 0.0) or 0.0),
                        "rmse_m": rmse,
                        "interception_time_s": row.get("interception_time_s"),
                        "closest_approach_m": row.get("closest_approach_m"),
                        "intercepted": bool(row.get("intercepted", False)),
                        "guidance_efficiency_mps2": float(row.get("guidance_efficiency_mps2", 0.0) or 0.0),
                        "spoofing_variance": spoofing_variance,
                        "compute_latency_ms": compute_latency_ms,
                        "energy_consumption_j": energy_consumption_j,
                        "mission_success_probability": mission_success_probability,
                        "packet_loss_probability": float(row.get("packet_loss_probability", 0.0) or 0.0),
                        "link_snr_db": float(row.get("link_snr_db", 0.0) or 0.0),
                        "risk_index_proxy": risk_index,
                    }
                )

    run_frame = pd.DataFrame(run_rows)
    per_target_frame = pd.DataFrame(per_target_rows)
    if run_frame.empty:
        risk_frame = pd.DataFrame(columns=["run_id", "risk_index_p90", "risk_index_mean", "targets"])
    elif per_target_frame.empty:
        risk_frame = pd.DataFrame(
            {"run_id": run_frame["run_id"], "risk_index_p90": 0.0, "risk_index_mean": 0.0, "targets": 0}
        )
    else:
        risk_frame = (
            per_target_frame.groupby("run_id")["risk_index_proxy"]
            .agg(
                risk_index_p90=lambda x: float(np.percentile(np.asarray(x, dtype=float), 90)),
                risk_index_mean=lambda x: float(np.mean(np.asarray(x, dtype=float))),
                targets=lambda x: int(len(x)),
            )
            .reset_index()
        )
    return DatasetBundle(run_frame=run_frame, per_target_frame=per_target_frame, risk_frame=risk_frame)


def _plot_packet_loss_surface(output_path: Path, k: float = 0.12, alpha: float = 1.8) -> None:
    snr_db = np.linspace(-10.0, 40.0, 220)
    distance_m = np.linspace(25.0, 500.0, 220)
    snr_grid, distance_grid = np.meshgrid(snr_db, distance_m)
    snr_linear = np.power(10.0, snr_grid / 10.0)
    normalized_snr = 1.0 / np.maximum(snr_linear, 1e-6)
    exponent = float(k) * normalized_snr / np.maximum(np.power(distance_grid, float(alpha)), 1e-6)
    packet_loss = 1.0 - np.exp(-exponent)
    packet_loss = np.clip(packet_loss, 0.0, 0.98)

    plt.figure(figsize=(11, 6))
    contour = plt.contourf(snr_grid, distance_grid, packet_loss, levels=24, cmap="magma")
    cbar = plt.colorbar(contour)
    cbar.set_label("Packet loss probability")
    plt.xlabel("Link SNR (dB)")
    plt.ylabel("Distance to interceptor (m)")
    plt.title("Packet Loss Surface: PL = 1 - exp(-k * SNR_inv / d^alpha), k=0.12, alpha=1.8")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_ekf_success_vs_noise(run_frame: pd.DataFrame, output_path: Path) -> None:
    if run_frame.empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, 6))
    spoof_off = run_frame[run_frame["enable_spoofing"] == False]
    spoof_on = run_frame[run_frame["enable_spoofing"] == True]
    ax.scatter(
        spoof_off["noise_std_m"],
        100.0 * spoof_off["ekf_success_rate"],
        alpha=0.35,
        s=28,
        label="Spoof OFF",
        color="#4caf50",
    )
    ax.scatter(
        spoof_on["noise_std_m"],
        100.0 * spoof_on["ekf_success_rate"],
        alpha=0.35,
        s=28,
        label="Spoof ON",
        color="#f44336",
    )
    if run_frame["noise_std_m"].notna().sum() > 1:
        grouped = run_frame.groupby(pd.cut(run_frame["noise_std_m"], bins=6), observed=False)["ekf_success_rate"].mean().dropna()
        x_vals = np.array([interval.mid for interval in grouped.index], dtype=float)
        y_vals = 100.0 * grouped.to_numpy(dtype=float)
        ax.plot(x_vals, y_vals, color="#00bcd4", linewidth=2.2, label="Binned mean")
    ax.set_xlabel("Noise std (m)")
    ax.set_ylabel("EKF success rate (%)")
    ax.set_title("EKF Success Rate vs Noise Stress (Run Registry)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_rmse_distribution(per_target_frame: pd.DataFrame, output_path: Path) -> None:
    if per_target_frame.empty:
        return
    data_off = per_target_frame.loc[per_target_frame["enable_spoofing"] == False, "rmse_m"].dropna().to_numpy(dtype=float)
    data_on = per_target_frame.loc[per_target_frame["enable_spoofing"] == True, "rmse_m"].dropna().to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    box_data = [data_off if data_off.size else np.array([0.0]), data_on if data_on.size else np.array([0.0])]
    ax.boxplot(box_data, labels=["Spoof OFF", "Spoof ON"], patch_artist=True)
    colors = ["#66bb6a", "#ef5350"]
    for patch, color in zip(ax.artists, colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Per-Target RMSE Distribution by Spoofing Mode")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_mahalanobis_probability(per_target_frame: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    d_m = np.linspace(0.0, 5.0, 300)
    p_theory = np.exp(-0.5 * d_m * d_m)
    ax.plot(d_m, p_theory, linewidth=2.3, color="#00bcd4", label="Theory: exp(-0.5*d_M^2)")

    if not per_target_frame.empty:
        threshold = np.maximum(per_target_frame["adaptive_ekf_success_threshold_m"].to_numpy(dtype=float), 1e-3)
        maha_proxy = np.clip(per_target_frame["rmse_m"].to_numpy(dtype=float) / threshold, 0.0, 8.0)
        probability = np.clip(per_target_frame["mission_success_probability"].to_numpy(dtype=float), 0.0, 1.0)
        ax.scatter(
            maha_proxy,
            probability,
            s=18,
            alpha=0.25,
            color="#ff9800",
            label="Empirical per-target samples",
        )

    ax.set_xlabel("Mahalanobis distance proxy d_M (RMSE / threshold)")
    ax.set_ylabel("Kill / mission success probability")
    ax.set_title("Probability Decay with Mahalanobis Distance")
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_risk_p90_distribution(risk_frame: pd.DataFrame, output_path: Path) -> None:
    if risk_frame.empty:
        return
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    ax.hist(risk_frame["risk_index_p90"].to_numpy(dtype=float), bins=28, color="#7e57c2", alpha=0.8)
    ax.set_xlabel("Threat Risk P90 (proxy)")
    ax.set_ylabel("Run count")
    ax.set_title("Distribution of Threat Risk P90 Across Mission Runs")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_clean_vs_spoof_heatmap(day9_profile_csv: Path, output_path: Path) -> None:
    if not day9_profile_csv.exists():
        return
    frame = pd.read_csv(day9_profile_csv)
    required_cols = {"true_x_m", "true_y_m", "spoofed_x_m", "spoofed_y_m"}
    if not required_cols.issubset(frame.columns):
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), sharex=True, sharey=True)
    hb0 = axes[0].hexbin(frame["true_x_m"], frame["true_y_m"], gridsize=28, cmap="Blues", mincnt=1)
    axes[0].set_title("Clean Trajectory Density")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    plt.colorbar(hb0, ax=axes[0], label="Counts")

    hb1 = axes[1].hexbin(frame["spoofed_x_m"], frame["spoofed_y_m"], gridsize=28, cmap="Reds", mincnt=1)
    axes[1].set_title("Spoofed Trajectory Density")
    axes[1].set_xlabel("X (m)")
    plt.colorbar(hb1, ax=axes[1], label="Counts")
    fig.suptitle("Trajectory Heatmaps: Clean vs Spoofed (Day 9 Profile)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_latency_histogram(per_target_frame: pd.DataFrame, output_path: Path) -> None:
    if per_target_frame.empty:
        return
    latency = per_target_frame["compute_latency_ms"].dropna().to_numpy(dtype=float)
    if latency.size == 0:
        return
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    ax.hist(latency, bins=32, color="#26c6da", alpha=0.85)
    ax.set_xlabel("Compute latency (ms)")
    ax.set_ylabel("Samples")
    ax.set_title("Compute Latency Histogram (Per Target)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_energy_vs_interception(per_target_frame: pd.DataFrame, output_path: Path) -> None:
    if per_target_frame.empty:
        return
    frame = per_target_frame.copy()
    frame = frame[pd.to_numeric(frame["interception_time_s"], errors="coerce").notna()]
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    colors = np.where(frame["intercepted"].astype(bool), "#66bb6a", "#ef5350")
    ax.scatter(
        frame["interception_time_s"].to_numpy(dtype=float),
        frame["energy_consumption_j"].to_numpy(dtype=float),
        c=colors,
        s=26,
        alpha=0.45,
    )
    ax.set_xlabel("Interception time (s)")
    ax.set_ylabel("Energy consumption (J)")
    ax.set_title("Energy vs Interception Time (Per Target)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_kill_probability_curve(output_path: Path) -> None:
    distance = np.linspace(0.0, 80.0, 400)
    probability = np.where(
        distance < 1.0,
        1.0,
        np.where(distance > 50.0, 0.0, 1.0 / (1.0 + np.exp(0.2 * (distance - 10.0)))),
    )
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    ax.plot(distance, probability, linewidth=2.2, color="#42a5f5")
    ax.set_xlabel("Interceptor-target distance (m)")
    ax.set_ylabel("Kill probability")
    ax.set_title("Backend Kill Probability Law")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _extract_notebook_workflow(notebook_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    categories = {
        "detection_vision": ["detection", "vision", "yolo", "detector"],
        "tracking_estimation": ["tracking", "kalman", "ekf", "estimation", "state"],
        "guidance_control": ["guidance", "control", "intercept", "navigation", "pursuit"],
        "spoofing_ew": ["spoof", "drift", "ew", "electronic"],
        "architecture_systems": ["architecture", "node graph", "system", "platform"],
        "simulation_sitl": ["airsim", "sitl", "replay", "simulation"],
        "validation_benchmark": ["validation", "benchmark", "metrics", "deliverables"],
    }
    for notebook_path in sorted(notebook_dir.glob("day*.ipynb")):
        payload = _read_json(notebook_path)
        if not payload:
            continue
        day_match = re.search(r"day(\d+)", notebook_path.name.lower())
        day = int(day_match.group(1)) if day_match else 0
        cells = payload.get("cells", []) if isinstance(payload.get("cells"), list) else []
        markdown_cells = [cell for cell in cells if isinstance(cell, dict) and cell.get("cell_type") == "markdown"]
        code_cells = [cell for cell in cells if isinstance(cell, dict) and cell.get("cell_type") == "code"]

        headings: list[str] = []
        for cell in markdown_cells:
            source = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
            for line in source.splitlines():
                s = line.strip()
                if s.startswith("#"):
                    headings.append(s.lstrip("# ").strip())
                    break
            if len(headings) >= 12:
                break

        text_blob = (notebook_path.name + " " + " ".join(headings)).lower()
        category_scores: dict[str, int] = {}
        for category, keywords in categories.items():
            score = 0
            for keyword in keywords:
                score += int(keyword in text_blob)
            category_scores[category] = score

        focus_sorted = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        top_focus = [name for name, score in focus_sorted if score > 0][:3]
        rows.append(
            {
                "day": day,
                "notebook": notebook_path.name,
                "cell_count": len(cells),
                "markdown_cells": len(markdown_cells),
                "code_cells": len(code_cells),
                "first_heading": headings[0] if headings else "",
                "headings_compact": " | ".join(headings[:6]),
                "primary_focus": ", ".join(top_focus) if top_focus else "general",
                **{key: int(value > 0) for key, value in category_scores.items()},
            }
        )
    return pd.DataFrame(rows).sort_values(["day", "notebook"]).reset_index(drop=True)


def _extract_day_progression(outputs_dir: Path) -> pd.DataFrame:
    def _json(path: Path) -> dict[str, Any]:
        return _read_json(path) if path.exists() else {}

    day5 = _json(outputs_dir / "day5_summary.json")
    day6 = _json(outputs_dir / "day6_summary.json")
    day7 = _json(outputs_dir / "day7_summary.json")
    day9 = _json(outputs_dir / "day9_summary.json")
    day10 = _json(outputs_dir / "day10_summary.json")

    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "day": 5,
            "label": "Day 5",
            "success_rate": float((day5.get("metrics") or {}).get("success_rate", 0.0)),
            "redirect_success_rate": np.nan,
            "tracking_precision_ratio": np.nan,
            "mean_tracking_error_m": np.nan,
            "mean_loop_fps": float((day5.get("metrics") or {}).get("fps", np.nan)),
            "rmse_m": float((day5.get("metrics") or {}).get("rmse_m", np.nan)),
        }
    )
    rows.append(
        {
            "day": 6,
            "label": "Day 6",
            "success_rate": float(1.0 if ((day6.get("metrics") or {}).get("success", False)) else 0.0),
            "redirect_success_rate": np.nan,
            "tracking_precision_ratio": np.nan,
            "mean_tracking_error_m": np.nan,
            "mean_loop_fps": float((day6.get("metrics") or {}).get("mean_loop_fps", np.nan)),
            "rmse_m": np.nan,
        }
    )
    rows.append(
        {
            "day": 7,
            "label": "Day 7",
            "success_rate": float((day7.get("metrics") or {}).get("success_rate", np.nan)),
            "redirect_success_rate": float((day7.get("metrics") or {}).get("redirection_success_rate", np.nan)),
            "tracking_precision_ratio": np.nan,
            "mean_tracking_error_m": np.nan,
            "mean_loop_fps": np.nan,
            "rmse_m": np.nan,
        }
    )
    rows.append(
        {
            "day": 9,
            "label": "Day 9",
            "success_rate": float(1.0 if ((day9.get("metrics") or {}).get("success", False)) else 0.0),
            "redirect_success_rate": float(1.0 if ((day9.get("metrics") or {}).get("redirected_to_safe_area", False)) else 0.0),
            "tracking_precision_ratio": float((day9.get("metrics") or {}).get("tracking_precision_ratio", np.nan)),
            "mean_tracking_error_m": float((day9.get("metrics") or {}).get("mean_tracking_error_m", np.nan)),
            "mean_loop_fps": float((day9.get("metrics") or {}).get("mean_loop_fps", np.nan)),
            "rmse_m": np.nan,
        }
    )
    rows.append(
        {
            "day": 10,
            "label": "Day 10",
            "success_rate": float((day10.get("benchmark") or {}).get("target_redirect_met", np.nan)),
            "redirect_success_rate": float((day10.get("benchmark") or {}).get("redirect_success_rate", np.nan)),
            "tracking_precision_ratio": float((day10.get("benchmark") or {}).get("tracking_precision_ratio", np.nan)),
            "mean_tracking_error_m": float((day10.get("benchmark") or {}).get("mean_tracking_error_m", np.nan)),
            "mean_loop_fps": np.nan,
            "rmse_m": np.nan,
        }
    )
    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True)


def _plot_notebook_timeline(workflow_frame: pd.DataFrame, output_path: Path) -> None:
    if workflow_frame.empty:
        return
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    y_positions = np.arange(len(workflow_frame))
    for idx, row in workflow_frame.iterrows():
        day = float(row["day"])
        ax.hlines(y=idx, xmin=day - 0.35, xmax=day + 0.35, colors="#4fc3f7", linewidth=2.8, alpha=0.8)
        ax.scatter([day], [idx], s=75, color="#ffca28", edgecolors="#37474f", linewidths=0.9, zorder=3)
        label = f'{row["notebook"]}\n{row["primary_focus"]}'
        ax.text(day + 0.42, idx, label, fontsize=8.5, va="center")
    ax.set_yticks([])
    ax.set_xlabel("Project Day")
    ax.set_title("Notebook Workflow Timeline and Focus Evolution")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(min(workflow_frame["day"]) - 0.8, max(workflow_frame["day"]) + 3.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_notebook_complexity(workflow_frame: pd.DataFrame, output_path: Path) -> None:
    if workflow_frame.empty:
        return
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    labels = workflow_frame["notebook"].tolist()
    code_cells = workflow_frame["code_cells"].to_numpy(dtype=float)
    markdown_cells = workflow_frame["markdown_cells"].to_numpy(dtype=float)
    x = np.arange(len(labels))
    ax.bar(x, markdown_cells, label="Markdown cells", color="#26a69a")
    ax.bar(x, code_cells, bottom=markdown_cells, label="Code cells", color="#42a5f5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("Cell count")
    ax.set_title("Notebook Complexity Profile (Markdown + Code)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_notebook_module_heatmap(workflow_frame: pd.DataFrame, output_path: Path) -> None:
    if workflow_frame.empty:
        return
    module_cols = [
        "detection_vision",
        "tracking_estimation",
        "guidance_control",
        "spoofing_ew",
        "architecture_systems",
        "simulation_sitl",
        "validation_benchmark",
    ]
    matrix = workflow_frame[module_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_yticks(np.arange(len(workflow_frame)))
    ax.set_yticklabels(workflow_frame["notebook"].tolist(), fontsize=8.3)
    ax.set_xticks(np.arange(len(module_cols)))
    ax.set_xticklabels(module_cols, rotation=25, ha="right", fontsize=8.6)
    ax.set_title("Notebook-to-Module Coverage Matrix")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Coverage (binary)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_day_progression(day_frame: pd.DataFrame, output_path: Path) -> None:
    if day_frame.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.5))
    day = day_frame["day"].to_numpy(dtype=float)

    axes[0, 0].plot(day, day_frame["success_rate"], marker="o", color="#66bb6a", linewidth=2.0)
    axes[0, 0].plot(day, day_frame["redirect_success_rate"], marker="s", color="#ab47bc", linewidth=1.8)
    axes[0, 0].set_title("Mission / Redirect Success Progression")
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].set_xlabel("Day")
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(day, day_frame["tracking_precision_ratio"], marker="o", color="#42a5f5", linewidth=2.0)
    axes[0, 1].set_title("Tracking Precision Ratio Progression")
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].set_xlabel("Day")
    axes[0, 1].set_ylabel("Ratio")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(day, day_frame["mean_tracking_error_m"], marker="o", color="#ef5350", linewidth=2.0)
    axes[1, 0].set_title("Mean Tracking Error (m)")
    axes[1, 0].set_xlabel("Day")
    axes[1, 0].set_ylabel("Error (m)")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(day, day_frame["mean_loop_fps"], marker="o", color="#26c6da", linewidth=2.0)
    axes[1, 1].set_title("Loop Throughput Progression (FPS)")
    axes[1, 1].set_xlabel("Day")
    axes[1, 1].set_ylabel("FPS")
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle("Notebook-Driven System Progression (Day 5 to Day 10)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_execution_workflow_diagram(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 7.2))
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        rect = plt.Rectangle((x, y), w, h, color=color, alpha=0.18, ec="#263238", lw=1.3)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)

    # Layered blocks
    box(0.04, 0.77, 0.22, 0.14, "Data Layer\nVisDrone + SITL + Run Registry", "#90caf9")
    box(0.30, 0.77, 0.22, 0.14, "Perception Layer\nYOLOv10-tiny / VisionNode", "#81c784")
    box(0.56, 0.77, 0.22, 0.14, "Estimation Layer\nEKF + Innovation Gating", "#ffcc80")
    box(0.80, 0.77, 0.16, 0.14, "Guidance Layer\nAPN + Lead Intercept", "#ce93d8")

    box(0.10, 0.52, 0.26, 0.16, "Spoofing Core (Defensive)\nDrift Planner + Safety Interlock\n(SDR Dry-Run)", "#ef9a9a")
    box(0.42, 0.52, 0.24, 0.16, "Mission Backend\nFastAPI /run_mission\nPer-target metrics", "#80cbc4")
    box(0.72, 0.52, 0.22, 0.16, "Frontend + Judge UX\nOverview / Live Analytics / 3D", "#f48fb1")

    box(0.22, 0.26, 0.22, 0.14, "Validation Engine\nMonte Carlo / Matrix12", "#b0bec5")
    box(0.50, 0.26, 0.22, 0.14, "Evidence Export\nHeatmaps / Reports / Artifacts", "#aed581")
    box(0.76, 0.26, 0.18, 0.14, "Deployment Bridge\nROS2 + MAVLink + Jetson", "#ffab91")

    arrows = [
        ((0.26, 0.84), (0.30, 0.84)),
        ((0.52, 0.84), (0.56, 0.84)),
        ((0.78, 0.84), (0.80, 0.84)),
        ((0.67, 0.77), (0.57, 0.68)),
        ((0.82, 0.77), (0.82, 0.68)),
        ((0.24, 0.77), (0.20, 0.68)),
        ((0.36, 0.60), (0.42, 0.60)),
        ((0.66, 0.60), (0.72, 0.60)),
        ((0.54, 0.52), (0.55, 0.40)),
        ((0.82, 0.52), (0.85, 0.40)),
        ((0.33, 0.52), (0.33, 0.40)),
        ((0.44, 0.33), (0.50, 0.33)),
        ((0.72, 0.33), (0.76, 0.33)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.6, color="#37474f"))

    ax.set_title("Combined Notebook-to-Runtime Execution Architecture", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_report_statistics(bundle: DatasetBundle, output_path: Path) -> None:
    run_frame = bundle.run_frame
    per_target_frame = bundle.per_target_frame
    risk_frame = bundle.risk_frame

    stats = {
        "run_registry_mission_runs": int(len(run_frame)),
        "per_target_samples": int(len(per_target_frame)),
        "targets_range": {
            "min": int(run_frame["num_targets"].min()) if not run_frame.empty else 0,
            "max": int(run_frame["num_targets"].max()) if not run_frame.empty else 0,
        },
        "ekf_success_rate_mean": float(run_frame["ekf_success_rate"].mean()) if not run_frame.empty else 0.0,
        "ekf_success_rate_p90": float(np.percentile(run_frame["ekf_success_rate"], 90)) if not run_frame.empty else 0.0,
        "rmse_mean_m": float(run_frame["rmse_m"].mean()) if not run_frame.empty else 0.0,
        "rmse_p90_m": float(np.percentile(run_frame["rmse_m"], 90)) if not run_frame.empty else 0.0,
        "mission_success_rate": float(run_frame["mission_success"].astype(float).mean()) if not run_frame.empty else 0.0,
        "spoof_enabled_fraction": float(run_frame["enable_spoofing"].astype(float).mean()) if not run_frame.empty else 0.0,
        "mean_compute_latency_ms": float(per_target_frame["compute_latency_ms"].mean()) if not per_target_frame.empty else 0.0,
        "mean_energy_consumption_j": float(per_target_frame["energy_consumption_j"].mean()) if not per_target_frame.empty else 0.0,
        "threat_risk_p90_mean": float(risk_frame["risk_index_p90"].mean()) if not risk_frame.empty else 0.0,
        "threat_risk_p90_p90": float(np.percentile(risk_frame["risk_index_p90"], 90)) if not risk_frame.empty else 0.0,
    }
    output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def generate_assets(
    *,
    outputs_dir: Path = DEFAULT_OUTPUTS,
    out_dir: Path = REPORT_OUTPUT_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_registry_dir = outputs_dir / "run_registry"
    bundle = _extract_datasets(run_registry_dir=run_registry_dir)

    bundle.run_frame.to_csv(out_dir / "run_mission_summary.csv", index=False)
    bundle.per_target_frame.to_csv(out_dir / "per_target_metrics.csv", index=False)
    bundle.risk_frame.to_csv(out_dir / "run_risk_index.csv", index=False)
    _write_report_statistics(bundle, out_dir / "report_stats.json")

    workflow_frame = _extract_notebook_workflow(NOTEBOOK_DIR)
    day_progress_frame = _extract_day_progression(outputs_dir)
    workflow_frame.to_csv(out_dir / "notebook_workflow_map.csv", index=False)
    day_progress_frame.to_csv(out_dir / "day_progression_metrics.csv", index=False)

    _plot_packet_loss_surface(out_dir / "fig_01_packet_loss_surface.png")
    _plot_ekf_success_vs_noise(bundle.run_frame, out_dir / "fig_02_ekf_success_vs_noise.png")
    _plot_rmse_distribution(bundle.per_target_frame, out_dir / "fig_03_rmse_distribution_spoof_on_off.png")
    _plot_mahalanobis_probability(bundle.per_target_frame, out_dir / "fig_04_mahalanobis_probability_curve.png")
    _plot_risk_p90_distribution(bundle.risk_frame, out_dir / "fig_05_threat_risk_p90_distribution.png")
    _plot_clean_vs_spoof_heatmap(outputs_dir / "day9_spoofing_profile.csv", out_dir / "fig_06_clean_vs_spoof_heatmap.png")
    _plot_latency_histogram(bundle.per_target_frame, out_dir / "fig_07_compute_latency_histogram.png")
    _plot_energy_vs_interception(bundle.per_target_frame, out_dir / "fig_08_energy_vs_interception_time.png")
    _plot_kill_probability_curve(out_dir / "fig_09_kill_probability_curve.png")
    _plot_notebook_timeline(workflow_frame, out_dir / "fig_10_notebook_workflow_timeline.png")
    _plot_notebook_complexity(workflow_frame, out_dir / "fig_11_notebook_complexity_profile.png")
    _plot_notebook_module_heatmap(workflow_frame, out_dir / "fig_12_notebook_module_heatmap.png")
    _plot_day_progression(day_progress_frame, out_dir / "fig_13_day_progression_dashboard.png")
    _plot_execution_workflow_diagram(out_dir / "fig_14_execution_architecture_flow.png")


def main() -> int:
    generate_assets()
    print(f"Sovereign report assets generated in: {REPORT_OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
