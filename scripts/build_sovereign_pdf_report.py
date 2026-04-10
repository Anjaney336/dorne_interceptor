from __future__ import annotations

import json
import math
import re
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUTS_DIR / "sovereign_report"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PDF_PATH = REPORT_DIR / "Sovereign_Interceptor_Full_Dossier.pdf"


@dataclass(frozen=True)
class FigureSpec:
    title: str
    path: Path
    caption: str


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _collect_day_artifact_inventory(outputs_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"day(\d+)", flags=re.IGNORECASE)
    for artifact in outputs_dir.glob("day*"):
        if not artifact.is_file():
            continue
        match = pattern.search(artifact.name)
        if not match:
            continue
        day = int(match.group(1))
        suffix = artifact.suffix.lower()
        category = "other"
        if suffix in {".png", ".jpg", ".jpeg"}:
            category = "image"
        elif suffix in {".mp4", ".avi", ".mov"}:
            category = "video"
        elif suffix in {".json", ".jsonl"}:
            category = "json"
        elif suffix in {".csv"}:
            category = "csv"
        rows.append(
            {
                "day": day,
                "file": artifact.name,
                "category": category,
                "size_mb": float(artifact.stat().st_size / (1024.0 * 1024.0)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["day", "artifacts", "image", "video", "json", "csv", "total_size_mb"])
    summary = frame.pivot_table(index="day", columns="category", values="file", aggfunc="count", fill_value=0).reset_index()
    for col in ["image", "video", "json", "csv", "other"]:
        if col not in summary.columns:
            summary[col] = 0
    size_sum = frame.groupby("day", as_index=False)["size_mb"].sum().rename(columns={"size_mb": "total_size_mb"})
    merged = summary.merge(size_sum, on="day", how="left")
    merged["artifacts"] = merged[["image", "video", "json", "csv", "other"]].sum(axis=1).astype(int)
    return merged.sort_values("day").reset_index(drop=True)


def _extract_notebook_headings(notebooks_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for notebook in sorted(notebooks_dir.glob("day*.ipynb")):
        payload = _read_json(notebook)
        if not payload:
            continue
        day_match = re.search(r"day(\d+)", notebook.name.lower())
        day = int(day_match.group(1)) if day_match else 0
        cells = payload.get("cells", []) if isinstance(payload.get("cells"), list) else []
        headings: list[str] = []
        code_cells = 0
        markdown_cells = 0
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            ctype = cell.get("cell_type")
            if ctype == "code":
                code_cells += 1
            if ctype == "markdown":
                markdown_cells += 1
                source = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
                for line in source.splitlines():
                    s = line.strip()
                    if s.startswith("#"):
                        headings.append(s.lstrip("# ").strip())
                        break
        rows.append(
            {
                "day": day,
                "notebook": notebook.name,
                "code_cells": code_cells,
                "markdown_cells": markdown_cells,
                "headings": " | ".join(headings[:8]),
            }
        )
    return pd.DataFrame(rows).sort_values(["day", "notebook"]).reset_index(drop=True)


def _load_report_data() -> dict[str, Any]:
    run_summary = pd.read_csv(REPORT_DIR / "run_mission_summary.csv") if (REPORT_DIR / "run_mission_summary.csv").exists() else pd.DataFrame()
    per_target = pd.read_csv(REPORT_DIR / "per_target_metrics.csv") if (REPORT_DIR / "per_target_metrics.csv").exists() else pd.DataFrame()
    risk = pd.read_csv(REPORT_DIR / "run_risk_index.csv") if (REPORT_DIR / "run_risk_index.csv").exists() else pd.DataFrame()
    workflow = pd.read_csv(REPORT_DIR / "notebook_workflow_map.csv") if (REPORT_DIR / "notebook_workflow_map.csv").exists() else _extract_notebook_headings(NOTEBOOKS_DIR)
    day_progress = pd.read_csv(REPORT_DIR / "day_progression_metrics.csv") if (REPORT_DIR / "day_progression_metrics.csv").exists() else pd.DataFrame()
    stats = _read_json(REPORT_DIR / "report_stats.json")
    judge10 = pd.read_csv(OUTPUTS_DIR / "judge_sweep_10_runs.csv") if (OUTPUTS_DIR / "judge_sweep_10_runs.csv").exists() else pd.DataFrame()
    judge15 = pd.read_csv(OUTPUTS_DIR / "judge_sweep_15_runs.csv") if (OUTPUTS_DIR / "judge_sweep_15_runs.csv").exists() else pd.DataFrame()
    bench10 = pd.read_csv(OUTPUTS_DIR / "day10_benchmark.csv") if (OUTPUTS_DIR / "day10_benchmark.csv").exists() else pd.DataFrame()
    day_inventory = _collect_day_artifact_inventory(OUTPUTS_DIR)
    return {
        "run_summary": run_summary,
        "per_target": per_target,
        "risk": risk,
        "workflow": workflow,
        "day_progress": day_progress,
        "stats": stats,
        "judge10": judge10,
        "judge15": judge15,
        "bench10": bench10,
        "day_inventory": day_inventory,
        "day5": _read_json(OUTPUTS_DIR / "day5_summary.json"),
        "day6": _read_json(OUTPUTS_DIR / "day6_summary.json"),
        "day7": _read_json(OUTPUTS_DIR / "day7_summary.json"),
        "day9": _read_json(OUTPUTS_DIR / "day9_summary.json"),
        "day10": _read_json(OUTPUTS_DIR / "day10_summary.json"),
        "notes_excerpts": _collect_notes_excerpts(DOCS_DIR),
    }


def _collect_notes_excerpts(docs_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for notes_file in sorted(docs_dir.glob("day*_notes.md")):
        match = re.search(r"day(\d+)", notes_file.name.lower())
        day = int(match.group(1)) if match else 0
        raw = _read_text(notes_file)
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        heading = ""
        for line in lines:
            if line.startswith("#"):
                heading = line.lstrip("# ").strip()
                break
        excerpt_lines: list[str] = []
        for line in lines:
            if line.startswith("#"):
                continue
            excerpt_lines.append(line)
            if len(" ".join(excerpt_lines)) > 350:
                break
        rows.append(
            {
                "day": day,
                "file": notes_file.name,
                "heading": heading,
                "excerpt": " ".join(excerpt_lines)[:420],
            }
        )
    if not rows:
        return pd.DataFrame(columns=["day", "file", "heading", "excerpt"])
    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True)


def _style_text_page(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.patch.set_facecolor("#f9fbff")
    fig.text(0.05, 0.94, title, fontsize=18, fontweight="bold", color="#102a43")
    if subtitle:
        fig.text(0.05, 0.905, subtitle, fontsize=10.5, color="#486581")
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.89, 0.89], color="#9fb3c8", linewidth=1.2))


def _add_wrapped_block(fig: plt.Figure, x: float, y: float, text: str, width: int = 110, fontsize: float = 10.5, line_spacing: float = 0.026) -> float:
    lines = textwrap.wrap(text, width=width, replace_whitespace=False)
    for line in lines:
        fig.text(x, y, line, fontsize=fontsize, color="#243b53")
        y -= line_spacing
    return y


def _write_paragraph_page(pdf: PdfPages, title: str, paragraphs: list[str], bullets: list[str] | None = None, subtitle: str | None = None, page_no: int = 0) -> int:
    fig = plt.figure(figsize=(11.69, 8.27))
    _style_text_page(fig, title=title, subtitle=subtitle)
    y = 0.85
    for para in paragraphs:
        y = _add_wrapped_block(fig, 0.06, y, para, width=140, fontsize=10.5, line_spacing=0.025)
        y -= 0.015
    if bullets:
        y -= 0.005
        for bullet in bullets:
            y = _add_wrapped_block(fig, 0.08, y, f"- {bullet}", width=132, fontsize=10.3, line_spacing=0.024)
            y -= 0.005
    fig.text(0.92, 0.03, f"Page {page_no}", fontsize=9, color="#627d98")
    pdf.savefig(fig, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return page_no + 1


def _write_table_page(pdf: PdfPages, title: str, frame: pd.DataFrame, subtitle: str | None = None, page_no: int = 0, max_rows: int = 16) -> int:
    if frame.empty:
        return _write_paragraph_page(
            pdf,
            title=title,
            subtitle=subtitle,
            paragraphs=["No tabular data available for this section."],
            page_no=page_no,
        )
    chunks = int(math.ceil(len(frame) / max_rows))
    current_page = page_no
    for idx in range(chunks):
        fig = plt.figure(figsize=(11.69, 8.27))
        _style_text_page(
            fig,
            title=title + (f" (Part {idx + 1}/{chunks})" if chunks > 1 else ""),
            subtitle=subtitle,
        )
        ax = fig.add_axes([0.04, 0.10, 0.92, 0.75])
        ax.axis("off")
        chunk = frame.iloc[idx * max_rows : (idx + 1) * max_rows].copy()
        for col in chunk.columns:
            if pd.api.types.is_float_dtype(chunk[col]):
                chunk[col] = chunk[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
        table = ax.table(
            cellText=chunk.values,
            colLabels=chunk.columns.tolist(),
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.4)
        table.scale(1, 1.35)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold", color="#102a43")
                cell.set_facecolor("#d9e2ec")
            else:
                cell.set_facecolor("#f0f4f8" if row % 2 == 0 else "#ffffff")
        fig.text(0.92, 0.03, f"Page {current_page}", fontsize=9, color="#627d98")
        pdf.savefig(fig, dpi=220, bbox_inches="tight")
        plt.close(fig)
        current_page += 1
    return current_page


def _write_figure_page(pdf: PdfPages, title: str, figure_spec: FigureSpec, page_no: int = 0) -> int:
    fig = plt.figure(figsize=(11.69, 8.27))
    _style_text_page(fig, title=title, subtitle=figure_spec.caption)
    ax = fig.add_axes([0.06, 0.12, 0.88, 0.72])
    ax.axis("off")
    if figure_spec.path.exists():
        image = plt.imread(figure_spec.path)
        ax.imshow(image)
    else:
        ax.text(0.5, 0.5, f"Missing figure:\n{figure_spec.path.name}", ha="center", va="center", fontsize=12)
    fig.text(0.92, 0.03, f"Page {page_no}", fontsize=9, color="#627d98")
    pdf.savefig(fig, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return page_no + 1


def _write_code_page(pdf: PdfPages, title: str, code_block: str, source_label: str, page_no: int = 0) -> int:
    fig = plt.figure(figsize=(11.69, 8.27))
    _style_text_page(fig, title=title, subtitle=f"Source: {source_label}")
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.78])
    ax.axis("off")
    wrapped_lines: list[str] = []
    for raw_line in code_block.strip("\n").splitlines():
        if len(raw_line) <= 130:
            wrapped_lines.append(raw_line)
        else:
            wrapped_lines.extend(textwrap.wrap(raw_line, width=130, subsequent_indent="    "))
    rendered = "\n".join(wrapped_lines[:58])
    ax.text(
        0.01,
        0.99,
        rendered,
        va="top",
        ha="left",
        fontsize=8.6,
        family="monospace",
        color="#102a43",
        linespacing=1.2,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", edgecolor="#9fb3c8"),
    )
    fig.text(0.92, 0.03, f"Page {page_no}", fontsize=9, color="#627d98")
    pdf.savefig(fig, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return page_no + 1


def _figure_inventory() -> list[FigureSpec]:
    return [
        FigureSpec("Packet Loss Surface", REPORT_DIR / "fig_01_packet_loss_surface.png", "Packet-loss model surface using PL = 1 - exp(-k * SNR_inv / d^alpha)."),
        FigureSpec("EKF Success vs Noise", REPORT_DIR / "fig_02_ekf_success_vs_noise.png", "Observed EKF success trend under noise stress."),
        FigureSpec("RMSE Distribution", REPORT_DIR / "fig_03_rmse_distribution_spoof_on_off.png", "Per-target RMSE split by spoof mode."),
        FigureSpec("Mahalanobis Probability", REPORT_DIR / "fig_04_mahalanobis_probability_curve.png", "Theoretical and empirical probability decay."),
        FigureSpec("Threat Risk P90", REPORT_DIR / "fig_05_threat_risk_p90_distribution.png", "Distribution of run-level risk P90."),
        FigureSpec("Clean vs Spoof Heatmaps", REPORT_DIR / "fig_06_clean_vs_spoof_heatmap.png", "Trajectory displacement density comparison."),
        FigureSpec("Compute Latency Histogram", REPORT_DIR / "fig_07_compute_latency_histogram.png", "Per-target compute latency distribution."),
        FigureSpec("Energy vs Interception", REPORT_DIR / "fig_08_energy_vs_interception_time.png", "Energy-interception coupling."),
        FigureSpec("Kill Probability Curve", REPORT_DIR / "fig_09_kill_probability_curve.png", "Backend kill-probability law."),
        FigureSpec("Notebook Timeline", REPORT_DIR / "fig_10_notebook_workflow_timeline.png", "Notebook workflow and focus evolution."),
        FigureSpec("Notebook Complexity", REPORT_DIR / "fig_11_notebook_complexity_profile.png", "Code/markdown complexity profile."),
        FigureSpec("Notebook Module Heatmap", REPORT_DIR / "fig_12_notebook_module_heatmap.png", "Module coverage across notebooks."),
        FigureSpec("Day Progression Dashboard", REPORT_DIR / "fig_13_day_progression_dashboard.png", "Day-wise capability progression."),
        FigureSpec("Execution Architecture Flow", REPORT_DIR / "fig_14_execution_architecture_flow.png", "System dataflow from detection to evidence export."),
    ]


def _make_additional_figures(data: dict[str, Any]) -> list[FigureSpec]:
    specs: list[FigureSpec] = []

    day_inventory: pd.DataFrame = data["day_inventory"]
    if not day_inventory.empty:
        path = REPORT_DIR / "fig_15_day_artifact_inventory.png"
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
        axes[0].bar(day_inventory["day"], day_inventory["artifacts"], color="#42a5f5")
        axes[0].set_title("Day-wise Artifact Count")
        axes[0].set_xlabel("Day")
        axes[0].set_ylabel("Artifact count")
        axes[0].grid(axis="y", alpha=0.25)

        width = 0.18
        days = day_inventory["day"].to_numpy(dtype=float)
        for idx, col in enumerate(["image", "video", "json", "csv"]):
            if col not in day_inventory.columns:
                continue
            axes[1].bar(days + (idx - 1.5) * width, day_inventory[col].to_numpy(dtype=float), width=width, label=col)
        axes[1].set_title("Artifact Type Mix by Day")
        axes[1].set_xlabel("Day")
        axes[1].set_ylabel("Count")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        specs.append(FigureSpec("Artifact Production Footprint", path, "Inventory coverage across Day 1 to Day 10 outputs."))

    run_summary: pd.DataFrame = data["run_summary"]
    if not run_summary.empty:
        path = REPORT_DIR / "fig_16_constraints_envelope.png"
        fig, ax = plt.subplots(figsize=(10.8, 5.8))
        scatter = ax.scatter(
            run_summary["drift_rate_mps"],
            run_summary["noise_std_m"],
            c=np.clip(run_summary["ekf_success_rate"], 0.0, 1.0),
            cmap="viridis",
            alpha=0.45,
            s=26,
        )
        ax.set_xlabel("Drift rate (m/s)")
        ax.set_ylabel("Noise std (m)")
        ax.set_title("Operational Constraint Envelope (Drift vs Noise)")
        ax.grid(alpha=0.25)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("EKF success rate")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        specs.append(FigureSpec("Constraint Envelope", path, "Observed operating envelope from run registry missions."))

    per_target: pd.DataFrame = data["per_target"]
    if not per_target.empty:
        path = REPORT_DIR / "fig_17_rmse_success_tradeoff.png"
        fig, ax = plt.subplots(figsize=(10.8, 5.8))
        scatter = ax.scatter(
            per_target["rmse_m"],
            per_target["mission_success_probability"],
            c=np.clip(per_target["compute_latency_ms"], 0.0, 320.0),
            cmap="plasma",
            alpha=0.35,
            s=22,
        )
        ax.set_xlabel("RMSE (m)")
        ax.set_ylabel("Mission success probability")
        ax.set_title("RMSE vs Mission Success (Color: Compute Latency ms)")
        ax.grid(alpha=0.25)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Compute latency (ms)")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        specs.append(FigureSpec("Performance Trade-Off Surface", path, "Per-target success sensitivity against error and latency."))

    judge10: pd.DataFrame = data["judge10"]
    if not judge10.empty:
        path = REPORT_DIR / "fig_18_validation_gate_matrix.png"
        matrix = judge10[["quality_gate", "deploy_gate", "tab_ready"]].astype(float).to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        im = ax.imshow(matrix, cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_yticks(np.arange(len(judge10)))
        ax.set_yticklabels([f'{r["profile"]}-{"ON" if bool(r["spoof_enabled"]) else "OFF"}' for _, r in judge10.iterrows()], fontsize=8.5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["quality_gate", "deploy_gate", "tab_ready"])
        ax.set_title("10-Simulation Judge Sweep Gate Matrix")
        plt.colorbar(im, ax=ax, label="Pass(1)/Fail(0)")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        specs.append(FigureSpec("Validation Gate Matrix", path, "Pass/fail profile across quality, deploy, and tab-readiness gates."))

    bench10: pd.DataFrame = data["bench10"]
    if not bench10.empty:
        path = REPORT_DIR / "fig_19_day10_benchmark_distributions.png"
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))
        axes[0].hist(bench10["tracking_precision_ratio"], bins=8, color="#26a69a", alpha=0.85)
        axes[0].set_title("Tracking precision ratio")
        axes[0].grid(axis="y", alpha=0.25)
        axes[1].hist(bench10["mean_tracking_error_m"], bins=8, color="#ef5350", alpha=0.85)
        axes[1].set_title("Mean tracking error (m)")
        axes[1].grid(axis="y", alpha=0.25)
        axes[2].hist(bench10["safe_zone_distance_improvement_m"], bins=8, color="#42a5f5", alpha=0.85)
        axes[2].set_title("Safe-zone distance improvement (m)")
        axes[2].grid(axis="y", alpha=0.25)
        fig.suptitle("Day10 Benchmark Distribution Summary")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        specs.append(FigureSpec("Day10 Benchmark Statistics", path, "Distribution view for Day10 benchmark runs."))

    return specs


def _extract_code_snippets() -> list[tuple[str, str, str]]:
    ekf = _read_text(PROJECT_ROOT / "src" / "drone_interceptor" / "navigation" / "ekf_filter.py")
    api = _read_text(PROJECT_ROOT / "src" / "drone_interceptor" / "simulation" / "telemetry_api.py")
    manager = _read_text(PROJECT_ROOT / "src" / "drone_interceptor" / "simulation" / "airsim_manager.py")
    spoof = _read_text(PROJECT_ROOT / "src" / "drone_interceptor" / "ros2" / "spoof_manager.py")
    snippets: list[tuple[str, str, str]] = []
    if ekf:
        snippets.append(
            (
                "EKF Core (Prediction + Innovation Gating)",
                "\n".join(
                    line
                    for line in ekf.splitlines()
                    if any(key in line for key in ["def predict", "def assess", "mahalanobis", "chi2", "trust_scale", "adapt_for_tracking_error", "return SpoofingAssessment"])
                )[:6500],
                "src/drone_interceptor/navigation/ekf_filter.py",
            )
        )
    if api:
        snippets.append(
            (
                "FastAPI Mission Endpoint and Per-Target Metrics",
                "\n".join(
                    line
                    for line in api.splitlines()
                    if any(
                        key in line
                        for key in [
                            "@app.post(\"/run_mission\")",
                            "async def run_mission_unified",
                            "mission_results.append",
                            "packet_loss_model",
                            "energy_model",
                            "_compute_target_result_from_replay",
                            "mission_success_probability",
                            "interception_time",
                        ]
                    )
                )[:7000],
                "src/drone_interceptor/simulation/telemetry_api.py",
            )
        )
    if manager:
        snippets.append(
            (
                "Mission Replay Dynamics and Packet-Loss Physics",
                "\n".join(
                    line
                    for line in manager.splitlines()
                    if any(
                        key in line
                        for key in ["_packet_loss_probability_from_link_model", "PL = 1 - exp", "_compute_effective_link_snr_db", "def run_replay", "_augmented_proportional_navigation", "_compute_threat_level"]
                    )
                )[:7000],
                "src/drone_interceptor/simulation/airsim_manager.py",
            )
        )
    if spoof:
        snippets.append(
            (
                "ROS2 Spoof-Manager Safety Interlock and Telemetry",
                "\n".join(
                    line
                    for line in spoof.splitlines()
                    if any(
                        key in line
                        for key in [
                            "class SafetyInterlock",
                            "def power_limit_dbm",
                            "def choose_frequency_hz",
                            "class SpoofManagerCore",
                            "def _spoof_confidence",
                            "def update",
                            "sdr_heatmap",
                            "SPOOFING ACTIVE (DRY-RUN)",
                        ]
                    )
                )[:7000],
                "src/drone_interceptor/ros2/spoof_manager.py",
            )
        )
    return snippets


def build_pdf() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_report_data()
    extra_figures = _make_additional_figures(data)
    stats = data["stats"]
    run_summary: pd.DataFrame = data["run_summary"]
    per_target: pd.DataFrame = data["per_target"]
    workflow: pd.DataFrame = data["workflow"]
    day_progress: pd.DataFrame = data["day_progress"]
    day_inventory: pd.DataFrame = data["day_inventory"]
    judge10: pd.DataFrame = data["judge10"]
    day10_json = data["day10"]
    notes_excerpts: pd.DataFrame = data["notes_excerpts"]

    page_no = 1
    with PdfPages(PDF_PATH) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("#f0f4f8")
        fig.text(0.05, 0.90, "SOVEREIGN INTERCEPTOR", fontsize=34, fontweight="bold", color="#102a43")
        fig.text(0.05, 0.84, "Dual-Layer Autonomous Drone Interceptor", fontsize=20, color="#334e68")
        fig.text(0.05, 0.80, "High-End Technical Dossier (Day 1 to Day 10)", fontsize=14, color="#486581")
        fig.add_artist(plt.Line2D([0.05, 0.95], [0.775, 0.775], color="#829ab1", linewidth=1.5))
        lines = [
            "Prepared for: Honeywell Design-Thon Defense Review",
            "Scope: Full chronology, mathematics, constraints, architecture, validation, and traceability.",
            "Data Basis: Repository-native outputs + run_registry + day summaries + notebooks + backend code.",
            f'Run Registry Missions: {stats.get("run_registry_mission_runs", 0)}',
            f'Per-Target Samples: {stats.get("per_target_samples", 0)}',
            f'EKF Success Mean/P90: {stats.get("ekf_success_rate_mean", 0.0):.4f} / {stats.get("ekf_success_rate_p90", 0.0):.4f}',
            f'RMSE Mean/P90 (m): {stats.get("rmse_mean_m", 0.0):.4f} / {stats.get("rmse_p90_m", 0.0):.4f}',
            f'Output PDF: {PDF_PATH.name}',
        ]
        y = 0.71
        for line in lines:
            fig.text(0.06, y, line, fontsize=12.2, color="#243b53")
            y -= 0.048
        fig.text(0.92, 0.03, f"Page {page_no}", fontsize=9, color="#627d98")
        pdf.savefig(fig, dpi=220, bbox_inches="tight")
        plt.close(fig)
        page_no += 1

        toc_items = [
            "1. Program Context and Day1-Day10 Execution History",
            "2. Unified Architecture and System Dataflow",
            "3. Mathematical Models and Derivations",
            "4. Constraints, Safety Interlocks, and Control Envelope",
            "5. Backend Mission Service and Telemetry Contract",
            "6. Validation Campaign and Judge-Sweep Evidence",
            "7. Figure Atlas (19 curated technical plots)",
            "8. Source Code Excerpts and Formula Traceability",
            "9. Conclusions and Winning-Path Recommendations",
        ]
        page_no = _write_paragraph_page(
            pdf,
            title="Table of Contents",
            subtitle="Sovereign Interceptor Full Dossier",
            paragraphs=["This dossier consolidates Day1-Day10 work into one professional technical report with mathematical and implementation traceability."],
            bullets=toc_items,
            page_no=page_no,
        )

        page_no = _write_paragraph_page(
            pdf,
            title="1. Program Context and Mission Intent",
            subtitle="From Day1 baseline to Day10 benchmark packaging",
            paragraphs=[
                "The objective is a state-first autonomous interceptor stack where all mission analytics are backend-derived and reproducible. The system evolved from day-level detection/tracking prototypes into a multi-target mission engine with spoof-aware telemetry and risk-based deployment gating.",
                "The mission pipeline integrates perception, EKF estimation, guidance/control, spoof-defense analytics, and validation/reporting. Frontend and report outputs are bound to backend mission snapshots and per-target result payloads.",
            ],
            bullets=[
                "Chronology scope: Day 1 through Day 10 artifacts and notebooks",
                "Math scope: EKF, innovation gating, packet-loss physics, guidance dynamics, energy modeling",
                "Validation scope: judge sweep campaigns, benchmark distributions, risk P90",
                "Systems scope: ROS2 spoof manager, Jetson deployment, FastAPI mission services",
            ],
            page_no=page_no,
        )

        day_table = []
        for day in range(1, 11):
            row = day_inventory[day_inventory["day"] == day]
            day_table.append(
                {
                    "day": day,
                    "artifact_count": int(row["artifacts"].iloc[0]) if not row.empty else 0,
                    "artifact_size_mb": float(row["total_size_mb"].iloc[0]) if not row.empty else 0.0,
                    "summary_json": "yes" if (OUTPUTS_DIR / f"day{day}_summary.json").exists() else "no",
                    "notes_md": "yes" if (DOCS_DIR / f"day{day}_notes.md").exists() else "no",
                }
            )
        page_no = _write_table_page(
            pdf,
            title="2. Day1-Day10 Artifact Timeline",
            subtitle="Repository evidence footprint",
            frame=pd.DataFrame(day_table),
            page_no=page_no,
            max_rows=12,
        )

        if not workflow.empty:
            wf_cols = [c for c in ["day", "notebook", "code_cells", "markdown_cells", "primary_focus", "headings_compact", "headings"] if c in workflow.columns]
            page_no = _write_table_page(
                pdf,
                title="3. Notebook Workflow Lineage",
                subtitle="Research-to-engineering progression",
                frame=workflow[wf_cols].copy(),
                page_no=page_no,
                max_rows=9,
            )

        if not notes_excerpts.empty:
            page_no = _write_table_page(
                pdf,
                title="3B. Day Notes Research Excerpts",
                subtitle="Documentation lineage from day notes markdown files",
                frame=notes_excerpts,
                page_no=page_no,
                max_rows=7,
            )

        page_no = _write_paragraph_page(
            pdf,
            title="4. Unified Architecture and Dataflow",
            subtitle="Three-layer system with state-first mission services",
            paragraphs=[
                "Layer 1 is flight-system integration (PX4/ArduPilot patterns, MAVLink status bridge). Layer 2 is AI+vision (YOLOv10-tiny and target-relative telemetry). Layer 3 is spoof-defense analytics (drift planner, interlock, runtime checks).",
                "FastAPI mission endpoints orchestrate preflight, mission run, state streaming, and validation. All scenario tables are generated from mission outputs rather than synthetic frontend placeholders.",
            ],
            bullets=[
                "Endpoints: /preflight, /mission/start, /mission/state, /run_mission, /validate",
                "Mission contract includes per-target EKF success, RMSE, interception time, guidance/latency/energy",
                "Deployment posture includes risk index, quality gate, and telemetry reliability gates",
            ],
            page_no=page_no,
        )

        page_no = _write_paragraph_page(
            pdf,
            title="5. Mathematical Models and Constraints",
            subtitle="Core equations used in backend mission service",
            paragraphs=[
                "RMSE is computed per target from EKF estimate versus ground truth over all frames. EKF success rate is the fraction of frames below adaptive threshold based on noise, drift, packet loss, and kill radius.",
                "Innovation gating uses Mahalanobis distance with chi-square confidence thresholds for soft and hard spoof detection. Packet-loss is modeled as PL = 1 - exp(-k * SNR_inv / d^alpha), then clipped and floor-bounded.",
                "Energy is computed using dynamic power: P = P_hover + c_drag*v^3 + eta*m*a*v, and integrated over time. Mission success probability blends estimation quality, intercept status, distance confidence, and temporal score.",
            ],
            bullets=[
                "Innovation: nu = z - Hx^- ; D_M^2 = nu^T S^-1 nu",
                "Kill kernel: exp(-0.5 * d_M^2)",
                "Adaptive threshold bounds: [0.45, 3.50] m",
                "Packet-loss probability bounds: [0.0, 0.98]",
                "SNR bounds in model: [-20 dB, 45 dB]",
            ],
            page_no=page_no,
        )

        constraints = pd.DataFrame(
            [
                {"constraint": "drift_rate_mps", "reference": "mission payload", "value_or_bound": "typical [0.25, 0.50]"},
                {"constraint": "noise_std_m", "reference": "mission payload", "value_or_bound": "observed [0.35, 1.30]"},
                {"constraint": "packet_loss_probability", "reference": "link model", "value_or_bound": "clipped [0.0, 0.98]"},
                {"constraint": "adaptive_ekf_threshold_m", "reference": "telemetry_api", "value_or_bound": "clipped [0.45, 3.50]"},
                {"constraint": "power_limit_dbm", "reference": "SafetyInterlock", "value_or_bound": "distance-throttled min/max"},
                {"constraint": "frequency_guard_band", "reference": "SafetyInterlock", "value_or_bound": "2.4/5.8 GHz protected"},
                {"constraint": "results_ready_gate", "reference": "dashboard", "value_or_bound": "complete mission + full per-target metrics"},
            ]
        )
        page_no = _write_table_page(
            pdf,
            title="6. Operational Constraints and Safety Gates",
            subtitle="Model and runtime control envelope",
            frame=constraints,
            page_no=page_no,
            max_rows=10,
        )

        if not run_summary.empty:
            summary_table = pd.DataFrame(
                [
                    {"metric": "mission_runs", "value": len(run_summary)},
                    {"metric": "ekf_success_rate_mean", "value": run_summary["ekf_success_rate"].mean()},
                    {"metric": "ekf_success_rate_p90", "value": np.percentile(run_summary["ekf_success_rate"], 90)},
                    {"metric": "rmse_mean_m", "value": run_summary["rmse_m"].mean()},
                    {"metric": "rmse_p90_m", "value": np.percentile(run_summary["rmse_m"], 90)},
                    {"metric": "spoof_enabled_fraction", "value": run_summary["enable_spoofing"].astype(float).mean()},
                ]
            )
            page_no = _write_table_page(
                pdf,
                title="7. Run Registry Quantitative Summary",
                subtitle="Aggregated statistics from sovereign_report datasets",
                frame=summary_table,
                page_no=page_no,
                max_rows=10,
            )

        if not per_target.empty:
            cols = [c for c in ["target_id", "rmse_m", "ekf_success_rate", "mission_success_probability", "compute_latency_ms", "energy_consumption_j", "risk_index_proxy"] if c in per_target.columns]
            page_no = _write_table_page(
                pdf,
                title="8. Per-Target Metrics Snapshot",
                subtitle="Sampled rows from complete per-target metrics dataset",
                frame=per_target[cols].head(20),
                page_no=page_no,
                max_rows=10,
            )

        if not judge10.empty:
            page_no = _write_table_page(
                pdf,
                title="9. Judge Sweep Evidence (10 runs)",
                subtitle="Quality gate / deploy gate / tab readiness status",
                frame=judge10,
                page_no=page_no,
                max_rows=10,
            )

        figs = _figure_inventory() + extra_figures
        for idx, spec in enumerate(figs, start=1):
            page_no = _write_figure_page(pdf, title=f"10.{idx} Figure Atlas", figure_spec=spec, page_no=page_no)

        for idx, (title, snippet, source) in enumerate(_extract_code_snippets(), start=1):
            if snippet.strip():
                page_no = _write_code_page(pdf, title=f"11.{idx} Code Traceability", code_block=snippet, source_label=source, page_no=page_no)

        if not day_progress.empty:
            page_no = _write_table_page(
                pdf,
                title="12. Day-Wise Capability Progression",
                subtitle="Metrics from day summary artifacts and benchmark reports",
                frame=day_progress,
                page_no=page_no,
                max_rows=12,
            )

        day10_b = day10_json.get("benchmark", {}) if isinstance(day10_json, dict) else {}
        page_no = _write_paragraph_page(
            pdf,
            title="13. Conclusions and Professional Recommendations",
            subtitle="Design-thon submission hardening path",
            paragraphs=[
                "The stack now has a complete state-first telemetry foundation, mathematical transparency, and reproducible evidence generation. This dossier demonstrates professional engineering quality with repository-grounded metrics and traceable formulas.",
                "Primary strengths are in architecture completeness, estimator instrumentation, per-target data integrity, and extensive benchmark artifacts. The remaining technical frontier is stress-regime closure under aggressive spoof/noise/packet-loss combinations.",
                f'Day10 benchmark snapshot: redirect_success_rate={day10_b.get("redirect_success_rate", "n/a")}, tracking_precision_ratio={day10_b.get("tracking_precision_ratio", "n/a")}, mean_tracking_error_m={day10_b.get("mean_tracking_error_m", "n/a")}.',
            ],
            bullets=[
                "Run focused stress sweeps and retune EKF covariance schedule/guidance gain",
                "Lock one canonical validation matrix for final judging demonstration",
                "Publish metric-to-formula trace index in frontend (for judge transparency)",
                "Add hardware-in-loop verification traces for final deployment credibility",
            ],
            page_no=page_no,
        )

    return PDF_PATH


def main() -> int:
    # Ensure sovereign base assets exist before PDF assembly.
    asset_builder = PROJECT_ROOT / "scripts" / "generate_sovereign_report_assets.py"
    if asset_builder.exists():
        subprocess.run(
            ["python", str(asset_builder)],
            check=True,
            cwd=str(PROJECT_ROOT),
        )
    output_path = build_pdf()
    print(f"Professional sovereign dossier PDF generated: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
