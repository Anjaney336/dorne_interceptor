from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_platform_3d_trajectory(
    target_positions: np.ndarray,
    interceptor_positions: np.ndarray,
    drifted_positions: np.ndarray,
    fused_positions: np.ndarray,
    output_path: str | Path,
    intercept_point: np.ndarray | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(12, 9))
    axis = figure.add_subplot(1, 1, 1, projection="3d")
    axis.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], linewidth=2.2, color="#d62728", label="Target")
    axis.plot(interceptor_positions[:, 0], interceptor_positions[:, 1], interceptor_positions[:, 2], linewidth=2.2, color="#1f77b4", label="Interceptor")
    axis.plot(drifted_positions[:, 0], drifted_positions[:, 1], drifted_positions[:, 2], linewidth=1.6, linestyle="--", color="#ff7f0e", label="Drifted")
    axis.plot(fused_positions[:, 0], fused_positions[:, 1], fused_positions[:, 2], linewidth=1.8, linestyle=":", color="#2ca02c", label="Fused")
    if intercept_point is not None:
        point = np.asarray(intercept_point, dtype=float)
        axis.scatter(point[0], point[1], point[2], marker="x", s=90, color="#9467bd", label="Intercept")
    axis.set_title("Final 3D Interception Geometry")
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_zlabel("Z [m]")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_distance_vs_time(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    threshold_m: float,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    figure, axis = plt.subplots(figsize=(12, 7))
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
    for index, (name, (times, distances)) in enumerate(series.items()):
        axis.plot(times, distances, linewidth=2.0, color=palette[index % len(palette)], label=name)
    axis.axhline(threshold_m, linestyle="--", linewidth=1.5, color="#444444", label=f"Intercept Threshold ({threshold_m:.2f} m)")
    axis.set_title("Distance vs Time")
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Distance [m]")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_rmse_vs_noise(
    scenario_names: list[str],
    noise_levels: list[float],
    rmses: list[float],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    figure, axis = plt.subplots(figsize=(11, 7))
    axis.scatter(noise_levels, rmses, color="#d62728", s=80)
    for name, x_value, y_value in zip(scenario_names, noise_levels, rmses, strict=False):
        axis.annotate(name, (x_value, y_value), textcoords="offset points", xytext=(6, 6), fontsize=9)
    axis.plot(noise_levels, rmses, color="#ff9896", linewidth=1.5)
    axis.set_title("RMSE vs Noise")
    axis.set_xlabel("Noise Level [relative]")
    axis.set_ylabel("RMSE [m]")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_cost_vs_iteration(
    iterations: np.ndarray,
    costs: np.ndarray,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    figure, axis = plt.subplots(figsize=(11, 7))
    axis.plot(iterations, costs, color="#1f77b4", linewidth=2.2)
    axis.set_title("Cost Function vs Iteration")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Stage Cost")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_fps_vs_model(
    model_labels: list[str],
    fps_values: list[float],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    figure, axis = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    axis.bar(model_labels, fps_values, color=colors[: len(model_labels)])
    axis.set_title("FPS vs Model")
    axis.set_xlabel("Model")
    axis.set_ylabel("FPS")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_success_rate_vs_scenario(
    scenario_names: list[str],
    success_values: list[float],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(scenario_names, success_values, color="#2ca02c")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("Success Rate vs Scenario")
    axis.set_xlabel("Scenario")
    axis.set_ylabel("Success Rate")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


__all__ = [
    "plot_cost_vs_iteration",
    "plot_distance_vs_time",
    "plot_fps_vs_model",
    "plot_platform_3d_trajectory",
    "plot_rmse_vs_noise",
    "plot_success_rate_vs_scenario",
]
