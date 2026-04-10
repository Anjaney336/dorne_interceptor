from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_trajectory_3d(
    target_positions: np.ndarray,
    interceptor_positions: np.ndarray,
    output_path: str | Path,
    intercept_point: np.ndarray | None = None,
    measured_target_positions: np.ndarray | None = None,
    filtered_target_positions: np.ndarray | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], label="Target", color="#d62728", linewidth=2)
    axis.plot(interceptor_positions[:, 0], interceptor_positions[:, 1], interceptor_positions[:, 2], label="Interceptor", color="#1f77b4", linewidth=2)
    if measured_target_positions is not None and len(measured_target_positions) > 0:
        axis.scatter(
            measured_target_positions[:, 0],
            measured_target_positions[:, 1],
            measured_target_positions[:, 2],
            label="Measured Target",
            color="#ff9896",
            s=12,
            alpha=0.5,
        )
    if filtered_target_positions is not None and len(filtered_target_positions) > 0:
        axis.plot(
            filtered_target_positions[:, 0],
            filtered_target_positions[:, 1],
            filtered_target_positions[:, 2],
            label="Filtered Target",
            color="#2ca02c",
            linewidth=2,
            linestyle="--",
        )
    if intercept_point is not None:
        intercept = np.asarray(intercept_point, dtype=float)
        axis.scatter(intercept[0], intercept[1], intercept[2], label="Intercept", color="#9467bd", marker="x", s=80)
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_zlabel("Z [m]")
    axis.set_title("3D Interception Trajectories")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_mission_dashboard(
    times: np.ndarray,
    target_positions: np.ndarray,
    interceptor_positions: np.ndarray,
    distances: np.ndarray,
    speed_commands: np.ndarray,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(target_positions[:, 0], target_positions[:, 1], label="Target", color="#d62728")
    axes[0, 0].plot(interceptor_positions[:, 0], interceptor_positions[:, 1], label="Interceptor", color="#1f77b4")
    axes[0, 0].set_title("Top-Down Trajectories")
    axes[0, 0].set_xlabel("X [m]")
    axes[0, 0].set_ylabel("Y [m]")
    axes[0, 0].legend()

    axes[0, 1].plot(times, distances, color="#2ca02c")
    axes[0, 1].set_title("Relative Distance")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Distance [m]")

    axes[1, 0].plot(times, target_positions[:, 2], label="Target Z", color="#ff7f0e")
    axes[1, 0].plot(times, interceptor_positions[:, 2], label="Interceptor Z", color="#9467bd")
    axes[1, 0].set_title("Altitude")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Z [m]")
    axes[1, 0].legend()

    axes[1, 1].plot(times, speed_commands, color="#17becf")
    axes[1, 1].set_title("Commanded Speed")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Speed [m/s]")

    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output
