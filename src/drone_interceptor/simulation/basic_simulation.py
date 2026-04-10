from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class BasicSimulationConfig:
    time_step: float = 0.1
    steps: int = 300
    arena_size_m: float = 1000.0
    interceptor_initial_position: tuple[float, float, float] = (0.0, 0.0, 100.0)
    interceptor_max_speed_mps: float = 55.0
    target_initial_position: tuple[float, float, float] = (300.0, 150.0, 120.0)
    target_initial_velocity: tuple[float, float, float] = (-6.0, 2.0, 0.0)
    target_max_speed_mps: float = 18.0
    target_acceleration_std_mps2: float = 3.0
    intercept_distance_m: float = 10.0
    random_seed: int = 7


@dataclass(frozen=True)
class SimulationResult:
    time_s: np.ndarray
    interceptor_positions: np.ndarray
    target_positions: np.ndarray
    interceptor_velocities: np.ndarray
    target_velocities: np.ndarray
    distances_m: np.ndarray
    intercepted: bool
    intercept_step: int | None


def simulate_basic_drone_scenario(config: BasicSimulationConfig) -> SimulationResult:
    rng = np.random.default_rng(config.random_seed)
    interceptor_position = np.array(config.interceptor_initial_position, dtype=float)
    interceptor_velocity = np.zeros(3, dtype=float)
    target_position = np.array(config.target_initial_position, dtype=float)
    target_velocity = np.array(config.target_initial_velocity, dtype=float)

    interceptor_positions = np.zeros((config.steps + 1, 3), dtype=float)
    target_positions = np.zeros((config.steps + 1, 3), dtype=float)
    interceptor_velocities = np.zeros((config.steps + 1, 3), dtype=float)
    target_velocities = np.zeros((config.steps + 1, 3), dtype=float)
    distances_m = np.zeros(config.steps + 1, dtype=float)

    interceptor_positions[0] = interceptor_position
    target_positions[0] = target_position
    interceptor_velocities[0] = interceptor_velocity
    target_velocities[0] = target_velocity
    distances_m[0] = np.linalg.norm(target_position - interceptor_position)

    intercepted = distances_m[0] <= config.intercept_distance_m
    intercept_step: int | None = 0 if intercepted else None
    last_index = 0
    if intercepted:
        return SimulationResult(
            time_s=np.array([0.0], dtype=float),
            interceptor_positions=interceptor_positions[:1],
            target_positions=target_positions[:1],
            interceptor_velocities=interceptor_velocities[:1],
            target_velocities=target_velocities[:1],
            distances_m=distances_m[:1],
            intercepted=True,
            intercept_step=0,
        )

    for step in range(1, config.steps + 1):
        target_velocity = _update_target_velocity(
            velocity=target_velocity,
            rng=rng,
            dt=config.time_step,
            max_speed=config.target_max_speed_mps,
            acceleration_std=config.target_acceleration_std_mps2,
        )
        target_position = target_position + target_velocity * config.time_step
        target_position, target_velocity = _reflect_from_bounds(
            position=target_position,
            velocity=target_velocity,
            arena_size=config.arena_size_m,
            altitude=target_position[2],
        )

        interceptor_velocity = _compute_interceptor_velocity(
            interceptor_position=interceptor_position,
            target_position=target_position,
            max_speed=config.interceptor_max_speed_mps,
        )
        interceptor_position = interceptor_position + interceptor_velocity * config.time_step

        interceptor_positions[step] = interceptor_position
        target_positions[step] = target_position
        interceptor_velocities[step] = interceptor_velocity
        target_velocities[step] = target_velocity
        distances_m[step] = np.linalg.norm(target_position - interceptor_position)
        last_index = step

        if distances_m[step] <= config.intercept_distance_m:
            intercepted = True
            intercept_step = step
            break

    time_s = np.arange(last_index + 1, dtype=float) * config.time_step
    return SimulationResult(
        time_s=time_s,
        interceptor_positions=interceptor_positions[: last_index + 1],
        target_positions=target_positions[: last_index + 1],
        interceptor_velocities=interceptor_velocities[: last_index + 1],
        target_velocities=target_velocities[: last_index + 1],
        distances_m=distances_m[: last_index + 1],
        intercepted=intercepted,
        intercept_step=intercept_step,
    )


def plot_simulation_trajectories(
    result: SimulationResult,
    output_path: str | Path | None = None,
    show: bool = True,
) -> Path | None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        result.target_positions[:, 0],
        result.target_positions[:, 1],
        label="Target Drone",
        color="tab:red",
        linewidth=2,
    )
    axes[0].plot(
        result.interceptor_positions[:, 0],
        result.interceptor_positions[:, 1],
        label="Interceptor Drone",
        color="tab:blue",
        linewidth=2,
    )
    axes[0].scatter(
        result.target_positions[0, 0],
        result.target_positions[0, 1],
        color="tab:red",
        marker="o",
        s=60,
        label="Target Start",
    )
    axes[0].scatter(
        result.interceptor_positions[0, 0],
        result.interceptor_positions[0, 1],
        color="tab:blue",
        marker="o",
        s=60,
        label="Interceptor Start",
    )
    axes[0].scatter(
        result.target_positions[-1, 0],
        result.target_positions[-1, 1],
        color="tab:red",
        marker="x",
        s=70,
        label="Target End",
    )
    axes[0].scatter(
        result.interceptor_positions[-1, 0],
        result.interceptor_positions[-1, 1],
        color="tab:blue",
        marker="x",
        s=70,
        label="Interceptor End",
    )
    axes[0].set_title("XY Trajectories")
    axes[0].set_xlabel("X Position (m)")
    axes[0].set_ylabel("Y Position (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].axis("equal")

    axes[1].plot(result.time_s, result.distances_m, color="tab:green", linewidth=2)
    axes[1].set_title("Interceptor Distance To Target")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Distance (m)")
    axes[1].grid(True, alpha=0.3)
    if result.intercept_step is not None:
        axes[1].axvline(
            result.time_s[result.intercept_step],
            color="tab:purple",
            linestyle="--",
            linewidth=1.5,
            label="Intercept",
        )
        axes[1].legend()

    figure.tight_layout()

    saved_path: Path | None = None
    if output_path is not None:
        saved_path = Path(output_path).resolve()
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(saved_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(figure)

    return saved_path


def save_position_log(result: SimulationResult, output_path: str | Path) -> Path:
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "time_s,"
        "interceptor_x,interceptor_y,interceptor_z,"
        "target_x,target_y,target_z,"
        "distance_m"
    )
    rows = np.column_stack(
        [
            result.time_s,
            result.interceptor_positions,
            result.target_positions,
            result.distances_m,
        ]
    )
    np.savetxt(destination, rows, delimiter=",", header=header, comments="", fmt="%.6f")
    return destination


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a basic drone interception simulation.")
    parser.add_argument("--steps", type=int, default=300, help="Number of simulation steps.")
    parser.add_argument("--time-step", type=float, default=0.1, help="Simulation time step in seconds.")
    parser.add_argument("--arena-size", type=float, default=1000.0, help="Square arena size in meters.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for target motion.")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("data/basic_simulation_positions.csv"),
        help="CSV output path for logged positions.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("data/basic_simulation_plot.png"),
        help="Image output path for the trajectory plot.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive Matplotlib display.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = BasicSimulationConfig(
        steps=args.steps,
        time_step=args.time_step,
        arena_size_m=args.arena_size,
        random_seed=args.seed,
    )
    result = simulate_basic_drone_scenario(config)
    log_path = save_position_log(result, args.log_path)
    plot_path = plot_simulation_trajectories(result, output_path=args.plot_path, show=not args.no_show)

    print(f"steps_simulated={len(result.time_s) - 1}")
    print(f"intercepted={result.intercepted}")
    print(f"final_distance_m={result.distances_m[-1]:.3f}")
    print(f"log_path={log_path}")
    if plot_path is not None:
        print(f"plot_path={plot_path}")


def _update_target_velocity(
    velocity: np.ndarray,
    rng: np.random.Generator,
    dt: float,
    max_speed: float,
    acceleration_std: float,
) -> np.ndarray:
    acceleration = rng.normal(0.0, acceleration_std, size=2)
    updated_velocity_xy = velocity[:2] + acceleration * dt
    speed = np.linalg.norm(updated_velocity_xy)
    if speed > max_speed and speed > 0.0:
        updated_velocity_xy = (updated_velocity_xy / speed) * max_speed

    updated_velocity = velocity.copy()
    updated_velocity[:2] = updated_velocity_xy
    updated_velocity[2] = 0.0
    return updated_velocity


def _reflect_from_bounds(
    position: np.ndarray,
    velocity: np.ndarray,
    arena_size: float,
    altitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    bounded_position = position.copy()
    bounded_velocity = velocity.copy()

    for index in (0, 1):
        if bounded_position[index] < 0.0:
            bounded_position[index] = -bounded_position[index]
            bounded_velocity[index] *= -1.0
        elif bounded_position[index] > arena_size:
            bounded_position[index] = (2.0 * arena_size) - bounded_position[index]
            bounded_velocity[index] *= -1.0

    bounded_position[2] = altitude
    bounded_velocity[2] = 0.0
    return bounded_position, bounded_velocity


def _compute_interceptor_velocity(
    interceptor_position: np.ndarray,
    target_position: np.ndarray,
    max_speed: float,
) -> np.ndarray:
    direction = target_position - interceptor_position
    distance = np.linalg.norm(direction)
    if distance <= 1e-9:
        return np.zeros(3, dtype=float)
    return (direction / distance) * min(distance, max_speed)


__all__ = [
    "BasicSimulationConfig",
    "SimulationResult",
    "plot_simulation_trajectories",
    "save_position_log",
    "simulate_basic_drone_scenario",
]


if __name__ == "__main__":
    main()
