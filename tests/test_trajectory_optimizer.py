from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.config import load_config
from drone_interceptor.optimization.trajectory_optimizer import InterceptionTrajectoryOptimizer


def test_trajectory_optimizer_returns_optimal_path_and_controls() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    optimizer = InterceptionTrajectoryOptimizer(
        config=config,
        horizon_steps=6,
        num_trajectories=10,
        random_seed=3,
    )

    result = optimizer.optimize(
        interceptor_state=np.array([0.0, 0.0, 2.0, 0.0]),
        target_state=np.array([20.0, 5.0, 0.0, 0.0]),
    )

    assert result.optimal_path.shape == (7, 2)
    assert result.optimal_controls.shape == (6, 2)
    assert result.evaluated_trajectories == 10
    assert result.optimal_cost > 0.0


def test_trajectory_optimizer_reduces_terminal_distance_for_best_path() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    optimizer = InterceptionTrajectoryOptimizer(
        config=config,
        horizon_steps=8,
        num_trajectories=12,
        random_seed=5,
    )

    result = optimizer.optimize(
        interceptor_state=np.array([0.0, 0.0, 0.0, 0.0]),
        target_state=np.array([15.0, 0.0, 0.0, 0.0]),
    )

    start_distance = np.linalg.norm(np.array([0.0, 0.0]) - np.array([15.0, 0.0]))
    end_distance = np.linalg.norm(result.optimal_path[-1] - np.array([15.0, 0.0]))
    assert end_distance < start_distance
