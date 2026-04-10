from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.simulation.basic_simulation import (
    BasicSimulationConfig,
    save_position_log,
    simulate_basic_drone_scenario,
)


def test_basic_simulation_logs_consistent_lengths(tmp_path: Path) -> None:
    config = BasicSimulationConfig(steps=20, random_seed=3)

    result = simulate_basic_drone_scenario(config)
    log_path = save_position_log(result, tmp_path / "positions.csv")

    assert result.time_s.shape[0] == result.interceptor_positions.shape[0]
    assert result.time_s.shape[0] == result.target_positions.shape[0]
    assert result.time_s.shape[0] == result.distances_m.shape[0]
    assert result.interceptor_positions.shape[1] == 3
    assert result.target_positions.shape[1] == 3
    assert np.all(result.distances_m >= 0.0)
    assert log_path.exists()


def test_basic_simulation_can_intercept_with_large_radius() -> None:
    config = BasicSimulationConfig(steps=5, intercept_distance_m=1_000.0)

    result = simulate_basic_drone_scenario(config)

    assert result.intercepted is True
    assert result.intercept_step == 0
