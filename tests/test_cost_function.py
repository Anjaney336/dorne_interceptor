from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.constraints import ConstraintStatus
from drone_interceptor.navigation import simulate_axis_drift, simulate_gps_with_drift
from drone_interceptor.optimization.cost import (
    InterceptionCostModel,
    compute_constraint_penalty,
    compute_interception_cost,
)


def test_simulated_gps_drift_matches_x_fake_equals_x_true_plus_k_t() -> None:
    assert simulate_axis_drift(10.0, 4.0, 0.3) == 11.2
    drifted = simulate_gps_with_drift(np.array([10.0, 5.0, 1.0]), 4.0, 0.3)
    assert np.allclose(drifted, np.array([11.2, 5.0, 1.0]))


def test_constraint_penalty_increases_with_violations() -> None:
    penalty = compute_constraint_penalty(
        ConstraintStatus(
            velocity_clipped=True,
            acceleration_clipped=False,
            tracking_ok=False,
            drift_rate_in_bounds=False,
            safety_override=True,
            distance_to_target_m=2.0,
        )
    )
    assert penalty > 0.0


def test_interception_cost_combines_distance_control_and_penalty_terms() -> None:
    status = ConstraintStatus(
        velocity_clipped=True,
        acceleration_clipped=False,
        tracking_ok=True,
        drift_rate_in_bounds=True,
        safety_override=False,
        distance_to_target_m=10.0,
    )
    cost = compute_interception_cost(
        interceptor_position=np.array([0.0, 0.0]),
        target_position=np.array([3.0, 4.0]),
        control_input=np.array([1.0, 2.0]),
        dt=0.1,
        alpha=0.5,
        beta=10.0,
        constraint_status=status,
    )
    expected = (25.0 + 0.5 * 5.0 + 10.0 * 1.0) * 0.1
    assert np.isclose(cost, expected)


def test_cost_model_stage_cost_uses_configured_weights() -> None:
    model = InterceptionCostModel(alpha=0.2, beta=5.0, dt=0.1)
    cost = model.stage_cost(
        interceptor_position=np.array([0.0, 0.0]),
        target_position=np.array([1.0, 0.0]),
        control_input=np.array([2.0, 0.0]),
        constraint_status=None,
    )
    assert np.isclose(cost, (1.0 + 0.2 * 4.0) * 0.1)
