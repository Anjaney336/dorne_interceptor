from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.prediction.trajectory import HybridTrajectoryPredictor


def test_hybrid_trajectory_predictor_uses_physics_rollout() -> None:
    predictor = HybridTrajectoryPredictor(dt=0.1, horizon_steps=3)
    past_positions = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [0.2, 0.4],
        ]
    )

    prediction = predictor.predict(past_positions)

    expected = np.array(
        [
            [0.3, 0.6],
            [0.4, 0.8],
            [0.5, 1.0],
        ]
    )
    assert np.allclose(prediction.predicted_positions, expected)
    assert np.allclose(prediction.estimated_velocity, np.array([1.0, 2.0]))
    assert np.allclose(prediction.estimated_acceleration, np.zeros(2))
    assert prediction.backend == "physics"


def test_hybrid_trajectory_predictor_estimates_acceleration() -> None:
    predictor = HybridTrajectoryPredictor(dt=1.0, horizon_steps=2)
    past_positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
        ]
    )

    prediction = predictor.predict(past_positions)

    assert np.allclose(prediction.estimated_velocity, np.array([2.0, 0.0]))
    assert np.allclose(prediction.estimated_acceleration, np.array([1.0, 0.0]))
    assert np.allclose(prediction.predicted_positions[0], np.array([5.5, 0.0]))


def test_hybrid_trajectory_predictor_validates_input_shape() -> None:
    predictor = HybridTrajectoryPredictor(dt=0.1, horizon_steps=2)

    with pytest.raises(ValueError):
        predictor.predict(np.array([[0.0, 1.0, 2.0]]))
