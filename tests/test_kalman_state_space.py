from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.dynamics.kalman import (
    build_observation_matrix,
    kalman_predict,
    kalman_update,
)


def test_build_observation_matrix_returns_position_measurement_model() -> None:
    h_matrix = build_observation_matrix()

    assert h_matrix.shape == (2, 4)
    assert np.allclose(h_matrix, np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))


def test_kalman_predict_and_update_shapes_are_correct() -> None:
    predicted_state, predicted_covariance = kalman_predict(
        state=np.array([0.0, 0.0, 1.0, 2.0]),
        covariance=np.eye(4),
        acceleration=np.array([0.5, -0.2]),
        dt=0.1,
        process_noise=np.eye(4) * 0.01,
    )

    updated_state, updated_covariance, kalman_gain = kalman_update(
        predicted_state=predicted_state,
        predicted_covariance=predicted_covariance,
        measurement=np.array([0.11, 0.20]),
        measurement_noise=np.eye(2) * 0.05,
    )

    assert predicted_state.shape == (4,)
    assert predicted_covariance.shape == (4, 4)
    assert updated_state.shape == (4,)
    assert updated_covariance.shape == (4, 4)
    assert kalman_gain.shape == (4, 2)
    assert np.all(np.diag(updated_covariance) <= np.diag(predicted_covariance) + 1e-9)
