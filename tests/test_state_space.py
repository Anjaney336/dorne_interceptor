from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.dynamics.state_space import build_state_matrices, update_state


def test_build_state_matrices_returns_expected_shapes() -> None:
    a_matrix, b_matrix = build_state_matrices(0.1)

    assert a_matrix.shape == (4, 4)
    assert b_matrix.shape == (4, 2)
    assert np.isclose(a_matrix[0, 2], 0.1)
    assert np.isclose(b_matrix[0, 0], 0.005)
    assert np.isclose(b_matrix[2, 0], 0.1)


def test_update_state_matches_discrete_kinematics() -> None:
    next_state = update_state(
        state=np.array([0.0, 0.0, 1.0, 2.0]),
        acceleration=np.array([0.5, -0.2]),
        dt=0.1,
    )

    expected = np.array([0.1025, 0.199, 1.05, 1.98])
    assert np.allclose(next_state, expected)


def test_update_state_rejects_invalid_input_shapes() -> None:
    with pytest.raises(ValueError):
        update_state(state=np.array([0.0, 1.0]), acceleration=np.array([0.0, 0.0]), dt=0.1)

    with pytest.raises(ValueError):
        update_state(state=np.zeros(4), acceleration=np.array([0.0]), dt=0.1)
