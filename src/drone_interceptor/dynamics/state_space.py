from __future__ import annotations

import numpy as np


def build_state_matrices(dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return discrete-time state and control matrices for planar drone motion.

    State vector:
        X = [x, y, vx, vy]^T

    Control input:
        u = [ax, ay]^T

    Continuous model:
        x_dot = vx
        y_dot = vy
        vx_dot = ax
        vy_dot = ay

    Discrete model:
        X[k+1] = A X[k] + B u[k]
    """

    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    a_matrix = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    b_matrix = np.array(
        [
            [0.5 * dt * dt, 0.0],
            [0.0, 0.5 * dt * dt],
            [dt, 0.0],
            [0.0, dt],
        ],
        dtype=float,
    )
    return a_matrix, b_matrix


def update_state(
    state: np.ndarray | list[float] | tuple[float, float, float, float],
    acceleration: np.ndarray | list[float] | tuple[float, float],
    dt: float,
) -> np.ndarray:
    """Propagate planar drone state by one discrete timestep.

    Args:
        state: [x, y, vx, vy]
        acceleration: [ax, ay]
        dt: timestep in seconds

    Returns:
        Updated state vector [x_next, y_next, vx_next, vy_next].
    """

    state_vector = np.asarray(state, dtype=float).reshape(-1)
    control_vector = np.asarray(acceleration, dtype=float).reshape(-1)

    if state_vector.shape != (4,):
        raise ValueError("state must have shape (4,) representing [x, y, vx, vy].")
    if control_vector.shape != (2,):
        raise ValueError("acceleration must have shape (2,) representing [ax, ay].")

    a_matrix, b_matrix = build_state_matrices(dt)
    next_state = a_matrix @ state_vector + b_matrix @ control_vector
    return next_state
