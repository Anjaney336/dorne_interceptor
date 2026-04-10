from __future__ import annotations

import numpy as np

from drone_interceptor.dynamics.state_space import build_state_matrices


def build_observation_matrix() -> np.ndarray:
    """Observation model for position-only measurements Z = H X."""

    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def kalman_predict(
    state: np.ndarray | list[float] | tuple[float, float, float, float],
    covariance: np.ndarray,
    acceleration: np.ndarray | list[float] | tuple[float, float],
    dt: float,
    process_noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the discrete-time Kalman prediction step."""

    state_vector = np.asarray(state, dtype=float).reshape(-1)
    covariance_matrix = np.asarray(covariance, dtype=float)
    control_vector = np.asarray(acceleration, dtype=float).reshape(-1)
    process_noise_matrix = np.asarray(process_noise, dtype=float)

    if state_vector.shape != (4,):
        raise ValueError("state must have shape (4,).")
    if covariance_matrix.shape != (4, 4):
        raise ValueError("covariance must have shape (4, 4).")
    if control_vector.shape != (2,):
        raise ValueError("acceleration must have shape (2,).")
    if process_noise_matrix.shape != (4, 4):
        raise ValueError("process_noise must have shape (4, 4).")

    a_matrix, b_matrix = build_state_matrices(dt)
    predicted_state = a_matrix @ state_vector + b_matrix @ control_vector
    predicted_covariance = a_matrix @ covariance_matrix @ a_matrix.T + process_noise_matrix
    return predicted_state, predicted_covariance


def kalman_update(
    predicted_state: np.ndarray | list[float] | tuple[float, float, float, float],
    predicted_covariance: np.ndarray,
    measurement: np.ndarray | list[float] | tuple[float, float],
    measurement_noise: np.ndarray,
    observation_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the Kalman correction step with position measurements."""

    state_vector = np.asarray(predicted_state, dtype=float).reshape(-1)
    covariance_matrix = np.asarray(predicted_covariance, dtype=float)
    measurement_vector = np.asarray(measurement, dtype=float).reshape(-1)
    measurement_noise_matrix = np.asarray(measurement_noise, dtype=float)
    h_matrix = build_observation_matrix() if observation_matrix is None else np.asarray(observation_matrix, dtype=float)

    if state_vector.shape != (4,):
        raise ValueError("predicted_state must have shape (4,).")
    if covariance_matrix.shape != (4, 4):
        raise ValueError("predicted_covariance must have shape (4, 4).")
    if measurement_vector.shape != (2,):
        raise ValueError("measurement must have shape (2,).")
    if measurement_noise_matrix.shape != (2, 2):
        raise ValueError("measurement_noise must have shape (2, 2).")
    if h_matrix.shape != (2, 4):
        raise ValueError("observation_matrix must have shape (2, 4).")

    innovation = measurement_vector - (h_matrix @ state_vector)
    innovation_covariance = h_matrix @ covariance_matrix @ h_matrix.T + measurement_noise_matrix
    kalman_gain = covariance_matrix @ h_matrix.T @ np.linalg.inv(innovation_covariance)
    updated_state = state_vector + kalman_gain @ innovation
    identity = np.eye(4, dtype=float)
    updated_covariance = (identity - kalman_gain @ h_matrix) @ covariance_matrix
    return updated_state, updated_covariance, kalman_gain
