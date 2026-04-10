from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from drone_interceptor.constraints import ConstraintStatus, load_constraint_envelope
from drone_interceptor.dynamics.state_space import build_state_matrices
from drone_interceptor.dynamics.state_space import update_state
from drone_interceptor.optimization.cost import InterceptionCostModel


@dataclass(slots=True)
class TrajectoryCandidate:
    controls: np.ndarray
    interceptor_states: np.ndarray
    target_states: np.ndarray
    target_covariances: np.ndarray
    total_cost: float


@dataclass(slots=True)
class OptimizationResult:
    optimal_path: np.ndarray
    optimal_controls: np.ndarray
    optimal_cost: float
    best_candidate_index: int
    evaluated_trajectories: int
    candidate_costs: np.ndarray
    best_cost_history: np.ndarray


class InterceptionTrajectoryOptimizer:
    """Sampling-based trajectory optimizer for minimum-cost interception."""

    def __init__(
        self,
        config: dict[str, Any],
        horizon_steps: int | None = None,
        num_trajectories: int | None = None,
        random_seed: int | None = None,
    ) -> None:
        mission = config["mission"]
        optimization = config.get("optimization", {})
        self._dt = float(mission["time_step"])
        self._horizon_steps = int(horizon_steps or optimization.get("horizon_steps", 12))
        self._num_trajectories = int(num_trajectories or optimization.get("num_trajectories", 25))
        self._cost_model = InterceptionCostModel.from_config(config)
        self._constraints = load_constraint_envelope(config)
        seed = random_seed if random_seed is not None else int(config.get("system", {}).get("random_seed", 7))
        self._rng = np.random.default_rng(seed)

    def optimize(
        self,
        interceptor_state: np.ndarray | list[float] | tuple[float, float, float, float],
        target_state: np.ndarray | list[float] | tuple[float, float, float, float],
        target_acceleration: np.ndarray | list[float] | tuple[float, float] | None = None,
        target_covariance: np.ndarray | None = None,
    ) -> OptimizationResult:
        interceptor_initial = np.asarray(interceptor_state, dtype=float).reshape(-1)
        target_initial = np.asarray(target_state, dtype=float).reshape(-1)
        target_control = (
            np.zeros(2, dtype=float)
            if target_acceleration is None
            else np.asarray(target_acceleration, dtype=float).reshape(-1)
        )

        if interceptor_initial.shape != (4,):
            raise ValueError("interceptor_state must have shape (4,) as [x, y, vx, vy].")
        if target_initial.shape != (4,):
            raise ValueError("target_state must have shape (4,) as [x, y, vx, vy].")
        if target_control.shape != (2,):
            raise ValueError("target_acceleration must have shape (2,).")
        if target_covariance is not None and np.asarray(target_covariance, dtype=float).shape != (4, 4):
            raise ValueError("target_covariance must have shape (4, 4).")

        candidates = self._generate_control_candidates(
            interceptor_initial=interceptor_initial,
            target_initial=target_initial,
        )
        best_candidate: TrajectoryCandidate | None = None
        best_index = -1
        candidate_costs: list[float] = []
        best_cost_history: list[float] = []

        for index, controls in enumerate(candidates):
            candidate = self._rollout_candidate(
                interceptor_initial=interceptor_initial,
                target_initial=target_initial,
                target_control=target_control,
                target_covariance=(
                    np.asarray(target_covariance, dtype=float).copy()
                    if target_covariance is not None
                    else np.eye(4, dtype=float) * self._cost_model.gamma
                ),
                controls=controls,
            )
            candidate_costs.append(candidate.total_cost)
            if best_candidate is None or candidate.total_cost < best_candidate.total_cost:
                best_candidate = candidate
                best_index = index
            assert best_candidate is not None
            best_cost_history.append(best_candidate.total_cost)

        assert best_candidate is not None
        return OptimizationResult(
            optimal_path=best_candidate.interceptor_states[:, :2].copy(),
            optimal_controls=best_candidate.controls.copy(),
            optimal_cost=best_candidate.total_cost,
            best_candidate_index=best_index,
            evaluated_trajectories=len(candidates),
            candidate_costs=np.asarray(candidate_costs, dtype=float),
            best_cost_history=np.asarray(best_cost_history, dtype=float),
        )

    def _generate_control_candidates(
        self,
        interceptor_initial: np.ndarray,
        target_initial: np.ndarray,
    ) -> list[np.ndarray]:
        candidates: list[np.ndarray] = []
        target_direction = target_initial[:2] - interceptor_initial[:2]
        target_distance = np.linalg.norm(target_direction)
        if target_distance < 1e-6:
            base_direction = np.array([1.0, 0.0], dtype=float)
        else:
            base_direction = target_direction / target_distance

        angle_offsets = np.linspace(-0.8, 0.8, max(self._num_trajectories, 3))
        magnitude_levels = np.linspace(
            0.3 * self._constraints.max_acceleration_mps2,
            self._constraints.max_acceleration_mps2,
            max(min(self._num_trajectories, 5), 2),
        )

        for magnitude in magnitude_levels:
            for angle in angle_offsets:
                rotated = _rotate_vector(base_direction, angle)
                control = rotated * magnitude
                candidates.append(np.repeat(control[None, :], self._horizon_steps, axis=0))
                if len(candidates) >= self._num_trajectories:
                    return candidates

        while len(candidates) < self._num_trajectories:
            random_control = self._rng.uniform(
                low=-self._constraints.max_acceleration_mps2,
                high=self._constraints.max_acceleration_mps2,
                size=(self._horizon_steps, 2),
            )
            norms = np.linalg.norm(random_control, axis=1, keepdims=True)
            clipped = np.where(
                norms > self._constraints.max_acceleration_mps2,
                random_control / np.maximum(norms, 1e-6) * self._constraints.max_acceleration_mps2,
                random_control,
            )
            candidates.append(clipped)
        return candidates

    def _rollout_candidate(
        self,
        interceptor_initial: np.ndarray,
        target_initial: np.ndarray,
        target_control: np.ndarray,
        target_covariance: np.ndarray,
        controls: np.ndarray,
    ) -> TrajectoryCandidate:
        interceptor_states = [interceptor_initial.copy()]
        target_states = [target_initial.copy()]
        target_covariances = [target_covariance.copy()]
        total_cost = 0.0
        current_interceptor = interceptor_initial.copy()
        current_target = target_initial.copy()
        current_target_covariance = target_covariance.copy()
        a_matrix, _ = build_state_matrices(self._dt)
        target_process_noise = self._build_process_noise(self._dt, scale=0.5)

        for control in controls:
            next_interceptor = update_state(current_interceptor, control, self._dt)
            next_target = update_state(current_target, target_control, self._dt)
            current_target_covariance = a_matrix @ current_target_covariance @ a_matrix.T + target_process_noise
            constraint_status = self._constraint_status(
                interceptor_state=next_interceptor,
                target_state=next_target,
                control=control,
            )
            stage_cost = self._cost_model.stage_cost(
                interceptor_position=next_interceptor[:2],
                target_position=next_target[:2],
                control_input=control,
                constraint_status=constraint_status,
                uncertainty_term=float(np.trace(current_target_covariance[:2, :2])),
            )
            total_cost += stage_cost
            interceptor_states.append(next_interceptor)
            target_states.append(next_target)
            target_covariances.append(current_target_covariance.copy())
            current_interceptor = next_interceptor
            current_target = next_target

        return TrajectoryCandidate(
            controls=controls,
            interceptor_states=np.asarray(interceptor_states, dtype=float),
            target_states=np.asarray(target_states, dtype=float),
            target_covariances=np.asarray(target_covariances, dtype=float),
            total_cost=float(total_cost),
        )

    def _constraint_status(
        self,
        interceptor_state: np.ndarray,
        target_state: np.ndarray,
        control: np.ndarray,
    ) -> ConstraintStatus:
        speed = float(np.linalg.norm(interceptor_state[2:4]))
        acceleration = float(np.linalg.norm(control))
        distance = float(np.linalg.norm(interceptor_state[:2] - target_state[:2]))
        return ConstraintStatus(
            velocity_clipped=speed > self._constraints.max_velocity_mps,
            acceleration_clipped=acceleration > self._constraints.max_acceleration_mps2,
            tracking_ok=True,
            drift_rate_in_bounds=True,
            safety_override=distance < self._constraints.min_separation_m,
            distance_to_target_m=distance,
        )

    def _build_process_noise(self, dt: float, scale: float = 1.0) -> np.ndarray:
        spectral_density = max(self._cost_model.gamma * scale, 1e-6)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        return spectral_density * np.array(
            [
                [0.25 * dt4, 0.0, 0.5 * dt3, 0.0],
                [0.0, 0.25 * dt4, 0.0, 0.5 * dt3],
                [0.5 * dt3, 0.0, dt2, 0.0],
                [0.0, 0.5 * dt3, 0.0, dt2],
            ],
            dtype=float,
        )


def _rotate_vector(vector: np.ndarray, angle_rad: float) -> np.ndarray:
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=float)
    return rotation @ vector
