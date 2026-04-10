from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from drone_interceptor.constraints import ConstraintStatus


def compute_constraint_penalty(
    constraint_status: ConstraintStatus | None,
    penalty_weight: float = 1.0,
) -> float:
    if constraint_status is None:
        return 0.0

    penalty = 0.0
    if constraint_status.velocity_clipped:
        penalty += 1.0
    if constraint_status.acceleration_clipped:
        penalty += 1.0
    if not constraint_status.tracking_ok:
        penalty += 2.0
    if not constraint_status.drift_rate_in_bounds:
        penalty += 2.0
    if constraint_status.safety_override:
        penalty += 5.0
    return float(penalty_weight * penalty)


def compute_interception_cost(
    interceptor_position: np.ndarray,
    target_position: np.ndarray,
    control_input: np.ndarray,
    dt: float,
    alpha: float = 0.1,
    beta: float = 10.0,
    gamma: float = 1.0,
    constraint_status: ConstraintStatus | None = None,
    uncertainty_term: float = 0.0,
) -> float:
    distance_term = float(np.linalg.norm(np.asarray(interceptor_position, dtype=float) - np.asarray(target_position, dtype=float)) ** 2)
    control_term = float(np.linalg.norm(np.asarray(control_input, dtype=float)) ** 2)
    penalty_term = compute_constraint_penalty(constraint_status)
    uncertainty_term = float(max(uncertainty_term, 0.0))
    return float((distance_term + alpha * control_term + beta * penalty_term + gamma * uncertainty_term) * float(dt))


@dataclass(slots=True)
class InterceptionCostModel:
    alpha: float = 0.1
    beta: float = 10.0
    gamma: float = 1.0
    dt: float = 0.1

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "InterceptionCostModel":
        optimization = config.get("optimization", {})
        mission = config.get("mission", {})
        return cls(
            alpha=float(optimization.get("alpha", 0.1)),
            beta=float(optimization.get("beta", 10.0)),
            gamma=float(optimization.get("gamma", optimization.get("uncertainty_weight", 1.0))),
            dt=float(mission.get("time_step", 0.1)),
        )

    def stage_cost(
        self,
        interceptor_position: np.ndarray,
        target_position: np.ndarray,
        control_input: np.ndarray,
        constraint_status: ConstraintStatus | None = None,
        uncertainty_term: float = 0.0,
    ) -> float:
        return compute_interception_cost(
            interceptor_position=interceptor_position,
            target_position=target_position,
            control_input=control_input,
            dt=self.dt,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            constraint_status=constraint_status,
            uncertainty_term=uncertainty_term,
        )
