from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from drone_interceptor.types import ControlCommand, TargetState


@dataclass(slots=True)
class ConstraintEnvelope:
    max_velocity_mps: float
    max_acceleration_mps2: float
    tracking_precision_m: float
    drift_rate_min_mps: float
    drift_rate_max_mps: float
    min_separation_m: float
    collision_radius_m: float


@dataclass(slots=True)
class ConstraintStatus:
    velocity_clipped: bool
    acceleration_clipped: bool
    tracking_ok: bool
    drift_rate_in_bounds: bool
    safety_override: bool
    distance_to_target_m: float


def load_constraint_envelope(config: dict[str, Any]) -> ConstraintEnvelope:
    constraints = config.get("constraints", {})
    physical = constraints.get("physical", {})
    tracking = constraints.get("tracking", {})
    drift = constraints.get("drift", {})
    safety = constraints.get("safety", {})
    planning = config.get("planning", {})
    control = config.get("control", {})

    return ConstraintEnvelope(
        max_velocity_mps=float(physical.get("max_velocity_mps", planning.get("max_speed_mps", 20.0))),
        max_acceleration_mps2=float(physical.get("max_acceleration_mps2", control.get("max_acceleration_mps2", 18.0))),
        tracking_precision_m=float(tracking.get("max_position_error_m", 0.5)),
        drift_rate_min_mps=float(drift.get("min_rate_mps", 0.2)),
        drift_rate_max_mps=float(drift.get("max_rate_mps", 0.5)),
        min_separation_m=float(safety.get("min_separation_m", 5.0)),
        collision_radius_m=float(safety.get("collision_radius_m", 1.0)),
    )


def clamp_drift_rate(drift_rate_mps: float, envelope: ConstraintEnvelope) -> float:
    return float(np.clip(drift_rate_mps, envelope.drift_rate_min_mps, envelope.drift_rate_max_mps))


def tracking_precision_ok(position_error_m: float, envelope: ConstraintEnvelope) -> bool:
    return bool(position_error_m <= envelope.tracking_precision_m)


class ConstraintModel:
    """Centralized physical, tracking, drift, and safety constraints."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._envelope = load_constraint_envelope(config)

    @property
    def envelope(self) -> ConstraintEnvelope:
        return self._envelope

    def enforce_guidance_command(
        self,
        interceptor_state: TargetState,
        target_state: TargetState,
        raw_acceleration: np.ndarray,
        dt: float,
        tracking_error_m: float = 0.0,
    ) -> tuple[ControlCommand, ConstraintStatus]:
        interceptor_position = _as_3d(interceptor_state.position)
        interceptor_velocity = _as_3d(interceptor_state.velocity)
        target_position = _as_3d(target_state.position)
        relative_position = target_position - interceptor_position
        distance = float(np.linalg.norm(relative_position))
        acceleration = _clip_vector(raw_acceleration, self._envelope.max_acceleration_mps2)
        acceleration_clipped = bool(np.linalg.norm(raw_acceleration) > self._envelope.max_acceleration_mps2 + 1e-9)
        safety_override = False

        if distance <= self._envelope.min_separation_m:
            safety_override = True
            if distance < 1e-6:
                escape_direction = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                escape_direction = -relative_position / distance
            acceleration = escape_direction * self._envelope.max_acceleration_mps2

        velocity = interceptor_velocity + acceleration * float(dt)
        clipped_velocity = _clip_vector(velocity, self._envelope.max_velocity_mps)
        velocity_clipped = bool(np.linalg.norm(velocity) > self._envelope.max_velocity_mps + 1e-9)

        command = ControlCommand(
            velocity_command=clipped_velocity,
            acceleration_command=acceleration,
            mode="pn",
            metadata={
                "constraint_max_velocity_mps": self._envelope.max_velocity_mps,
                "constraint_max_acceleration_mps2": self._envelope.max_acceleration_mps2,
                "constraint_min_separation_m": self._envelope.min_separation_m,
            },
        )
        status = ConstraintStatus(
            velocity_clipped=velocity_clipped,
            acceleration_clipped=acceleration_clipped,
            tracking_ok=tracking_precision_ok(tracking_error_m, self._envelope),
            drift_rate_in_bounds=True,
            safety_override=safety_override,
            distance_to_target_m=distance,
        )
        return command, status


def _clip_vector(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm < 1e-6:
        return vector.astype(float)
    return vector.astype(float) / norm * max_norm


def _as_3d(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=float).reshape(-1)
    if array.shape == (3,):
        return array
    if array.shape == (2,):
        return np.array([array[0], array[1], 0.0], dtype=float)
    raise ValueError("Constraint model expects 2D or 3D state vectors.")
