from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


DriftMode = Literal["linear", "circular", "directed"]


@dataclass(frozen=True, slots=True)
class DriftSample:
    true_position: np.ndarray
    fake_position: np.ndarray
    drift_offset: np.ndarray
    drift_direction: np.ndarray
    noise: np.ndarray
    adaptive_rate_mps: float
    mode: DriftMode
    distance_to_interceptor_m: float
    distance_to_safe_zone_m: float


class IntelligentDriftEngine:
    """Adaptive spoofing model used to bias target navigation in simulation."""

    def __init__(
        self,
        min_rate_mps: float = 0.2,
        max_rate_mps: float = 0.5,
        near_distance_m: float = 120.0,
        noise_std_m: float = 0.12,
        safe_zone_position: np.ndarray | None = None,
        circular_frequency_hz: float = 0.18,
        circular_vertical_gain: float = 0.08,
        safe_zone_bias_gain: float = 0.65,
        distance_bias_scale_m: float = 180.0,
        random_seed: int = 7,
    ) -> None:
        self._min_rate = float(min_rate_mps)
        self._max_rate = float(max(max_rate_mps, min_rate_mps))
        self._near_distance = float(max(near_distance_m, 1.0))
        self._noise_std = float(max(noise_std_m, 0.0))
        self._safe_zone = (
            np.asarray(safe_zone_position, dtype=float).copy()
            if safe_zone_position is not None
            else np.array([120.0, -120.0, 120.0], dtype=float)
        )
        self._circular_frequency_hz = float(max(circular_frequency_hz, 1e-3))
        self._circular_vertical_gain = float(max(circular_vertical_gain, 0.0))
        self._safe_zone_bias_gain = float(np.clip(safe_zone_bias_gain, 0.0, 1.0))
        self._distance_bias_scale_m = float(max(distance_bias_scale_m, 1.0))
        self._rng = np.random.default_rng(int(random_seed))

    @property
    def safe_zone(self) -> np.ndarray:
        return self._safe_zone.copy()

    def sample(
        self,
        true_position: np.ndarray,
        interceptor_position: np.ndarray,
        time_s: float,
        mode: DriftMode = "directed",
    ) -> DriftSample:
        true_position = np.asarray(true_position, dtype=float)
        interceptor_position = np.asarray(interceptor_position, dtype=float)
        distance_to_interceptor_m = float(np.linalg.norm(true_position - interceptor_position))
        distance_to_safe_zone_m = float(np.linalg.norm(self._safe_zone - true_position))
        adaptive_rate = self._adaptive_rate(
            distance_to_interceptor_m=distance_to_interceptor_m,
            distance_to_safe_zone_m=distance_to_safe_zone_m,
        )
        direction = self._direction_for_mode(
            mode=mode,
            true_position=true_position,
            interceptor_position=interceptor_position,
            time_s=float(time_s),
        )
        noise = self._sample_noise()
        drift_offset = adaptive_rate * float(time_s) * direction + noise
        fake_position = true_position + drift_offset
        return DriftSample(
            true_position=true_position.copy(),
            fake_position=fake_position,
            drift_offset=drift_offset,
            drift_direction=direction,
            noise=noise,
            adaptive_rate_mps=adaptive_rate,
            mode=mode,
            distance_to_interceptor_m=distance_to_interceptor_m,
            distance_to_safe_zone_m=distance_to_safe_zone_m,
        )

    def _adaptive_rate(self, distance_to_interceptor_m: float, distance_to_safe_zone_m: float) -> float:
        proximity = 1.0 - min(max(distance_to_interceptor_m / self._near_distance, 0.0), 1.0)
        progress_bias = min(max(distance_to_safe_zone_m / self._distance_bias_scale_m, 0.0), 1.0)
        combined_weight = np.clip(0.7 * proximity + 0.3 * progress_bias, 0.0, 1.0)
        return float(self._min_rate + (self._max_rate - self._min_rate) * combined_weight)

    def _direction_for_mode(
        self,
        mode: DriftMode,
        true_position: np.ndarray,
        interceptor_position: np.ndarray,
        time_s: float,
    ) -> np.ndarray:
        if mode == "linear":
            return np.array([1.0, 0.15, 0.0], dtype=float) / np.linalg.norm(np.array([1.0, 0.15, 0.0], dtype=float))
        if mode == "circular":
            angle = 2.0 * np.pi * self._circular_frequency_hz * time_s
            orbital_vector = np.array(
                [
                    np.cos(angle),
                    np.sin(angle),
                    self._circular_vertical_gain * np.sin(0.5 * angle),
                ],
                dtype=float,
            )
            safe_zone_direction = _normalize(self._safe_zone - true_position)
            vector = ((1.0 - self._safe_zone_bias_gain) * orbital_vector) + (self._safe_zone_bias_gain * safe_zone_direction)
            return vector / max(float(np.linalg.norm(vector)), 1e-6)
        del interceptor_position
        return _normalize(self._safe_zone - true_position)

    def _sample_noise(self) -> np.ndarray:
        if self._noise_std <= 0.0:
            return np.zeros(3, dtype=float)
        noise = self._rng.normal(0.0, self._noise_std, size=3)
        noise[2] *= 0.35
        return np.asarray(noise, dtype=float)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.zeros_like(vector, dtype=float)
    return np.asarray(vector, dtype=float) / norm


__all__ = ["DriftMode", "DriftSample", "IntelligentDriftEngine"]
