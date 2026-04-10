from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScenarioResultRecord:
    scenario: str
    success: bool
    interception_time_s: float | None
    rmse_m: float
    mean_loop_fps: float
    mean_stage_cost: float
    final_distance_m: float
    noise_level: float
    model_label: str
    log_file: str


@dataclass(frozen=True)
class PlatformMetrics:
    success_rate: float
    mean_interception_time_s: float
    mean_rmse_m: float
    mean_fps: float


def rmse(values: list[float]) -> float:
    if not values:
        return 0.0
    data = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(data))))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


__all__ = ["PlatformMetrics", "ScenarioResultRecord", "mean", "rmse"]
