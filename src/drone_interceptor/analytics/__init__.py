"""Analytics and reporting utilities."""

from drone_interceptor.analytics.metrics import PlatformMetrics, ScenarioResultRecord
from drone_interceptor.analytics.visualization import (
    plot_cost_vs_iteration,
    plot_distance_vs_time,
    plot_fps_vs_model,
    plot_platform_3d_trajectory,
    plot_rmse_vs_noise,
    plot_success_rate_vs_scenario,
)

__all__ = [
    "PlatformMetrics",
    "ScenarioResultRecord",
    "plot_cost_vs_iteration",
    "plot_distance_vs_time",
    "plot_fps_vs_model",
    "plot_platform_3d_trajectory",
    "plot_rmse_vs_noise",
    "plot_success_rate_vs_scenario",
]
