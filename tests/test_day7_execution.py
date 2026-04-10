from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.navigation.drift_model import IntelligentDriftEngine
from drone_interceptor.validation.day7 import run_day7_execution


def test_intelligent_drift_engine_adapts_rate_and_points_to_safe_zone() -> None:
    engine = IntelligentDriftEngine(
        min_rate_mps=0.2,
        max_rate_mps=0.5,
        near_distance_m=120.0,
        noise_std_m=0.0,
        safe_zone_position=np.array([100.0, 100.0, 0.0], dtype=float),
        random_seed=17,
    )

    far_sample = engine.sample(
        true_position=np.array([0.0, 0.0, 0.0], dtype=float),
        interceptor_position=np.array([300.0, 0.0, 0.0], dtype=float),
        time_s=5.0,
        mode="directed",
    )
    near_sample = engine.sample(
        true_position=np.array([0.0, 0.0, 0.0], dtype=float),
        interceptor_position=np.array([20.0, 0.0, 0.0], dtype=float),
        time_s=5.0,
        mode="directed",
    )

    assert near_sample.adaptive_rate_mps > far_sample.adaptive_rate_mps
    assert np.allclose(far_sample.drift_direction[:2], np.array([np.sqrt(0.5), np.sqrt(0.5)]), atol=1e-3)


def test_day7_execution_writes_outputs() -> None:
    summaries, metrics, artifacts = run_day7_execution(
        project_root=ROOT,
        random_seed=73,
        max_steps_override=40,
    )

    assert len(summaries) == 3
    assert 0.0 <= metrics.success_rate <= 1.0
    assert 0.0 <= metrics.redirection_success_rate <= 1.0
    assert artifacts.trajectory_plot.exists()
    assert artifacts.demo_video.exists()
    assert artifacts.log_file.exists()
