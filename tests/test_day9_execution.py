from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.navigation.drift_model import DP5CoordinateSpoofingToolkit
from drone_interceptor.validation.day9 import run_day9_execution


def test_dp5_coordinate_spoofing_toolkit_stays_within_gradient_band() -> None:
    toolkit = DP5CoordinateSpoofingToolkit(
        safe_zone_position=np.array([100.0, 100.0, 0.0], dtype=float),
        min_rate_mps=0.2,
        max_rate_mps=0.5,
        noise_std_m=0.0,
        random_seed=17,
    )
    true_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 2.0, 0.0],
            [20.0, 4.0, 0.0],
        ],
        dtype=float,
    )
    interceptor_positions = np.array(
        [
            [250.0, 0.0, 0.0],
            [90.0, 0.0, 0.0],
            [15.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    samples = toolkit.generate_profile(true_positions, interceptor_positions, dt=0.1, mode="directed")

    assert len(samples) == 3
    assert all(0.2 <= sample.drift_rate_mps <= 0.5 for sample in samples)
    assert samples[-1].drift_rate_mps > samples[0].drift_rate_mps


def test_day9_execution_writes_outputs() -> None:
    metrics, compliance, artifacts = run_day9_execution(
        project_root=ROOT,
        random_seed=73,
        max_steps_override=40,
    )

    assert artifacts.demo_video.exists()
    assert artifacts.hero_image.exists()
    assert artifacts.spoofing_profile_csv.exists()
    assert artifacts.benchmark_csv.exists()
    assert artifacts.summary_json.exists()
    assert artifacts.log_file.exists()
    assert 0.0 <= metrics.tracking_precision_ratio <= 1.0
    assert 0.2 <= metrics.min_drift_rate_mps <= 0.5
    assert 0.2 <= metrics.max_drift_rate_mps <= 0.5
    assert compliance.simulation_ready is True
    assert compliance.rf_integrity_ready is False
