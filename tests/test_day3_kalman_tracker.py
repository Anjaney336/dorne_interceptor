from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.config import load_config
from drone_interceptor.tracking.tracker import TargetTracker
from drone_interceptor.types import Detection


def test_kalman_target_tracker_reduces_noisy_measurement_rmse() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["tracking"]["mode"] = "kalman"
    tracker = TargetTracker(config)

    true_positions: list[np.ndarray] = []
    measured_positions: list[np.ndarray] = []
    filtered_positions: list[np.ndarray] = []
    rng = np.random.default_rng(11)
    dt = float(config["mission"]["time_step"])

    for step in range(30):
        timestamp = step * dt
        truth = np.array([2.0 * timestamp, -1.0 + 0.5 * timestamp, 10.0], dtype=float)
        measurement = truth + rng.normal(0.0, [1.0, 1.0, 0.2], size=3)
        track = tracker.update(
            Detection(
                position=measurement,
                confidence=0.9,
                timestamp=timestamp,
                metadata={"backend": "synthetic"},
            )
        )
        true_positions.append(truth)
        measured_positions.append(measurement)
        filtered_positions.append(track.position.copy())

    truth_array = np.asarray(true_positions, dtype=float)
    measured_array = np.asarray(measured_positions, dtype=float)
    filtered_array = np.asarray(filtered_positions, dtype=float)

    raw_rmse = float(np.sqrt(np.mean((measured_array - truth_array) ** 2)))
    filtered_rmse = float(np.sqrt(np.mean((filtered_array - truth_array) ** 2)))

    assert filtered_rmse < raw_rmse


def test_kalman_innovation_gating_with_spoofing_jump() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["tracking"]["mode"] = "kalman"
    tracker = TargetTracker(config)

    dt = float(config["mission"]["time_step"])
    mahalanobis_distances: list[float] = []

    # Normal tracking
    for step in range(10):
        timestamp = step * dt
        truth = np.array([2.0 * timestamp, -1.0 + 0.5 * timestamp, 10.0], dtype=float)
        measurement = truth + np.random.normal(0.0, [0.5, 0.5, 0.1], size=3)
        track = tracker.update(
            Detection(
                position=measurement,
                confidence=0.9,
                timestamp=timestamp,
                metadata={"backend": "synthetic"},
            )
        )
        mahalanobis_distances.append(track.metadata.get("mahalanobis_distance", 0.0))

    # Spoofing jump at step 10
    timestamp = 10 * dt
    truth = np.array([2.0 * timestamp, -1.0 + 0.5 * timestamp, 10.0], dtype=float)
    spoofed_measurement = truth + np.array([10.0, 0.0, 0.0], dtype=float)  # 10m jump
    track = tracker.update(
        Detection(
            position=spoofed_measurement,
            confidence=0.9,
            timestamp=timestamp,
            metadata={"backend": "synthetic"},
        )
    )
    spoof_mahalanobis = track.metadata.get("mahalanobis_distance", 0.0)
    mahalanobis_distances.append(spoof_mahalanobis)

    # Check that normal mahalanobis are low
    normal_mean = np.mean(mahalanobis_distances[:-1])
    assert normal_mean < 10.0  # Chi-squared for 2 dof is about 5.99 for 95%

    # Spoof mahalanobis is high
    assert spoof_mahalanobis > 20.0  # Should be much higher

    # Could add gating logic here, but for now just check the distance
