from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.config import load_config
from drone_interceptor.navigation.drift_model import AttackProfile, DP5CoordinateSpoofingToolkit


def test_attack_profile_supports_delayed_and_intermittent_activation() -> None:
    toolkit = DP5CoordinateSpoofingToolkit(
        safe_zone_position=np.array([100.0, 100.0, 0.0], dtype=float),
        random_seed=17,
    )
    profile = AttackProfile(
        name="delayed_intermittent",
        mode="directed",
        onset_time_s=1.0,
        intermittent_period_s=1.0,
        duty_cycle=0.5,
    )
    samples = toolkit.generate_profile(
        true_positions=np.tile(np.array([[0.0, 0.0, 0.0]], dtype=float), (5, 1)),
        interceptor_positions=np.tile(np.array([[50.0, 0.0, 0.0]], dtype=float), (5, 1)),
        dt=0.5,
        attack_profile=profile,
    )
    assert samples[0].attack_active is False
    assert any(sample.attack_active for sample in samples[2:])


def test_defense_sweep_returns_multiple_modes() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    toolkit = DP5CoordinateSpoofingToolkit(
        safe_zone_position=np.array([100.0, 100.0, 0.0], dtype=float),
        random_seed=17,
    )
    true_positions = np.array([[0.0 + 2.0 * idx, 1.0 * idx, 0.0] for idx in range(10)], dtype=float)
    true_velocities = np.array([[2.0, 1.0, 0.0] for _ in range(10)], dtype=float)
    interceptor_positions = np.array([[40.0, 0.0, 0.0] for _ in range(10)], dtype=float)
    results = toolkit.run_defense_sweep(
        config=config,
        true_positions=true_positions,
        true_velocities=true_velocities,
        interceptor_positions=interceptor_positions,
        dt=0.1,
        attack_profile=AttackProfile(name="directed", mode="directed"),
        packet_loss_rate=0.1,
    )
    modes = {result.defense_mode for result in results}
    assert "raw_gps" in modes
    assert "kalman_fusion" in modes
    assert "ekf_innovation_gate" in modes
    assert "adaptive_trust" in modes
