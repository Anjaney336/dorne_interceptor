from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.navigation.ekf_filter import InterceptorEKF
from drone_interceptor.simulation.airsim_manager import AirSimMissionManager


def test_ekf_detects_large_spoofing_innovation() -> None:
    ekf = InterceptorEKF(dt=0.1, process_noise=0.01, measurement_noise=0.2)
    ekf.initialize(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    ekf.predict(drift_rate_mps=0.0)
    assessment = ekf.assess(np.array([40.0, 25.0, 0.0]))
    assert assessment.spoofing_detected is True
    previous_position = ekf.position.copy()
    ekf.update(np.array([40.0, 25.0, 0.0]), is_spoofing_detected=True)
    assert np.linalg.norm(ekf.position - previous_position) < 1.0


def test_airsim_manager_runs_multi_target_replay_and_validation() -> None:
    manager = AirSimMissionManager(connect=False)
    setup = manager.setup_swarm(3, random_seed=11)
    telemetry = manager.get_live_telemetry()
    spoofed = manager.apply_spoofing(setup["target_names"][0], drift_rate=0.3, time_s=5.0, noise_std_m=0.0)
    replay = manager.run_replay(
        num_targets=3,
        use_ekf=True,
        drift_rate_mps=0.3,
        noise_std_m=0.45,
        packet_loss_rate=0.2,
        max_steps=60,
    )
    validation = manager.run_monte_carlo_validation(
        iterations=10,
        num_targets=3,
        drift_rate_mps=0.3,
        noise_std_m=0.45,
        packet_loss_rate=0.2,
    )

    assert len(setup["target_names"]) == 3
    assert setup["advanced_visuals"]["ready"] is True
    assert "Interceptor_pursuit" in telemetry
    assert spoofed.shape == (3,)
    assert len(replay.frames) > 0
    assert not replay.map_frame.empty
    assert not replay.distance_frame.empty
    assert "packet_loss_events" in replay.validation
    assert "threat_order" in replay.validation
    assert replay.validation["use_ekf_anti_spoofing"] is True
    assert any("threat_level" in row for row in replay.map_frame.to_dict("records")[:5])
    assert any(target.innovation_gate >= 0.0 for target in replay.frames[-1].targets)
    assert 0.0 <= validation.ekf_success_rate <= 1.0
    assert validation.ekf_success_rate >= validation.raw_success_rate
    assert validation.ekf_success_rate == np.mean([float(row["ekf_success"]) for row in validation.iteration_records])
    assert validation.ekf_mean_miss_distance_m >= 0.0
    assert validation.ekf_mean_kill_probability >= 0.0
    assert len(validation.per_target_summary) == 3
    assert len(validation.iteration_records) == 10
    assert all(0.0 <= float(row["ekf_success_rate"]) <= 1.0 for row in validation.per_target_summary)


def test_airsim_manager_exports_cinematic_demo(tmp_path: Path) -> None:
    manager = AirSimMissionManager(connect=False)
    replay = manager.run_replay(
        num_targets=2,
        use_ekf=True,
        drift_rate_mps=0.3,
        noise_std_m=0.45,
        packet_loss_rate=0.1,
        max_steps=20,
    )
    manager._cinematic_recorder = manager._cinematic_recorder.__class__(client=None, output_dir=tmp_path, fps=20.44)
    output = manager.export_cinematic_demo(replay, prefix="test_bms")

    assert output.exists()
    assert output.suffix == ".mp4"


def test_run_replay_emits_dynamic_detection_fps() -> None:
    replay = AirSimMissionManager(connect=False).run_replay(
        num_targets=3,
        use_ekf=True,
        drift_rate_mps=0.45,
        noise_std_m=1.2,
        packet_loss_rate=0.3,
        max_steps=24,
    )
    fps_values = [round(float(frame.detection_fps), 3) for frame in replay.frames]
    assert len(set(fps_values)) > 1
