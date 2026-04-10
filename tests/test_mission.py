from pathlib import Path
import sys
import asyncio
import time

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.backend.mission_service import MissionConfig, MissionController
from drone_interceptor.backend.run_store import FileRunStore
from drone_interceptor.simulation import telemetry_api as telemetry_api_module
from drone_interceptor.simulation.telemetry_api import app


def test_gnc_interceptor_converges_to_target(tmp_path: Path) -> None:
    config = MissionConfig(
        num_targets=1,
        target_speed_mps=4.0,
        interceptor_speed_mps=20.0,
        drift_rate_mps=0.0,
        noise_level_m=0.10,
        telemetry_latency_ms=0.0,
        packet_loss_rate=0.0,
        max_steps=int(20.0 / 0.05),
        dt=0.05,
        use_ekf=True,
    )
    controller = MissionController(config, output_dir=tmp_path / "outputs")
    result = asyncio.run(controller.run_mission())

    assert result["mission_success"] is True
    final_distance = min(np.linalg.norm(controller.interceptor.position - target.position) for target in controller.targets)
    assert final_distance < config.kill_radius_m


def test_ekf_rmse_below_threshold_under_high_drift(tmp_path: Path) -> None:
    config = MissionConfig(
        num_targets=1,
        target_speed_mps=6.0,
        interceptor_speed_mps=20.0,
        drift_rate_mps=0.5,
        noise_level_m=0.15,
        telemetry_latency_ms=0.0,
        packet_loss_rate=0.0,
        max_steps=int(20.0 / 0.05),
        dt=0.05,
        use_ekf=True,
    )
    controller = MissionController(config, output_dir=tmp_path / "outputs")
    asyncio.run(controller.run_mission())

    rmse_list = [frame.rmse_m for frame in controller.telemetry_log]
    mean_rmse = float(np.mean(np.asarray(rmse_list, dtype=float))) if rmse_list else 0.0
    assert mean_rmse < 0.3


def test_artifacts_endpoint_returns_valid_download_url(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(telemetry_api_module, "RUN_STORE", FileRunStore(tmp_path / "run_registry"))
    monkeypatch.setattr(telemetry_api_module, "OUTPUTS", tmp_path / "outputs")
    client = TestClient(app)

    response = client.post(
        "/mission/start/v2",
        json={
            "num_targets": 1,
            "max_steps": 20,
            "dt": 0.05,
            "use_ekf": True,
            "drift_rate_mps": 0.1,
            "noise_level_m": 0.1,
            "telemetry_latency_ms": 0.0,
            "packet_loss_rate": 0.0,
        },
    )
    assert response.status_code == 200
    run_id = response.json()["run_id"]

    run_record = None
    for _ in range(30):
        run_record = telemetry_api_module.RUN_STORE.get_run(run_id)
        if run_record.status == "complete":
            break
        time.sleep(0.5)
    assert run_record is not None
    assert run_record.status == "complete"

    artifacts_response = client.get(f"/mission/{run_id}/artifacts")
    assert artifacts_response.status_code == 200
    artifacts_payload = artifacts_response.json()
    assert "fpv_video_mp4" in artifacts_payload["artifacts"]
    video_artifact = artifacts_payload["artifacts"]["fpv_video_mp4"]
    assert video_artifact["download_url"].startswith("file://")
    assert Path(video_artifact["path"]).exists()
