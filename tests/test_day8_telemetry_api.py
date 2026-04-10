from pathlib import Path
import sys
import time
from math import isclose

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.simulation.airsim_manager import AirSimMissionManager
from drone_interceptor.simulation import telemetry_api as telemetry_api_module
from drone_interceptor.simulation.telemetry_api import _build_terminal_snapshot, app


client = TestClient(app)


def test_preflight_and_validate_endpoints() -> None:
    preflight = client.post(
        "/preflight",
        json={
            "num_targets": 3,
            "connect_airsim": False,
            "interceptor_speed_mps": 28.0,
        },
    )
    assert preflight.status_code == 200
    preflight_body = preflight.json()
    assert preflight_body["ready"] is True
    assert preflight_body["spawned_targets"] == 3

    validation = client.post(
        "/validate",
        json={
            "iterations": 5,
            "num_targets": 3,
            "connect_airsim": False,
            "drift_rate_mps": 0.3,
            "noise_std_m": 0.45,
            "packet_loss_rate": 0.2,
        },
    )
    assert validation.status_code == 200
    validation_body = validation.json()
    assert validation_body["validation_success"] is bool(
        validation_body["ekf_success_rate"] >= 1.0
        and validation_body["ekf_mean_miss_distance_m"] <= 0.5
    )
    assert 0.0 <= validation_body["ekf_success_rate"] <= 1.0
    assert validation_body["ekf_success_rate"] >= validation_body["raw_success_rate"]
    assert validation_body["ekf_mean_kill_probability"] >= 0.0
    assert len(validation_body["per_target_summary"]) == 3
    assert len(validation_body["scenario_results"]) == 5
    assert len(validation_body["iteration_records"]) == 5
    assert validation_body["scenario_results"] == validation_body["iteration_records"]
    assert validation_body["ekf_success_rate"] == sum(
        1 for row in validation_body["iteration_records"] if row["ekf_success"]
    ) / len(validation_body["iteration_records"])


def test_validate_multiple_targets_with_varying_noise() -> None:
    # Test with 3 targets and varying noise levels
    validation = client.post(
        "/validate",
        json={
            "iterations": 3,
            "num_targets": 3,
            "connect_airsim": False,
            "drift_rate_mps": 0.2,
            "noise_std_m": 0.5,  # Varying noise
            "packet_loss_rate": 0.0,
        },
    )
    assert validation.status_code == 200
    validation_body = validation.json()
    assert len(validation_body["per_target_summary"]) == 3
    target_names = [row["target"] for row in validation_body["per_target_summary"]]
    assert len(set(target_names)) == 3  # Unique target_ids
    for row in validation_body["per_target_summary"]:
        assert "ekf_success_rate" in row
        assert "ekf_mean_miss_distance_m" in row
        assert 0.0 <= row["ekf_success_rate"] <= 1.0


def test_validate_edge_case_zero_vs_high_latency() -> None:
    # Test zero latency
    validation_zero = client.post(
        "/validate",
        json={
            "iterations": 2,
            "num_targets": 2,
            "connect_airsim": False,
            "drift_rate_mps": 0.1,
            "noise_std_m": 0.3,
            "packet_loss_rate": 0.0,
            "latency_ms": 0.0,
        },
    )
    assert validation_zero.status_code == 200
    zero_body = validation_zero.json()
    assert zero_body["ekf_success_rate"] >= 0.0

    # Test high latency
    validation_high = client.post(
        "/validate",
        json={
            "iterations": 2,
            "num_targets": 2,
            "connect_airsim": False,
            "drift_rate_mps": 0.1,
            "noise_std_m": 0.3,
            "packet_loss_rate": 0.0,
            "latency_ms": 400.0,
        },
    )
    assert validation_high.status_code == 200
    high_body = validation_high.json()
    assert high_body["ekf_success_rate"] >= 0.0
    # High latency should not crash, but may have lower success


def test_start_mission_and_state_endpoint() -> None:
    started = client.post(
        "/mission/start",
        json={
            "num_targets": 3,
            "connect_airsim": False,
            "use_ekf": True,
            "drift_rate_mps": 0.3,
            "noise_std_m": 0.45,
            "packet_loss_rate": 0.2,
            "max_steps": 20,
            "dt": 0.05,
        },
    )
    assert started.status_code == 200
    time.sleep(0.25)
    state = client.get("/mission/state")
    assert state.status_code == 200
    payload = state.json()
    assert "snapshot" in payload
    assert payload["snapshot"]["status"] in {"preparing", "running", "complete"}
    assert isinstance(payload["snapshot"].get("targets", []), list)


def test_completed_mission_state_uses_terminal_replay_metrics() -> None:
    replay = AirSimMissionManager(connect=False).run_replay(
        num_targets=1,
        use_ekf=True,
        packet_loss_rate=0.0,
        latency_ms=0.0,
        max_steps=6,
        dt=0.05,
    )
    snapshot = _build_terminal_snapshot(replay, mission_id=7, throughput_fps=60.0)
    assert snapshot["status"] == "complete"
    assert snapshot["terminal_metric_basis"] == "closest_approach"
    assert "closest_approach_m" in snapshot
    assert snapshot["kill_probability"] >= 0.0


def test_stream_endpoint_yields_packets() -> None:
    response = client.post(
        "/stream",
        json={
            "num_targets": 2,
            "connect_airsim": False,
            "use_ekf": True,
            "max_steps": 5,
            "dt": 0.05,
        },
    )
    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.strip()]
    assert len(lines) >= 2
    assert '"status": "running"' in lines[0] or '"status":"running"' in lines[0]


def test_status_endpoint_reports_lifecycle() -> None:
    status = client.get("/status")
    assert status.status_code == 200
    payload = status.json()
    assert payload["service_status"] == "ok"
    assert payload["lifecycle"] in {"ACTIVE", "COMPLETE", "PREPARED", "STANDBY"}


def test_telemetry_heartbeat_endpoint_returns_json_schema() -> None:
    telemetry = client.get("/telemetry")
    assert telemetry.status_code == 200
    payload = telemetry.json()
    assert payload["schema_version"] == "1.0"
    assert payload["type"] == "MISSION_HEARTBEAT"
    assert "snapshot" in payload
    assert "mission_status" in payload
    assert "target_count" in payload


def test_status_uses_preflight_snapshot_when_idle() -> None:
    client.post("/mission/stop")
    preflight = client.post(
        "/preflight",
        json={
            "num_targets": 3,
            "connect_airsim": False,
            "interceptor_speed_mps": 28.0,
        },
    )
    assert preflight.status_code == 200
    status = client.get("/status")
    payload = status.json()
    assert payload["stage"] == "Preflight"
    assert payload["target_count"] == 3
    assert payload["lifecycle"] == "PREPARED"


def test_ws_mission_emits_terminal_replay_packet() -> None:
    with client.websocket_connect("/ws/mission") as websocket:
        websocket.send_json(
            {
                "num_targets": 1,
                "connect_airsim": False,
                "use_ekf": True,
                "max_steps": 6,
                "dt": 0.05,
            }
        )
        terminal_message = None
        for _ in range(20):
            message = websocket.receive_json()
            if message.get("type") == "MISSION_COMPLETE":
                terminal_message = message
                break
        assert terminal_message is not None
        assert terminal_message["status"] == "COMPLETED"
        assert isinstance(terminal_message["full_replay"], list)
        assert terminal_message["artifact_url"].endswith("platform_preview.html")


def test_validate_supports_matrix12_mode() -> None:
    response = client.post(
        "/validate",
        json={
            "num_targets": 3,
            "connect_airsim": False,
            "validation_mode": "matrix12",
            "max_steps": 24,
            "dt": 0.05,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["validation_mode"] == "matrix12"
    assert payload["iterations"] == 12
    assert len(payload["scenario_results"]) == 12
    assert {row["category"] for row in payload["scenario_results"]} == {"nominal", "drift_stress", "anti_spoofing", "edge"}


def test_run_mission_returns_multi_target_results_with_ekf_metrics() -> None:
    payload = {
        "target_ids": [f"Track_{index + 1}" for index in range(5)],
        "num_targets": 5,
        "connect_airsim": False,
        "use_ekf": True,
        "use_ekf_anti_spoofing": True,
        "drift_rate_mps": 0.3,
        "noise_std_m": 1.2,
        "latency_ms": 80.0,
        "packet_loss_rate": 0.05,
        "max_steps": 12,
        "dt": 0.05,
    }
    response = client.post("/run_mission", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["workflow_status"] == "success"

    results = body["results"]
    assert len(results) == 5
    assert len(body["validation"]["per_target_summary"]) == 5
    assert len(body["status"]["targets"]) == 5

    for row in results:
        assert "target_id" in row
        assert "ekf_success_rate" in row
        assert "interception_time" in row
        assert "rmse" in row
        assert "closest_approach_m" in row
        assert "intercepted" in row
        assert 0.0 <= float(row["ekf_success_rate"]) <= 1.0

    expected_rmse = sum(float(row["rmse"]) for row in results) / max(len(results), 1)
    assert isclose(float(body["snapshot"]["rmse_measured_true_m"]), expected_rmse, rel_tol=1e-9)
    assert isclose(float(body["snapshot"]["rmse_m"]), expected_rmse, rel_tol=1e-9)
    command_readiness = body.get("mission_insights", {}).get("command_readiness", {})
    assert "readiness_score" in command_readiness
    assert "security_posture" in command_readiness
    assert "quality_gate_passed" in command_readiness
    assert isinstance(body.get("mission_insights", {}).get("engagement_priority_queue", []), list)


def test_run_mission_without_spoof_injection_keeps_spoof_flags_off() -> None:
    payload = {
        "target_ids": ["Target_1", "Target_2", "Target_3"],
        "num_targets": 3,
        "connect_airsim": False,
        "use_ekf": True,
        "use_ekf_anti_spoofing": True,
        "enable_spoofing": False,
        "drift_rate_mps": 0.4,
        "noise_std_m": 1.3,
        "latency_ms": 60.0,
        "packet_loss_rate": 0.0,
        "max_steps": 24,
        "dt": 0.05,
    }
    response = client.post("/run_mission", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["workflow_status"] == "success"
    status_targets = (body.get("status") or {}).get("targets", [])
    for target in status_targets:
        assert bool(target.get("spoofing_active", False)) is False
        assert bool(target.get("spoofing_detected", False)) is False
    assert bool((body.get("snapshot") or {}).get("spoofing_active", False)) is False


def test_runs_endpoint_persists_validation_records(tmp_path, monkeypatch) -> None:
    from drone_interceptor.backend.run_store import FileRunStore

    monkeypatch.setattr(telemetry_api_module, "RUN_STORE", FileRunStore(tmp_path / "run_registry"))
    validation = client.post(
        "/validate",
        json={
            "iterations": 2,
            "num_targets": 2,
            "connect_airsim": False,
            "drift_rate_mps": 0.3,
            "noise_std_m": 0.45,
            "packet_loss_rate": 0.0,
        },
    )
    assert validation.status_code == 200
    run_id = validation.json()["run_id"]

    runs = client.get("/runs")
    assert runs.status_code == 200
    runs_payload = runs.json()
    assert runs_payload["schema_version"] == "1.0"
    assert any(run["run_id"] == run_id for run in runs_payload["runs"])

    run_payload = client.get(f"/runs/{run_id}")
    assert run_payload.status_code == 200
    assert run_payload.json()["kind"] == "validation"


def test_monte_carlo_validation_uses_actual_trial_results(monkeypatch) -> None:
    scripted_results = iter(
        [
            {
                "iteration": 1,
                "raw_mean_miss_distance_m": 0.8,
                "ekf_mean_miss_distance_m": 0.3,
                "raw_success": False,
                "ekf_success": True,
                "num_targets": 2,
                "raw_by_target": {"Target_1": 0.9, "Target_2": 0.7},
                "ekf_by_target": {"Target_1": 0.4, "Target_2": 0.2},
                "raw_kill_probability": 0.1,
                "ekf_kill_probability": 0.8,
            },
            {
                "iteration": 2,
                "raw_mean_miss_distance_m": 0.4,
                "ekf_mean_miss_distance_m": 0.6,
                "raw_success": True,
                "ekf_success": False,
                "num_targets": 2,
                "raw_by_target": {"Target_1": 0.2, "Target_2": 0.6},
                "ekf_by_target": {"Target_1": 0.3, "Target_2": 0.9},
                "raw_kill_probability": 0.6,
                "ekf_kill_probability": 0.4,
            },
        ]
    )

    from drone_interceptor.simulation import airsim_manager as airsim_manager_module

    monkeypatch.setattr(
        airsim_manager_module,
        "_run_validation_trial",
        lambda args: next(scripted_results),
    )

    summary = AirSimMissionManager(connect=False).run_monte_carlo_validation(
        iterations=2,
        num_targets=2,
        use_multiprocessing=False,
    )

    assert isclose(summary.raw_success_rate, 0.5)
    assert isclose(summary.ekf_success_rate, 0.5)
    assert isclose(summary.raw_mean_miss_distance_m, 0.6)
    assert isclose(summary.ekf_mean_miss_distance_m, 0.45)
    per_target = {row["target"]: row for row in summary.per_target_summary}
    assert isclose(per_target["Target_1"]["raw_success_rate"], 0.5)
    assert isclose(per_target["Target_1"]["ekf_success_rate"], 1.0)
    assert isclose(per_target["Target_2"]["raw_success_rate"], 0.0)
    assert isclose(per_target["Target_2"]["ekf_success_rate"], 0.5)
