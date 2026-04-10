from pathlib import Path
import sys
from math import isclose

import streamlit as st
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.dashboard import app as dashboard_app
from drone_interceptor.simulation.telemetry_api import app as telemetry_app


client = TestClient(telemetry_app)


def _request_via_test_client(
    host: str,
    port: int,
    path: str,
    payload: dict | None = None,
    timeout_s: float = 5.0,
) -> dict:
    del host, port, timeout_s
    if payload is None:
        response = client.get(path)
    else:
        response = client.post(path, json=payload)
    response.raise_for_status()
    return response.json()


def _fast_validation_via_test_client(host: str, port: int, payload: dict) -> tuple[dict, object]:
    del host, port
    tuned_payload = dict(payload)
    tuned_payload["iterations"] = 10
    response = client.post("/validate", json=tuned_payload)
    response.raise_for_status()
    report = response.json()
    return report, dashboard_app.pd.DataFrame(report.get("scenario_results", []))


def test_dashboard_sync_backend_service_consumes_fastapi_validation(monkeypatch) -> None:
    st.session_state.clear()
    monkeypatch.setattr(dashboard_app, "_backend_request_json", _request_via_test_client)

    controls = {
        "target_speed": 6.0,
        "interceptor_speed": 20.0,
        "drift_rate": 0.3,
        "noise_level": 0.45,
        "num_targets": 3,
        "latency_ms": 80.0,
        "packet_loss_rate": 0.2,
        "use_ekf": True,
        "use_ekf_anti_spoofing": True,
        "connect_airsim": False,
        "run_validation_suite": True,
        "scenario_type": "normal",
        "compare_without_drift": True,
        "animate_frontend": False,
        "playback_fps_hz": 45.78,
        "run_clicked": True,
        "backend_host": "127.0.0.1",
        "backend_port": 8765,
        "backend_live_mode": True,
        "backend_preflight": False,
        "backend_start": False,
        "backend_validate": False,
        "deploy_clicked": False,
    }

    backend_state = dashboard_app._sync_backend_service(controls)

    assert backend_state is not None
    assert backend_state["snapshot"]["status"] in {"preparing", "running", "complete"}
    assert backend_state["status_endpoint"]["service_status"] == "ok"
    validation = st.session_state["backend_validation"]
    assert st.session_state["backend_deploy_ready"] is bool(validation["validation_success"])
    assert len(st.session_state["results_table"]) == 3
    assert len(st.session_state["scenario_results_df"]) == 3
    assert len(st.session_state["backend_validation_frame"]) == 3
    result_columns = set(st.session_state["results_table"].columns)
    assert {
        "target",
        "ekf_success_rate",
        "interception_time_s",
        "rmse_m",
        "mission_success_probability",
        "guidance_efficiency_mps2",
        "spoofing_variance",
        "compute_latency_ms",
        "energy_consumption_j",
    }.issubset(result_columns)
    assert len(validation["per_target_summary"]) == 3
    assert 0.0 <= validation["ekf_success_rate"] <= 1.0
    assert isclose(
        float(validation["ekf_success_rate"]),
        sum(float(row["ekf_success_rate"]) for row in validation["per_target_summary"])
        / len(validation["per_target_summary"]),
        rel_tol=1e-9,
    )
    mission_history = st.session_state.get("mission_history", [])
    assert isinstance(mission_history, list) and mission_history
    assert "readiness_score" in mission_history[-1]
    assert "quality_gate_passed" in mission_history[-1]


def test_dashboard_artifact_video_candidates_prioritize_day9(monkeypatch, tmp_path) -> None:
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    day9 = outputs_dir / "day9_dp5_demo.mp4"
    final_demo = outputs_dir / "final_demo.mp4"
    day9.write_bytes(b"day9")
    final_demo.write_bytes(b"final")
    monkeypatch.setattr(dashboard_app, "OUTPUTS", outputs_dir)
    st.session_state.clear()

    candidates = dashboard_app._artifact_video_candidates()

    assert candidates[0] == day9
    assert final_demo in candidates


def test_results_frame_is_backend_derived_and_fully_populated() -> None:
    st.session_state.clear()
    st.session_state["live_control_state"] = {"num_targets": 2}
    replay = dashboard_app.MissionReplay(
        frames=tuple(),
        validation={},
        map_frame=dashboard_app.pd.DataFrame(),
        distance_frame=dashboard_app.pd.DataFrame(),
        safe_intercepts=0,
    )
    validation = {
        "per_target_summary": [
            {
                "target": "Target_1",
                "ekf_success_rate": 0.82,
                "rmse": 0.31,
                "interception_time_s": 4.15,
                "mission_success_probability": 0.78,
                "guidance_efficiency_mps2": 1.24,
                "spoofing_variance": 0.44,
                "compute_latency_ms": 102.1,
                "energy_consumption_j": 62.5,
            },
            {
                "target": "Target_2",
                "ekf_success_rate": 0.76,
                "rmse": 0.38,
                "interception_time_s": 4.55,
                "mission_success_probability": 0.73,
                "guidance_efficiency_mps2": 1.11,
                "spoofing_variance": 0.49,
                "compute_latency_ms": 103.2,
                "energy_consumption_j": 59.2,
            },
        ]
    }
    st.session_state["backend_validation"] = validation
    frame = dashboard_app._build_day8_target_results_frame(replay, None)
    assert len(frame) == 2
    assert list(frame["target"]) == ["Target_1", "Target_2"]
    assert not frame.isna().any().any()
    assert dashboard_app._results_ready_for_display({"snapshot": {"status": "complete"}}, frame) is True


def test_resolve_dashboard_state_populates_benchmark_on_run_click() -> None:
    st.session_state.clear()
    controls = {
        "target_speed": 6.0,
        "interceptor_speed": 20.0,
        "drift_rate": 0.3,
        "noise_level": 0.45,
        "scenario_type": "normal",
        "compare_without_drift": True,
        "run_clicked": False,
    }
    _, benchmark_frame_initial, _ = dashboard_app._resolve_dashboard_state(controls)
    assert benchmark_frame_initial.empty

    controls["run_clicked"] = True
    _, benchmark_frame_run, _ = dashboard_app._resolve_dashboard_state(controls)
    assert not benchmark_frame_run.empty
    assert {"scenario", "success", "interception_time_s", "rmse_m"}.issubset(set(benchmark_frame_run.columns))


def test_live_analytics_frame_falls_back_to_results_table() -> None:
    st.session_state.clear()
    st.session_state["results_table"] = dashboard_app.pd.DataFrame(
        [
            {
                "target": "Target_1",
                "ekf_success_rate": 82.0,
                "interception_time_s": 4.1,
                "rmse_m": 0.31,
                "measured_rmse_m": 0.31,
                "mission_success_probability": 78.0,
                "guidance_efficiency_mps2": 1.22,
                "spoofing_variance": 0.45,
                "compute_latency_ms": 102.0,
                "energy_consumption_j": 61.0,
            }
        ]
    )
    st.session_state["live_control_state"] = {"noise_level": 0.45}
    st.session_state["backend_state"] = {
        "snapshot": {
            "detection_fps": 41.5,
            "active_distance_m": 12.0,
            "targets": [{"target_id": "Target_1", "distance_m": 11.2}],
        }
    }
    frame = dashboard_app._build_live_analytics_frame_from_results()
    assert len(frame) == 1
    assert frame.iloc[0]["scenario"] == "Target_1"
    assert frame.iloc[0]["success"] == "YES"
