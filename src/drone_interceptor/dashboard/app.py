from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from collections import Counter
from urllib import error as urlerror
from urllib import request as urlrequest

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from plotly.subplots import make_subplots

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[3]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from drone_interceptor.config import load_config
from drone_interceptor.constraints import ConstraintStatus, load_constraint_envelope
from drone_interceptor.control.controller import InterceptionController
from drone_interceptor.navigation.drift_model.dp5_safe import AttackProfile, DP5CoordinateSpoofingToolkit
from drone_interceptor.navigation.state_estimator import GPSIMUKalmanFusion, simulate_gps_with_drift
from drone_interceptor.optimization.cost import InterceptionCostModel
from drone_interceptor.perception.detector import TargetDetector, score_weighted_detection_targets
from drone_interceptor.planning.planner import InterceptPlanner
from drone_interceptor.prediction.predictor import TargetPredictor
from drone_interceptor.simulation.airsim_manager import AirSimMissionManager, MissionReplay, MonteCarloSummary, local_position_to_lla
from drone_interceptor.simulation.environment import DroneInterceptionEnv
from drone_interceptor.tracking.tracker import TargetTracker
from drone_interceptor.types import Detection, TargetState
from drone_interceptor.validation.day4 import _apply_day4_tuning


ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = ROOT / "outputs"
LOGS = ROOT / "logs"
LIVE_REPLAY_FPS = 45.78
UI_BUILD = "2026.04.09.9"
DECK_MAP_PROVIDER = "carto"
DECK_MAP_STYLE = "dark"
DEFAULT_BACKEND_HOST = "127.0.0.1"
DEFAULT_BACKEND_PORT = 8000
BACKEND_BOOTSTRAP_SLEEP_S = 2.0
BACKEND_PROCESS_REF = None  # Global reference to background backend process
BACKEND_BOOTSTRAP_DONE = False
SOLIDWORKS_MODEL_FALLBACK = Path(r"C:\Users\hp\Downloads\Honeywell drone.glb")


def _is_backend_reachable(host: str, port: int, timeout_s: float = 2.0) -> bool:
    """Check if backend is reachable on the given host:port."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout_s)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def _start_backend_process(
    host: str = DEFAULT_BACKEND_HOST,
    port: int = DEFAULT_BACKEND_PORT,
    startup_wait_s: float = BACKEND_BOOTSTRAP_SLEEP_S,
    session_flag: bool = True,
) -> bool:
    """Attempt to start the telemetry API backend via subprocess if not already running."""
    global BACKEND_PROCESS_REF

    if _is_backend_reachable(host, port):
        return True  # Already running

    try:
        src_dir = ROOT / "src"
        module_path = src_dir / "drone_interceptor" / "simulation" / "telemetry_api.py"
        if not module_path.exists():
            return False

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{src_dir}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(src_dir)
        )

        popen_kwargs: dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "env": env,
            "cwd": str(ROOT),
        }
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_kwargs["start_new_session"] = True

        BACKEND_PROCESS_REF = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "drone_interceptor.simulation.telemetry_api:app",
                "--host",
                str(host),
                "--port",
                str(int(port)),
            ],
            **popen_kwargs,
        )

        startup_wait_s = max(float(startup_wait_s), 0.0)
        deadline = time.time() + startup_wait_s
        while time.time() < deadline:
            if _is_backend_reachable(host, port):
                if session_flag:
                    st.session_state["backend_auto_started"] = True
                return True
            time.sleep(0.1)
        if _is_backend_reachable(host, port):
            if session_flag:
                st.session_state["backend_auto_started"] = True
            return True

        return False
    except Exception:
        return False


def _bootstrap_backend_service() -> None:
    global BACKEND_BOOTSTRAP_DONE
    if BACKEND_BOOTSTRAP_DONE:
        return
    BACKEND_BOOTSTRAP_DONE = True
    if _is_backend_reachable(DEFAULT_BACKEND_HOST, DEFAULT_BACKEND_PORT, timeout_s=0.5):
        return
    _start_backend_process(
        host=DEFAULT_BACKEND_HOST,
        port=DEFAULT_BACKEND_PORT,
        startup_wait_s=BACKEND_BOOTSTRAP_SLEEP_S,
        session_flag=False,
    )


_bootstrap_backend_service()


def main() -> None:
    st.set_page_config(page_title="Drone Interceptor Mission Control", layout="wide")
    set_custom_style()
    _refresh_frontend_build_state()

    st.title("Drone Interceptor Battle Management System")
    st.caption(
        "High-fidelity battle management frontend for UAV interception. "
        "Adjust mission parameters, inspect the autonomy stack, and review real-time multi-target telemetry, spoofing resilience, and mission outputs."
    )
    st.caption("Change drift, see the path change, and watch the interception and redirect sequence update in real time.")
    st.caption(f"Frontend build: {UI_BUILD}")

    controls = _control_panel()
    # ── TEST MODE banner ──────────────────────────────────────────────────────
    if bool(controls.get("enable_spoofing", False)):
        st.markdown(
            """
            <div class="global-alert-banner">
              <span class="alert-tag">Critical System Alert</span>
              <span class="alert-text">SPOOF INJECTION MODE: ON</span>
              <span class="alert-subtext">Adversarial stress profile active. EKF anti-spoofing remains enabled.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    _sync_live_control_state(controls)
    
    # ── AUTO HEALTH CHECK & BACKEND SUPERVISION ────────────────────────────
    backend_host = str(controls.get("backend_host", DEFAULT_BACKEND_HOST))
    backend_port = int(controls.get("backend_port", DEFAULT_BACKEND_PORT))
    if not st.session_state.get("backend_health_checked", False):
        if not _is_backend_reachable(backend_host, backend_port):
            with st.spinner("Backend not reachable. Auto-starting..."):
                if _start_backend_process(
                    host=backend_host,
                    port=backend_port,
                    startup_wait_s=BACKEND_BOOTSTRAP_SLEEP_S,
                    session_flag=True,
                ):
                    st.session_state["backend_auto_started"] = True
                else:
                    st.warning(
                        "Could not auto-start backend. Please ensure the backend service is running on port 8000, "
                        "or check that the telemetry_api.py is accessible."
                    )
        st.session_state["backend_health_checked"] = True
    
    simulation, benchmark_frame, replay_requested = _resolve_dashboard_state(controls)
    backend_state = _sync_backend_service(controls, simulation)

    day8_replay, day8_validation = _resolve_day8_state(controls)
    header_placeholder = st.empty()
    _render_tactical_header(
        header_placeholder=header_placeholder,
        simulation=simulation,
        status=str(st.session_state.get("dashboard_status", "STOPPED")),
        active_stage=str(st.session_state.get("dashboard_active_stage", "Detection")),
    )

    tabs = st.tabs(["Overview", "3D Mission View", "Map", "Live Analytics", "Architecture & Hardware"])
    with tabs[0]:
        st.markdown('<div class="tactical-panel-label">Overview</div>', unsafe_allow_html=True)
        metric_columns = st.columns(4)
        metric_placeholders = [column.empty() for column in metric_columns]
        _render_summary_metrics(metric_placeholders, simulation)
        overview_left, overview_center, overview_right = st.columns([1.15, 1.0, 0.85], gap="large")
        with overview_left:
            st.markdown('<div class="tactical-panel-label">Live Simulation Panel [Trajectory]</div>', unsafe_allow_html=True)
            live_panel_placeholder = st.empty()
        with overview_center:
            st.markdown('<div class="tactical-panel-label">Mission Playback [Visual Feed]</div>', unsafe_allow_html=True)
            mission_demo_placeholder = st.empty()
            st.markdown('<div class="tactical-panel-label">Scenario Results [Per Target]</div>', unsafe_allow_html=True)
            st.caption("These results are computed per target from the current mission and validation settings.")
            scenario_results_placeholder = st.empty()
        with overview_right:
            st.markdown('<div class="tactical-panel-label">Backend Mission Service [Health]</div>', unsafe_allow_html=True)
            backend_status_placeholder = st.empty()
    with tabs[1]:
        st.markdown('<div class="tactical-panel-label">Advanced 3D Mission View</div>', unsafe_allow_html=True)
        three_d_placeholder = st.empty()
    with tabs[2]:
        st.markdown('<div class="tactical-panel-label">Tactical Map</div>', unsafe_allow_html=True)
        map_placeholder = st.empty()
        telemetry_placeholder = st.empty()
        st.markdown('<div class="tactical-panel-label">AirSim Swarm Map</div>', unsafe_allow_html=True)
        day8_map_placeholder = st.empty()
        st.markdown('<div class="tactical-panel-label">Swarm Telemetry Stream</div>', unsafe_allow_html=True)
        day8_telemetry_placeholder = st.empty()
        st.markdown('<div class="tactical-panel-label">Mission HUD</div>', unsafe_allow_html=True)
        backend_stream_placeholder = st.empty()
    with tabs[3]:
        st.markdown('<div class="tactical-panel-label">Live Analytics</div>', unsafe_allow_html=True)
        realtime_placeholder = st.empty()
        analytics_placeholder = st.empty()
        st.markdown('<div class="tactical-panel-label">Post-Flight Analysis</div>', unsafe_allow_html=True)
        comparison_placeholder = st.empty()
        st.markdown('<div class="tactical-panel-label">Multi-UAV Validation</div>', unsafe_allow_html=True)
        day8_distance_placeholder = st.empty()
        day8_validation_plot_placeholder = st.empty()
        day8_validation_placeholder = st.empty()
        st.markdown('<div class="tactical-panel-label">Backend Architecture</div>', unsafe_allow_html=True)
        day8_architecture_placeholder = st.empty()
    with tabs[4]:
        st.markdown('<div class="tactical-panel-label">Architecture and Hardware</div>', unsafe_allow_html=True)
        _render_ops_and_architecture_panel(controls=controls, backend_state=backend_state)

    if replay_requested and controls["animate_frontend"]:
        _run_live_simulation_callback(
            simulation=simulation,
            benchmark_frame=benchmark_frame,
            day8_replay=day8_replay,
            day8_validation=day8_validation,
            header_placeholder=header_placeholder,
            metric_placeholders=metric_placeholders,
            three_d_placeholder=three_d_placeholder,
            live_panel_placeholder=live_panel_placeholder,
            map_placeholder=map_placeholder,
            telemetry_placeholder=telemetry_placeholder,
            day8_map_placeholder=day8_map_placeholder,
            day8_telemetry_placeholder=day8_telemetry_placeholder,
            realtime_placeholder=realtime_placeholder,
            comparison_placeholder=comparison_placeholder,
            scenario_results_placeholder=scenario_results_placeholder,
            mission_demo_placeholder=mission_demo_placeholder,
            analytics_placeholder=analytics_placeholder,
            day8_distance_placeholder=day8_distance_placeholder,
            day8_validation_plot_placeholder=day8_validation_plot_placeholder,
            day8_validation_placeholder=day8_validation_placeholder,
            day8_architecture_placeholder=day8_architecture_placeholder,
            backend_status_placeholder=backend_status_placeholder,
            backend_stream_placeholder=backend_stream_placeholder,
            backend_state=backend_state,
            backend_host=controls["backend_host"],
            backend_port=controls["backend_port"],
            playback_fps_hz=controls["playback_fps_hz"],
        )
        st.session_state["dashboard_replay_requested"] = False
    else:
        _render_static_frontend(
            simulation=simulation,
            benchmark_frame=benchmark_frame,
            day8_replay=day8_replay,
            day8_validation=day8_validation,
            header_placeholder=header_placeholder,
            status=str(st.session_state.get("dashboard_status", "STOPPED")),
            three_d_placeholder=three_d_placeholder,
            live_panel_placeholder=live_panel_placeholder,
            map_placeholder=map_placeholder,
            telemetry_placeholder=telemetry_placeholder,
            day8_map_placeholder=day8_map_placeholder,
            day8_telemetry_placeholder=day8_telemetry_placeholder,
            realtime_placeholder=realtime_placeholder,
            comparison_placeholder=comparison_placeholder,
            scenario_results_placeholder=scenario_results_placeholder,
            mission_demo_placeholder=mission_demo_placeholder,
            analytics_placeholder=analytics_placeholder,
            day8_distance_placeholder=day8_distance_placeholder,
            day8_validation_plot_placeholder=day8_validation_plot_placeholder,
            day8_validation_placeholder=day8_validation_placeholder,
            day8_architecture_placeholder=day8_architecture_placeholder,
            backend_status_placeholder=backend_status_placeholder,
            backend_stream_placeholder=backend_stream_placeholder,
            backend_state=backend_state,
            backend_host=controls["backend_host"],
            backend_port=controls["backend_port"],
        )

def _control_panel() -> dict[str, Any]:
    with st.sidebar:
        st.header("Interactive Control Panel")
        st.caption("Run the mission from the current inputs and let the frontend build plots as the workflow advances.")
        with st.form("mission_controls", clear_on_submit=False):
            target_speed = st.slider("Target Speed [m/s]", min_value=3.0, max_value=12.0, value=6.0, step=0.1)
            interceptor_speed = st.slider("Interceptor Speed [m/s]", min_value=12.0, max_value=24.0, value=20.0, step=0.5)
            interceptor_mass_kg = st.slider("Interceptor Mass [kg]", min_value=1.0, max_value=25.0, value=6.5, step=0.1)
            drift_rate = st.slider("Drift Rate [m/s]", min_value=0.2, max_value=0.5, value=0.3, step=0.01)
            noise_level = st.slider("Noise Level", min_value=0.1, max_value=1.5, value=0.45, step=0.05)
            num_targets = st.number_input("Number of Targets", min_value=1, max_value=10, value=3, step=1)
            latency_ms = st.slider("Telemetry Latency [ms]", min_value=0.0, max_value=400.0, value=80.0, step=10.0)
            packet_loss_rate = st.slider("Packet Loss Rate", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
            link_snr_db = st.slider("Link SNR [dB]", min_value=5.0, max_value=45.0, value=28.0, step=0.5)
            packet_loss_k = st.slider("Packet Loss Model k", min_value=0.01, max_value=0.50, value=0.12, step=0.01)
            packet_loss_alpha = st.slider("Packet Loss Model alpha", min_value=1.0, max_value=3.5, value=1.8, step=0.1)
            st.toggle("Use EKF Anti-Spoofing", value=True, disabled=True)
            use_ekf = True
            connect_airsim = False
            run_validation_suite = st.toggle("Run 10x Validation", value=False)
            st.toggle(
                "Enable Spoof Injection (Test Mode)",
                value=True,
                disabled=True,
                help="Inject adversarial noise / frame corruption into the pipeline while the mission runs. "
                     "Feature-flagged: leave OFF in production.",
            )
            enable_spoofing = True
            scenario_type = st.selectbox(
                "Scenario Type",
                options=["normal", "fast", "noisy", "drift", "zigzag"],
                index=0,
            )
            compare_without_drift = st.toggle("Compare With / Without Drift", value=True)
            animate_frontend = st.toggle("Animate Frontend Replay", value=True)
            playback_fps_hz = st.slider("Replay FPS [Hz]", min_value=10.0, max_value=60.0, value=LIVE_REPLAY_FPS, step=0.5)
            run_clicked = st.form_submit_button("Run Live Simulation", use_container_width=True, type="primary")
        with st.expander("How Inputs Change Output", expanded=False):
            st.markdown(
                "- `Use EKF Anti-Spoofing`: defensive estimator/gating; changes EKF success %, RMSE, confidence, and mission success probability.\n"
                "- `Enable Spoof Injection`: adversarial disturbance source; changes spoof active/detected counts, spoofing variance, and readiness posture.\n"
                "- `Run 10x Validation`: does not change single-run mission physics; runs Monte Carlo checks and updates validation/deploy-gate confidence."
            )
            st.info(
                "`EKF Anti-Spoofing` and `Spoof Injection` are different controls: one defends, one attacks. "
                "This profile locks both enabled to continuously test resilience."
            )
        with st.expander("Overview vs Architecture Tabs", expanded=False):
            st.markdown(
                "- `Overview`: mission-output bound (`/run_mission` + snapshot). Results table is released after full per-target metrics are populated.\n"
                "- `Architecture & Hardware`: system model/introspection (equations, ROS2 roles, hardware links, spoof A/B diagnostics). It is not a second mission engine."
            )
        run_seed = int(st.session_state.get("mission_run_seed", int(time.time() * 1000.0) % 1_000_000_000))
        if run_clicked:
            run_seed = int(time.time() * 1000.0) % 1_000_000_000
            st.session_state["mission_run_seed"] = run_seed
        st.markdown("### Backend Service")
        backend_host = st.text_input("FastAPI Host", value=DEFAULT_BACKEND_HOST, key="backend_host_input")
        backend_port = int(
            st.number_input(
                "FastAPI Port",
                min_value=1,
                max_value=65535,
                value=DEFAULT_BACKEND_PORT,
                step=1,
                key="backend_port_input",
            )
        )
        backend_live_mode = st.toggle("Backend Live HUD", value=True)
        st.caption("Note: The 'Run Live Simulation' button will automatically trigger preflight, start mission, and populate results.")
        backend_deploy_ready = bool(st.session_state.get("backend_deploy_ready", False))
        deploy_clicked = st.button("Deploy Mission", use_container_width=True, disabled=not backend_deploy_ready, type="primary")
        st.markdown(
            """
            <div class="stage-card">
              <strong>Platform Stages</strong><br/>
              Detection → Tracking → Interception → Drift Applied → Path Changes → Target Redirected
            </div>
            """,
            unsafe_allow_html=True,
        )
    return {
        "target_speed": target_speed,
        "interceptor_speed": interceptor_speed,
        "interceptor_mass_kg": float(interceptor_mass_kg),
        "drift_rate": drift_rate,
        "noise_level": noise_level,
        "num_targets": int(num_targets),
        "latency_ms": float(latency_ms),
        "packet_loss_rate": float(packet_loss_rate),
        "link_snr_db": float(link_snr_db),
        "packet_loss_k": float(packet_loss_k),
        "packet_loss_alpha": float(packet_loss_alpha),
        "use_ekf": bool(use_ekf),
        "use_ekf_anti_spoofing": bool(use_ekf),
        "connect_airsim": bool(connect_airsim),
        "run_validation_suite": bool(run_validation_suite),
        "enable_spoofing": bool(enable_spoofing),
        "scenario_type": scenario_type,
        "compare_without_drift": compare_without_drift,
        "animate_frontend": animate_frontend,
        "playback_fps_hz": playback_fps_hz,
        "run_seed": int(run_seed),
        "run_clicked": run_clicked,
        "backend_host": backend_host,
        "backend_port": backend_port,
        "backend_live_mode": bool(backend_live_mode),
        "backend_preflight": False,
        "backend_start": False,
        "backend_validate": False,
        "deploy_clicked": bool(deploy_clicked),
    }


def _run_local_python_script(script_path: Path, args: list[str], timeout_s: float = 120.0) -> dict[str, Any]:
    if not script_path.exists():
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Script not found: {script_path}",
            "command": [str(script_path), *args],
        }
    command = [sys.executable, str(script_path), *args]
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=max(float(timeout_s), 5.0),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": -2,
            "stdout": (exc.stdout or "")[:5000],
            "stderr": f"Timeout after {timeout_s:.1f}s",
            "command": command,
        }
    return {
        "ok": completed.returncode == 0,
        "returncode": int(completed.returncode),
        "stdout": (completed.stdout or "")[:15000],
        "stderr": (completed.stderr or "")[:8000],
        "command": command,
    }


def _run_spoof_ab_comparison(host: str, port: int, controls: dict[str, Any]) -> pd.DataFrame:
    base_payload = _build_backend_payload(controls)
    rows: list[dict[str, Any]] = []
    for label, enable_spoofing in (("Spoof OFF", False), ("Spoof ON", True)):
        payload = dict(base_payload)
        payload["enable_spoofing"] = bool(enable_spoofing)
        response = _backend_request_json(host, port, "/run_mission", payload, timeout_s=220.0)
        if response.get("workflow_status") != "success":
            rows.append(
                {
                    "scenario": label,
                    "status": "ERROR",
                    "error": str(response.get("error", "unknown")),
                }
            )
            continue
        mission_summary = response.get("mission_summary", {})
        mission_insights = response.get("mission_insights", {})
        snapshot = response.get("snapshot", {}) if isinstance(response, dict) else {}
        global_metrics = mission_insights.get("global_metrics", {}) if isinstance(mission_insights, dict) else {}
        readiness = mission_insights.get("command_readiness", {}) if isinstance(mission_insights, dict) else {}
        rows.append(
            {
                "scenario": label,
                "status": "SUCCESS",
                "targets": int(mission_summary.get("target_count", 0)),
                "ekf_success_pct": round(float(mission_summary.get("ekf_success_rate", 0.0)) * 100.0, 2),
                "mean_success_prob_pct": round(float(global_metrics.get("mean_mission_success_probability", 0.0)) * 100.0, 2),
                "spoof_active_count": int(global_metrics.get("active_spoofing_count", 0)),
                "spoof_detect_count": int(global_metrics.get("spoofing_detected_count", 0)),
                "spoof_detect_rate_pct": round(float(global_metrics.get("spoofing_detection_rate", 0.0)) * 100.0, 2),
                "kill_probability_pct": round(float(snapshot.get("kill_probability", 0.0)) * 100.0, 2),
                "kill_probability_target": str(
                    snapshot.get("kill_probability_target_id", snapshot.get("active_target", "n/a"))
                ),
                "closest_approach_m": round(
                    float(snapshot.get("closest_approach_m", snapshot.get("active_distance_m", 0.0))),
                    3,
                ),
                "closest_approach_target": str(
                    snapshot.get("closest_approach_target_id", snapshot.get("active_target", "n/a"))
                ),
                "quality_gate": "PASS" if bool(readiness.get("quality_gate_passed", False)) else "HOLD",
                "deploy_gate": "PASS" if bool(readiness.get("deployment_gate_passed", False)) else "HOLD",
                "readiness_score": round(float(readiness.get("readiness_score", 0.0)), 2),
            }
        )
    return pd.DataFrame(rows)


def _run_spoof_toggle_verification(
    host: str,
    port: int,
    controls: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _run_spoof_ab_comparison(host=host, port=port, controls=controls)
    verdict = {
        "passed": False,
        "checks": {},
        "message": "Verification could not run.",
    }
    if frame.empty or "scenario" not in frame.columns:
        return frame, verdict

    by_label = {str(row.get("scenario", "")): row for _, row in frame.iterrows()}
    off_row = by_label.get("Spoof OFF")
    on_row = by_label.get("Spoof ON")
    if off_row is None or on_row is None:
        verdict["message"] = "Missing Spoof OFF/ON rows."
        return frame, verdict

    off_active = int(_safe_float(off_row.get("spoof_active_count", 0), 0))
    off_detect = int(_safe_float(off_row.get("spoof_detect_count", 0), 0))
    on_active = int(_safe_float(on_row.get("spoof_active_count", 0), 0))
    on_detect = int(_safe_float(on_row.get("spoof_detect_count", 0), 0))
    off_targets = int(_safe_float(off_row.get("targets", 0), 0))
    on_targets = int(_safe_float(on_row.get("targets", 0), 0))
    expected_targets = max(int(controls.get("num_targets", 1)), 1)

    checks = {
        "off_has_no_spoof_activity": bool(off_active == 0 and off_detect == 0),
        "on_has_spoof_activity": bool(on_active > 0 or on_detect > 0),
        "target_count_preserved": bool(off_targets == expected_targets and on_targets == expected_targets),
    }
    passed = bool(all(checks.values()))
    message = (
        "PASS: spoof toggle behavior is deterministic and consistent."
        if passed
        else "HOLD: spoof toggle behavior deviates from expected OFF/ON semantics."
    )
    verdict = {
        "passed": passed,
        "checks": checks,
        "message": message,
        "off_active": off_active,
        "off_detect": off_detect,
        "on_active": on_active,
        "on_detect": on_detect,
    }
    return frame, verdict


def _run_spoof_ab_target_deltas(
    host: str,
    port: int,
    controls: dict[str, Any],
) -> pd.DataFrame:
    base_payload = _build_backend_payload(controls)
    responses: dict[str, dict[str, Any]] = {}
    for label, enable_spoofing in (("off", False), ("on", True)):
        payload = dict(base_payload)
        payload["enable_spoofing"] = bool(enable_spoofing)
        response = _backend_request_json(host, port, "/run_mission", payload, timeout_s=260.0)
        if response.get("workflow_status") != "success":
            return pd.DataFrame()
        responses[label] = response

    off_results = responses["off"].get("results", [])
    on_results = responses["on"].get("results", [])
    if not isinstance(off_results, list) or not isinstance(on_results, list):
        return pd.DataFrame()

    off_frame = pd.DataFrame(off_results)
    on_frame = pd.DataFrame(on_results)
    if off_frame.empty or on_frame.empty:
        return pd.DataFrame()

    keep_columns = [
        "target_id",
        "ekf_success_rate",
        "mission_success_probability",
        "rmse",
        "interception_time",
        "closest_approach_m",
        "compute_latency_ms",
        "packet_loss_probability",
        "link_snr_db",
        "spoofing_variance",
    ]
    off_frame = off_frame[[column for column in keep_columns if column in off_frame.columns]].copy()
    on_frame = on_frame[[column for column in keep_columns if column in on_frame.columns]].copy()
    merged = off_frame.merge(on_frame, on="target_id", suffixes=("_off", "_on"), how="outer")
    if merged.empty:
        return pd.DataFrame()

    numeric_metrics = [
        "ekf_success_rate",
        "mission_success_probability",
        "rmse",
        "interception_time",
        "closest_approach_m",
        "compute_latency_ms",
        "packet_loss_probability",
        "link_snr_db",
        "spoofing_variance",
    ]
    for metric in numeric_metrics:
        off_col = f"{metric}_off"
        on_col = f"{metric}_on"
        if off_col in merged.columns and on_col in merged.columns:
            merged[f"{metric}_delta"] = merged[on_col].astype(float) - merged[off_col].astype(float)

    merged = merged.rename(columns={"target_id": "target"})
    return merged


def _build_spoof_delta_heatmap(delta_frame: pd.DataFrame) -> go.Figure:
    if delta_frame.empty:
        figure = go.Figure()
        figure.update_layout(
            title="No spoof A/B delta data available.",
            template="plotly_dark",
            height=420,
        )
        return figure

    metric_map = {
        "ekf_success_rate_delta": "Delta EKF Success",
        "mission_success_probability_delta": "Delta Mission Success P",
        "rmse_delta": "Delta RMSE [m]",
        "closest_approach_m_delta": "Delta Closest Approach [m]",
        "interception_time_delta": "Delta Intercept Time [s]",
        "compute_latency_ms_delta": "Delta Compute Latency [ms]",
        "packet_loss_probability_delta": "Delta Packet Loss P",
        "link_snr_db_delta": "Delta Link SNR [dB]",
        "spoofing_variance_delta": "Delta Spoof Variance",
    }
    available_metrics = [column for column in metric_map.keys() if column in delta_frame.columns]
    if not available_metrics:
        figure = go.Figure()
        figure.update_layout(
            title="No numeric spoof delta metrics to render.",
            template="plotly_dark",
            height=420,
        )
        return figure

    targets = [str(value) for value in delta_frame["target"].tolist()]
    z_matrix = np.asarray(
        [
            [float(delta_frame.iloc[row_index][metric]) for metric in available_metrics]
            for row_index in range(len(delta_frame))
        ],
        dtype=float,
    )
    figure = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=[metric_map[metric] for metric in available_metrics],
            y=targets,
            colorscale="RdBu",
            zmid=0.0,
            colorbar=dict(title="ON - OFF"),
        )
    )
    figure.update_layout(
        title="Per-Target Spoof ON vs OFF Delta Heatmap",
        xaxis_title="Metric",
        yaxis_title="Target",
        template="plotly_dark",
        height=460,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return figure


def _run_profile_ab_sweep(
    host: str,
    port: int,
    controls: dict[str, Any],
) -> pd.DataFrame:
    profiles = [
        {"profile": "Nominal", "noise_std_m": 0.40, "drift_rate_mps": 0.25, "packet_loss_rate": 0.03, "link_snr_db": 32.0},
        {"profile": "Noisy", "noise_std_m": 1.10, "drift_rate_mps": 0.30, "packet_loss_rate": 0.06, "link_snr_db": 28.0},
        {"profile": "Drift Stress", "noise_std_m": 0.55, "drift_rate_mps": 0.48, "packet_loss_rate": 0.05, "link_snr_db": 26.0},
        {"profile": "Link Stress", "noise_std_m": 0.65, "drift_rate_mps": 0.35, "packet_loss_rate": 0.25, "link_snr_db": 18.0},
        {"profile": "Combined Stress", "noise_std_m": 1.20, "drift_rate_mps": 0.48, "packet_loss_rate": 0.28, "link_snr_db": 15.0},
    ]
    rows: list[dict[str, Any]] = []
    base_payload = _build_backend_payload(controls)
    for profile in profiles:
        scenario_data: dict[str, dict[str, Any]] = {}
        for label, spoof in (("off", False), ("on", True)):
            payload = dict(base_payload)
            payload.update(
                {
                    "noise_std_m": float(profile["noise_std_m"]),
                    "drift_rate_mps": float(profile["drift_rate_mps"]),
                    "packet_loss_rate": float(profile["packet_loss_rate"]),
                    "link_snr_db": float(profile["link_snr_db"]),
                    "enable_spoofing": bool(spoof),
                }
            )
            response = _backend_request_json(host, port, "/run_mission", payload, timeout_s=260.0)
            if response.get("workflow_status") != "success":
                scenario_data[label] = {}
                continue
            mission_insights = response.get("mission_insights", {})
            global_metrics = mission_insights.get("global_metrics", {}) if isinstance(mission_insights, dict) else {}
            readiness = mission_insights.get("command_readiness", {}) if isinstance(mission_insights, dict) else {}
            snapshot = response.get("snapshot", {}) if isinstance(response, dict) else {}
            scenario_data[label] = {
                "weighted_success": float(global_metrics.get("weighted_mission_success", 0.0)),
                "rmse_p95": float(global_metrics.get("rmse_p95_m", 0.0)),
                "packet_loss_obs": float(global_metrics.get("packet_loss_observed_rate", 0.0)),
                "spoof_detect_rate": float(global_metrics.get("spoofing_detection_rate", 0.0)),
                "quality_gate": bool(readiness.get("quality_gate_passed", False)),
                "kill_probability": float(snapshot.get("kill_probability", 0.0)),
            }
        off = scenario_data.get("off", {})
        on = scenario_data.get("on", {})
        if not off or not on:
            continue
        rows.append(
            {
                "profile": str(profile["profile"]),
                "weighted_success_off_pct": round(float(off.get("weighted_success", 0.0)) * 100.0, 2),
                "weighted_success_on_pct": round(float(on.get("weighted_success", 0.0)) * 100.0, 2),
                "weighted_success_delta_pct": round((float(on.get("weighted_success", 0.0)) - float(off.get("weighted_success", 0.0))) * 100.0, 2),
                "rmse_p95_delta_m": round(float(on.get("rmse_p95", 0.0)) - float(off.get("rmse_p95", 0.0)), 4),
                "packet_loss_observed_delta_pct": round((float(on.get("packet_loss_obs", 0.0)) - float(off.get("packet_loss_obs", 0.0))) * 100.0, 2),
                "spoof_detect_rate_on_pct": round(float(on.get("spoof_detect_rate", 0.0)) * 100.0, 2),
                "kill_probability_delta_pct": round((float(on.get("kill_probability", 0.0)) - float(off.get("kill_probability", 0.0))) * 100.0, 2),
                "quality_gate_off": "PASS" if bool(off.get("quality_gate", False)) else "HOLD",
                "quality_gate_on": "PASS" if bool(on.get("quality_gate", False)) else "HOLD",
            }
        )
    return pd.DataFrame(rows)


def _build_profile_sweep_heatmap(profile_frame: pd.DataFrame) -> go.Figure:
    if profile_frame.empty:
        figure = go.Figure()
        figure.update_layout(
            title="No profile sweep data available.",
            template="plotly_dark",
            height=380,
        )
        return figure

    metric_columns = [
        "weighted_success_delta_pct",
        "rmse_p95_delta_m",
        "packet_loss_observed_delta_pct",
        "kill_probability_delta_pct",
        "spoof_detect_rate_on_pct",
    ]
    metric_labels = {
        "weighted_success_delta_pct": "Delta Weighted Success [%]",
        "rmse_p95_delta_m": "Delta RMSE P95 [m]",
        "packet_loss_observed_delta_pct": "Delta Packet Loss Obs [%]",
        "kill_probability_delta_pct": "Delta Kill Probability [%]",
        "spoof_detect_rate_on_pct": "Spoof Detect Rate ON [%]",
    }
    available_metrics = [metric for metric in metric_columns if metric in profile_frame.columns]
    z_matrix = np.asarray(
        [
            [float(profile_frame.iloc[row_idx][metric]) for metric in available_metrics]
            for row_idx in range(len(profile_frame))
        ],
        dtype=float,
    )
    figure = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=[metric_labels[metric] for metric in available_metrics],
            y=[str(value) for value in profile_frame["profile"].tolist()],
            colorscale="RdBu",
            zmid=0.0,
            colorbar=dict(title="Profile Delta"),
        )
    )
    figure.update_layout(
        title="Spoof ON/OFF Profile Sweep Heatmap",
        xaxis_title="Metric",
        yaxis_title="Profile",
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return figure


def _build_judge_sweep_cases() -> list[dict[str, Any]]:
    return [
        {"name": "nominal_a", "num_targets": 3, "noise_std_m": 0.35, "drift_rate_mps": 0.25, "latency_ms": 60.0, "packet_loss_rate": 0.02, "enable_spoofing": False},
        {"name": "nominal_b", "num_targets": 4, "noise_std_m": 0.45, "drift_rate_mps": 0.30, "latency_ms": 80.0, "packet_loss_rate": 0.05, "enable_spoofing": False},
        {"name": "nominal_spoof", "num_targets": 4, "noise_std_m": 0.45, "drift_rate_mps": 0.30, "latency_ms": 80.0, "packet_loss_rate": 0.05, "enable_spoofing": True},
        {"name": "noise_medium", "num_targets": 5, "noise_std_m": 0.85, "drift_rate_mps": 0.30, "latency_ms": 90.0, "packet_loss_rate": 0.05, "enable_spoofing": False},
        {"name": "noise_high", "num_targets": 6, "noise_std_m": 1.20, "drift_rate_mps": 0.30, "latency_ms": 90.0, "packet_loss_rate": 0.07, "enable_spoofing": False},
        {"name": "noise_high_spoof", "num_targets": 6, "noise_std_m": 1.20, "drift_rate_mps": 0.30, "latency_ms": 90.0, "packet_loss_rate": 0.07, "enable_spoofing": True},
        {"name": "drift_medium", "num_targets": 5, "noise_std_m": 0.45, "drift_rate_mps": 0.40, "latency_ms": 90.0, "packet_loss_rate": 0.05, "enable_spoofing": False},
        {"name": "drift_high", "num_targets": 6, "noise_std_m": 0.45, "drift_rate_mps": 0.48, "latency_ms": 100.0, "packet_loss_rate": 0.05, "enable_spoofing": False},
        {"name": "drift_high_spoof", "num_targets": 6, "noise_std_m": 0.45, "drift_rate_mps": 0.48, "latency_ms": 100.0, "packet_loss_rate": 0.05, "enable_spoofing": True},
        {"name": "latency_high", "num_targets": 7, "noise_std_m": 0.55, "drift_rate_mps": 0.33, "latency_ms": 220.0, "packet_loss_rate": 0.08, "enable_spoofing": False},
        {"name": "latency_extreme", "num_targets": 8, "noise_std_m": 0.65, "drift_rate_mps": 0.33, "latency_ms": 280.0, "packet_loss_rate": 0.08, "enable_spoofing": False},
        {"name": "packet_loss_high", "num_targets": 8, "noise_std_m": 0.65, "drift_rate_mps": 0.33, "latency_ms": 120.0, "packet_loss_rate": 0.28, "enable_spoofing": False},
        {"name": "stress_combo", "num_targets": 9, "noise_std_m": 1.00, "drift_rate_mps": 0.45, "latency_ms": 240.0, "packet_loss_rate": 0.20, "enable_spoofing": False},
        {"name": "stress_combo_spoof", "num_targets": 9, "noise_std_m": 1.00, "drift_rate_mps": 0.45, "latency_ms": 240.0, "packet_loss_rate": 0.20, "enable_spoofing": True},
        {"name": "max_stress_spoof", "num_targets": 10, "noise_std_m": 1.10, "drift_rate_mps": 0.48, "latency_ms": 260.0, "packet_loss_rate": 0.25, "enable_spoofing": True},
    ]


def _run_judge_sweep(
    host: str,
    port: int,
    controls: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_fields = [
        "target_id",
        "ekf_success_rate",
        "interception_time",
        "rmse",
        "mission_success_probability",
        "guidance_efficiency_mps2",
        "spoofing_variance",
        "compute_latency_ms",
        "energy_consumption_j",
    ]
    rows: list[dict[str, Any]] = []
    for case in _build_judge_sweep_cases():
        payload = _build_backend_payload(controls)
        payload.update(
            {
                "num_targets": int(case["num_targets"]),
                "target_ids": [f"Target_{index + 1}" for index in range(int(case["num_targets"]))],
                "noise_std_m": float(case["noise_std_m"]),
                "drift_rate_mps": float(case["drift_rate_mps"]),
                "latency_ms": float(case["latency_ms"]),
                "packet_loss_rate": float(case["packet_loss_rate"]),
                "enable_spoofing": bool(case["enable_spoofing"]),
            }
        )
        response = _backend_request_json(host, port, "/run_mission", payload, timeout_s=260.0)
        results = response.get("results", []) if isinstance(response, dict) else []
        mission_insights = response.get("mission_insights", {}) if isinstance(response, dict) else {}
        global_metrics = mission_insights.get("global_metrics", {}) if isinstance(mission_insights, dict) else {}
        readiness = mission_insights.get("command_readiness", {}) if isinstance(mission_insights, dict) else {}
        missing_fields = 0
        for result in results:
            if not isinstance(result, dict):
                missing_fields += len(required_fields)
                continue
            for field in required_fields:
                value = result.get(field)
                if value is None:
                    missing_fields += 1
        rows.append(
            {
                "case": str(case["name"]),
                "targets_req": int(case["num_targets"]),
                "targets_out": int(len(results)),
                "spoof_enabled": bool(case["enable_spoofing"]),
                "spoof_active_count": int(global_metrics.get("active_spoofing_count", 0)),
                "spoof_detect_count": int(global_metrics.get("spoofing_detected_count", 0)),
                "spoof_detect_rate_pct": round(float(global_metrics.get("spoofing_detection_rate", 0.0)) * 100.0, 2),
                "readiness_score": round(float(readiness.get("readiness_score", 0.0)), 2),
                "quality_gate": bool(readiness.get("quality_gate_passed", False)),
                "deploy_gate": bool(readiness.get("deployment_gate_passed", False)),
                "missing_fields": int(missing_fields),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, {
            "runs": 0,
            "target_mismatch": 0,
            "missing_metrics_runs": 0,
            "false_spoof_off_runs": 0,
            "quality_gate_pass_runs": 0,
            "deploy_gate_pass_runs": 0,
        }
    summary = {
        "runs": int(len(frame)),
        "target_mismatch": int((frame["targets_out"] != frame["targets_req"]).sum()),
        "missing_metrics_runs": int((frame["missing_fields"] > 0).sum()),
        "false_spoof_off_runs": int(((frame["spoof_enabled"] == False) & (frame["spoof_active_count"] > 0)).sum()),
        "quality_gate_pass_runs": int((frame["quality_gate"] == True).sum()),
        "deploy_gate_pass_runs": int((frame["deploy_gate"] == True).sum()),
    }
    runs = max(int(summary["runs"]), 1)
    score_components = {
        "target_match": 1.0 - (float(summary["target_mismatch"]) / runs),
        "metrics_completeness": 1.0 - (float(summary["missing_metrics_runs"]) / runs),
        "spoof_toggle_integrity": 1.0 - (float(summary["false_spoof_off_runs"]) / max(int((frame["spoof_enabled"] == False).sum()), 1)),
        "quality_gate_rate": float(summary["quality_gate_pass_runs"]) / runs,
        "deploy_gate_rate": float(summary["deploy_gate_pass_runs"]) / runs,
    }
    designathon_score = float(
        np.clip(
            10.0
            * (
                0.25 * score_components["target_match"]
                + 0.25 * score_components["metrics_completeness"]
                + 0.15 * score_components["spoof_toggle_integrity"]
                + 0.20 * score_components["quality_gate_rate"]
                + 0.15 * score_components["deploy_gate_rate"]
            ),
            0.0,
            10.0,
        )
    )
    summary["designathon_score"] = round(designathon_score, 2)
    summary["designathon_tier"] = (
        "Top Tier"
        if designathon_score >= 9.0
        else ("Competitive" if designathon_score >= 8.0 else "Needs Hardening")
    )
    return frame, summary


def _render_ops_and_architecture_panel(
    controls: dict[str, Any],
    backend_state: dict[str, Any] | None,
) -> None:
    st.subheader("Command Center and Final Architecture")
    tabs = st.tabs(
        [
            "Command Center",
            "Spoof Toggle Logic",
            "Jetson Deployment",
            "Final Architecture",
            "Hardware Requirements",
            "SolidWorks Guide",
            "Judge Sweep (15)",
        ]
    )
    backend_host = str(controls.get("backend_host", DEFAULT_BACKEND_HOST))
    backend_port = int(controls.get("backend_port", DEFAULT_BACKEND_PORT))

    with tabs[0]:
        st.caption("Run validation commands from inside the software. All commands below are defensive dry-run workflows.")
        st.code(
            "\n".join(
                [
                    "python test_spoof.py --jetson-host 192.168.55.1 --spoof-enable",
                    "python scripts/run_spoof_sitl_scenario.py --sim synthetic --steps 240 --spoof-enable",
                    "python -m drone_interceptor.ros2.vision_node --config configs/default.yaml",
                    "python -m drone_interceptor.ros2.spoof_node --config configs/default.yaml --spoof-enable",
                ]
            ),
            language="bash",
        )
        cmd_cols = st.columns(2)
        if cmd_cols[0].button("Run Spoof Diagnostic", key="cmd_run_spoof_diag"):
            with st.spinner("Running test_spoof.py..."):
                st.session_state["cmd_spoof_diag_output"] = _run_local_python_script(
                    script_path=ROOT / "test_spoof.py",
                    args=["--jetson-host", "127.0.0.1", "--spoof-enable"],
                    timeout_s=45.0,
                )
        if cmd_cols[1].button("Run SITL Spoof Scenario", key="cmd_run_sitl_spoof"):
            with st.spinner("Running SITL spoof scenario script..."):
                st.session_state["cmd_sitl_spoof_output"] = _run_local_python_script(
                    script_path=ROOT / "scripts" / "run_spoof_sitl_scenario.py",
                    args=["--sim", "synthetic", "--steps", "120", "--spoof-enable"],
                    timeout_s=90.0,
                )
        for key, label in (
            ("cmd_spoof_diag_output", "Spoof Diagnostic Output"),
            ("cmd_sitl_spoof_output", "SITL Scenario Output"),
        ):
            output = st.session_state.get(key)
            if isinstance(output, dict):
                st.markdown(f"**{label}**")
                st.write(f"Return code: {output.get('returncode')} | OK: {output.get('ok')}")
                stdout_text = str(output.get("stdout", "")).strip()
                stderr_text = str(output.get("stderr", "")).strip()
                if stdout_text:
                    st.code(stdout_text, language="text")
                if stderr_text:
                    st.code(stderr_text, language="text")

    with tabs[1]:
        st.caption("Toggle behavior is deterministic and backend-driven.")
        st.markdown(
            "- `Enable Spoof Injection = OFF`: backend sends `enable_spoofing=false`, spoof active/detected counters should remain zero.\n"
            "- `Enable Spoof Injection = ON`: spoof engine is activated, offsets are applied, and detection counters become non-zero if anti-spoofing catches events.\n"
            "- Quality/deploy gates are computed from mission outputs; no random UI logic."
        )
        st.markdown("#### Note Session: EKF Anti-Spoofing vs Spoof Injection")
        st.info(
            "EKF Anti-Spoofing is the defensive estimator and gating logic. "
            "Spoof Injection is the adversarial disturbance source. "
            "They are not the same switch: one attacks, one defends. "
            "This profile locks both ON to continuously evaluate resilience."
        )
        st.markdown("#### Input-to-Output Effect Matrix (Primary 3 Inputs)")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Input": "Enable Spoof Injection",
                        "Backend Effect": "enable_spoofing flag; spoof offsets injected into raw measurements",
                        "Observed Output Changes": "active_spoofing_count, spoofing_detected_count, spoof variance, kill probability shift",
                    },
                    {
                        "Input": "Use EKF Anti-Spoofing",
                        "Backend Effect": "innovation gating + filtered state correction + prediction hold under packet drop",
                        "Observed Output Changes": "EKF success rate, RMSE, confidence, mission_success_probability",
                    },
                    {
                        "Input": "Packet Loss Model alpha",
                        "Backend Effect": "modulates PL = 1-exp(-k*SNR/d^alpha); changes packet drop likelihood",
                        "Observed Output Changes": "packet_loss_probability, telemetry reliability, latency/risk, deployment gate",
                    },
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        status_endpoint = ((backend_state or {}).get("status_endpoint") or {}) if isinstance(backend_state, dict) else {}
        mission_insights = status_endpoint.get("mission_insights", {}) if isinstance(status_endpoint, dict) else {}
        global_metrics = mission_insights.get("global_metrics", {}) if isinstance(mission_insights, dict) else {}
        metrics_cols = st.columns(4)
        metrics_cols[0].metric("Spoof Active", int(global_metrics.get("active_spoofing_count", 0)))
        metrics_cols[1].metric("Spoof Detected", int(global_metrics.get("spoofing_detected_count", 0)))
        metrics_cols[2].metric("Detect Rate", f"{float(global_metrics.get('spoofing_detection_rate', 0.0)) * 100:.1f}%")
        metrics_cols[3].metric("Quality Gate", "PASS" if bool(((mission_insights.get("command_readiness", {}) if isinstance(mission_insights, dict) else {}).get("quality_gate_passed", False))) else "HOLD")
        if st.button("Run A/B Spoof Comparison", key="cmd_run_spoof_ab"):
            with st.spinner("Running back-to-back OFF/ON missions..."):
                st.session_state["spoof_ab_frame"] = _run_spoof_ab_comparison(
                    host=backend_host,
                    port=backend_port,
                    controls=controls,
                )
        if st.button("Verify Spoof Toggle Semantics", key="cmd_verify_spoof_toggle"):
            with st.spinner("Running deterministic OFF/ON spoof verification..."):
                frame, verdict = _run_spoof_toggle_verification(
                    host=backend_host,
                    port=backend_port,
                    controls=controls,
                )
                st.session_state["spoof_ab_frame"] = frame
                st.session_state["spoof_toggle_verdict"] = verdict
        if st.button("Run Per-Target Spoof Delta Heatmap", key="cmd_run_spoof_delta_heatmap"):
            with st.spinner("Running OFF/ON mission pair and building per-target heatmap..."):
                st.session_state["spoof_ab_delta_frame"] = _run_spoof_ab_target_deltas(
                    host=backend_host,
                    port=backend_port,
                    controls=controls,
                )
        if st.button("Run 5-Profile Spoof Validation Sweep", key="cmd_run_spoof_profile_sweep"):
            with st.spinner("Running 5 profiles with spoof OFF/ON..."):
                st.session_state["spoof_profile_sweep_frame"] = _run_profile_ab_sweep(
                    host=backend_host,
                    port=backend_port,
                    controls=controls,
                )
        spoof_ab_frame = st.session_state.get("spoof_ab_frame")
        if isinstance(spoof_ab_frame, pd.DataFrame) and not spoof_ab_frame.empty:
            st.dataframe(spoof_ab_frame, width="stretch", hide_index=True)
        verdict = st.session_state.get("spoof_toggle_verdict")
        if isinstance(verdict, dict) and verdict:
            if bool(verdict.get("passed", False)):
                st.success(str(verdict.get("message", "PASS")))
            else:
                st.warning(str(verdict.get("message", "HOLD")))
            checks = verdict.get("checks", {})
            if isinstance(checks, dict) and checks:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Check": str(key), "Result": bool(value)}
                            for key, value in checks.items()
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
        delta_frame = st.session_state.get("spoof_ab_delta_frame")
        if isinstance(delta_frame, pd.DataFrame) and not delta_frame.empty:
            st.dataframe(delta_frame, width="stretch", hide_index=True)
            st.plotly_chart(
                _build_spoof_delta_heatmap(delta_frame),
                width="stretch",
                key="spoof_delta_heatmap_live",
                config={"displayModeBar": False},
            )
        profile_sweep_frame = st.session_state.get("spoof_profile_sweep_frame")
        if isinstance(profile_sweep_frame, pd.DataFrame) and not profile_sweep_frame.empty:
            st.dataframe(profile_sweep_frame, width="stretch", hide_index=True)
            st.plotly_chart(
                _build_profile_sweep_heatmap(profile_sweep_frame),
                width="stretch",
                key="spoof_profile_heatmap_live",
                config={"displayModeBar": False},
            )
            st.caption(
                "Interpretation: positive deltas increase from OFF to ON; use these profiles to verify spoofing resilience across different user-input stress levels."
            )

    with tabs[2]:
        st.caption("Jetson Nano runtime model and startup commands.")
        jetson_script_path = ROOT / "scripts" / "jetson_nano_bootstrap.sh"
        if jetson_script_path.exists():
            st.markdown(f"[Open Jetson bootstrap script]({jetson_script_path.as_posix()})")
            st.code(jetson_script_path.read_text(encoding="utf-8"), language="bash")
        st.code(
            "\n".join(
                [
                    "# On Jetson Nano / Orin Nano",
                    "python -m drone_interceptor.ros2.vision_node --config configs/default.yaml",
                    "python -m drone_interceptor.ros2.spoof_node --config configs/default.yaml --spoof-enable",
                    "# Optional combined node",
                    "python -m drone_interceptor.ros2.spoof_manager --config configs/default.yaml --spoof-enable",
                ]
            ),
            language="bash",
        )
        st.markdown(
            "- Vision pipeline: YOLOv10-tiny inference -> relative target coordinates\n"
            "- Spoof manager (defensive dry-run): drift plan + safety interlock + telemetry status\n"
            "- MAVLink bridge pushes STATUSTEXT events to GCS."
        )
        st.markdown("#### ROS2 Node Roles")
        st.code(
            "\n".join(
                [
                    "/camera/image_raw -> VisionNode -> /spoof/target_relative",
                    "/mavros/global_position/raw + /spoof/target_relative -> SpoofNode -> /spoof/status",
                    "SpoofNode -> MavlinkBridge -> STATUSTEXT to GCS",
                ]
            ),
            language="text",
        )
        st.markdown("#### ROS2 Role Matrix (Operational)")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Node": "vision_node",
                        "Inputs": "/camera/image_raw",
                        "Outputs": "/spoof/target_relative, /vision/detections",
                        "Rate": "30-45 Hz",
                        "Purpose": "YOLOv10-tiny target localization",
                    },
                    {
                        "Node": "spoof_node",
                        "Inputs": "/mavros/global_position/raw, /spoof/target_relative",
                        "Outputs": "/spoof/status, /spoof/drift_plan",
                        "Rate": "20-40 Hz",
                        "Purpose": "Compute defensive spoof-aware drift response",
                    },
                    {
                        "Node": "mavlink_bridge",
                        "Inputs": "/spoof/status, /mission/state",
                        "Outputs": "GCS STATUSTEXT",
                        "Rate": "5-10 Hz",
                        "Purpose": "Operator awareness and command signaling",
                    },
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.markdown("#### Jetson Topic I/O Examples")
        st.code(
            """
# VisionNode output example (/spoof/target_relative)
{
  "target_id": "Target_3",
  "bbox_px": [512, 208, 96, 74],
  "relative_ned_m": {"x": 42.3, "y": -7.1, "z": -2.4},
  "confidence": 0.93,
  "stamp_ms": 1744203100123
}

# SpoofNode status example (/spoof/status)
{
  "spoof_enable": true,
  "spoofing_active": true,
  "spoofing_detected": true,
  "innovation_m": 3.84,
  "innovation_gate_m": 1.95,
  "target_id": "Target_3",
  "drift_plan_m": {"north": 6.2, "east": -1.7},
  "safety_interlock": {"power_scale": 0.42, "sdr_to_gnss_distance_m": 0.78}
}

# MavlinkBridge STATUSTEXT examples
TARGET ACQUIRED: Target_3 conf=0.93
SPOOFING ACTIVE: innovation=3.84m gate=1.95m
SAFETY INTERLOCK: SDR power throttled to 42%
            """.strip(),
            language="json",
        )
        st.markdown("#### Spoof Logic Gate (Reference)")
        st.code(
            """
spoof_enable = bool(config.spoof_enable and ui.enable_spoofing)
innovation_ratio = innovation_m / max(innovation_gate_m, 1e-3)
spoofing_detected = innovation_ratio > 1.0

if not spoof_enable:
    spoofing_active = False
    drift_plan = (0.0, 0.0)
else:
    drift_plan = compute_drift_plan(target_relative_state, gps_state)
    spoofing_active = norm(drift_plan) > 0.0

power_scale = safety_interlock_scale(sdr_to_gnss_distance_m)
apply_sdr_limits(power_scale)
publish_status(spoof_enable, spoofing_active, spoofing_detected, drift_plan)
            """.strip(),
            language="python",
        )
        st.markdown("#### Operating Sequence (Jetson + ROS2)")
        st.code(
            "\n".join(
                [
                    "source /opt/ros/humble/setup.bash",
                    "ros2 launch mavros px4.launch fcu_url:=udp://:14540@127.0.0.1:14557",
                    "python -m drone_interceptor.ros2.vision_node --config configs/default.yaml",
                    "python -m drone_interceptor.ros2.spoof_node --config configs/default.yaml --spoof-enable",
                    "python -m drone_interceptor.ros2.mavlink_bridge --config configs/default.yaml",
                    "ros2 topic echo /spoof/status",
                ]
            ),
            language="bash",
        )

    with tabs[3]:
        st.markdown(
            "### Layer 1: Flight System (PX4 / ArduPilot)\n"
            "- Pixhawk flight controller\n"
            "- ESC + motors\n"
            "- Real GNSS for navigation and autopilot safety loops\n\n"
            "### Layer 2: AI + Vision System\n"
            "- Jetson Nano / Orin Nano\n"
            "- Camera input\n"
            "- YOLOv10-tiny (>30 FPS target on optimized edge profile)\n\n"
            "### Layer 3: Spoofing-Aware Defense System (USP)\n"
            "- SDR runtime checks (HackRF One tooling)\n"
            "- Directional RF safety interlock model\n"
            "- GNURadio/gps-sdr-sim dry-run planning and spoof-risk analytics\n"
            "- No live RF transmission in this software build"
        )
        st.markdown("### Electronic System Architecture (Signal + Power)")
        electronic_bus_rows = [
            {"From": "Pixhawk FCU", "To": "Jetson Nano", "Interface": "MAVLink over UART", "Signal": "global_position/raw, attitude, heartbeat", "Rate": "10-50 Hz"},
            {"From": "Camera", "To": "Jetson Nano", "Interface": "CSI/USB", "Signal": "image_raw", "Rate": "30-60 FPS"},
            {"From": "Jetson VisionNode", "To": "Jetson SpoofNode", "Interface": "ROS2 topic", "Signal": "target_relative_ned + detection confidence", "Rate": "20-45 Hz"},
            {"From": "SpoofNode", "To": "MavlinkBridge", "Interface": "ROS2 topic", "Signal": "spoof_status, innovation, drift_plan", "Rate": "5-20 Hz"},
            {"From": "MavlinkBridge", "To": "GCS", "Interface": "MAVLink STATUSTEXT", "Signal": "TARGET ACQUIRED / SPOOFING ACTIVE / INTERLOCK", "Rate": "1-5 Hz"},
            {"From": "Jetson Safety Interlock", "To": "HackRF Planner", "Interface": "Local control API", "Signal": "power_scale, channel_allow, no_tx_guard", "Rate": "on-change"},
        ]
        power_tree_rows = [
            {"Rail": "Battery Main", "Consumers": "ESC/Motors", "Voltage": "11.1-22.2V", "Notes": "High current propulsion rail"},
            {"Rail": "BEC 5V", "Consumers": "Pixhawk, GPS, telemetry radio", "Voltage": "5V", "Notes": "Flight-critical avionics rail"},
            {"Rail": "Jetson DC", "Consumers": "Jetson Nano/Orin + camera", "Voltage": "5-19V (board dependent)", "Notes": "AI compute + vision rail"},
            {"Rail": "USB 5V", "Consumers": "HackRF One (diagnostics)", "Voltage": "5V", "Notes": "Software interlock enforces dry-run/no-TX policy"},
        ]
        st.dataframe(pd.DataFrame(electronic_bus_rows), width="stretch", hide_index=True)
        st.dataframe(pd.DataFrame(power_tree_rows), width="stretch", hide_index=True)
        st.markdown("#### Signal Exchange Examples")
        st.code(
            """
# FCU -> Jetson (MAVLink payload proxy)
{
  "topic": "/mavros/global_position/raw",
  "lat": 12.971612,
  "lon": 77.594588,
  "alt_m": 122.4,
  "ground_speed_mps": 13.2,
  "stamp_ms": 1744203201450
}

# VisionNode -> SpoofNode (ROS2)
{
  "topic": "/spoof/target_relative",
  "target_id": "Target_4",
  "relative_ned_m": {"x": 38.2, "y": -6.4, "z": -1.8},
  "confidence": 0.91
}

# SpoofNode -> MAVLink bridge -> GCS
{
  "topic": "/spoof/status",
  "spoof_enable": true,
  "anti_spoof_enable": true,
  "innovation_m": 2.84,
  "innovation_gate_m": 1.65,
  "spoofing_detected": true,
  "safety_interlock": {"power_scale": 0.40, "channel_allow": false}
}
            """.strip(),
            language="json",
        )
        status_endpoint = ((backend_state or {}).get("status_endpoint") or {}) if isinstance(backend_state, dict) else {}
        mission_insights = status_endpoint.get("mission_insights", {}) if isinstance(status_endpoint, dict) else {}
        mission_model = mission_insights.get("mission_model", {}) if isinstance(mission_insights, dict) else {}
        packet_model = mission_model.get("packet_loss_model", {}) if isinstance(mission_model, dict) else {}
        energy_model = mission_model.get("energy_model", {}) if isinstance(mission_model, dict) else {}
        mass_kg = float(energy_model.get("interceptor_mass_kg", controls.get("interceptor_mass_kg", 6.5)))
        hover_w = float(energy_model.get("hover_power_w", 90.0))
        drag_coeff = float(energy_model.get("drag_power_coeff", 0.02))
        accel_coeff = float(energy_model.get("accel_power_coeff", 0.45))
        st.markdown("### Mission Service Equations")
        st.latex(r"T_{int} = t_{impact} - t_{launch}")
        st.latex(r"RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_{true,i} - x_{est,i})^2}")
        st.latex(r"PL = 1 - e^{-k \cdot SNR / d^\alpha}")
        st.latex(r"P_t = P_{hover} + c_{drag}\lVert v_t \rVert^3 + \eta \, m \, \lVert a_t \rVert \, \lVert v_t \rVert")
        st.latex(r"E = \sum_t P_t \Delta t")
        st.latex(r"E_{\text{fallback}} = (P_{hover} + \eta\,m\,\bar{g}\,v_{ref}) \cdot T")
        st.caption(
            "Energy model in backend: "
            "per-frame power uses hover + aerodynamic drag + acceleration power, then integrates over time."
        )
        st.caption(
            f"Energy params in current run: mass={mass_kg:.2f} kg, P_hover={hover_w:.1f} W, "
            f"c_drag={drag_coeff:.4f}, eta={accel_coeff:.3f}"
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"Term": "m", "Meaning": "Interceptor mass", "Units": "kg", "Value": round(mass_kg, 3)},
                    {"Term": "P_hover", "Meaning": "Base hover/avionics power draw", "Units": "W", "Value": round(hover_w, 3)},
                    {"Term": "c_drag", "Meaning": "Aerodynamic drag power coefficient", "Units": "W/(m/s)^3", "Value": round(drag_coeff, 5)},
                    {"Term": "eta", "Meaning": "Acceleration-to-power coefficient", "Units": "1", "Value": round(accel_coeff, 4)},
                    {"Term": "|v_t|", "Meaning": "Estimated interceptor speed at frame t", "Units": "m/s", "Value": "runtime"},
                    {"Term": "|a_t|", "Meaning": "Estimated interceptor acceleration at frame t", "Units": "m/s^2", "Value": "runtime"},
                    {"Term": "Δt", "Meaning": "Frame time step", "Units": "s", "Value": "runtime"},
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"Packet model values in current run: "
            f"k={float(packet_model.get('k', controls.get('packet_loss_k', 0.12))):.3f}, "
            f"alpha={float(packet_model.get('alpha', controls.get('packet_loss_alpha', 1.8))):.2f}, "
            f"base_link_snr_db={float(packet_model.get('base_link_snr_db', controls.get('link_snr_db', 28.0))):.2f}"
        )
        st.caption(
            "Hard constraints: interceptor_speed_mps > target_speed_mps, "
            "packet_loss_probability in [0, 0.98], and mission table release only after full per-target metrics are populated."
        )

    with tabs[4]:
        requirements_rows = [
            {"Layer": "Flight System", "Component": "Pixhawk (PX4/ArduPilot)", "Role": "Low-level flight control", "Interface": "MAVLink / PWM"},
            {"Layer": "Flight System", "Component": "ESC + Motors", "Role": "Actuation and thrust", "Interface": "PWM / DShot"},
            {"Layer": "Flight System", "Component": "GNSS Module", "Role": "Real navigation", "Interface": "UART/I2C"},
            {"Layer": "AI + Vision", "Component": "Jetson Nano / Orin Nano", "Role": "Inference + mission logic", "Interface": "ROS2 + CUDA"},
            {"Layer": "AI + Vision", "Component": "Camera", "Role": "Target detection feed", "Interface": "CSI/USB"},
            {"Layer": "AI + Vision", "Component": "YOLOv10-tiny", "Role": "Real-time detection >30 FPS", "Interface": "TensorRT/PyTorch"},
            {"Layer": "Spoofing-Aware Defense", "Component": "HackRF One", "Role": "SDR diagnostics / dry-run planning", "Interface": "USB"},
            {"Layer": "Spoofing-Aware Defense", "Component": "Directional Antenna", "Role": "Directional RF path", "Interface": "RF front-end"},
            {"Layer": "Spoofing-Aware Defense", "Component": "GNU Radio Blocks", "Role": "Signal chain orchestration", "Interface": "grc/python"},
        ]
        st.dataframe(pd.DataFrame(requirements_rows), width="stretch", hide_index=True)
        st.markdown("#### Component Connection Map")
        st.code(
            "\n".join(
                [
                    "Battery -> PDB -> ESC x4 -> Motors x4",
                    "Battery/BEC -> Pixhawk -> MAVLink telemetry radio -> GCS",
                    "Battery/DC rail -> Jetson Nano -> Camera + ROS2 stack",
                    "Pixhawk <-> Jetson via MAVLink bridge (position/attitude/health)",
                    "Jetson VisionNode -> SpoofNode -> MavlinkBridge -> STATUSTEXT/GCS",
                    "Jetson -> HackRF planner (diagnostic dry-run, interlock constrained)",
                ]
            ),
            language="text",
        )
        st.markdown(
            "#### Jetson Nano Use Case"
            "\n- Runs ROS2 `VisionNode`, `SpoofNode`, and optional `SpoofManager`."
            "\n- Consumes `/camera/image_raw` and `/mavros/global_position/raw`."
            "\n- Publishes target-relative coordinates, spoof-status telemetry, and GCS status text."
        )
        st.code(
            "\n".join(
                [
                    "python -m drone_interceptor.ros2.vision_node --config configs/default.yaml",
                    "python -m drone_interceptor.ros2.spoof_node --config configs/default.yaml --spoof-enable",
                    "python test_spoof.py --jetson-host 192.168.55.1 --spoof-enable",
                ]
            ),
            language="bash",
        )

    with tabs[5]:
        st.markdown(
            "### SolidWorks Integration Steps\n"
            "1. Import base frame CAD (F450/X500 class).\n"
            "2. Add Jetson mount at center of gravity with airflow vents.\n"
            "3. Place SDR module on side/bottom arm with metal shielding enclosure.\n"
            "4. Mount directional antenna oriented toward external threat axis.\n"
            "5. Place GNSS module on top mast, far from SDR, with shielded casing.\n"
            "6. Place camera at front with slight downward tilt and vibration isolation.\n"
            "7. Export CAD offsets (`sdr_to_gnss_distance_m`) and feed them to safety interlock config."
        )
        st.info("Safety interlock in software explicitly uses SDR-to-own-GNSS distance to throttle RF power in dry-run planning.")
        st.markdown("### SolidWorks Frontend Model")
        st.caption(
            "The frontend now supports direct GLB visualization for CAD review without affecting the mission workflow."
        )
        _render_solidworks_model_panel()

    with tabs[6]:
        st.caption("Run 15 parameterized missions to validate table completeness and spoof toggle correctness.")
        if st.button("Run 15-Case Judge Sweep", key="cmd_run_judge_sweep15"):
            with st.spinner("Running 15-case sweep... this can take a few minutes."):
                frame, summary = _run_judge_sweep(
                    host=backend_host,
                    port=backend_port,
                    controls=controls,
                )
                st.session_state["judge_sweep_frame"] = frame
                st.session_state["judge_sweep_summary"] = summary
        summary = st.session_state.get("judge_sweep_summary")
        frame = st.session_state.get("judge_sweep_frame")
        if isinstance(summary, dict):
            cols = st.columns(7)
            cols[0].metric("Runs", int(summary.get("runs", 0)))
            cols[1].metric("Target Mismatch", int(summary.get("target_mismatch", 0)))
            cols[2].metric("Missing Metrics Runs", int(summary.get("missing_metrics_runs", 0)))
            cols[3].metric("False Spoof OFF Runs", int(summary.get("false_spoof_off_runs", 0)))
            cols[4].metric("Quality Gate Pass", int(summary.get("quality_gate_pass_runs", 0)))
            cols[5].metric("Deploy Gate Pass", int(summary.get("deploy_gate_pass_runs", 0)))
            cols[6].metric(
                "Designathon Score",
                f"{float(summary.get('designathon_score', 0.0)):.2f}/10",
                delta=str(summary.get("designathon_tier", "n/a")),
            )
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            st.dataframe(frame, width="stretch", hide_index=True)


def _refresh_frontend_build_state() -> None:
    previous_build = st.session_state.get("ui_build")
    if previous_build == UI_BUILD:
        return
    st.cache_data.clear()
    for key in (
        "dashboard_controls_signature",
        "dashboard_simulation",
        "dashboard_benchmark_frame",
        "day8_controls_signature",
        "day8_replay",
        "day8_compare_replay",
        "day8_validation",
        "backend_state",
        "backend_validation",
        "backend_validation_frame",
        "results_table",
        "scenario_results_df",
        "mission_results_ready",
        "mission_expected_targets",
        "backend_preflight",
        "backend_start_result",
    ):
        st.session_state.pop(key, None)
    st.session_state["ui_build"] = UI_BUILD


def _build_backend_payload(controls: dict[str, Any]) -> dict[str, Any]:
    target_count = int(controls["num_targets"])
    max_steps = 180 if target_count <= 3 else 220
    target_ids = [f"Target_{index + 1}" for index in range(target_count)]
    noise_level = float(controls["noise_level"])
    drift_rate = float(controls["drift_rate"])
    packet_loss = float(controls["packet_loss_rate"])
    ekf_process_noise = max(0.05, 0.18 * drift_rate + 0.04 * noise_level)
    ekf_measurement_noise = [
        max(noise_level, 0.2),
        max(noise_level, 0.2),
        max(noise_level * 0.67, 0.18),
    ]
    guidance_gain = float(
        np.clip(
            6.0
            + 0.30 * max(target_count - 1, 0)
            + 1.80 * drift_rate
            + 0.90 * noise_level
            + 0.80 * packet_loss,
            6.0,
            9.2,
        )
    )
    kill_radius_m = float(
        np.clip(
            1.0
            + 0.12 * max(target_count - 1, 0)
            + 0.45 * noise_level
            + 0.70 * packet_loss
            + 0.35 * drift_rate,
            1.0,
            2.8,
        )
    )
    run_seed = int(controls.get("run_seed", int(time.time() * 1000.0) % 1_000_000_000))
    return {
        "num_targets": target_count,
        "target_ids": target_ids,
        "use_ekf": True,
        "use_ekf_anti_spoofing": True,
        "drift_rate_mps": drift_rate,
        "noise_std_m": noise_level,
        "latency_ms": float(controls["latency_ms"]),
        "packet_loss_rate": packet_loss,
        "link_snr_db": float(controls.get("link_snr_db", 28.0)),
        "packet_loss_k": float(controls.get("packet_loss_k", 0.12)),
        "packet_loss_alpha": float(controls.get("packet_loss_alpha", 1.8)),
        "connect_airsim": False,
        "enable_spoofing": True,
        "iterations": 10,
        "max_steps": max_steps,
        "dt": 0.05,
        "guidance_gain": guidance_gain,
        "kill_radius_m": kill_radius_m,
        "target_speed_mps": float(controls["target_speed"]),
        "interceptor_speed_mps": float(controls["interceptor_speed"]),
        "interceptor_mass_kg": float(controls.get("interceptor_mass_kg", 6.5)),
        "hover_power_w": 90.0,
        "drag_power_coeff": 0.02,
        "accel_power_coeff": 0.45,
        "ekf_process_noise": float(ekf_process_noise),
        "ekf_measurement_noise": ekf_measurement_noise,
        "random_seed": int(run_seed),
    }


def _estimate_kill_probability(distance_m: float, uncertainty_m: float = 0.5) -> float:
    sigma = max(float(uncertainty_m), 0.25)
    return float(np.exp(-0.5 * (float(distance_m) / sigma) ** 2))


def _build_backend_simulation_state(
    simulation: dict[str, Any],
    controls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_count = int(controls["num_targets"]) if controls and controls.get("num_targets") is not None else len(simulation.get("target_positions", []))
    spoof_enabled = bool(controls.get("enable_spoofing", False)) if controls else False
    success = bool(simulation.get("success"))
    final_distance = float(simulation.get("final_distance_m", 0.0))
    mean_fps = float(simulation.get("mean_loop_fps", 0.0))
    rmse = float(simulation.get("rmse_m", 0.0))
    snapshot = {
        "mission_id": 0,
        "status": "complete" if success else "running",
        "active_stage": "Intercept Complete" if success else "Simulation",
        "stage": "Intercept Complete" if success else "Simulation",
        "active_target": f"Target_1" if target_count > 0 else "n/a",
        "backend_throughput_fps": mean_fps,
        "detection_fps": mean_fps,
        "detection_fps_window_avg": mean_fps,
        "rmse_m": rmse,
        "rmse_measured_true_m": rmse,
        "mean_uncertainty_m": float(
            np.mean(simulation.get("tracking_errors", [0.0])) if simulation.get("tracking_errors") else 0.0
        ),
        "kill_probability": 1.0 if success else _estimate_kill_probability(final_distance),
        "active_distance_m": final_distance,
        "closing_velocity_mps": 0.0,
        "los_rate_rps": 0.0,
        "innovation_m": float(
            np.mean(simulation.get("tracking_errors", [0.0])) if simulation.get("tracking_errors") else 0.0
        ),
        "innovation_gate": 0.5,
        "confidence_score": float(simulation.get("success_rate", 0.0)),
        "target_count": target_count,
        "spoofing_active": spoof_enabled,
        "ekf_lock": bool(controls.get("use_ekf", True)) if controls else True,
    }
    synthetic_targets: list[dict[str, Any]] = []
    for idx in range(target_count):
        synthetic_targets.append(
            {
                "target_id": f"Target_{idx+1}",
                "name": f"Target_{idx+1}",
                "status": "ACTIVE",
                "threat_level": 0.0,
                "distance_m": float(final_distance),
                "spoofing_active": spoof_enabled,
                "innovation_m": float(simulation.get("rmse_m", 0.0)),
                "innovation_gate": 0.5,
                "estimated_error_m": float(final_distance),
                "drift_rate_mps": float(controls.get("drift_rate", 0.0)) if controls else 0.0,
                "spoof_offset_m": float(simulation.get("noise_level", 0.0)) if spoof_enabled else 0.0,
                "spoofing_detected": False,
                "jammed": False,
            }
        )
    return {
        "snapshot": {**snapshot, "targets": synthetic_targets},
        "validation": {
            "validation_success": success,
            "ekf_success_rate": float(simulation.get("success_rate", 0.0)),
            "ekf_mean_miss_distance_m": float(final_distance),
            "per_target_summary": [
                {
                    "target": target["target_id"],
                    "ekf_success_rate": float(simulation.get("success_rate", 0.0)),
                    "ekf_mean_miss_distance_m": float(final_distance),
                    "drift_rate_mps": float(controls.get("drift_rate", 0.0)) if controls else 0.0,
                    "spoof_offset_m": float(simulation.get("noise_level", 0.0)),
                    "time_to_recovery_s": 0.0,
                }
                for target in synthetic_targets
            ],
        },
        "status_endpoint": {
            "service_status": "ok",
            "schema_version": "1.0",
            "mission_status": "simulation",
            "lifecycle": "SIMULATION",
            "target_count": target_count,
            "heartbeat_live": False,
            "heartbeat_age_ms": None,
            "deploy_ready": success,
            "validation_success": success,
            "run_id": None,
        },
        "preflight": {
            "ready": True,
            "spawned_targets": target_count,
            "requested_targets": target_count,
            "fallback_mode": True,
        },
        "deploy_ready": success,
    }


def _backend_request_json(
    host: str,
    port: int,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    url = f"http://{host}:{port}{path}"
    request_payload = None
    headers = {}
    method = "GET"
    if payload is not None:
        request_payload = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urlrequest.Request(url, data=request_payload, headers=headers, method=method)
    with urlrequest.urlopen(request, timeout=max(float(timeout_s), 0.1)) as response:
        return json.loads(response.read().decode("utf-8"))


def run_100x_validation(host: str, port: int, payload: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    report = _backend_request_json(host, port, "/validate", payload)
    scenario_frame = pd.DataFrame(report.get("scenario_results", []))
    return report, scenario_frame


def _run_coro(coro: Any) -> Any:
    return asyncio.run(coro)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


def _safe_series_mean(values: pd.Series, default: float = 0.0) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return float(default)
    return float(numeric.mean())


def _resolve_solidworks_model_path() -> Path | None:
    candidates = [
        ROOT / "outputs" / "solidworks" / "Honeywell_drone.glb",
        ROOT / "outputs" / "solidworks" / "Honeywell drone.glb",
        ROOT / "outputs" / "assets" / "Honeywell_drone.glb",
        ROOT / "outputs" / "assets" / "Honeywell drone.glb",
        ROOT / "Honeywell_drone.glb",
        ROOT / "Honeywell drone.glb",
        SOLIDWORKS_MODEL_FALLBACK,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".glb":
            return candidate
    solidworks_dir = ROOT / "outputs" / "solidworks"
    if solidworks_dir.exists():
        glb_files = sorted(solidworks_dir.glob("*.glb"))
        if glb_files:
            return glb_files[0]
    return None


def _artifact_video_candidates() -> list[Path]:
    candidates: list[Path] = []
    if not OUTPUTS.exists():
        return candidates

    preferred_names = (
        "day9_dp5_demo.mp4",
        "day8_bms_demo.mp4",
        "final_demo.mp4",
    )
    for name in preferred_names:
        path = OUTPUTS / name
        if path.exists() and path.is_file():
            candidates.append(path)

    mp4_files = sorted(OUTPUTS.glob("*.mp4"), key=lambda p: p.name.lower())
    avi_files = sorted(OUTPUTS.glob("*.avi"), key=lambda p: p.name.lower())
    for path in (*mp4_files, *avi_files):
        if path not in candidates:
            candidates.append(path)
    return candidates


@st.cache_data(show_spinner=False)
def _load_glb_data_url(path_str: str) -> str:
    payload = Path(path_str).read_bytes()
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:model/gltf-binary;base64,{encoded}"


def _render_solidworks_model_panel() -> None:
    model_path = _resolve_solidworks_model_path()
    if model_path is None:
        st.warning(
            "No `.glb` CAD model was found. Place the file at "
            "`outputs/solidworks/Honeywell_drone.glb` or keep it at "
            "`C:\\Users\\hp\\Downloads\\Honeywell drone.glb`."
        )
        return

    model_size_mb = float(model_path.stat().st_size / (1024.0 * 1024.0))
    st.caption(f"Loaded model: `{model_path.name}` ({model_size_mb:.2f} MB)")
    auto_rotate = st.toggle("Auto rotate model", value=True, key="solidworks_auto_rotate")
    exposure = st.slider("Model exposure", min_value=0.6, max_value=2.0, value=1.0, step=0.1, key="solidworks_exposure")
    camera_orbit = st.selectbox(
        "Default camera orbit",
        options=["35deg 70deg 250%", "0deg 75deg 280%", "90deg 65deg 260%"],
        index=0,
        key="solidworks_camera_orbit",
    )
    if st.button("Load 3D SolidWorks Model", key="solidworks_load_model"):
        st.session_state["solidworks_model_loaded"] = True
    if not bool(st.session_state.get("solidworks_model_loaded", False)):
        st.info("Press `Load 3D SolidWorks Model` to render the CAD model in the frontend.")
        st.download_button(
            "Download GLB",
            data=model_path.read_bytes(),
            file_name=model_path.name,
            mime="model/gltf-binary",
            key="solidworks_download_glb_idle",
        )
        return

    try:
        model_data_url = _load_glb_data_url(str(model_path))
    except Exception as exc:
        st.error(f"Failed to read GLB model: {exc}")
        return

    auto_rotate_attr = "auto-rotate" if auto_rotate else ""
    viewer_html = f"""
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    <style>
      .solidworks-shell {{
        width: 100%;
        height: 560px;
        border: 1px solid #1f2937;
        border-radius: 12px;
        overflow: hidden;
        background: linear-gradient(180deg, #0f172a, #020617);
      }}
      model-viewer {{
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at 20% 10%, #1e293b 0%, #020617 70%);
      }}
    </style>
    <div class="solidworks-shell">
      <model-viewer
        src="{model_data_url}"
        camera-controls
        {auto_rotate_attr}
        shadow-intensity="1"
        tone-mapping="aces"
        exposure="{exposure}"
        camera-orbit="{camera_orbit}"
        interaction-prompt="auto">
      </model-viewer>
    </div>
    """
    components.html(viewer_html, height=590, scrolling=False)
    st.download_button(
        "Download GLB",
        data=model_path.read_bytes(),
        file_name=model_path.name,
        mime="model/gltf-binary",
        key="solidworks_download_glb_loaded",
    )
    st.caption(f"Model source: `{model_path}`")


def _resolve_backend_canonical_metrics(simulation: dict[str, Any] | None = None) -> dict[str, Any]:
    fallback = simulation or {}
    metrics: dict[str, Any] = {
        "success_rate": _normalize_success_ratio(
            fallback.get("success_rate", 1.0 if bool(fallback.get("success", False)) else 0.0)
        ),
        "earliest_interception_time_s": fallback.get("interception_time_s"),
        "mean_interception_time_s": fallback.get("interception_time_s"),
        "interception_target": "n/a",
        "rmse_m": _safe_float(fallback.get("rmse_m"), 0.0),
        "final_distance_m": _safe_float(fallback.get("final_distance_m"), 0.0),
        "source": "simulation",
    }

    results_table = st.session_state.get("results_table")
    if isinstance(results_table, pd.DataFrame) and not results_table.empty:
        if "ekf_success_rate" in results_table.columns:
            numeric_success = pd.to_numeric(results_table["ekf_success_rate"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if not numeric_success.empty:
                metrics["success_rate"] = _normalize_success_ratio(float(numeric_success.mean()))
                metrics["source"] = "backend_results_table"
        if "interception_time_s" in results_table.columns:
            numeric_times = pd.to_numeric(results_table["interception_time_s"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if not numeric_times.empty:
                earliest = float(numeric_times.min())
                metrics["earliest_interception_time_s"] = earliest
                metrics["mean_interception_time_s"] = float(numeric_times.mean())
                if "target" in results_table.columns:
                    earliest_idx = numeric_times.idxmin()
                    metrics["interception_target"] = str(results_table.loc[earliest_idx, "target"])
                metrics["source"] = "backend_results_table"
        if "rmse_m" in results_table.columns:
            metrics["rmse_m"] = _safe_series_mean(results_table["rmse_m"], metrics["rmse_m"])
            metrics["source"] = "backend_results_table"

    backend_state = st.session_state.get("backend_state")
    if isinstance(backend_state, dict):
        snapshot = backend_state.get("snapshot", {}) if isinstance(backend_state.get("snapshot"), dict) else {}
        status_endpoint = backend_state.get("status_endpoint", {}) if isinstance(backend_state.get("status_endpoint"), dict) else {}
        mission_insights = status_endpoint.get("mission_insights", {}) if isinstance(status_endpoint, dict) else {}
        global_metrics = mission_insights.get("global_metrics", {}) if isinstance(mission_insights, dict) else {}
        mission_status = str(snapshot.get("status", "")).lower()
        metrics["mission_complete"] = mission_status == "complete"

        weighted_success = global_metrics.get("weighted_mission_success")
        if weighted_success is not None:
            metrics["success_rate"] = _normalize_success_ratio(weighted_success)
            metrics["source"] = "backend_mission_insights"
        elif snapshot.get("mission_success") is not None and str(metrics.get("source", "simulation")) == "simulation":
            metrics["success_rate"] = 1.0 if bool(snapshot.get("mission_success", False)) else 0.0

        earliest_global = global_metrics.get("earliest_intercept_s")
        if earliest_global is not None and np.isfinite(_safe_float(earliest_global, np.nan)):
            metrics["earliest_interception_time_s"] = _safe_float(earliest_global, metrics["earliest_interception_time_s"] or 0.0)
            if metrics["mean_interception_time_s"] is None:
                metrics["mean_interception_time_s"] = metrics["earliest_interception_time_s"]
            metrics["source"] = "backend_mission_insights"

        distance_complete = snapshot.get("closest_approach_m")
        if distance_complete is not None and np.isfinite(_safe_float(distance_complete, np.nan)):
            metrics["final_distance_m"] = _safe_float(distance_complete, metrics["final_distance_m"])
            metrics["source"] = "backend_snapshot"
        elif snapshot.get("active_distance_m") is not None:
            metrics["final_distance_m"] = _safe_float(snapshot.get("active_distance_m"), metrics["final_distance_m"])

        if snapshot.get("rmse_m") is not None:
            metrics["rmse_m"] = _safe_float(snapshot.get("rmse_m"), metrics["rmse_m"])

    return metrics


def _normalize_success_ratio(value: Any) -> float:
    ratio = _safe_float(value, 0.0)
    if ratio > 1.0 and ratio <= 100.0:
        ratio /= 100.0
    return float(np.clip(ratio, 0.0, 1.0))


def _estimate_result_probability(success_rate: float, rmse_m: float, threshold_m: float = 0.5) -> float:
    threshold = max(_safe_float(threshold_m, 0.5), 1e-3)
    distance_component = float(np.exp(-0.5 * (_safe_float(rmse_m, 0.0) / threshold) ** 2))
    return float(np.clip(0.55 * _normalize_success_ratio(success_rate) + 0.45 * distance_component, 0.0, 1.0))


def _derive_results_from_validation(validation_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = validation_payload.get("per_target_summary", []) if isinstance(validation_payload, dict) else []
    if not isinstance(rows, list):
        return []
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        target_id = str(row.get("target") or row.get("target_id") or f"Target_{index + 1}")
        success_rate = _normalize_success_ratio(row.get("ekf_success_rate", 0.0))
        rmse_value = _safe_float(row.get("rmse", row.get("ekf_mean_miss_distance_m", 0.0)), 0.0)
        interception_time = row.get("interception_time_s")
        if interception_time is None:
            t_launch = row.get("t_launch")
            t_impact = row.get("t_impact")
            if t_launch is not None and t_impact is not None:
                interception_time = max(_safe_float(t_impact, 0.0) - _safe_float(t_launch, 0.0), 0.0)
        mission_success_probability = row.get("mission_success_probability")
        if mission_success_probability is None:
            mission_success_probability = _estimate_result_probability(success_rate, rmse_value)
        results.append(
            {
                "target_id": target_id,
                "ekf_success_rate": success_rate,
                "interception_time": _safe_float(interception_time, 0.0),
                "rmse": rmse_value,
                "guidance_efficiency_mps2": _safe_float(row.get("guidance_efficiency_mps2", 0.0), 0.0),
                "spoofing_variance": _safe_float(row.get("spoofing_variance", 0.0), 0.0),
                "compute_latency_ms": _safe_float(row.get("compute_latency_ms", 0.0), 0.0),
                "energy_consumption_j": _safe_float(row.get("energy_consumption_j", 0.0), 0.0),
                "mission_success_probability": _normalize_success_ratio(mission_success_probability),
                "t_launch": row.get("t_launch"),
                "t_impact": row.get("t_impact"),
            }
        )
    return results


def _complete_results_for_targets(
    results: list[dict[str, Any]],
    target_ids: list[str],
    validation_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    by_target: dict[str, dict[str, Any]] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        target_id = str(row.get("target_id") or row.get("target") or "").strip()
        if target_id:
            by_target[target_id] = dict(row)

    validation_rows = _derive_results_from_validation(validation_payload)
    for row in validation_rows:
        target_id = str(row.get("target_id") or "").strip()
        if not target_id:
            continue
        if target_id in by_target:
            merged = dict(row)
            merged.update({k: v for k, v in by_target[target_id].items() if v is not None})
            by_target[target_id] = merged
        else:
            by_target[target_id] = dict(row)

    if not target_ids:
        return list(by_target.values())

    completed: list[dict[str, Any]] = []
    for target_id in target_ids:
        if target_id in by_target:
            completed.append(dict(by_target[target_id]))
    for target_id, row in by_target.items():
        if target_id not in target_ids:
            completed.append(dict(row))
    return completed


def _mission_results_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in results:
        success_rate = _normalize_success_ratio(row.get("ekf_success_rate", 0.0))
        rmse_value = _safe_float(row.get("rmse", row.get("rmse_m", row.get("estimated_error_m", 0.0))), 0.0)
        interception_time = row.get("interception_time")
        if interception_time is None:
            interception_time = row.get("interception_time_s")
        if interception_time is None:
            t_launch = row.get("t_launch")
            t_impact = row.get("t_impact")
            if t_launch is not None and t_impact is not None:
                interception_time = max(_safe_float(t_impact, 0.0) - _safe_float(t_launch, 0.0), 0.0)
        interception_time_s = _safe_float(interception_time, 0.0)

        raw_probability = row.get("mission_success_probability", row.get("kill_probability"))
        mission_success_probability = (
            _normalize_success_ratio(raw_probability)
            if raw_probability is not None
            else _estimate_result_probability(success_rate, rmse_value)
        )
        guidance_efficiency_mps2 = _safe_float(row.get("guidance_efficiency_mps2", 0.0), 0.0)
        spoofing_variance = _safe_float(row.get("spoofing_variance", 0.0), 0.0)
        compute_latency_ms = _safe_float(row.get("compute_latency_ms", row.get("latency_ms", 0.0)), 0.0)
        energy_consumption_j = _safe_float(row.get("energy_consumption_j", 0.0), 0.0)
        rows.append(
            {
                "target": str(row.get("target_id") or row.get("target") or "unknown"),
                "ekf_success_rate": round(success_rate * 100.0, 1),
                "interception_time_s": round(interception_time_s, 3),
                "rmse_m": round(rmse_value, 3),
                "measured_rmse_m": round(rmse_value, 3),
                "mission_success_probability": round(mission_success_probability * 100.0, 1),
                "guidance_efficiency_mps2": round(guidance_efficiency_mps2, 4),
                "spoofing_variance": round(spoofing_variance, 6),
                "compute_latency_ms": round(compute_latency_ms, 3),
                "energy_consumption_j": round(energy_consumption_j, 3),
            }
        )
    frame = pd.DataFrame(rows)
    expected_columns = [
        "target",
        "ekf_success_rate",
        "interception_time_s",
        "rmse_m",
        "measured_rmse_m",
        "mission_success_probability",
        "guidance_efficiency_mps2",
        "spoofing_variance",
        "compute_latency_ms",
        "energy_consumption_j",
    ]
    if frame.empty:
        return pd.DataFrame(columns=expected_columns)
    for column in expected_columns:
        if column not in frame.columns:
            frame[column] = 0.0 if column != "target" else ""
    return frame[expected_columns].copy()


def _derive_results_from_status_targets(status_payload: dict[str, Any]) -> list[dict[str, Any]]:
    targets = status_payload.get("targets", []) if isinstance(status_payload, dict) else []
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            continue
        target_id = str(target.get("target_id") or target.get("name") or f"Target_{index + 1}")
        rmse_value = _safe_float(target.get("rmse", target.get("estimated_error_m", target.get("distance_m", 0.0))), 0.0)
        success_rate = _normalize_success_ratio(target.get("ekf_success_rate", 1.0 if rmse_value <= 0.5 else 0.0))
        mission_success_probability = target.get("mission_success_probability")
        if mission_success_probability is None:
            mission_success_probability = target.get("kill_probability")
        if mission_success_probability is None:
            mission_success_probability = _estimate_result_probability(success_rate, rmse_value)
        rows.append(
            {
                "target_id": target_id,
                "ekf_success_rate": success_rate,
                "interception_time": _safe_float(target.get("interception_time_s"), 0.0),
                "rmse": rmse_value,
                "guidance_efficiency_mps2": _safe_float(target.get("guidance_efficiency_mps2", 0.0), 0.0),
                "spoofing_variance": _safe_float(target.get("spoofing_variance", 0.0), 0.0),
                "compute_latency_ms": _safe_float(target.get("compute_latency_ms", 0.0), 0.0),
                "energy_consumption_j": _safe_float(target.get("energy_consumption_j", 0.0), 0.0),
                "mission_success_probability": _normalize_success_ratio(mission_success_probability),
            }
        )
    return rows


def _extract_mission_results(response: dict[str, Any], controls: dict[str, Any]) -> list[dict[str, Any]]:
    results = response.get("results")
    if not isinstance(results, list):
        results = []
    if not results and isinstance(response.get("mission_summary"), dict):
        summary_results = response["mission_summary"].get("results", [])
        if isinstance(summary_results, list):
            results = summary_results
    validation_payload = response.get("validation", {})
    if not isinstance(validation_payload, dict):
        validation_payload = ((response.get("state") or {}).get("validation") or {})
    if not results and isinstance(validation_payload, dict):
        results = _derive_results_from_validation(validation_payload)
    if not results and isinstance(response.get("status"), dict):
        results = _derive_results_from_status_targets(response.get("status", {}))
    if not results and isinstance(response.get("targets"), list):
        results = _derive_results_from_status_targets({"targets": response.get("targets", [])})
    expected_target_ids: list[str] = []
    status_targets = response.get("targets", [])
    if isinstance(status_targets, list):
        for target in status_targets:
            if isinstance(target, dict):
                target_id = str(target.get("target_id") or target.get("name") or "").strip()
                if target_id:
                    expected_target_ids.append(target_id)
    if not expected_target_ids:
        expected_target_ids = [f"Target_{index + 1}" for index in range(max(int(controls.get("num_targets", 1)), 1))]
    completed = _complete_results_for_targets(
        results=results,
        target_ids=expected_target_ids,
        validation_payload=validation_payload,
    )
    return completed


def _build_fallback_results_table(simulation: dict[str, Any], controls: dict[str, Any]) -> pd.DataFrame:
    target_count = max(int(controls.get("num_targets", 1)), 1)
    success_rate = 1.0 if bool(simulation.get("success", False)) else 0.0
    rmse_value = float(simulation.get("rmse_m", 0.0))
    interception_time = simulation.get("interception_time_s")
    rows = [
        {
            "target_id": f"Target_{index + 1}",
            "ekf_success_rate": success_rate,
            "interception_time": interception_time,
            "rmse": rmse_value,
            "guidance_efficiency_mps2": 0.0,
            "spoofing_variance": 0.0,
            "compute_latency_ms": float(controls.get("latency_ms", 0.0)),
            "energy_consumption_j": 0.0,
            "mission_success_probability": success_rate,
        }
        for index in range(target_count)
    ]
    return _mission_results_to_dataframe(rows)


async def _run_unified_mission_workflow_async(
    host: str,
    port: int,
    controls: dict[str, Any],
) -> dict[str, Any] | None:
    payload = _build_backend_payload(controls)
    expected_targets = max(int(payload.get("num_targets", 1)), 1)
    st.session_state["mission_expected_targets"] = int(expected_targets)
    mission_timeout_s = min(
        240.0,
        max(
            45.0,
            15.0
            + float(payload.get("max_steps", 0)) * 0.45
            + float(payload.get("num_targets", 1)) * 4.0,
        ),
    )
    try:
        response = await asyncio.to_thread(
            _backend_request_json,
            host,
            port,
            "/run_mission",
            payload,
            mission_timeout_s,
        )
    except Exception as exc:
        st.session_state["backend_error"] = str(exc)
        st.session_state["mission_results_ready"] = False
        return None

    if response.get("workflow_status") != "success":
        st.session_state["backend_error"] = response.get("error", "Unknown error in /run_mission")
        st.session_state["mission_results_ready"] = False
        return None

    results = _extract_mission_results(response, controls)
    if len(results) < expected_targets:
        for _ in range(3):
            await asyncio.sleep(0.35)
            try:
                state_payload = _backend_request_json(host, port, "/mission/state", timeout_s=6.0)
                status_payload = _backend_request_json(host, port, "/status", timeout_s=6.0)
            except Exception:
                continue
            refill = _extract_mission_results(
                {
                    "state": state_payload,
                    "validation": state_payload.get("validation", {}),
                    "status": status_payload,
                    "targets": status_payload.get("targets", []),
                },
                controls,
            )
            if len(refill) >= expected_targets:
                results = refill
                break

    results_frame = _mission_results_to_dataframe(results)
    if not results_frame.empty:
        st.session_state["mission_expected_targets"] = int(len(results_frame))
    st.session_state["results_table"] = results_frame
    st.session_state["scenario_results_df"] = results_frame.copy()

    validation = response.get("validation")
    if not isinstance(validation, dict):
        validation = ((response.get("state") or {}).get("validation") or {})
    st.session_state["backend_validation"] = validation
    st.session_state["backend_validation_frame"] = results_frame.copy()
    st.session_state["backend_deploy_ready"] = bool(validation.get("validation_success", False))

    targets = response.get("targets", [])
    mission_insights = response.get("mission_insights", {})
    snapshot = response.get("snapshot", {})
    state = response.get("state", {})
    st.session_state["backend_run_mission_targets"] = targets
    st.session_state["backend_mission_insights"] = mission_insights
    st.session_state["backend_snapshot_latest"] = snapshot
    st.session_state["backend_state_latest"] = state

    global_metrics = mission_insights.get("global_metrics", {}) if isinstance(mission_insights, dict) else {}
    command_readiness = mission_insights.get("command_readiness", {}) if isinstance(mission_insights, dict) else {}
    mission_history = st.session_state.get("mission_history", [])
    if not isinstance(mission_history, list):
        mission_history = []
    mission_history.append(
        {
            "run_id": str(response.get("run_id", "n/a")),
            "run_time": time.strftime("%H:%M:%S", time.localtime()),
            "target_count": int(global_metrics.get("total_targets", len(results_frame))),
            "weighted_success_pct": round(float(global_metrics.get("weighted_mission_success", 0.0)) * 100.0, 2),
            "rmse_p95_m": round(float(global_metrics.get("rmse_p95_m", 0.0)), 4),
            "risk_p90": round(float(global_metrics.get("risk_index_p90", 0.0)), 4),
            "spoof_detect_rate_pct": round(float(global_metrics.get("spoofing_detection_rate", 0.0)) * 100.0, 2),
            "readiness_score": round(float(command_readiness.get("readiness_score", 0.0)), 2),
            "security_posture": str(command_readiness.get("security_posture", "UNKNOWN")),
            "quality_gate_passed": bool(command_readiness.get("quality_gate_passed", False)),
            "deployment_gate_passed": bool(command_readiness.get("deployment_gate_passed", False)),
        }
    )
    st.session_state["mission_history"] = mission_history[-40:]

    mission_status = str(((state.get("snapshot", {}) if isinstance(state, dict) else {}).get("status", "") or (snapshot.get("status") if isinstance(snapshot, dict) else ""))).lower()
    mission_complete = mission_status == "complete" or str((response.get("mission", {}) or {}).get("status", "")).lower() == "complete"
    required_columns = [
        "target",
        "ekf_success_rate",
        "interception_time_s",
        "rmse_m",
        "measured_rmse_m",
        "mission_success_probability",
        "guidance_efficiency_mps2",
        "spoofing_variance",
        "compute_latency_ms",
        "energy_consumption_j",
    ]
    frame_complete = (
        not results_frame.empty
        and len(results_frame) >= expected_targets
        and set(required_columns).issubset(set(results_frame.columns))
        and not bool(results_frame[required_columns].isna().any().any())
    )
    st.session_state["mission_results_ready"] = bool(mission_complete and frame_complete)

    st.session_state.pop("backend_error", None)
    return response


def _run_unified_mission_workflow(
    host: str,
    port: int,
    controls: dict[str, Any],
) -> dict[str, Any] | None:
    """Execute unified mission workflow with async request/await semantics."""
    return _run_coro(_run_unified_mission_workflow_async(host=host, port=port, controls=controls))


def _sync_backend_service(
    controls: dict[str, Any],
    simulation: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    host = str(controls["backend_host"])
    port = int(controls["backend_port"])
    backend_state = st.session_state.get("backend_state")

    try:
        # Auto-trigger unified workflow on Run button click.
        if controls["run_clicked"]:
            st.session_state["mission_results_ready"] = False
            if not _is_backend_reachable(host, port, timeout_s=0.6):
                with st.spinner("Backend service unreachable. Auto-launching and waiting 2 seconds..."):
                    launched = _start_backend_process(
                        host=host,
                        port=port,
                        startup_wait_s=BACKEND_BOOTSTRAP_SLEEP_S,
                        session_flag=True,
                    )
                if not launched and not _is_backend_reachable(host, port, timeout_s=0.6):
                    raise OSError(f"Backend service is unavailable on {host}:{port}")

            mission_response = _run_unified_mission_workflow(host, port, controls)
            if mission_response is not None and mission_response.get("workflow_status") == "success":
                backend_state = mission_response.get("state") or backend_state
                targets = mission_response.get("targets", [])
                if backend_state is None:
                    backend_state = {"snapshot": {}, "validation": None}
                if isinstance(backend_state.get("snapshot"), dict):
                    backend_state["snapshot"]["targets"] = targets
                if isinstance(mission_response.get("validation"), dict):
                    backend_state["validation"] = mission_response.get("validation")
                status_payload = mission_response.get("status")
                if isinstance(status_payload, dict):
                    backend_state["status_endpoint"] = status_payload
                else:
                    backend_state["status_endpoint"] = _backend_request_json(host, port, "/status")
                st.session_state["backend_state"] = backend_state
                st.session_state["dashboard_status"] = "SIMULATION"
                st.session_state["backend_health_failures"] = 0
                st.session_state.pop("backend_error", None)
                return backend_state

        # Live mode polling
        if controls["backend_live_mode"]:
            backend_state = _backend_request_json(host, port, "/mission/state")
            backend_state["status_endpoint"] = _backend_request_json(host, port, "/status")
            st.session_state["backend_state"] = backend_state
            snapshot = backend_state.get("snapshot", {})
            if snapshot:
                st.session_state["dashboard_status"] = str(snapshot.get("status", "STOPPED")).upper()
                st.session_state["dashboard_active_stage"] = str(snapshot.get("active_stage", "Detection"))
            st.session_state["backend_health_failures"] = 0
            st.session_state.pop("backend_error", None)
            return backend_state
        
        if controls["deploy_clicked"]:
            st.session_state["backend_deployed"] = True

        return backend_state

    except (urlerror.URLError, TimeoutError, OSError) as exc:
        st.session_state["backend_error"] = str(exc)
        failure_count = int(st.session_state.get("backend_health_failures", 0)) + 1
        st.session_state["backend_health_failures"] = failure_count
        st.session_state["backend_heartbeat_live"] = False

        if failure_count == 1:
            with st.spinner("Backend unreachable. Attempting to start backend process..."):
                if _start_backend_process(
                    host=host,
                    port=port,
                    startup_wait_s=BACKEND_BOOTSTRAP_SLEEP_S,
                    session_flag=True,
                ):
                    if controls.get("run_clicked", False):
                        mission_response = _run_unified_mission_workflow(host, port, controls)
                        if mission_response is not None and mission_response.get("workflow_status") == "success":
                            backend_state = mission_response.get("state") or {"snapshot": {}, "validation": None}
                            backend_state["status_endpoint"] = mission_response.get("status", {})
                            st.session_state["backend_state"] = backend_state
                            st.session_state["backend_health_failures"] = 0
                            st.session_state.pop("backend_error", None)
                            return backend_state

        if failure_count >= 2:
            st.session_state["backend_failsafe_land"] = True
            st.session_state["dashboard_status"] = "FAILSAFE LAND"

        if simulation is not None and controls["run_clicked"]:
            empty_frame = _mission_results_to_dataframe([])
            st.session_state["backend_state"] = None
            st.session_state["results_table"] = empty_frame
            st.session_state["scenario_results_df"] = empty_frame.copy()
            st.session_state["mission_results_ready"] = False
            st.session_state["backend_validation"] = {}
            st.session_state["backend_validation_frame"] = empty_frame.copy()
            st.session_state["backend_deploy_ready"] = False
            st.session_state["dashboard_status"] = "BACKEND ERROR"
        backend_state = None

    return backend_state


def _resolve_dashboard_state(controls: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, bool]:
    control_signature = (
        round(float(controls["target_speed"]), 3),
        round(float(controls["interceptor_speed"]), 3),
        round(float(controls["drift_rate"]), 3),
        round(float(controls["noise_level"]), 3),
        str(controls["scenario_type"]),
        bool(controls["compare_without_drift"]),
        int(controls.get("run_seed", 61)),
    )
    controls_changed = (
        "dashboard_controls_signature" not in st.session_state
        or st.session_state["dashboard_controls_signature"] != control_signature
    )
    if controls_changed or "dashboard_simulation" not in st.session_state:
        simulation = _simulate_dashboard(
            target_speed=controls["target_speed"],
            interceptor_speed=controls["interceptor_speed"],
            drift_rate=controls["drift_rate"],
            noise_level=controls["noise_level"],
            scenario_type=controls["scenario_type"],
            compare_without_drift=controls["compare_without_drift"],
            run_seed=int(controls.get("run_seed", 61)),
        )
        st.session_state["dashboard_controls_signature"] = control_signature
        st.session_state["dashboard_simulation"] = simulation
        if "dashboard_benchmark_frame" not in st.session_state:
            st.session_state["dashboard_benchmark_frame"] = []

    if controls["run_clicked"]:
        benchmark_frame = _run_live_benchmark(
            target_speed=controls["target_speed"],
            interceptor_speed=controls["interceptor_speed"],
            drift_rate=controls["drift_rate"],
            noise_level=controls["noise_level"],
            run_seed=int(controls.get("run_seed", 61)),
        )
        st.session_state["dashboard_benchmark_frame"] = benchmark_frame.to_dict("records")
    elif controls_changed:
        st.session_state["dashboard_benchmark_frame"] = []
    st.session_state.setdefault("dashboard_status", "STOPPED")
    st.session_state.setdefault("dashboard_active_stage", "Detection")
    if controls["run_clicked"]:
        st.session_state["dashboard_replay_requested"] = True
        st.session_state["dashboard_status"] = "RUNNING"
    simulation = copy.deepcopy(st.session_state["dashboard_simulation"])
    benchmark_frame = pd.DataFrame(st.session_state["dashboard_benchmark_frame"])
    replay_requested = bool(st.session_state.get("dashboard_replay_requested", False))
    return simulation, benchmark_frame, replay_requested


def _resolve_day8_state(controls: dict[str, Any]) -> tuple[MissionReplay, MonteCarloSummary | None]:
    control_signature = (
        int(controls["num_targets"]),
        round(float(controls["drift_rate"]), 3),
        round(float(controls["noise_level"]), 3),
        round(float(controls["latency_ms"]), 1),
        round(float(controls["packet_loss_rate"]), 3),
        bool(controls["use_ekf"]),
        bool(controls["connect_airsim"]),
        int(controls.get("run_seed", 61)),
    )
    should_build_replay = bool(
        controls.get("run_clicked", False)
        or controls.get("backend_start", False)
        or controls.get("backend_preflight", False)
        or controls.get("backend_validate", False)
    )
    if (
        should_build_replay
        and (
            "day8_controls_signature" not in st.session_state
            or st.session_state["day8_controls_signature"] != control_signature
        )
    ):
        manager = AirSimMissionManager(connect=bool(controls["connect_airsim"]))
        with st.spinner("Building AirSim swarm replay and telemetry stream..."):
            replay = manager.run_replay(
                num_targets=int(controls["num_targets"]),
                use_ekf=bool(controls["use_ekf"]),
                drift_rate_mps=float(controls["drift_rate"]),
                noise_std_m=float(controls["noise_level"]),
                latency_ms=float(controls["latency_ms"]),
                packet_loss_rate=float(controls["packet_loss_rate"]),
                random_seed=int(controls.get("run_seed", 61)),
                max_steps=180,
                kill_radius_m=1.0,
                guidance_gain=6.0,
                use_ekf_anti_spoofing=bool(controls.get("use_ekf_anti_spoofing", controls.get("use_ekf", False))),
                enable_spoofing=bool(controls.get("enable_spoofing", False)),
            )
            if bool(controls.get("compare_without_drift", False)):
                st.session_state["day8_compare_replay"] = manager.run_replay(
                    num_targets=int(controls["num_targets"]),
                    use_ekf=bool(controls["use_ekf"]),
                    drift_rate_mps=0.0,
                    noise_std_m=float(controls["noise_level"]),
                    latency_ms=float(controls["latency_ms"]),
                    packet_loss_rate=float(controls["packet_loss_rate"]),
                    random_seed=int(controls.get("run_seed", 61)) + 23,
                    max_steps=180,
                    kill_radius_m=1.0,
                    guidance_gain=6.0,
                    use_ekf_anti_spoofing=bool(controls.get("use_ekf_anti_spoofing", controls.get("use_ekf", False))),
                    enable_spoofing=bool(controls.get("enable_spoofing", False)),
                )
            else:
                st.session_state["day8_compare_replay"] = MissionReplay(
                    frames=tuple(),
                    validation={},
                    map_frame=pd.DataFrame(),
                    distance_frame=pd.DataFrame(),
                    safe_intercepts=0,
                )
        cinematic_path = manager.export_cinematic_demo(replay, prefix="day8_bms_demo")
        st.session_state["day8_controls_signature"] = control_signature
        st.session_state["day8_replay"] = replay
        st.session_state["day8_cinematic_path"] = str(cinematic_path)
    elif "day8_replay" not in st.session_state:
        st.session_state["day8_controls_signature"] = control_signature
        st.session_state["day8_replay"] = MissionReplay(
            frames=tuple(),
            validation={},
            map_frame=pd.DataFrame(),
            distance_frame=pd.DataFrame(),
            safe_intercepts=0,
        )
        st.session_state["day8_compare_replay"] = MissionReplay(
            frames=tuple(),
            validation={},
            map_frame=pd.DataFrame(),
            distance_frame=pd.DataFrame(),
            safe_intercepts=0,
        )
    if controls["run_clicked"] and bool(controls["run_validation_suite"]):
        manager = AirSimMissionManager(connect=False)
        with st.spinner("Running 10-iteration EKF vs raw Monte Carlo validation..."):
            st.session_state["day8_validation"] = manager.run_monte_carlo_validation(
                iterations=10,
                num_targets=int(controls["num_targets"]),
                drift_rate_mps=float(controls["drift_rate"]),
                noise_std_m=float(controls["noise_level"]),
                packet_loss_rate=float(controls["packet_loss_rate"]),
            )
    return st.session_state["day8_replay"], st.session_state.get("day8_validation")


def _render_summary_metrics(metric_placeholders: list[Any], simulation: dict[str, Any], upto_index: int | None = None) -> None:
    backend_snapshot = ((st.session_state.get("backend_state") or {}).get("snapshot") or {})
    canonical = _resolve_backend_canonical_metrics(simulation)
    mission_final_step = upto_index is None or upto_index >= len(simulation["times"]) - 1
    if upto_index is None:
        success_value = _normalize_success_ratio(canonical.get("success_rate", simulation["success_rate"]))
        interception_time = canonical.get("earliest_interception_time_s", simulation["interception_time_s"])
        rmse_value = float(canonical.get("rmse_m", simulation["rmse_m"]))
        fps_value = simulation["mean_loop_fps"]
    else:
        sample_count = max(upto_index + 1, 1)
        success_value = 1.0 if upto_index >= len(simulation["times"]) - 1 and simulation["success"] else 0.0
        interception_time = simulation["times"][upto_index] if success_value > 0.5 else None
        rmse_value = float(np.sqrt(np.mean(np.square(np.asarray(simulation["tracking_errors"][:sample_count], dtype=float)))))
        fps_value = float(np.mean(np.asarray(simulation["fps_samples"][:sample_count], dtype=float)))
    if backend_snapshot:
        rmse_value = float(canonical.get("rmse_m", backend_snapshot.get("rmse_m", rmse_value)))
        fps_value = float(backend_snapshot.get("detection_fps_window_avg", backend_snapshot.get("detection_fps", fps_value)))
        if mission_final_step and canonical.get("success_rate") is not None:
            success_value = _normalize_success_ratio(canonical.get("success_rate"))
        elif backend_snapshot.get("mission_success") is not None and bool(backend_snapshot.get("mission_success", False)):
            success_value = max(success_value, 1.0)
        if (
            canonical.get("earliest_interception_time_s") is not None
            and (
                mission_final_step
                or bool(backend_snapshot.get("mission_success", False))
                or bool(canonical.get("mission_complete", False))
            )
        ):
            interception_time = _safe_float(canonical.get("earliest_interception_time_s"), interception_time if interception_time is not None else 0.0)
    if mission_final_step and canonical.get("success_rate") is not None:
        success_value = _normalize_success_ratio(canonical.get("success_rate"))
    if mission_final_step and canonical.get("earliest_interception_time_s") is not None:
        interception_time = _safe_float(canonical.get("earliest_interception_time_s"), 0.0)

    interception_numeric = _safe_float(interception_time, np.nan) if interception_time is not None else np.nan
    interception_label = f"{interception_numeric:.2f} s" if np.isfinite(interception_numeric) else "n/a"
    metric_placeholders[0].metric("Success Rate", f"{success_value * 100:.0f}%")
    metric_placeholders[1].metric("Interception Time", interception_label)
    metric_placeholders[2].metric("RMSE", f"{rmse_value:.3f} m")
    metric_placeholders[3].metric("FPS", f"{fps_value:.2f}")


def _sync_live_control_state(controls: dict[str, Any]) -> None:
    st.session_state["live_control_state"] = {
        "target_speed": float(controls["target_speed"]),
        "interceptor_speed": float(controls["interceptor_speed"]),
        "interceptor_mass_kg": float(controls.get("interceptor_mass_kg", 6.5)),
        "drift_rate": float(controls["drift_rate"]),
        "noise_level": float(controls["noise_level"]),
        "num_targets": int(controls["num_targets"]),
        "latency_ms": float(controls["latency_ms"]),
        "packet_loss_rate": float(controls["packet_loss_rate"]),
        "use_ekf": bool(controls["use_ekf"]),
        "connect_airsim": bool(controls["connect_airsim"]),
        "run_validation_suite": bool(controls["run_validation_suite"]),
        "scenario_type": str(controls["scenario_type"]),
        "compare_without_drift": bool(controls["compare_without_drift"]),
        "animate_frontend": bool(controls["animate_frontend"]),
        "playback_fps_hz": float(controls["playback_fps_hz"]),
        "run_seed": int(controls.get("run_seed", 61)),
        "use_ekf": True,
        "enable_spoofing": True,
        "backend_host": str(controls["backend_host"]),
        "backend_port": int(controls["backend_port"]),
    }


def _render_tactical_header(
    header_placeholder: Any,
    simulation: dict[str, Any],
    status: str,
    active_stage: str,
    upto_index: int | None = None,
) -> None:
    backend_snapshot = ((st.session_state.get("backend_state") or {}).get("snapshot") or {})
    if upto_index is None:
        detection_fps = float(simulation["mean_loop_fps"])
        rmse_value = float(simulation["rmse_m"])
    else:
        limit = max(upto_index + 1, 1)
        detection_fps = float(np.mean(np.asarray(simulation["fps_samples"][:limit], dtype=float)))
        rmse_value = float(np.sqrt(np.mean(np.square(np.asarray(simulation["tracking_errors"][:limit], dtype=float)))))
    if backend_snapshot:
        detection_fps = float(backend_snapshot.get("detection_fps_window_avg", backend_snapshot.get("detection_fps", detection_fps)))
        rmse_value = float(backend_snapshot.get("rmse_m", rmse_value))
    backend_heartbeat_live = bool(backend_snapshot.get("heartbeat_live", st.session_state.get("backend_heartbeat_live", False)))
    failsafe_land = bool(st.session_state.get("backend_failsafe_land", False))
    heartbeat_class = "heartbeat-live" if backend_heartbeat_live else "heartbeat-idle"
    heartbeat_label = "FAILSAFE" if failsafe_land else ("LIVE" if backend_heartbeat_live else "STANDBY")
    with header_placeholder.container():
        st.markdown('<div class="tactical-panel">', unsafe_allow_html=True)
        left_col, center_col, right_col = st.columns([1.0, 0.8, 1.5])
        with left_col:
            st.markdown(
                f"""
                <div class="hud-card">
                  <div class="hud-label">System Status</div>
                  <div class="hud-value">{str(status).upper()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with center_col:
            st.markdown(
                f"""
                <div class="hud-card hud-heartbeat">
                  <div class="hud-label">Mission Heartbeat</div>
                  <div class="heartbeat-row"><span class="heartbeat-dot {heartbeat_class}"></span><span class="hud-inline">{heartbeat_label}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right_col:
            st.markdown(
                f"""
                <div class="hud-card">
                  <div class="hud-label">Detection FPS</div>
                  <div class="hud-inline">{detection_fps:.2f}</div>
                  <div class="hud-label">RMSE</div>
                  <div class="hud-inline">{rmse_value:.3f} m</div>
                  <div class="hud-label">Current Stage</div>
                  <div class="hud-inline">{active_stage}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


def _render_tactical_map(map_placeholder: Any, simulation: dict[str, Any], upto_index: int | None = None) -> None:
    deck = _build_map_deck(simulation, upto_index=upto_index)
    with map_placeholder.container():
        st.caption("Interceptor range heatmap: brighter cyan indicates higher interceptor occupancy/coverage confidence.")
        st.pydeck_chart(deck, width="stretch")


def _render_telemetry_overlay(telemetry_placeholder: Any, simulation: dict[str, Any], upto_index: int | None = None) -> None:
    if upto_index is None:
        upto_index = len(simulation["times"]) - 1
    interceptor_positions = np.asarray(simulation["interceptor_positions"], dtype=float)
    if len(interceptor_positions) == 0:
        return
    point = interceptor_positions[upto_index]
    speed = float(simulation["commanded_speed"][upto_index])
    st.session_state["dashboard_telemetry"] = {
        "x_m": float(point[0]),
        "y_m": float(point[1]),
        "z_m": float(point[2]),
        "velocity_mps": speed,
    }
    with telemetry_placeholder.container():
        st.markdown(
            f"""
            <div class="telemetry-overlay tactical-panel">
              <div class="hud-label">Interceptor Telemetry</div>
              <div class="telemetry-grid">
                <div>X: {point[0]:7.2f} m</div>
                <div>Y: {point[1]:7.2f} m</div>
                <div>Z: {point[2]:7.2f} m</div>
                <div>Velocity: {speed:6.2f} m/s</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


@st.cache_data(show_spinner=False)
def _simulate_dashboard(
    target_speed: float,
    interceptor_speed: float,
    drift_rate: float,
    noise_level: float,
    scenario_type: str,
    compare_without_drift: bool,
    run_seed: int = 61,
) -> dict[str, Any]:
    primary = _run_dashboard_case(
        target_speed=target_speed,
        interceptor_speed=interceptor_speed,
        drift_rate=drift_rate,
        noise_level=noise_level,
        scenario_type=scenario_type,
        random_seed=int(run_seed),
    )
    comparison = None
    if compare_without_drift:
        comparison = _run_dashboard_case(
            target_speed=target_speed,
            interceptor_speed=interceptor_speed,
            drift_rate=0.0,
            noise_level=noise_level,
            scenario_type=scenario_type,
            random_seed=int(run_seed) + 37,
        )
    primary["comparison"] = comparison
    return primary


def _run_dashboard_case(
    target_speed: float,
    interceptor_speed: float,
    drift_rate: float,
    noise_level: float,
    scenario_type: str,
    random_seed: int = 61,
) -> dict[str, Any]:
    config = load_config(ROOT / "configs" / "default.yaml")
    seed_value = int(random_seed)
    rng = np.random.default_rng(seed_value)
    _apply_day4_tuning(config)
    config.setdefault("system", {})["random_seed"] = seed_value
    config.setdefault("tracking", {})["mode"] = "kalman"
    config.setdefault("control", {})["mode"] = "mpc"
    config.setdefault("mission", {})["max_steps"] = 160
    config.setdefault("planning", {})["max_speed_mps"] = float(interceptor_speed)
    config.setdefault("perception", {})["synthetic_measurement_noise_std_m"] = float(noise_level)
    config.setdefault("tracking", {})["measurement_noise"] = max(float(noise_level * 0.4), 0.1)
    config.setdefault("tracking", {})["process_noise"] = max(float(noise_level * 0.12), 0.05)
    config.setdefault("navigation", {})["gps_drift_rate_mps"] = float(drift_rate)
    config.setdefault("navigation", {})["gps_noise_std_m"] = max(1.2, float(1.1 + 0.5 * noise_level))
    config.setdefault("prediction", {})["process_noise"] = max(
        float(config["prediction"].get("process_noise", 0.15)),
        0.12 + 0.45 * float(drift_rate) + 0.08 * float(noise_level),
    )
    config["prediction"]["acceleration_damping"] = max(0.35, 0.60 - 0.18 * float(drift_rate))
    scenario_spawn_jitter = np.array(
        [
            float(rng.uniform(-30.0, 30.0)),
            float(rng.uniform(-35.0, 35.0)),
            float(rng.uniform(-8.0, 8.0)),
        ],
        dtype=float,
    )
    scenario_velocity_jitter = np.array(
        [
            float(rng.uniform(-0.9, 0.9)),
            float(rng.uniform(-1.2, 1.2)),
            0.0,
        ],
        dtype=float,
    )
    config.setdefault("simulation", {})["target_initial_velocity"] = (
        np.array([-float(target_speed), 1.8, 0.0], dtype=float) + scenario_velocity_jitter
    ).astype(float).tolist()
    config["simulation"]["target_process_noise_std_mps2"] = max(0.10, float(0.10 + 0.08 * noise_level))
    config["simulation"]["wind_disturbance_std_mps2"] = max(0.04, float(0.03 + 0.04 * noise_level))
    config["simulation"]["target_initial_position"] = (
        np.array([280.0, 145.0, 120.0], dtype=float) + scenario_spawn_jitter
    ).astype(float).tolist()

    if scenario_type == "fast":
        fast_velocity = np.array([-float(max(target_speed, 8.5)), 3.0, 0.0], dtype=float) + scenario_velocity_jitter
        config["simulation"]["target_initial_velocity"] = fast_velocity.astype(float).tolist()
        config["simulation"]["target_max_acceleration_mps2"] = 4.5
    elif scenario_type == "noisy":
        config["perception"]["synthetic_measurement_noise_std_m"] = max(noise_level, 0.9)
        config["tracking"]["measurement_noise"] = max(config["tracking"]["measurement_noise"], 0.45)
        config["simulation"]["target_initial_position"] = (
            np.array([255.0, 135.0, 120.0], dtype=float) + scenario_spawn_jitter
        ).astype(float).tolist()
    elif scenario_type == "drift":
        config["navigation"]["gps_drift_rate_mps"] = max(drift_rate, 0.35)
        config["navigation"]["measurement_noise_scale"] = 1.10
        config["planning"]["desired_intercept_distance_m"] = max(
            float(config["planning"].get("desired_intercept_distance_m", 10.25)),
            12.0,
        )
    elif scenario_type == "zigzag":
        config.setdefault("scenario", {})["zigzag_amplitude_mps"] = 1.8
        config["scenario"]["zigzag_frequency_hz"] = 0.18
        zigzag_velocity = np.array([-float(target_speed), 0.0, 0.0], dtype=float) + scenario_velocity_jitter
        config["simulation"]["target_initial_velocity"] = zigzag_velocity.astype(float).tolist()

    env = DroneInterceptionEnv(config)
    detector = TargetDetector(config)
    tracker = TargetTracker(config)
    predictor = TargetPredictor(config)
    planner = InterceptPlanner(config)
    controller = InterceptionController(config)
    navigator = GPSIMUKalmanFusion(config)
    cost_model = InterceptionCostModel.from_config(config)
    constraint_envelope = load_constraint_envelope(config)
    spoof_toolkit = DP5CoordinateSpoofingToolkit(
        safe_zone_position=np.array([35.0, 0.0, 110.0], dtype=float),
        min_rate_mps=0.2,
        max_rate_mps=max(float(drift_rate), 0.5),
        noise_std_m=max(float(noise_level) * 0.12, 0.04),
        random_seed=int(random_seed),
    )
    spoof_profile = AttackProfile(name=f"dashboard_{scenario_type}", mode="directed", onset_time_s=0.0)

    observation = env.reset()
    dt = float(config["mission"]["time_step"])
    max_steps = int(config["mission"]["max_steps"])
    start_time = time.perf_counter()

    times: list[float] = []
    target_positions: list[list[float]] = []
    interceptor_positions: list[list[float]] = []
    drifted_positions: list[list[float]] = []
    fused_positions: list[list[float]] = []
    predicted_positions: list[list[float]] = []
    distances: list[float] = []
    control_effort: list[float] = []
    commanded_speed: list[float] = []
    stage_costs: list[float] = []
    tracking_errors: list[float] = []
    fps_samples: list[float] = []
    detections: list[list[float]] = []
    detection_events: list[dict[str, Any]] = []

    success = False
    interception_time_s: float | None = None
    final_distance_m = float("inf")

    for step in range(max_steps):
        if scenario_type == "zigzag":
            env.target_state.velocity[1] = 1.8 * math.sin(2.0 * math.pi * 0.18 * step * dt)

        navigation_state = navigator.update(observation["sensor_packet"])
        interceptor_estimate = TargetState(
            position=navigation_state.position.copy(),
            velocity=navigation_state.velocity.copy(),
            covariance=None if navigation_state.covariance is None else navigation_state.covariance.copy(),
            timestamp=navigation_state.timestamp,
            metadata=dict(navigation_state.metadata),
        )
        detection = detector.detect(observation)
        track = tracker.update(detection)
        prediction = predictor.predict(track)
        plan = planner.plan(interceptor_estimate, prediction)
        plan.metadata["current_target_position"] = track.position.copy()
        plan.metadata["current_target_velocity"] = track.velocity.copy()
        plan.metadata["current_target_acceleration"] = (
            track.acceleration.copy() if track.acceleration is not None else np.zeros(3, dtype=float)
        )
        plan.metadata["current_target_covariance"] = (
            None if track.covariance is None else np.asarray(track.covariance, dtype=float).copy()
        )
        tracking_error_m = float(np.linalg.norm(track.position - env.target_state.position))
        plan.metadata["tracking_error_m"] = tracking_error_m
        command = controller.compute_command(interceptor_estimate, plan)
        observation, done, info = env.step(command)

        constraint_status = ConstraintStatus(
            velocity_clipped=bool(command.metadata.get("velocity_clipped", False)),
            acceleration_clipped=bool(command.metadata.get("acceleration_clipped", False)),
            tracking_ok=tracking_error_m <= constraint_envelope.tracking_precision_m,
            drift_rate_in_bounds=bool(navigation_state.metadata.get("drift_rate_in_bounds", True)),
            safety_override=bool(command.metadata.get("safety_override", False)),
            distance_to_target_m=float(info["distance_to_target"]),
        )
        stage_cost = cost_model.stage_cost(
            interceptor_position=env.interceptor_state.position,
            target_position=env.target_state.position,
            control_input=command.acceleration_command if command.acceleration_command is not None else command.velocity_command,
            constraint_status=constraint_status,
            uncertainty_term=float(plan.metadata.get("uncertainty_trace", 0.0)),
        )
        spoof_sample = spoof_toolkit.sample(
            true_position=env.target_state.position,
            interceptor_position=env.interceptor_state.position,
            time_s=float(observation["time"][0]),
            attack_profile=spoof_profile,
        )
        drifted_position = spoof_sample.spoofed_position
        elapsed = max(time.perf_counter() - start_time, 1e-6)
        loop_fps = float((step + 1) / elapsed)

        times.append(float(observation["time"][0]))
        target_positions.append(env.target_state.position.astype(float).tolist())
        interceptor_positions.append(env.interceptor_state.position.astype(float).tolist())
        drifted_positions.append(np.asarray(drifted_position, dtype=float).tolist())
        fused_positions.append(track.position.astype(float).tolist())
        predicted_positions.append(_resolve_prediction_position(prediction, track).astype(float).tolist())
        distances.append(float(info["distance_to_target"]))
        control_effort.append(float(np.linalg.norm(command.acceleration_command if command.acceleration_command is not None else np.zeros(3, dtype=float))))
        commanded_speed.append(float(np.linalg.norm(command.velocity_command)))
        stage_costs.append(float(stage_cost))
        tracking_errors.append(tracking_error_m)
        fps_samples.append(loop_fps)
        detections.append(np.asarray(detection.position, dtype=float).tolist())
        weighted_summary = score_weighted_detection_targets(detection)
        weighted_targets = weighted_summary.get("weighted_targets", [])
        primary_target = weighted_targets[0] if weighted_targets else {}
        detection_events.append(
            {
                "step": int(step),
                "time_s": float(observation["time"][0]),
                "weighted_total_score": float(weighted_summary.get("weighted_total_score", 0.0)),
                "normalized_weighted_score": float(weighted_summary.get("normalized_weighted_score", 0.0)),
                "drone_focus_score": float(weighted_summary.get("drone_focus_score", 0.0)),
                "target_count": int(weighted_summary.get("target_count", 0)),
                "drone_count": int(weighted_summary.get("drone_count", 0)),
                "mean_confidence": float(weighted_summary.get("mean_confidence", 0.0)),
                "primary_label": str(primary_target.get("class_name", "unknown")),
                "class_histogram": dict(weighted_summary.get("class_histogram", {})),
            }
        )
        final_distance_m = float(info["distance_to_target"])

        if done:
            success = final_distance_m <= float(config["planning"]["desired_intercept_distance_m"])
            interception_time_s = float(observation["time"][0]) if success else None
            break

    mean_loop_fps = float(len(times) / max(time.perf_counter() - start_time, 1e-6)) if times else 0.0
    return {
        "times": times,
        "target_positions": target_positions,
        "interceptor_positions": interceptor_positions,
        "drifted_positions": drifted_positions,
        "fused_positions": fused_positions,
        "predicted_positions": predicted_positions,
        "distances": distances,
        "control_effort": control_effort,
        "commanded_speed": commanded_speed,
        "stage_costs": stage_costs,
        "tracking_errors": tracking_errors,
        "fps_samples": fps_samples,
        "detections": detections,
        "detection_events": detection_events,
        "success": success,
        "success_rate": 1.0 if success else 0.0,
        "interception_time_s": interception_time_s,
        "mean_loop_fps": mean_loop_fps,
        "rmse_m": float(np.sqrt(np.mean(np.square(np.asarray(tracking_errors, dtype=float))))) if tracking_errors else 0.0,
        "final_distance_m": final_distance_m,
        "scenario_type": scenario_type,
        "drift_rate": drift_rate,
        "noise_level": noise_level,
        "target_speed": target_speed,
        "interceptor_speed": interceptor_speed,
    }


def _resolve_prediction_position(prediction: Any, track: TargetState) -> np.ndarray:
    if isinstance(prediction, list):
        if prediction:
            first_state = prediction[0]
            if hasattr(first_state, "position"):
                return np.asarray(first_state.position, dtype=float)
        return np.asarray(track.position, dtype=float)
    predicted_states = getattr(prediction, "predicted_states", None)
    if predicted_states:
        first_state = predicted_states[0]
        if hasattr(first_state, "position"):
            return np.asarray(first_state.position, dtype=float)
    return np.asarray(track.position, dtype=float)


@st.cache_data(show_spinner=False)
def _run_live_benchmark(
    target_speed: float,
    interceptor_speed: float,
    drift_rate: float,
    noise_level: float,
    run_seed: int = 61,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scenario_order = ["normal", "fast", "noisy", "drift", "zigzag"]
    for scenario_index, scenario_type in enumerate(scenario_order):
        result = _run_dashboard_case(
            target_speed=target_speed,
            interceptor_speed=interceptor_speed,
            drift_rate=drift_rate,
            noise_level=noise_level,
            scenario_type=scenario_type,
            random_seed=int(run_seed) + int(scenario_index * 131),
        )
        rows.append(
            {
                "scenario": scenario_type,
                "success": "YES" if result["success"] else "NO",
                "interception_time_s": round(result["interception_time_s"], 2) if result["interception_time_s"] is not None else None,
                "rmse_m": round(result["rmse_m"], 3),
                "mean_loop_fps": round(result["mean_loop_fps"], 2),
                "final_distance_m": round(result["final_distance_m"], 3),
                "noise_level": round(result["noise_level"], 2),
            }
        )
    return pd.DataFrame(rows)


def _extract_backend_replay_plot_data(upto_index: int | None = None) -> dict[str, Any] | None:
    backend_state = st.session_state.get("backend_state", {})
    if not isinstance(backend_state, dict):
        return None
    replay_data = backend_state.get("replay_data", {})
    if not isinstance(replay_data, dict):
        return None
    frames = replay_data.get("frames", [])
    if not isinstance(frames, list) or len(frames) < 2:
        return None
    limit = len(frames) if upto_index is None else min(max(int(upto_index) + 1, 1), len(frames))
    frames = frames[:limit]

    times: list[float] = []
    target_positions: list[list[float]] = []
    interceptor_positions: list[list[float]] = []
    drifted_positions: list[list[float]] = []
    fused_positions: list[list[float]] = []
    predicted_positions: list[list[float]] = []
    distances: list[float] = []
    control_effort: list[float] = []
    commanded_speed: list[float] = []
    stage_costs: list[float] = []
    tracking_errors: list[float] = []
    fps_samples: list[float] = []
    detections: list[list[float]] = []

    for frame in frames:
        if not isinstance(frame, dict):
            continue
        interceptor_position = np.asarray(frame.get("interceptor_position", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        interceptor_velocity = np.asarray(frame.get("interceptor_velocity", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        targets = frame.get("targets", [])
        if not isinstance(targets, list) or not targets:
            continue
        active_target = str(frame.get("active_target", "")).strip()
        selected_target: dict[str, Any] | None = None
        for candidate in targets:
            if isinstance(candidate, dict) and str(candidate.get("name", "")).strip() == active_target:
                selected_target = candidate
                break
        if selected_target is None:
            selected_target = next((item for item in targets if isinstance(item, dict)), None)
        if selected_target is None:
            continue

        true_position = np.asarray(selected_target.get("true_position", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        spoofed_position = np.asarray(selected_target.get("spoofed_position", true_position), dtype=float).reshape(3)
        estimated_position = np.asarray(selected_target.get("estimated_position", true_position), dtype=float).reshape(3)
        detection_position = np.asarray(selected_target.get("spoofed_position", true_position), dtype=float).reshape(3)

        times.append(float(frame.get("time_s", len(times) * 0.05)))
        interceptor_positions.append(interceptor_position.astype(float).tolist())
        target_positions.append(true_position.astype(float).tolist())
        drifted_positions.append(spoofed_position.astype(float).tolist())
        fused_positions.append(estimated_position.astype(float).tolist())
        predicted_positions.append(estimated_position.astype(float).tolist())
        detections.append(detection_position.astype(float).tolist())

        distance_m = float(np.linalg.norm(true_position - interceptor_position))
        distances.append(distance_m)
        fps_samples.append(float(frame.get("detection_fps", 0.0)))
        commanded_speed.append(float(np.linalg.norm(interceptor_velocity)))
        control_effort.append(0.0)
        stage_costs.append(0.0)
        tracking_errors.append(float(np.linalg.norm(estimated_position - true_position)))

    if len(times) < 2:
        return None

    snapshot = backend_state.get("snapshot", {}) if isinstance(backend_state.get("snapshot"), dict) else {}
    mission_complete = str(snapshot.get("status", "")).lower() == "complete"
    return {
        "times": times,
        "target_positions": target_positions,
        "interceptor_positions": interceptor_positions,
        "drifted_positions": drifted_positions,
        "fused_positions": fused_positions,
        "predicted_positions": predicted_positions,
        "distances": distances,
        "control_effort": control_effort,
        "commanded_speed": commanded_speed,
        "stage_costs": stage_costs,
        "tracking_errors": tracking_errors,
        "fps_samples": fps_samples,
        "detections": detections,
        "success": bool(mission_complete),
        "success_rate": 1.0 if mission_complete else 0.0,
        "interception_time_s": _safe_float(snapshot.get("time_s", times[-1] if times else 0.0), 0.0),
        "mean_loop_fps": float(np.mean(np.asarray(fps_samples, dtype=float))) if fps_samples else 0.0,
        "rmse_m": float(np.sqrt(np.mean(np.square(np.asarray(tracking_errors, dtype=float))))) if tracking_errors else 0.0,
        "final_distance_m": float(distances[-1]) if distances else 0.0,
        "scenario_type": "backend_replay",
        "comparison": None,
        "origin_lat_lon": tuple(replay_data.get("origin_lat_lon", (37.7749, -122.4194))),
        "source": "backend_replay",
    }


def _build_3d_figure(simulation: dict[str, Any], upto_index: int | None = None) -> go.Figure:
    plot_data = _extract_backend_replay_plot_data(upto_index)
    source = "Backend Replay"
    if plot_data is None:
        plot_data = simulation
        source = "Frontend Simulation"
        if upto_index is not None:
            limit = upto_index + 1
            plot_data = dict(plot_data)
            for key in ("target_positions", "interceptor_positions", "drifted_positions", "fused_positions"):
                plot_data[key] = plot_data[key][:limit]

    target = np.asarray(plot_data["target_positions"], dtype=float)
    interceptor = np.asarray(plot_data["interceptor_positions"], dtype=float)
    drifted = np.asarray(plot_data["drifted_positions"], dtype=float)
    fused = np.asarray(plot_data["fused_positions"], dtype=float)
    comparison = plot_data.get("comparison")
    spoof_offset_m = np.linalg.norm(drifted - target, axis=1) if len(target) == len(drifted) and len(target) > 0 else np.asarray([0.0], dtype=float)

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=target[:, 0], y=target[:, 1], mode="lines", name="Target True", line=dict(color="#ff845c", width=6)))
    figure.add_trace(go.Scatter(x=interceptor[:, 0], y=interceptor[:, 1], mode="lines", name="Interceptor", line=dict(color="#5ed2ff", width=6)))
    figure.add_trace(go.Scatter(x=fused[:, 0], y=fused[:, 1], mode="lines", name="EKF Filtered", line=dict(color="#73f0a0", width=5)))
    figure.add_trace(go.Scatter(x=drifted[:, 0], y=drifted[:, 1], mode="lines", name="Raw Drifted", line=dict(color="#ff4b4b", width=4, dash="dash")))
    if len(target) > 0 and len(drifted) > 0:
        connector_step = max(1, int(len(target) / 10))
        connector_x: list[float | None] = []
        connector_y: list[float | None] = []
        for index in range(0, len(target), connector_step):
            connector_x.extend([float(target[index, 0]), float(drifted[index, 0]), None])
            connector_y.extend([float(target[index, 1]), float(drifted[index, 1]), None])
        if connector_x and connector_y:
            figure.add_trace(
                go.Scatter(
                    x=connector_x,
                    y=connector_y,
                    mode="lines",
                    name="True↔Spoof Offset",
                    line=dict(color="#ffd36a", width=1.8, dash="dot"),
                    opacity=0.65,
                )
            )

    if len(target) >= 2 and len(interceptor) >= 2:
        target_head = target[-1]
        interceptor_head = interceptor[-1]
        target_vec = target[-1] - target[-2]
        interceptor_vec = interceptor[-1] - interceptor[-2]
        figure.add_trace(
            go.Scatter(
                x=[target_head[0], target_head[0] + target_vec[0] * 4.0],
                y=[target_head[1], target_head[1] + target_vec[1] * 4.0],
                mode="lines+markers",
                name="Target Velocity",
                line=dict(color="#ff845c", width=8),
                marker=dict(size=3, color="#ff845c"),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[interceptor_head[0], interceptor_head[0] + interceptor_vec[0] * 4.0],
                y=[interceptor_head[1], interceptor_head[1] + interceptor_vec[1] * 4.0],
                mode="lines+markers",
                name="Interceptor Velocity",
                line=dict(color="#5ed2ff", width=8),
                marker=dict(size=3, color="#5ed2ff"),
            )
        )

    if len(target) > 0 and len(drifted) > 0:
        current_true = target[-1]
        current_spoof = drifted[-1]
        figure.add_trace(
            go.Scatter(
                x=[float(current_true[0]), float(current_spoof[0])],
                y=[float(current_true[1]), float(current_spoof[1])],
                mode="lines",
                name="Current Spoof Offset",
                line=dict(color="#ffd36a", width=3.5, dash="dashdot"),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[float(current_true[0])],
                y=[float(current_true[1])],
                mode="markers+text",
                name="Actual Position (Current)",
                text=["Actual"],
                textposition="top center",
                marker=dict(size=12, color="#ff845c", line=dict(color="#ffe9df", width=1.6), symbol="circle"),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[float(current_spoof[0])],
                y=[float(current_spoof[1])],
                mode="markers+text",
                name="Spoofed Position (Current)",
                text=["Spoofed"],
                textposition="bottom center",
                marker=dict(size=12, color="#ff4b4b", line=dict(color="#ffd9d9", width=1.6), symbol="x"),
            )
        )

    if comparison is not None:
        comparison_interceptor = np.asarray(comparison["interceptor_positions"], dtype=float)
        if upto_index is not None:
            comparison_interceptor = comparison_interceptor[: upto_index + 1]
        figure.add_trace(
            go.Scatter(
                x=comparison_interceptor[:, 0],
                y=comparison_interceptor[:, 1],
                mode="lines",
                name="Without Drift",
                line=dict(color="#7bf7a5", width=4, dash="dot"),
            )
        )

    if plot_data["success"] and len(interceptor) > 0 and (upto_index is None or upto_index >= len(plot_data["times"]) - 1):
        point = interceptor[-1]
        figure.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode="markers", name="Intercept", marker=dict(size=7, color="#ff5f99", symbol="diamond")))

    frames = []
    for index in range(len(target)):
        frame_traces = [
            go.Scatter(x=target[: index + 1, 0], y=target[: index + 1, 1], mode="lines", line=dict(color="#ff845c", width=6)),
            go.Scatter(x=interceptor[: index + 1, 0], y=interceptor[: index + 1, 1], mode="lines", line=dict(color="#5ed2ff", width=6)),
            go.Scatter(x=fused[: index + 1, 0], y=fused[: index + 1, 1], mode="lines", line=dict(color="#73f0a0", width=5)),
            go.Scatter(x=drifted[: index + 1, 0], y=drifted[: index + 1, 1], mode="lines", line=dict(color="#ff4b4b", width=4, dash="dash")),
        ]
        if comparison is not None:
            comparison_interceptor = np.asarray(comparison["interceptor_positions"], dtype=float)
            frame_traces.append(
                go.Scatter(
                    x=comparison_interceptor[: index + 1, 0],
                    y=comparison_interceptor[: index + 1, 1],
                    mode="lines",
                    line=dict(color="#7bf7a5", width=4, dash="dot"),
                )
            )
        frames.append(go.Frame(data=frame_traces, name=str(index)))
    figure.frames = frames

    figure.update_layout(
        height=720,
        margin=dict(l=0, r=0, t=30, b=0),
        title=(
            f"Trajectory Plot Source: {source} | "
            f"Spoof Offset Mean={float(np.mean(spoof_offset_m)):.2f} m, "
            f"Current={float(spoof_offset_m[-1]) if len(spoof_offset_m) > 0 else 0.0:.2f} m"
        ),
        xaxis=dict(title="X [m]", gridcolor="#2a3948"),
        yaxis=dict(title="Y [m]", gridcolor="#2a3948", scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", y=1.02, x=0.0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=1.02,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 55, "redraw": True}, "fromcurrent": True}],
                    )
                ],
            )
        ],
    )
    return figure


def _build_spoofing_diagnostics_figure(simulation: dict[str, Any], upto_index: int | None = None) -> go.Figure:
    plot_data = _extract_backend_replay_plot_data(upto_index)
    source = "Backend Replay"
    if plot_data is None:
        plot_data = simulation
        source = "Frontend Simulation"
        if upto_index is not None:
            limit = upto_index + 1
            plot_data = dict(plot_data)
            for key in ("times", "target_positions", "drifted_positions", "fused_positions"):
                if key in plot_data:
                    plot_data[key] = plot_data[key][:limit]
    times = np.asarray(plot_data.get("times", []), dtype=float)
    target = np.asarray(plot_data.get("target_positions", []), dtype=float)
    drifted = np.asarray(plot_data.get("drifted_positions", []), dtype=float)
    fused = np.asarray(plot_data.get("fused_positions", []), dtype=float)
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("Spoof Offset Magnitude", "True vs Spoofed XY Delta"),
    )
    if len(times) > 0 and len(target) == len(drifted) and len(target) > 0:
        spoof_offset_m = np.linalg.norm(drifted - target, axis=1)
        dx = drifted[:, 0] - target[:, 0]
        dy = drifted[:, 1] - target[:, 1]
        figure.add_trace(
            go.Scatter(
                x=times,
                y=spoof_offset_m,
                mode="lines",
                name="|Spoof - Actual| [m]",
                line=dict(color="#ffd36a", width=3),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=times,
                y=dx,
                mode="lines",
                name="Delta X [m]",
                line=dict(color="#ff4b4b", width=2.5),
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=times,
                y=dy,
                mode="lines",
                name="Delta Y [m]",
                line=dict(color="#ff845c", width=2.5),
            ),
            row=2,
            col=1,
        )
        if len(fused) == len(target):
            ekf_error = np.linalg.norm(fused - target, axis=1)
            figure.add_trace(
                go.Scatter(
                    x=times,
                    y=ekf_error,
                    mode="lines",
                    name="EKF Error [m]",
                    line=dict(color="#73f0a0", width=2.2),
                ),
                row=1,
                col=1,
            )
    else:
        figure.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No spoof diagnostics available.",
            showarrow=False,
            font=dict(color="#b8c7d6", size=13),
        )
    figure.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=45, b=25),
        title=f"Spoofing Diagnostics Source: {source}",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.18, x=0.0),
    )
    figure.update_yaxes(title_text="Offset [m]", row=1, col=1)
    figure.update_yaxes(title_text="Delta [m]", row=2, col=1)
    figure.update_xaxes(title_text="Time [s]", row=2, col=1)
    return figure


def _render_3d_mission_panel(
    placeholder: Any,
    simulation: dict[str, Any],
    upto_index: int | None,
    key_suffix: str,
) -> None:
    with placeholder.container():
        st.plotly_chart(
            _build_3d_figure(simulation, upto_index=upto_index),
            width="stretch",
            key=f"three_d_{key_suffix}",
            config={"displayModeBar": True},
        )
        st.plotly_chart(
            _build_spoofing_diagnostics_figure(simulation, upto_index=upto_index),
            width="stretch",
            key=f"three_d_spoof_diag_{key_suffix}",
            config={"displayModeBar": False},
        )


def _build_interceptor_range_heatmap_points(
    interceptor_positions: np.ndarray,
    origin_lat_lon: tuple[float, float],
    range_radius_m: float = 70.0,
) -> pd.DataFrame:
    if len(interceptor_positions) == 0:
        return pd.DataFrame(columns=["lon", "lat", "weight"])
    sample_count = min(int(len(interceptor_positions)), 120)
    sample_indices = np.linspace(0, len(interceptor_positions) - 1, sample_count, dtype=int)
    bearings_deg = np.asarray([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0], dtype=float)
    radius_scales = np.asarray([0.25, 0.50, 0.80, 1.00], dtype=float)
    rows: list[dict[str, float]] = []
    for order, idx in enumerate(sample_indices):
        anchor = np.asarray(interceptor_positions[int(idx)], dtype=float).reshape(3)
        age_weight = float((order + 1) / max(len(sample_indices), 1))
        anchor_geo = _positions_to_geo(np.asarray([anchor], dtype=float), origin_lat_lon=origin_lat_lon)[0]
        rows.append({"lon": float(anchor_geo[0]), "lat": float(anchor_geo[1]), "weight": float(1.8 * age_weight)})
        for radius_scale in radius_scales:
            radius_m = float(max(range_radius_m * radius_scale, 1e-3))
            for bearing_deg in bearings_deg:
                bearing_rad = math.radians(float(bearing_deg))
                probe = anchor.copy()
                probe[0] = float(anchor[0] + radius_m * math.cos(bearing_rad))
                probe[1] = float(anchor[1] + radius_m * math.sin(bearing_rad))
                probe_geo = _positions_to_geo(np.asarray([probe], dtype=float), origin_lat_lon=origin_lat_lon)[0]
                rows.append(
                    {
                        "lon": float(probe_geo[0]),
                        "lat": float(probe_geo[1]),
                        "weight": float((1.0 - 0.62 * radius_scale) * age_weight),
                    }
                )
    return pd.DataFrame(rows)


def _build_map_deck(simulation: dict[str, Any], upto_index: int | None = None) -> pdk.Deck:
    plot_data = _extract_backend_replay_plot_data(upto_index)
    if plot_data is None:
        if upto_index is None:
            upto_index = len(simulation["times"]) - 1
        limit = max(upto_index + 1, 1)
        target_positions = np.asarray(simulation["target_positions"][:limit], dtype=float)
        interceptor_positions = np.asarray(simulation["interceptor_positions"][:limit], dtype=float)
        drifted_positions = np.asarray(simulation["drifted_positions"][:limit], dtype=float)
        fused_positions = np.asarray(simulation["fused_positions"][:limit], dtype=float)
        origin_lat_lon = (37.7749, -122.4194)
    else:
        target_positions = np.asarray(plot_data["target_positions"], dtype=float)
        interceptor_positions = np.asarray(plot_data["interceptor_positions"], dtype=float)
        drifted_positions = np.asarray(plot_data["drifted_positions"], dtype=float)
        fused_positions = np.asarray(plot_data["fused_positions"], dtype=float)
        origin_lat_lon = tuple(plot_data.get("origin_lat_lon", (37.7749, -122.4194)))

    target_path = _positions_to_geo(target_positions, origin_lat_lon=origin_lat_lon)
    interceptor_path = _positions_to_geo(interceptor_positions, origin_lat_lon=origin_lat_lon)
    drifted_path = _positions_to_geo(drifted_positions, origin_lat_lon=origin_lat_lon)
    fused_path = _positions_to_geo(fused_positions, origin_lat_lon=origin_lat_lon)
    current_target = target_path[-1]
    current_interceptor = interceptor_path[-1]
    current_drifted = drifted_path[-1]
    current_fused = fused_path[-1]

    path_frame = pd.DataFrame(
        [
            {"name": "Target True", "path": target_path, "color": [255, 132, 92], "width": 5},
            {"name": "Interceptor", "path": interceptor_path, "color": [0, 242, 255], "width": 5},
            {"name": "Raw Drift", "path": drifted_path, "color": [255, 75, 75], "width": 4},
            {"name": "EKF Fused", "path": fused_path, "color": [115, 240, 160], "width": 4},
        ]
    )
    points_frame = pd.DataFrame(
        [
            {"name": "Target True", "lon": current_target[0], "lat": current_target[1], "color": [255, 132, 92], "radius": 110},
            {"name": "Interceptor", "lon": current_interceptor[0], "lat": current_interceptor[1], "color": [0, 242, 255], "radius": 110},
            {"name": "Raw Drift", "lon": current_drifted[0], "lat": current_drifted[1], "color": [255, 75, 75], "radius": 88},
            {"name": "EKF Fused", "lon": current_fused[0], "lat": current_fused[1], "color": [115, 240, 160], "radius": 88},
        ]
    )
    range_radius_m = 65.0
    heatmap_frame = _build_interceptor_range_heatmap_points(
        interceptor_positions=interceptor_positions,
        origin_lat_lon=origin_lat_lon,
        range_radius_m=range_radius_m,
    )
    range_frame = pd.DataFrame(
        [
            {
                "name": "Interceptor Coverage Range",
                "lon": float(current_interceptor[0]),
                "lat": float(current_interceptor[1]),
                "radius": float(range_radius_m),
                "color": [0, 242, 255, 28],
            }
        ]
    )
    layers = [
        pdk.Layer(
            "HeatmapLayer",
            data=heatmap_frame,
            get_position="[lon, lat]",
            get_weight="weight",
            radiusPixels=52,
            intensity=1.1,
            threshold=0.03,
            colorRange=[
                [7, 21, 46],
                [0, 86, 164],
                [0, 185, 255],
                [112, 255, 250],
                [198, 255, 242],
                [255, 255, 255],
            ],
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=range_frame,
            get_position="[lon, lat]",
            get_radius="radius",
            get_fill_color="color",
            get_line_color=[0, 242, 255, 165],
            stroked=True,
            filled=True,
            line_width_min_pixels=2,
        ),
        pdk.Layer(
            "PathLayer",
            data=path_frame,
            get_path="path",
            get_color="color",
            width_scale=10,
            width_min_pixels=2,
            get_width="width",
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=points_frame,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius="radius",
            pickable=True,
        ),
    ]
    center_lon = float((current_target[0] + current_interceptor[0]) / 2.0)
    center_lat = float((current_target[1] + current_interceptor[1]) / 2.0)
    return pdk.Deck(
        map_provider=DECK_MAP_PROVIDER,
        map_style=DECK_MAP_STYLE,
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=13.2, pitch=50, bearing=18),
        layers=layers,
        tooltip={"text": "{name}"},
    )


def _positions_to_geo(
    positions: np.ndarray,
    origin_lat_lon: tuple[float, float] = (37.7749, -122.4194),
) -> list[list[float]]:
    if len(positions) == 0:
        return [[float(origin_lat_lon[1]), float(origin_lat_lon[0])]]
    geo_points: list[list[float]] = []
    for point in positions:
        lat, lon, _ = local_position_to_lla(np.asarray(point, dtype=float), origin_lat_lon)
        geo_points.append([lon, lat])
    return geo_points


def _build_timeseries_figure(simulation: dict[str, Any], upto_index: int | None = None) -> go.Figure:
    plot_data = _extract_backend_replay_plot_data(upto_index)
    source = "Backend Replay"
    if plot_data is None:
        plot_data = simulation
        source = "Frontend Simulation"
        if upto_index is None:
            upto_index = len(simulation["times"]) - 1
        limit = upto_index + 1
    else:
        limit = len(plot_data["times"])
    times = np.asarray(plot_data["times"][:limit], dtype=float)
    figure = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            f"Distance vs Time ({source})",
            "Cost vs Time",
            "Velocity vs Time",
            "Control Effort vs Time",
        ),
        vertical_spacing=0.08,
    )
    figure.add_trace(go.Scatter(x=times, y=plot_data["distances"][:limit], mode="lines", name="Distance", line=dict(color="#5ed2ff", width=3)), row=1, col=1)
    figure.add_trace(go.Scatter(x=times, y=plot_data["stage_costs"][:limit], mode="lines", name="Cost", line=dict(color="#ff845c", width=3)), row=2, col=1)
    figure.add_trace(go.Scatter(x=times, y=plot_data["commanded_speed"][:limit], mode="lines", name="Velocity", line=dict(color="#7bf7a5", width=3)), row=3, col=1)
    figure.add_trace(go.Scatter(x=times, y=plot_data["control_effort"][:limit], mode="lines", name="Control Effort", line=dict(color="#ffd36a", width=3)), row=4, col=1)
    figure.update_xaxes(title_text="Time [s]", row=4, col=1)
    figure.update_yaxes(title_text="Distance [m]", row=1, col=1)
    figure.update_yaxes(title_text="Cost", row=2, col=1)
    figure.update_yaxes(title_text="Velocity [m/s]", row=3, col=1)
    figure.update_yaxes(title_text="Effort", row=4, col=1)
    figure.update_layout(height=920, margin=dict(l=30, r=20, t=70, b=35), showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return figure


def _build_comparison_figure(simulation: dict[str, Any], upto_index: int | None = None) -> go.Figure:
    comparison = simulation["comparison"]
    if upto_index is None:
        upto_index = min(len(simulation["times"]), len(comparison["times"])) - 1
    limit = upto_index + 1
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=simulation["times"][:limit], y=simulation["distances"][:limit], mode="lines", name="With Drift", line=dict(color="#ff845c", width=3)))
    figure.add_trace(go.Scatter(x=comparison["times"][:limit], y=comparison["distances"][:limit], mode="lines", name="Without Drift", line=dict(color="#7bf7a5", width=3, dash="dot")))
    figure.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Time [s]",
        yaxis_title="Distance [m]",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _build_live_rmse_noise_figure(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["noise_level"],
            y=frame["rmse_m"],
            mode="markers+lines+text",
            text=frame["scenario"],
            textposition="top center",
            marker=dict(size=11, color="#ff845c"),
            line=dict(color="#ffb28e", width=2),
            name="RMSE",
        )
    )
    figure.update_layout(
        title="Live RMSE vs Noise",
        xaxis_title="Noise Level",
        yaxis_title="RMSE [m]",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _build_live_success_figure(frame: pd.DataFrame) -> go.Figure:
    values = [1.0 if item == "YES" else 0.0 for item in frame["success"]]
    figure = go.Figure(
        go.Bar(
            x=frame["scenario"],
            y=values,
            marker_color=["#73f0a0" if value > 0.5 else "#ff8e8e" for value in values],
            name="Success Rate",
        )
    )
    figure.update_layout(
        title="Live Success by Scenario",
        yaxis_title="Success",
        yaxis=dict(range=[0, 1.05]),
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _build_live_fps_figure(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=frame["scenario"],
            y=frame["mean_loop_fps"],
            marker_color="#5ed2ff",
            name="FPS",
        )
    )
    figure.update_layout(
        title="Live FPS by Scenario",
        yaxis_title="FPS",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _build_live_terminal_distance_figure(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=frame["scenario"],
            y=frame["final_distance_m"],
            marker_color="#ffd36a",
            name="Final Distance",
        )
    )
    figure.update_layout(
        title="Live Final Distance by Scenario",
        yaxis_title="Distance [m]",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _render_static_frontend(
    simulation: dict[str, Any],
    benchmark_frame: pd.DataFrame,
    day8_replay: MissionReplay,
    day8_validation: MonteCarloSummary | None,
    header_placeholder: Any,
    status: str,
    three_d_placeholder: Any,
    live_panel_placeholder: Any,
    map_placeholder: Any,
    telemetry_placeholder: Any,
    day8_map_placeholder: Any,
    day8_telemetry_placeholder: Any,
    realtime_placeholder: Any,
    comparison_placeholder: Any,
    scenario_results_placeholder: Any,
    mission_demo_placeholder: Any,
    analytics_placeholder: Any,
    day8_distance_placeholder: Any,
    day8_validation_plot_placeholder: Any,
    day8_validation_placeholder: Any,
    day8_architecture_placeholder: Any,
    backend_status_placeholder: Any,
    backend_stream_placeholder: Any,
    backend_state: dict[str, Any] | None,
    backend_host: str,
    backend_port: int,
) -> None:
    active_validation = backend_state.get("validation") if backend_state and backend_state.get("validation") is not None else day8_validation
    _render_tactical_header(
        header_placeholder=header_placeholder,
        simulation=simulation,
        status=status,
        active_stage=str(st.session_state.get("dashboard_active_stage", "Detection")),
    )
    _render_3d_mission_panel(
        placeholder=three_d_placeholder,
        simulation=simulation,
        upto_index=None,
        key_suffix="static",
    )
    with live_panel_placeholder.container():
        _render_live_simulation_panel(simulation, backend_host=backend_host, backend_port=backend_port)
    _render_tactical_map(map_placeholder, simulation)
    _render_telemetry_overlay(telemetry_placeholder, simulation)
    _render_day8_map(day8_map_placeholder, day8_replay)
    _render_day8_telemetry(day8_telemetry_placeholder, day8_replay)
    realtime_placeholder.plotly_chart(
        _build_timeseries_figure(simulation),
        width="stretch",
        key="timeseries_static",
        config={"displayModeBar": False},
    )
    if simulation["comparison"] is not None:
        comparison_placeholder.plotly_chart(
            _build_comparison_figure(simulation),
            width="stretch",
            key="comparison_static",
            config={"displayModeBar": False},
        )
    else:
        comparison_placeholder.info("Enable `Compare With / Without Drift` in the control panel to visualize both trajectories.")
    results_frame = _build_day8_target_results_frame(
        day8_replay,
        backend_state.get("validation") if backend_state and backend_state.get("validation") is not None else day8_validation,
        simulation=simulation,
    )
    if _results_ready_for_display(backend_state, results_frame):
        with scenario_results_placeholder.container():
            _render_results_data_tiles(results_frame=results_frame)
    else:
        scenario_results_placeholder.info(
            "Scenario results will display as soon as all required per-target backend metrics are populated."
        )
    with mission_demo_placeholder.container():
        _show_live_mission_demo(simulation, backend_state=backend_state, key_suffix="static")
    _render_backend_status_panel(
        backend_status_placeholder,
        backend_state,
        backend_host,
        backend_port,
        simulation=simulation,
    )
    _render_backend_live_stream(backend_stream_placeholder, backend_host, backend_port)
    with analytics_placeholder.container():
        _render_live_analytics_gallery(benchmark_frame, key_suffix="static")
    _render_day8_distance_plot(day8_distance_placeholder, day8_replay)
    _render_day8_validation_plot(day8_validation_plot_placeholder, active_validation, key_suffix="static")
    _render_day8_validation(day8_validation_placeholder, active_validation)
    _render_day8_architecture(day8_architecture_placeholder, day8_replay, active_validation)


def _run_live_simulation_callback(
    simulation: dict[str, Any],
    benchmark_frame: pd.DataFrame,
    day8_replay: MissionReplay,
    day8_validation: MonteCarloSummary | None,
    header_placeholder: Any,
    metric_placeholders: list[Any],
    three_d_placeholder: Any,
    live_panel_placeholder: Any,
    map_placeholder: Any,
    telemetry_placeholder: Any,
    day8_map_placeholder: Any,
    day8_telemetry_placeholder: Any,
    realtime_placeholder: Any,
    comparison_placeholder: Any,
    scenario_results_placeholder: Any,
    mission_demo_placeholder: Any,
    analytics_placeholder: Any,
    day8_distance_placeholder: Any,
    day8_validation_plot_placeholder: Any,
    day8_validation_placeholder: Any,
    day8_architecture_placeholder: Any,
    backend_status_placeholder: Any,
    backend_stream_placeholder: Any,
    backend_state: dict[str, Any] | None,
    backend_host: str,
    backend_port: int,
    playback_fps_hz: float,
) -> None:
    active_validation = backend_state.get("validation") if backend_state and backend_state.get("validation") is not None else day8_validation
    total_steps = len(simulation["times"])
    if total_steps <= 0:
        _render_static_frontend(
            simulation=simulation,
            benchmark_frame=benchmark_frame,
            day8_replay=day8_replay,
            day8_validation=day8_validation,
            header_placeholder=header_placeholder,
            status="STOPPED",
            three_d_placeholder=three_d_placeholder,
            live_panel_placeholder=live_panel_placeholder,
            map_placeholder=map_placeholder,
            telemetry_placeholder=telemetry_placeholder,
            day8_map_placeholder=day8_map_placeholder,
            day8_telemetry_placeholder=day8_telemetry_placeholder,
            realtime_placeholder=realtime_placeholder,
            comparison_placeholder=comparison_placeholder,
            scenario_results_placeholder=scenario_results_placeholder,
            mission_demo_placeholder=mission_demo_placeholder,
            analytics_placeholder=analytics_placeholder,
            day8_distance_placeholder=day8_distance_placeholder,
            day8_validation_plot_placeholder=day8_validation_plot_placeholder,
            day8_validation_placeholder=day8_validation_placeholder,
            day8_architecture_placeholder=day8_architecture_placeholder,
            backend_status_placeholder=backend_status_placeholder,
            backend_stream_placeholder=backend_stream_placeholder,
            backend_state=backend_state,
            backend_host=backend_host,
            backend_port=backend_port,
        )
        return

    progress_bar = st.progress(0.0, text="Executing autonomy stack replay...")
    replay_steps = _replay_step_indices(total_steps, max_updates=26)
    backend_throughput_hz = float(((backend_state or {}).get("snapshot") or {}).get("backend_throughput_fps", playback_fps_hz))
    effective_fps_hz = float(playback_fps_hz)
    reduced_visuals = False
    if backend_throughput_hz < 15.0:
        effective_fps_hz = max(10.0, min(float(playback_fps_hz), backend_throughput_hz * 0.9))
        reduced_visuals = True
    playback_interval_s = 1.0 / max(effective_fps_hz, 1.0)

    heavy_stride = 1 if len(replay_steps) <= 12 else (2 if len(replay_steps) <= 20 else 3)
    for replay_index, step_index in enumerate(replay_steps, start=1):
        ratio = replay_index / max(len(replay_steps), 1)
        heavy_update = bool(
            replay_index == 1
            or replay_index == len(replay_steps)
            or ((replay_index - 1) % heavy_stride == 0)
        )
        partial_rows = max(1, int(np.ceil(ratio * max(len(benchmark_frame), 1))))
        partial_frame = _slice_benchmark_frame(benchmark_frame, partial_rows)
        active_stage = _current_stage_name(simulation, step_index)
        st.session_state["dashboard_active_stage"] = active_stage
        st.session_state["playback_step"] = step_index
        st.session_state["dashboard_status"] = "RUNNING"

        _render_tactical_header(
            header_placeholder=header_placeholder,
            simulation=simulation,
            status="RUNNING",
            active_stage=active_stage,
            upto_index=step_index,
        )

        _render_summary_metrics(metric_placeholders, simulation, upto_index=step_index)
        if heavy_update and ((not reduced_visuals) or (step_index % 2 == 0)):
            _render_3d_mission_panel(
                placeholder=three_d_placeholder,
                simulation=simulation,
                upto_index=step_index,
                key_suffix=f"replay_{step_index}",
            )
        with live_panel_placeholder.container():
            _render_live_simulation_panel(
                simulation,
                backend_host=backend_host,
                backend_port=backend_port,
                active_step_override=step_index,
            )
        if heavy_update:
            _render_tactical_map(map_placeholder, simulation, upto_index=step_index)
            _render_telemetry_overlay(telemetry_placeholder, simulation, upto_index=step_index)
            _render_day8_map(day8_map_placeholder, day8_replay, upto_index=min(step_index, max(len(day8_replay.frames) - 1, 0)))
            _render_day8_telemetry(day8_telemetry_placeholder, day8_replay, upto_index=min(step_index, max(len(day8_replay.frames) - 1, 0)))
            realtime_placeholder.plotly_chart(
                _build_timeseries_figure(simulation, upto_index=step_index),
                width="stretch",
                key=f"timeseries_replay_{step_index}",
                config={"displayModeBar": False},
            )
            if simulation["comparison"] is not None and step_index >= len(simulation["times"]) - 1:
                comparison_upto = min(step_index, len(simulation["comparison"]["times"]) - 1)
                comparison_placeholder.plotly_chart(
                    _build_comparison_figure(simulation, upto_index=comparison_upto),
                    width="stretch",
                    key=f"comparison_replay_{step_index}",
                    config={"displayModeBar": False},
                )
            else:
                comparison_placeholder.info("Comparison mode will render automatically when the active mission finishes.")
        replay_results = _build_day8_target_results_frame(
            day8_replay,
            active_validation,
            upto_index=min(step_index, max(len(day8_replay.frames) - 1, 0)),
            simulation=simulation,
        )
        if ratio >= 1.0 and _results_ready_for_display(backend_state, replay_results):
            with scenario_results_placeholder.container():
                _render_results_data_tiles(results_frame=replay_results)
        else:
            if ratio < 1.0:
                scenario_results_placeholder.info(
                    f"Scenario results unlock at replay completion (100%). Current progress: {ratio * 100:.0f}%."
                )
            else:
                scenario_results_placeholder.info(
                    "Replay is complete. Waiting for backend per-target metrics to finish populating."
                )
        with mission_demo_placeholder.container():
            _show_live_mission_demo(simulation, backend_state=backend_state, upto_index=step_index, key_suffix=f"replay_{step_index}")
        _render_backend_status_panel(
            backend_status_placeholder,
            backend_state,
            backend_host,
            backend_port,
            simulation=simulation,
        )
        _render_backend_live_stream(backend_stream_placeholder, backend_host, backend_port)
        if heavy_update and ((not reduced_visuals) or (step_index % 2 == 0)):
            with analytics_placeholder.container():
                _render_live_analytics_gallery(partial_frame, key_suffix=f"replay_{step_index}")
        if heavy_update:
            _render_day8_distance_plot(day8_distance_placeholder, day8_replay, upto_index=min(step_index, max(len(day8_replay.frames) - 1, 0)))
            _render_day8_validation_plot(day8_validation_plot_placeholder, active_validation, key_suffix=f"replay_{step_index}")
            _render_day8_validation(day8_validation_placeholder, active_validation)
            _render_day8_architecture(day8_architecture_placeholder, day8_replay, active_validation)
        replay_text = f"Executing autonomy stack replay... {active_stage} @ {effective_fps_hz:.1f} Hz"
        if reduced_visuals:
            replay_text += " | reduced visuals"
        progress_bar.progress(ratio, text=replay_text)
        time.sleep(playback_interval_s)

    progress_bar.empty()
    st.session_state["dashboard_status"] = "STOPPED"
    _render_static_frontend(
        simulation=simulation,
        benchmark_frame=benchmark_frame,
        day8_replay=day8_replay,
        day8_validation=active_validation,
        header_placeholder=header_placeholder,
        status="STOPPED",
        three_d_placeholder=three_d_placeholder,
        live_panel_placeholder=live_panel_placeholder,
        map_placeholder=map_placeholder,
        telemetry_placeholder=telemetry_placeholder,
        day8_map_placeholder=day8_map_placeholder,
        day8_telemetry_placeholder=day8_telemetry_placeholder,
        realtime_placeholder=realtime_placeholder,
        comparison_placeholder=comparison_placeholder,
        scenario_results_placeholder=scenario_results_placeholder,
        mission_demo_placeholder=mission_demo_placeholder,
        analytics_placeholder=analytics_placeholder,
        day8_distance_placeholder=day8_distance_placeholder,
        day8_validation_plot_placeholder=day8_validation_plot_placeholder,
        day8_validation_placeholder=day8_validation_placeholder,
        day8_architecture_placeholder=day8_architecture_placeholder,
        backend_status_placeholder=backend_status_placeholder,
        backend_stream_placeholder=backend_stream_placeholder,
        backend_state=backend_state,
        backend_host=backend_host,
        backend_port=backend_port,
    )


def _build_live_analytics_frame_from_results() -> pd.DataFrame:
    results_table = st.session_state.get("results_table")
    if not isinstance(results_table, pd.DataFrame) or results_table.empty:
        return pd.DataFrame()
    if "target" not in results_table.columns:
        return pd.DataFrame()

    snapshot = ((st.session_state.get("backend_state") or {}).get("snapshot") or {})
    detection_fps = _safe_float(
        snapshot.get("detection_fps_window_avg", snapshot.get("detection_fps", 0.0)),
        0.0,
    )
    default_distance = _safe_float(snapshot.get("active_distance_m", 0.0), 0.0)
    noise_level = _safe_float((st.session_state.get("live_control_state") or {}).get("noise_level", 0.0), 0.0)
    target_distance_by_id: dict[str, float] = {}
    snapshot_targets = snapshot.get("targets", [])
    if isinstance(snapshot_targets, list):
        for index, target in enumerate(snapshot_targets):
            if not isinstance(target, dict):
                continue
            target_id = str(target.get("target_id") or target.get("name") or f"Target_{index + 1}")
            target_distance_by_id[target_id] = _safe_float(target.get("distance_m", default_distance), default_distance)

    rows: list[dict[str, Any]] = []
    for _, row in results_table.iterrows():
        target_id = str(row.get("target", "unknown"))
        rmse_m = _safe_float(row.get("rmse_m", 0.0), 0.0)
        interception_time_s = _safe_float(row.get("interception_time_s", 0.0), 0.0)
        mission_success_probability = _safe_float(row.get("mission_success_probability", 0.0), 0.0)
        success_label = "YES" if mission_success_probability >= 0.50 else "NO"
        rows.append(
            {
                "scenario": target_id,
                "success": success_label,
                "interception_time_s": interception_time_s,
                "rmse_m": rmse_m,
                "mean_loop_fps": detection_fps,
                "final_distance_m": target_distance_by_id.get(target_id, default_distance),
                "noise_level": noise_level,
            }
        )
    return pd.DataFrame(rows)


def _build_mission_history_frame() -> pd.DataFrame:
    mission_history = st.session_state.get("mission_history", [])
    if not isinstance(mission_history, list) or not mission_history:
        return pd.DataFrame()
    history_frame = pd.DataFrame(mission_history)
    if "run_seq" not in history_frame.columns:
        history_frame["run_seq"] = np.arange(1, len(history_frame) + 1, dtype=int)
    return history_frame


def _build_weighted_detection_frame() -> pd.DataFrame:
    simulation = st.session_state.get("dashboard_simulation")
    if not isinstance(simulation, dict):
        return pd.DataFrame()
    events = simulation.get("detection_events")
    if not isinstance(events, list) or not events:
        return pd.DataFrame()
    frame = pd.DataFrame(events)
    required = {
        "time_s",
        "weighted_total_score",
        "normalized_weighted_score",
        "drone_focus_score",
        "target_count",
        "drone_count",
        "mean_confidence",
    }
    if not required.issubset(set(frame.columns)):
        return pd.DataFrame()
    return frame


def _build_weighted_detection_figure(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["time_s"],
            y=frame["normalized_weighted_score"],
            mode="lines+markers",
            name="Normalized Weighted Score",
            line=dict(color="#00F0FF", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["time_s"],
            y=frame["drone_focus_score"],
            mode="lines+markers",
            name="Drone Focus Score",
            line=dict(color="#73F0A0", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["time_s"],
            y=frame["mean_confidence"],
            mode="lines+markers",
            name="Mean Confidence",
            line=dict(color="#FFB800", width=2),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Weighted Object and Drone Detection Confidence Over Time",
        xaxis_title="Mission Time [s]",
        yaxis=dict(title="Weighted Score", rangemode="tozero"),
        yaxis2=dict(title="Mean Confidence", overlaying="y", side="right", rangemode="tozero"),
        height=340,
        margin=dict(l=20, r=20, t=45, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    return fig


def _build_weighted_detection_class_bar(frame: pd.DataFrame) -> go.Figure:
    histogram = Counter()
    for item in frame.get("class_histogram", []):
        if isinstance(item, dict):
            for label, count in item.items():
                histogram[str(label)] += int(count)
    labels = list(histogram.keys()) or ["drone"]
    counts = [histogram.get(label, 0) for label in labels] or [0]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=counts,
                marker=dict(color="#5ED2FF"),
                name="Detections",
            )
        ]
    )
    fig.update_layout(
        title="Object-Class Detection Distribution",
        xaxis_title="Class",
        yaxis_title="Count",
        height=320,
        margin=dict(l=20, r=20, t=45, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _render_live_analytics_gallery(frame: pd.DataFrame, key_suffix: str) -> None:
    canonical_backend_frame = _build_live_analytics_frame_from_results()
    if not canonical_backend_frame.empty:
        frame = canonical_backend_frame
    elif frame.empty:
        frame = canonical_backend_frame
    if frame.empty:
        st.info("Live analytics will populate after a completed backend mission with per-target results.")
        return
    backend_snapshot = ((st.session_state.get("backend_state") or {}).get("snapshot") or {})
    status_endpoint = ((st.session_state.get("backend_state") or {}).get("status_endpoint") or {})
    mission_insights = status_endpoint.get("mission_insights", {}) if isinstance(status_endpoint, dict) else {}
    global_metrics = mission_insights.get("global_metrics", {}) if isinstance(mission_insights, dict) else {}
    results_table = st.session_state.get("results_table")
    if isinstance(results_table, pd.DataFrame) and not results_table.empty and "rmse_m" in results_table.columns:
        try:
            table_rmse = float(results_table["rmse_m"].astype(float).mean())
            backend_snapshot = dict(backend_snapshot)
            backend_snapshot["rmse_m"] = table_rmse
            backend_snapshot["rmse_measured_true_m"] = table_rmse
        except Exception:
            pass
    metric_cols = st.columns(6)
    metric_cols[0].metric("Backend RMSE", f"{float(backend_snapshot.get('rmse_m', 0.0)):.3f} m")
    metric_cols[1].metric("Innovation", f"{float(backend_snapshot.get('innovation_m', 0.0)):.3f} m")
    metric_cols[2].metric("Confidence", f"{float(backend_snapshot.get('confidence_score', 0.0)) * 100:.1f}%")
    metric_cols[3].metric("Measured RMSE", f"{float(backend_snapshot.get('rmse_measured_true_m', backend_snapshot.get('rmse_m', 0.0))):.3f} m")
    metric_cols[4].metric("Threat Risk P90", f"{float(global_metrics.get('risk_index_p90', 0.0)):.3f}")
    metric_cols[5].metric("Spoof Detect Rate", f"{float(global_metrics.get('spoofing_detection_rate', 0.0)) * 100:.1f}%")
    tabs = st.tabs(["Noise / RMSE", "Scenario Success", "Runtime / Terminal", "Mission Trends", "Weighted Detection"])
    with tabs[0]:
        st.plotly_chart(
            _build_live_rmse_noise_figure(frame),
            width="stretch",
            key=f"rmse_chart_{key_suffix}",
            config={"displayModeBar": False},
        )
    with tabs[1]:
        success_left, success_right = st.columns([1.05, 0.95])
        with success_left:
            st.plotly_chart(
                _build_live_success_figure(frame),
                width="stretch",
                key=f"success_chart_{key_suffix}",
                config={"displayModeBar": False},
            )
        with success_right:
            st.dataframe(
                frame[["scenario", "success", "interception_time_s", "rmse_m"]].rename(
                    columns={
                        "scenario": "Scenario",
                        "success": "Success",
                        "interception_time_s": "Intercept [s]",
                        "rmse_m": "RMSE [m]",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
    with tabs[2]:
        runtime_left, runtime_right = st.columns(2)
        with runtime_left:
            st.plotly_chart(
                _build_live_fps_figure(frame),
                width="stretch",
                key=f"fps_chart_{key_suffix}",
                config={"displayModeBar": False},
            )
        with runtime_right:
            st.plotly_chart(
                _build_live_terminal_distance_figure(frame),
                width="stretch",
                key=f"terminal_chart_{key_suffix}",
                config={"displayModeBar": False},
            )
    with tabs[3]:
        history_frame = _build_mission_history_frame()
        if history_frame.empty:
            st.info("Run at least one mission to populate command-readiness trends.")
        else:
            trend_figure = go.Figure()
            trend_figure.add_trace(
                go.Scatter(
                    x=history_frame["run_seq"],
                    y=history_frame["readiness_score"],
                    mode="lines+markers",
                    name="Readiness Score",
                    line=dict(color="#8ef7a6", width=2),
                )
            )
            trend_figure.add_trace(
                go.Scatter(
                    x=history_frame["run_seq"],
                    y=history_frame["weighted_success_pct"],
                    mode="lines+markers",
                    name="Weighted Success [%]",
                    line=dict(color="#5ed2ff", width=2),
                )
            )
            trend_figure.add_trace(
                go.Scatter(
                    x=history_frame["run_seq"],
                    y=history_frame["rmse_p95_m"],
                    mode="lines+markers",
                    name="RMSE P95 [m]",
                    line=dict(color="#ffd36a", width=2),
                    yaxis="y2",
                )
            )
            trend_figure.update_layout(
                title="Mission Performance Trend",
                xaxis_title="Run Sequence",
                yaxis=dict(title="Score / Success [%]", rangemode="tozero"),
                yaxis2=dict(title="RMSE P95 [m]", overlaying="y", side="right", rangemode="tozero"),
                height=340,
                margin=dict(l=20, r=20, t=45, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
            )
            st.plotly_chart(trend_figure, width="stretch", key=f"mission_trends_{key_suffix}", config={"displayModeBar": False})
            st.dataframe(
                history_frame[[
                    "run_seq",
                    "run_time",
                    "target_count",
                    "readiness_score",
                    "security_posture",
                    "quality_gate_passed",
                    "weighted_success_pct",
                    "rmse_p95_m",
                    "risk_p90",
                    "spoof_detect_rate_pct",
                ]].rename(
                    columns={
                        "run_seq": "Run #",
                        "run_time": "Time",
                        "target_count": "Targets",
                        "readiness_score": "Readiness",
                        "security_posture": "Posture",
                        "quality_gate_passed": "Quality Gate",
                        "weighted_success_pct": "Weighted Success [%]",
                        "rmse_p95_m": "RMSE P95 [m]",
                        "risk_p90": "Risk P90",
                        "spoof_detect_rate_pct": "Spoof Detect [%]",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
    with tabs[4]:
        detection_frame = _build_weighted_detection_frame()
        if detection_frame.empty:
            st.info("Weighted detection analytics will populate after running a live mission.")
        else:
            metrics = st.columns(4)
            metrics[0].metric(
                "Weighted Score Avg",
                f"{float(pd.to_numeric(detection_frame['normalized_weighted_score'], errors='coerce').mean()):.3f}",
            )
            metrics[1].metric(
                "Drone Focus Peak",
                f"{float(pd.to_numeric(detection_frame['drone_focus_score'], errors='coerce').max()):.3f}",
            )
            metrics[2].metric(
                "Drone Detections (sum)",
                f"{int(pd.to_numeric(detection_frame['drone_count'], errors='coerce').fillna(0).sum())}",
            )
            metrics[3].metric(
                "Mean Confidence Avg",
                f"{float(pd.to_numeric(detection_frame['mean_confidence'], errors='coerce').mean()):.3f}",
            )
            left, right = st.columns([1.2, 0.8])
            with left:
                st.plotly_chart(
                    _build_weighted_detection_figure(detection_frame),
                    width="stretch",
                    key=f"weighted_detection_{key_suffix}",
                    config={"displayModeBar": False},
                )
            with right:
                st.plotly_chart(
                    _build_weighted_detection_class_bar(detection_frame),
                    width="stretch",
                    key=f"weighted_detection_class_{key_suffix}",
                    config={"displayModeBar": False},
                )
                table_cols = [
                    "time_s",
                    "primary_label",
                    "target_count",
                    "drone_count",
                    "normalized_weighted_score",
                    "mean_confidence",
                ]
                preview = detection_frame[table_cols].tail(12).copy()
                preview.columns = [
                    "Time [s]",
                    "Primary Label",
                    "Targets",
                    "Drones",
                    "Weighted Score",
                    "Mean Conf.",
                ]
                st.dataframe(preview, width="stretch", hide_index=True)


def _slice_benchmark_frame(frame: pd.DataFrame, rows: int) -> pd.DataFrame:
    return frame.iloc[:rows].reset_index(drop=True)


def _validation_metric(validation: MonteCarloSummary | dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if validation is None:
        return float(default)
    if isinstance(validation, MonteCarloSummary):
        return float(getattr(validation, key, default))
    return float(validation.get(key, default))


def _validation_passed(validation: MonteCarloSummary | dict[str, Any] | None) -> bool:
    if validation is None:
        return False
    if isinstance(validation, dict) and "validation_success" in validation:
        return bool(validation.get("validation_success", False))
    return bool(
        _validation_metric(validation, "ekf_success_rate") >= 1.0
        and _validation_metric(validation, "ekf_mean_miss_distance_m") <= 0.5
    )


def _render_results_data_tiles(
    results_frame: pd.DataFrame,
    max_tiles: int = 8,
) -> None:
    if not isinstance(results_frame, pd.DataFrame) or results_frame.empty:
        st.info("No per-target mission outputs are available yet.")
        return

    required = [
        "target",
        "ekf_success_rate",
        "interception_time_s",
        "rmse_m",
        "mission_success_probability",
    ]
    if not set(required).issubset(set(results_frame.columns)):
        st.dataframe(results_frame, width="stretch", hide_index=True, height=360)
        return

    tiles = results_frame.copy().reset_index(drop=True)
    tile_count = min(max(int(max_tiles), 1), len(tiles))
    columns = st.columns(min(4, tile_count))
    for index in range(tile_count):
        row = tiles.iloc[index]
        success_pct = _normalize_success_ratio(row.get("ekf_success_rate", 0.0)) * 100.0
        hit_prob_pct = _normalize_success_ratio(row.get("mission_success_probability", 0.0)) * 100.0
        intercept_s = _safe_float(row.get("interception_time_s", np.nan), np.nan)
        rmse_m = _safe_float(row.get("rmse_m", np.nan), np.nan)
        tile_class = "data-tile-ok" if success_pct >= 50.0 else "data-tile-warn"
        intercept_label = f"{intercept_s:.2f} s" if np.isfinite(intercept_s) else "n/a"
        rmse_label = f"{rmse_m:.3f} m" if np.isfinite(rmse_m) else "n/a"
        with columns[index % len(columns)]:
            st.markdown(
                f"""
                <div class="data-tile {tile_class}">
                  <div class="data-tile-target">{str(row.get("target", f"Target_{index + 1}"))}</div>
                  <div class="data-tile-grid">
                    <div><span>EKF Success</span><strong>{success_pct:.1f}%</strong></div>
                    <div><span>Intercept</span><strong>{intercept_label}</strong></div>
                    <div><span>RMSE</span><strong>{rmse_label}</strong></div>
                    <div><span>Hit Prob</span><strong>{hit_prob_pct:.1f}%</strong></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("Per-target detailed table", expanded=False):
        st.dataframe(
            results_frame,
            width="stretch",
            hide_index=True,
            height=360,
        )


def _results_ready_for_display(
    backend_state: dict[str, Any] | None,
    results_frame: pd.DataFrame,
) -> bool:
    if not isinstance(results_frame, pd.DataFrame) or results_frame.empty:
        return False
    required_columns = [
        "target",
        "ekf_success_rate",
        "interception_time_s",
        "rmse_m",
        "measured_rmse_m",
        "mission_success_probability",
        "guidance_efficiency_mps2",
        "spoofing_variance",
        "compute_latency_ms",
        "energy_consumption_j",
    ]
    for column in required_columns:
        if column not in results_frame.columns:
            return False
    if bool(results_frame[required_columns].isna().any().any()):
        return False

    expected_targets = int(st.session_state.get("mission_expected_targets", 0) or 0)
    if expected_targets <= 0 and isinstance(backend_state, dict):
        snapshot = backend_state.get("snapshot", {}) if isinstance(backend_state.get("snapshot"), dict) else {}
        expected_targets = int(snapshot.get("target_count", 0) or 0)
    if expected_targets <= 0:
        control_state = st.session_state.get("live_control_state", {})
        expected_targets = max(int(control_state.get("num_targets", len(results_frame))), 1)
    if len(results_frame) < expected_targets:
        return False

    return True


def _build_day8_target_results_frame(
    replay: MissionReplay,
    validation_report: MonteCarloSummary | dict[str, Any] | None,
    upto_index: int | None = None,
    simulation: dict[str, Any] | None = None,
) -> pd.DataFrame:
    del replay, validation_report, upto_index, simulation
    session_results = st.session_state.get("results_table")
    if isinstance(session_results, pd.DataFrame) and not session_results.empty:
        return session_results.copy()
    if isinstance(session_results, list) and session_results:
        return _mission_results_to_dataframe(session_results)

    active_validation = st.session_state.get("backend_validation")
    if isinstance(active_validation, MonteCarloSummary):
        active_validation = {
            "per_target_summary": list(active_validation.per_target_summary),
        }
    if isinstance(active_validation, dict):
        validation_rows = _derive_results_from_validation(active_validation)
        if validation_rows:
            return _mission_results_to_dataframe(validation_rows)

    backend_state = st.session_state.get("backend_state", {})
    status_payload = (
        backend_state.get("status_endpoint", {})
        if isinstance(backend_state, dict)
        else {}
    )
    if isinstance(status_payload, dict):
        status_rows = _derive_results_from_status_targets(status_payload)
        if status_rows:
            return _mission_results_to_dataframe(status_rows)

    return _mission_results_to_dataframe([])


def _render_day8_map(day8_map_placeholder: Any, replay: MissionReplay, upto_index: int | None = None) -> None:
    if len(replay.frames) == 0:
        day8_map_placeholder.info("No AirSim swarm replay available.")
        return
    if upto_index is None:
        upto_index = len(replay.frames) - 1
    step = min(max(int(upto_index), 0), len(replay.frames) - 1)
    current_frame = replay.frames[step]
    history = replay.frames[max(0, step - 49) : step + 1]
    if not history:
        day8_map_placeholder.info("Awaiting AirSim swarm coordinates.")
        return
    validation = replay.validation
    origin_lat_lon = tuple(validation.get("origin_lat_lon", (37.7749, -122.4194)))
    zone = validation.get("no_fly_zone", {})
    no_fly_lon, no_fly_lat = _positions_to_geo(
        np.asarray([[zone.get("center_x_m", 35.0), zone.get("center_y_m", 0.0), zone.get("center_z_m", 110.0)]], dtype=float),
        origin_lat_lon=origin_lat_lon,
    )[0]

    point_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []

    interceptor_path = _positions_to_geo(np.asarray([frame.interceptor_position for frame in history], dtype=float), origin_lat_lon=origin_lat_lon)
    path_rows.append({"name": "Interceptor", "path": interceptor_path, "color": [0, 242, 255]})
    current_interceptor_lon, current_interceptor_lat = interceptor_path[-1]
    point_rows.append(
        {
            "name": "Interceptor",
            "lon": current_interceptor_lon,
            "lat": current_interceptor_lat,
            "color": [0, 242, 255],
            "radius": 95,
        }
    )

    target_names = [target.name for target in current_frame.targets]
    for target_name in target_names:
        estimated_positions: list[np.ndarray] = []
        spoofed_positions: list[np.ndarray] = []
        current_target = None
        for frame in history:
            target = next((item for item in frame.targets if item.name == target_name), None)
            if target is None:
                continue
            current_target = target
            estimated_positions.append(np.asarray(target.filtered_estimate, dtype=float))
            spoofed_positions.append(np.asarray(target.raw_measurement, dtype=float))
        if current_target is None:
            continue
        estimated_path = _positions_to_geo(np.asarray(estimated_positions, dtype=float), origin_lat_lon=origin_lat_lon)
        spoofed_path = _positions_to_geo(np.asarray(spoofed_positions, dtype=float), origin_lat_lon=origin_lat_lon)
        path_rows.append({"name": f"{target_name} EKF", "path": estimated_path, "color": [115, 240, 160]})
        path_rows.append({"name": f"{target_name} Spoofed", "path": spoofed_path, "color": [255, 75, 75]})
        est_lon, est_lat = estimated_path[-1]
        spoof_lon, spoof_lat = spoofed_path[-1]
        point_rows.append({"name": f"{target_name} EKF", "lon": est_lon, "lat": est_lat, "color": [115, 240, 160], "radius": 75})
        point_rows.append({"name": f"{target_name} Spoofed", "lon": spoof_lon, "lat": spoof_lat, "color": [255, 75, 75], "radius": 55})
        uncertainty_rows.append(
            {
                "name": f"{target_name} Uncertainty",
                "lon": est_lon,
                "lat": est_lat,
                "radius": max(float(current_target.uncertainty_radius_m) * 8.5, 25.0),
                "color": [115, 240, 160, 35],
            }
        )

    frame = replay.map_frame[replay.map_frame["step"] <= step].copy()
    deck = pdk.Deck(
        map_provider=DECK_MAP_PROVIDER,
        map_style=DECK_MAP_STYLE,
        initial_view_state=pdk.ViewState(
            latitude=float(frame["lat"].mean()),
            longitude=float(frame["lon"].mean()),
            zoom=12.8,
            pitch=48,
            bearing=14,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame(
                    [
                        {
                            "name": "No-Fly Zone",
                            "lon": no_fly_lon,
                            "lat": no_fly_lat,
                            "radius": float(zone.get("radius_m", 30.0)) * 8.5,
                            "color": [255, 75, 75, 45],
                        }
                    ]
                ),
                get_position="[lon, lat]",
                get_fill_color="color",
                get_line_color=[255, 75, 75],
                get_radius="radius",
                stroked=True,
                filled=True,
                line_width_min_pixels=2,
            ),
            pdk.Layer("PathLayer", data=path_rows, get_path="path", get_color="color", width_scale=8, width_min_pixels=2, get_width=4),
            pdk.Layer(
                "ScatterplotLayer",
                data=uncertainty_rows,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius="radius",
                stroked=False,
                filled=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=point_rows,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
            ),
        ],
        tooltip={"text": "{name}"},
    )
    with day8_map_placeholder.container():
        threat_order = ", ".join(validation.get("threat_order", []))
        st.caption(
            f"Active threat ranking: {threat_order or 'n/a'} | "
            f"packet-loss events: {validation.get('packet_loss_events', 0)} | "
            f"no-fly radius: {zone.get('radius_m', 30.0):.1f} m | "
            f"green=EKF / red=spoofed"
        )
        st.pydeck_chart(deck, width="stretch")


def _render_day8_telemetry(day8_telemetry_placeholder: Any, replay: MissionReplay, upto_index: int | None = None) -> None:
    if len(replay.frames) == 0:
        day8_telemetry_placeholder.info("No live swarm telemetry available.")
        return
    if upto_index is None:
        upto_index = len(replay.frames) - 1
    frame_index = min(max(int(upto_index), 0), len(replay.frames) - 1)
    mission_frame = replay.frames[frame_index]
    origin_lat_lon = tuple(replay.validation.get("origin_lat_lon", (37.7749, -122.4194)))
    rows: list[dict[str, Any]] = []
    lon_i, lat_i = _positions_to_geo(
        np.asarray([mission_frame.interceptor_position], dtype=float),
        origin_lat_lon=origin_lat_lon,
    )[0]
    rows.append(
        {
            "name": "Interceptor",
            "role": "interceptor",
            "lat": round(lat_i, 6),
            "lon": round(lon_i, 6),
            "altitude_m": round(float(mission_frame.interceptor_position[2]), 2),
            "speed_mps": round(float(np.linalg.norm(mission_frame.interceptor_velocity)), 2),
            "threat_level": 0.0,
            "packet_dropped": False,
            "spoofing_detected": False,
            "jammed": False,
        }
    )
    for target in mission_frame.targets:
        lon_t, lat_t = _positions_to_geo(
            np.asarray([target.position], dtype=float),
            origin_lat_lon=origin_lat_lon,
        )[0]
        measured_error = float(np.linalg.norm(target.raw_measurement - target.position))
        estimated_error = float(np.linalg.norm(target.filtered_estimate - target.position))
        rows.append(
            {
                "name": target.name,
                "role": "target",
                "lat": round(lat_t, 6),
                "lon": round(lon_t, 6),
                "altitude_m": round(float(target.position[2]), 2),
                "speed_mps": round(float(np.linalg.norm(target.velocity)), 2),
                "threat_level": round(float(target.threat_level), 4),
                "spoofing_active": bool(target.spoofing_active),
                "drift_rate_mps": round(float(target.drift_rate_mps), 3),
                "spoof_offset_m": round(float(target.spoof_offset_m), 3),
                "uncertainty_m": round(float(target.uncertainty_radius_m), 3),
                "innovation_m": round(float(target.innovation_m), 3),
                "innovation_gate": round(float(target.innovation_gate), 3),
                "measured_error_m": round(measured_error, 3),
                "estimated_error_m": round(estimated_error, 3),
                "packet_dropped": bool(target.packet_dropped),
                "spoofing_detected": bool(target.spoofing_detected),
                "jammed": bool(target.jammed),
            }
        )
    with day8_telemetry_placeholder.container():
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_day8_distance_plot(day8_distance_placeholder: Any, replay: MissionReplay, upto_index: int | None = None) -> None:
    frame = replay.distance_frame.copy()
    if frame.empty:
        day8_distance_placeholder.info("No multi-UAV distance telemetry available.")
        return
    if upto_index is not None and len(replay.frames) > 0:
        max_time = replay.frames[min(max(int(upto_index), 0), len(replay.frames) - 1)].time_s
        frame = frame[frame["time_s"] <= max_time]
    figure = go.Figure()
    for target_name, group in frame.groupby("target"):
        figure.add_trace(
            go.Scatter(
                x=group["time_s"],
                y=group["distance_m"],
                mode="lines",
                name=f"{target_name} spoofing",
            )
        )
    comparison_replay = st.session_state.get("day8_compare_replay")
    if isinstance(comparison_replay, MissionReplay) and not comparison_replay.distance_frame.empty:
        comparison_frame = comparison_replay.distance_frame.copy()
        if upto_index is not None and len(comparison_replay.frames) > 0:
            max_time = comparison_replay.frames[min(max(int(upto_index), 0), len(comparison_replay.frames) - 1)].time_s
            comparison_frame = comparison_frame[comparison_frame["time_s"] <= max_time]
        for target_name, group in comparison_frame.groupby("target"):
            figure.add_trace(
                go.Scatter(
                    x=group["time_s"],
                    y=group["distance_m"],
                    mode="lines",
                    name=f"{target_name} clean",
                    line=dict(dash="dot"),
                )
            )
    figure.update_layout(
        title="Distance to Target per UAV: spoofing vs clean baseline",
        xaxis_title="Time [s]",
        yaxis_title="Distance [m]",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    day8_distance_placeholder.plotly_chart(figure, width="stretch", key=f"day8_distance_{upto_index if upto_index is not None else 'static'}", config={"displayModeBar": False})


def _render_day8_validation_plot(day8_validation_plot_placeholder: Any, validation: MonteCarloSummary | dict[str, Any] | None, key_suffix: str) -> None:
    if validation is None:
        day8_validation_plot_placeholder.info("Validation comparison will appear after a 100-iteration run.")
        return
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Success Rate", "Mean Miss Distance"),
    )
    figure.add_trace(
        go.Bar(
            x=["Raw", "EKF"],
            y=[_validation_metric(validation, "raw_success_rate"), _validation_metric(validation, "ekf_success_rate")],
            marker_color=["#ff845c", "#73f0a0"],
            name="Success",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=["Raw", "EKF"],
            y=[
                _validation_metric(validation, "raw_mean_miss_distance_m"),
                _validation_metric(validation, "ekf_mean_miss_distance_m"),
            ],
            marker_color=["#ff845c", "#5ed2ff"],
            name="Miss Distance",
        ),
        row=1,
        col=2,
    )
    figure.update_yaxes(title_text="Rate", range=[0.0, 1.05], row=1, col=1)
    figure.update_yaxes(title_text="Distance [m]", row=1, col=2)
    figure.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    day8_validation_plot_placeholder.plotly_chart(
        figure,
        width="stretch",
        key=f"day8_validation_comparison_{key_suffix}",
        config={"displayModeBar": False},
    )


def _render_day8_validation(day8_validation_placeholder: Any, validation: MonteCarloSummary | dict[str, Any] | None) -> None:
    with day8_validation_placeholder.container():
        if validation is None:
            st.info("Enable `Run 100x Validation` and start a run to compute raw vs EKF Monte Carlo results.")
            return
        cols = st.columns(5)
        cols[0].metric("EKF Success", f"{_validation_metric(validation, 'ekf_success_rate') * 100:.0f}%")
        cols[1].metric("Raw Success", f"{_validation_metric(validation, 'raw_success_rate') * 100:.0f}%")
        cols[2].metric("EKF Miss", f"{_validation_metric(validation, 'ekf_mean_miss_distance_m'):.3f} m")
        cols[3].metric("EKF Kill Prob", f"{_validation_metric(validation, 'ekf_mean_kill_probability') * 100:.1f}%")
        cols[4].metric("Iterations", f"{int(_validation_metric(validation, 'iterations', 0))}")
        control_state = st.session_state.get("live_control_state", {})
        if control_state:
            st.caption(
                f"Validation profile: targets={control_state.get('num_targets', 'n/a')}, "
                f"latency={control_state.get('latency_ms', 0.0):.0f} ms, "
                f"packet loss={control_state.get('packet_loss_rate', 0.0):.2f}"
            )
        if _validation_passed(validation):
            st.success(
                f"Validation passed: EKF success {_validation_metric(validation, 'ekf_success_rate') * 100:.0f}% "
                f"across {int(_validation_metric(validation, 'iterations', 0))} iterations."
            )
        else:
            st.warning(
                f"Validation hold: EKF success {_validation_metric(validation, 'ekf_success_rate') * 100:.0f}% "
                f"with mean miss {_validation_metric(validation, 'ekf_mean_miss_distance_m'):.3f} m."
            )


def _render_backend_status_panel(
    backend_status_placeholder: Any,
    backend_state: dict[str, Any] | None,
    backend_host: str,
    backend_port: int,
    simulation: dict[str, Any] | None = None,
) -> None:
    del simulation
    with backend_status_placeholder.container():
        backend_error = st.session_state.get("backend_error")
        if backend_error and backend_state is None:
            st.warning(f"Backend unavailable: {backend_error}")
            return
        if not backend_state:
            st.info(
                "Backend service idle. Run a mission to populate backend status, target telemetry, and mission insights."
            )
            return
        snapshot = backend_state.get("snapshot", {})
        validation = backend_state.get("validation")
        preflight = backend_state.get("preflight")
        status_endpoint = backend_state.get("status_endpoint", {})
        status_html = f"""
        <style>
          html, body {{
            margin: 0;
            min-height: 100%;
            background: #08111B;
            color: #FFFFFF;
            font-family: 'Roboto Mono', 'JetBrains Mono', 'Courier New', monospace;
          }}
          body, div, span, strong {{
            color: #FFFFFF;
            font-family: inherit;
          }}
          .backend-status-root {{
            color: #FFFFFF;
            background: linear-gradient(180deg, rgba(18, 24, 34, 0.92), rgba(7, 11, 18, 0.96));
            border: 1px solid rgba(0, 229, 255, 0.82);
            border-radius: 14px;
            box-shadow: 0 0 10px #00F2FF33;
            padding: 10px;
            overflow-y: auto;
          }}
          .backend-status-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
          }}
          .backend-status-grid.three {{
            grid-template-columns: repeat(3, minmax(0, 1fr));
          }}
          .stage-card {{
            padding: 14px 16px;
            margin: 0;
            line-height: 1.55;
            color: #FFFFFF;
            border: 1px solid rgba(0, 229, 255, 0.82);
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(10, 15, 20, 0.94), rgba(6, 9, 15, 0.90));
            box-shadow: 0 0 10px #00F2FF33;
          }}
          .stage-card strong, .stage-card span {{
            color: #FFFFFF;
          }}
          .status-red {{
            border-color: #FF5A5A;
            box-shadow: 0 0 14px rgba(255, 90, 90, 0.30);
          }}
          .status-green {{
            border-color: #00FF41;
            box-shadow: 0 0 14px rgba(0, 255, 65, 0.30);
          }}
          .backend-alert {{
            margin-top: 10px;
            color: #FF7B7B;
            font-weight: 700;
          }}
          @media (max-width: 900px) {{
            .backend-status-grid,
            .backend-status-grid.three {{
              grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
          }}
        </style>
        <div class="backend-status-root">
        <div class="backend-status-grid">
          <div id="svc-status" class="stage-card {'status-red' if snapshot.get('spoofing_active', False) else 'status-green'}"><strong>Mission Status</strong><br/>{snapshot.get('status', 'idle').upper()}<br/>Stage: {snapshot.get('active_stage', 'n/a')}</div>
          <div id="svc-ekf" class="stage-card {'status-green' if snapshot.get('ekf_lock', False) else 'status-red'}"><strong>EKF Lock</strong><br/>{'ACTIVE' if snapshot.get('ekf_lock', False) else 'OFF'}<br/>Target: {snapshot.get('active_target', 'n/a')}</div>
          <div class="stage-card"><strong>Backend Throughput</strong><br/><span id="svc-throughput">{snapshot.get('backend_throughput_fps', 0.0):.2f}</span> Hz<br/>Detection FPS: <span id="svc-detect">{snapshot.get('detection_fps_window_avg', snapshot.get('detection_fps', 0.0)):.2f}</span></div>
          <div class="stage-card"><strong>Kill Probability</strong><br/><span id="svc-kill">{snapshot.get('kill_probability', 0.0) * 100:.1f}</span>% (Target: <span id="svc-kill-target">{snapshot.get('kill_probability_target_id', snapshot.get('active_target', 'n/a'))}</span>)<br/><span id="svc-dist-label">{'Closest Approach' if snapshot.get('status') == 'complete' else 'Distance'}</span>: <span id="svc-dist">{(snapshot.get('closest_approach_m', snapshot.get('active_distance_m', 0.0)) if snapshot.get('status') == 'complete' else snapshot.get('active_distance_m', 0.0)):.2f}</span> m (Target: <span id="svc-dist-target">{snapshot.get('closest_approach_target_id', snapshot.get('active_target', 'n/a'))}</span>)</div>
        </div>
        <div id="svc-alert" class="backend-alert"></div>
        <div class="backend-status-grid" style="margin-top:12px;">
          <div class="stage-card"><strong>Measured RMSE</strong><br/><span id="svc-mrmse">{snapshot.get('rmse_measured_true_m', 0.0):.3f}</span> m</div>
          <div class="stage-card"><strong>EKF RMSE</strong><br/><span id="svc-ermse">{snapshot.get('rmse_m', 0.0):.3f}</span> m</div>
          <div class="stage-card"><strong>Uncertainty</strong><br/><span id="svc-unc">{snapshot.get('mean_uncertainty_m', 0.0):.3f}</span> m</div>
          <div class="stage-card"><strong>Deploy Ready</strong><br/><span id="svc-deploy">{'YES' if backend_state.get('deploy_ready', False) else 'NO'}</span></div>
        </div>
        <div class="backend-status-grid three" style="margin-top:12px;">
          <div class="stage-card"><strong>LOS Rate</strong><br/><span id="svc-los">{snapshot.get('los_rate_rps', 0.0):.3f}</span> rad/s</div>
          <div class="stage-card"><strong>Closing Velocity</strong><br/><span id="svc-close">{snapshot.get('closing_velocity_mps', 0.0):.2f}</span> m/s</div>
          <div class="stage-card"><strong>Innovation</strong><br/><span id="svc-innov">{snapshot.get('innovation_m', 0.0):.3f}</span> m</div>
        </div>
        <div class="backend-status-grid three" style="margin-top:12px;">
          <div class="stage-card"><strong>Confidence Score</strong><br/><span id="svc-confidence">{snapshot.get('confidence_score', 0.0) * 100:.1f}</span>%</div>
          <div class="stage-card"><strong>Target Count</strong><br/><span id="svc-count">{snapshot.get('target_count', 0)}</span></div>
          <div class="stage-card"><strong>Validation State</strong><br/><span id="svc-validate">{'PASS' if backend_state.get('deploy_ready', False) else 'HOLD'}</span></div>
        </div>
        </div>
        <script>
          (() => {{
            const applyPayload = (payload) => {{
              const snapshot = payload.snapshot || {{}};
              const alertNode = document.getElementById('svc-alert');
              const statusNode = document.getElementById('svc-status');
              const ekfNode = document.getElementById('svc-ekf');
              statusNode.innerHTML = `<strong>Mission Status</strong><br/>${{(snapshot.status || 'idle').toUpperCase()}}<br/>Stage: ${{snapshot.active_stage || 'n/a'}}`;
              ekfNode.innerHTML = `<strong>EKF Lock</strong><br/>${{snapshot.ekf_lock ? 'ACTIVE' : 'OFF'}}<br/>Target: ${{snapshot.active_target || 'n/a'}}`;
              statusNode.className = `stage-card ${{snapshot.spoofing_active ? 'status-red' : 'status-green'}}`;
              ekfNode.className = `stage-card ${{snapshot.ekf_lock ? 'status-green' : 'status-red'}}`;
              document.getElementById('svc-throughput').textContent = Number(snapshot.backend_throughput_fps || 0).toFixed(2);
              document.getElementById('svc-detect').textContent = Number(snapshot.detection_fps_window_avg || snapshot.detection_fps || 0).toFixed(2);
              document.getElementById('svc-kill').textContent = (Number(snapshot.kill_probability || 0) * 100).toFixed(1);
              document.getElementById('svc-kill-target').textContent = String(snapshot.kill_probability_target_id || snapshot.active_target || 'n/a');
              document.getElementById('svc-dist-label').textContent = (snapshot.status || 'idle') === 'complete' ? 'Closest Approach' : 'Distance';
              const distanceValue = (snapshot.status || 'idle') === 'complete'
                ? Number(snapshot.closest_approach_m || snapshot.active_distance_m || 0)
                : Number(snapshot.active_distance_m || 0);
              document.getElementById('svc-dist').textContent = distanceValue.toFixed(2);
              const distanceTarget = (snapshot.status || 'idle') === 'complete'
                ? String(snapshot.closest_approach_target_id || snapshot.active_target || 'n/a')
                : String(snapshot.active_target || 'n/a');
              document.getElementById('svc-dist-target').textContent = distanceTarget;
              document.getElementById('svc-mrmse').textContent = Number(snapshot.rmse_measured_true_m || 0).toFixed(3);
              document.getElementById('svc-ermse').textContent = Number(snapshot.rmse_m || 0).toFixed(3);
              document.getElementById('svc-unc').textContent = Number(snapshot.mean_uncertainty_m || 0).toFixed(3);
              document.getElementById('svc-deploy').textContent = payload.deploy_ready ? 'YES' : 'NO';
              document.getElementById('svc-los').textContent = Number(snapshot.los_rate_rps || 0).toFixed(3);
              document.getElementById('svc-close').textContent = Number(snapshot.closing_velocity_mps || 0).toFixed(2);
              document.getElementById('svc-innov').textContent = Number(snapshot.innovation_m || 0).toFixed(3);
              document.getElementById('svc-confidence').textContent = (Number(snapshot.confidence_score || 0) * 100).toFixed(1);
              document.getElementById('svc-count').textContent = Number(snapshot.target_count || 0).toFixed(0);
              document.getElementById('svc-validate').textContent = payload.deploy_ready ? 'PASS' : 'HOLD';
              if (payload.event === 'mission_complete' && payload.replay_data) {{
                alertNode.textContent = `REPLAY READY | frames=${{Number(payload.replay_data.frame_count || 0)}}`;
              }} else if (snapshot.spoofing_active) {{
                alertNode.textContent = 'SPOOFING DETECTED';
              }} else {{
                alertNode.textContent = '';
              }}
            }};
            const fetchFallbackState = () => {{
              fetch("http://{backend_host}:{backend_port}/mission/state")
                .then((response) => response.ok ? response.json() : null)
                .then((payload) => {{
                  if (payload) applyPayload(payload);
                }})
                .catch(() => null);
            }};
            try {{
              const socket = new WebSocket("ws://{backend_host}:{backend_port}/ws/state");
              socket.onmessage = (event) => {{
                const payload = JSON.parse(event.data);
                applyPayload(payload);
              }};
              socket.onerror = () => fetchFallbackState();
              socket.onclose = () => fetchFallbackState();
            }} catch (err) {{
              fetchFallbackState();
            }}
            fetchFallbackState();
            setInterval(fetchFallbackState, 1500);
          }})();
        </script>
        """
        components.html(status_html, height=700)
        if preflight is not None:
            st.caption(
                f"Preflight: ready={preflight.get('ready', False)} | "
                f"spawned={preflight.get('spawned_targets', 0)}/{preflight.get('requested_targets', 0)} | "
                f"fallback={preflight.get('fallback_mode', False)}"
            )
        if status_endpoint:
            st.caption(
                f"Service /status: lifecycle={status_endpoint.get('lifecycle', 'ACTIVE')} | "
                f"stage={status_endpoint.get('stage', 'idle')} | "
                f"targets={status_endpoint.get('target_count', 0)}"
            )
            readiness = ((status_endpoint.get("mission_insights") or {}).get("command_readiness") or {})
            if isinstance(readiness, dict) and readiness:
                st.caption(
                    f"Command readiness={float(readiness.get('readiness_score', 0.0)):.1f}/100 | "
                    f"posture={str(readiness.get('security_posture', 'UNKNOWN'))} | "
                    f"quality_gate={'PASS' if bool(readiness.get('quality_gate_passed', False)) else 'HOLD'}"
                )
        if validation is not None:
            st.caption(
                f"Backend validation: EKF={validation.get('ekf_success_rate', 0.0) * 100:.0f}% | "
                f"RAW={validation.get('raw_success_rate', 0.0) * 100:.0f}% | "
                f"miss={validation.get('ekf_mean_miss_distance_m', 0.0):.3f} m | "
                f"kill={validation.get('ekf_mean_kill_probability', 0.0) * 100:.1f}%"
            )
        
        mission_insights = status_endpoint.get("mission_insights")
        if mission_insights is not None:
            with st.expander("Mission Insights - Global and Per-Target Metrics"):
                global_metrics = mission_insights.get("global_metrics", {})
                target_details = mission_insights.get("target_details", [])
                command_readiness = mission_insights.get("command_readiness", {})
                priority_queue = mission_insights.get("engagement_priority_queue", [])
                tactical_assessment = mission_insights.get("tactical_assessment", {})

                insights_cols = st.columns(4)
                insights_cols[0].metric(
                    "Total Targets",
                    int(global_metrics.get("total_targets", 0)),
                )
                insights_cols[1].metric(
                    "Weighted Success",
                    f"{float(global_metrics.get('weighted_mission_success', global_metrics.get('average_mission_success', 0.0))) * 100:.1f}%",
                )
                insights_cols[2].metric(
                    "Interception Completion",
                    f"{float(global_metrics.get('interception_completion_rate', 0.0)) * 100:.1f}%",
                )
                insights_cols[3].metric(
                    "Spoofing Detection",
                    f"{float(global_metrics.get('spoofing_detection_rate', 0.0)) * 100:.1f}%",
                )

                system_cols = st.columns(4)
                system_cols[0].metric(
                    "System RMSE",
                    f"{float(global_metrics.get('system_wide_rmse', 0.0)):.4f} m",
                )
                system_cols[1].metric(
                    "RMSE P95",
                    f"{float(global_metrics.get('rmse_p95_m', 0.0)):.4f} m",
                )
                system_cols[2].metric(
                    "Mean Compute Latency",
                    f"{float(global_metrics.get('mean_compute_latency_ms', 0.0)):.2f} ms",
                )
                system_cols[3].metric(
                    "Total Energy",
                    f"{float(global_metrics.get('total_energy_consumption_j', 0.0)):.2f} J",
                )

                analytics_cols = st.columns(4)
                analytics_cols[0].metric(
                    "Mean Guidance Efficiency",
                    f"{float(global_metrics.get('mean_guidance_efficiency_mps2', 0.0)):.4f} m/s^2",
                )
                analytics_cols[1].metric(
                    "Mean Spoofing Variance",
                    f"{float(global_metrics.get('mean_spoofing_variance', 0.0)):.6f}",
                )
                analytics_cols[2].metric(
                    "Mean Mission Success P",
                    f"{float(global_metrics.get('mean_mission_success_probability', 0.0)) * 100:.1f}%",
                )
                analytics_cols[3].metric(
                    "Risk P90",
                    f"{float(global_metrics.get('risk_index_p90', 0.0)):.3f}",
                )

                link_cols = st.columns(4)
                link_cols[0].metric(
                    "Observed Packet Loss",
                    f"{float(global_metrics.get('packet_loss_observed_rate', 0.0)) * 100:.2f}%",
                )
                link_cols[1].metric(
                    "Effective Packet Loss",
                    f"{float(global_metrics.get('packet_loss_effective_rate', 0.0)) * 100:.2f}%",
                )
                link_cols[2].metric(
                    "Mean Link SNR",
                    f"{float(global_metrics.get('mean_link_snr_db', 0.0)):.2f} dB",
                )
                link_cols[3].metric(
                    "Mean Packet Loss P",
                    f"{float(global_metrics.get('mean_packet_loss_probability', 0.0)) * 100:.2f}%",
                )
                targeting_cols = st.columns(4)
                targeting_cols[0].metric(
                    "Best Kill Target",
                    str(global_metrics.get("best_kill_probability_target", "n/a")),
                )
                targeting_cols[1].metric(
                    "Best Kill Probability",
                    f"{float(global_metrics.get('best_kill_probability', 0.0)) * 100:.1f}%",
                )
                targeting_cols[2].metric(
                    "Closest Target",
                    str(global_metrics.get("closest_approach_target", "n/a")),
                )
                targeting_cols[3].metric(
                    "Closest Approach",
                    f"{float(global_metrics.get('closest_approach_m', 0.0)):.3f} m",
                )

                mission_model = mission_insights.get("mission_model", {})
                packet_model = mission_model.get("packet_loss_model", {}) if isinstance(mission_model, dict) else {}
                live_controls = st.session_state.get("live_control_state", {})
                st.caption(
                    "Packet Loss Model: "
                    + str(mission_model.get("packet_loss_formula", "PL = 1 - exp(-k * SNR / d^alpha)"))
                )
                st.caption(
                    f"Model Params: k={float(packet_model.get('k', live_controls.get('packet_loss_k', 0.12))):.3f}, "
                    f"alpha={float(packet_model.get('alpha', live_controls.get('packet_loss_alpha', 1.8))):.2f}, "
                    f"base_link_snr_db={float(packet_model.get('base_link_snr_db', live_controls.get('link_snr_db', 28.0))):.2f}"
                )
                st.markdown("#### Metric Formula Reference")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Metric": "Interception Time",
                                "Formula": "T_int = t_impact - t_launch",
                                "Source": "Per-target replay timeline",
                            },
                            {
                                "Metric": "EKF Success Rate",
                                "Formula": "sum(||x_true - x_ekf|| < threshold) / T",
                                "Source": "All frames per target",
                            },
                            {
                                "Metric": "RMSE",
                                "Formula": "sqrt((1/n) * sum((x_true - x_est)^2))",
                                "Source": "EKF estimate vs true state",
                            },
                            {
                                "Metric": "Kill Probability",
                                "Formula": "P_k = exp(-0.5 * d_M^2)",
                                "Source": "Mahalanobis distance + uncertainty",
                            },
                            {
                                "Metric": "Packet Loss Probability",
                                "Formula": "PL = 1 - exp(-k * SNR / d^alpha)",
                                "Source": "Link SNR + distance model",
                            },
                            {
                                "Metric": "Energy Consumption",
                                "Formula": "E = sum((P_hover + c_drag*|v|^3 + eta*m*|a|*|v|) * dt)",
                                "Source": "Per-frame power integration from replay state",
                            },
                            {
                                "Metric": "Energy Fallback",
                                "Formula": "E_fb = (P_hover + eta*m*g_bar*v_ref) * T",
                                "Source": "Used only when acceleration samples are unavailable",
                            },
                            {
                                "Metric": "Mission Success Probability",
                                "Formula": "weighted(EKF success, tracking, intercept, distance, time)",
                                "Source": "Per-target fused score",
                            },
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
                energy_model = mission_model.get("energy_model", {}) if isinstance(mission_model, dict) else {}
                mass_kg = float(energy_model.get("interceptor_mass_kg", live_controls.get("interceptor_mass_kg", 6.5)))
                hover_w = float(energy_model.get("hover_power_w", 90.0))
                drag_coeff = float(energy_model.get("drag_power_coeff", 0.02))
                accel_coeff = float(energy_model.get("accel_power_coeff", 0.45))
                st.markdown("#### Energy Model Terms")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Term": "m", "Meaning": "Interceptor mass", "Units": "kg", "Value": round(mass_kg, 3)},
                            {"Term": "P_hover", "Meaning": "Base hover/compute power", "Units": "W", "Value": round(hover_w, 3)},
                            {"Term": "c_drag", "Meaning": "Drag power coefficient", "Units": "W/(m/s)^3", "Value": round(drag_coeff, 5)},
                            {"Term": "eta", "Meaning": "Acceleration power coefficient", "Units": "1", "Value": round(accel_coeff, 4)},
                            {"Term": "|v|", "Meaning": "Interceptor speed magnitude", "Units": "m/s", "Value": "runtime"},
                            {"Term": "|a|", "Meaning": "Interceptor acceleration magnitude", "Units": "m/s^2", "Value": "runtime"},
                            {"Term": "dt", "Meaning": "Per-frame time delta", "Units": "s", "Value": "runtime"},
                            {"Term": "T", "Meaning": "Fallback duration horizon", "Units": "s", "Value": "runtime"},
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )

                if isinstance(command_readiness, dict) and command_readiness:
                    readiness_cols = st.columns(5)
                    readiness_cols[0].metric(
                        "Command Readiness",
                        f"{float(command_readiness.get('readiness_score', 0.0)):.1f}/100",
                    )
                    readiness_cols[1].metric(
                        "Security Posture",
                        str(command_readiness.get("security_posture", "UNKNOWN")),
                    )
                    readiness_cols[2].metric(
                        "Quality Gate",
                        "PASS" if bool(command_readiness.get("quality_gate_passed", False)) else "HOLD",
                    )
                    readiness_cols[3].metric(
                        "Deploy Gate",
                        "PASS" if bool(command_readiness.get("deployment_gate_passed", False)) else "HOLD",
                    )
                    readiness_cols[4].metric(
                        "Telemetry Reliability",
                        f"{float(command_readiness.get('telemetry_reliability_score', 0.0)) * 100:.1f}%",
                    )
                    gate_cols = st.columns(3)
                    gate_cols[0].metric(
                        "Stress Index",
                        f"{float(command_readiness.get('stress_index', 0.0)) * 100:.1f}%",
                    )
                    gate_cols[1].metric(
                        "Deploy Margin",
                        f"{float(command_readiness.get('deployment_margin_score', 0.0)):.1f}/100",
                    )
                    checks = command_readiness.get("deployment_checks", {})
                    if isinstance(checks, dict) and checks:
                        passed = int(sum(1 for value in checks.values() if bool(value)))
                        total = int(len(checks))
                    else:
                        passed, total = 0, 0
                    gate_cols[2].metric("Deploy Checks", f"{passed}/{total}")

                    thresholds = command_readiness.get("deployment_thresholds", {})
                    if isinstance(thresholds, dict) and thresholds:
                        st.markdown("#### Adaptive Deployment Thresholds")
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {"Threshold": "Mission Success Min", "Value": float(thresholds.get("mission_success_min", 0.0))},
                                    {"Threshold": "Completion Min", "Value": float(thresholds.get("completion_min", 0.0))},
                                    {"Threshold": "RMSE P95 Max [m]", "Value": float(thresholds.get("rmse_p95_max", 0.0))},
                                    {"Threshold": "Latency Max [ms]", "Value": float(thresholds.get("latency_max_ms", 0.0))},
                                    {"Threshold": "Telemetry Reliability Min", "Value": float(thresholds.get("telemetry_reliability_min", 0.0))},
                                    {"Threshold": "Risk P90 Max", "Value": float(thresholds.get("risk_p90_max", 0.0))},
                                    {"Threshold": "Spoof Detect Min", "Value": float(thresholds.get("spoof_detect_min", 0.0))},
                                ]
                            ),
                            width="stretch",
                            hide_index=True,
                        )
                    component_scores = command_readiness.get("component_scores", {})
                    if isinstance(component_scores, dict) and component_scores:
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "Component": str(key).replace("_", " ").title(),
                                        "Score": float(value),
                                    }
                                    for key, value in component_scores.items()
                                ]
                            ),
                            width="stretch",
                            hide_index=True,
                        )

                if target_details:
                    target_rows = []
                    for target in target_details:
                        target_rows.append(
                            {
                                "Target ID": str(target.get("id", "unknown")),
                                "Engagement State": str(target.get("engagement_state", "ACTIVE")),
                                "EKF Success [%]": round(float(target.get("ekf_success_rate", 0.0)) * 100.0, 1),
                                "RMSE [m]": round(float(target.get("rmse_m", target.get("ekf_mean_distance_m", 0.0))), 4),
                                "Mission Success P [%]": round(float(target.get("mission_success_probability", 0.0)) * 100.0, 1),
                                "Kill Probability [%]": round(float(target.get("kill_probability", target.get("mission_success_probability", 0.0))) * 100.0, 1),
                                "Closest Approach [m]": round(float(target.get("closest_approach_m", target.get("distance_m", 0.0))), 4),
                                "Guidance Efficiency [m/s^2]": round(float(target.get("guidance_efficiency_mps2", 0.0)), 4),
                                "Spoofing Variance": round(float(target.get("spoofing_variance", 0.0)), 6),
                                "Link SNR [dB]": round(float(target.get("link_snr_db", 0.0)), 3),
                                "Packet Loss P [%]": round(float(target.get("packet_loss_probability", 0.0)) * 100.0, 2),
                                "Compute Latency [ms]": round(float(target.get("compute_latency_ms", 0.0)), 3),
                                "Energy [J]": round(float(target.get("energy_consumption_j", 0.0)), 3),
                                "Risk Index": round(float(target.get("risk_index", 0.0)), 3),
                                "Innovation Ratio": round(float(target.get("innovation_ratio", 0.0)), 3),
                            }
                        )
                    st.subheader("Per-Target Details")
                    st.dataframe(pd.DataFrame(target_rows), use_container_width=True, hide_index=True)

                if isinstance(priority_queue, list) and priority_queue:
                    st.subheader("Engagement Priority Queue")
                    st.dataframe(
                        pd.DataFrame(priority_queue).rename(
                            columns={
                                "target_id": "Target",
                                "engagement_state": "State",
                                "priority_score": "Priority Score",
                                "risk_index": "Risk Index",
                                "distance_m": "Distance [m]",
                                "mission_success_probability": "Mission Success P",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                recommendations = tactical_assessment.get("recommendations", [])
                if isinstance(recommendations, list) and recommendations:
                    st.caption("Tactical Assessment")
                    for recommendation in recommendations:
                        st.write(f"- {recommendation}")
                elif recommendations:
                    st.caption(f"Tactical Assessment: {recommendations}")

                directives = command_readiness.get("directives", []) if isinstance(command_readiness, dict) else []
                if isinstance(directives, list) and directives:
                    st.caption("Command Directives")
                    for directive in directives:
                        st.write(f"- {directive}")
        
        if st.session_state.get("backend_deployed", False):
            st.success("Deploy command armed. Mission console is in validated deployable state.")


def _render_backend_live_stream(backend_stream_placeholder: Any, backend_host: str, backend_port: int) -> None:
    del backend_host, backend_port
    snapshot = ((st.session_state.get("backend_state") or {}).get("snapshot") or {})
    targets = snapshot.get("targets", [])
    if not isinstance(targets, list):
        targets = []
    with backend_stream_placeholder.container():
        st.caption("Mission HUD (direct state view). WebSocket stream removed for stability.")
        cols = st.columns(4)
        cols[0].metric("Stage", str(snapshot.get("active_stage", "n/a")))
        cols[1].metric("Target", str(snapshot.get("active_target", "n/a")))
        cols[2].metric("EKF RMSE", f"{_safe_float(snapshot.get('rmse_m', 0.0), 0.0):.3f} m")
        cols[3].metric(
            "Kill Probability",
            f"{_safe_float(snapshot.get('kill_probability', 0.0), 0.0) * 100.0:.1f}%",
        )
        cols2 = st.columns(4)
        cols2[0].metric("Closest Approach", f"{_safe_float(snapshot.get('closest_approach_m', snapshot.get('active_distance_m', 0.0)), 0.0):.2f} m")
        cols2[1].metric("LOS Rate", f"{_safe_float(snapshot.get('los_rate_rps', 0.0), 0.0):.3f} rad/s")
        cols2[2].metric("Closing Velocity", f"{_safe_float(snapshot.get('closing_velocity_mps', 0.0), 0.0):.2f} m/s")
        cols2[3].metric("Confidence", f"{_safe_float(snapshot.get('confidence_score', 0.0), 0.0) * 100.0:.1f}%")
        if targets:
            frame = pd.DataFrame(
                [
                    {
                        "target": str(target.get("target_id", target.get("name", f"Target_{index + 1}"))),
                        "distance_m": round(_safe_float(target.get("distance_m", 0.0), 0.0), 3),
                        "spoofing_active": bool(target.get("spoofing_active", False)),
                        "spoofing_detected": bool(target.get("spoofing_detected", False)),
                        "innovation_m": round(_safe_float(target.get("innovation_m", 0.0), 0.0), 3),
                    }
                    for index, target in enumerate(targets)
                    if isinstance(target, dict)
                ]
            )
            if not frame.empty:
                st.dataframe(frame, width="stretch", hide_index=True, height=220)


def _render_day8_architecture(
    day8_architecture_placeholder: Any,
    replay: MissionReplay,
    validation: MonteCarloSummary | dict[str, Any] | None,
) -> None:
    validation_status = "PASS" if _validation_passed(validation) else "HOLD"
    mission_status = "COMPLETE" if replay.validation.get("success", False) else "ACTIVE / PARTIAL"
    threat_order = ", ".join(replay.validation.get("threat_order", [])) or "n/a"
    advanced_visuals = replay.validation.get("advanced_visuals", {})
    cinematic_path = st.session_state.get("day8_cinematic_path")
    cinematic_label = Path(cinematic_path).name if cinematic_path else "pending"
    with day8_architecture_placeholder.container():
        st.markdown(
            f"""
            <div class="panel-grid">
              <div class="stage-card"><strong>Simulation Engine</strong><br/>Dynamic multi-target spawn and replay path is implemented through <code>AirSimMissionManager.setup_swarm(...)</code>. Mission status: {mission_status}. Advanced visuals ready: {advanced_visuals.get('ready', False)}.</div>
              <div class="stage-card"><strong>Telemetry Backend</strong><br/>FastAPI endpoints: <code>/preflight</code>, <code>/mission/start</code>, <code>/mission/state</code>, <code>/validate</code>, with direct state polling and mission replay payloads for frontend rendering.</div>
              <div class="stage-card"><strong>Tracking Resilience</strong><br/>Constant-velocity EKF suppresses spoofed GPS updates via innovation gating and prediction hold during packet loss. Confidence score is streamed to the HUD.</div>
              <div class="stage-card"><strong>Threat Logic</strong><br/>Threat order is ranked against the no-fly zone and interceptor proximity: {threat_order}.</div>
              <div class="stage-card"><strong>Cinematic Recorder</strong><br/>Timestamped mission export: <code>{cinematic_label}</code>. Browser-safe fallback video writing is active even when external AirSim capture is unavailable.</div>
              <div class="stage-card"><strong>Validation</strong><br/>Thread A uses raw spoofed telemetry. Thread B uses EKF-filtered telemetry. Current validation status: {validation_status}.</div>
              <div class="stage-card"><strong>Notebook</strong><br/>Day 8 documentation is available in <code>notebooks/day8_multi_uav_airsim_architecture.ipynb</code>.</div>
              <div class="stage-card"><strong>Segmentation / Visual Lock</strong><br/>Configured targets: {len(advanced_visuals.get('segmented_targets', []))}. Camera: <code>{advanced_visuals.get('camera_name', 'cinematic_cam')}</code>.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _replay_step_indices(total_steps: int, max_updates: int = 28) -> list[int]:
    if total_steps <= 1:
        return [0]
    stride = max(1, total_steps // max_updates)
    indices = list(range(0, total_steps, stride))
    if indices[-1] != total_steps - 1:
        indices.append(total_steps - 1)
    return indices


def _render_live_simulation_panel(
    simulation: dict[str, Any],
    backend_host: str,
    backend_port: int,
    active_step_override: int | None = None,
) -> None:
    del backend_host, backend_port
    tabs = st.tabs(["Animated Mission", "Frame Inspector", "Workflow"])
    if active_step_override is None:
        step_index = st.slider(
            "Playback Step",
            min_value=0,
            max_value=max(len(simulation["times"]) - 1, 0),
            value=max(len(simulation["times"]) - 1, 0),
            key="playback_step",
        )
    else:
        step_index = int(np.clip(active_step_override, 0, max(len(simulation["times"]) - 1, 0)))
        st.session_state["playback_step"] = int(step_index)
    stage_name = _current_stage_name(simulation, int(step_index))
    stage_ratio = 0.0
    if len(simulation["times"]) > 1:
        stage_ratio = float(step_index / max(len(simulation["times"]) - 1, 1))

    stage_left, stage_right = st.columns([1.25, 0.75])
    with stage_left:
        st.markdown(_build_live_stage_progress_html(stage_name, stage_ratio), unsafe_allow_html=True)
    with stage_right:
        st.markdown("**Live Running Code**")
        st.code(_stage_logic_snippet(stage_name), language="python")

    with tabs[0]:
        if active_step_override is not None:
            st.caption(f"Live replay step: {int(step_index)}")
        _render_live_video_or_fallback(simulation, int(step_index))
    with tabs[1]:
        _render_live_stage_panel(simulation, int(step_index))
    with tabs[2]:
        workflow_left, workflow_right = st.columns([0.9, 1.1])
        with workflow_left:
            _render_live_code_inspector(simulation, int(step_index), heading="Autonomy Stack Mirror")
            st.markdown(_build_explanation_html(simulation, int(step_index)), unsafe_allow_html=True)
        with workflow_right:
            st.markdown(_build_workflow_html(simulation), unsafe_allow_html=True)


def _render_live_backend_animation(backend_host: str, backend_port: int, height: int = 420) -> None:
    live_control_state = st.session_state.get("live_control_state", {})
    replay_fps_hz = max(float(live_control_state.get("playback_fps_hz", LIVE_REPLAY_FPS)), 1.0)
    animate_frontend = bool(live_control_state.get("animate_frontend", True))
    animation_html = f"""
    <div style="border:1px solid #00F2FF;box-shadow:0 0 10px #00F2FF33;border-radius:12px;padding:12px;background:#06090F;color:#E6FCFF;font-family:'Roboto Mono','JetBrains Mono',monospace;">
      <div style="color:#7DEBFF;margin-bottom:8px;">Live Backend Animation Feed</div>
      <div id="anim-status">Connecting...</div>
      <canvas id="anim-canvas" width="980" height="420" style="width:100%;height:420px;border:1px solid #183142;border-radius:10px;background:#08111B;margin-top:10px;"></canvas>
    </div>
    <script>
      const statusNode = document.getElementById("anim-status");
      const canvas = document.getElementById("anim-canvas");
      const ctx = canvas.getContext("2d");
      let previousSnapshot = null;
      let currentSnapshot = null;
      let replayFrames = [];
      let replayIndex = 0;
      let lastPacketTs = 0;
      const replayIntervalMs = {1000.0 / replay_fps_hz:.4f};
      const animateFrontendReplay = {"true" if animate_frontend else "false"};
      const lerp = (a, b, t) => a + (b - a) * t;
      const lerpPoint = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
      const mapFactory = (snapshot) => {{
        const targets = snapshot.targets || [];
        const all = [];
        if (snapshot.interceptor_position) all.push(snapshot.interceptor_position);
        targets.forEach((target) => {{
          all.push(target.true_position || [0, 0, 0]);
          all.push(target.spoofed_position || [0, 0, 0]);
          all.push(target.estimated_position || [0, 0, 0]);
        }});
        const xs = all.map((point) => point[0]);
        const ys = all.map((point) => point[1]);
        const minX = Math.min(...xs, 0), maxX = Math.max(...xs, 1);
        const minY = Math.min(...ys, 0), maxY = Math.max(...ys, 1);
        const spanX = Math.max(maxX - minX, 1), spanY = Math.max(maxY - minY, 1);
        return (point) => [48 + ((point[0] - minX) / spanX) * 884, 360 - ((point[1] - minY) / spanY) * 300];
      }};
      const drawSnapshot = (snapshot, title) => {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#08111B";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#E6FCFF";
        ctx.font = "24px JetBrains Mono";
        ctx.fillText(title, 24, 34);
        const mapPoint = mapFactory(snapshot);
        (snapshot.targets || []).forEach((target) => {{
          const truth = mapPoint(target.true_position || [0, 0, 0]);
          const spoof = mapPoint(target.spoofed_position || [0, 0, 0]);
          const fused = mapPoint(target.estimated_position || [0, 0, 0]);
          ctx.strokeStyle = "rgba(255,75,75,0.75)";
          ctx.beginPath();
          ctx.moveTo(truth[0], truth[1]);
          ctx.lineTo(spoof[0], spoof[1]);
          ctx.stroke();
          ctx.fillStyle = "#ff845c";
          ctx.beginPath(); ctx.arc(truth[0], truth[1], 6, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = "#ff4b4b";
          ctx.beginPath(); ctx.arc(spoof[0], spoof[1], 5, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = "#73f0a0";
          ctx.beginPath(); ctx.arc(fused[0], fused[1], 5, 0, Math.PI * 2); ctx.fill();
        }});
        if (snapshot.interceptor_position) {{
          const interceptor = mapPoint(snapshot.interceptor_position);
          ctx.fillStyle = "#00F2FF";
          ctx.beginPath(); ctx.arc(interceptor[0], interceptor[1], 6, 0, Math.PI * 2); ctx.fill();
        }}
      }};
      const interpolateSnapshot = () => {{
        if (!currentSnapshot) return null;
        if (!previousSnapshot) return currentSnapshot;
        const alpha = Math.min(Math.max((performance.now() - lastPacketTs) / 80.0, 0), 1);
        const next = JSON.parse(JSON.stringify(currentSnapshot));
        if (previousSnapshot.interceptor_position && currentSnapshot.interceptor_position) {{
          next.interceptor_position = lerpPoint(previousSnapshot.interceptor_position, currentSnapshot.interceptor_position, alpha);
        }}
        const prevTargets = new Map((previousSnapshot.targets || []).map((target) => [target.name, target]));
        next.targets = (currentSnapshot.targets || []).map((target) => {{
          const previous = prevTargets.get(target.name);
          if (!previous) return target;
          return {{
            ...target,
            true_position: lerpPoint(previous.true_position || target.true_position, target.true_position || previous.true_position, alpha),
            spoofed_position: lerpPoint(previous.spoofed_position || target.spoofed_position, target.spoofed_position || previous.spoofed_position, alpha),
            estimated_position: lerpPoint(previous.estimated_position || target.estimated_position, target.estimated_position || previous.estimated_position, alpha),
          }};
        }});
        return next;
      }};
      const renderLoop = () => {{
        if (replayFrames.length) {{
          drawSnapshot(replayFrames[replayIndex], "Replay Buffer");
          replayIndex = (replayIndex + 1) % replayFrames.length;
        }} else {{
          const snapshot = interpolateSnapshot();
          if (snapshot) drawSnapshot(snapshot, "Live Mission Feed");
        }}
        requestAnimationFrame(() => setTimeout(renderLoop, replayIntervalMs));
      }};
      const socket = new WebSocket("ws://{backend_host}:{backend_port}/ws");
      socket.onopen = () => {{ statusNode.textContent = "WebSocket: CONNECTED"; statusNode.style.color = "#73F0A0"; }};
      socket.onclose = () => {{ statusNode.textContent = "WebSocket: CLOSED"; statusNode.style.color = "#FF4B4B"; }};
      socket.onerror = () => {{ statusNode.textContent = "WebSocket: ERROR"; statusNode.style.color = "#FF4B4B"; }};
      socket.onmessage = (event) => {{
        const message = JSON.parse(event.data);
        if (message.type === "MISSION_COMPLETE" && message.replay_data) {{
          replayFrames = animateFrontendReplay ? (message.replay_data.frames || []) : [];
          replayIndex = 0;
          const completedStage = ((message.snapshot || {{}}).active_stage || message.stage || "Mission Complete");
          statusNode.textContent = `REPLAY READY - ${{completedStage}}`;
          statusNode.style.color = "#73F0A0";
          previousSnapshot = currentSnapshot || (message.snapshot || {{}});
          currentSnapshot = message.snapshot || currentSnapshot;
          lastPacketTs = performance.now();
          return;
        }}
        const snapshot = message.snapshot || message.payload || {{}};
        previousSnapshot = currentSnapshot || snapshot;
        currentSnapshot = snapshot;
        lastPacketTs = performance.now();
      }};
      renderLoop();
    </script>
    """
    components.html(animation_html, height=height + 60)


def _render_live_stage_panel(simulation: dict[str, Any], step_index: int) -> None:
    st.markdown(f"**Active Stage:** `{_current_stage_name(simulation, step_index)}`")
    panel_left, panel_right = st.columns([1.2, 0.8])
    with panel_left:
        _render_live_video_or_fallback(simulation, step_index)
    with panel_right:
        _render_live_code_inspector(simulation, step_index, heading="Live Code Inspector")
        st.markdown(_build_explanation_html(simulation, step_index), unsafe_allow_html=True)


def _render_live_video_or_fallback(simulation: dict[str, Any], step_index: int) -> None:
    try:
        if step_index >= len(simulation["times"]) - 1:
            st.image(
                _build_live_animation_bytes(simulation),
                use_container_width=True,
                caption="Synthetic YOLO-style feed dynamically animated from the completed simulation replay.",
            )
        else:
            st.image(
                _render_simulation_frame(simulation, step_index),
                use_container_width=True,
                caption="Synthetic YOLO-style feed streamed from the active simulation run.",
            )
    except Exception:
        st.warning("Simulation frame stream failed. Showing vector-based mission animation fallback.")
        st.plotly_chart(
            _build_3d_figure(simulation, upto_index=step_index),
            width="stretch",
            config={"displayModeBar": False},
        )


def _render_live_code_inspector(simulation: dict[str, Any], step_index: int, heading: str) -> None:
    stage_name = _current_stage_name(simulation, step_index)
    st.session_state["dashboard_active_stage"] = stage_name
    pulse_ratio = (step_index + 1) / max(len(simulation["times"]), 1)
    st.markdown(f"**{heading}**")
    st.caption(f"Stage pulse: {stage_name} | progress {pulse_ratio * 100:.0f}%")
    st.code(_stage_logic_snippet(stage_name), language="python")


def _current_stage_name(simulation: dict[str, Any], step_index: int) -> str:
    total_steps = max(len(simulation["times"]) - 1, 1)
    ratio = step_index / total_steps
    if ratio < 0.14:
        return "Detection"
    if ratio < 0.28:
        return "Tracking"
    if ratio < 0.48:
        return "Interception"
    if ratio < 0.64:
        return "Drift Applied"
    if ratio < 0.82:
        return "Path Changes"
    return "Target Redirected"


def _stage_logic_snippet(stage_name: str) -> str:
    snippets = {
        "Detection": """
observation = {"target_position": target_state.position, "time": np.array([time_s])}
detection = detector.detect(observation)
bbox_center = detection.position
confidence = detection.confidence
        """.strip(),
        "Tracking": """
predicted_state, predicted_covariance = kalman_predict(
    state=self._state,
    covariance=self._covariance,
    acceleration=control,
    dt=dt,
    process_noise=process_noise,
)
updated_state, updated_covariance, kalman_gain = kalman_update(
    predicted_state=predicted_state,
    predicted_covariance=predicted_covariance,
    measurement=measurement_xy,
    measurement_noise=measurement_noise,
)
        """.strip(),
        "Interception": """
prediction = predictor.predict(track)
plan = planner.plan(interceptor_estimate, prediction)
command = controller.compute_command(interceptor_estimate, plan)
        """.strip(),
        "Drift Applied": """
spoof_sample = spoof_toolkit.sample(
    true_position=target_state.position,
    interceptor_position=interceptor_state.position,
    time_s=time_s,
    attack_profile=directed_profile,
)
drifted_position = spoof_sample.spoofed_position
navigator_state = navigator.update(sensor_packet)
        """.strip(),
        "Path Changes": """
fused_path.append(navigator_state.position.copy())
interceptor_path.append(interceptor_state.position.copy())
target_path.append(target_state.position.copy())
        """.strip(),
        "Target Redirected": """
mission_success = final_distance_m <= desired_intercept_distance_m
redirect_status = "redirected" if mission_success else "still converging"
        """.strip(),
    }
    return snippets.get(stage_name, "# Awaiting mission start...")


@st.cache_data(show_spinner=False)
def _build_live_animation_bytes(simulation: dict[str, Any]) -> bytes:
    frame_count = len(simulation["times"])
    if frame_count == 0:
        return b""

    stride = max(1, frame_count // 36)
    images: list[Image.Image] = []
    for step_index in range(0, frame_count, stride):
        frame = _render_simulation_frame(simulation, step_index)
        images.append(Image.fromarray(frame))
    if images[-1] != Image.fromarray(_render_simulation_frame(simulation, frame_count - 1)):
        images.append(Image.fromarray(_render_simulation_frame(simulation, frame_count - 1)))

    payload = io.BytesIO()
    images[0].save(
        payload,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=120,
        loop=0,
        optimize=False,
    )
    return payload.getvalue()


def _render_simulation_frame(simulation: dict[str, Any], step_index: int) -> np.ndarray:
    canvas = np.full((540, 820, 3), 18, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (819, 539), (38, 54, 70), 2)
    cv2.putText(canvas, "Simulation Feed", (28, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 238, 244), 2, cv2.LINE_AA)

    target_positions = np.asarray(simulation["target_positions"], dtype=float)
    interceptor_positions = np.asarray(simulation["interceptor_positions"], dtype=float)
    detections = np.asarray(simulation["detections"], dtype=float)
    predicted_positions = np.asarray(simulation["predicted_positions"], dtype=float)
    target = target_positions[step_index]
    interceptor = interceptor_positions[step_index]
    detection = detections[step_index]
    predicted = predicted_positions[step_index]
    drifted_positions = np.asarray(simulation["drifted_positions"], dtype=float)
    drifted = drifted_positions[step_index]
    spoof_offset_m = float(np.linalg.norm(drifted - target))
    baseline_distance = None
    if simulation.get("comparison") is not None and step_index < len(simulation["comparison"]["distances"]):
        baseline_distance = float(simulation["comparison"]["distances"][step_index])
    confidence_score = max(
        0.0,
        min(
            1.0,
            1.0 - (
                float(np.linalg.norm(predicted - target)) /
                max(float(simulation["distances"][step_index]), 1.0)
            ),
        ),
    )
    canonical = _resolve_backend_canonical_metrics(simulation)
    canonical_rmse = _safe_float(canonical.get("rmse_m"), simulation["rmse_m"])
    canonical_intercept = canonical.get("earliest_interception_time_s")

    all_xy = np.vstack([target_positions[:, :2], interceptor_positions[:, :2], drifted_positions[:, :2], predicted_positions[:, :2]])
    min_xy = np.min(all_xy, axis=0)
    max_xy = np.max(all_xy, axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)

    def map_point(point_xy: np.ndarray) -> tuple[int, int]:
        x = int(70 + ((point_xy[0] - min_xy[0]) / span[0]) * 660)
        y = int(470 - ((point_xy[1] - min_xy[1]) / span[1]) * 360)
        return x, y

    for array, color, thickness in (
        (target_positions[: step_index + 1], (90, 145, 255), 3),
        (interceptor_positions[: step_index + 1], (255, 210, 94), 3),
        (drifted_positions[: step_index + 1], (120, 245, 160), 2),
    ):
        points = np.asarray([map_point(point[:2]) for point in array], dtype=np.int32).reshape((-1, 1, 2))
        if len(points) >= 2:
            cv2.polylines(canvas, [points], False, color, thickness)

    target_point = map_point(target[:2])
    interceptor_point = map_point(interceptor[:2])
    detection_point = map_point(detection[:2])
    predicted_point = map_point(predicted[:2])
    drifted_point = map_point(drifted[:2])
    bbox = (target_point[0] - 26, target_point[1] - 18, target_point[0] + 26, target_point[1] + 18)
    cv2.rectangle(canvas, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (85, 238, 135), 2)
    cv2.putText(canvas, "YOLO Target", (bbox[0], bbox[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (85, 238, 135), 2, cv2.LINE_AA)
    cv2.circle(canvas, target_point, 8, (90, 145, 255), -1)
    cv2.circle(canvas, interceptor_point, 8, (255, 210, 94), -1)
    cv2.circle(canvas, drifted_point, 6, (120, 245, 160), -1)
    cv2.circle(canvas, detection_point, 5, (255, 255, 255), -1)
    cv2.line(canvas, interceptor_point, predicted_point, (0, 242, 255), 1, cv2.LINE_AA)
    cv2.circle(canvas, predicted_point, 16, (0, 242, 255), 1)
    cv2.line(canvas, (predicted_point[0] - 20, predicted_point[1]), (predicted_point[0] + 20, predicted_point[1]), (0, 242, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (predicted_point[0], predicted_point[1] - 20), (predicted_point[0], predicted_point[1] + 20), (0, 242, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Lead Pursuit", (predicted_point[0] - 38, predicted_point[1] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 242, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Track ID: kalman-target", (560, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (214, 227, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Distance: {simulation['distances'][step_index]:6.2f} m", (560, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (214, 227, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"RMSE: {canonical_rmse:.3f} m", (560, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (214, 227, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Drift Rate: {simulation['drift_rate']:.2f} m/s", (560, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (214, 227, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Scenario: {simulation['scenario_type']}", (560, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (214, 227, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Time: {simulation['times'][step_index]:.2f} s", (560, 254), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (214, 227, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"EKF Confidence: {confidence_score * 100:5.1f}%", (560, 286), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 65), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Spoof Offset: {spoof_offset_m:5.2f} m", (560, 318), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 120, 120), 2, cv2.LINE_AA)
    if canonical_intercept is not None:
        cv2.putText(canvas, f"Intercept Time*: {_safe_float(canonical_intercept, 0.0):5.2f} s", (560, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 245, 160), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"*source: {str(canonical.get('source', 'backend'))}", (560, 382), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 245, 160), 1, cv2.LINE_AA)
    if baseline_distance is not None:
        cv2.putText(canvas, f"Clean Baseline: {baseline_distance:6.2f} m", (560, 414), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (115, 240, 160), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Legend", (560, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 238, 244), 2, cv2.LINE_AA)
    cv2.circle(canvas, (575, 462), 6, (90, 145, 255), -1)
    cv2.putText(canvas, "Target", (590, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (214, 227, 240), 2, cv2.LINE_AA)
    cv2.circle(canvas, (575, 486), 6, (255, 210, 94), -1)
    cv2.putText(canvas, "Interceptor", (590, 492), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (214, 227, 240), 2, cv2.LINE_AA)
    cv2.circle(canvas, (575, 510), 6, (120, 245, 160), -1)
    cv2.putText(canvas, "Drifted Nav Path", (590, 516), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (214, 227, 240), 2, cv2.LINE_AA)
    cv2.circle(canvas, (575, 532), 5, (255, 255, 255), -1)
    cv2.putText(canvas, "Detection", (590, 536), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (214, 227, 240), 1, cv2.LINE_AA)
    if simulation["success"] and step_index >= len(simulation["times"]) - 1:
        cv2.putText(canvas, "INTERCEPT ACHIEVED", (76, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (120, 245, 160), 3, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _build_explanation_html(simulation: dict[str, Any], step_index: int) -> str:
    times = simulation["times"]
    positions_i = np.asarray(simulation["interceptor_positions"], dtype=float)
    positions_t = np.asarray(simulation["target_positions"], dtype=float)
    drifted = np.asarray(simulation["drifted_positions"], dtype=float)
    relative = positions_t[step_index] - positions_i[step_index]
    distance = float(np.linalg.norm(relative))
    drift_shift = float(drifted[step_index][0] - positions_i[step_index][0])
    if step_index == 0:
        closing_velocity = 0.0
    else:
        previous_distance = float(np.linalg.norm(positions_t[step_index - 1] - positions_i[step_index - 1]))
        dt = max(times[step_index] - times[step_index - 1], 1e-6)
        closing_velocity = max((previous_distance - distance) / dt, 0.0)
    spoof_offset = float(np.linalg.norm(drifted[step_index] - positions_t[step_index]))
    comparison = simulation.get("comparison")
    comparison_note = ""
    if comparison is not None and step_index < len(comparison["distances"]):
        distance_delta = float(simulation["distances"][step_index] - comparison["distances"][step_index])
        comparison_note = f" Relative to the clean baseline, distance is shifted by {distance_delta:+.2f} m."

    canonical = _resolve_backend_canonical_metrics(simulation)
    canonical_rmse = _safe_float(canonical.get("rmse_m"), simulation["tracking_errors"][step_index])
    earliest_intercept = canonical.get("earliest_interception_time_s")

    explanation = (
        f"The interceptor is increasing closing velocity to {closing_velocity:.2f} m/s while the Kalman filter "
        f"holds tracking RMSE near {canonical_rmse:.2f} m. "
        f"Adaptive spoofing offset is {spoof_offset:.2f} m at this step. "
        "We simulate how coordinate drift affects the navigation estimation layer (EKF), resulting in gradual trajectory deviation."
        f"{comparison_note}"
    )
    if earliest_intercept is not None:
        explanation += (
            f" Backend canonical earliest intercept time is "
            f"{_safe_float(earliest_intercept, 0.0):.3f} s."
        )
    return f"""
    <div class="workflow-stack">
      <div class="workflow-callout"><strong>Mission Interpretation</strong><br/>{explanation}</div>
      <div class="workflow-mini-grid">
        <div class="workflow-mini-card"><strong>Detection</strong><br/>YOLO localizes the drone using a bounding-box measurement.</div>
        <div class="workflow-mini-card"><strong>Tracking</strong><br/>Kalman fusion smooths the target state and suppresses noise.</div>
        <div class="workflow-mini-card"><strong>Interception</strong><br/>Guidance closes geometry and commits the interceptor to the target corridor.</div>
        <div class="workflow-mini-card"><strong>Drift Applied</strong><br/>Coordinate drift is injected into the navigation estimate instead of the true path.</div>
        <div class="workflow-mini-card"><strong>Path Changes</strong><br/>The visible route bends as the estimate and controller respond together.</div>
        <div class="workflow-mini-card"><strong>Target Redirected</strong><br/>Mission success is judged by where the target path ends up.</div>
      </div>
    </div>
    """


def _build_workflow_html(simulation: dict[str, Any]) -> str:
    canonical = _resolve_backend_canonical_metrics(simulation)
    intercept_status = "Interception confirmed" if simulation["success"] else "Interceptor still outside terminal envelope"
    final_distance_m = _safe_float(canonical.get("final_distance_m"), simulation["final_distance_m"])
    final_rmse_m = _safe_float(canonical.get("rmse_m"), simulation["rmse_m"])
    earliest_intercept = canonical.get("earliest_interception_time_s")
    mean_intercept = canonical.get("mean_interception_time_s")
    intercept_line = (
        f"Earliest intercept: {_safe_float(earliest_intercept, 0.0):.3f} s, "
        f"mean intercept across targets: {_safe_float(mean_intercept, 0.0):.3f} s."
        if earliest_intercept is not None
        else "Interception time pending."
    )
    return f"""
    <div class="workflow-stack">
      <div class="workflow-step">
        <div class="workflow-index">01</div>
        <div class="workflow-body"><strong>Detection</strong><br/>The perception engine emits a measurement aligned to the target silhouette.</div>
      </div>
      <div class="workflow-step">
        <div class="workflow-index">02</div>
        <div class="workflow-body"><strong>Tracking + Kalman</strong><br/>The tracker fuses measurements over time to stabilize state and reject jitter.</div>
      </div>
      <div class="workflow-step">
        <div class="workflow-index">03</div>
        <div class="workflow-body"><strong>Interception</strong><br/>Guidance closes geometry and pushes the interceptor into the target corridor.</div>
      </div>
      <div class="workflow-step">
        <div class="workflow-index">04</div>
        <div class="workflow-body"><strong>Drift Applied</strong><br/>We simulate how coordinate drift affects the navigation estimation layer (EKF), resulting in gradual trajectory deviation.</div>
      </div>
      <div class="workflow-step">
        <div class="workflow-index">05</div>
        <div class="workflow-body"><strong>Path Changes</strong><br/>The estimated route shifts first, then the controller responds and the visible path bends.</div>
      </div>
      <div class="workflow-step">
        <div class="workflow-index">06</div>
        <div class="workflow-body"><strong>Target Redirected</strong><br/>{intercept_status}. Final miss distance is {final_distance_m:.2f} m and RMSE is {final_rmse_m:.3f} m. {intercept_line}</div>
      </div>
    </div>
    """


def _build_live_stage_progress_html(stage_name: str, ratio: float) -> str:
    ordered_stages = [
        "Detection",
        "Tracking",
        "Interception",
        "Drift Applied",
        "Path Changes",
        "Target Redirected",
    ]
    active_index = ordered_stages.index(stage_name) if stage_name in ordered_stages else 0
    chips: list[str] = []
    for index, name in enumerate(ordered_stages):
        is_active = index == active_index
        is_done = index < active_index
        if is_active:
            bg = "rgba(0,242,255,0.22)"
            border = "#00F2FF"
            text = "#E8FCFF"
        elif is_done:
            bg = "rgba(115,240,160,0.16)"
            border = "#73F0A0"
            text = "#DDFCEF"
        else:
            bg = "rgba(22,34,46,0.92)"
            border = "#24384A"
            text = "#9FB8CA"
        chips.append(
            f'<span style="display:inline-block;padding:6px 10px;border-radius:10px;'
            f'border:1px solid {border};background:{bg};color:{text};font-size:0.82rem;'
            f'margin:0 6px 6px 0;">{name}</span>'
        )
    progress_pct = float(np.clip(ratio, 0.0, 1.0) * 100.0)
    return (
        '<div class="stage-card" style="margin-bottom:8px;">'
        '<strong>Platform Stages (Live)</strong><br/>'
        + "".join(chips)
        + f'<div style="margin-top:8px;color:#8FE8FF;">Progress: {progress_pct:.0f}% | Active: {stage_name}</div>'
        + "</div>"
    )


def _show_live_mission_demo(
    simulation: dict[str, Any],
    backend_state: dict[str, Any] | None = None,
     upto_index: int | None = None,
    key_suffix: str = "static",
) -> None:
    animated_payload = _build_live_animation_bytes(simulation) if upto_index is None else None
    main_frame_payload = _render_simulation_frame(simulation, upto_index) if upto_index is not None else None
    left_col, right_col = st.columns([1.35, 0.65])
    with left_col:
        if upto_index is None:
            st.image(
                animated_payload,
                use_container_width=True,
                caption="Primary mission playback dynamically animated from the active run.",
            )
        else:
            st.image(
                main_frame_payload,
                use_container_width=True,
                caption="Primary mission playback is being generated from the active run.",
            )
    with right_col:
        step = max(0, min(int(upto_index if upto_index is not None else len(simulation["times"]) - 1), len(simulation["times"]) - 1))
        st.markdown('<div class="pip-label">Picture-in-Picture Views</div>', unsafe_allow_html=True)
        if upto_index is None:
            st.image(animated_payload, use_container_width=True, caption="Interceptor FPV (Animated Replay)")
        else:
            st.image(_render_simulation_frame(simulation, step), use_container_width=True, caption="Interceptor FPV (Live Frame)")
        st.plotly_chart(
            _build_3d_figure(simulation, upto_index=step),
            width="stretch",
            key=f"pip_3d_{key_suffix}_{step}",
            config={"displayModeBar": False},
        )
    columns = st.columns(3)
    canonical = _resolve_backend_canonical_metrics(simulation)
    columns[0].metric("Frames", f"{len(simulation['times']) if upto_index is None else upto_index + 1}")
    if upto_index is None:
        distance_value = _safe_float(canonical.get("final_distance_m"), simulation["final_distance_m"])
    else:
        distance_value = simulation["distances"][upto_index]
    columns[1].metric("Final Distance", f"{distance_value:.2f} m")
    snapshot = (backend_state or {}).get("snapshot", {}) if isinstance(backend_state, dict) else {}
    backend_complete = str(snapshot.get("status", "")).lower() == "complete"
    mission_result = "INTERCEPT" if (backend_complete or simulation["success"]) and (upto_index is None or upto_index >= len(simulation["times"]) - 1) else "ACTIVE"
    columns[2].metric("Mission Result", mission_result)
    if upto_index is None and canonical.get("earliest_interception_time_s") is not None:
        st.caption(
            f"Backend canonical intercept time (earliest target): "
            f"{_safe_float(canonical.get('earliest_interception_time_s'), 0.0):.3f} s "
            f"[source: {canonical.get('source', 'n/a')}]"
        )


def set_custom_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --obsidian: #0A0A0B;
            --surface: #101319;
            --surface-2: #0E1117;
            --stroke: rgba(0, 240, 255, 0.32);
            --stroke-strong: rgba(0, 240, 255, 0.60);
            --cyber-cyan: #00F0FF;
            --amber: #FFB800;
            --text-main: #EAF4FF;
            --text-dim: #9CB4C8;
            --ok: #4DFFB2;
        }
        html, body, [data-testid="stAppViewContainer"], .stApp {
            height: 100%;
        }
        .stApp {
            background:
                radial-gradient(circle at 20% 0%, rgba(0, 240, 255, 0.12), transparent 28%),
                radial-gradient(circle at 80% 0%, rgba(255, 184, 0, 0.08), transparent 24%),
                linear-gradient(180deg, #07090D 0%, #0A0A0B 62%, #090B10 100%);
            color: var(--text-main);
            font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            overflow-x: hidden;
            overflow-y: auto;
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 100% !important;
            padding-top: 0.55rem;
            padding-bottom: 0.30rem;
            height: calc(100vh - 1.1rem);
            overflow-x: hidden;
            overflow-y: auto;
        }
        [data-baseweb="tab-panel"] {
            height: calc(100vh - 248px);
            overflow-y: auto;
            overflow-x: hidden;
            padding-right: 0.35rem;
        }
        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(12, 15, 22, 0.98), rgba(7, 9, 14, 0.98));
            border-right: 1px solid var(--stroke);
        }
        [data-testid="stSidebar"] .stForm {
            background: linear-gradient(180deg, rgba(18, 23, 32, 0.76), rgba(10, 13, 20, 0.72));
            border: 1px solid rgba(0, 240, 255, 0.22);
            border-radius: 14px;
            padding: 0.65rem 0.65rem 0.35rem 0.65rem;
        }
        [data-testid="stSidebar"] button {
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }
        [data-testid="stSidebar"] button:hover {
            transform: translateY(-1px);
            box-shadow: 0 0 16px rgba(0, 240, 255, 0.28);
            border-color: rgba(0, 240, 255, 0.70);
        }
        [data-testid="stSidebar"] button[kind="primary"] {
            background: linear-gradient(120deg, rgba(0, 240, 255, 0.32), rgba(0, 180, 220, 0.20));
            border: 1px solid rgba(0, 240, 255, 0.70);
            color: #E8FDFF;
            font-weight: 700;
            letter-spacing: 0.02rem;
        }
        [data-testid="stSidebar"] [data-testid="stSlider"] {
            border: 1px solid rgba(0, 240, 255, 0.16);
            border-radius: 10px;
            padding: 0.18rem 0.35rem;
            background: rgba(7, 12, 18, 0.45);
            transition: border-color 0.18s ease, box-shadow 0.18s ease;
        }
        [data-testid="stSidebar"] [data-testid="stSlider"]:hover {
            border-color: rgba(0, 240, 255, 0.42);
            box-shadow: 0 0 10px rgba(0, 240, 255, 0.12);
        }
        h1, h2, h3 {
            color: var(--cyber-cyan) !important;
            letter-spacing: 0.08rem;
            text-transform: uppercase;
        }
        [data-testid="stMetric"] {
            border: 1px solid var(--stroke) !important;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02), 0 10px 24px rgba(0, 0, 0, 0.35);
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(18, 22, 30, 0.76), rgba(10, 13, 20, 0.80));
            backdrop-filter: blur(14px);
        }
        [data-testid="stPlotlyChart"],
        [data-testid="stDataFrame"],
        [data-testid="stImage"],
        [data-testid="stVideo"] {
            border: 1px solid rgba(0, 240, 255, 0.22);
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(12, 16, 24, 0.72), rgba(8, 11, 16, 0.76));
            padding: 0.22rem;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.30);
        }
        [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"],
        [data-testid="stDataFrame"] * {
            font-family: "JetBrains Mono", "Consolas", "SFMono-Regular", monospace !important;
            font-variant-numeric: tabular-nums;
        }
        .tactical-panel {
            padding: 0.6rem 0.8rem;
            margin-bottom: 0.75rem;
            border: 1px solid var(--stroke);
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(16, 20, 28, 0.72), rgba(10, 12, 18, 0.78));
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.28);
        }
        .tactical-panel-label {
            color: var(--cyber-cyan);
            text-transform: uppercase;
            letter-spacing: 0.16rem;
            font-size: 0.78rem;
            font-weight: 700;
            margin: 0.28rem 0 0.55rem 0;
            border-left: 3px solid rgba(0, 240, 255, 0.7);
            padding-left: 0.45rem;
        }
        .hud-card {
            padding: 0.85rem 1rem;
            min-height: 108px;
            border: 1px solid rgba(0, 240, 255, 0.32);
            border-radius: 12px;
            background: linear-gradient(180deg, rgba(17, 21, 31, 0.78), rgba(11, 14, 22, 0.82));
        }
        .hud-label {
            color: #9ad9e6;
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.10rem;
            margin-bottom: 0.35rem;
        }
        .hud-value {
            color: #e9f9ff;
            font-size: 1.35rem;
            font-weight: 700;
            text-shadow: 0 0 14px rgba(0, 240, 255, 0.30);
            font-family: "JetBrains Mono", "Consolas", monospace;
            font-variant-numeric: tabular-nums;
        }
        .hud-inline {
            color: #F3FCFF;
            font-size: 0.98rem;
            margin-bottom: 0.22rem;
        }
        .hud-heartbeat {
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .heartbeat-row {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin-top: 0.35rem;
        }
        .heartbeat-dot {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            display: inline-block;
            background: rgba(77, 255, 178, 0.16);
            border: 2px solid rgba(77, 255, 178, 0.95);
        }
        .heartbeat-live {
            box-shadow: 0 0 0 rgba(77, 255, 178, 0.78);
            animation: pulse-cyber 1.2s infinite;
        }
        .heartbeat-idle {
            opacity: 0.35;
        }
        .global-alert-banner {
            border: 1px solid rgba(255, 184, 0, 0.75);
            border-radius: 12px;
            background:
                linear-gradient(95deg, rgba(45, 33, 8, 0.86), rgba(28, 22, 8, 0.92)),
                radial-gradient(circle at 8% 50%, rgba(255, 184, 0, 0.22), transparent 46%);
            padding: 0.70rem 0.88rem;
            margin: 0.40rem 0 0.70rem 0;
            box-shadow: 0 0 18px rgba(255, 184, 0, 0.25);
            display: grid;
            grid-template-columns: auto auto 1fr;
            gap: 0.7rem;
            align-items: center;
        }
        .alert-tag {
            font-size: 0.72rem;
            font-weight: 700;
            color: #2B1C03;
            background: var(--amber);
            border-radius: 999px;
            padding: 0.20rem 0.58rem;
            text-transform: uppercase;
            letter-spacing: 0.08rem;
        }
        .alert-text {
            color: #FFE6AE;
            font-weight: 700;
            letter-spacing: 0.04rem;
            font-family: "JetBrains Mono", "Consolas", monospace;
        }
        .alert-subtext {
            color: #FFD992;
            opacity: 0.95;
            font-size: 0.90rem;
        }
        .telemetry-overlay {
            position: sticky;
            top: 1rem;
            padding: 0.85rem 1rem;
            margin-top: 0.8rem;
            border: 1px solid rgba(0, 240, 255, 0.28);
            border-radius: 12px;
            background: linear-gradient(180deg, rgba(14, 18, 26, 0.75), rgba(9, 12, 18, 0.8));
        }
        .telemetry-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.5rem 0.8rem;
            color: #F3FCFF;
            font-size: 0.92rem;
            font-family: "JetBrains Mono", "Consolas", monospace;
            font-variant-numeric: tabular-nums;
        }
        .stage-card {
            padding: 14px 16px;
            margin-bottom: 12px;
            line-height: 1.55;
            color: #eaf2f7;
            background: linear-gradient(180deg, rgba(14, 18, 26, 0.88), rgba(10, 13, 20, 0.86));
            border: 1px solid rgba(0, 240, 255, 0.26);
            border-radius: 12px;
        }
        .status-red {
            border-color: #FF3131 !important;
            box-shadow: 0 0 14px rgba(255, 49, 49, 0.35) !important;
        }
        .status-green {
            border-color: #00FF41 !important;
            box-shadow: 0 0 14px rgba(0, 255, 65, 0.35) !important;
        }
        .panel-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin-top: 10px;
        }
        .pip-label {
            color: var(--cyber-cyan);
            text-transform: uppercase;
            letter-spacing: 0.12rem;
            font-size: 0.76rem;
            margin-bottom: 0.45rem;
        }
        .workflow-stack {
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
        }
        .workflow-step {
            display: grid;
            grid-template-columns: 64px 1fr;
            gap: 0.85rem;
            align-items: stretch;
        }
        .workflow-index {
            border: 1px solid rgba(0, 255, 65, 0.6);
            border-radius: 12px;
            background: rgba(0, 255, 65, 0.08);
            color: #00FF41;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.0rem;
            min-height: 64px;
        }
        .workflow-body, .workflow-callout, .workflow-mini-card {
            border: 1px solid rgba(0, 229, 255, 0.42);
            border-radius: 14px;
            background: rgba(8, 17, 27, 0.72);
            padding: 0.9rem 1rem;
            line-height: 1.55;
        }
        .workflow-callout {
            border-color: rgba(0, 255, 65, 0.55);
            box-shadow: inset 0 0 0 1px rgba(0, 255, 65, 0.08);
        }
        .workflow-mini-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
        }
        .bms-glass {
            background: linear-gradient(180deg, rgba(18, 24, 34, 0.72), rgba(7, 11, 18, 0.78));
            border: 1px solid rgba(0, 240, 255, 0.26);
            border-radius: 12px;
        }
        [data-testid="stTabs"] button {
            color: #93d7e7;
            font-family: "Inter", "Segoe UI", sans-serif !important;
            font-weight: 600;
            transition: color 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--cyber-cyan);
            border-bottom: 2px solid var(--cyber-cyan);
            box-shadow: 0 2px 0 0 rgba(0, 240, 255, 0.35);
        }
        .data-tile {
            border: 1px solid rgba(0, 240, 255, 0.30);
            border-radius: 12px;
            background: linear-gradient(180deg, rgba(14, 18, 27, 0.88), rgba(8, 12, 19, 0.92));
            padding: 0.75rem 0.80rem;
            margin-bottom: 0.56rem;
            transition: border-color 0.16s ease, box-shadow 0.16s ease, transform 0.16s ease;
        }
        .data-tile:hover {
            border-color: rgba(0, 240, 255, 0.58);
            box-shadow: 0 0 14px rgba(0, 240, 255, 0.22);
            transform: translateY(-1px);
        }
        .data-tile-ok {
            border-left: 4px solid rgba(77, 255, 178, 0.82);
        }
        .data-tile-warn {
            border-left: 4px solid rgba(255, 184, 0, 0.88);
        }
        .data-tile-target {
            color: #DDF5FF;
            font-size: 0.86rem;
            font-weight: 700;
            margin-bottom: 0.40rem;
            letter-spacing: 0.03rem;
        }
        .data-tile-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.35rem 0.8rem;
        }
        .data-tile-grid span {
            display: block;
            font-size: 0.67rem;
            color: #88a8bf;
            text-transform: uppercase;
            letter-spacing: 0.06rem;
        }
        .data-tile-grid strong {
            display: block;
            margin-top: 0.05rem;
            color: #F1FAFF;
            font-family: "JetBrains Mono", "Consolas", monospace;
            font-variant-numeric: tabular-nums;
            font-size: 0.92rem;
        }
        @keyframes pulse-cyber {
            0% { box-shadow: 0 0 0 0 rgba(77, 255, 178, 0.7); opacity: 0.70; }
            70% { box-shadow: 0 0 0 9px rgba(77, 255, 178, 0.0); opacity: 1.0; }
            100% { box-shadow: 0 0 0 0 rgba(77, 255, 178, 0.0); opacity: 0.70; }
        }
        @media (max-width: 1080px) {
            .stApp {
                overflow: auto;
            }
            [data-testid="stMainBlockContainer"] {
                height: auto;
                overflow: visible;
            }
            [data-baseweb="tab-panel"] {
                height: auto;
                overflow: visible;
            }
            .panel-grid {
                grid-template-columns: 1fr;
            }
            .telemetry-grid {
                grid-template-columns: 1fr;
            }
            .global-alert-banner {
                grid-template-columns: 1fr;
                gap: 0.4rem;
            }
            .workflow-step,
            .workflow-mini-grid {
                grid-template-columns: 1fr;
            }
            .data-tile-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _inject_styles() -> None:
    set_custom_style()


if __name__ == "__main__":
    main()
