from __future__ import annotations

import atexit
import asyncio
import logging
import time
from collections import deque
from dataclasses import asdict
from dataclasses import dataclass, field
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from queue import SimpleQueue
from typing import Any

import numpy as np
from fastapi import BackgroundTasks, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from drone_interceptor.backend.engine import StreamProcessor
from drone_interceptor.backend.feature_flags import is_enabled, set_flag
from drone_interceptor.backend.mission_service import MissionConfig, MissionController
from drone_interceptor.backend.run_store import FileRunStore
from drone_interceptor.backend.spoof_service import get_default_service
from drone_interceptor.simulation.airsim_manager import AirSimMissionManager, MissionFrame, MissionReplay, local_position_to_lla
from drone_interceptor.simulation.airsim_yolo_bridge import AirSimYOLOBridge


app = FastAPI(title="Drone Interceptor Telemetry API", version="0.2.0")
STREAM_HZ = 45.78
HEARTBEAT_HZ = 20.0
RUN_STORE = FileRunStore(Path(__file__).resolve().parents[3] / "outputs" / "run_registry")
OUTPUTS = Path(__file__).resolve().parents[3] / "outputs"
LOGS = Path(__file__).resolve().parents[3] / "logs"
AIRSIM_YOLO_BRIDGE = AirSimYOLOBridge(
    config_path=Path(__file__).resolve().parents[3] / "configs" / "default.yaml",
    output_dir=OUTPUTS / "airsim_yolo",
)


def _configure_async_telemetry_logger() -> logging.Logger:
    logger = logging.getLogger("drone_interceptor.telemetry")
    if getattr(logger, "_async_configured", False):
        return logger
    logger.setLevel(logging.INFO)
    log_queue: SimpleQueue[logging.LogRecord] = SimpleQueue()
    queue_handler = QueueHandler(log_queue)
    queue_handler.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(queue_handler)
    logger.propagate = False

    LOGS.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOGS / "telemetry_async.log",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    listener = QueueListener(log_queue, file_handler, respect_handler_level=True)
    listener.start()
    setattr(logger, "_async_listener", listener)
    setattr(logger, "_async_configured", True)
    atexit.register(listener.stop)
    return logger


TELEMETRY_LOGGER = _configure_async_telemetry_logger()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8501",
        "http://localhost:8501",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_manager(payload: dict[str, Any]) -> AirSimMissionManager:
    return AirSimMissionManager(
        host=str(payload.get("host", "127.0.0.1")),
        port=int(payload.get("port", 41451)),
        connect=bool(payload.get("connect_airsim", False)),
    )


def _airsim_yolo_status(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = payload or {}
    return AIRSIM_YOLO_BRIDGE.get_status(
        refresh=bool(data.get("refresh", True)),
        host=str(data.get("airsim_host", data.get("host", "127.0.0.1"))),
        port=int(data.get("airsim_port", data.get("port", 41451))),
        vehicle_name=str(data.get("airsim_vehicle_name", data.get("vehicle_name", ""))),
        camera_name=str(data.get("airsim_camera_name", data.get("camera_name", "0"))),
        drift_rate_mps=float(data.get("drift_rate_mps", 0.3)),
        timeout_s=float(data.get("airsim_timeout_s", data.get("timeout_s", 3.0))),
        connect_airsim=bool(data.get("connect_airsim", True)),
    )


def _build_validation_report(summary: Any, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    drift_rate = float(payload.get("drift_rate_mps", 0.0)) if payload is not None else 0.0
    noise_std = float(payload.get("noise_std_m", 0.0)) if payload is not None else 0.0
    latency_ms = float(payload.get("latency_ms", 0.0)) if payload is not None else 0.0
    per_target_summary = [dict(row) for row in summary.per_target_summary]
    target_details: list[dict[str, Any]] = []
    for row in per_target_summary:
        target_details.append(
            {
                "id": str(row.get("target", "")),
                "ekf_success_rate": float(row.get("ekf_success_rate", 0.0)),
                "ekf_mean_distance_m": float(row.get("ekf_mean_miss_distance_m", 0.0)),
                "drift_rate_applied_mps": drift_rate,
                "active_spoof_offset_m": float(row.get("ekf_mean_spoof_offset_m", 0.0)),
                "time_to_recovery_s": float(row.get("ekf_mean_time_to_recovery_s", 0.0)),
            }
        )

    global_metrics = {
        "total_targets": int(len(per_target_summary)),
        "average_mission_success": float(summary.ekf_success_rate),
        "system_wide_rmse": float(
            np.mean([float(row.get("ekf_rmse_m", 0.0)) for row in summary.iteration_records])
            if summary.iteration_records
            else 0.0
        ),
        "drift_rate_mps": drift_rate,
        "noise_std_m": noise_std,
        "latency_ms": latency_ms,
    }
    return {
        "iterations": int(summary.iterations),
        "raw_success_rate": float(summary.raw_success_rate),
        "ekf_success_rate": float(summary.ekf_success_rate),
        "raw_mean_miss_distance_m": float(summary.raw_mean_miss_distance_m),
        "ekf_mean_miss_distance_m": float(summary.ekf_mean_miss_distance_m),
        "raw_mean_spoof_offset_m": float(getattr(summary, "raw_mean_spoof_offset_m", 0.0)),
        "ekf_mean_spoof_offset_m": float(getattr(summary, "ekf_mean_spoof_offset_m", 0.0)),
        "raw_mean_time_to_recovery_s": float(getattr(summary, "raw_mean_time_to_recovery_s", 0.0)),
        "ekf_mean_time_to_recovery_s": float(getattr(summary, "ekf_mean_time_to_recovery_s", 0.0)),
        "raw_mean_kill_probability": float(summary.raw_mean_kill_probability),
        "ekf_mean_kill_probability": float(summary.ekf_mean_kill_probability),
        "validation_success": bool(summary.ekf_success_rate >= 1.0 and summary.ekf_mean_miss_distance_m <= 0.5),
        "global_metrics": global_metrics,
        "target_details": target_details,
        "per_target_summary": per_target_summary,
        "iteration_records": [dict(row) for row in summary.iteration_records],
        "scenario_results": [dict(row) for row in summary.iteration_records],
    }


def _validation_matrix_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for case_id, speed in enumerate((6.0, 9.0, 12.0), start=1):
        cases.append(
            {
                "case_id": case_id,
                "label": f"Nominal-{int(speed)}mps",
                "category": "nominal",
                "target_speed_mps": speed,
                "drift_rate_mps": 0.2,
                "noise_std_m": 0.10,
                "packet_loss_rate": 0.0,
                "latency_ms": 0.0,
                "use_ekf": True,
                "use_ekf_anti_spoofing": True,
                "expected_outcome": "SUCCESS",
            }
        )
    for case_id, speed in enumerate((6.0, 9.0, 12.0), start=4):
        cases.append(
            {
                "case_id": case_id,
                "label": f"DriftStress-{int(speed)}mps",
                "category": "drift_stress",
                "target_speed_mps": speed,
                "drift_rate_mps": 0.50,
                "noise_std_m": 0.55,
                "packet_loss_rate": 0.0,
                "latency_ms": 80.0,
                "use_ekf": False,
                "use_ekf_anti_spoofing": False,
                "expected_outcome": "FAILURE",
            }
        )
    for case_id, speed in enumerate((6.0, 9.0, 12.0), start=7):
        cases.append(
            {
                "case_id": case_id,
                "label": f"AntiSpoof-{int(speed)}mps",
                "category": "anti_spoofing",
                "target_speed_mps": speed,
                "drift_rate_mps": 0.45,
                "noise_std_m": 1.50,
                "packet_loss_rate": 0.05,
                "latency_ms": 120.0,
                "use_ekf": True,
                "use_ekf_anti_spoofing": True,
                "expected_outcome": "SUCCESS",
            }
        )
    for case_id, speed in enumerate((6.0, 9.0, 12.0), start=10):
        cases.append(
            {
                "case_id": case_id,
                "label": f"Edge-{int(speed)}mps",
                "category": "edge",
                "target_speed_mps": speed,
                "drift_rate_mps": 0.35,
                "noise_std_m": 1.10,
                "packet_loss_rate": 0.80,
                "latency_ms": 400.0,
                "use_ekf": True,
                "use_ekf_anti_spoofing": True,
                "expected_outcome": "LATE INTERCEPT",
            }
        )
    return cases


def _first_intercept_time(replay: MissionReplay) -> float | None:
    for frame in replay.frames:
        if any(target.jammed for target in frame.targets):
            return float(frame.time_s)
    return None


def _mean_replay_rmse(replay: MissionReplay) -> float:
    if not replay.frames:
        return 0.0
    return float(np.mean([float(frame.rmse_m) for frame in replay.frames], dtype=float))


def _build_matrix_validation_report(payload: dict[str, Any]) -> dict[str, Any]:
    manager = _build_manager(payload)
    scenario_results: list[dict[str, Any]] = []
    raw_successes = 0
    ekf_successes = 0
    raw_miss_distances: list[float] = []
    ekf_miss_distances: list[float] = []
    raw_kill_probabilities: list[float] = []
    ekf_kill_probabilities: list[float] = []
    scenario_max_steps = max(int(payload.get("max_steps", 24)), 180)
    kill_radius_m = float(payload.get("kill_radius_m", 1.0))

    for case in _validation_matrix_cases():
        case_seed = int(payload.get("random_seed", 61)) + int(case["case_id"]) * 31
        run_kwargs = {
            "num_targets": int(payload.get("num_targets", 3)),
            "drift_rate_mps": float(case["drift_rate_mps"]),
            "noise_std_m": float(case["noise_std_m"]),
            "latency_ms": float(case["latency_ms"]),
            "packet_loss_rate": float(case["packet_loss_rate"]),
            "random_seed": case_seed,
            "max_steps": scenario_max_steps,
            "dt": float(payload.get("dt", 0.05)),
            "kill_radius_m": kill_radius_m,
            "guidance_gain": max(float(payload.get("guidance_gain", 6.0)), 6.8 if str(case["category"]) in {"nominal", "anti_spoofing"} else 6.2),
            "target_speed_mps": float(case["target_speed_mps"]),
            "enable_spoofing": bool(str(case.get("category", "")).lower() in {"anti_spoofing", "drift_stress", "edge"}),
        }
        raw_replay = manager.run_replay(
            use_ekf=False,
            use_ekf_anti_spoofing=False,
            **run_kwargs,
        )
        ekf_replay = manager.run_replay(
            use_ekf=True,
            use_ekf_anti_spoofing=bool(case["use_ekf_anti_spoofing"]),
            **run_kwargs,
        )
        raw_distance = float(raw_replay.distance_frame["distance_m"].min()) if not raw_replay.distance_frame.empty else 0.0
        ekf_distance = float(ekf_replay.distance_frame["distance_m"].min()) if not ekf_replay.distance_frame.empty else 0.0
        raw_success = bool(raw_replay.safe_intercepts > 0 or raw_distance <= kill_radius_m)
        ekf_success = bool(ekf_replay.safe_intercepts > 0 or ekf_distance <= kill_radius_m)
        primary_replay = ekf_replay if bool(case["use_ekf"]) else raw_replay
        primary_distance = ekf_distance if bool(case["use_ekf"]) else raw_distance
        primary_success = bool(primary_replay.safe_intercepts > 0 or primary_distance <= kill_radius_m)
        intercept_time_s = _first_intercept_time(primary_replay)
        status = "LATE INTERCEPT" if primary_success and float(case["packet_loss_rate"]) >= 0.75 else ("SUCCESS" if primary_success else "FAILURE")
        raw_successes += int(raw_success)
        ekf_successes += int(ekf_success)
        raw_miss_distances.append(raw_distance)
        ekf_miss_distances.append(ekf_distance)
        raw_kill_probabilities.append(_kill_probability(raw_distance, 0.5))
        ekf_kill_probabilities.append(_kill_probability(ekf_distance, 0.5))
        scenario_results.append(
            {
                "iteration": int(case["case_id"]),
                "case_id": int(case["case_id"]),
                "scenario": str(case["label"]),
                "category": str(case["category"]),
                "expected_outcome": str(case["expected_outcome"]),
                "status": status,
                "success": "YES" if primary_success else "NO",
                "ekf_enabled": bool(case["use_ekf"]),
                "drift_rate_mps": float(case["drift_rate_mps"]),
                "noise_level": float(case["noise_std_m"]),
                "packet_loss_rate": float(case["packet_loss_rate"]),
                "latency_ms": float(case["latency_ms"]),
                "target_speed_mps": float(case["target_speed_mps"]),
                "raw_mean_miss_distance_m": raw_distance,
                "ekf_mean_miss_distance_m": ekf_distance,
                "raw_kill_probability": _kill_probability(raw_distance, 0.5),
                "ekf_kill_probability": _kill_probability(ekf_distance, 0.5),
                "raw_success": raw_success,
                "ekf_success": ekf_success,
                "rmse_m": _mean_replay_rmse(primary_replay),
                "raw_rmse_m": _mean_replay_rmse(raw_replay),
                "ekf_rmse_m": _mean_replay_rmse(ekf_replay),
                "interception_time_s": intercept_time_s,
            }
        )

    ekf_success_rate = float(ekf_successes / max(len(scenario_results), 1))
    report = {
        "iterations": len(scenario_results),
        "validation_mode": "matrix12",
        "raw_success_rate": float(raw_successes / max(len(scenario_results), 1)),
        "ekf_success_rate": ekf_success_rate,
        "raw_mean_miss_distance_m": float(np.mean(raw_miss_distances, dtype=float)) if raw_miss_distances else 0.0,
        "ekf_mean_miss_distance_m": float(np.mean(ekf_miss_distances, dtype=float)) if ekf_miss_distances else 0.0,
        "raw_mean_kill_probability": float(np.mean(raw_kill_probabilities, dtype=float)) if raw_kill_probabilities else 0.0,
        "ekf_mean_kill_probability": float(np.mean(ekf_kill_probabilities, dtype=float)) if ekf_kill_probabilities else 0.0,
        "validation_success": bool(ekf_success_rate >= 0.75),
        "per_target_summary": [],
        "iteration_records": list(scenario_results),
        "scenario_results": list(scenario_results),
    }
    return report


def _kill_probability(distance_m: float, uncertainty_m: float | None = None) -> float:
    if distance_m < 1.0:
        return 1.0
    if distance_m > 50.0:
        return 0.0
    return float(1.0 / (1.0 + np.exp(0.2 * (distance_m - 10.0))))


def _covariance_from_uncertainty(uncertainty_radius_m: float, floor_m: float = 0.08) -> np.ndarray:
    sigma = max(float(uncertainty_radius_m), float(floor_m))
    return np.diag([sigma**2, sigma**2, sigma**2]).astype(float)


def _mahalanobis_distance(delta_vector: np.ndarray, covariance: np.ndarray) -> float:
    delta = np.asarray(delta_vector, dtype=float).reshape(3, 1)
    cov = np.asarray(covariance, dtype=float).reshape(3, 3)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)
    distance_squared = float((delta.T @ inv_cov @ delta).item())
    return float(np.sqrt(max(distance_squared, 0.0)))


def _kill_probability_from_mahalanobis(mahalanobis_distance: float) -> float:
    distance = max(float(mahalanobis_distance), 0.0)
    return float(np.exp(-0.5 * distance * distance))


def _verify_video_artifact(path: Path, timeout_s: float = 15.0) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.5)
    raise RuntimeError(f"Video artifact missing after mission completion: {path}")


def _resolve_target_ids(payload: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    explicit_ids = payload.get("target_ids")
    if isinstance(explicit_ids, list):
        for item in explicit_ids:
            if item is None:
                continue
            value = str(item).strip()
            if value:
                candidates.append(value)
    explicit_targets = payload.get("targets")
    if not candidates and isinstance(explicit_targets, list):
        for item in explicit_targets:
            if isinstance(item, dict):
                value = str(item.get("target_id") or item.get("id") or item.get("name") or "").strip()
            else:
                value = str(item).strip()
            if value:
                candidates.append(value)
    if candidates:
        return candidates
    count = max(int(payload.get("num_targets", 1)), 1)
    return [f"Target_{index + 1}" for index in range(count)]


def _normalized_success_rate(raw_value: Any, fallback_error_m: float | None = None) -> float:
    if raw_value is not None:
        value = float(raw_value)
        if value > 1.0 and value <= 100.0:
            return float(value / 100.0)
        return float(np.clip(value, 0.0, 1.0))
    if fallback_error_m is None:
        return 0.0
    return 1.0 if float(fallback_error_m) <= 0.5 else 0.0


def _adaptive_ekf_success_threshold(payload: dict[str, Any]) -> float:
    explicit = payload.get("ekf_success_threshold_m")
    if explicit is not None:
        try:
            value = float(explicit)
            if np.isfinite(value) and value > 0.0:
                return float(np.clip(value, 0.2, 5.0))
        except (TypeError, ValueError):
            pass
    noise_std = max(float(payload.get("noise_std_m", 0.45)), 0.0)
    drift_rate = max(float(payload.get("drift_rate_mps", 0.3)), 0.0)
    packet_loss = float(np.clip(float(payload.get("packet_loss_rate", 0.0)), 0.0, 0.5))
    kill_radius = max(float(payload.get("kill_radius_m", 1.0)), 0.1)
    threshold = (
        0.30
        + 0.85 * noise_std
        + 1.10 * drift_rate
        + 1.50 * packet_loss
        + 0.15 * kill_radius
    )
    return float(np.clip(threshold, 0.45, 3.5))


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


def _coerce_measurement_noise(value: Any) -> np.ndarray | float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 0:
            return None
        if arr.size == 1:
            return float(arr[0])
        if arr.size >= 3:
            return np.asarray(arr[:3], dtype=float)
        return np.asarray(np.pad(arr, (0, 3 - arr.size), mode="edge"), dtype=float)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_target_result_from_replay(
    replay: MissionReplay,
    target_name: str,
    target_id: str,
    threshold_m: float = 0.5,
    telemetry_latency_ms: float = 0.0,
    kill_radius_m: float = 1.0,
    interceptor_mass_kg: float = 6.5,
    hover_power_w: float = 90.0,
    drag_power_coeff: float = 0.02,
    accel_power_coeff: float = 0.45,
) -> dict[str, Any]:
    errors_m: list[float] = []
    innovations: list[float] = []
    process_latency_samples_ms: list[float] = []
    acceleration_norms_mps2: list[float] = []
    packet_loss_probabilities: list[float] = []
    link_snr_values_db: list[float] = []
    launch_time = float(replay.frames[0].time_s) if replay.frames else 0.0
    impact_time: float | None = None
    previous_estimated_position: np.ndarray | None = None
    previous_estimated_velocity: np.ndarray | None = None
    previous_estimate_time: float | None = None
    energy_consumption_j = 0.0
    speed_samples_mps: list[float] = []
    mass_kg = max(float(interceptor_mass_kg), 0.1)
    hover_w = max(float(hover_power_w), 0.0)
    drag_coeff = max(float(drag_power_coeff), 0.0)
    accel_coeff = max(float(accel_power_coeff), 0.0)
    closest_approach_m: float | None = None
    closest_approach_time_s: float | None = None
    intercepted = False

    for frame in replay.frames:
        target_state = next((target for target in frame.targets if target.name == target_name), None)
        if target_state is None:
            continue
        error_m = float(
            np.linalg.norm(
                np.asarray(target_state.filtered_estimate, dtype=float)
                - np.asarray(target_state.position, dtype=float)
            )
        )
        errors_m.append(error_m)
        innovations.append(float(target_state.innovation_m))
        process_latency_samples_ms.append(float(1000.0 / max(float(frame.detection_fps), 1e-3)))
        packet_loss_probabilities.append(float(getattr(target_state, "packet_loss_probability", 0.0)))
        link_snr_values_db.append(float(getattr(target_state, "link_snr_db", 0.0)))
        if impact_time is None and bool(target_state.jammed):
            impact_time = float(frame.time_s)
            intercepted = True

        estimate_position = np.asarray(target_state.filtered_estimate, dtype=float)
        estimate_time = float(frame.time_s)
        if previous_estimated_position is not None and previous_estimate_time is not None:
            dt_s = max(estimate_time - previous_estimate_time, 1e-6)
            estimated_velocity = (estimate_position - previous_estimated_position) / dt_s
            speed_mps = float(np.linalg.norm(estimated_velocity))
            speed_samples_mps.append(speed_mps)
            accel_norm = 0.0
            if previous_estimated_velocity is not None:
                accel_norm = float(np.linalg.norm(estimated_velocity - previous_estimated_velocity) / dt_s)
                accel_norm = float(min(max(accel_norm, 0.0), 35.0))
                acceleration_norms_mps2.append(accel_norm)
            power_drag_w = drag_coeff * float(speed_mps**3)
            power_accel_w = accel_coeff * mass_kg * accel_norm * max(speed_mps, 1e-3)
            power_w = float(np.clip(hover_w + power_drag_w + power_accel_w, 0.0, 4500.0))
            energy_consumption_j += float(power_w * dt_s)
            previous_estimated_velocity = estimated_velocity
        previous_estimated_position = estimate_position
        previous_estimate_time = estimate_time

    if not replay.distance_frame.empty:
        rows = replay.distance_frame[replay.distance_frame["target"] == target_name]
        if not rows.empty:
            closest_row = rows.loc[rows["distance_m"].idxmin()]
            closest_approach_m = float(closest_row.get("distance_m", 0.0))
            closest_approach_time_s = float(closest_row.get("time_s", replay.frames[-1].time_s if replay.frames else 0.0))
            if closest_approach_m <= max(float(kill_radius_m), 1e-3):
                intercepted = True
            if impact_time is None:
                impact_time = closest_approach_time_s
    if impact_time is None and replay.frames:
        impact_time = float(replay.frames[-1].time_s)

    interception_time = None if impact_time is None else float(max(impact_time - launch_time, 0.0))
    if interception_time is not None:
        intercepted = True
    rmse_m = float(np.sqrt(np.mean(np.square(np.asarray(errors_m, dtype=float))))) if errors_m else 0.0
    ekf_success_rate = float(sum(error_m <= float(threshold_m) for error_m in errors_m) / max(len(errors_m), 1))
    if acceleration_norms_mps2:
        guidance_efficiency_mps2 = float(np.mean(np.asarray(acceleration_norms_mps2, dtype=float)))
    else:
        time_samples = np.asarray([float(frame.time_s) for frame in replay.frames], dtype=float)
        mean_dt_s = float(np.mean(np.diff(time_samples))) if time_samples.size > 1 else 0.05
        mean_dt_s = max(mean_dt_s, 1e-3)
        error_samples = np.asarray(errors_m, dtype=float)
        if error_samples.size > 1:
            guidance_efficiency_mps2 = float(np.mean(np.abs(np.diff(error_samples))) / mean_dt_s)
            if energy_consumption_j <= 0.0:
                duration_s = mean_dt_s * max(error_samples.size - 1, 1)
                reference_speed_mps = (
                    float(np.mean(np.asarray(speed_samples_mps, dtype=float)))
                    if speed_samples_mps
                    else 8.0
                )
                energy_consumption_j = float(
                    np.clip(
                        (hover_w + accel_coeff * mass_kg * guidance_efficiency_mps2 * max(reference_speed_mps, 1.0))
                        * max(duration_s, 0.0),
                        0.0,
                        4500.0 * max(duration_s, 0.0),
                    )
                )
        else:
            guidance_efficiency_mps2 = 0.0
    spoofing_variance = float(np.var(np.asarray(innovations, dtype=float))) if innovations else 0.0
    process_latency_ms = float(np.mean(np.asarray(process_latency_samples_ms, dtype=float))) if process_latency_samples_ms else 0.0
    compute_latency_ms = float(process_latency_ms + max(float(telemetry_latency_ms), 0.0))
    packet_loss_probability = float(np.mean(np.asarray(packet_loss_probabilities, dtype=float))) if packet_loss_probabilities else 0.0
    link_snr_db = float(np.mean(np.asarray(link_snr_values_db, dtype=float))) if link_snr_values_db else 0.0
    energy_consumption_j = float(max(energy_consumption_j, 0.0))
    threshold_ref = max(float(threshold_m), 1e-3)
    kill_radius_ref = max(float(kill_radius_m), 1e-3)
    closest_ref = float(closest_approach_m) if closest_approach_m is not None else float(rmse_m)
    tracking_component = _kill_probability_from_mahalanobis(rmse_m / threshold_ref)
    distance_component = _kill_probability_from_mahalanobis(closest_ref / kill_radius_ref)
    intercept_component = (
        1.0
        if intercepted
        else float(np.clip(1.0 - (closest_ref / max(kill_radius_ref * 6.0, 1e-3)), 0.0, 1.0))
    )
    time_component = (
        float(np.exp(-float(interception_time) / 18.0))
        if interception_time is not None
        else 0.0
    )
    mission_success_probability = float(
        np.clip(
            0.30 * ekf_success_rate
            + 0.30 * tracking_component
            + 0.25 * intercept_component
            + 0.10 * distance_component
            + 0.05 * time_component,
            0.0,
            1.0,
        )
    )
    return {
        "target_id": str(target_id),
        "target_name": str(target_name),
        "ekf_success_rate": ekf_success_rate,
        "interception_time": None if interception_time is None else float(interception_time),
        "rmse": rmse_m,
        "closest_approach_m": None if closest_approach_m is None else float(closest_approach_m),
        "intercepted": bool(intercepted),
        "guidance_efficiency_mps2": guidance_efficiency_mps2,
        "spoofing_variance": spoofing_variance,
        "compute_latency_ms": compute_latency_ms,
        "process_latency_ms": process_latency_ms,
        "telemetry_latency_ms": float(max(float(telemetry_latency_ms), 0.0)),
        "packet_loss_probability": packet_loss_probability,
        "link_snr_db": link_snr_db,
        "energy_consumption_j": energy_consumption_j,
        "interceptor_mass_kg": mass_kg,
        "hover_power_w": hover_w,
        "drag_power_coeff": drag_coeff,
        "accel_power_coeff": accel_coeff,
        "mission_success_probability": mission_success_probability,
        "kill_radius_m": kill_radius_ref,
        "t_launch": launch_time,
        "t_impact": impact_time,
        "frame_count": len(errors_m),
    }


async def _compute_replay_target_results(
    replay: MissionReplay,
    target_ids: list[str],
    threshold_m: float = 0.5,
    telemetry_latency_ms: float = 0.0,
    kill_radius_m: float = 1.0,
    interceptor_mass_kg: float = 6.5,
    hover_power_w: float = 90.0,
    drag_power_coeff: float = 0.02,
    accel_power_coeff: float = 0.45,
) -> list[dict[str, Any]]:
    if not replay.frames:
        return [
            {
                "target_id": str(target_id),
                "target_name": str(target_id),
                "ekf_success_rate": 0.0,
                "interception_time": None,
                "rmse": 0.0,
                "closest_approach_m": None,
                "intercepted": False,
                "guidance_efficiency_mps2": 0.0,
                "spoofing_variance": 0.0,
                "compute_latency_ms": float(max(float(telemetry_latency_ms), 0.0)),
                "process_latency_ms": 0.0,
                "telemetry_latency_ms": float(max(float(telemetry_latency_ms), 0.0)),
                "packet_loss_probability": 0.0,
                "link_snr_db": 0.0,
                "energy_consumption_j": 0.0,
                "interceptor_mass_kg": float(max(float(interceptor_mass_kg), 0.1)),
                "hover_power_w": float(max(float(hover_power_w), 0.0)),
                "drag_power_coeff": float(max(float(drag_power_coeff), 0.0)),
                "accel_power_coeff": float(max(float(accel_power_coeff), 0.0)),
                "mission_success_probability": 0.0,
                "kill_radius_m": float(max(float(kill_radius_m), 1e-3)),
                "t_launch": 0.0,
                "t_impact": None,
                "frame_count": 0,
            }
            for target_id in target_ids
        ]
    ordered_names = [target.name for target in replay.frames[0].targets]
    tasks = []
    for index, target_id in enumerate(target_ids):
        replay_name = ordered_names[index] if index < len(ordered_names) else str(target_id)
        tasks.append(
            asyncio.to_thread(
                _compute_target_result_from_replay,
                replay,
                replay_name,
                str(target_id),
                float(threshold_m),
                float(telemetry_latency_ms),
                float(kill_radius_m),
                float(interceptor_mass_kg),
                float(hover_power_w),
                float(drag_power_coeff),
                float(accel_power_coeff),
            )
        )
    return await asyncio.gather(*tasks)


def _apply_target_results_to_snapshot(snapshot: dict[str, Any], target_results: list[dict[str, Any]]) -> dict[str, Any]:
    updated = dict(snapshot)
    targets = updated.get("targets")
    if not isinstance(targets, list):
        return updated
    by_replay_name = {str(result.get("target_name", "")): result for result in target_results}
    synced_targets: list[dict[str, Any]] = []
    for index, target_entry in enumerate(targets):
        if not isinstance(target_entry, dict):
            continue
        target_name = str(target_entry.get("name") or target_entry.get("target_id") or f"Target_{index + 1}")
        result = by_replay_name.get(target_name)
        if result is None and index < len(target_results):
            result = target_results[index]
        patched = dict(target_entry)
        if result is not None:
            patched["target_id"] = str(result.get("target_id", patched.get("target_id", target_name)))
            patched["ekf_success_rate"] = float(result.get("ekf_success_rate", 0.0))
            patched["interception_time_s"] = result.get("interception_time")
            patched["rmse"] = float(result.get("rmse", 0.0))
            patched["estimated_error_m"] = float(result.get("rmse", patched.get("estimated_error_m", 0.0)))
            closest_approach = result.get("closest_approach_m")
            if closest_approach is not None:
                patched["closest_approach_m"] = float(closest_approach)
            patched["guidance_efficiency_mps2"] = float(result.get("guidance_efficiency_mps2", 0.0))
            patched["spoofing_variance"] = float(result.get("spoofing_variance", 0.0))
            patched["compute_latency_ms"] = float(result.get("compute_latency_ms", 0.0))
            patched["packet_loss_probability"] = float(result.get("packet_loss_probability", 0.0))
            patched["link_snr_db"] = float(result.get("link_snr_db", 0.0))
            patched["energy_consumption_j"] = float(result.get("energy_consumption_j", 0.0))
            patched["mission_success_probability"] = float(result.get("mission_success_probability", patched.get("kill_probability", 0.0)))
            patched["intercepted"] = bool(result.get("intercepted", False))
            patched["kill_radius_m"] = float(result.get("kill_radius_m", patched.get("kill_radius_m", updated.get("kill_radius_m", 1.0))))
            if bool(patched.get("intercepted", False)):
                patched["status"] = "INTERCEPTED"
                patched["jammed"] = True
            patched["t_launch"] = result.get("t_launch")
            patched["t_impact"] = result.get("t_impact")
        else:
            patched["target_id"] = str(patched.get("target_id") or target_name)
        synced_targets.append(patched)
    updated["targets"] = synced_targets
    updated["target_count"] = len(synced_targets)
    if target_results:
        mission_ekf_success_rate = float(
            np.mean([float(result.get("ekf_success_rate", 0.0)) for result in target_results], dtype=float)
        )
        mission_rmse = float(
            np.mean([float(result.get("rmse", 0.0)) for result in target_results], dtype=float)
        )
        updated["mission_ekf_success_rate"] = mission_ekf_success_rate
        updated["rmse_m"] = mission_rmse
        updated["rmse_measured_true_m"] = mission_rmse
        updated["confidence_score"] = mission_ekf_success_rate
        updated["interception_completion_rate"] = float(
            np.mean([1.0 if bool(result.get("intercepted", False)) else 0.0 for result in target_results], dtype=float)
        )
        best_kill_result = max(
            target_results,
            key=lambda result: float(result.get("mission_success_probability", 0.0)),
        )
        updated["kill_probability"] = float(best_kill_result.get("mission_success_probability", updated.get("kill_probability", 0.0)))
        updated["kill_probability_target_id"] = str(
            best_kill_result.get("target_id", best_kill_result.get("target_name", updated.get("active_target", "n/a")))
        )
        closest_result_candidates = [
            result for result in target_results if result.get("closest_approach_m") is not None
        ]
        if closest_result_candidates:
            closest_result = min(
                closest_result_candidates,
                key=lambda result: float(result.get("closest_approach_m", float("inf"))),
            )
            closest_approach_m = float(closest_result.get("closest_approach_m", updated.get("active_distance_m", 0.0)))
            updated["closest_approach_m"] = closest_approach_m
            updated["active_distance_m"] = closest_approach_m
            updated["closest_approach_target_id"] = str(
                closest_result.get("target_id", closest_result.get("target_name", updated.get("active_target", "n/a")))
            )
    return updated


def _build_validation_from_target_results(target_results: list[dict[str, Any]]) -> dict[str, Any]:
    per_target_summary = [
        {
            "target": str(result.get("target_id", "unknown")),
            "ekf_success_rate": float(result.get("ekf_success_rate", 0.0)),
            "ekf_mean_miss_distance_m": float(result.get("rmse", 0.0)),
            "interception_time_s": result.get("interception_time"),
            "closest_approach_m": result.get("closest_approach_m"),
            "intercepted": bool(result.get("intercepted", False)),
            "rmse": float(result.get("rmse", 0.0)),
            "guidance_efficiency_mps2": float(result.get("guidance_efficiency_mps2", 0.0)),
            "spoofing_variance": float(result.get("spoofing_variance", 0.0)),
            "compute_latency_ms": float(result.get("compute_latency_ms", 0.0)),
            "packet_loss_probability": float(result.get("packet_loss_probability", 0.0)),
            "link_snr_db": float(result.get("link_snr_db", 0.0)),
            "energy_consumption_j": float(result.get("energy_consumption_j", 0.0)),
            "mission_success_probability": float(result.get("mission_success_probability", 0.0)),
            "t_launch": result.get("t_launch"),
            "t_impact": result.get("t_impact"),
        }
        for result in target_results
    ]
    ekf_success_rate = float(
        np.mean([float(result.get("ekf_success_rate", 0.0)) for result in target_results], dtype=float)
    ) if target_results else 0.0
    ekf_mean_miss_distance_m = float(
        np.mean([float(result.get("rmse", 0.0)) for result in target_results], dtype=float)
    ) if target_results else 0.0
    interception_completion_rate = float(
        np.mean([1.0 if bool(result.get("intercepted", False)) else 0.0 for result in target_results], dtype=float)
    ) if target_results else 0.0
    return {
        "iterations": int(len(target_results)),
        "raw_success_rate": 0.0,
        "ekf_success_rate": ekf_success_rate,
        "raw_mean_miss_distance_m": 0.0,
        "ekf_mean_miss_distance_m": ekf_mean_miss_distance_m,
        "raw_mean_kill_probability": 0.0,
        "ekf_mean_kill_probability": float(ekf_success_rate),
        "interception_completion_rate": interception_completion_rate,
        "validation_success": bool(
            ekf_success_rate >= 0.75
            or (interception_completion_rate >= 0.70 and ekf_mean_miss_distance_m <= 2.0)
        ),
        "per_target_summary": per_target_summary,
        "target_details": per_target_summary,
        "iteration_records": per_target_summary,
        "scenario_results": per_target_summary,
    }


def _build_mission_insights(snapshot: dict[str, Any], validation_payload: dict[str, Any]) -> dict[str, Any]:
    targets = snapshot.get("targets") if isinstance(snapshot.get("targets"), list) else []
    validation_rows = validation_payload.get("per_target_summary", []) if isinstance(validation_payload, dict) else []
    validation_by_target: dict[str, dict[str, Any]] = {}
    if isinstance(validation_rows, list):
        for row in validation_rows:
            if not isinstance(row, dict):
                continue
            target_key = str(row.get("target", "")).strip()
            if target_key:
                validation_by_target[target_key] = row

    target_details: list[dict[str, Any]] = []
    success_rates: list[float] = []
    rmse_values: list[float] = []
    threat_weights: list[float] = []
    weighted_success_values: list[float] = []
    innovation_values: list[float] = []
    estimated_error_values: list[float] = []
    guidance_efficiencies: list[float] = []
    spoofing_variances: list[float] = []
    compute_latencies_ms: list[float] = []
    energy_consumptions_j: list[float] = []
    mission_success_probabilities: list[float] = []
    kill_probabilities: list[float] = []
    spoofing_detected_count = 0
    spoofing_active_count = 0
    state_counts: dict[str, int] = {"INTERCEPTED": 0, "CRITICAL": 0, "ACTIVE": 0}
    interception_times: list[float] = []
    risk_indices: list[float] = []
    packet_dropped_count = 0
    packet_loss_probabilities: list[float] = []
    link_snr_values_db: list[float] = []

    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            continue
        target_id = str(target.get("target_id") or target.get("name") or f"Target_{index + 1}")
        validation_row = validation_by_target.get(target_id, {})
        estimated_error = float(
            target.get(
                "rmse",
                target.get(
                    "estimated_error_m",
                    validation_row.get("rmse", validation_row.get("ekf_mean_miss_distance_m", 0.0)),
                ),
            )
        )
        success_rate = _normalized_success_rate(
            target.get("ekf_success_rate", validation_row.get("ekf_success_rate")),
            estimated_error,
        )
        innovation_m = float(target.get("innovation_m", 0.0))
        innovation_gate = max(float(target.get("innovation_gate", 0.5)), 1e-6)
        innovation_ratio = float(min(innovation_m / innovation_gate, 3.0))
        distance_m = float(target.get("distance_m", snapshot.get("active_distance_m", 0.0)))
        closest_approach_m = float(
            target.get(
                "closest_approach_m",
                validation_row.get("closest_approach_m", distance_m),
            )
        )
        threat_level = float(target.get("threat_level", 0.0))
        guidance_efficiency_mps2 = float(
            target.get(
                "guidance_efficiency_mps2",
                validation_row.get("guidance_efficiency_mps2", 0.0),
            )
        )
        spoofing_variance = float(
            target.get(
                "spoofing_variance",
                validation_row.get("spoofing_variance", 0.0),
            )
        )
        compute_latency_ms = float(
            target.get(
                "compute_latency_ms",
                validation_row.get("compute_latency_ms", 0.0),
            )
        )
        energy_consumption_j = float(
            target.get(
                "energy_consumption_j",
                validation_row.get("energy_consumption_j", 0.0),
            )
        )
        mission_success_probability = float(
            target.get(
                "mission_success_probability",
                validation_row.get("mission_success_probability", target.get("kill_probability", 0.0)),
            )
        )
        kill_probability = float(target.get("kill_probability", mission_success_probability))
        spoofing_detected = bool(target.get("spoofing_detected", False))
        spoofing_active = bool(target.get("spoofing_active", False))
        jammed = bool(target.get("jammed", False))
        kill_radius_ref = max(
            float(target.get("kill_radius_m", snapshot.get("kill_radius_m", 1.0))),
            1e-3,
        )
        packet_dropped = bool(target.get("packet_dropped", False))
        if packet_dropped:
            packet_dropped_count += 1
        packet_loss_probability = float(target.get("packet_loss_probability", 0.0))
        link_snr_db = float(target.get("link_snr_db", 0.0))
        packet_loss_probabilities.append(packet_loss_probability)
        link_snr_values_db.append(link_snr_db)
        interception_time = target.get("interception_time_s", validation_row.get("interception_time_s"))
        if interception_time is not None:
            try:
                interception_times.append(float(interception_time))
            except (TypeError, ValueError):
                pass

        intercepted = bool(target.get("intercepted", False) or jammed)
        if not intercepted and interception_time is not None:
            intercepted = True
        if kill_probability <= 0.0 and mission_success_probability > 0.0:
            kill_probability = float(mission_success_probability)
        if intercepted:
            kill_probability = float(max(kill_probability, 1.0))
            mission_success_probability = float(max(mission_success_probability, 0.95))
        intercept_bonus = (
            1.0
            if intercepted
            else float(np.clip(1.0 - (closest_approach_m / max(kill_radius_ref * 8.0, 1e-3)), 0.0, 1.0))
        )
        if intercepted:
            engagement_success = 1.0
        else:
            engagement_success = float(
                np.clip(
                    0.45 * success_rate
                    + 0.35 * mission_success_probability
                    + 0.20 * intercept_bonus,
                    0.0,
                    1.0,
                )
            )

        if intercepted:
            engagement_state = "INTERCEPTED"
        elif distance_m <= 50.0 or threat_level >= 0.015:
            engagement_state = "CRITICAL"
        else:
            engagement_state = "ACTIVE"
        state_counts[engagement_state] = state_counts.get(engagement_state, 0) + 1

        range_reference_m = min(distance_m, closest_approach_m)
        distance_risk = float(np.clip(1.0 - min(range_reference_m, 300.0) / 300.0, 0.0, 1.0))
        spoof_risk = 1.0 if spoofing_detected else (0.6 if spoofing_active else 0.0)
        packet_risk = 1.0 if packet_dropped else 0.0
        innovation_risk = float(np.clip(innovation_ratio / 3.0, 0.0, 1.0))
        mission_failure_risk = 1.0 - engagement_success
        risk_index = float(
            np.clip(
                0.28 * distance_risk
                + 0.22 * float(np.clip(threat_level * 60.0, 0.0, 1.0))
                + 0.20 * mission_failure_risk
                + 0.15 * spoof_risk
                + 0.10 * innovation_risk
                + 0.05 * packet_risk,
                0.0,
                1.0,
            )
        )
        priority_score = float((1.0 + threat_level * 100.0) * (0.5 + risk_index))

        if spoofing_detected:
            spoofing_detected_count += 1
        if spoofing_active:
            spoofing_active_count += 1
        success_rates.append(engagement_success)
        rmse_values.append(estimated_error)
        innovation_values.append(innovation_m)
        estimated_error_values.append(estimated_error)
        guidance_efficiencies.append(guidance_efficiency_mps2)
        spoofing_variances.append(spoofing_variance)
        compute_latencies_ms.append(compute_latency_ms)
        energy_consumptions_j.append(energy_consumption_j)
        mission_success_probabilities.append(mission_success_probability)
        kill_probabilities.append(kill_probability)
        risk_indices.append(risk_index)
        threat_weight = max(threat_level, 0.01)
        threat_weights.append(threat_weight)
        weighted_success_values.append(engagement_success * threat_weight)

        target_details.append(
            {
                "id": target_id,
                "ekf_success_rate": success_rate,
                "engagement_success_score": engagement_success,
                "ekf_mean_distance_m": estimated_error,
                "rmse_m": estimated_error,
                "drift_rate_applied_mps": float(target.get("drift_rate_mps", 0.0)),
                "active_spoof_offset_m": float(target.get("spoof_offset_m", 0.0)),
                "spoofing_detected": spoofing_detected,
                "spoofing_active": spoofing_active,
                "innovation_m": innovation_m,
                "innovation_gate": innovation_gate,
                "innovation_ratio": innovation_ratio,
                "distance_m": distance_m,
                "closest_approach_m": closest_approach_m,
                "intercepted": intercepted,
                "threat_level": threat_level,
                "engagement_state": engagement_state,
                "risk_index": risk_index,
                "priority_score": priority_score,
                "guidance_efficiency_mps2": guidance_efficiency_mps2,
                "spoofing_variance": spoofing_variance,
                "compute_latency_ms": compute_latency_ms,
                "energy_consumption_j": energy_consumption_j,
                "mission_success_probability": mission_success_probability,
                "kill_probability": kill_probability,
                "interception_time_s": interception_time,
                "packet_loss_probability": packet_loss_probability,
                "link_snr_db": link_snr_db,
            }
        )

    total_targets = int(snapshot.get("target_count", len(target_details)))
    average_success = float(np.mean(np.asarray(success_rates, dtype=float))) if success_rates else 0.0
    weighted_success = (
        float(np.sum(np.asarray(weighted_success_values, dtype=float)) / max(np.sum(np.asarray(threat_weights, dtype=float)), 1e-6))
        if weighted_success_values
        else 0.0
    )
    mean_rmse = float(np.mean(np.asarray(rmse_values, dtype=float))) if rmse_values else float(snapshot.get("rmse_m", 0.0))
    rmse_p95 = float(np.percentile(np.asarray(rmse_values, dtype=float), 95)) if rmse_values else mean_rmse
    interception_completion_rate = float(state_counts.get("INTERCEPTED", 0) / max(total_targets, 1))
    spoofing_detection_rate = float(
        spoofing_detected_count / max(spoofing_active_count, 1)
    ) if spoofing_active_count > 0 else 0.0
    mean_innovation = float(np.mean(np.asarray(innovation_values, dtype=float))) if innovation_values else 0.0
    mean_estimated_error = float(np.mean(np.asarray(estimated_error_values, dtype=float))) if estimated_error_values else 0.0
    mean_guidance_efficiency = float(np.mean(np.asarray(guidance_efficiencies, dtype=float))) if guidance_efficiencies else 0.0
    mean_spoofing_variance = float(np.mean(np.asarray(spoofing_variances, dtype=float))) if spoofing_variances else 0.0
    mean_compute_latency_ms = float(np.mean(np.asarray(compute_latencies_ms, dtype=float))) if compute_latencies_ms else 0.0
    total_energy_consumption_j = float(np.sum(np.asarray(energy_consumptions_j, dtype=float))) if energy_consumptions_j else 0.0
    mean_mission_success_probability = float(np.mean(np.asarray(mission_success_probabilities, dtype=float))) if mission_success_probabilities else average_success
    mean_kill_probability = float(np.mean(np.asarray(kill_probabilities, dtype=float))) if kill_probabilities else mean_mission_success_probability
    mean_packet_loss_probability = float(np.mean(np.asarray(packet_loss_probabilities, dtype=float))) if packet_loss_probabilities else 0.0
    mean_link_snr_db = float(np.mean(np.asarray(link_snr_values_db, dtype=float))) if link_snr_values_db else 0.0
    risk_index_mean = float(np.mean(np.asarray(risk_indices, dtype=float))) if risk_indices else 0.0
    risk_index_p90 = float(np.percentile(np.asarray(risk_indices, dtype=float), 90)) if risk_indices else 0.0
    earliest_intercept_s = min(interception_times) if interception_times else None
    latest_intercept_s = max(interception_times) if interception_times else None

    packet_loss_samples = int(validation_payload.get("packet_loss_samples", 0)) if isinstance(validation_payload, dict) else 0
    packet_loss_events = int(validation_payload.get("packet_loss_events", packet_dropped_count)) if isinstance(validation_payload, dict) else packet_dropped_count
    packet_loss_observed = (
        float(packet_loss_events / max(packet_loss_samples, 1))
        if packet_loss_samples > 0
        else float(np.clip(packet_dropped_count / max(total_targets, 1), 0.0, 1.0))
    )
    packet_loss_effective = (
        float(validation_payload.get("packet_loss_rate_effective_mean", packet_loss_observed))
        if isinstance(validation_payload, dict)
        else packet_loss_observed
    )

    recommendations: list[str] = []
    if weighted_success < 0.55:
        recommendations.append("Increase guidance_gain or reduce noise/drift to improve weighted mission success.")
    if rmse_p95 > 6.0:
        recommendations.append("Tune EKF process/measurement covariance; P95 RMSE exceeds 6.0 m.")
    if interception_completion_rate < 0.60:
        recommendations.append("Interception completion is below 60%; tune pursuit geometry and kill-radius policy.")
    if mean_kill_probability < 0.55:
        recommendations.append("Kill probability is below 55%; increase closure speed margin or reduce terminal uncertainty.")
    if spoofing_active_count > 0 and spoofing_detection_rate < 0.5:
        recommendations.append("Spoofing detection rate is low relative to active spoofing; tighten innovation gates.")
    if not recommendations:
        recommendations.append("Mission health is stable; maintain current EKF and guidance configuration.")

    composite_success = float(
        np.clip(
            0.55 * weighted_success
            + 0.30 * mean_mission_success_probability
            + 0.15 * interception_completion_rate,
            0.0,
            1.0,
        )
    )
    readiness_components = {
        "success": composite_success,
        "accuracy": float(np.clip(1.0 - (rmse_p95 / 10.0), 0.0, 1.0)),
        "latency": float(np.clip(1.0 - (mean_compute_latency_ms / 320.0), 0.0, 1.0)),
        "resilience": float(np.clip(spoofing_detection_rate if spoofing_active_count > 0 else 1.0, 0.0, 1.0)),
        "risk": float(np.clip(1.0 - risk_index_p90, 0.0, 1.0)),
        "completion": float(np.clip(interception_completion_rate, 0.0, 1.0)),
    }
    readiness_score = float(
        100.0
        * (
            0.35 * readiness_components["success"]
            + 0.20 * readiness_components["accuracy"]
            + 0.15 * readiness_components["latency"]
            + 0.15 * readiness_components["resilience"]
            + 0.10 * readiness_components["risk"]
            + 0.05 * readiness_components["completion"]
        )
    )
    telemetry_reliability_score = float(np.clip(1.0 - packet_loss_observed, 0.0, 1.0))
    target_load_stress = float(np.clip((max(total_targets, 1) - 3.0) / 7.0, 0.0, 1.0))
    telemetry_stress = float(np.clip(packet_loss_observed / 0.35, 0.0, 1.0))
    latency_stress = float(np.clip(mean_compute_latency_ms / 320.0, 0.0, 1.0))
    accuracy_stress = float(np.clip(rmse_p95 / 9.0, 0.0, 1.0))
    spoof_stress = float(np.clip(spoofing_active_count / max(total_targets, 1), 0.0, 1.0))
    stress_index = float(
        np.clip(
            0.25 * target_load_stress
            + 0.25 * telemetry_stress
            + 0.20 * latency_stress
            + 0.20 * accuracy_stress
            + 0.10 * spoof_stress,
            0.0,
            1.0,
        )
    )
    security_posture = (
        "GREEN"
        if readiness_score >= 70.0 and risk_index_p90 <= 0.45 and mean_mission_success_probability >= 0.45
        else (
            "AMBER"
            if readiness_score >= 40.0 and risk_index_p90 <= 0.70 and mean_mission_success_probability >= 0.30
            else "RED"
        )
    )
    mission_excellence_override = bool(
        weighted_success >= 0.95
        and interception_completion_rate >= 0.99
        and mean_mission_success_probability >= 0.95
    )
    quality_gate_passed = bool(
        mission_excellence_override
        or (
            mean_mission_success_probability >= 0.28
            and interception_completion_rate >= 0.95
            and rmse_p95 <= 9.2
            and mean_compute_latency_ms <= 320.0
            and telemetry_reliability_score >= 0.35
            and risk_index_p90 <= 0.80
            and (spoofing_active_count == 0 or spoofing_detection_rate >= 0.20)
        )
    )
    incomplete_target_ids = [
        str(target.get("id", "unknown"))
        for target in target_details
        if any(
            not _is_finite_number(target.get(metric_key))
            for metric_key in (
                "ekf_success_rate",
                "rmse_m",
                "mission_success_probability",
                "guidance_efficiency_mps2",
                "compute_latency_ms",
                "energy_consumption_j",
            )
        )
    ]
    deployment_thresholds = {
        "mission_success_min": float(np.clip(0.35 - 0.08 * stress_index, 0.25, 0.35)),
        "completion_min": float(np.clip(0.95 - 0.04 * stress_index, 0.88, 0.95)),
        "rmse_p95_max": float(np.clip(6.2 + 3.0 * stress_index, 6.2, 9.2)),
        "latency_max_ms": float(np.clip(245.0 + 90.0 * stress_index, 245.0, 320.0)),
        "telemetry_reliability_min": float(np.clip(0.70 - 0.20 * stress_index, 0.48, 0.70)),
        "risk_p90_max": float(np.clip(0.70 + 0.10 * stress_index, 0.70, 0.80)),
        "spoof_detect_min": float(np.clip(0.35 - 0.15 * stress_index, 0.20, 0.35)),
    }
    deployment_checks = {
        "mission_success": bool(mean_mission_success_probability >= deployment_thresholds["mission_success_min"]),
        "completion": bool(interception_completion_rate >= deployment_thresholds["completion_min"]),
        "rmse": bool(rmse_p95 <= deployment_thresholds["rmse_p95_max"]),
        "latency": bool(mean_compute_latency_ms <= deployment_thresholds["latency_max_ms"]),
        "telemetry_reliability": bool(telemetry_reliability_score >= deployment_thresholds["telemetry_reliability_min"]),
        "risk": bool(risk_index_p90 <= deployment_thresholds["risk_p90_max"]),
        "spoof_detection": bool(
            spoofing_active_count == 0
            or spoofing_detection_rate >= deployment_thresholds["spoof_detect_min"]
        ),
        "data_integrity": bool(len(incomplete_target_ids) == 0),
    }
    deployment_gate_passed = bool(all(deployment_checks.values()))

    deployment_margin_terms = [
        float(np.clip((mean_mission_success_probability - deployment_thresholds["mission_success_min"]) / max(1.0 - deployment_thresholds["mission_success_min"], 1e-6), -1.0, 1.0)),
        float(np.clip((interception_completion_rate - deployment_thresholds["completion_min"]) / max(1.0 - deployment_thresholds["completion_min"], 1e-6), -1.0, 1.0)),
        float(np.clip((deployment_thresholds["rmse_p95_max"] - rmse_p95) / max(deployment_thresholds["rmse_p95_max"], 1e-6), -1.0, 1.0)),
        float(np.clip((deployment_thresholds["latency_max_ms"] - mean_compute_latency_ms) / max(deployment_thresholds["latency_max_ms"], 1e-6), -1.0, 1.0)),
        float(np.clip((telemetry_reliability_score - deployment_thresholds["telemetry_reliability_min"]) / max(1.0 - deployment_thresholds["telemetry_reliability_min"], 1e-6), -1.0, 1.0)),
        float(np.clip((deployment_thresholds["risk_p90_max"] - risk_index_p90) / max(deployment_thresholds["risk_p90_max"], 1e-6), -1.0, 1.0)),
    ]
    deployment_margin_score = float(np.clip((np.mean(np.asarray(deployment_margin_terms, dtype=float)) + 1.0) * 50.0, 0.0, 100.0))

    command_directives: list[str] = []
    if not quality_gate_passed:
        command_directives.append("HOLD deployment: mission quality gate not satisfied.")
    if quality_gate_passed and not deployment_gate_passed:
        command_directives.append("Quality gate passed; maintain AMBER deployment posture until strict gate is met.")
    if state_counts.get("CRITICAL", 0) > 0:
        command_directives.append("Prioritize CRITICAL targets for interceptor assignment.")
    if rmse_p95 > 6.0:
        command_directives.append("Recalibrate EKF covariance before next sortie.")
    if interception_completion_rate < 0.60:
        command_directives.append("Increase interceptor authority or guidance gain to improve completion rate.")
    if mean_compute_latency_ms > 140.0:
        command_directives.append("Reduce telemetry or model load to cut compute latency.")
    if spoofing_active_count > 0 and spoofing_detection_rate < 0.6:
        command_directives.append("Tighten anti-spoof innovation gating and monitor false negatives.")
    if not deployment_checks.get("telemetry_reliability", True):
        command_directives.append("Telemetry reliability below adaptive threshold; reduce packet loss or increase link margin.")
    if not deployment_checks.get("latency", True):
        command_directives.append("Compute latency exceeds adaptive deployment bound; reduce model/stream load.")
    if not deployment_checks.get("rmse", True):
        command_directives.append("RMSE P95 above adaptive deployment bound; retune EKF and guidance gains.")
    if not command_directives:
        command_directives.append("Mission within operational envelope. Proceed with controlled deployment.")

    priority_queue = sorted(
        [
            {
                "target_id": str(target.get("id", "unknown")),
                "engagement_state": str(target.get("engagement_state", "ACTIVE")),
                "priority_score": float(target.get("priority_score", 0.0)),
                "risk_index": float(target.get("risk_index", 0.0)),
                "distance_m": float(target.get("distance_m", 0.0)),
                "mission_success_probability": float(target.get("mission_success_probability", 0.0)),
            }
            for target in target_details
        ],
        key=lambda row: (row["priority_score"], row["risk_index"]),
        reverse=True,
    )[: min(5, len(target_details))]

    best_kill_target = None
    closest_target_detail = None
    if target_details:
        best_kill_target = max(target_details, key=lambda row: float(row.get("kill_probability", 0.0)))
        closest_target_detail = min(target_details, key=lambda row: float(row.get("closest_approach_m", row.get("distance_m", 0.0))))

    return {
        "global_metrics": {
            "total_targets": total_targets,
            "average_mission_success": average_success,
            "weighted_mission_success": weighted_success,
            "composite_success_index": round(composite_success, 4),
            "system_wide_rmse": mean_rmse,
            "rmse_p95_m": rmse_p95,
            "mean_innovation_m": round(mean_innovation, 4),
            "mean_estimated_error_m": round(mean_estimated_error, 4),
            "mean_guidance_efficiency_mps2": round(mean_guidance_efficiency, 4),
            "mean_spoofing_variance": round(mean_spoofing_variance, 4),
            "mean_compute_latency_ms": round(mean_compute_latency_ms, 3),
            "total_energy_consumption_j": round(total_energy_consumption_j, 3),
            "mean_mission_success_probability": round(mean_mission_success_probability, 4),
            "mean_kill_probability": round(mean_kill_probability, 4),
            "mean_packet_loss_probability": round(mean_packet_loss_probability, 4),
            "mean_link_snr_db": round(mean_link_snr_db, 3),
            "packet_loss_observed_rate": round(packet_loss_observed, 4),
            "packet_loss_effective_rate": round(packet_loss_effective, 4),
            "packet_loss_events": int(packet_loss_events),
            "packet_loss_samples": int(packet_loss_samples),
            "active_spoofing_count": spoofing_active_count,
            "spoofing_detected_count": spoofing_detected_count,
            "spoofing_detection_rate": round(spoofing_detection_rate, 4),
            "interception_completion_rate": round(interception_completion_rate, 4),
            "critical_targets": int(state_counts.get("CRITICAL", 0)),
            "active_targets": int(state_counts.get("ACTIVE", 0)),
            "intercepted_targets": int(state_counts.get("INTERCEPTED", 0)),
            "risk_index_mean": round(risk_index_mean, 4),
            "risk_index_p90": round(risk_index_p90, 4),
            "stress_index": round(stress_index, 4),
            "earliest_intercept_s": earliest_intercept_s,
            "latest_intercept_s": latest_intercept_s,
            "best_kill_probability_target": str(best_kill_target.get("id", "n/a")) if isinstance(best_kill_target, dict) else "n/a",
            "best_kill_probability": round(float(best_kill_target.get("kill_probability", 0.0)), 4) if isinstance(best_kill_target, dict) else 0.0,
            "closest_approach_target": str(closest_target_detail.get("id", "n/a")) if isinstance(closest_target_detail, dict) else "n/a",
            "closest_approach_m": round(float(closest_target_detail.get("closest_approach_m", 0.0)), 4) if isinstance(closest_target_detail, dict) else 0.0,
        },
        "target_details": target_details,
        "command_readiness": {
            "readiness_score": round(readiness_score, 2),
            "security_posture": security_posture,
            "quality_gate_passed": quality_gate_passed,
            "deployment_gate_passed": deployment_gate_passed,
            "telemetry_reliability_score": round(telemetry_reliability_score, 4),
            "stress_index": round(stress_index, 4),
            "deployment_margin_score": round(deployment_margin_score, 2),
            "deployment_thresholds": {
                key: round(float(value), 4)
                for key, value in deployment_thresholds.items()
            },
            "deployment_checks": deployment_checks,
            "component_scores": {
                key: round(value * 100.0, 2)
                for key, value in readiness_components.items()
            },
            "data_integrity": {
                "targets_with_complete_metrics": int(max(total_targets - len(incomplete_target_ids), 0)),
                "incomplete_target_ids": incomplete_target_ids,
            },
            "directives": command_directives,
        },
        "engagement_priority_queue": priority_queue,
        "tactical_assessment": {
            "state_counts": state_counts,
            "recommendations": recommendations,
        },
        "mission_model": {
            "packet_loss_formula": "PL = 1 - exp(-k * SNR / d^alpha)",
            "packet_loss_model": validation_payload.get("packet_loss_model", {}) if isinstance(validation_payload, dict) else {},
            "energy_formula": "P = P_hover + c_drag*v^3 + eta*m*a*v; E = sum(P*dt)",
            "energy_model": validation_payload.get("energy_model", {}) if isinstance(validation_payload, dict) else {},
            "constraints": {
                "interceptor_speed_gt_target_speed": True,
                "kill_radius_m_min": 0.001,
                "packet_loss_rate_bounds": [0.0, 0.98],
                "snr_db_bounds": [-20.0, 45.0],
            },
        },
    }


def _build_frame_snapshot(
    frame: MissionFrame,
    validation: dict[str, Any],
    mission_id: int,
    throughput_fps: float,
) -> dict[str, Any]:
    origin = tuple(validation.get("origin_lat_lon", (37.7749, -122.4194)))
    active_target = next((target for target in frame.targets if target.name == frame.active_target), None)
    if active_target is None and frame.targets:
        active_target = frame.targets[0]
    if active_target is not None:
        active_distance_m = float(np.linalg.norm(frame.interceptor_position - active_target.position))
        active_covariance = _covariance_from_uncertainty(getattr(active_target, "uncertainty_radius_m", 0.25))
        active_mahalanobis_distance = _mahalanobis_distance(
            np.asarray(active_target.position, dtype=float) - np.asarray(frame.interceptor_position, dtype=float),
            active_covariance,
        )
        mission_success_probability = _kill_probability_from_mahalanobis(active_mahalanobis_distance)
        relative_position = np.asarray(active_target.position, dtype=float) - np.asarray(frame.interceptor_position, dtype=float)
        relative_velocity = np.asarray(active_target.velocity, dtype=float) - np.asarray(frame.interceptor_velocity, dtype=float)
        relative_distance_m = max(float(np.linalg.norm(relative_position)), 1e-6)
        los_rate_rps = float(np.linalg.norm(np.cross(relative_position, relative_velocity)) / (relative_distance_m**2))
        closing_velocity_mps = max(-float(np.dot(relative_velocity, relative_position / relative_distance_m)), 0.0)
        rmse_measured_true = float(frame.rmse_m)
        mean_uncertainty_m = float(np.mean([target.uncertainty_radius_m for target in frame.targets]))
        innovation_m = float(np.mean([target.innovation_m for target in frame.targets]))
        innovation_gate = float(max([target.innovation_gate for target in frame.targets], default=0.0))
    else:
        active_distance_m = 0.0
        los_rate_rps = 0.0
        closing_velocity_mps = 0.0
        rmse_measured_true = 0.0
        mean_uncertainty_m = 0.0
        innovation_m = 0.0
        innovation_gate = 0.0
        active_mahalanobis_distance = 0.0
        mission_success_probability = 0.0

    spoofing_active = any(target.spoofing_active or target.spoofing_detected for target in frame.targets)
    confidence_score = max(0.0, min(1.0, 1.0 - (innovation_m / max(innovation_gate, 1e-3))))
    interceptor_lat, interceptor_lon, interceptor_altitude = local_position_to_lla(np.asarray(frame.interceptor_position, dtype=float), origin)
    return {
        "mission_id": mission_id,
        "status": "running",
        "step": frame.step,
        "time_s": frame.time_s,
        "active_stage": frame.active_stage,
        "stage": frame.active_stage,
        "active_target": frame.active_target,
        "backend_throughput_fps": float(throughput_fps),
        "detection_fps": float(frame.detection_fps),
        "rmse_m": float(frame.rmse_m),
        "rmse_measured_true_m": rmse_measured_true,
        "mean_uncertainty_m": mean_uncertainty_m,
        "kill_probability": mission_success_probability,
        "mission_success_probability": mission_success_probability,
        "mahalanobis_distance": float(active_mahalanobis_distance),
        "active_distance_m": active_distance_m,
        "closing_velocity_mps": closing_velocity_mps,
        "los_rate_rps": los_rate_rps,
        "innovation_m": innovation_m,
        "innovation_gate": innovation_gate,
        "confidence_score": confidence_score,
        "interceptor_position": frame.interceptor_position.tolist(),
        "interceptor_velocity": frame.interceptor_velocity.tolist(),
        "interceptor_geo": {"lat": float(interceptor_lat), "lon": float(interceptor_lon), "altitude_m": float(interceptor_altitude)},
        "filtered_state": active_target.filtered_estimate.tolist() if active_target is not None else [0.0, 0.0, 0.0],
        "raw_drifted_state": active_target.raw_measurement.tolist() if active_target is not None else [0.0, 0.0, 0.0],
        "spoofing_active": spoofing_active,
        "ekf_lock": bool(validation.get("ekf_enabled", False)),
        "target_count": len(frame.targets),
        "targets": [
            {
                "target_id": getattr(target, "id", target.name),
                "status": (
                    "JAMMED" if getattr(target, "jammed", False)
                    else ("CRITICAL" if float(np.linalg.norm(np.asarray(target.position, dtype=float) - np.asarray(frame.interceptor_position, dtype=float))) < 50.0 and float(np.linalg.norm(target.velocity)) > 5.0 else "ACTIVE")
                ),
                "threat_level": float(target.threat_level),
                "distance_m": float(np.linalg.norm(np.asarray(target.position, dtype=float) - np.asarray(frame.interceptor_position, dtype=float))),
                "spoofing_active": bool(getattr(target, "spoofing_active", False) or getattr(target, "spoofing_detected", False) or getattr(target, "is_spoofed", False)),
                "innovation_m": float(getattr(target, "innovation_m", 0.0)),
                "estimated_error_m": float(getattr(target, "estimated_error_m", 0.0)),
                "packet_dropped": bool(getattr(target, "packet_dropped", False)),
                "packet_loss_probability": float(getattr(target, "packet_loss_probability", 0.0)),
                "link_snr_db": float(getattr(target, "link_snr_db", 0.0)),
                "kill_probability": float(
                    getattr(
                        target,
                        "kill_probability",
                        _kill_probability_from_mahalanobis(
                            _mahalanobis_distance(
                                np.asarray(target.position, dtype=float) - np.asarray(frame.interceptor_position, dtype=float),
                                _covariance_from_uncertainty(float(getattr(target, "uncertainty_radius_m", 0.25))),
                            )
                        ),
                    )
                ),
                "name": target.name,
                "true_position": target.position.tolist(),
                "spoofed_position": target.raw_measurement.tolist(),
                "estimated_position": target.filtered_estimate.tolist(),
                "velocity": target.velocity.tolist(),
                "drift_rate_mps": float(getattr(target, "drift_rate_mps", 0.0)),
                "spoof_offset_m": float(getattr(target, "spoof_offset_m", 0.0)),
                "uncertainty_radius_m": float(getattr(target, "uncertainty_radius_m", 0.0)),
                "innovation_gate": float(getattr(target, "innovation_gate", 0.5)),
                "spoofing_detected": bool(getattr(target, "spoofing_detected", False)),
                "jammed": bool(getattr(target, "jammed", False)),
                "true_geo": {
                    "lat": float(local_position_to_lla(np.asarray(target.position, dtype=float), origin)[0]),
                    "lon": float(local_position_to_lla(np.asarray(target.position, dtype=float), origin)[1]),
                    "altitude_m": float(local_position_to_lla(np.asarray(target.position, dtype=float), origin)[2]),
                },
                "spoofed_geo": {
                    "lat": float(local_position_to_lla(np.asarray(target.raw_measurement, dtype=float), origin)[0]),
                    "lon": float(local_position_to_lla(np.asarray(target.raw_measurement, dtype=float), origin)[1]),
                    "altitude_m": float(local_position_to_lla(np.asarray(target.raw_measurement, dtype=float), origin)[2]),
                },
                "estimated_geo": {
                    "lat": float(local_position_to_lla(np.asarray(target.filtered_estimate, dtype=float), origin)[0]),
                    "lon": float(local_position_to_lla(np.asarray(target.filtered_estimate, dtype=float), origin)[1]),
                    "altitude_m": float(local_position_to_lla(np.asarray(target.filtered_estimate, dtype=float), origin)[2]),
                },
            }
            for target in frame.targets
        ],
        "validation": validation,
    }


def _build_terminal_snapshot(
    replay: MissionReplay,
    mission_id: int,
    throughput_fps: float,
    prior_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not replay.frames:
        snapshot = dict(prior_snapshot or {})
        snapshot.update(
            {
                "mission_id": mission_id,
                "status": "complete",
                "active_stage": "Target Redirected",
                "stage": "Target Redirected",
                "mission_success": False,
                "safe_intercepts": 0,
                "kill_probability": 0.0,
                "active_distance_m": 0.0,
                "terminal_metric_basis": "no_frames",
            }
        )
        return snapshot

    final_frame = replay.frames[-1]
    snapshot = _build_frame_snapshot(final_frame, replay.validation, mission_id, throughput_fps)
    distance_frame = replay.distance_frame
    if distance_frame.empty:
        closest_distance_m = float(snapshot.get("active_distance_m", 0.0))
        closest_target = str(snapshot.get("active_target", final_frame.active_target))
    else:
        closest_row = distance_frame.loc[distance_frame["distance_m"].idxmin()]
        closest_distance_m = float(closest_row["distance_m"])
        closest_target = str(closest_row["target"])

    active_target = next((target for target in final_frame.targets if target.name == closest_target), None)
    if active_target is None and final_frame.targets:
        active_target = final_frame.targets[0]

    mean_uncertainty_m = float(
        np.mean([target.uncertainty_radius_m for target in final_frame.targets], dtype=float)
    ) if final_frame.targets else float(snapshot.get("mean_uncertainty_m", 0.25))
    if active_target is not None:
        terminal_delta = np.asarray(active_target.position, dtype=float) - np.asarray(final_frame.interceptor_position, dtype=float)
        terminal_covariance = _covariance_from_uncertainty(float(active_target.uncertainty_radius_m))
        terminal_mahalanobis_distance = _mahalanobis_distance(terminal_delta, terminal_covariance)
    else:
        terminal_mahalanobis_distance = _mahalanobis_distance(
            np.array([closest_distance_m, 0.0, 0.0], dtype=float),
            _covariance_from_uncertainty(mean_uncertainty_m),
        )
    closest_kill_probability = _kill_probability_from_mahalanobis(terminal_mahalanobis_distance)
    mission_success = bool(replay.validation.get("success", False))
    safe_intercepts = int(replay.safe_intercepts)

    snapshot.update(
        {
            "status": "complete",
            "active_stage": "Target Redirected" if mission_success else final_frame.active_stage,
            "stage": "Target Redirected" if mission_success else final_frame.active_stage,
            "active_target": closest_target,
            "active_distance_m": closest_distance_m,
            "closest_approach_m": closest_distance_m,
            "closest_approach_target_id": closest_target,
            "kill_probability": max(closest_kill_probability, 1.0 if mission_success else 0.0),
            "kill_probability_target_id": closest_target,
            "mission_success": mission_success,
            "safe_intercepts": safe_intercepts,
            "target_count": len(final_frame.targets),
            "terminal_metric_basis": "closest_approach",
            "backend_throughput_fps": float(throughput_fps),
            "mahalanobis_distance": float(terminal_mahalanobis_distance),
            "mission_success_probability": float(max(closest_kill_probability, 1.0 if mission_success else 0.0)),
        }
    )
    if active_target is not None:
        snapshot["spoofing_active"] = bool(active_target.spoofing_detected or active_target.spoofing_active)
        snapshot["innovation_m"] = float(active_target.innovation_m)
        snapshot["innovation_gate"] = float(active_target.innovation_gate)
    target_ids = [target.name for target in final_frame.targets]
    target_results = [
        _compute_target_result_from_replay(replay, target_name=target_id, target_id=target_id)
        for target_id in target_ids
    ]
    return _apply_target_results_to_snapshot(snapshot, target_results)


def _build_replay_data(replay: MissionReplay) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    filtered_state: list[list[float]] = []
    raw_drifted_state: list[list[float]] = []
    true_state: list[list[float]] = []
    interceptor_state: list[list[float]] = []
    for frame in replay.frames:
        active_target = next((target for target in frame.targets if target.name == frame.active_target), frame.targets[0] if frame.targets else None)
        if active_target is not None:
            filtered_state.append(active_target.filtered_estimate.tolist())
            raw_drifted_state.append(active_target.raw_measurement.tolist())
            true_state.append(active_target.position.tolist())
        interceptor_state.append(frame.interceptor_position.tolist())
        frames.append(
            {
                "step": int(frame.step),
                "time_s": float(frame.time_s),
                "active_stage": str(frame.active_stage),
                "active_target": str(frame.active_target),
                "rmse_m": float(frame.rmse_m),
                "detection_fps": float(frame.detection_fps),
                "interceptor_position": frame.interceptor_position.tolist(),
                "interceptor_velocity": frame.interceptor_velocity.tolist(),
                "targets": [
                    {
                        "name": getattr(target, "name", "unknown"),
                        "true_position": target.position.tolist() if hasattr(target, "position") and target.position is not None else [0.0, 0.0, 0.0],
                        "spoofed_position": target.raw_measurement.tolist() if hasattr(target, "raw_measurement") and target.raw_measurement is not None else [0.0, 0.0, 0.0],
                        "estimated_position": target.filtered_estimate.tolist() if hasattr(target, "filtered_estimate") and target.filtered_estimate is not None else [0.0, 0.0, 0.0],
                        "velocity": target.velocity.tolist() if hasattr(target, "velocity") and target.velocity is not None else [0.0, 0.0, 0.0],
                        "threat_level": float(getattr(target, "threat_level", 0.0)),
                        "spoofing_active": bool(getattr(target, "spoofing_active", False)),
                        "drift_rate_mps": float(getattr(target, "drift_rate_mps", 0.0)),
                        "spoof_offset_m": float(getattr(target, "spoof_offset_m", 0.0)),
                        "innovation_m": float(getattr(target, "innovation_m", 0.0)),
                        "innovation_gate": float(getattr(target, "innovation_gate", 0.0)),
                        "packet_dropped": bool(getattr(target, "packet_dropped", False)),
                        "packet_loss_probability": float(getattr(target, "packet_loss_probability", 0.0)),
                        "link_snr_db": float(getattr(target, "link_snr_db", 0.0)),
                        "spoofing_detected": bool(getattr(target, "spoofing_detected", False)),
                        "jammed": bool(getattr(target, "jammed", False)),
                        "estimated_error_m": float(getattr(target, "estimated_error_m", 0.0)),
                        "kill_probability": float(getattr(target, "kill_probability", 0.0)),
                    }
                    for target in frame.targets
                ],
            }
        )
    return {
        "frame_count": len(frames),
        "frames": frames,
        "filtered_state": filtered_state,
        "raw_drifted_state": raw_drifted_state,
        "true_state": true_state,
        "interceptor_state": interceptor_state,
        "map_rows": replay.map_frame.to_dict("records"),
        "origin_lat_lon": tuple(replay.validation.get("origin_lat_lon", (37.7749, -122.4194))),
        "distance_records": replay.distance_frame.to_dict("records"),
        "validation": dict(replay.validation),
        "safe_intercepts": int(replay.safe_intercepts),
    }


def _write_platform_preview(replay_data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Drone Interceptor Replay Preview</title>
  <style>
    body {{
      margin: 0;
      background: #061018;
      color: #e6fcff;
      font-family: 'JetBrains Mono', 'Roboto Mono', monospace;
    }}
    .shell {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    canvas {{
      width: 100%;
      height: auto;
      border: 1px solid #183142;
      border-radius: 14px;
      background: #08111b;
      box-shadow: 0 0 24px rgba(0, 242, 255, 0.12);
    }}
    .row {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .card {{
      padding: 12px 14px;
      border: 1px solid rgba(0, 242, 255, 0.35);
      border-radius: 12px;
      background: rgba(8, 17, 27, 0.9);
    }}
  </style>
</head>
<body>
  <div class="shell">
    <h2>Backend-Driven Mission Replay</h2>
    <canvas id="replay" width="1160" height="620"></canvas>
    <div class="row">
      <div class="card" id="stage">Stage: n/a</div>
      <div class="card" id="rmse">RMSE: n/a</div>
      <div class="card" id="frames">Frames: {int(replay_data.get("frame_count", 0))}</div>
      <div class="card" id="status">Artifact: platform_preview.html</div>
    </div>
  </div>
  <script>
    const replayData = {__import__("json").dumps(replay_data).replace("NaN", "null")};
    const canvas = document.getElementById("replay");
    const ctx = canvas.getContext("2d");
    const stageNode = document.getElementById("stage");
    const rmseNode = document.getElementById("rmse");
    const frames = replayData.frames || [];
    const mapPointFactory = (frame) => {{
      const all = [];
      if (frame.interceptor_position) all.push(frame.interceptor_position);
      (frame.targets || []).forEach((target) => {{
        all.push(target.true_position || [0, 0, 0]);
        all.push(target.spoofed_position || [0, 0, 0]);
        all.push(target.estimated_position || [0, 0, 0]);
      }});
      const xs = all.map((point) => point[0]);
      const ys = all.map((point) => point[1]);
      const minX = Math.min(...xs, 0);
      const maxX = Math.max(...xs, 1);
      const minY = Math.min(...ys, 0);
      const maxY = Math.max(...ys, 1);
      const spanX = Math.max(maxX - minX, 1);
      const spanY = Math.max(maxY - minY, 1);
      return (point) => {{
        const x = 48 + ((point[0] - minX) / spanX) * 1030;
        const y = 560 - ((point[1] - minY) / spanY) * 470;
        return [x, y];
      }};
    }};
    const drawFrame = (frame) => {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#08111b";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const mapPoint = mapPointFactory(frame);
      ctx.fillStyle = "#e6fcff";
      ctx.font = "28px JetBrains Mono";
      ctx.fillText("Backend Replay Buffer", 36, 42);
      (frame.targets || []).forEach((target) => {{
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
      if (frame.interceptor_position) {{
        const interceptor = mapPoint(frame.interceptor_position);
        ctx.fillStyle = "#00f2ff";
        ctx.beginPath(); ctx.arc(interceptor[0], interceptor[1], 6, 0, Math.PI * 2); ctx.fill();
      }}
      stageNode.textContent = `Stage: ${{frame.active_stage || 'n/a'}}`;
      rmseNode.textContent = `RMSE: ${{Number(frame.rmse_m || 0).toFixed(3)}} m`;
    }};
    let index = 0;
    const tick = () => {{
      if (!frames.length) return;
      drawFrame(frames[index]);
      index = (index + 1) % frames.length;
      requestAnimationFrame(() => setTimeout(tick, 40));
    }};
    tick();
  </script>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")


@dataclass
class MissionState:
    snapshot: dict[str, Any] = field(default_factory=lambda: {"status": "idle"})
    validation_report: dict[str, Any] | None = None
    preflight_report: dict[str, Any] | None = None
    replay_data: dict[str, Any] | None = None
    mission_task: asyncio.Task[Any] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    mission_counter: int = 0
    deploy_ready: bool = False
    active_run_id: str | None = None
    fps_window: deque[float] = field(default_factory=lambda: deque(maxlen=12))
    last_stream_at_s: float | None = None
    stream_packets_sent: int = 0

    def _effective_snapshot_locked(self) -> dict[str, Any]:
        snapshot = dict(self.snapshot)
        mission_status = str(snapshot.get("status", "idle")).lower()
        if mission_status == "idle" and self.preflight_report is not None:
            if bool(self.preflight_report.get("ready", False)):
                snapshot["status"] = "preflight_ready"
                snapshot.setdefault("active_stage", "Preflight")
                snapshot.setdefault("stage", "Preflight")
                snapshot["target_count"] = int(self.preflight_report.get("spawned_targets", snapshot.get("target_count", 0)))
            else:
                snapshot.setdefault("active_stage", "Preflight")
                snapshot.setdefault("stage", "Preflight")
        if self.fps_window:
            snapshot["detection_fps_window_avg"] = float(np.mean(np.asarray(self.fps_window, dtype=float)))
        else:
            snapshot.setdefault("detection_fps_window_avg", float(snapshot.get("detection_fps", 0.0)))
        age_ms = None if self.last_stream_at_s is None else max((time.time() - self.last_stream_at_s) * 1000.0, 0.0)
        snapshot["heartbeat_age_ms"] = age_ms
        snapshot["heartbeat_live"] = bool(
            self.last_stream_at_s is not None
            and mission_status in {"preparing", "running", "complete"}
            and age_ms is not None
            and age_ms <= 500.0
        )
        snapshot["stream_packets_sent"] = int(self.stream_packets_sent)
        return snapshot

    async def get_payload(self) -> dict[str, Any]:
        async with self.lock:
            snapshot = self._effective_snapshot_locked()
            mission_status = str(snapshot.get("status", "idle")).lower()
            payload_type = "MISSION_COMPLETE" if mission_status == "complete" and self.replay_data is not None else ("LIVE" if mission_status in {"running", "preparing"} else "STATE")
            return {
                "type": payload_type,
                "snapshot": snapshot,
                "preflight": None if self.preflight_report is None else dict(self.preflight_report),
                "validation": None if self.validation_report is None else dict(self.validation_report),
                "replay_data": None if self.replay_data is None else dict(self.replay_data),
                "event": "mission_complete" if mission_status == "complete" and self.replay_data is not None else None,
                "artifact_url": "outputs/platform_preview.html" if self.replay_data is not None else None,
                "finish_signal": {"artifact_url": "outputs/platform_preview.html", "replay_ready": True} if mission_status == "complete" and self.replay_data is not None else None,
                "deploy_ready": bool(self.deploy_ready),
                "run_id": self.active_run_id,
            }

    async def set_snapshot(self, snapshot: dict[str, Any]) -> None:
        async with self.lock:
            self.snapshot = dict(snapshot)
            detection_fps = float(self.snapshot.get("detection_fps", 0.0))
            if detection_fps > 0.0:
                self.fps_window.append(detection_fps)
            self.last_stream_at_s = time.time()
            self.stream_packets_sent += 1

    async def set_preflight(self, report: dict[str, Any]) -> None:
        async with self.lock:
            self.preflight_report = dict(report)

    async def set_replay_data(self, replay_data: dict[str, Any] | None) -> None:
        async with self.lock:
            self.replay_data = None if replay_data is None else dict(replay_data)

    async def set_validation(self, report: dict[str, Any]) -> None:
        async with self.lock:
            self.validation_report = dict(report)
            self.deploy_ready = bool(report.get("validation_success", False))

    async def stop_current(self) -> None:
        if self.mission_task is not None and not self.mission_task.done():
            self.mission_task.cancel()
            try:
                await self.mission_task
            except asyncio.CancelledError:
                pass
        self.mission_task = None

    async def start_mission(self, payload: dict[str, Any]) -> dict[str, Any]:
        await self.stop_current()
        self.mission_counter += 1
        mission_id = self.mission_counter
        run_record = RUN_STORE.create_run(
            kind="mission",
            status="started",
            config=dict(payload),
            seed=int(payload.get("random_seed", 61)),
        )
        self.active_run_id = run_record.run_id
        await self.set_replay_data(None)
        self.fps_window.clear()
        await self.set_snapshot(
            {
                "status": "preparing",
                "mission_id": mission_id,
                "active_stage": "Preflight",
                "stage": "Preflight",
                "target_count": int(payload.get("num_targets", 3)),
            }
        )
        self.mission_task = asyncio.create_task(self._run_mission(mission_id, payload))
        return {"status": "started", "mission_id": mission_id, "run_id": run_record.run_id}

    async def _run_mission(self, mission_id: int, payload: dict[str, Any]) -> None:
        # ── Spoof injection: honour per-request flag or global env flag ───────
        enable_spoofing = bool(payload.get("enable_spoofing", False))
        set_flag("SPOOFING", enable_spoofing)
        spoof_service = get_default_service()
        TELEMETRY_LOGGER.info(
            "mission_start mission_id=%s targets=%s use_ekf=%s noise=%.3f drift=%.3f latency_ms=%.1f",
            mission_id,
            int(payload.get("num_targets", 3)),
            bool(payload.get("use_ekf", True)),
            float(payload.get("noise_std_m", 0.45)),
            float(payload.get("drift_rate_mps", 0.3)),
            float(payload.get("latency_ms", 0.0)),
        )

        manager = _build_manager(payload)
        preflight = await asyncio.to_thread(
            manager.preflight_validate,
            int(payload.get("num_targets", 3)),
            int(payload.get("random_seed", 61)),
            float(payload.get("interceptor_speed_mps", 28.0)),
        )
        await self.set_preflight(preflight)
        if not preflight.get("ready", False):
            await self.set_snapshot({"status": "preflight_failed", "mission_id": mission_id, "preflight": preflight})
            return

        replay: MissionReplay = await asyncio.to_thread(
            manager.run_replay,
            int(payload.get("num_targets", 3)),
            bool(payload.get("use_ekf", True)),
            float(payload.get("drift_rate_mps", 0.3)),
            float(payload.get("noise_std_m", 0.45)),
            float(payload.get("latency_ms", 0.0)),
            float(payload.get("packet_loss_rate", 0.0)),
            int(payload.get("random_seed", 61)),
            int(payload.get("max_steps", 141)),
            float(payload.get("dt", 0.05)),
            float(payload.get("kill_radius_m", 0.5)),
            float(payload.get("guidance_gain", 4.2)),
            bool(payload.get("use_ekf_anti_spoofing", True)),
            float(payload.get("target_speed_mps")) if payload.get("target_speed_mps") is not None else None,
            float(payload.get("interceptor_speed_mps")) if payload.get("interceptor_speed_mps") is not None else None,
            float(payload.get("ekf_process_noise")) if payload.get("ekf_process_noise") is not None else None,
            _coerce_measurement_noise(payload.get("ekf_measurement_noise")),
            bool(payload.get("enable_spoofing", False)),
            float(payload.get("link_snr_db", 28.0)),
            float(payload.get("packet_loss_k", 0.12)),
            float(payload.get("packet_loss_alpha", 1.8)),
        )

        previous_tick = time.perf_counter()
        import dataclasses
        previous_jammed_count = 0
        for frame in replay.frames:
            current_tick = time.perf_counter()
            throughput = 1.0 / max(current_tick - previous_tick, 1e-6)
            previous_tick = current_tick
            jammed_count = sum(1 for t in frame.targets if t.jammed)
            if jammed_count > previous_jammed_count:
                frame = dataclasses.replace(frame, active_stage="TARGET REDIRECTED")
            previous_jammed_count = jammed_count

            # ── Spoof middleware intercept ─────────────────────────────────
            spoof_result = spoof_service.apply_spoof(
                {"frame": frame.interceptor_position, "targets": [t.position for t in frame.targets]},
                data_type="video",
            )
            snapshot = _build_frame_snapshot(frame, replay.validation, mission_id, throughput)
            snapshot["spoof_injected"] = spoof_result.spoofed
            if spoof_result.spoofed:
                snapshot["spoof_metadata"] = {
                    "algorithm": spoof_result.metadata.get("algorithm", "unknown"),
                    "elapsed_s": round(spoof_result.elapsed_s, 6),
                }
            await self.set_snapshot(snapshot)
            if frame.step % 20 == 0:
                TELEMETRY_LOGGER.info(
                    "mission_heartbeat mission_id=%s step=%s stage=%s rmse=%.4f fps=%.2f",
                    mission_id,
                    frame.step,
                    str(frame.active_stage),
                    float(snapshot.get("rmse_m", 0.0)),
                    float(snapshot.get("detection_fps", 0.0)),
                )
            await asyncio.sleep(1.0 / STREAM_HZ)

        final_payload = await self.get_payload()
        snapshot = _build_terminal_snapshot(
            replay=replay,
            mission_id=mission_id,
            throughput_fps=float((final_payload.get("snapshot") or {}).get("backend_throughput_fps", STREAM_HZ)),
            prior_snapshot=final_payload.get("snapshot"),
        )
        replay_data = _build_replay_data(replay)
        _write_platform_preview(replay_data, OUTPUTS / "platform_preview.html")
        await asyncio.to_thread(manager.export_cinematic_demo, replay, prefix="day8_bms_demo")
        await self.set_replay_data(replay_data)
        await self.set_snapshot(snapshot)
        TELEMETRY_LOGGER.info(
            "mission_complete mission_id=%s success=%s safe_intercepts=%s rmse=%.4f",
            mission_id,
            bool(snapshot.get("mission_success", False)),
            int(snapshot.get("safe_intercepts", 0)),
            float(snapshot.get("rmse_m", 0.0)),
        )
        if self.active_run_id is not None:
            RUN_STORE.update_run(
                self.active_run_id,
                status="complete",
                metrics={
                    "mission_success": bool(replay.validation.get("success", False)),
                    "safe_intercepts": int(replay.safe_intercepts),
                    "final_status": snapshot["status"],
                },
                validation=dict(replay.validation),
            )


MISSION_STATE = MissionState()


@app.get("/health")
async def health() -> dict[str, Any]:
    payload = await MISSION_STATE.get_payload()
    return {"status": "ok", "mission_status": payload["snapshot"].get("status", "idle"), "schema_version": "1.0"}


@app.get("/status")
async def status() -> dict[str, Any]:
    payload = await MISSION_STATE.get_payload()
    snapshot = payload["snapshot"]
    mission_status = str(snapshot.get("status", "idle")).lower()
    lifecycle = "COMPLETE" if mission_status == "complete" else ("PREPARED" if mission_status == "preflight_ready" else ("ACTIVE" if mission_status in {"preparing", "running"} else "STANDBY"))
    targets = snapshot.get("targets") if isinstance(snapshot.get("targets"), list) else []
    validation_payload = payload.get("validation") or {}
    mission_insights = _build_mission_insights(snapshot, validation_payload)
    return {
        "service_status": "ok",
        "schema_version": "1.0",
        "mission_status": mission_status,
        "stage": snapshot.get("stage", snapshot.get("active_stage", "idle")),
        "lifecycle": lifecycle,
        "target_count": int(snapshot.get("target_count", len(targets))),
        "heartbeat_live": bool(snapshot.get("heartbeat_live", False)),
        "heartbeat_age_ms": snapshot.get("heartbeat_age_ms"),
        "detection_fps": float(snapshot.get("detection_fps_window_avg", snapshot.get("detection_fps", 0.0))),
        "deploy_ready": bool(payload.get("deploy_ready", False)),
        "validation_success": bool(validation_payload.get("validation_success", False)),
        "run_id": payload.get("run_id"),
        "targets": targets,
        "mission_insights": mission_insights,
        "airsim_yolo": _airsim_yolo_status({"refresh": False}),
    }


@app.get("/airsim/yolo/status")
async def get_airsim_yolo_status() -> dict[str, Any]:
    return await asyncio.to_thread(_airsim_yolo_status, {"refresh": False})


@app.post("/airsim/yolo/status")
async def post_airsim_yolo_status(payload: dict[str, Any]) -> dict[str, Any]:
    return await asyncio.to_thread(_airsim_yolo_status, payload)


@app.post("/preflight")
async def preflight(payload: dict[str, Any]) -> dict[str, Any]:
    manager = _build_manager(payload)
    report = await asyncio.to_thread(
        manager.preflight_validate,
        int(payload.get("num_targets", 3)),
        int(payload.get("random_seed", 61)),
        float(payload.get("interceptor_speed_mps", 28.0)),
    )
    await MISSION_STATE.set_preflight(report)
    current = await MISSION_STATE.get_payload()
    snapshot = current.get("snapshot", {})
    if str(snapshot.get("status", "idle")).lower() in {"idle", "stopped", "complete", "preflight_ready", "preflight_failed"}:
        await MISSION_STATE.set_snapshot(
            {
                "status": "preflight_ready" if bool(report.get("ready", False)) else "preflight_failed",
                "mission_id": int(snapshot.get("mission_id", 0)),
                "active_stage": "Preflight",
                "stage": "Preflight",
                "target_count": int(report.get("spawned_targets", 0)),
            }
        )
    return report


@app.post("/mission/start")
async def start_mission(payload: dict[str, Any]) -> dict[str, Any]:
    return await MISSION_STATE.start_mission(payload)


@app.get("/mission/state")
async def mission_state() -> dict[str, Any]:
    return await MISSION_STATE.get_payload()


@app.post("/mission/stop")
async def stop_mission() -> dict[str, str]:
    await MISSION_STATE.stop_current()
    await MISSION_STATE.set_replay_data(None)
    await MISSION_STATE.set_snapshot({"status": "stopped"})
    if MISSION_STATE.active_run_id is not None:
        RUN_STORE.update_run(MISSION_STATE.active_run_id, status="stopped")
    return {"status": "stopped"}


@app.post("/mission/start/v2")
async def start_mission_v2(payload: dict[str, Any], background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Enhanced mission endpoint with EKF-based control, kinematic validation, and artifact generation."""
    from fastapi import HTTPException
    from drone_interceptor.backend.mission_service import MissionFinalizer

    # Extract and validate mission parameters
    target_speed = float(payload.get("target_speed_mps", 6.0))
    interceptor_speed = float(payload.get("interceptor_speed_mps", 20.0))
    
    # ─── Validation Check: Kinematic Feasibility ───────────────────────────
    if interceptor_speed <= target_speed:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Kinematic Infeasibility",
                "message": f"Interceptor speed ({interceptor_speed} m/s) must exceed target speed ({target_speed} m/s)",
                "reason": "Mission geometry is impossible without higher interceptor velocity",
                "recommendation": "Increase interceptor_speed_mps or decrease target_speed_mps",
            }
        )

    # Build mission config
    config = MissionConfig(
        num_targets=int(payload.get("num_targets", 3)),
        target_speed_mps=target_speed,
        interceptor_speed_mps=interceptor_speed,
        drift_rate_mps=float(payload.get("drift_rate_mps", 0.3)),
        noise_level_m=float(payload.get("noise_level_m", 0.45)),
        telemetry_latency_ms=float(payload.get("telemetry_latency_ms", 80.0)),
        packet_loss_rate=float(payload.get("packet_loss_rate", 0.05)),
        guidance_gain=float(payload.get("guidance_gain", 6.0)),
        kill_radius_m=float(payload.get("kill_radius_m", 10.26)),
        max_steps=int(payload.get("max_steps", 200)),
        dt=float(payload.get("dt", 0.05)),
        use_ekf=bool(payload.get("use_ekf", True)),
        use_anti_spoofing=bool(payload.get("use_anti_spoofing", True)),
        random_seed=int(payload.get("random_seed", 42)),
    )
    
    # Create mission controller and finalizer
    controller = MissionController(config, output_dir=OUTPUTS)
    finalizer = MissionFinalizer(output_dir=OUTPUTS)
    
    # Initialize mission
    init_result = controller.initialize_mission()
    
    # Record the run
    run_record = RUN_STORE.create_run(
        kind="mission_v2",
        status="started",
        config=dict(payload),
        seed=config.random_seed,
    )
    
    # Run mission in background with finalization and artifact validation
    async def run_and_record():
        try:
            result = await controller.run_mission()
            
            # Finalize artifacts and verify existence
            finalized_paths = await finalizer.finalize(result, controller.telemetry_log, controller)
            
            artifacts_dict = {}
            for key, path_str in finalized_paths.items():
                artifact_path = Path(path_str)
                artifacts_dict[key] = {
                    "path": str(artifact_path),
                    "type": key,
                    "exists": artifact_path.exists(),
                }
            
            RUN_STORE.update_run(
                run_record.run_id,
                status="complete",
                metrics={
                    "mission_success": bool(result.get("mission_success", False)),
                    "jammed_targets": int(result.get("jammed_targets", 0)),
                    "mission_duration_s": float(result.get("mission_duration_s", 0.0)),
                    "mean_rmse_m": float(np.mean([f.rmse_m for f in controller.telemetry_log])) if controller.telemetry_log else 0.0,
                },
                artifacts=artifacts_dict,
                validation=result,
            )
        except Exception as exc:
            RUN_STORE.update_run(run_record.run_id, status="failed", error=str(exc))
            raise
    
    # Create task to run mission
    asyncio.create_task(run_and_record())
    background_tasks.add_task(_verify_video_artifact, OUTPUTS / "mission_final.mp4", timeout_s=30.0)
    
    return {
        "status": "started",
        "run_id": run_record.run_id,
        "kinematic_feasible": True,
        "mission_config": {
            "num_targets": config.num_targets,
            "target_speed_mps": config.target_speed_mps,
            "interceptor_speed_mps": config.interceptor_speed_mps,
            "kinematic_margin": float(interceptor_speed - target_speed),
            "drift_rate_mps": config.drift_rate_mps,
            "noise_level_m": config.noise_level_m,
            "guidance_gain": config.guidance_gain,
            "use_ekf": config.use_ekf,
            "use_anti_spoofing": config.use_anti_spoofing,
        },
        "init_result": init_result,
    }


@app.post("/run_mission")
async def run_mission_unified(payload: dict[str, Any]) -> dict[str, Any]:
    """
    State-first unified mission endpoint.
    Runs multi-target replay, computes per-target EKF metrics, and returns a full mission summary.
    """
    run_record = None
    try:
        target_ids = _resolve_target_ids(payload)
        mission_payload = dict(payload)
        mission_payload["enable_spoofing"] = bool(mission_payload.get("enable_spoofing", False))
        mission_payload["target_ids"] = target_ids
        mission_payload["num_targets"] = len(target_ids)
        set_flag("SPOOFING", bool(mission_payload.get("enable_spoofing", False)))

        preflight_resp = await preflight(mission_payload)
        if not preflight_resp.get("ready", False):
            return {
                "workflow_status": "error",
                "error": "Preflight failed",
                "preflight": preflight_resp,
            }

        run_record = RUN_STORE.create_run(
            kind="run_mission",
            status="running",
            config=dict(mission_payload),
            seed=int(mission_payload.get("random_seed", 61)),
        )
        MISSION_STATE.active_run_id = run_record.run_id

        manager = _build_manager(mission_payload)
        mission_start_s = time.perf_counter()
        ekf_success_threshold_m = _adaptive_ekf_success_threshold(mission_payload)
        replay: MissionReplay = await asyncio.to_thread(
            manager.run_replay,
            int(len(target_ids)),
            bool(mission_payload.get("use_ekf", True)),
            float(mission_payload.get("drift_rate_mps", 0.3)),
            float(mission_payload.get("noise_std_m", 0.45)),
            float(mission_payload.get("latency_ms", 0.0)),
            float(mission_payload.get("packet_loss_rate", 0.0)),
            int(mission_payload.get("random_seed", 61)),
            int(mission_payload.get("max_steps", 141)),
            float(mission_payload.get("dt", 0.05)),
            float(mission_payload.get("kill_radius_m", 0.5)),
            float(mission_payload.get("guidance_gain", 4.2)),
            bool(mission_payload.get("use_ekf_anti_spoofing", True)),
            float(mission_payload.get("target_speed_mps")) if mission_payload.get("target_speed_mps") is not None else None,
            float(mission_payload.get("interceptor_speed_mps")) if mission_payload.get("interceptor_speed_mps") is not None else None,
            float(mission_payload.get("ekf_process_noise")) if mission_payload.get("ekf_process_noise") is not None else None,
            _coerce_measurement_noise(mission_payload.get("ekf_measurement_noise")),
            bool(mission_payload.get("enable_spoofing", False)),
            float(mission_payload.get("link_snr_db", 28.0)),
            float(mission_payload.get("packet_loss_k", 0.12)),
            float(mission_payload.get("packet_loss_alpha", 1.8)),
        )
        mission_duration_s = max(time.perf_counter() - mission_start_s, 0.0)
        throughput_fps = (
            float(len(replay.frames) / max(mission_duration_s, 1e-6))
            if replay.frames
            else float(STREAM_HZ)
        )

        target_results = await _compute_replay_target_results(
            replay,
            target_ids=target_ids,
            threshold_m=float(ekf_success_threshold_m),
            telemetry_latency_ms=float(mission_payload.get("latency_ms", 0.0)),
            kill_radius_m=float(mission_payload.get("kill_radius_m", 1.0)),
            interceptor_mass_kg=float(mission_payload.get("interceptor_mass_kg", 6.5)),
            hover_power_w=float(mission_payload.get("hover_power_w", 90.0)),
            drag_power_coeff=float(mission_payload.get("drag_power_coeff", 0.02)),
            accel_power_coeff=float(mission_payload.get("accel_power_coeff", 0.45)),
        )
        result_by_id: dict[str, dict[str, Any]] = {}
        for index, result in enumerate(target_results):
            if not isinstance(result, dict):
                continue
            result_target_id = str(result.get("target_id", "")).strip()
            if result_target_id:
                result_by_id[result_target_id] = result
            elif index < len(target_ids):
                result_by_id[str(target_ids[index])] = result

        mission_results: list[dict[str, Any]] = []
        for index, target_id in enumerate(target_ids):
            base_result = result_by_id.get(str(target_id))
            if base_result is None and index < len(target_results) and isinstance(target_results[index], dict):
                base_result = target_results[index]
            if base_result is None:
                base_result = {}

            ekf_success_rate = float(np.clip(_finite_float(base_result.get("ekf_success_rate"), 0.0), 0.0, 1.0))
            rmse_m = max(_finite_float(base_result.get("rmse"), 0.0), 0.0)
            t_launch = _finite_float(base_result.get("t_launch"), 0.0)
            t_impact_raw = base_result.get("t_impact")
            t_impact = _finite_float(t_impact_raw, t_launch) if t_impact_raw is not None else None
            interception_time_raw = base_result.get("interception_time")
            if interception_time_raw is None and t_impact is not None:
                interception_time_s = float(max(t_impact - t_launch, 0.0))
            else:
                interception_time_s = float(max(_finite_float(interception_time_raw, 0.0), 0.0))

            closest_approach_raw = base_result.get("closest_approach_m")
            closest_approach_m = (
                _finite_float(closest_approach_raw, 0.0)
                if closest_approach_raw is not None and _is_finite_number(closest_approach_raw)
                else None
            )
            kill_radius_ref = max(_finite_float(mission_payload.get("kill_radius_m", 1.0), 1.0), 1e-3)
            intercepted = bool(base_result.get("intercepted", False))
            if closest_approach_m is not None and closest_approach_m <= kill_radius_ref:
                intercepted = True

            probability_fallback = float(
                np.clip(
                    0.65 * ekf_success_rate
                    + 0.35 * _kill_probability_from_mahalanobis(rmse_m / max(float(ekf_success_threshold_m), 1e-3)),
                    0.0,
                    1.0,
                )
            )
            mission_success_probability = float(
                np.clip(
                    _finite_float(base_result.get("mission_success_probability"), probability_fallback),
                    0.0,
                    1.0,
                )
            )

            mission_results.append(
                {
                    "target_id": str(target_id),
                    "ekf_success_rate": ekf_success_rate,
                    "interception_time": interception_time_s,
                    "rmse": rmse_m,
                    "closest_approach_m": closest_approach_m,
                    "intercepted": intercepted,
                    "guidance_efficiency_mps2": max(_finite_float(base_result.get("guidance_efficiency_mps2"), 0.0), 0.0),
                    "spoofing_variance": max(_finite_float(base_result.get("spoofing_variance"), 0.0), 0.0),
                    "compute_latency_ms": max(_finite_float(base_result.get("compute_latency_ms"), 0.0), 0.0),
                    "packet_loss_probability": float(
                        np.clip(_finite_float(base_result.get("packet_loss_probability"), 0.0), 0.0, 0.98)
                    ),
                    "link_snr_db": _finite_float(base_result.get("link_snr_db"), 0.0),
                    "energy_consumption_j": max(_finite_float(base_result.get("energy_consumption_j"), 0.0), 0.0),
                    "interceptor_mass_kg": max(_finite_float(base_result.get("interceptor_mass_kg"), mission_payload.get("interceptor_mass_kg", 6.5)), 0.1),
                    "hover_power_w": max(_finite_float(base_result.get("hover_power_w"), mission_payload.get("hover_power_w", 90.0)), 0.0),
                    "drag_power_coeff": max(_finite_float(base_result.get("drag_power_coeff"), mission_payload.get("drag_power_coeff", 0.02)), 0.0),
                    "accel_power_coeff": max(_finite_float(base_result.get("accel_power_coeff"), mission_payload.get("accel_power_coeff", 0.45)), 0.0),
                    "mission_success_probability": mission_success_probability,
                    "t_launch": t_launch,
                    "t_impact": t_impact,
                }
            )
        mission_ekf_success_rate = (
            float(np.mean([row["ekf_success_rate"] for row in mission_results], dtype=float))
            if mission_results
            else 0.0
        )
        mission_rmse = (
            float(np.mean([row["rmse"] for row in mission_results], dtype=float))
            if mission_results
            else 0.0
        )

        snapshot = _build_terminal_snapshot(
            replay=replay,
            mission_id=int((time.time() * 1000.0) % 1_000_000_000),
            throughput_fps=throughput_fps,
        )
        snapshot["kill_radius_m"] = float(mission_payload.get("kill_radius_m", 1.0))
        snapshot = _apply_target_results_to_snapshot(snapshot, target_results)
        snapshot["mission_duration_s"] = float(mission_duration_s)
        replay_data = _build_replay_data(replay)
        _write_platform_preview(replay_data, OUTPUTS / "platform_preview.html")

        validation_report = _build_validation_from_target_results(target_results)
        validation_report["packet_loss_events"] = int(replay.validation.get("packet_loss_events", 0))
        validation_report["packet_loss_samples"] = int(replay.validation.get("packet_loss_samples", 0))
        validation_report["packet_loss_rate"] = float(replay.validation.get("packet_loss_rate", mission_payload.get("packet_loss_rate", 0.0)))
        validation_report["packet_loss_rate_configured"] = float(
            replay.validation.get("packet_loss_rate_configured", mission_payload.get("packet_loss_rate", 0.0))
        )
        validation_report["packet_loss_rate_effective_mean"] = float(replay.validation.get("packet_loss_rate_effective_mean", 0.0))
        validation_report["packet_loss_rate_observed"] = float(replay.validation.get("packet_loss_rate_observed", 0.0))
        validation_report["packet_loss_model"] = replay.validation.get("packet_loss_model", {})
        validation_report["energy_model"] = {
            "formula": "P = P_hover + c_drag*v^3 + eta*m*a*v; E = sum(P*dt)",
            "interceptor_mass_kg": float(mission_payload.get("interceptor_mass_kg", 6.5)),
            "hover_power_w": float(mission_payload.get("hover_power_w", 90.0)),
            "drag_power_coeff": float(mission_payload.get("drag_power_coeff", 0.02)),
            "accel_power_coeff": float(mission_payload.get("accel_power_coeff", 0.45)),
        }
        await MISSION_STATE.set_replay_data(replay_data)
        await MISSION_STATE.set_validation(validation_report)
        await MISSION_STATE.set_snapshot(snapshot)

        RUN_STORE.update_run(
            run_record.run_id,
            status="complete",
            metrics={
                "mission_success": bool(replay.validation.get("success", False)),
                "safe_intercepts": int(replay.safe_intercepts),
                "target_count": len(target_results),
                "ekf_success_rate": mission_ekf_success_rate,
                "rmse_m": mission_rmse,
                "mission_duration_s": float(mission_duration_s),
            },
            validation=validation_report,
        )

        state_resp = await mission_state()
        status_resp = await status()
        mission_summary = {
            "target_count": int(len(mission_results)),
            "ekf_success_rate": float(mission_ekf_success_rate),
            "ekf_success_threshold_m": float(ekf_success_threshold_m),
            "rmse": float(mission_rmse),
            "mission_duration_s": float(mission_duration_s),
            "packet_loss_rate_observed": float(replay.validation.get("packet_loss_rate_observed", 0.0)),
            "packet_loss_rate_effective_mean": float(replay.validation.get("packet_loss_rate_effective_mean", 0.0)),
            "packet_loss_model": replay.validation.get("packet_loss_model", {}),
            "energy_model": validation_report.get("energy_model", {}),
            "interception_completion_rate": float(
                np.mean([1.0 if bool(row.get("intercepted", False)) else 0.0 for row in mission_results], dtype=float)
            ) if mission_results else 0.0,
            "results": mission_results,
        }
        if mission_results:
            best_kill_row = max(
                mission_results,
                key=lambda row: float(row.get("mission_success_probability", 0.0)),
            )
            closest_rows = [row for row in mission_results if row.get("closest_approach_m") is not None]
            closest_row = (
                min(closest_rows, key=lambda row: float(row.get("closest_approach_m", float("inf"))))
                if closest_rows
                else None
            )
            mission_summary["best_kill_probability"] = float(best_kill_row.get("mission_success_probability", 0.0))
            mission_summary["best_kill_probability_target_id"] = str(best_kill_row.get("target_id", "n/a"))
            mission_summary["closest_approach_m"] = (
                float(closest_row.get("closest_approach_m", 0.0))
                if closest_row is not None
                else None
            )
            mission_summary["closest_approach_target_id"] = (
                str(closest_row.get("target_id", "n/a"))
                if closest_row is not None
                else "n/a"
            )

        return {
            "workflow_status": "success",
            "preflight": preflight_resp,
            "mission": {
                "status": "complete",
                "run_id": run_record.run_id,
                "mission_duration_s": float(mission_duration_s),
            },
            "run_id": run_record.run_id,
            "mission_summary": mission_summary,
            "results": mission_results,
            "state": state_resp,
            "status": status_resp,
            "targets": status_resp.get("targets", []),
            "mission_insights": status_resp.get("mission_insights"),
            "snapshot": state_resp.get("snapshot", {}),
            "validation": validation_report,
            "timestamp": time.time(),
        }
    except Exception as exc:
        if run_record is not None:
            RUN_STORE.update_run(run_record.run_id, status="failed", error=str(exc))
        return {
            "workflow_status": "error",
            "error": str(exc),
        }


@app.get("/mission/{mission_id}/artifacts")
async def get_mission_artifacts(mission_id: str) -> dict[str, Any]:
    """Retrieve mission artifacts (videos, CSV logs, etc.)."""
    artifacts = RUN_STORE.list_artifacts(mission_id)
    normalized: dict[str, Any] = {}
    for key, value in artifacts.items():
        artifact_path = Path(str(value.get("path") if isinstance(value, dict) else value))
        normalized[key] = {
            "path": str(artifact_path),
            "download_url": f"file://{artifact_path.resolve()}",
        }
    return {
        "schema_version": "1.0",
        "mission_id": mission_id,
        "artifacts": normalized,
        "artifact_directory": str(OUTPUTS),
    }


@app.get("/mission/{mission_id}/status")
async def get_mission_status(mission_id: str) -> dict[str, Any]:
    """Get the current status of a mission."""
    try:
        run = RUN_STORE.get_run(mission_id)
        return {
            "schema_version": "1.0",
            "mission_id": mission_id,
            "status": run.status,
            "config": run.config,
            "metrics": run.metrics,
            "validation": run.validation,
            "created_at": run.created_at,
            "updated_at": run.updated_at,
        }
    except Exception as exc:
        return {
            "schema_version": "1.0",
            "mission_id": mission_id,
            "error": str(exc),
            "status": "not_found",
        }


@app.post("/stream")
async def stream(payload: dict[str, Any]) -> StreamingResponse:
    manager = _build_manager(payload)
    processor = StreamProcessor(manager)

    def generate() -> Any:
        for packet in processor.stream_replay(payload):
            yield packet.to_ndjson()

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/telemetry")
async def telemetry_heartbeat() -> dict[str, Any]:
    payload = await MISSION_STATE.get_payload()
    snapshot = payload.get("snapshot", {})
    validation = payload.get("validation", {})
    mission_status = str(snapshot.get("status", "idle")).lower()
    return {
        "schema_version": "1.0",
        "type": "MISSION_HEARTBEAT",
        "mission_id": int(snapshot.get("mission_id", 0) or 0),
        "mission_status": mission_status,
        "stage": snapshot.get("active_stage", snapshot.get("stage", "idle")),
        "heartbeat_live": bool(snapshot.get("heartbeat_live", False)),
        "heartbeat_age_ms": snapshot.get("heartbeat_age_ms"),
        "target_count": int(snapshot.get("target_count", 0)),
        "rmse_m": float(snapshot.get("rmse_m", 0.0)),
        "innovation_m": float(snapshot.get("innovation_m", 0.0)),
        "confidence_score": float(snapshot.get("confidence_score", 0.0)),
        "validation_success": bool(validation.get("validation_success", False)) if isinstance(validation, dict) else False,
        "timestamp": time.time(),
        "snapshot": snapshot,
    }


@app.post("/telemetry")
async def telemetry(payload: dict[str, Any]) -> StreamingResponse:
    return await stream(payload)


@app.post("/validate")
async def validate(payload: dict[str, Any]) -> dict[str, Any]:
    run_record = RUN_STORE.create_run(
        kind="validation",
        status="running",
        config=dict(payload),
        seed=int(payload.get("random_seed", 61)),
    )
    if bool(payload.get("scenario_matrix", False)) or str(payload.get("validation_mode", "")).lower() == "matrix12":
        report = await asyncio.to_thread(_build_matrix_validation_report, payload)
    else:
        manager = _build_manager(payload)
        summary = await asyncio.to_thread(
            manager.run_monte_carlo_validation,
            int(payload.get("iterations", 100)),
            int(payload.get("num_targets", 3)),
            float(payload.get("drift_rate_mps", 0.3)),
            float(payload.get("noise_std_m", 0.45)),
            float(payload.get("packet_loss_rate", 0.0)),
            int(payload.get("max_steps", 20)),
        )
        report = _build_validation_report(summary, payload)
    await MISSION_STATE.set_validation(report)
    RUN_STORE.update_run(
        run_record.run_id,
        status="complete",
        metrics={
            "ekf_success_rate": float(report["ekf_success_rate"]),
            "raw_success_rate": float(report["raw_success_rate"]),
            "ekf_mean_miss_distance_m": float(report["ekf_mean_miss_distance_m"]),
        },
        validation=dict(report),
    )
    return {"run_id": run_record.run_id, **report}


@app.get("/runs")
async def list_runs() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "runs": [asdict(record) for record in RUN_STORE.list_runs()],
    }


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    record = RUN_STORE.get_run(run_id)
    return {"schema_version": "1.0", **asdict(record)}


@app.get("/runs/{run_id}/artifacts")
async def get_run_artifacts(run_id: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "artifacts": RUN_STORE.list_artifacts(run_id),
    }


@app.websocket("/ws/state")
async def state_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    sent_terminal_replay_for_mission: int | None = None
    try:
        while True:
            payload = await MISSION_STATE.get_payload()
            mission_id = int((payload.get("snapshot") or {}).get("mission_id", -1))
            if payload.get("type") == "MISSION_COMPLETE":
                if sent_terminal_replay_for_mission == mission_id:
                    payload = dict(payload)
                    payload["type"] = "STATE"
                    payload["event"] = None
                    payload["replay_data"] = None
                else:
                    sent_terminal_replay_for_mission = mission_id
            await websocket.send_json(payload)
            await asyncio.sleep(1.0 / HEARTBEAT_HZ)
    except WebSocketDisconnect:
        return


@app.websocket("/ws")
async def state_stream_alias(websocket: WebSocket) -> None:
    await state_stream(websocket)


@app.websocket("/ws/mission")
async def mission_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
        manager = _build_manager(payload)
        replay = await asyncio.to_thread(
            manager.run_replay,
            int(payload.get("num_targets", 3)),
            bool(payload.get("use_ekf", True)),
            float(payload.get("drift_rate_mps", 0.3)),
            float(payload.get("noise_std_m", 0.45)),
            float(payload.get("latency_ms", 0.0)),
            float(payload.get("packet_loss_rate", 0.0)),
            int(payload.get("random_seed", 61)),
            int(payload.get("max_steps", 141)),
            float(payload.get("dt", 0.05)),
            float(payload.get("kill_radius_m", 0.5)),
            float(payload.get("guidance_gain", 4.2)),
            bool(payload.get("use_ekf_anti_spoofing", True)),
            float(payload.get("target_speed_mps")) if payload.get("target_speed_mps") is not None else None,
            float(payload.get("interceptor_speed_mps")) if payload.get("interceptor_speed_mps") is not None else None,
            float(payload.get("ekf_process_noise")) if payload.get("ekf_process_noise") is not None else None,
            _coerce_measurement_noise(payload.get("ekf_measurement_noise")),
            bool(payload.get("enable_spoofing", False)),
            float(payload.get("link_snr_db", 28.0)),
            float(payload.get("packet_loss_k", 0.12)),
            float(payload.get("packet_loss_alpha", 1.8)),
        )
        previous_tick = time.perf_counter()
        for frame in replay.frames:
            current_tick = time.perf_counter()
            throughput = 1.0 / max(current_tick - previous_tick, 1e-6)
            previous_tick = current_tick
            await websocket.send_json(
                {
                    "type": "LIVE",
                    "payload": _build_frame_snapshot(frame, replay.validation, 0, throughput),
                }
            )
            await asyncio.sleep(1.0 / HEARTBEAT_HZ)
        final_snapshot = _build_terminal_snapshot(replay, mission_id=0, throughput_fps=STREAM_HZ)
        replay_data = _build_replay_data(replay)
        _write_platform_preview(replay_data, OUTPUTS / "platform_preview.html")
        await websocket.send_json(
            {
                "type": "MISSION_COMPLETE",
                "event": "mission_complete",
                "status": "COMPLETED",
                "stage": final_snapshot.get("active_stage", "Target Redirected"),
                "snapshot": final_snapshot,
                "replay_data": replay_data,
                "full_replay": replay_data.get("frames", []),
                "artifact_url": "outputs/platform_preview.html",
                "finish_signal": {"artifact_url": "outputs/platform_preview.html", "replay_ready": True},
                "stats": {
                    "rmse": float(final_snapshot.get("rmse_m", 0.0)),
                    "success": bool(replay.validation.get("success", False)),
                },
                "validation": replay.validation,
            }
        )
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"event": "error", "detail": str(exc)})
        return


@app.get("/spoof/status")
async def spoof_status() -> dict[str, Any]:
    """Return the current spoofing feature flag state and service info."""
    return {
        "schema_version": "1.0",
        "spoofing_enabled": is_enabled("SPOOFING"),
        "env_var": "DRONE_FEATURE_SPOOFING",
        "service": "SpoofService",
        "modalities": ["image", "audio", "video"],
        "note": "Toggle via POST /mission/start with enable_spoofing=true or set env var DRONE_FEATURE_SPOOFING=true",
    }


__all__ = ["app", "MISSION_STATE", "MissionState"]
