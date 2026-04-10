from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from drone_interceptor.simulation.airsim_manager import AirSimMissionManager, MissionFrame, MissionReplay


@dataclass(frozen=True, slots=True)
class TelemetryPacket:
    payload: dict[str, Any]

    def to_ndjson(self) -> bytes:
        return (json.dumps(self.payload) + "\n").encode("utf-8")


class StreamProcessor:
    """Consumes the mission manager and yields telemetry packets frame-by-frame."""

    def __init__(self, manager: AirSimMissionManager) -> None:
        self._manager = manager

    def stream_replay(self, payload: dict[str, Any]) -> Iterator[TelemetryPacket]:
        try:
            replay = self._manager.run_replay(
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
            )
        except Exception as exc:
            yield TelemetryPacket({"schema_version": "1.0", "status": "connection_lost", "detail": str(exc)})
            return

        previous_tick = time.perf_counter()
        previous_frame: MissionFrame | None = None
        for frame in replay.frames:
            current_tick = time.perf_counter()
            throughput_hz = 1.0 / max(current_tick - previous_tick, 1e-6)
            previous_tick = current_tick
            packet = _build_packet(frame, replay.validation, throughput_hz, previous_frame)
            previous_frame = frame
            yield TelemetryPacket(packet)

        yield TelemetryPacket({"schema_version": "1.0", "status": "complete", "validation": replay.validation})


def _build_packet(
    frame: MissionFrame,
    validation: dict[str, Any],
    throughput_hz: float,
    previous_frame: MissionFrame | None,
) -> dict[str, Any]:
    active_target = next((target for target in frame.targets if target.name == frame.active_target), None)
    if active_target is None and frame.targets:
        active_target = frame.targets[0]

    if active_target is None:
        return {
            "schema_version": "1.0",
            "status": "running",
            "step": frame.step,
            "throughput_hz": throughput_hz,
            "ekf_rmse": float(frame.rmse_m),
            "kill_prob": 0.0,
        }

    relative_position = np.asarray(active_target.position, dtype=float) - np.asarray(frame.interceptor_position, dtype=float)
    relative_velocity = np.asarray(active_target.velocity, dtype=float) - np.asarray(frame.interceptor_velocity, dtype=float)
    distance_m = max(float(np.linalg.norm(relative_position)), 1e-6)
    los_rate = float(np.linalg.norm(np.cross(relative_position, relative_velocity)) / (distance_m**2))
    closing_velocity = max(-float(np.dot(relative_velocity, relative_position / distance_m)), 0.0)
    uncertainty_m = max(float(active_target.uncertainty_radius_m), 0.25)
    kill_prob = _gaussian_kill_probability(distance_m, uncertainty_m)
    innovation = float(np.linalg.norm(np.asarray(active_target.raw_measurement) - np.asarray(active_target.filtered_estimate)))

    return {
        "schema_version": "1.0",
        "status": "running",
        "step": int(frame.step),
        "time_s": float(frame.time_s),
        "active_stage": frame.active_stage,
        "active_target": frame.active_target,
        "target_count": len(frame.targets),
        "throughput_hz": float(throughput_hz),
        "detection_fps": float(frame.detection_fps),
        "ekf_rmse": float(frame.rmse_m),
        "kill_prob": float(kill_prob),
        "relative_distance_m": float(distance_m),
        "closing_velocity_mps": float(closing_velocity),
        "los_rate_rps": float(los_rate),
        "innovation_m": innovation,
        "uncertainty_radius_m": uncertainty_m,
        "ekf_lock": bool(validation.get("ekf_enabled", False)),
        "spoofing_detected": bool(active_target.spoofing_detected or active_target.packet_dropped),
        "interceptor_position": frame.interceptor_position.tolist(),
        "interceptor_velocity": frame.interceptor_velocity.tolist(),
        "targets": [
            {
                "name": target.name,
                "true_position": target.position.tolist(),
                "spoofed_position": target.raw_measurement.tolist(),
                "estimated_position": target.filtered_estimate.tolist(),
                "uncertainty_radius_m": float(target.uncertainty_radius_m),
                "innovation_m": float(target.innovation_m),
                "innovation_gate": float(target.innovation_gate),
                "spoofing_detected": bool(target.spoofing_detected),
                "packet_dropped": bool(target.packet_dropped),
                "jammed": bool(target.jammed),
            }
            for target in frame.targets
        ],
    }


def _gaussian_kill_probability(distance_m: float, sigma_m: float) -> float:
    sigma_m = max(float(sigma_m), 0.25)
    return float(math.exp(-0.5 * (float(distance_m) / sigma_m) ** 2))


__all__ = ["StreamProcessor", "TelemetryPacket"]
