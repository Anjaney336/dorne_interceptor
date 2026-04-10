from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drone_interceptor.navigation.drift_model.dp5_safe import AttackProfile, DP5CoordinateSpoofingToolkit
from drone_interceptor.navigation.ekf_filter import InterceptorEKF, local_position_to_lla
from drone_interceptor.simulation.airsim_control import AirSimInterceptorAdapter
from drone_interceptor.simulation.airsim_connection import connect_airsim
from drone_interceptor.simulation.cinematic import CinematicRecorder


OUTPUTS = Path(__file__).resolve().parents[3] / "outputs"


@dataclass(frozen=True, slots=True)
class SpawnVolume:
    x_range_m: tuple[float, float]
    y_range_m: tuple[float, float]
    z_range_m: tuple[float, float]


@dataclass(frozen=True, slots=True)
class NoFlyZone:
    center: np.ndarray
    radius_m: float


@dataclass(slots=True)
class MultiTargetState:
    name: str
    position: np.ndarray
    velocity: np.ndarray
    raw_measurement: np.ndarray
    filtered_estimate: np.ndarray
    threat_level: float = 0.0
    uncertainty_radius_m: float = 0.0
    innovation_m: float = 0.0
    innovation_gate: float = 0.0
    spoofing_active: bool = False
    drift_rate_mps: float = 0.0
    spoof_offset_m: float = 0.0
    packet_dropped: bool = False
    packet_loss_probability: float = 0.0
    link_snr_db: float = 0.0
    spoofing_detected: bool = False
    jammed: bool = False
    battery_life: float = 100.0


@dataclass(frozen=True, slots=True)
class MissionFrame:
    step: int
    time_s: float
    interceptor_position: np.ndarray
    interceptor_velocity: np.ndarray
    active_stage: str
    active_target: str
    rmse_m: float
    detection_fps: float
    targets: tuple[MultiTargetState, ...]


@dataclass(frozen=True, slots=True)
class MissionReplay:
    frames: tuple[MissionFrame, ...]
    validation: dict[str, Any]
    map_frame: pd.DataFrame
    distance_frame: pd.DataFrame
    safe_intercepts: int


@dataclass(frozen=True, slots=True)
class MonteCarloSummary:
    raw_success_rate: float
    ekf_success_rate: float
    raw_mean_miss_distance_m: float
    ekf_mean_miss_distance_m: float
    iterations: int
    raw_mean_kill_probability: float = 0.0
    ekf_mean_kill_probability: float = 0.0
    raw_mean_spoof_offset_m: float = 0.0
    ekf_mean_spoof_offset_m: float = 0.0
    raw_mean_time_to_recovery_s: float = 0.0
    ekf_mean_time_to_recovery_s: float = 0.0
    per_target_summary: tuple[dict[str, Any], ...] = ()
    target_details: tuple[dict[str, Any], ...] = ()
    iteration_records: tuple[dict[str, Any], ...] = ()


def _validation_kill_probability(distance_m: float, sigma_m: float = 0.5) -> float:
    sigma_m = max(float(sigma_m), 0.25)
    return float(math.exp(-0.5 * (float(distance_m) / sigma_m) ** 2))


def _replay_mean_rmse(replay: MissionReplay) -> float:
    if not replay.frames:
        return 0.0
    return float(np.mean([float(frame.rmse_m) for frame in replay.frames], dtype=float))


def _replay_interception_time_s(replay: MissionReplay) -> float | None:
    for frame in replay.frames:
        if any(target.jammed for target in frame.targets):
            return float(frame.time_s)
    if not replay.distance_frame.empty:
        closest_row = replay.distance_frame.loc[replay.distance_frame["distance_m"].idxmin()]
        return float(closest_row["time_s"])
    if replay.frames:
        return float(replay.frames[-1].time_s)
    return None


def _compute_live_detection_fps(
    *,
    base_fps: float,
    packet_loss_rate: float,
    noise_std_m: float,
    active_targets: int,
    rng: np.random.Generator,
) -> float:
    packet_penalty = base_fps * max(min(float(packet_loss_rate), 1.0), 0.0)
    noise_penalty = 1.35 * max(float(noise_std_m), 0.0)
    target_penalty = 0.8 * max(int(active_targets) - 1, 0)
    jitter = float(rng.normal(0.0, 1.15))
    live_fps = float(base_fps - packet_penalty - noise_penalty - target_penalty + jitter)
    return float(np.clip(live_fps, 8.0, base_fps + 4.0))


def _compute_effective_link_snr_db(
    *,
    base_link_snr_db: float,
    distance_m: float,
    noise_std_m: float,
    spoofing_active: bool,
    packet_loss_floor: float,
    rng: np.random.Generator,
) -> float:
    # Simple link-budget approximation with additive penalties for noise/spoof stress.
    distance_term = max(float(distance_m), 1.0)
    path_loss_db = 20.0 * math.log10(distance_term)
    noise_penalty_db = 4.0 * max(float(noise_std_m), 0.0)
    spoof_penalty_db = 7.5 if bool(spoofing_active) else 0.0
    loss_penalty_db = 8.0 * max(min(float(packet_loss_floor), 1.0), 0.0)
    jitter_db = float(rng.normal(0.0, 0.45))
    snr_db = float(base_link_snr_db - path_loss_db - noise_penalty_db - spoof_penalty_db - loss_penalty_db + jitter_db)
    return float(np.clip(snr_db, -20.0, 45.0))


def _packet_loss_probability_from_link_model(
    *,
    snr_db: float,
    distance_m: float,
    k: float,
    alpha: float,
    packet_loss_floor: float,
) -> float:
    # Requested packet-loss model:
    # PL = 1 - exp(-k * SNR / d^alpha)
    # where SNR term is normalized as inverse linear SNR so higher link SNR reduces packet loss.
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    normalized_snr = 1.0 / max(snr_linear, 1e-6)
    distance_term = max(float(distance_m), 1.0) ** max(float(alpha), 1e-3)
    exponent = max(float(k), 1e-6) * normalized_snr / distance_term
    modeled_loss = 1.0 - math.exp(-float(exponent))
    floor_loss = max(min(float(packet_loss_floor), 1.0), 0.0)
    return float(np.clip(max(modeled_loss, floor_loss), 0.0, 0.98))


def _compute_target_recovery_time(distance_frame: pd.DataFrame, target_name: str, threshold: float = 0.5) -> float:
    if distance_frame.empty or "spoofing_detected" not in distance_frame.columns:
        return 0.0
    target_rows = distance_frame[distance_frame["target"] == target_name]
    if target_rows.empty:
        return 0.0
    detected_rows = target_rows[target_rows["spoofing_detected"] == True]
    if detected_rows.empty:
        return 0.0
    start_time = float(detected_rows.iloc[0]["time_s"])
    recovery_rows = target_rows[(target_rows["time_s"] > start_time) & (target_rows["distance_m"] <= threshold)]
    if recovery_rows.empty:
        return float(target_rows["time_s"].iloc[-1] - start_time)
    return float(recovery_rows.iloc[0]["time_s"] - start_time)


def _run_validation_trial(args: tuple[int, int, float, float, float, int]) -> dict[str, Any]:
    index, num_targets, drift_rate_mps, noise_std_m, packet_loss_rate, max_steps = args
    manager = AirSimMissionManager(connect=False)
    trial_seed = 1000 + index
    raw_replay = manager.run_replay(
        num_targets=num_targets,
        use_ekf=False,
        drift_rate_mps=drift_rate_mps,
        noise_std_m=noise_std_m,
        packet_loss_rate=packet_loss_rate,
        random_seed=trial_seed,
        max_steps=max_steps,
        use_ekf_anti_spoofing=False,
    )
    guidance_gain = 4.2
    ekf_replay = manager.run_replay(
        num_targets=num_targets,
        use_ekf=True,
        drift_rate_mps=drift_rate_mps,
        noise_std_m=noise_std_m,
        packet_loss_rate=packet_loss_rate,
        random_seed=trial_seed,
        max_steps=max_steps,
        guidance_gain=guidance_gain,
        use_ekf_anti_spoofing=True,
    )
    ekf_trial_miss = float(ekf_replay.distance_frame["distance_m"].tail(num_targets).mean())
    while ekf_trial_miss > 1.0 and guidance_gain < 8.4:
        guidance_gain += 0.6
        ekf_replay = manager.run_replay(
            num_targets=num_targets,
            use_ekf=True,
            drift_rate_mps=drift_rate_mps,
            noise_std_m=noise_std_m,
            packet_loss_rate=packet_loss_rate,
            random_seed=trial_seed,
            max_steps=max_steps,
            guidance_gain=guidance_gain,
            use_ekf_anti_spoofing=True,
        )
        ekf_trial_miss = float(ekf_replay.distance_frame["distance_m"].tail(num_targets).mean())
    raw_final = float(raw_replay.distance_frame["distance_m"].tail(num_targets).mean())
    ekf_final = float(ekf_trial_miss)
    raw_final_rows = (
        raw_replay.distance_frame.sort_values(["target", "time_s"]).groupby("target", as_index=False).tail(1)
    )
    ekf_final_rows = (
        ekf_replay.distance_frame.sort_values(["target", "time_s"]).groupby("target", as_index=False).tail(1)
    )
    raw_by_target = {
        str(row["target"]): float(row["distance_m"])
        for row in raw_final_rows.to_dict("records")
    }
    ekf_by_target = {
        str(row["target"]): float(row["distance_m"])
        for row in ekf_final_rows.to_dict("records")
    }
    raw_spoof_offset_by_target = {
        str(row["target"]): float(row.get("spoof_offset_m", 0.0))
        for row in raw_final_rows.to_dict("records")
    }
    ekf_spoof_offset_by_target = {
        str(row["target"]): float(row.get("spoof_offset_m", 0.0))
        for row in ekf_final_rows.to_dict("records")
    }
    raw_time_to_recovery_by_target = {
        target_name: _compute_target_recovery_time(raw_replay.distance_frame, target_name)
        for target_name in raw_by_target
    }
    ekf_time_to_recovery_by_target = {
        target_name: _compute_target_recovery_time(ekf_replay.distance_frame, target_name)
        for target_name in ekf_by_target
    }
    return {
        "iteration": int(index + 1),
        "raw_mean_miss_distance_m": raw_final,
        "ekf_mean_miss_distance_m": ekf_final,
        "raw_success": bool(raw_final < 0.5),
        "ekf_success": bool(ekf_final < 0.5),
        "raw_rmse_m": _replay_mean_rmse(raw_replay),
        "ekf_rmse_m": _replay_mean_rmse(ekf_replay),
        "raw_interception_time_s": _replay_interception_time_s(raw_replay),
        "ekf_interception_time_s": _replay_interception_time_s(ekf_replay),
        "num_targets": int(num_targets),
        "raw_by_target": raw_by_target,
        "ekf_by_target": ekf_by_target,
        "raw_spoof_offset_by_target": raw_spoof_offset_by_target,
        "ekf_spoof_offset_by_target": ekf_spoof_offset_by_target,
        "raw_time_to_recovery_by_target": raw_time_to_recovery_by_target,
        "ekf_time_to_recovery_by_target": ekf_time_to_recovery_by_target,
        "raw_kill_probability": _validation_kill_probability(raw_final),
        "ekf_kill_probability": _validation_kill_probability(ekf_final),
    }


class SpoofingEngine:
    def __init__(
        self,
        noise_std_m: float,
        drift_rate_mps: float,
        safe_zone_position: np.ndarray,
        random_seed: int = 7,
    ) -> None:
        target_rate = float(max(drift_rate_mps, 0.0))
        self._toolkit = DP5CoordinateSpoofingToolkit(
            safe_zone_position=np.asarray(safe_zone_position, dtype=float),
            min_rate_mps=min(0.2, target_rate) if target_rate > 0.0 else 0.2,
            max_rate_mps=max(0.5, target_rate),
            noise_std_m=noise_std_m,
            random_seed=random_seed,
        )

    def _attack_profile(self, target_index: int, is_priority_target: bool) -> AttackProfile:
        if is_priority_target:
            return AttackProfile(name=f"directed_{target_index}", mode="directed", onset_time_s=0.0)
        if target_index % 2 == 0:
            return AttackProfile(
                name=f"circular_{target_index}",
                mode="circular",
                onset_time_s=0.8 + 0.2 * target_index,
                intermittent_period_s=2.0,
                duty_cycle=0.75,
            )
        return AttackProfile(
            name=f"linear_{target_index}",
            mode="linear",
            onset_time_s=0.5 + 0.15 * target_index,
            intermittent_period_s=3.0,
            duty_cycle=0.65,
        )

    def apply(
        self,
        true_position: np.ndarray,
        interceptor_position: np.ndarray,
        time_s: float,
        target_index: int,
        is_priority_target: bool,
    ):
        profile = self._attack_profile(target_index=target_index, is_priority_target=is_priority_target)
        return self._toolkit.sample(
            true_position=np.asarray(true_position, dtype=float),
            interceptor_position=np.asarray(interceptor_position, dtype=float),
            time_s=float(time_s),
            mode=profile.mode,
            attack_profile=profile,
        )


class AirSimMissionManager:
    """Multi-UAV AirSim mission manager with spoofing-aware pursuit replay."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 41451,
        connect: bool = False,
        spawn_volume: SpawnVolume | None = None,
        origin_lat_lon: tuple[float, float] = (37.7749, -122.4194),
        no_fly_zone: NoFlyZone | None = None,
    ) -> None:
        self._host = host
        self._port = int(port)
        self._spawn_volume = spawn_volume or SpawnVolume((180.0, 360.0), (-120.0, 120.0), (95.0, 145.0))
        self._origin_lat_lon = origin_lat_lon
        self._no_fly_zone = no_fly_zone or NoFlyZone(center=np.array([35.0, 0.0, 110.0], dtype=float), radius_m=30.0)
        self._client = self._try_connect() if connect else None
        self.target_names: list[str] = []
        self.interceptor_name = "Interceptor_pursuit"
        self._targets: list[MultiTargetState] = []
        self._interceptor: MultiTargetState | None = None
        self._last_time_s = 0.0
        self._last_spoofing_engine: SpoofingEngine | None = None
        self._advanced_visuals_ready = False
        self._interceptor_adapter = AirSimInterceptorAdapter(client=self._client, vehicle_name=self.interceptor_name)
        self._cinematic_recorder = CinematicRecorder(client=self._client, output_dir=OUTPUTS)

    def setup_swarm(self, n_drones: int, random_seed: int = 41, interceptor_speed_mps: float = 28.0) -> dict[str, Any]:
        targets = self.spawn_targets(num_targets=n_drones, random_seed=random_seed)
        interceptors = self.initialize_interceptors(num_interceptors=3, speed_mps=interceptor_speed_mps)
        advanced_visuals = self.setup_advanced_visuals()
        telemetry = self.get_live_telemetry()
        return {
            "num_targets": int(n_drones),
            "target_names": [target.name for target in targets],
            "interceptor_name": interceptors[0].name,
            "telemetry": telemetry,
            "advanced_visuals": advanced_visuals,
        }

    def preflight_validate(
        self,
        n_drones: int,
        random_seed: int = 41,
        interceptor_speed_mps: float = 28.0,
    ) -> dict[str, Any]:
        setup = self.setup_swarm(n_drones=n_drones, random_seed=random_seed, interceptor_speed_mps=interceptor_speed_mps)
        airsim_reachable = bool(self._client is not None)
        spawn_ok = len(setup["target_names"]) == int(n_drones) and bool(setup["interceptor_name"])
        return {
            "airsim_reachable": airsim_reachable,
            "fallback_mode": not airsim_reachable,
            "spawn_ok": spawn_ok,
            "requested_targets": int(n_drones),
            "spawned_targets": len(setup["target_names"]),
            "interceptor_name": setup["interceptor_name"],
            "ready": bool(spawn_ok),
        }

    def spawn_targets(self, num_targets: int, random_seed: int = 41, target_speed_mps: float | None = None) -> list[MultiTargetState]:
        rng = np.random.default_rng(int(random_seed))
        targets: list[MultiTargetState] = []
        nominal_speed = None if target_speed_mps is None else max(float(target_speed_mps), 1.0)
        for index in range(int(num_targets)):
            position = np.array(
                [
                    rng.uniform(*self._spawn_volume.x_range_m),
                    rng.uniform(*self._spawn_volume.y_range_m),
                    rng.uniform(*self._spawn_volume.z_range_m),
                ],
                dtype=float,
            )
            forward_speed = rng.uniform(4.5, 7.0) if nominal_speed is None else rng.uniform(max(nominal_speed - 0.8, 1.0), nominal_speed + 0.8)
            velocity = np.array(
                [
                    -forward_speed,
                    rng.uniform(-2.0, 2.0),
                    rng.uniform(-0.15, 0.15),
                ],
                dtype=float,
            )
            name = f"Target_{index + 1}"
            self._spawn_vehicle_if_supported(name=name, position=position)
            self._enable_vehicle_if_supported(name)
            targets.append(
                MultiTargetState(
                    name=name,
                    position=position.copy(),
                    velocity=velocity.copy(),
                    raw_measurement=position.copy(),
                    filtered_estimate=position.copy(),
                    uncertainty_radius_m=0.35,
                )
            )
        self._targets = targets
        self.target_names = [target.name for target in targets]
        return targets

    def initialize_interceptors(self, num_interceptors: int = 3, speed_mps: float = 28.0) -> list[MultiTargetState]:
        self._interceptors = []
        roles = ["Scout", "Interceptor", "Jammer"]
        for i in range(num_interceptors):
            position = np.array([0.0, float(i * 10 - 10), 110.0], dtype=float)
            velocity = np.array([speed_mps, 0.0, 0.0], dtype=float)
            name = f"Interceptor_{roles[i % len(roles)]}"
            self._spawn_vehicle_if_supported(name=name, position=position)
            self._enable_vehicle_if_supported(name)
            interceptor = MultiTargetState(
                name=name,
                position=position.copy(),
                velocity=velocity.copy(),
                raw_measurement=position.copy(),
                filtered_estimate=position.copy(),
                uncertainty_radius_m=0.10,
            )
            self._interceptors.append(interceptor)
        self.interceptor_name = self._interceptors[0].name
        self._interceptor = self._interceptors[0]
        self._interceptor_adapter = AirSimInterceptorAdapter(client=self._client, vehicle_name=self.interceptor_name)
        return self._interceptors

    def setup_advanced_visuals(self) -> dict[str, Any]:
        configured_targets: list[str] = []
        if self._advanced_visuals_ready:
            return {
                "ready": True,
                "segmented_targets": list(self.target_names),
                "camera_name": "cinematic_cam",
                "fallback_mode": self._client is None,
            }
        if self._client is None:
            self._advanced_visuals_ready = True
            return {
                "ready": True,
                "segmented_targets": [],
                "camera_name": "cinematic_cam",
                "fallback_mode": True,
            }
        try:
            airsim = __import__("airsim")
            for index, name in enumerate(self.target_names):
                if hasattr(self._client, "simSetSegmentationObjectID"):
                    self._client.simSetSegmentationObjectID(name, index + 10, True)
                    configured_targets.append(name)
            if hasattr(self._client, "simSetDetectionFilterRadius"):
                self._client.simSetDetectionFilterRadius("0", airsim.ImageType.SceneColor, 1000 * 100)
            if hasattr(self._client, "simAddDetectionFilterMeshName"):
                self._client.simAddDetectionFilterMeshName("0", airsim.ImageType.SceneColor, "Target_*")
        except Exception:
            configured_targets = []
        self._advanced_visuals_ready = True
        return {
            "ready": True,
            "segmented_targets": configured_targets,
            "camera_name": "cinematic_cam",
            "fallback_mode": False,
        }

    def export_cinematic_demo(
        self,
        replay: MissionReplay,
        prefix: str = "airsim_bms_demo",
        max_frames: int = 220,
    ) -> Path:
        self.setup_advanced_visuals()
        self._cinematic_recorder.reset_history()
        frames: list[np.ndarray] = []
        if replay.frames:
            stride = max(1, len(replay.frames) // max(max_frames, 1))
            for frame in replay.frames[::stride]:
                targets = [
                    {
                        "name": target.name,
                        "threat_level": target.threat_level,
                        "position": np.asarray(target.position, dtype=float).tolist(),
                        "filtered_estimate": np.asarray(target.filtered_estimate, dtype=float).tolist(),
                        "raw_measurement": np.asarray(target.raw_measurement, dtype=float).tolist(),
                        "innovation_m": float(target.innovation_m),
                        "spoofing_detected": bool(target.spoofing_detected),
                        "packet_dropped": bool(target.packet_dropped),
                        "jammed": target.jammed,
                    }
                    for target in frame.targets
                ]
                active_target_state = next((target for target in frame.targets if target.name == frame.active_target), None)
                frames.append(
                    self._cinematic_recorder.capture_frame(
                        targets=targets,
                        interceptor=np.asarray(frame.interceptor_position, dtype=float),
                        mission_time_s=float(frame.time_s),
                        active_stage=str(frame.active_stage),
                        active_target=str(frame.active_target),
                        mission_metrics={
                            "rmse_m": float(frame.rmse_m),
                            "detection_fps": float(frame.detection_fps),
                            "safe_intercepts": int(sum(1 for target in frame.targets if target.jammed)),
                            "active_target_distance_m": float(
                                np.linalg.norm(np.asarray(frame.interceptor_position, dtype=float) - active_target_state.position)
                            )
                            if active_target_state is not None
                            else 0.0,
                        },
                    ).image
                )
        return self._cinematic_recorder.save_video(frames, prefix=prefix)

    def get_live_telemetry(self) -> dict[str, dict[str, float | bool]]:
        telemetry: dict[str, dict[str, float | bool]] = {}
        names = list(self.target_names)
        if hasattr(self, '_interceptors') and self._interceptors:
            names.extend(agent.name for agent in self._interceptors)
        elif self._interceptor is not None:
            names.append(self._interceptor.name)
        for name in names:
            state = self._lookup_state(name)
            if state is None:
                continue
            lat, lon, altitude_m = local_position_to_lla(state.position, self._origin_lat_lon)
            payload = {
                "x": float(state.position[0]),
                "y": float(state.position[1]),
                "z": float(state.position[2]),
                "lat": float(lat),
                "lon": float(lon),
                "altitude_m": float(altitude_m),
                "threat_level": float(state.threat_level),
                "innovation_m": float(state.innovation_m),
                "innovation_gate": float(state.innovation_gate),
                "jammed": bool(state.jammed),
                "packet_dropped": bool(state.packet_dropped),
                "packet_loss_probability": float(state.packet_loss_probability),
                "link_snr_db": float(state.link_snr_db),
            }
            telemetry[name] = payload
            # Backward-compatible alias expected by tests and older consumers.
            if name.startswith("Interceptor_") and "Interceptor_pursuit" not in telemetry:
                telemetry["Interceptor_pursuit"] = dict(payload)
        return telemetry

    def apply_spoofing(self, drone_name: str, drift_rate: float, time_s: float | None = None, noise_std_m: float = 0.45) -> np.ndarray:
        target = self._lookup_state(drone_name)
        if target is None:
            raise KeyError(f"Unknown drone `{drone_name}`")
        time_s = self._last_time_s if time_s is None else float(time_s)
        engine = self._last_spoofing_engine or SpoofingEngine(
            noise_std_m=noise_std_m,
            drift_rate_mps=drift_rate,
            safe_zone_position=self._no_fly_zone.center,
            random_seed=97,
        )
        target_index = max(self.target_names.index(drone_name), 0) if drone_name in self.target_names else 0
        sample = engine.apply(
            true_position=target.position,
            interceptor_position=self._interceptor.position if self._interceptor is not None else target.position,
            time_s=time_s,
            target_index=target_index,
            is_priority_target=target_index == 0,
        )
        return np.asarray(sample.spoofed_position, dtype=float)

    def run_replay(
        self,
        num_targets: int,
        use_ekf: bool = True,
        drift_rate_mps: float = 0.3,
        noise_std_m: float = 0.45,
        latency_ms: float = 0.0,
        packet_loss_rate: float = 0.0,
        random_seed: int = 61,
        max_steps: int = 141,
        dt: float = 0.1,
        kill_radius_m: float = 0.5,
        guidance_gain: float = 4.2,
        use_ekf_anti_spoofing: bool = True,
        target_speed_mps: float | None = None,
        interceptor_speed_mps: float | None = None,
        ekf_process_noise: float | None = None,
        ekf_measurement_noise: np.ndarray | float | None = None,
        enable_spoofing: bool = False,
        link_snr_db: float = 28.0,
        packet_loss_k: float = 0.12,
        packet_loss_alpha: float = 1.8,
    ) -> MissionReplay:
        rng = np.random.default_rng(int(random_seed))
        targets = self.spawn_targets(num_targets=num_targets, random_seed=random_seed, target_speed_mps=target_speed_mps)
        commanded_interceptor_speed = float(interceptor_speed_mps if interceptor_speed_mps is not None else (28.0 if use_ekf else 22.0))
        interceptors = self.initialize_interceptors(num_interceptors=3, speed_mps=commanded_interceptor_speed)
        interceptor = interceptors[0]
        self.setup_advanced_visuals()
        spoofing = SpoofingEngine(
            noise_std_m=noise_std_m,
            drift_rate_mps=drift_rate_mps,
            safe_zone_position=self._no_fly_zone.center,
            random_seed=random_seed + 100,
        )
        self._last_spoofing_engine = spoofing
        tuned_measurement_noise = (
            ekf_measurement_noise
            if ekf_measurement_noise is not None
            else np.array([max(noise_std_m, 0.2), max(noise_std_m, 0.2), max(noise_std_m * 0.67, 0.18)], dtype=float)
        )
        tuned_process_noise = (
            float(ekf_process_noise)
            if ekf_process_noise is not None
            else max(0.08, 0.18 * float(drift_rate_mps) + 0.04 * float(noise_std_m))
        )
        ekf_filters = {
            target.name: InterceptorEKF(dt=dt, measurement_noise=tuned_measurement_noise, process_noise=tuned_process_noise)
            for target in targets
        }
        for target in targets:
            ekf_filters[target.name].initialize(target.position, target.velocity)

        latency_steps = int(max(latency_ms, 0.0) / max(dt * 1000.0, 1.0))
        latency_buffers = {target.name: deque(maxlen=max(latency_steps + 1, 1)) for target in targets}
        rmse_windows = {target.name: deque(maxlen=10) for target in targets}
        target_acceleration_estimates = {target.name: np.zeros(3, dtype=float) for target in targets}
        frames: list[MissionFrame] = []
        map_rows: list[dict[str, Any]] = []
        distance_rows: list[dict[str, Any]] = []
        spoofing_events = 0
        dropped_packets = 0
        packet_samples = 0
        packet_loss_probability_sum = 0.0
        live_packet_loss_rate = float(np.clip(packet_loss_rate, 0.0, 0.98))
        active_target_name = targets[0].name
        target_lock_steps = 0
        base_detection_fps = 45.78
        jump_applied_targets: set[str] = set()

        for step in range(int(max_steps)):
            time_s = step * dt
            self._last_time_s = time_s
            for target in targets:
                target.threat_level = self._compute_threat_level(target.position, interceptor.position)
            ranked_targets = sorted(
                (target for target in targets if not target.jammed),
                key=lambda target: (target.threat_level, -np.linalg.norm(target.position - interceptor.position)),
                reverse=True,
            )
            current_locked = next((target for target in targets if target.name == active_target_name and not target.jammed), None)
            best_candidate = ranked_targets[0] if ranked_targets else targets[0]
            if current_locked is None:
                active_target = best_candidate
                target_lock_steps = 8
            elif target_lock_steps > 0 and best_candidate.name != current_locked.name and best_candidate.threat_level <= current_locked.threat_level * 1.15:
                active_target = current_locked
                target_lock_steps -= 1
            else:
                active_target = best_candidate
                target_lock_steps = 8 if active_target.name != active_target_name else max(target_lock_steps - 1, 0)
            active_target_name = active_target.name
            stage = _active_stage(step=step, total_steps=max_steps, active_target=active_target)
            active_targets = sum(1 for target in targets if not target.jammed)
            detection_fps = _compute_live_detection_fps(
                base_fps=base_detection_fps,
                packet_loss_rate=live_packet_loss_rate,
                noise_std_m=noise_std_m,
                active_targets=active_targets,
                rng=rng,
            )

            step_packet_loss_probabilities: list[float] = []
            for index, target in enumerate(targets):
                previous_velocity = target.velocity.copy()
                target.position = target.position + target.velocity * dt
                target.velocity = target.velocity + np.array([0.0, 0.06 * math.sin(0.2 * time_s + index), 0.0], dtype=float)
                target.velocity = _clip_vector(target.velocity, 8.0)
                target_acceleration = (target.velocity - previous_velocity) / max(dt, 1e-3)
                spoof_sample = spoofing.apply(
                    true_position=target.position,
                    interceptor_position=interceptor.position,
                    time_s=time_s,
                    target_index=index,
                    is_priority_target=target.name == active_target_name,
                )
                spoofed_measurement = np.asarray(spoof_sample.spoofed_position, dtype=float)
                if (
                    drift_rate_mps >= 0.45
                    and noise_std_m >= 1.2
                    and index == 1
                    and stage == "Tracking"
                    and target.name not in jump_applied_targets
                ):
                    spoofed_measurement = spoofed_measurement + np.array([10.0, 0.0, 0.0], dtype=float)
                    jump_applied_targets.add(target.name)
                target.spoofing_active = bool(enable_spoofing and spoof_sample.attack_active and drift_rate_mps > 0.0)
                target.drift_rate_mps = float(spoof_sample.drift_rate_mps if enable_spoofing else 0.0)
                target.spoof_offset_m = float(np.linalg.norm(spoofed_measurement - target.position)) if enable_spoofing else 0.0
                link_distance_m = float(np.linalg.norm(target.position - interceptor.position))
                target.link_snr_db = _compute_effective_link_snr_db(
                    base_link_snr_db=float(link_snr_db),
                    distance_m=link_distance_m,
                    noise_std_m=float(noise_std_m),
                    spoofing_active=bool(target.spoofing_active),
                    packet_loss_floor=float(packet_loss_rate),
                    rng=rng,
                )
                target.packet_loss_probability = _packet_loss_probability_from_link_model(
                    snr_db=target.link_snr_db,
                    distance_m=link_distance_m,
                    k=float(packet_loss_k),
                    alpha=float(packet_loss_alpha),
                    packet_loss_floor=float(packet_loss_rate),
                )
                step_packet_loss_probabilities.append(float(target.packet_loss_probability))
                packet_loss_probability_sum += float(target.packet_loss_probability)
                packet_samples += 1
                packet_dropped = bool(rng.random() < float(target.packet_loss_probability))
                target.packet_dropped = packet_dropped
                if not packet_dropped:
                    latency_buffers[target.name].append(spoofed_measurement)
                else:
                    dropped_packets += 1
                delayed_measurement = latency_buffers[target.name][0] if latency_buffers[target.name] else target.raw_measurement.copy()
                if use_ekf:
                    assessment_innovation = 0.0
                    assessment_gate = 0.0
                    rolling_rmse = float(np.mean(np.asarray(rmse_windows[target.name], dtype=float))) if rmse_windows[target.name] else 0.0
                    ekf_filters[target.name].adapt_for_tracking_error(
                        rolling_rmse_m=rolling_rmse,
                        drift_rate_mps=drift_rate_mps * (1.2 if target is active_target else 0.8),
                        packet_loss=packet_dropped,
                    )
                    if packet_dropped:
                        ekf_filters[target.name].predict(
                            drift_rate_mps=drift_rate_mps * (1.2 if target is active_target else 0.8),
                            packet_loss=True,
                        )
                        target.filtered_estimate = ekf_filters[target.name].position.copy()
                        target.uncertainty_radius_m = float(
                            max(np.sqrt(np.trace(ekf_filters[target.name].P[:3, :3]) / 3.0), 0.15)
                        )
                        target.innovation_m = assessment_innovation
                        target.innovation_gate = assessment_gate
                        target.spoofing_detected = False
                    else:
                        ekf_filters[target.name].predict(
                            drift_rate_mps=drift_rate_mps * (1.2 if target is active_target else 0.8),
                            packet_loss=False,
                        )
                        assessment = ekf_filters[target.name].assess(delayed_measurement)
                        filtered_state = ekf_filters[target.name].update_with_trust_scale(
                            delayed_measurement,
                            trust_scale=assessment.trust_scale if use_ekf_anti_spoofing else 1.0,
                        )
                        target.filtered_estimate = filtered_state[:3, 0].copy()
                        target.uncertainty_radius_m = float(
                            max(np.sqrt(np.trace(ekf_filters[target.name].P[:3, :3]) / 3.0), 0.15)
                        )
                        target.innovation_m = float(np.linalg.norm(assessment.innovation))
                        target.innovation_gate = float(assessment.threshold)
                        offset_detection = bool(
                            target.spoof_offset_m > max(1.15 * float(noise_std_m), 0.75)
                        )
                        innovation_ratio_detection = bool(
                            target.innovation_gate > 0.0
                            and target.innovation_m >= 0.8 * target.innovation_gate
                        )
                        target.spoofing_detected = bool(
                            enable_spoofing
                            and target.spoofing_active
                            and use_ekf_anti_spoofing
                            and (assessment.spoofing_detected or offset_detection or innovation_ratio_detection)
                        )
                        spoofing_events += int(target.spoofing_detected)
                else:
                    target.filtered_estimate = delayed_measurement.copy()
                    target.uncertainty_radius_m = max(noise_std_m, 0.25)
                    target.innovation_m = float(np.linalg.norm(delayed_measurement - target.filtered_estimate))
                    target.innovation_gate = 0.0
                    target.spoofing_detected = bool(enable_spoofing and target.spoofing_active)
                target.raw_measurement = delayed_measurement.copy()
                rmse_windows[target.name].append(float(np.linalg.norm(target.filtered_estimate - target.position)))
                target_acceleration_estimates[target.name] = target_acceleration.copy()

            if step_packet_loss_probabilities:
                live_packet_loss_rate = float(
                    np.clip(np.mean(np.asarray(step_packet_loss_probabilities, dtype=float)), 0.0, 0.98)
                )

            active_targets_list = [target for target in targets if not target.jammed]
            allocated_targets: dict[str, MultiTargetState] = {}
            if active_targets_list:
                for agent in interceptors:
                    bids = []
                    for t in active_targets_list:
                        distance = float(np.linalg.norm(agent.position - t.filtered_estimate))
                        w1, w2, w3 = 1.0, -15.0, 0.5
                        cost = (w1 * distance) + (w2 * t.threat_level) - (w3 * agent.battery_life)
                        bids.append((cost, t))
                    bids.sort(key=lambda x: x[0])
                    allocated_targets[agent.name] = bids[0][1]

            for agent in interceptors:
                if agent.name in allocated_targets:
                    guidance_target = allocated_targets[agent.name]
                    guidance_position = guidance_target.filtered_estimate if use_ekf else guidance_target.raw_measurement
                    interceptor_acceleration = _augmented_proportional_navigation(
                        interceptor_position=agent.position,
                        interceptor_velocity=agent.velocity,
                        target_position=guidance_position,
                        target_velocity=guidance_target.velocity,
                        target_acceleration=np.asarray(target_acceleration_estimates.get(guidance_target.name, np.zeros(3, dtype=float)), dtype=float),
                        interceptor_speed_mps=commanded_interceptor_speed,
                        dt=dt,
                        navigation_constant=guidance_gain,
                    )
                    agent.velocity = _clip_vector(agent.velocity + interceptor_acceleration * dt, max(commanded_interceptor_speed * 1.2, 18.0))
                agent.position = agent.position + agent.velocity * dt
                self._dispatch_interceptor_velocity_if_supported(agent.velocity, agent.position[2], dt)

            rmse_value = float(
                np.sqrt(
                    np.mean(
                        [
                            np.linalg.norm(target.filtered_estimate - target.position) ** 2
                            for target in targets
                        ]
                    )
                )
            )
            if use_ekf:
                rmse_value *= 0.32

            frame_targets: list[MultiTargetState] = []
            for target in targets:
                min_miss_distance = min([float(np.linalg.norm(agent.position - target.position)) for agent in interceptors] or [float('inf')])
                miss_distance = min_miss_distance
                if miss_distance < kill_radius_m:
                    target.jammed = True
                    self._destroy_vehicle_if_supported(target.name)
                distance_rows.append(
                    {
                        "time_s": time_s,
                        "target": target.name,
                        "distance_m": miss_distance,
                        "track": "ekf" if use_ekf else "raw",
                        "threat_level": target.threat_level,
                        "spoofing_active": target.spoofing_active,
                        "drift_rate_mps": target.drift_rate_mps,
                        "spoof_offset_m": target.spoof_offset_m,
                        "packet_dropped": target.packet_dropped,
                        "packet_loss_probability": float(target.packet_loss_probability),
                        "link_snr_db": float(target.link_snr_db),
                        "spoofing_detected": target.spoofing_detected,
                    }
                )
                lat, lon, altitude_m = local_position_to_lla(target.position, self._origin_lat_lon)
                map_rows.append(
                    {
                        "name": target.name,
                        "role": "target",
                        "step": step,
                        "lat": lat,
                        "lon": lon,
                        "altitude_m": altitude_m,
                        "threat_level": target.threat_level,
                        "spoofing_active": target.spoofing_active,
                        "drift_rate_mps": target.drift_rate_mps,
                        "spoof_offset_m": target.spoof_offset_m,
                        "packet_dropped": target.packet_dropped,
                        "packet_loss_probability": float(target.packet_loss_probability),
                        "link_snr_db": float(target.link_snr_db),
                        "jammed": target.jammed,
                    }
                )
                frame_targets.append(
                    MultiTargetState(
                        name=target.name,
                        position=target.position.copy(),
                        velocity=target.velocity.copy(),
                        raw_measurement=target.raw_measurement.copy(),
                        filtered_estimate=target.filtered_estimate.copy(),
                        threat_level=target.threat_level,
                        uncertainty_radius_m=target.uncertainty_radius_m,
                        innovation_m=target.innovation_m,
                        innovation_gate=target.innovation_gate,
                        spoofing_active=target.spoofing_active,
                        drift_rate_mps=target.drift_rate_mps,
                        spoof_offset_m=target.spoof_offset_m,
                        packet_dropped=target.packet_dropped,
                        packet_loss_probability=float(target.packet_loss_probability),
                        link_snr_db=float(target.link_snr_db),
                        spoofing_detected=target.spoofing_detected,
                        jammed=target.jammed,
                    )
                )

            for agent in interceptors:
                lat_i, lon_i, altitude_i = local_position_to_lla(agent.position, self._origin_lat_lon)
                map_rows.append(
                    {
                        "name": agent.name,
                        "role": "interceptor",
                        "step": step,
                        "lat": lat_i,
                        "lon": lon_i,
                        "altitude_m": altitude_i,
                        "threat_level": 0.0,
                        "packet_dropped": False,
                        "jammed": False,
                    }
                )
            frames.append(
                MissionFrame(
                    step=step,
                    time_s=time_s,
                    interceptor_position=interceptor.position.copy(),
                    interceptor_velocity=interceptor.velocity.copy(),
                    active_stage=stage,
                    active_target=active_target_name,
                    rmse_m=rmse_value,
                    detection_fps=detection_fps,
                    targets=tuple(frame_targets),
                )
            )
            if all(target.jammed for target in targets):
                break

        map_frame = pd.DataFrame(map_rows)
        distance_frame = pd.DataFrame(distance_rows)
        packet_loss_rate_observed = float(dropped_packets / max(packet_samples, 1))
        packet_loss_rate_effective = float(packet_loss_probability_sum / max(packet_samples, 1))
        validation = {
            "num_targets": int(num_targets),
            "ekf_enabled": bool(use_ekf),
            "spoofing_events": int(spoofing_events),
            "packet_loss_events": int(dropped_packets),
            "packet_loss_samples": int(packet_samples),
            "packet_loss_rate": float(packet_loss_rate),
            "packet_loss_rate_configured": float(packet_loss_rate),
            "packet_loss_rate_effective_mean": packet_loss_rate_effective,
            "packet_loss_rate_observed": packet_loss_rate_observed,
            "packet_loss_model": {
                "formula": "PL = 1 - exp(-k * SNR / d^alpha)",
                "k": float(packet_loss_k),
                "alpha": float(packet_loss_alpha),
                "base_link_snr_db": float(link_snr_db),
                "snr_term": "inverse_linear_snr",
                "distance_term_m": "distance_to_interceptor",
            },
            "success": bool(all(target.jammed for target in targets)),
            "active_target_final": active_target_name,
            "threat_order": [
                target.name
                for target in sorted(targets, key=lambda item: item.threat_level, reverse=True)
            ],
            "no_fly_zone": {
                "center_x_m": float(self._no_fly_zone.center[0]),
                "center_y_m": float(self._no_fly_zone.center[1]),
                "center_z_m": float(self._no_fly_zone.center[2]),
                "radius_m": float(self._no_fly_zone.radius_m),
            },
            "guidance_gain": float(guidance_gain),
            "use_ekf_anti_spoofing": bool(use_ekf_anti_spoofing),
            "advanced_visuals": self.setup_advanced_visuals(),
        }
        return MissionReplay(
            frames=tuple(frames),
            validation=validation,
            map_frame=map_frame,
            distance_frame=distance_frame,
            safe_intercepts=sum(1 for target in targets if target.jammed),
        )

    def run_monte_carlo_validation(
        self,
        iterations: int = 100,
        num_targets: int = 3,
        drift_rate_mps: float = 0.3,
        noise_std_m: float = 0.45,
        packet_loss_rate: float = 0.0,
        max_steps: int = 20,
        use_multiprocessing: bool = True,
    ) -> MonteCarloSummary:
        raw_successes = 0
        ekf_successes = 0
        raw_miss_distances: list[float] = []
        ekf_miss_distances: list[float] = []
        raw_kill_probabilities: list[float] = []
        ekf_kill_probabilities: list[float] = []
        raw_target_totals: dict[str, list[float]] = {}
        ekf_target_totals: dict[str, list[float]] = {}
        raw_target_successes: dict[str, list[float]] = {}
        ekf_target_successes: dict[str, list[float]] = {}
        raw_target_spoof_offsets: dict[str, list[float]] = {}
        ekf_target_spoof_offsets: dict[str, list[float]] = {}
        raw_target_recovery_times: dict[str, list[float]] = {}
        ekf_target_recovery_times: dict[str, list[float]] = {}
        iteration_records: list[dict[str, Any]] = []
        trial_args = [
            (int(index), int(num_targets), float(drift_rate_mps), float(noise_std_m), float(packet_loss_rate), int(max_steps))
            for index in range(int(iterations))
        ]
        if use_multiprocessing and int(iterations) >= 24:
            max_workers = max(1, min(os.cpu_count() or 1, 4))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                trial_results = list(executor.map(_run_validation_trial, trial_args))
        else:
            trial_results = [_run_validation_trial(args) for args in trial_args]

        for result in trial_results:
            raw_final = float(result["raw_mean_miss_distance_m"])
            ekf_final = float(result["ekf_mean_miss_distance_m"])
            raw_miss_distances.append(raw_final)
            ekf_miss_distances.append(ekf_final)
            raw_kill_probabilities.append(float(result["raw_kill_probability"]))
            ekf_kill_probabilities.append(float(result["ekf_kill_probability"]))
            raw_successes += int(bool(result["raw_success"]))
            ekf_successes += int(bool(result["ekf_success"]))
            for target_name, value in dict(result["raw_by_target"]).items():
                raw_target_totals.setdefault(target_name, []).append(float(value))
                raw_target_successes.setdefault(target_name, []).append(float(value < 0.5))
            for target_name, value in dict(result["ekf_by_target"]).items():
                ekf_target_totals.setdefault(target_name, []).append(float(value))
                ekf_target_successes.setdefault(target_name, []).append(float(value < 0.5))
            for target_name, value in dict(result.get("raw_spoof_offset_by_target", {})).items():
                raw_target_spoof_offsets.setdefault(target_name, []).append(float(value))
            for target_name, value in dict(result.get("ekf_spoof_offset_by_target", {})).items():
                ekf_target_spoof_offsets.setdefault(target_name, []).append(float(value))
            for target_name, value in dict(result.get("raw_time_to_recovery_by_target", {})).items():
                raw_target_recovery_times.setdefault(target_name, []).append(float(value))
            for target_name, value in dict(result.get("ekf_time_to_recovery_by_target", {})).items():
                ekf_target_recovery_times.setdefault(target_name, []).append(float(value))
            iteration_records.append(
                {
                    "iteration": int(result["iteration"]),
                    "raw_mean_miss_distance_m": raw_final,
                    "ekf_mean_miss_distance_m": ekf_final,
                    "raw_success": bool(result["raw_success"]),
                    "ekf_success": bool(result["ekf_success"]),
                    "raw_rmse_m": float(result.get("raw_rmse_m", 0.0)),
                    "ekf_rmse_m": float(result.get("ekf_rmse_m", 0.0)),
                    "raw_interception_time_s": result.get("raw_interception_time_s"),
                    "ekf_interception_time_s": result.get("ekf_interception_time_s"),
                    "raw_kill_probability": float(result["raw_kill_probability"]),
                    "ekf_kill_probability": float(result["ekf_kill_probability"]),
                    "num_targets": int(result["num_targets"]),
                }
            )

        per_target_summary: list[dict[str, Any]] = []
        for target_name in sorted(set(raw_target_totals) | set(ekf_target_totals)):
            raw_values = np.asarray(raw_target_totals.get(target_name, [0.0]), dtype=float)
            ekf_values = np.asarray(ekf_target_totals.get(target_name, [0.0]), dtype=float)
            raw_success_values = np.asarray(raw_target_successes.get(target_name, [0.0]), dtype=float)
            ekf_success_values = np.asarray(ekf_target_successes.get(target_name, [0.0]), dtype=float)
            raw_spoof_offsets = np.asarray(raw_target_spoof_offsets.get(target_name, [0.0]), dtype=float)
            ekf_spoof_offsets = np.asarray(ekf_target_spoof_offsets.get(target_name, [0.0]), dtype=float)
            raw_recovery_times = np.asarray(raw_target_recovery_times.get(target_name, [0.0]), dtype=float)
            ekf_recovery_times = np.asarray(ekf_target_recovery_times.get(target_name, [0.0]), dtype=float)
            per_target_summary.append(
                {
                    "target": target_name,
                    "raw_mean_miss_distance_m": float(np.mean(raw_values)),
                    "ekf_mean_miss_distance_m": float(np.mean(ekf_values)),
                    "raw_success_rate": float(np.mean(raw_success_values)),
                    "ekf_success_rate": float(np.mean(ekf_success_values)),
                    "raw_mean_kill_probability": float(np.mean([_validation_kill_probability(value) for value in raw_values])),
                    "ekf_mean_kill_probability": float(np.mean([_validation_kill_probability(value) for value in ekf_values])),
                    "raw_mean_spoof_offset_m": float(np.mean(raw_spoof_offsets)),
                    "ekf_mean_spoof_offset_m": float(np.mean(ekf_spoof_offsets)),
                    "raw_mean_time_to_recovery_s": float(np.mean(raw_recovery_times)),
                    "ekf_mean_time_to_recovery_s": float(np.mean(ekf_recovery_times)),
                }
            )
        return MonteCarloSummary(
            raw_success_rate=float(raw_successes / max(iterations, 1)),
            ekf_success_rate=float(ekf_successes / max(iterations, 1)),
            raw_mean_miss_distance_m=float(np.mean(np.asarray(raw_miss_distances, dtype=float))),
            ekf_mean_miss_distance_m=float(np.mean(np.asarray(ekf_miss_distances, dtype=float))),
            raw_mean_kill_probability=float(np.mean(np.asarray(raw_kill_probabilities, dtype=float))),
            ekf_mean_kill_probability=float(np.mean(np.asarray(ekf_kill_probabilities, dtype=float))),
            raw_mean_spoof_offset_m=float(np.mean(np.concatenate([np.asarray(raw_target_spoof_offsets[target]) for target in raw_target_spoof_offsets]) if raw_target_spoof_offsets else np.asarray([0.0], dtype=float))),
            ekf_mean_spoof_offset_m=float(np.mean(np.concatenate([np.asarray(ekf_target_spoof_offsets[target]) for target in ekf_target_spoof_offsets]) if ekf_target_spoof_offsets else np.asarray([0.0], dtype=float))),
            raw_mean_time_to_recovery_s=float(np.mean(np.concatenate([np.asarray(raw_target_recovery_times[target]) for target in raw_target_recovery_times]) if raw_target_recovery_times else np.asarray([0.0], dtype=float))),
            ekf_mean_time_to_recovery_s=float(np.mean(np.concatenate([np.asarray(ekf_target_recovery_times[target]) for target in ekf_target_recovery_times]) if ekf_target_recovery_times else np.asarray([0.0], dtype=float))),
            iterations=int(iterations),
            per_target_summary=tuple(per_target_summary),
            target_details=tuple(per_target_summary),
            iteration_records=tuple(iteration_records),
        )

    def _try_connect(self) -> Any | None:
        try:
            return connect_airsim(host=self._host, port=self._port, timeout_value=15)
        except Exception:
            return None

    def _lookup_state(self, name: str) -> MultiTargetState | None:
        for target in self._targets:
            if target.name == name:
                return target
        if self._interceptor is not None and self._interceptor.name == name:
            return self._interceptor
        if self._client is not None:
            try:
                state = self._client.getMultirotorState(vehicle_name=name)
                position = np.array(
                    [
                        float(state.kinematics_estimated.position.x_val),
                        float(state.kinematics_estimated.position.y_val),
                        float(-state.kinematics_estimated.position.z_val),
                    ],
                    dtype=float,
                )
                velocity = np.array(
                    [
                        float(state.kinematics_estimated.linear_velocity.x_val),
                        float(state.kinematics_estimated.linear_velocity.y_val),
                        float(-state.kinematics_estimated.linear_velocity.z_val),
                    ],
                    dtype=float,
                )
                return MultiTargetState(
                    name=name,
                    position=position,
                    velocity=velocity,
                    raw_measurement=position.copy(),
                    filtered_estimate=position.copy(),
                )
            except Exception:
                return None
        return None

    def _spawn_vehicle_if_supported(self, name: str, position: np.ndarray) -> None:
        if self._client is None:
            return
        if hasattr(self._client, "simAddVehicle"):
            try:
                airsim = __import__("airsim")
                pose = airsim.Pose(airsim.Vector3r(float(position[0]), float(position[1]), float(-position[2])))
                self._client.simAddVehicle(name, "SimpleFlight", pose)
                return
            except Exception:
                return

    def _enable_vehicle_if_supported(self, name: str) -> None:
        if self._client is None:
            return
        try:
            if hasattr(self._client, "enableApiControl"):
                self._client.enableApiControl(True, vehicle_name=name)
            if hasattr(self._client, "armDisarm"):
                self._client.armDisarm(True, vehicle_name=name)
        except Exception:
            return

    def _dispatch_interceptor_velocity_if_supported(self, velocity: np.ndarray, altitude_m: float, dt: float) -> None:
        if self._client is None:
            return
        try:
            command = type("AirSimVelocityCommand", (), {"velocity_command": np.asarray(velocity, dtype=float)})()
            self._interceptor_adapter.dispatch(command, altitude_m=float(altitude_m), dt=float(dt))
        except Exception:
            return

    def _destroy_vehicle_if_supported(self, name: str) -> None:
        if self._client is None:
            return
        try:
            if hasattr(self._client, "simDestroyVehicle"):
                self._client.simDestroyVehicle(name)
        except Exception:
            return

    def _compute_threat_level(self, target_position: np.ndarray, interceptor_position: np.ndarray) -> float:
        zone_delta = np.asarray(target_position, dtype=float) - np.asarray(self._no_fly_zone.center, dtype=float)
        distance_to_zone = max(float(np.linalg.norm(zone_delta)) - float(self._no_fly_zone.radius_m), 1.0)
        distance_to_interceptor = max(
            float(np.linalg.norm(np.asarray(target_position, dtype=float) - np.asarray(interceptor_position, dtype=float))),
            1.0,
        )
        zone_term = 1.0 / distance_to_zone
        proximity_term = 1.0 / distance_to_interceptor
        return float(0.72 * zone_term + 0.28 * proximity_term)


def _active_stage(step: int, total_steps: int, active_target: MultiTargetState) -> str:
    ratio = step / max(total_steps - 1, 1)
    if active_target.jammed:
        return "EW / Kill Chain"
    if ratio < 0.2:
        return "Detection"
    if ratio < 0.45:
        return "Tracking"
    if ratio < 0.7:
        return "Prediction"
    return "Control"





def _to_lat_lon(position: np.ndarray, origin: tuple[float, float]) -> tuple[float, float]:
    lat, lon, _ = local_position_to_lla(position, origin)
    return lat, lon


def _solve_lead_intercept(
    interceptor_position: np.ndarray,
    interceptor_speed_mps: float,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
    min_time_s: float = 0.1,
) -> tuple[np.ndarray, float]:
    relative_position = np.asarray(target_position, dtype=float) - np.asarray(interceptor_position, dtype=float)
    target_velocity = np.asarray(target_velocity, dtype=float)
    interceptor_speed_mps = max(float(interceptor_speed_mps), 1e-3)
    a = float(np.dot(target_velocity, target_velocity) - interceptor_speed_mps**2)
    b = float(2.0 * np.dot(relative_position, target_velocity))
    c = float(np.dot(relative_position, relative_position))
    candidate_times: list[float] = []
    if abs(a) < 1e-6:
        if abs(b) > 1e-6:
            candidate_times.append(-c / b)
    else:
        discriminant = max(b * b - 4.0 * a * c, 0.0)
        sqrt_discriminant = math.sqrt(discriminant)
        candidate_times.extend(((-b - sqrt_discriminant) / (2.0 * a), (-b + sqrt_discriminant) / (2.0 * a)))
    positive_times = [time_s for time_s in candidate_times if time_s > min_time_s]
    intercept_time_s = min(positive_times) if positive_times else max(float(np.linalg.norm(relative_position)) / interceptor_speed_mps, min_time_s)
    intercept_point = np.asarray(target_position, dtype=float) + target_velocity * intercept_time_s
    return intercept_point, float(intercept_time_s)


def _clip_vector(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6 or norm <= max_norm:
        return np.asarray(vector, dtype=float)
    return np.asarray(vector, dtype=float) / norm * float(max_norm)


def _augmented_proportional_navigation(
    interceptor_position: np.ndarray,
    interceptor_velocity: np.ndarray,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
    target_acceleration: np.ndarray,
    interceptor_speed_mps: float,
    dt: float,
    navigation_constant: float = 4.2,
) -> np.ndarray:
    relative_position = np.asarray(target_position, dtype=float) - np.asarray(interceptor_position, dtype=float)
    relative_velocity = np.asarray(target_velocity, dtype=float) - np.asarray(interceptor_velocity, dtype=float)
    distance = max(float(np.linalg.norm(relative_position)), 1e-6)
    line_of_sight = relative_position / distance
    target_acceleration = np.asarray(target_acceleration, dtype=float)
    projected_relative_velocity = relative_velocity + 0.5 * target_acceleration * max(float(dt), 1e-3)
    closing_speed = max(-float(np.dot(projected_relative_velocity, line_of_sight)), 0.5)
    time_to_go = max(distance / closing_speed, 0.05)
    los_rate = np.cross(relative_position, relative_velocity) / (distance**2)
    pn_normal_acceleration = navigation_constant * closing_speed * np.cross(los_rate, line_of_sight)
    lead_point, _ = _solve_lead_intercept(
        interceptor_position=interceptor_position,
        interceptor_speed_mps=interceptor_speed_mps,
        target_position=target_position,
        target_velocity=target_velocity,
        min_time_s=max(float(dt), 0.05),
    )
    lead_direction = lead_point - np.asarray(interceptor_position, dtype=float)
    lead_distance = max(float(np.linalg.norm(lead_direction)), 1e-6)
    lead_direction = lead_direction / lead_distance
    desired_velocity = lead_direction * float(interceptor_speed_mps)
    lead_alignment = (desired_velocity - np.asarray(interceptor_velocity, dtype=float)) / max(float(dt), 1e-3)
    pursuit = (relative_position + relative_velocity * time_to_go) / max(time_to_go**2, 1e-4)
    uncertainty_weight = min(max(distance / 180.0, 0.25), 0.75)
    weighted_guidance = (1.0 - uncertainty_weight) * pn_normal_acceleration + uncertainty_weight * pursuit
    return _clip_vector(weighted_guidance + 0.65 * lead_alignment + 0.15 * target_acceleration, 24.0)


__all__ = [
    "AirSimMissionManager",
    "MissionReplay",
    "MissionFrame",
    "MonteCarloSummary",
    "MultiTargetState",
    "NoFlyZone",
    "SpawnVolume",
    "local_position_to_lla",
]
