from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from drone_interceptor.navigation.ekf_filter import InterceptorEKF
from drone_interceptor.navigation.drift_model.intelligent import DriftMode, IntelligentDriftEngine
from drone_interceptor.navigation.state_estimator import GPSIMUKalmanFusion
from drone_interceptor.types import SensorPacket


@dataclass(frozen=True, slots=True)
class TelemetrySpoofSample:
    time_s: float
    true_position: np.ndarray
    spoofed_position: np.ndarray
    drift_offset: np.ndarray
    drift_rate_mps: float
    distance_to_interceptor_m: float
    distance_to_safe_zone_m: float
    attack_active: bool


@dataclass(frozen=True, slots=True)
class AttackProfile:
    name: str
    mode: DriftMode
    onset_time_s: float = 0.0
    intermittent_period_s: float = 0.0
    duty_cycle: float = 1.0


@dataclass(frozen=True, slots=True)
class DefenseSweepResult:
    defense_mode: str
    mean_position_error_m: float
    max_position_error_m: float
    spoofing_detection_rate: float
    packet_loss_rate: float


class DP5CoordinateSpoofingToolkit:
    """Simulation-only coordinate drift toolkit for safe DP5 validation."""

    def __init__(
        self,
        safe_zone_position: np.ndarray,
        min_rate_mps: float = 0.2,
        max_rate_mps: float = 0.5,
        near_distance_m: float = 120.0,
        noise_std_m: float = 0.0,
        circular_frequency_hz: float = 0.12,
        random_seed: int = 7,
    ) -> None:
        self._engine = IntelligentDriftEngine(
            min_rate_mps=min_rate_mps,
            max_rate_mps=max_rate_mps,
            near_distance_m=near_distance_m,
            noise_std_m=noise_std_m,
            safe_zone_position=np.asarray(safe_zone_position, dtype=float),
            circular_frequency_hz=circular_frequency_hz,
            random_seed=random_seed,
        )

    @property
    def safe_zone(self) -> np.ndarray:
        return self._engine.safe_zone

    def sample(
        self,
        true_position: np.ndarray,
        interceptor_position: np.ndarray,
        time_s: float,
        mode: DriftMode = "directed",
        attack_profile: AttackProfile | None = None,
    ) -> TelemetrySpoofSample:
        profile = attack_profile or AttackProfile(name=str(mode), mode=mode)
        attack_active = self._attack_active(profile, float(time_s))
        active_mode = profile.mode if attack_active else "directed"
        sample = self._engine.sample(
            true_position=np.asarray(true_position, dtype=float),
            interceptor_position=np.asarray(interceptor_position, dtype=float),
            time_s=float(time_s),
            mode=active_mode,
        )
        spoofed_position = sample.fake_position.copy() if attack_active else sample.true_position.copy()
        drift_offset = sample.drift_offset.copy() if attack_active else np.zeros_like(sample.drift_offset, dtype=float)
        drift_rate = float(sample.adaptive_rate_mps) if attack_active else 0.0
        return TelemetrySpoofSample(
            time_s=float(time_s),
            true_position=sample.true_position.copy(),
            spoofed_position=spoofed_position,
            drift_offset=drift_offset,
            drift_rate_mps=drift_rate,
            distance_to_interceptor_m=float(sample.distance_to_interceptor_m),
            distance_to_safe_zone_m=float(sample.distance_to_safe_zone_m),
            attack_active=attack_active,
        )

    def generate_profile(
        self,
        true_positions: np.ndarray,
        interceptor_positions: np.ndarray,
        dt: float,
        mode: DriftMode = "directed",
        attack_profile: AttackProfile | None = None,
    ) -> list[TelemetrySpoofSample]:
        true_positions = np.asarray(true_positions, dtype=float)
        interceptor_positions = np.asarray(interceptor_positions, dtype=float)
        length = min(len(true_positions), len(interceptor_positions))
        return [
            self.sample(
                true_position=true_positions[index],
                interceptor_position=interceptor_positions[index],
                time_s=float(index) * float(dt),
                mode=mode,
                attack_profile=attack_profile,
            )
            for index in range(length)
        ]

    def export_profile_rows(
        self,
        true_positions: np.ndarray,
        interceptor_positions: np.ndarray,
        dt: float,
        mode: DriftMode = "directed",
        attack_profile: AttackProfile | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for sample in self.generate_profile(true_positions, interceptor_positions, dt=dt, mode=mode, attack_profile=attack_profile):
            rows.append(
                {
                    "time_s": sample.time_s,
                    "true_x_m": float(sample.true_position[0]),
                    "true_y_m": float(sample.true_position[1]),
                    "true_z_m": float(sample.true_position[2]),
                    "spoofed_x_m": float(sample.spoofed_position[0]),
                    "spoofed_y_m": float(sample.spoofed_position[1]),
                    "spoofed_z_m": float(sample.spoofed_position[2]),
                    "offset_x_m": float(sample.drift_offset[0]),
                    "offset_y_m": float(sample.drift_offset[1]),
                    "offset_z_m": float(sample.drift_offset[2]),
                    "drift_rate_mps": float(sample.drift_rate_mps),
                    "distance_to_interceptor_m": float(sample.distance_to_interceptor_m),
                    "distance_to_safe_zone_m": float(sample.distance_to_safe_zone_m),
                    "attack_active": bool(sample.attack_active),
                }
            )
        return rows

    def run_defense_sweep(
        self,
        config: dict[str, Any],
        true_positions: np.ndarray,
        true_velocities: np.ndarray,
        interceptor_positions: np.ndarray,
        dt: float,
        attack_profile: AttackProfile,
        packet_loss_rate: float = 0.0,
    ) -> list[DefenseSweepResult]:
        samples = self.generate_profile(
            true_positions=true_positions,
            interceptor_positions=interceptor_positions,
            dt=dt,
            mode=attack_profile.mode,
            attack_profile=attack_profile,
        )
        results: list[DefenseSweepResult] = []
        rng = np.random.default_rng(17)
        for defense_mode in ("raw_gps", "kalman_fusion", "ekf_innovation_gate", "adaptive_trust"):
            errors: list[float] = []
            detections: list[float] = []
            if defense_mode == "kalman_fusion":
                fusion = GPSIMUKalmanFusion(config)
            else:
                fusion = None
            ekf = InterceptorEKF(
                dt=float(dt),
                process_noise=float(config.get("tracking", {}).get("process_noise", 0.1)),
                measurement_noise=float(config.get("tracking", {}).get("measurement_noise", 0.2)),
            )
            ekf.initialize(np.asarray(true_positions[0], dtype=float), np.asarray(true_velocities[0], dtype=float))
            adaptive_estimate = np.asarray(true_positions[0], dtype=float).copy()
            for index, sample in enumerate(samples):
                measured_position = sample.spoofed_position.copy()
                if rng.random() < max(min(packet_loss_rate, 1.0), 0.0):
                    measured_position = sample.true_position.copy()
                if defense_mode == "raw_gps":
                    estimate = measured_position
                    spoofing_detected = float(sample.attack_active)
                elif defense_mode == "kalman_fusion":
                    assert fusion is not None
                    state = fusion.update(
                        SensorPacket(
                            gps_position=measured_position,
                            imu_acceleration=np.zeros(3, dtype=float),
                            timestamp=float(index) * float(dt),
                            true_position=np.asarray(sample.true_position, dtype=float),
                            true_velocity=np.asarray(true_velocities[min(index, len(true_velocities) - 1)], dtype=float),
                        )
                    )
                    estimate = np.asarray(state.position, dtype=float)
                    spoofing_detected = 0.0
                elif defense_mode == "ekf_innovation_gate":
                    ekf.predict(drift_rate_mps=0.0)
                    assessment = ekf.assess(measured_position)
                    ekf.update(measured_position, is_spoofing_detected=assessment.spoofing_detected)
                    estimate = ekf.position.copy()
                    spoofing_detected = float(assessment.spoofing_detected)
                else:
                    ekf.predict(drift_rate_mps=0.0)
                    assessment = ekf.assess(measured_position)
                    if assessment.spoofing_detected:
                        adaptive_estimate = adaptive_estimate + 0.35 * (
                            np.asarray(true_velocities[min(index, len(true_velocities) - 1)], dtype=float) * float(dt)
                        )
                    else:
                        adaptive_estimate = 0.75 * adaptive_estimate + 0.25 * measured_position
                    estimate = adaptive_estimate.copy()
                    spoofing_detected = float(assessment.spoofing_detected)
                errors.append(float(np.linalg.norm(np.asarray(estimate, dtype=float) - np.asarray(sample.true_position, dtype=float))))
                detections.append(spoofing_detected)
            results.append(
                DefenseSweepResult(
                    defense_mode=defense_mode,
                    mean_position_error_m=float(np.mean(errors)) if errors else 0.0,
                    max_position_error_m=float(np.max(errors)) if errors else 0.0,
                    spoofing_detection_rate=float(np.mean(detections)) if detections else 0.0,
                    packet_loss_rate=float(packet_loss_rate),
                )
            )
        return results

    def _attack_active(self, attack_profile: AttackProfile, time_s: float) -> bool:
        if time_s < float(attack_profile.onset_time_s):
            return False
        if attack_profile.intermittent_period_s <= 0.0:
            return True
        phase = (time_s - float(attack_profile.onset_time_s)) % float(attack_profile.intermittent_period_s)
        return phase <= float(attack_profile.duty_cycle) * float(attack_profile.intermittent_period_s)


__all__ = [
    "AttackProfile",
    "DP5CoordinateSpoofingToolkit",
    "DefenseSweepResult",
    "TelemetrySpoofSample",
]
