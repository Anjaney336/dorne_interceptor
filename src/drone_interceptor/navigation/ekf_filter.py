from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True, slots=True)
class SpoofingAssessment:
    innovation: np.ndarray
    mahalanobis_distance: float
    threshold: float
    soft_threshold: float
    trust_scale: float
    soft_spoofing_detected: bool
    spoofing_detected: bool


class InterceptorEKF:
    """Constant-velocity EKF with adaptive GPS trust for spoofing resilience."""
    _SOFT_SPOOF_TRUST_SCALE = 8.0
    _HARD_SPOOF_TRUST_SCALE = 4096.0

    def __init__(
        self,
        dt: float = 0.05,
        process_noise: float = 0.08,
        measurement_noise: float = 0.45,
        spoofing_soft_confidence: float = 0.975,
        spoofing_confidence: float = 0.995,
    ) -> None:
        self.dt = float(dt)
        self.X = np.zeros((6, 1), dtype=float)
        self.P = np.eye(6, dtype=float)
        self._base_process_noise = float(process_noise)
        self._base_measurement_noise = measurement_noise
        self._adaptive_process_scale = 1.0
        self._adaptive_measurement_scale = 1.0
        self.Q = self._build_process_covariance(self._base_process_noise)
        self.R = self._build_measurement_covariance(self._base_measurement_noise)
        self.A = np.eye(6, dtype=float)
        for index in range(3):
            self.A[index, index + 3] = self.dt
        self.H = np.hstack((np.eye(3, dtype=float), np.zeros((3, 3), dtype=float)))
        self._spoofing_soft_threshold = float(chi2.ppf(spoofing_soft_confidence, df=3))
        self._spoofing_threshold = float(chi2.ppf(spoofing_confidence, df=3))
        self._last_innovation: np.ndarray = np.zeros((3, 1), dtype=float)
        self._last_innovation_covariance: np.ndarray = np.eye(3, dtype=float)

    def _build_process_covariance(self, accel_sigma: float) -> np.ndarray:
        accel_sigma = max(float(accel_sigma), 1e-3)
        dt2 = self.dt**2
        dt3 = self.dt**3
        dt4 = self.dt**4
        q_xy = accel_sigma**2
        q_z = (0.7 * accel_sigma) ** 2
        covariance = np.zeros((6, 6), dtype=float)
        for axis, q_axis in enumerate((q_xy, q_xy, q_z)):
            covariance[axis, axis] = 0.25 * dt4 * q_axis
            covariance[axis, axis + 3] = 0.5 * dt3 * q_axis
            covariance[axis + 3, axis] = 0.5 * dt3 * q_axis
            covariance[axis + 3, axis + 3] = dt2 * q_axis
        return covariance

    def _build_measurement_covariance(self, measurement_noise: float | np.ndarray) -> np.ndarray:
        noise_array = np.asarray(measurement_noise, dtype=float)
        if noise_array.ndim == 0:
            xy_sigma = max(float(noise_array), 1e-3)
            z_sigma = max(0.67 * xy_sigma, 1e-3)
            sigmas = np.array([xy_sigma, xy_sigma, z_sigma], dtype=float)
        else:
            flattened = noise_array.reshape(-1)
            if flattened.size != 3:
                raise ValueError("measurement_noise must be scalar or length-3")
            sigmas = np.maximum(flattened[:3], 1e-3)
        return np.diag(np.square(sigmas))

    def set_noise_levels(self, process_noise: float | None = None, measurement_noise: float | np.ndarray | None = None) -> None:
        if process_noise is not None:
            self._base_process_noise = float(process_noise)
            self.Q = self._build_process_covariance(self._base_process_noise)
        if measurement_noise is not None:
            self._base_measurement_noise = measurement_noise
            self.R = self._build_measurement_covariance(self._base_measurement_noise)

    def set_adaptive_scales(self, process_scale: float = 1.0, measurement_scale: float = 1.0) -> None:
        self._adaptive_process_scale = max(float(process_scale), 1e-3)
        self._adaptive_measurement_scale = max(float(measurement_scale), 1e-3)

    def adapt_for_tracking_error(self, rolling_rmse_m: float, drift_rate_mps: float = 0.0, packet_loss: bool = False) -> None:
        process_scale = 1.0 + 1.8 * max(float(drift_rate_mps), 0.0)
        measurement_scale = 1.0 + 1.2 * max(float(drift_rate_mps), 0.0)
        if float(rolling_rmse_m) > 0.5:
            process_scale *= 1.35
            measurement_scale *= 1.15
            self.Q = self._build_process_covariance(self._base_process_noise * process_scale)
        if packet_loss:
            process_scale *= 1.2
            measurement_scale *= 1.1
        self.set_adaptive_scales(process_scale=process_scale, measurement_scale=measurement_scale)

    def initialize(self, position: np.ndarray, velocity: np.ndarray | None = None) -> None:
        position = np.asarray(position, dtype=float).reshape(3)
        self.X[:3, 0] = position
        if velocity is not None:
            self.X[3:, 0] = np.asarray(velocity, dtype=float).reshape(3)

    def predict(self, drift_rate_mps: float = 0.0, packet_loss: bool = False) -> np.ndarray:
        drift_bias = np.array([[drift_rate_mps * self.dt], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype=float)
        self.X = self.A @ self.X + drift_bias
        q_scale = self._adaptive_process_scale * (1.0 + 2.0 * max(float(drift_rate_mps), 0.0)) * (1.15 if packet_loss else 1.0)
        self.P = self.A @ self.P @ self.A.T + self.Q * q_scale
        return self.position

    def assess(self, z_measured: np.ndarray) -> SpoofingAssessment:
        z = np.asarray(z_measured, dtype=float).reshape(3, 1)
        innovation = z - (self.H @ self.X)
        innovation_covariance = self.H @ self.P @ self.H.T + (self.R * self._adaptive_measurement_scale)
        self._last_innovation = innovation.copy()
        self._last_innovation_covariance = innovation_covariance.copy()
        mahalanobis = float((innovation.T @ np.linalg.inv(innovation_covariance) @ innovation).item())
        soft_spoofing_detected = mahalanobis > self._spoofing_soft_threshold
        spoofing_detected = mahalanobis > self._spoofing_threshold
        if spoofing_detected:
            trust_scale = self._HARD_SPOOF_TRUST_SCALE
        elif soft_spoofing_detected:
            trust_scale = self._SOFT_SPOOF_TRUST_SCALE
        else:
            trust_scale = 1.0
        return SpoofingAssessment(
            innovation=innovation.reshape(3),
            mahalanobis_distance=mahalanobis,
            threshold=self._spoofing_threshold,
            soft_threshold=self._spoofing_soft_threshold,
            trust_scale=trust_scale,
            soft_spoofing_detected=soft_spoofing_detected,
            spoofing_detected=spoofing_detected,
        )

    def update(self, z_measured: np.ndarray, is_spoofing_detected: bool = False) -> np.ndarray:
        trust_scale = self._HARD_SPOOF_TRUST_SCALE if is_spoofing_detected else 1.0
        return self.update_with_trust_scale(z_measured=z_measured, trust_scale=trust_scale)

    def update_with_trust_scale(self, z_measured: np.ndarray, trust_scale: float = 1.0) -> np.ndarray:
        z = np.asarray(z_measured, dtype=float).reshape(3, 1)
        adaptive_r = self.R * max(float(trust_scale) * self._adaptive_measurement_scale, 1e-6)
        innovation = z - (self.H @ self.X)
        innovation_covariance = self.H @ self.P @ self.H.T + adaptive_r
        kalman_gain = self.P @ self.H.T @ np.linalg.inv(innovation_covariance)
        self.X = self.X + kalman_gain @ innovation
        self.P = self.P - kalman_gain @ self.H @ self.P
        return self.X.copy()

    @property
    def innovation_vector(self) -> np.ndarray:
        return self._last_innovation.reshape(3).copy()

    @property
    def residual_covariance(self) -> np.ndarray:
        return self._last_innovation_covariance.copy()

    @property
    def gate_threshold(self) -> float:
        return float(self._spoofing_threshold)

    def step(self, z_measured: np.ndarray, drift_rate_mps: float = 0.0) -> tuple[np.ndarray, SpoofingAssessment]:
        self.predict(drift_rate_mps=drift_rate_mps)
        assessment = self.assess(z_measured)
        self.update_with_trust_scale(z_measured, trust_scale=assessment.trust_scale)
        return self.X.copy(), assessment

    @property
    def position(self) -> np.ndarray:
        return self.X[:3, 0].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.X[3:, 0].copy()


__all__ = ["InterceptorEKF", "SpoofingAssessment", "lla_to_local_position", "local_position_to_lla"]

def lla_to_local_position(lat: float, lon: float, origin: tuple[float, float]) -> np.ndarray:
    lat0, lon0 = origin
    R_earth = 6378137.0
    d_lat = np.radians(lat - lat0)
    d_lon = np.radians(lon - lon0)
    x = d_lat * R_earth
    y = d_lon * R_earth * np.cos(np.radians(lat0))
    return np.array([x, y, 0.0], dtype=float)

def local_position_to_lla(position: np.ndarray, origin: tuple[float, float]) -> tuple[float, float, float]:
    lat0, lon0 = origin
    R_earth = 6378137.0
    lat = lat0 + np.degrees(position[0] / R_earth)
    lon = lon0 + np.degrees(position[1] / (R_earth * np.cos(np.radians(lat0))))
    altitude_m = float(position[2])
    return lat, lon, altitude_m
