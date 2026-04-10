from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from drone_interceptor.dynamics.state_space import build_state_matrices
from drone_interceptor.types import TargetState


class _FallbackResidualModel:
    def __call__(self, sequence: np.ndarray, horizon_steps: int) -> np.ndarray:
        del sequence
        return np.zeros((horizon_steps, 3), dtype=float)


class TargetPredictor:
    """Physics-first trajectory predictor with an optional LSTM residual model."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._mission = config["mission"]
        self._prediction = config["prediction"]
        self._dt = float(self._mission["time_step"])
        self._horizon = int(self._prediction["horizon_steps"])
        self._sequence_length = int(self._prediction.get("history_steps", 8))
        self._residual_gain = float(self._prediction.get("lstm_residual_gain", 0.25))
        self._process_noise_std = float(self._prediction.get("process_noise", config["tracking"].get("process_noise", 0.15)))
        self._acceleration_damping = float(self._prediction.get("acceleration_damping", 0.55))
        self._history: deque[TargetState] = deque(maxlen=self._sequence_length)
        self._residual_model = self._load_residual_model(
            checkpoint_path=self._prediction.get("lstm_checkpoint"),
        )

    def predict(self, track: TargetState) -> list[TargetState]:
        self._history.append(self._copy_state(track))
        physics_prediction = self._physics_rollout(track)
        residuals = self._predict_residuals()

        for index, state in enumerate(physics_prediction):
            state.position = state.position + residuals[index] * self._residual_gain
            if index > 0:
                previous_position = physics_prediction[index - 1].position
                state.velocity = (state.position - previous_position) / self._dt
            else:
                acceleration = track.acceleration if track.acceleration is not None else np.zeros(3)
                state.velocity = track.velocity + acceleration * self._dt
            state.metadata["prediction_backend"] = self.model_name
            state.metadata["prediction_step"] = index + 1
        return physics_prediction

    @property
    def model_name(self) -> str:
        if isinstance(self._residual_model, _FallbackResidualModel):
            return "physics"
        return "physics_lstm_hybrid"

    def _physics_rollout(self, track: TargetState) -> list[TargetState]:
        states: list[TargetState] = []
        acceleration = (
            np.asarray(track.acceleration, dtype=float)
            if track.acceleration is not None
            else np.zeros(3, dtype=float)
        )
        acceleration = acceleration * self._acceleration_damping
        planar_state = np.array(
            [track.position[0], track.position[1], track.velocity[0], track.velocity[1]],
            dtype=float,
        )
        planar_covariance = self._resolve_planar_covariance(track.covariance)
        a_matrix, b_matrix = build_state_matrices(self._dt)
        process_noise = self._build_process_noise(self._dt)
        vertical_variance = float(track.metadata.get("vertical_variance", 0.25))
        vertical_velocity_variance = float(track.metadata.get("vertical_velocity_variance", 0.25))

        for step in range(1, self._horizon + 1):
            planar_state = a_matrix @ planar_state + b_matrix @ acceleration[:2]
            planar_covariance = a_matrix @ planar_covariance @ a_matrix.T + process_noise
            position = np.array(
                [
                    planar_state[0],
                    planar_state[1],
                    track.position[2] + track.velocity[2] * step * self._dt + 0.5 * acceleration[2] * ((step * self._dt) ** 2),
                ],
                dtype=float,
            )
            velocity = np.array(
                [planar_state[2], planar_state[3], track.velocity[2] + acceleration[2] * step * self._dt],
                dtype=float,
            )
            states.append(
                TargetState(
                    position=position,
                    velocity=velocity,
                    acceleration=acceleration.copy(),
                    covariance=planar_covariance.copy(),
                    timestamp=(track.timestamp or 0.0) + step * self._dt,
                    track_id=track.track_id,
                    metadata={
                        "source": "physics_rollout",
                        "covariance_trace": float(np.trace(planar_covariance)),
                        "vertical_variance": vertical_variance,
                        "vertical_velocity_variance": vertical_velocity_variance,
                    },
                )
            )
        return states

    def _predict_residuals(self) -> np.ndarray:
        if len(self._history) < self._sequence_length:
            return np.zeros((self._horizon, 3), dtype=float)

        sequence = np.stack(
            [
                np.concatenate((state.position.astype(float), state.velocity.astype(float)))
                for state in self._history
            ],
            axis=0,
        )
        return np.asarray(self._residual_model(sequence, self._horizon), dtype=float)

    def _load_residual_model(self, checkpoint_path: str | None) -> Any:
        if not checkpoint_path:
            return _FallbackResidualModel()

        candidate = Path(checkpoint_path)
        if not candidate.exists():
            return _FallbackResidualModel()

        try:
            import torch
            from torch import nn
        except ImportError:
            return _FallbackResidualModel()

        class ResidualLSTM(nn.Module):
            def __init__(self, input_size: int, hidden_size: int, horizon_steps: int) -> None:
                super().__init__()
                self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
                self.head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, horizon_steps * 3),
                )
                self.horizon_steps = horizon_steps

            def forward(self, sequence: torch.Tensor) -> torch.Tensor:
                _, (hidden_state, _) = self.lstm(sequence)
                output = self.head(hidden_state[-1])
                return output.view(sequence.shape[0], self.horizon_steps, 3)

        model = ResidualLSTM(input_size=6, hidden_size=32, horizon_steps=self._horizon)
        state_dict = torch.load(candidate, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        def predictor(sequence: np.ndarray, horizon_steps: int) -> np.ndarray:
            del horizon_steps
            with torch.no_grad():
                tensor = torch.tensor(sequence[None, :, :], dtype=torch.float32)
                output = model(tensor).cpu().numpy()[0]
            return output

        return predictor

    def _copy_state(self, track: TargetState) -> TargetState:
        return TargetState(
            position=np.asarray(track.position, dtype=float).copy(),
            velocity=np.asarray(track.velocity, dtype=float).copy(),
            acceleration=(
                np.asarray(track.acceleration, dtype=float).copy()
                if track.acceleration is not None
                else np.zeros(3, dtype=float)
            ),
            covariance=None if track.covariance is None else np.asarray(track.covariance, dtype=float).copy(),
            timestamp=track.timestamp,
            track_id=track.track_id,
            metadata=dict(track.metadata),
        )

    def _resolve_planar_covariance(self, covariance: np.ndarray | None) -> np.ndarray:
        if covariance is None:
            return np.eye(4, dtype=float) * max(self._process_noise_std**2, 1e-4)

        covariance_matrix = np.asarray(covariance, dtype=float)
        if covariance_matrix.shape == (4, 4):
            return covariance_matrix.copy()
        if covariance_matrix.shape == (3, 3):
            planar = np.eye(4, dtype=float) * max(self._process_noise_std**2, 1e-4)
            planar[0, 0] = covariance_matrix[0, 0]
            planar[1, 1] = covariance_matrix[1, 1]
            return planar
        raise ValueError("Target covariance must be 3x3 or 4x4.")

    def _build_process_noise(self, dt: float) -> np.ndarray:
        spectral_density = max(self._process_noise_std**2, 1e-8)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        return spectral_density * np.array(
            [
                [0.25 * dt4, 0.0, 0.5 * dt3, 0.0],
                [0.0, 0.25 * dt4, 0.0, 0.5 * dt3],
                [0.5 * dt3, 0.0, dt2, 0.0],
                [0.0, 0.5 * dt3, 0.0, dt2],
            ],
            dtype=float,
        )
