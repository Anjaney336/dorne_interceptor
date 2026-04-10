from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class TrajectoryPrediction:
    predicted_positions: np.ndarray
    estimated_velocity: np.ndarray
    estimated_acceleration: np.ndarray
    backend: str


class _ZeroResidualTrajectoryModel:
    def __call__(self, sequence: np.ndarray, horizon_steps: int) -> np.ndarray:
        del sequence
        return np.zeros((horizon_steps, 2), dtype=float)


class HybridTrajectoryPredictor:
    """Predict future 2D target positions from past observations."""

    def __init__(
        self,
        dt: float,
        horizon_steps: int,
        history_steps: int = 6,
        lstm_checkpoint: str | Path | None = None,
        lstm_residual_gain: float = 0.25,
    ) -> None:
        self._dt = float(dt)
        self._horizon_steps = int(horizon_steps)
        self._history_steps = int(history_steps)
        self._residual_gain = float(lstm_residual_gain)
        self._residual_model = self._load_lstm_extension(lstm_checkpoint)

    @property
    def backend(self) -> str:
        if isinstance(self._residual_model, _ZeroResidualTrajectoryModel):
            return "physics"
        return "physics_lstm_hybrid"

    def predict(
        self,
        past_positions: np.ndarray | list[list[float]] | list[tuple[float, float]],
        acceleration: np.ndarray | list[float] | tuple[float, float] | None = None,
        horizon_steps: int | None = None,
    ) -> TrajectoryPrediction:
        positions = np.asarray(past_positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("past_positions must have shape (T, 2).")
        if positions.shape[0] < 2:
            raise ValueError("past_positions must contain at least two samples.")

        horizon = self._horizon_steps if horizon_steps is None else int(horizon_steps)
        velocity = self._estimate_velocity(positions)
        accel = self._estimate_acceleration(positions) if acceleration is None else np.asarray(acceleration, dtype=float)
        if accel.shape != (2,):
            raise ValueError("acceleration must have shape (2,).")

        predicted_positions = self._physics_rollout(
            last_position=positions[-1],
            velocity=velocity,
            acceleration=accel,
            horizon_steps=horizon,
        )
        residuals = self._predict_residuals(positions, horizon)
        predicted_positions = predicted_positions + residuals * self._residual_gain

        return TrajectoryPrediction(
            predicted_positions=predicted_positions,
            estimated_velocity=velocity,
            estimated_acceleration=accel,
            backend=self.backend,
        )

    def _physics_rollout(
        self,
        last_position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        horizon_steps: int,
    ) -> np.ndarray:
        trajectory = np.zeros((horizon_steps, 2), dtype=float)
        for step in range(1, horizon_steps + 1):
            time_horizon = step * self._dt
            trajectory[step - 1] = (
                last_position
                + velocity * time_horizon
                + 0.5 * acceleration * (time_horizon**2)
            )
        return trajectory

    def _estimate_velocity(self, positions: np.ndarray) -> np.ndarray:
        return (positions[-1] - positions[-2]) / self._dt

    def _estimate_acceleration(self, positions: np.ndarray) -> np.ndarray:
        if positions.shape[0] < 3:
            return np.zeros(2, dtype=float)
        last_velocity = (positions[-1] - positions[-2]) / self._dt
        prev_velocity = (positions[-2] - positions[-3]) / self._dt
        return (last_velocity - prev_velocity) / self._dt

    def _predict_residuals(self, positions: np.ndarray, horizon_steps: int) -> np.ndarray:
        if positions.shape[0] < self._history_steps:
            return np.zeros((horizon_steps, 2), dtype=float)

        recent_positions = positions[-self._history_steps :]
        velocities = np.diff(recent_positions, axis=0, prepend=recent_positions[:1]) / self._dt
        sequence = np.concatenate((recent_positions, velocities), axis=1)
        return np.asarray(self._residual_model(sequence, horizon_steps), dtype=float)

    def _load_lstm_extension(self, checkpoint_path: str | Path | None) -> Any:
        if checkpoint_path is None:
            return _ZeroResidualTrajectoryModel()

        candidate = Path(checkpoint_path)
        if not candidate.exists():
            return _ZeroResidualTrajectoryModel()

        try:
            import torch
            from torch import nn
        except ImportError:
            return _ZeroResidualTrajectoryModel()

        class ResidualLSTM(nn.Module):
            def __init__(self, input_size: int, hidden_size: int) -> None:
                super().__init__()
                self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
                self.hidden_size = hidden_size
                self.projection: nn.Module | None = None

            def build_head(self, horizon_steps: int) -> None:
                self.projection = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, horizon_steps * 2),
                )

            def forward(self, sequence: torch.Tensor, horizon_steps: int) -> torch.Tensor:
                if self.projection is None:
                    self.build_head(horizon_steps)
                _, (hidden_state, _) = self.lstm(sequence)
                output = self.projection(hidden_state[-1])
                return output.view(sequence.shape[0], horizon_steps, 2)

        model = ResidualLSTM(input_size=4, hidden_size=32)
        state_dict = torch.load(candidate, map_location="cpu")
        model.build_head(self._horizon_steps)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        def predictor(sequence: np.ndarray, horizon_steps: int) -> np.ndarray:
            with torch.no_grad():
                tensor = torch.tensor(sequence[None, :, :], dtype=torch.float32)
                output = model(tensor, horizon_steps).cpu().numpy()[0]
            return output

        return predictor
