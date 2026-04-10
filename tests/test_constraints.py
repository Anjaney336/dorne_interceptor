from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.config import load_config
from drone_interceptor.constraints import (
    ConstraintModel,
    clamp_drift_rate,
    load_constraint_envelope,
    tracking_precision_ok,
)
from drone_interceptor.types import TargetState


def test_constraint_model_enforces_velocity_and_acceleration_limits() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    model = ConstraintModel(config)

    command, status = model.enforce_guidance_command(
        interceptor_state=TargetState(
            position=np.zeros(3),
            velocity=np.array([19.5, 0.0, 0.0]),
        ),
        target_state=TargetState(
            position=np.array([100.0, 0.0, 0.0]),
            velocity=np.zeros(3),
        ),
        raw_acceleration=np.array([50.0, 0.0, 0.0]),
        dt=0.1,
    )

    assert np.linalg.norm(command.velocity_command) <= model.envelope.max_velocity_mps + 1e-9
    assert np.linalg.norm(command.acceleration_command) <= model.envelope.max_acceleration_mps2 + 1e-9
    assert status.acceleration_clipped
    assert status.velocity_clipped


def test_constraint_model_applies_safety_override_inside_minimum_separation() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    model = ConstraintModel(config)

    command, status = model.enforce_guidance_command(
        interceptor_state=TargetState(
            position=np.array([0.0, 0.0]),
            velocity=np.zeros(2),
        ),
        target_state=TargetState(
            position=np.array([1.0, 0.0]),
            velocity=np.zeros(2),
        ),
        raw_acceleration=np.array([0.0, 0.0, 0.0]),
        dt=0.1,
    )

    assert status.safety_override
    assert command.acceleration_command[0] < 0.0


def test_drift_and_tracking_constraints_are_checked() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    envelope = load_constraint_envelope(config)

    assert clamp_drift_rate(0.1, envelope) == 0.2
    assert clamp_drift_rate(0.7, envelope) == 0.5
    assert tracking_precision_ok(0.49, envelope)
    assert not tracking_precision_ok(0.51, envelope)
