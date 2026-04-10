from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.control.guidance import ProportionalNavigationGuidance
from drone_interceptor.types import TargetState


def test_proportional_navigation_guidance_returns_control_command() -> None:
    guidance = ProportionalNavigationGuidance(
        dt=0.1,
        navigation_constant=3.0,
        max_acceleration=20.0,
        max_speed=60.0,
    )
    interceptor = TargetState(
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([10.0, 0.0, 0.0]),
    )
    target = TargetState(
        position=np.array([100.0, 10.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
    )

    command = guidance.compute_command(interceptor_state=interceptor, target_state=target)

    assert command.acceleration_command is not None
    assert command.acceleration_command.shape == (3,)
    assert command.velocity_command.shape == (3,)
    assert command.metadata["time_to_go"] > 0.0
    assert command.metadata["closing_speed"] > 0.0
    assert command.acceleration_command[1] > 0.0


def test_proportional_navigation_guidance_limits_acceleration_norm() -> None:
    guidance = ProportionalNavigationGuidance(
        dt=0.1,
        navigation_constant=6.0,
        max_acceleration=5.0,
        max_speed=60.0,
        time_to_go_gain=3.0,
    )
    interceptor = TargetState(
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
    )
    target = TargetState(
        position=np.array([2.0, 20.0]),
        velocity=np.array([0.0, -5.0]),
    )

    command = guidance.compute_command(interceptor_state=interceptor, target_state=target)

    assert np.linalg.norm(command.acceleration_command) <= 5.0 + 1e-9
