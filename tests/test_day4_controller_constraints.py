from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.config import load_config
from drone_interceptor.control.controller import InterceptionController
from drone_interceptor.types import Plan, TargetState


def test_controller_uses_current_target_for_safety_constraints() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["control"]["mode"] = "mpc"

    controller = InterceptionController(config)
    interceptor = TargetState(
        position=np.zeros(3, dtype=float),
        velocity=np.zeros(3, dtype=float),
        acceleration=np.zeros(3, dtype=float),
    )
    plan = Plan(
        intercept_point=np.array([2.0, 0.0, 0.0], dtype=float),
        desired_velocity=np.array([5.0, 0.0, 0.0], dtype=float),
        desired_acceleration=np.zeros(3, dtype=float),
        metadata={
            "target_velocity": np.zeros(3, dtype=float),
            "target_acceleration": np.zeros(3, dtype=float),
            "current_target_position": np.array([50.0, 0.0, 0.0], dtype=float),
            "current_target_velocity": np.zeros(3, dtype=float),
            "current_target_acceleration": np.zeros(3, dtype=float),
            "tracking_error_m": 0.0,
        },
    )

    command = controller.compute_command(interceptor, plan)

    assert command.metadata["safety_override"] is False
