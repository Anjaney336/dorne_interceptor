from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.autonomy.system import AutonomousInterceptorSystem
from drone_interceptor.config import load_config
from drone_interceptor.control.controller import InterceptionController
from drone_interceptor.navigation.state_estimator import GPSIMUKalmanFusion, simulate_gps_with_drift
from drone_interceptor.prediction.predictor import TargetPredictor
from drone_interceptor.types import Plan, SensorPacket, TargetState


def test_navigation_drift_model_and_fusion() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    navigator = GPSIMUKalmanFusion(config)
    drifted = simulate_gps_with_drift(np.array([1.0, 2.0, 3.0]), time_s=10.0, drift_rate_mps=0.1)
    assert np.allclose(drifted, np.array([2.0, 2.0, 3.0]))

    state = navigator.update(
        SensorPacket(
            gps_position=np.array([5.0, 0.0, 1.0]),
            imu_acceleration=np.array([0.0, 0.0, 0.0]),
            timestamp=0.0,
        )
    )
    assert state.position.shape == (3,)
    assert state.velocity.shape == (3,)


def test_predictor_and_controller_generate_commands() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    predictor = TargetPredictor(config)
    controller = InterceptionController(config)
    track = TargetState(
        position=np.array([100.0, 20.0, 5.0]),
        velocity=np.array([-4.0, 1.0, 0.0]),
        acceleration=np.array([0.2, 0.0, 0.0]),
    )
    predicted = predictor.predict(track)
    assert len(predicted) == int(config["prediction"]["horizon_steps"])

    command = controller.compute_command(
        interceptor_state=TargetState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
        ),
        plan=Plan(
            intercept_point=predicted[0].position,
            desired_velocity=np.array([5.0, 0.0, 0.0]),
            desired_acceleration=np.array([0.1, 0.0, 0.0]),
            metadata={"target_velocity": predicted[0].velocity},
        ),
    )
    assert command.velocity_command.shape == (3,)
    assert command.acceleration_command is not None


def test_autonomous_system_runs_and_writes_outputs(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["mission"]["max_steps"] = 12
    config["visualization"]["output_dir"] = str(tmp_path)
    system = AutonomousInterceptorSystem(config)

    result = system.run()

    assert result.steps_executed > 0
    assert result.mean_loop_fps > 0.0
    assert len(result.output_paths) == 2
    for output_path in result.output_paths:
        assert output_path.exists()
