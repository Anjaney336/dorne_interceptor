from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.simulation.airsim_connection import AirSimPosition, format_position, get_drone_position


class _Vector3r:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x_val = x
        self.y_val = y
        self.z_val = z


class _State:
    def __init__(self, position: _Vector3r) -> None:
        self.kinematics_estimated = type("Kinematics", (), {"position": position})()


class _Pose:
    def __init__(self, position: _Vector3r) -> None:
        self.position = position


class _Client:
    def __init__(self, state_position: _Vector3r | None, pose_position: _Vector3r | None = None) -> None:
        self._state_position = state_position
        self._pose_position = pose_position

    def getMultirotorState(self, vehicle_name: str = "") -> object:
        if self._state_position is None:
            return type("State", (), {"kinematics_estimated": None})()
        return _State(self._state_position)

    def simGetVehiclePose(self, vehicle_name: str = "") -> object:
        return _Pose(self._pose_position)


def test_get_drone_position_reads_kinematics_position() -> None:
    client = _Client(state_position=_Vector3r(1.0, 2.0, -3.0))

    position = get_drone_position(client)

    assert position == AirSimPosition(x=1.0, y=2.0, z=-3.0)


def test_get_drone_position_falls_back_to_pose() -> None:
    client = _Client(state_position=None, pose_position=_Vector3r(4.0, 5.0, -6.0))

    position = get_drone_position(client)

    assert position == AirSimPosition(x=4.0, y=5.0, z=-6.0)


def test_format_position_outputs_readable_coordinates() -> None:
    formatted = format_position(AirSimPosition(x=1.23456, y=2.0, z=-3.4567))

    assert formatted == "x=1.235, y=2.000, z=-3.457"
