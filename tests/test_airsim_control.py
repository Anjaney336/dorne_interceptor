from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.simulation.airsim_control import AirSimInterceptorAdapter
from drone_interceptor.types import ControlCommand


class _Future:
    def __init__(self) -> None:
        self.joined = False

    def join(self) -> None:
        self.joined = True


class _Client:
    def __init__(self) -> None:
        self.calls: list[tuple[float, float, float, float, str]] = []
        self.future = _Future()

    def moveByVelocityZAsync(
        self,
        vx: float,
        vy: float,
        z: float,
        duration: float,
        vehicle_name: str = "",
    ) -> _Future:
        self.calls.append((vx, vy, z, duration, vehicle_name))
        return self.future


def test_airsim_adapter_formats_dry_run_command() -> None:
    adapter = AirSimInterceptorAdapter(client=None, vehicle_name="Interceptor", use_ned_z=True)
    packet = adapter.dispatch(
        command=ControlCommand(
            velocity_command=np.array([3.0, -1.0, 0.2]),
            acceleration_command=np.array([0.0, 0.0, 0.0]),
        ),
        altitude_m=120.0,
        dt=0.1,
    )

    assert packet.dispatched is False
    assert packet.mode == "dry_run"
    assert np.allclose(packet.velocity_command, np.array([3.0, -1.0, 0.2]))
    assert packet.z_setpoint == -120.0


def test_airsim_adapter_dispatches_supported_rpc_call() -> None:
    client = _Client()
    adapter = AirSimInterceptorAdapter(client=client, vehicle_name="Interceptor", use_ned_z=True)
    packet = adapter.dispatch(
        command=ControlCommand(
            velocity_command=np.array([4.0, 2.0, 0.0]),
            acceleration_command=np.array([0.0, 0.0, 0.0]),
        ),
        altitude_m=95.0,
        dt=0.2,
    )

    assert packet.dispatched is True
    assert client.calls == [(4.0, 2.0, -95.0, 0.2, "Interceptor")]
    assert client.future.joined is True
