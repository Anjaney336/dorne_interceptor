from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from drone_interceptor.simulation.airsim_connection import connect_airsim
from drone_interceptor.types import ControlCommand


@dataclass(slots=True)
class AirSimCommandPacket:
    velocity_command: np.ndarray
    z_setpoint: float
    duration_s: float
    mode: str
    vehicle_name: str
    dispatched: bool


class AirSimInterceptorAdapter:
    """Translate interceptor commands into optional AirSim RPC calls."""

    def __init__(
        self,
        client: Any | None = None,
        vehicle_name: str = "",
        use_ned_z: bool = True,
    ) -> None:
        self._client = client
        self._vehicle_name = vehicle_name
        self._use_ned_z = bool(use_ned_z)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        connect: bool = False,
    ) -> "AirSimInterceptorAdapter":
        airsim_config = config.get("airsim", {})
        client = None
        if connect:
            client = connect_airsim(
                host=str(airsim_config.get("host", "127.0.0.1")),
                port=int(airsim_config.get("port", 41451)),
                timeout_value=int(airsim_config.get("timeout_s", 60)),
                vehicle_name=str(airsim_config.get("vehicle_name", "")),
            )
        return cls(
            client=client,
            vehicle_name=str(airsim_config.get("vehicle_name", "")),
            use_ned_z=bool(airsim_config.get("use_ned_z", True)),
        )

    def dispatch(
        self,
        command: ControlCommand,
        altitude_m: float,
        dt: float,
    ) -> AirSimCommandPacket:
        velocity = np.asarray(command.velocity_command, dtype=float).reshape(-1)
        if velocity.shape != (3,):
            raise ValueError("AirSim control expects a 3D velocity command.")

        z_setpoint = -float(altitude_m) if self._use_ned_z else float(altitude_m)
        packet = AirSimCommandPacket(
            velocity_command=velocity.copy(),
            z_setpoint=z_setpoint,
            duration_s=float(dt),
            mode="dry_run",
            vehicle_name=self._vehicle_name,
            dispatched=False,
        )

        if self._client is None:
            return packet

        if hasattr(self._client, "moveByVelocityZAsync"):
            future = self._client.moveByVelocityZAsync(
                vx=float(velocity[0]),
                vy=float(velocity[1]),
                z=float(z_setpoint),
                duration=float(dt),
                vehicle_name=self._vehicle_name,
            )
            packet.mode = "moveByVelocityZAsync"
        elif hasattr(self._client, "moveByVelocityAsync"):
            future = self._client.moveByVelocityAsync(
                vx=float(velocity[0]),
                vy=float(velocity[1]),
                vz=float(velocity[2]),
                duration=float(dt),
                vehicle_name=self._vehicle_name,
            )
            packet.mode = "moveByVelocityAsync"
        else:
            raise RuntimeError("AirSim client does not expose a supported velocity command API.")

        if hasattr(future, "join"):
            future.join()

        packet.dispatched = True
        return packet


__all__ = ["AirSimCommandPacket", "AirSimInterceptorAdapter"]
