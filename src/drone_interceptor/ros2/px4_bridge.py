from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from drone_interceptor.types import ControlCommand


class PX4OffboardCommandAdapter:
    """Converts internal commands into PX4-style NED velocity and acceleration setpoints."""

    def to_offboard_setpoint(self, command: ControlCommand) -> dict[str, list[float] | str]:
        velocity = np.asarray(command.velocity_command, dtype=float)
        acceleration = (
            np.asarray(command.acceleration_command, dtype=float)
            if command.acceleration_command is not None
            else np.zeros(3, dtype=float)
        )
        return {
            "coordinate_frame": "NED",
            "mode": command.mode,
            "velocity_setpoint": [float(velocity[0]), float(velocity[1]), float(-velocity[2])],
            "acceleration_setpoint": [float(acceleration[0]), float(acceleration[1]), float(-acceleration[2])],
        }


@dataclass(slots=True)
class PX4SITLPacket:
    mode: str
    vehicle_model: str
    offboard_setpoint: dict[str, list[float] | str]
    sitl_command: str
    target_system: str


class PX4SITLAdapter:
    """Packages internal commands into PX4 SITL-ready offboard packets."""

    def __init__(
        self,
        px4_root: str | Path = "PX4-Autopilot",
        vehicle_model: str = "iris",
        target_system: str = "px4_sitl",
    ) -> None:
        self._px4_root = Path(px4_root)
        self._vehicle_model = vehicle_model
        self._target_system = target_system

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PX4SITLAdapter":
        airsim = config.get("airsim", {})
        return cls(
            px4_root=str(airsim.get("px4_root", "PX4-Autopilot")),
            vehicle_model=str(airsim.get("vehicle_model", "iris")),
            target_system=str(airsim.get("target_system", "px4_sitl")),
        )

    def command_packet(self, offboard_setpoint: dict[str, list[float] | str]) -> PX4SITLPacket:
        return PX4SITLPacket(
            mode="offboard_velocity",
            vehicle_model=self._vehicle_model,
            offboard_setpoint=offboard_setpoint,
            sitl_command=px4_sitl_command(self._px4_root, vehicle_model=self._vehicle_model),
            target_system=self._target_system,
        )


def px4_sitl_command(px4_root: str | Path, vehicle_model: str = "iris") -> str:
    root = Path(px4_root)
    return f"cd {root} && make px4_sitl gazebo_{vehicle_model} PX4_SYS_AUTOSTART=4001"
