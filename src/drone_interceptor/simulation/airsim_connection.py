from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import numpy as np


if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[3]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


@dataclass(frozen=True)
class AirSimPosition:
    x: float
    y: float
    z: float

    def as_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


def connect_airsim(
    host: str = "127.0.0.1",
    port: int = 41451,
    timeout_value: int = 60,
    vehicle_name: str = "",
) -> Any:
    airsim = _import_airsim()
    client = airsim.MultirotorClient(ip=host, port=port, timeout_value=timeout_value)
    client.confirmConnection()
    # Touch the state API immediately so connection errors surface here.
    client.getMultirotorState(vehicle_name=vehicle_name)
    return client


def get_drone_position(client: Any, vehicle_name: str = "") -> AirSimPosition:
    state = client.getMultirotorState(vehicle_name=vehicle_name)
    estimated = getattr(state, "kinematics_estimated", None)
    if estimated is not None and getattr(estimated, "position", None) is not None:
        position = estimated.position
        return AirSimPosition(
            x=float(position.x_val),
            y=float(position.y_val),
            z=float(position.z_val),
        )

    pose = client.simGetVehiclePose(vehicle_name=vehicle_name)
    if getattr(pose, "position", None) is None:
        raise RuntimeError("AirSim returned a state without position data.")

    return AirSimPosition(
        x=float(pose.position.x_val),
        y=float(pose.position.y_val),
        z=float(pose.position.z_val),
    )


def format_position(position: AirSimPosition) -> str:
    return f"x={position.x:.3f}, y={position.y:.3f}, z={position.z:.3f}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Connect to AirSim and print the current drone coordinates.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="AirSim RPC host.")
    parser.add_argument("--port", type=int, default=41451, help="AirSim RPC port.")
    parser.add_argument("--timeout", type=int, default=60, help="AirSim RPC timeout in seconds.")
    parser.add_argument("--vehicle-name", type=str, default="", help="Optional AirSim vehicle name.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    client = connect_airsim(
        host=args.host,
        port=args.port,
        timeout_value=args.timeout,
        vehicle_name=args.vehicle_name,
    )
    position = get_drone_position(client, vehicle_name=args.vehicle_name)
    print("connected=True")
    print(format_position(position))


def _import_airsim() -> ModuleType:
    try:
        import airsim
    except ImportError as exc:
        raise ImportError(
            "AirSim Python client is not installed in the active environment. "
            "Install it before using the AirSim connection module."
        ) from exc

    return airsim


__all__ = [
    "AirSimPosition",
    "connect_airsim",
    "format_position",
    "get_drone_position",
]


if __name__ == "__main__":
    main()
