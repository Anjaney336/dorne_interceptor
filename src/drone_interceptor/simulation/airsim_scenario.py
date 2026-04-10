from __future__ import annotations

from dataclasses import dataclass

from drone_interceptor.simulation.airsim_connection import connect_airsim, get_drone_position


@dataclass(slots=True)
class MultiDronePositions:
    interceptor: tuple[float, float, float]
    target: tuple[float, float, float]


class AirSimMultiDroneScenario:
    """Thin wrapper around AirSim for target and interceptor position sampling."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 41451,
        interceptor_name: str = "Interceptor",
        target_name: str = "Target",
    ) -> None:
        self._host = host
        self._port = port
        self._interceptor_name = interceptor_name
        self._target_name = target_name

    def sample_positions(self) -> MultiDronePositions:
        client = connect_airsim(host=self._host, port=self._port)
        interceptor = get_drone_position(client, vehicle_name=self._interceptor_name)
        target = get_drone_position(client, vehicle_name=self._target_name)
        return MultiDronePositions(
            interceptor=(interceptor.x, interceptor.y, interceptor.z),
            target=(target.x, target.y, target.z),
        )
