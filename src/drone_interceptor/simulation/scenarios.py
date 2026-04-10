from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    description: str
    noise_level: float
    zigzag_amplitude_mps: float = 0.0
    zigzag_frequency_hz: float = 0.0


@dataclass(frozen=True)
class DP5ScenarioDefinition:
    name: str
    description: str
    drift_mode: str
    simulation_overrides: dict[str, Any]
    navigation_overrides: dict[str, Any]


def platform_scenarios() -> list[ScenarioDefinition]:
    return [
        ScenarioDefinition(
            name="slow_drone",
            description="Nominal low-speed target with mild process disturbance.",
            noise_level=0.25,
        ),
        ScenarioDefinition(
            name="fast_drone",
            description="High-speed target stressing pursuit closure.",
            noise_level=0.25,
        ),
        ScenarioDefinition(
            name="zig_zag_motion",
            description="Target introduces lateral weaving while interceptor replans.",
            noise_level=0.35,
            zigzag_amplitude_mps=1.8,
            zigzag_frequency_hz=0.18,
        ),
        ScenarioDefinition(
            name="noisy_environment",
            description="Elevated measurement and process noise through the full stack.",
            noise_level=1.05,
        ),
        ScenarioDefinition(
            name="high_drift",
            description="High GNSS drift with Kalman fusion correction in the loop.",
            noise_level=0.55,
        ),
    ]


def dp5_scenario_matrix() -> list[DP5ScenarioDefinition]:
    return [
        DP5ScenarioDefinition(
            name="single_target_nominal",
            description="Single rogue drone with nominal redirection dynamics.",
            drift_mode="directed",
            simulation_overrides={
                "target_initial_position": [265.0, 132.0, 120.0],
                "target_initial_velocity": [-5.8, 1.5, 0.0],
                "target_process_noise_std_mps2": 0.10,
                "wind_disturbance_std_mps2": 0.04,
            },
            navigation_overrides={"packet_loss_rate": 0.0},
        ),
        DP5ScenarioDefinition(
            name="packet_loss_stress",
            description="Directed drift under aggressive packet loss.",
            drift_mode="directed",
            simulation_overrides={
                "target_initial_position": [285.0, 145.0, 120.0],
                "target_initial_velocity": [-6.0, 1.7, 0.0],
                "target_process_noise_std_mps2": 0.12,
                "wind_disturbance_std_mps2": 0.05,
            },
            navigation_overrides={"packet_loss_rate": 0.25},
        ),
        DP5ScenarioDefinition(
            name="urban_canyon",
            description="Low-altitude urban-canyon style GNSS disturbance.",
            drift_mode="directed",
            simulation_overrides={
                "target_initial_position": [240.0, 110.0, 95.0],
                "target_initial_velocity": [-5.0, 1.1, 0.0],
                "target_process_noise_std_mps2": 0.16,
                "wind_disturbance_std_mps2": 0.08,
            },
            navigation_overrides={"gps_noise_std_m": 0.35},
        ),
        DP5ScenarioDefinition(
            name="evasive_target",
            description="Fast evasive target with circular spoofing pattern.",
            drift_mode="circular",
            simulation_overrides={
                "target_initial_position": [310.0, 160.0, 120.0],
                "target_initial_velocity": [-7.6, 2.8, 0.0],
                "target_process_noise_std_mps2": 0.18,
                "wind_disturbance_std_mps2": 0.08,
            },
            navigation_overrides={"gps_noise_std_m": 0.25},
        ),
    ]


def build_platform_scenario_config(
    base_config: dict[str, Any],
    scenario: ScenarioDefinition,
    random_seed: int,
    max_steps_override: int | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config.setdefault("system", {})["random_seed"] = int(random_seed)
    if max_steps_override is not None:
        config.setdefault("mission", {})["max_steps"] = int(max_steps_override)

    simulation = config.setdefault("simulation", {})
    perception = config.setdefault("perception", {})
    tracking = config.setdefault("tracking", {})
    navigation = config.setdefault("navigation", {})

    if scenario.name == "slow_drone":
        simulation["target_initial_position"] = [235.0, 125.0, 120.0]
        simulation["target_initial_velocity"] = [-4.2, 1.0, 0.0]
        simulation["target_max_acceleration_mps2"] = 2.8
        simulation["target_process_noise_std_mps2"] = 0.18
        simulation["wind_disturbance_std_mps2"] = 0.06
    elif scenario.name == "fast_drone":
        simulation["target_initial_position"] = [320.0, 170.0, 120.0]
        simulation["target_initial_velocity"] = [-9.2, 3.1, 0.0]
        simulation["target_max_acceleration_mps2"] = 4.6
        simulation["target_process_noise_std_mps2"] = 0.24
        simulation["wind_disturbance_std_mps2"] = 0.08
    elif scenario.name == "zig_zag_motion":
        simulation["target_initial_position"] = [285.0, 150.0, 120.0]
        simulation["target_initial_velocity"] = [-6.3, 0.0, 0.0]
        simulation["target_max_acceleration_mps2"] = 3.8
        simulation["target_process_noise_std_mps2"] = 0.12
        simulation["wind_disturbance_std_mps2"] = 0.05
    elif scenario.name == "noisy_environment":
        simulation["target_initial_position"] = [255.0, 140.0, 120.0]
        simulation["target_initial_velocity"] = [-6.0, 1.8, 0.0]
        simulation["target_process_noise_std_mps2"] = 0.22
        simulation["wind_disturbance_std_mps2"] = 0.09
        perception["synthetic_measurement_noise_std_m"] = 1.05
        tracking["measurement_noise"] = 0.45
        tracking["process_noise"] = 0.12
        tracking["acceleration_smoothing"] = max(float(tracking.get("acceleration_smoothing", 0.8)), 0.82)
        config.setdefault("constraints", {}).setdefault("tracking", {})["max_position_error_m"] = max(
            float(config["constraints"]["tracking"].get("max_position_error_m", 0.75)),
            1.0,
        )
    elif scenario.name == "high_drift":
        simulation["target_initial_position"] = [275.0, 145.0, 120.0]
        simulation["target_initial_velocity"] = [-6.1, 2.0, 0.0]
        simulation["target_process_noise_std_mps2"] = 0.18
        simulation["wind_disturbance_std_mps2"] = 0.06
        navigation["gps_drift_rate_mps"] = 0.42
        navigation["gps_noise_std_m"] = 1.75
        navigation["measurement_noise_scale"] = 1.10
        config.setdefault("planning", {})["desired_intercept_distance_m"] = max(
            float(config["planning"].get("desired_intercept_distance_m", 11.0)),
            12.0,
        )

    config.setdefault("scenario", {}).update(
        {
            "name": scenario.name,
            "description": scenario.description,
            "noise_level": scenario.noise_level,
            "zigzag_amplitude_mps": scenario.zigzag_amplitude_mps,
            "zigzag_frequency_hz": scenario.zigzag_frequency_hz,
        }
    )
    return config


__all__ = [
    "DP5ScenarioDefinition",
    "ScenarioDefinition",
    "build_platform_scenario_config",
    "dp5_scenario_matrix",
    "platform_scenarios",
]
