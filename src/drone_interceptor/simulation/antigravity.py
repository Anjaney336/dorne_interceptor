import math
import numpy as np
from typing import Any
from drone_interceptor.simulation.airsim_manager import AirSimMissionManager, MultiTargetState, _augmented_proportional_navigation

def _antigravity_pn(
    interceptor_position: np.ndarray,
    interceptor_velocity: np.ndarray,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
    target_acceleration: np.ndarray,
    interceptor_speed_mps: float,
    dt: float,
    navigation_constant: float = 4.2,
) -> np.ndarray:
    # Get standard proportional navigation
    accel = _augmented_proportional_navigation(
        interceptor_position, interceptor_velocity, 
        target_position, target_velocity, target_acceleration, 
        interceptor_speed_mps, dt, navigation_constant
    )
    # Simulate 0g environment lack of vertical drag/gravity constraints
    # Controllers might over-command the Z-axis since there is no gravity to pull it down.
    # We amplify the Z command to simulate the over-oscillation effect in 0g.
    accel[2] *= 1.35 
    return accel

import drone_interceptor.simulation.airsim_manager as asm

class AntigravityMissionManager(AirSimMissionManager):
    """Testbed simulating 0g kinematics and high vertical thrust."""
    
    def run_replay(self, *args, **kwargs):
        # Override the PN module reference momentarily to apply antigravity physics
        original_pn = asm._augmented_proportional_navigation
        asm._augmented_proportional_navigation = _antigravity_pn
        try:
            return super().run_replay(*args, **kwargs)
        finally:
            asm._augmented_proportional_navigation = original_pn

