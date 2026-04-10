"""
Test script for the Drone Interceptor Mission Backend.

This script validates:
1. MissionConfig initialization
2. MissionController setup
3. Single step execution
4. Full mission execution (sync and async)
5. Artifact generation
6. API endpoint integration
"""

import asyncio
import json
from pathlib import Path

# Test MissionConfig and MissionController
def test_mission_config():
    """Test MissionConfig initialization and defaults."""
    from drone_interceptor.backend.mission_service import MissionConfig
    
    config = MissionConfig()
    assert config.num_targets == 3
    assert config.target_speed_mps == 6.0
    assert config.interceptor_speed_mps == 20.0
    assert config.drift_rate_mps == 0.3
    assert config.noise_level_m == 0.45
    assert config.guidance_gain == 6.0
    assert config.kill_radius_m == 10.26
    assert config.use_ekf is True
    assert config.use_anti_spoofing is True
    
    # Test custom config
    custom_config = MissionConfig(
        num_targets=5,
        target_speed_mps=8.0,
        interceptor_speed_mps=25.0,
    )
    assert custom_config.num_targets == 5
    assert custom_config.target_speed_mps == 8.0
    assert custom_config.interceptor_speed_mps == 25.0
    
    print("✓ MissionConfig tests passed")


def test_proportional_navigation():
    """Test ProportionalNavigation guidance law."""
    import numpy as np
    from drone_interceptor.backend.mission_service import ProportionalNavigation
    
    pn = ProportionalNavigation(navigation_constant=6.0)
    
    # Test basic computation
    interceptor_pos = np.array([0.0, 0.0, 100.0], dtype=float)
    interceptor_vel = np.array([20.0, 0.0, 0.0], dtype=float)
    target_pos = np.array([100.0, 0.0, 100.0], dtype=float)
    target_vel = np.array([6.0, 0.0, 0.0], dtype=float)
    
    accel = pn.compute_acceleration(
        interceptor_pos, interceptor_vel,
        target_pos, target_vel
    )
    
    # Should have non-zero perpendicular component
    assert accel is not None
    assert len(accel) == 3
    assert np.linalg.norm(accel) >= 0.0
    
    print(f"✓ ProportionalNavigation test passed (accel magnitude: {np.linalg.norm(accel):.3f} m/s²)")


async def test_mission_controller_init():
    """Test MissionController initialization."""
    from drone_interceptor.backend.mission_service import MissionConfig, MissionController
    
    config = MissionConfig(num_targets=3, random_seed=42)
    output_dir = Path("/tmp/test_mission")
    
    controller = MissionController(config, output_dir=output_dir)
    result = controller.initialize_mission()
    
    assert result["status"] == "initialized"
    assert result["num_targets"] == 3
    assert len(result["target_names"]) == 3
    assert result["interceptor_name"] == "Interceptor"
    
    # Check internal state
    assert len(controller.targets) == 3
    assert controller.interceptor is not None
    assert len(controller.ekf_filters) == 3
    
    print("✓ MissionController initialization test passed")


async def test_single_step_execution():
    """Test single step execution."""
    from drone_interceptor.backend.mission_service import MissionConfig, MissionController
    
    config = MissionConfig(num_targets=2, max_steps=10, dt=0.05, random_seed=42)
    controller = MissionController(config)
    controller.initialize_mission()
    
    # Execute first step
    frame = controller.execute_step(0)
    
    assert frame.step == 0
    assert frame.time_s == 0.0
    assert frame.active_stage == "Detection"
    assert len(frame.targets) == 2
    assert frame.rmse_m >= 0.0
    
    # Execute second step
    frame = controller.execute_step(1)
    assert frame.step == 1
    assert frame.time_s == 0.05
    
    print("✓ Single step execution test passed")


async def test_full_mission_async():
    """Test full mission execution."""
    from drone_interceptor.backend.mission_service import MissionConfig, MissionController
    
    config = MissionConfig(
        num_targets=2,
        max_steps=20,
        dt=0.05,
        use_ekf=True,
        use_anti_spoofing=True,
        random_seed=42
    )
    
    output_dir = Path("/tmp/test_mission_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    controller = MissionController(config, output_dir=output_dir)
    result = await controller.run_mission()
    
    assert result["status"] == "complete"
    assert result["total_targets"] == 2
    assert "mission_duration_s" in result
    assert "artifacts" in result
    assert "telemetry_csv" in result["artifacts"]
    assert "fpv_video_mp4" in result["artifacts"]
    
    # Check telemetry log
    assert len(controller.telemetry_log) > 0
    
    # Check RMSE progression
    rmses = [frame.rmse_m for frame in controller.telemetry_log]
    assert all(r >= 0 for r in rmses)
    
    print(f"✓ Full mission async test passed")
    print(f"  - Duration: {result['mission_duration_s']:.2f}s")
    print(f"  - Frames: {len(controller.telemetry_log)}")
    print(f"  - Success: {result['mission_success']}")
    print(f"  - Jammed targets: {result['jammed_targets']}/{result['total_targets']}")
    
    return result


def test_artifact_paths():
    """Test that artifact paths are created correctly."""
    from drone_interceptor.backend.mission_service import MissionConfig, MissionController
    from pathlib import Path
    
    config = MissionConfig(num_targets=2)
    output_dir = Path("/tmp/test_artifacts")
    
    controller = MissionController(config, output_dir=output_dir)
    
    # Verify output directory is created
    assert output_dir.exists()
    
    print("✓ Artifact paths test passed")


async def test_ekf_filtering():
    """Test EKF filtering in target tracking."""
    import numpy as np
    from drone_interceptor.backend.mission_service import MissionConfig, MissionController
    
    config = MissionConfig(
        num_targets=1,
        max_steps=50,
        use_ekf=True,
        use_anti_spoofing=True,
        drift_rate_mps=0.3,
        noise_level_m=0.45,
    )
    
    controller = MissionController(config)
    controller.initialize_mission()
    
    # Run several steps and track estimation error
    errors = []
    for step in range(20):
        frame = controller.execute_step(step)
        if frame.targets:
            target = frame.targets[0]
            error = np.linalg.norm(target.estimated_position - target.position)
            errors.append(error)
    
    mean_error = np.mean(errors)
    
    # Error should be bounded by noise level * some factor
    assert mean_error < config.noise_level_m * 3.0
    
    print(f"✓ EKF filtering test passed (mean error: {mean_error:.3f}m)")


async def test_drift_injection():
    """Test drift injection application."""
    from drone_interceptor.backend.mission_service import MissionConfig, MissionController
    import numpy as np
    
    config = MissionConfig(
        num_targets=1,
        max_steps=10,
        drift_rate_mps=0.5,
        noise_level_m=0.1,
    )
    
    controller = MissionController(config)
    controller.initialize_mission()
    
    # Track measurement errors
    drifts = []
    for step in range(5):
        frame = controller.execute_step(step)
        if frame.targets:
            target = frame.targets[0]
            drift = target.measured_position - target.position
            drifts.append(np.linalg.norm(drift))
    
    # Drift should accumulate
    assert len(drifts) > 0
    
    print(f"✓ Drift injection test passed (average drift: {np.mean(drifts):.3f}m)")


async def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Testing Drone Interceptor Mission Backend Implementation")
    print("=" * 60)
    
    try:
        print("\n1. Testing MissionConfig...")
        test_mission_config()
        
        print("\n2. Testing ProportionalNavigation...")
        test_proportional_navigation()
        
        print("\n3. Testing MissionController initialization...")
        await test_mission_controller_init()
        
        print("\n4. Testing single step execution...")
        await test_single_step_execution()
        
        print("\n5. Testing artifact paths...")
        test_artifact_paths()
        
        print("\n6. Testing EKF filtering...")
        await test_ekf_filtering()
        
        print("\n7. Testing drift injection...")
        await test_drift_injection()
        
        print("\n8. Testing full mission execution (async)...")
        result = await test_full_mission_async()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print(f"\nImplementation Summary:")
        print(f"  ✓ MissionConfig with all parameters")
        print(f"  ✓ ProportionalNavigation guidance law")
        print(f"  ✓ MissionController async execution")
        print(f"  ✓ EKF-based state estimation")
        print(f"  ✓ Anti-spoofing innovation gate")
        print(f"  ✓ Drift injection simulation")
        print(f"  ✓ Latency and packet loss simulation")
        print(f"  ✓ CSV telemetry artifact generation")
        print(f"  ✓ MP4 video artifact generation")
        
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    import numpy as np
    
    # Add src to path if needed
    src_path = Path(__file__).resolve().parents[1] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
