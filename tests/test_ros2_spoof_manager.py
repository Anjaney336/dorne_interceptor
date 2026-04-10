from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.ros2.spoof_manager import (  # noqa: E402
    DefensiveDriftPlanner,
    GeoFix,
    SafetyInterlock,
    SafetyInterlockConfig,
    SpoofManagerConfig,
    SpoofManagerCore,
)


def test_safety_interlock_power_limit_scales_with_distance() -> None:
    interlock = SafetyInterlock(
        SafetyInterlockConfig(
            min_safe_distance_m=1.0,
            fade_distance_m=4.0,
            min_power_dbm=-40.0,
            max_power_dbm=-10.0,
        )
    )
    near = interlock.power_limit_dbm(0.5)
    mid = interlock.power_limit_dbm(2.0)
    far = interlock.power_limit_dbm(6.0)
    assert near <= mid <= far
    assert near == -40.0
    assert far == -10.0


def test_safety_interlock_avoids_guard_bands() -> None:
    interlock = SafetyInterlock()
    frequency, switched = interlock.choose_frequency_hz(
        desired_frequency_hz=2.41e9,
        interference_frequency_hz=2.412e9,
    )
    assert switched is True
    assert abs(frequency - 2.4e9) > 60e6


def test_defensive_drift_planner_returns_finite_offsets() -> None:
    planner = DefensiveDriftPlanner(lead_gain=0.35, max_offset_m=30.0)
    fix = GeoFix(lat_deg=37.7749, lon_deg=-122.4194, alt_m=30.0, timestamp_s=0.0)
    plan = planner.plan(fix=fix, relative_xyz_m=np.array([100.0, -40.0, 8.0], dtype=float))
    assert np.isfinite(plan.spoof_lat_deg)
    assert np.isfinite(plan.spoof_lon_deg)
    assert abs(plan.east_offset_m) <= 30.0
    assert abs(plan.north_offset_m) <= 30.0


def test_spoof_manager_core_emits_confidence_and_heatmap(tmp_path: Path) -> None:
    core = SpoofManagerCore(
        config=SpoofManagerConfig(
            spoof_enable=True,
            log_path=tmp_path / "spoof_telemetry.jsonl",
            heatmap_bins=6,
        )
    )
    fix = GeoFix(lat_deg=37.7749, lon_deg=-122.4194, alt_m=30.0, timestamp_s=0.0)
    payload = core.update(
        fix=fix,
        relative_xyz_m=np.array([80.0, 10.0, 2.0], dtype=float),
        detection_confidence=0.88,
    )
    assert 0.0 <= float(payload["spoof_confidence_score"]) <= 1.0
    heatmap = payload["sdr_heatmap"]
    assert isinstance(heatmap, list)
    assert len(heatmap) == 6
    assert len(heatmap[0]) == 6
