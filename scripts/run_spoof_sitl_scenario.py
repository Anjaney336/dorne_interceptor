from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from drone_interceptor.ros2.spoof_manager import GeoFix, SpoofManagerConfig, SpoofManagerCore


def _attacker_relative_position(step: int, dt: float, approach_speed_mps: float = 7.0) -> np.ndarray:
    time_s = float(step) * float(dt)
    x_m = max(180.0 - approach_speed_mps * time_s, -25.0)
    y_m = 25.0 * math.sin(0.12 * time_s)
    z_m = 6.0 * math.sin(0.08 * time_s)
    return np.array([x_m, y_m, z_m], dtype=float)


def run_sitl_scenario(
    sim_mode: str,
    steps: int,
    dt: float,
    output_path: Path,
    spoof_enable: bool,
) -> list[dict[str, object]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    core = SpoofManagerCore(
        config=SpoofManagerConfig(
            spoof_enable=bool(spoof_enable),
            log_path=output_path.parent / "spoof_manager_telemetry.jsonl",
        )
    )
    base_fix = GeoFix(
        lat_deg=37.7749,
        lon_deg=-122.4194,
        alt_m=32.0,
        timestamp_s=time.time(),
    )

    rows: list[dict[str, object]] = []
    for step in range(max(int(steps), 1)):
        relative = _attacker_relative_position(step=step, dt=dt)
        confidence = float(np.clip(0.92 - 0.002 * step, 0.45, 0.92))
        fix = GeoFix(
            lat_deg=base_fix.lat_deg,
            lon_deg=base_fix.lon_deg,
            alt_m=base_fix.alt_m,
            timestamp_s=time.time(),
        )
        status = core.update(fix=fix, relative_xyz_m=relative, detection_confidence=confidence)
        rows.append(
            {
                "step": int(step),
                "time_s": round(float(step * dt), 3),
                "sim_mode": str(sim_mode),
                "attacker_relative_m": relative.round(4).tolist(),
                "status": str(status.get("status", "UNKNOWN")),
                "spoof_confidence_score": float(status.get("spoof_confidence_score", 0.0)),
                "interlock_power_dbm": float(
                    (status.get("safety_interlock") or {}).get("power_limit_dbm", 0.0)
                ),
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="SITL spoof-check scenario runner (defensive, dry-run only).")
    parser.add_argument("--sim", choices=("synthetic", "airsim", "gazebo"), default="synthetic")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--spoof-enable", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "sitl_spoof_scenario.jsonl",
    )
    args = parser.parse_args()

    rows = run_sitl_scenario(
        sim_mode=str(args.sim),
        steps=int(args.steps),
        dt=float(args.dt),
        output_path=Path(args.output),
        spoof_enable=bool(args.spoof_enable),
    )
    acquired = sum(1 for row in rows if str(row.get("status", "")).startswith("TARGET ACQUIRED"))
    active = sum(1 for row in rows if "ACTIVE" in str(row.get("status", "")))
    print(
        json.dumps(
            {
                "sim_mode": str(args.sim),
                "samples": len(rows),
                "target_acquired_samples": acquired,
                "spoof_active_samples": active,
                "output": str(Path(args.output)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
