from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.ros2.spoof_manager import (  # noqa: E402
    GeoFix,
    SDRDryRunInterface,
    SpoofManagerConfig,
    SpoofManagerCore,
)


def _ping_host(host: str, timeout_s: float) -> tuple[bool, str]:
    timeout_ms = max(int(float(timeout_s) * 1000), 200)
    if platform.system().lower().startswith("win"):
        command = ["ping", "-n", "1", "-w", str(timeout_ms), host]
    else:
        timeout_value = max(int(timeout_s), 1)
        command = ["ping", "-c", "1", "-W", str(timeout_value), host]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except Exception as exc:
        return False, str(exc)
    return completed.returncode == 0, (completed.stdout or completed.stderr).strip()[:400]


def run_diagnostics(
    jetson_host: str,
    timeout_s: float,
    spoof_enable: bool,
) -> dict[str, object]:
    ping_ok, ping_output = _ping_host(jetson_host, timeout_s=timeout_s)
    sdr = SDRDryRunInterface()
    runtime = sdr.inspect_runtime(timeout_s=timeout_s)

    core = SpoofManagerCore(
        config=SpoofManagerConfig(
            spoof_enable=bool(spoof_enable),
            log_path=Path("logs") / "spoof_manager_telemetry.jsonl",
        )
    )
    fix = GeoFix(
        lat_deg=37.7749,
        lon_deg=-122.4194,
        alt_m=32.0,
        timestamp_s=time.time(),
    )
    relative = np.array([120.0, -18.0, 4.0], dtype=float)
    dry_run = core.update(fix=fix, relative_xyz_m=relative, detection_confidence=0.86)
    dry_run_ok = bool(
        np.isfinite(float(dry_run.get("spoof_confidence_score", 0.0)))
        and "drift_plan" in dry_run
        and "sdr_heatmap" in dry_run
    )

    return {
        "jetson_ping_ok": bool(ping_ok),
        "jetson_ping_output": ping_output,
        "hackrf_info_available": bool(runtime.hackrf_info_available),
        "hackrf_info_ok": bool(runtime.hackrf_info_ok),
        "gps_sdr_sim_available": bool(runtime.gps_sdr_sim_available),
        "dry_run_math_ok": bool(dry_run_ok),
        "spoof_confidence_score": float(dry_run.get("spoof_confidence_score", 0.0)),
        "status": str(dry_run.get("status", "UNKNOWN")),
        "note": "This utility performs defensive dry-run checks only. RF transmission is not executed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Spoof-check utility (defensive dry-run only).")
    parser.add_argument("--jetson-host", type=str, default="192.168.55.1")
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--spoof-enable", action="store_true", help="Enable dry-run spoof-manager state.")
    parser.add_argument(
        "--strict-hardware",
        action="store_true",
        help="Fail if HackRF tools are missing or not initialized.",
    )
    args = parser.parse_args()

    result = run_diagnostics(
        jetson_host=str(args.jetson_host),
        timeout_s=float(args.timeout_s),
        spoof_enable=bool(args.spoof_enable),
    )
    print(json.dumps(result, indent=2))

    checks = [bool(result["jetson_ping_ok"]), bool(result["dry_run_math_ok"])]
    if bool(args.strict_hardware):
        checks.append(bool(result["hackrf_info_available"]))
        checks.append(bool(result["hackrf_info_ok"]))
    if all(checks):
        raise SystemExit(0)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
