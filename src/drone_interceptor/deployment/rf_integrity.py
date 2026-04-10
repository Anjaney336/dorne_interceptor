from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RFIntegrityChecklist:
    simulation_only: bool
    gps_module_shielded: bool
    sdr_enclosure_shielded: bool
    antenna_separation_verified: bool
    front_end_filtering_present: bool
    power_isolation_present: bool
    legal_test_mode_declared: bool
    measured_isolation_db: float | None
    ready: bool
    notes: tuple[str, ...]


def build_rf_integrity_manifest(
    output_path: str | Path,
    measured_isolation_db: float | None = None,
    simulation_only: bool = True,
) -> Path:
    isolation_ready = measured_isolation_db is not None and float(measured_isolation_db) >= 40.0
    checklist = RFIntegrityChecklist(
        simulation_only=bool(simulation_only),
        gps_module_shielded=False,
        sdr_enclosure_shielded=False,
        antenna_separation_verified=False,
        front_end_filtering_present=False,
        power_isolation_present=False,
        legal_test_mode_declared=bool(simulation_only),
        measured_isolation_db=None if measured_isolation_db is None else float(measured_isolation_db),
        ready=bool(isolation_ready and not simulation_only),
        notes=(
            "This repository remains in simulation-only RF mode until shielding and compliance measurements are captured on hardware.",
            "Required hardware evidence: enclosure shielding, antenna isolation, filter chain validation, and regulated test-environment signoff.",
        ),
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(checklist), indent=2), encoding="utf-8")
    return destination


__all__ = ["RFIntegrityChecklist", "build_rf_integrity_manifest"]
