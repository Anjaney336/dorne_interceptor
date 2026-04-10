from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.validation.day10 import run_day10_execution


def test_day10_execution_writes_evidence_artifacts() -> None:
    summary, artifacts = run_day10_execution(
        project_root=ROOT,
        benchmark_total_runs=8,
        base_random_seed=97,
    )

    assert artifacts.benchmark_csv.exists()
    assert artifacts.summary_json.exists()
    assert artifacts.detector_benchmark_json.exists()
    assert artifacts.edge_manifest_json.exists()
    assert artifacts.sitl_bag_jsonl.exists()
    assert artifacts.sitl_replay_json.exists()
    assert artifacts.rf_integrity_json.exists()
    assert summary.tracking_precision_ratio >= 0.0
    assert summary.peak_speed_mps > 0.0

    detector_payload = json.loads(artifacts.detector_benchmark_json.read_text(encoding="utf-8"))
    assert "domain_analysis" in detector_payload
    assert "effective_benchmark" in detector_payload

    sitl_payload = json.loads(artifacts.sitl_replay_json.read_text(encoding="utf-8"))
    assert sitl_payload["messages"] > 0
    assert "latency_budget_ms" in sitl_payload
