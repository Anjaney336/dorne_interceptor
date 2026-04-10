from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def _run(script_name: str) -> None:
    script = ROOT / 'scripts' / script_name
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"{script_name} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


def test_manifest_validator_runs() -> None:
    _run('validate_repo_manifests.py')


def test_traceability_validator_runs() -> None:
    _run('validate_requirements_traceability.py')


def test_performance_budget_gate_runs() -> None:
    _run('check_performance_budgets.py')
