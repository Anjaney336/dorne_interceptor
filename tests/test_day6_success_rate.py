from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.validation.day6 import run_day6_validation_sweep


def test_day6_validation_sweep_reaches_full_success_rate() -> None:
    metrics = run_day6_validation_sweep(
        project_root=ROOT,
        seeds=(41, 42, 43, 44),
        max_steps_override=160,
    )

    assert metrics.runs == 4
    assert metrics.success_rate == 1.0
