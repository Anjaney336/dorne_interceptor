from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.validation.day4 import run_day4_validation


def test_day4_validation_runs_and_writes_outputs() -> None:
    statuses, metrics, artifacts = run_day4_validation(
        project_root=ROOT,
        control_runs=2,
        random_seed=13,
        max_steps_override=20,
        use_airsim=False,
    )

    assert statuses
    assert metrics.mean_loop_fps > 0.0
    assert artifacts.physics_plot.exists()
    assert artifacts.demo_video.exists()
