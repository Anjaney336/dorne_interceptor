from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.validation.day5 import run_day5_execution


def test_day5_execution_runs_and_writes_outputs() -> None:
    summaries, metrics, artifacts = run_day5_execution(
        project_root=ROOT,
        random_seed=31,
        max_steps_override=20,
        use_airsim=False,
    )

    assert len(summaries) == 4
    assert metrics.fps > 0.0
    assert artifacts.trajectory_plot.exists()
    assert artifacts.distance_plot.exists()
    assert artifacts.demo_video.exists()
    assert artifacts.final_log.exists()
