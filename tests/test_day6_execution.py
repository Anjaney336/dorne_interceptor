from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.validation.day6 import run_day6_execution


def test_day6_execution_runs_and_writes_outputs() -> None:
    metrics, artifacts = run_day6_execution(
        project_root=ROOT,
        random_seed=51,
        max_steps_override=40,
        use_airsim=False,
    )

    assert metrics.mean_loop_fps > 0.0
    assert metrics.edge_mode_fps > 0.0
    assert artifacts.trajectory_plot.exists()
    assert artifacts.demo_video.exists()
    assert artifacts.log_file.exists()
