from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.platform import run_platform_demo


def test_platform_demo_writes_outputs() -> None:
    records, metrics, artifacts = run_platform_demo(
        project_root=ROOT,
        random_seed=71,
        max_steps_override=40,
        use_airsim=False,
    )

    assert len(records) == 5
    assert metrics.mean_fps > 0.0
    assert artifacts.final_demo_video.exists()
    assert artifacts.final_3d_plot.exists()
    assert artifacts.scenario_results_csv.exists()
    assert artifacts.dashboard_preview_html.exists()
