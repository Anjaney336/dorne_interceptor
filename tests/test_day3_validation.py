from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.config import load_config
from drone_interceptor.validation.day3 import run_day3_validation


def test_day3_validation_runs_and_writes_outputs(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "configs" / "default.yaml")
    config["mission"]["max_steps"] = 60
    config["prediction"]["horizon_steps"] = 8
    config["control"]["mpc_horizon_steps"] = 8
    config["control"]["mpc_num_trajectories"] = 12
    config["optimization"]["horizon_steps"] = 8
    config["optimization"]["num_trajectories"] = 12
    config["visualization"]["output_dir"] = str(project_root / "outputs")

    with (project_root / "configs" / "default.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    statuses, metrics, artifacts = run_day3_validation(
        project_root=project_root,
        monte_carlo_runs=1,
        noise_runs=1,
        control_runs=1,
        random_seed=5,
    )

    assert any(status.name == "Integration" for status in statuses)
    assert artifacts.trajectory_3d_plot.exists()
    assert artifacts.metrics_plot.exists()
    assert metrics.mean_loop_fps > 0.0
