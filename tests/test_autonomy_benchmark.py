from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.autonomy import run_autonomy_benchmark
from drone_interceptor.config import load_config


def test_autonomy_benchmark_writes_summary_json(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["mission"]["max_steps"] = 10
    config["visualization"]["output_dir"] = str(tmp_path / "outputs")

    report_path = tmp_path / "benchmark" / "summary.json"
    summary = run_autonomy_benchmark(config=config, runs=3, output_path=report_path)

    assert summary.runs == 3
    assert len(summary.run_results) == 3
    assert summary.report_path == report_path
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["runs"] == 3
    assert payload["base_seed"] == int(config["system"]["random_seed"])
    assert len(payload["run_results"]) == 3
    assert [run["seed"] for run in payload["run_results"]] == [
        int(config["system"]["random_seed"]) + offset for offset in range(3)
    ]
