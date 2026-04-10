from __future__ import annotations

import argparse
from pathlib import Path

from drone_interceptor.autonomy import AutonomousInterceptorSystem, run_autonomy_benchmark
from drone_interceptor.config import load_config
from drone_interceptor.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the drone interception pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=1,
        help="Run multiple seeded autonomy rollouts and write an aggregate JSON report.",
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=None,
        help="Optional output path for the benchmark JSON report.",
    )
    return parser.parse_args()


def run_pipeline(
    config_path: str | Path,
    benchmark_runs: int = 1,
    benchmark_output: str | Path | None = None,
):
    config = load_config(config_path)
    setup_logging(config.get("system", {}).get("log_level", "INFO"))
    if benchmark_runs > 1:
        return run_autonomy_benchmark(config=config, runs=benchmark_runs, output_path=benchmark_output)
    system = AutonomousInterceptorSystem(config)
    return system.run()


def main() -> None:
    args = parse_args()
    run_pipeline(
        config_path=args.config,
        benchmark_runs=args.benchmark_runs,
        benchmark_output=args.benchmark_output,
    )


if __name__ == "__main__":
    main()
