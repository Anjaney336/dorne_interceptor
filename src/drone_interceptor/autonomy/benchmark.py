from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drone_interceptor.autonomy.system import AutonomousInterceptorSystem, SystemRunResult


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    seed: int
    steps_executed: int
    intercepted: bool
    mean_loop_fps: float
    final_distance_m: float
    total_cost: float


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    runs: int
    base_seed: int
    success_rate: float
    mean_steps_executed: float
    mean_loop_fps: float
    mean_final_distance_m: float
    mean_total_cost: float
    best_run_seed: int
    worst_run_seed: int
    report_path: Path
    run_results: list[BenchmarkRunResult]


def run_autonomy_benchmark(
    config: dict[str, Any],
    runs: int = 5,
    output_path: str | Path | None = None,
) -> BenchmarkSummary:
    if runs < 1:
        raise ValueError("Benchmark runs must be at least 1.")

    base_seed = int(config.get("system", {}).get("random_seed", 7))
    report_path = _resolve_report_path(config=config, output_path=output_path)
    run_results: list[BenchmarkRunResult] = []

    for offset in range(runs):
        seed = base_seed + offset
        config_variant = copy.deepcopy(config)
        config_variant.setdefault("system", {})["random_seed"] = seed
        config_variant.setdefault("visualization", {})["save_outputs"] = False

        result = AutonomousInterceptorSystem(config_variant).run()
        run_results.append(_to_benchmark_run_result(seed=seed, result=result))

    mean_steps_executed = _mean(result.steps_executed for result in run_results)
    mean_loop_fps = _mean(result.mean_loop_fps for result in run_results)
    mean_final_distance_m = _mean(result.final_distance_m for result in run_results)
    mean_total_cost = _mean(result.total_cost for result in run_results)
    success_rate = _mean(1.0 if result.intercepted else 0.0 for result in run_results)
    best_run = min(run_results, key=lambda result: (result.final_distance_m, result.total_cost))
    worst_run = max(run_results, key=lambda result: (result.final_distance_m, result.total_cost))

    summary = BenchmarkSummary(
        runs=runs,
        base_seed=base_seed,
        success_rate=success_rate,
        mean_steps_executed=mean_steps_executed,
        mean_loop_fps=mean_loop_fps,
        mean_final_distance_m=mean_final_distance_m,
        mean_total_cost=mean_total_cost,
        best_run_seed=best_run.seed,
        worst_run_seed=worst_run.seed,
        report_path=report_path,
        run_results=run_results,
    )
    _write_report(summary)
    return summary


def _resolve_report_path(config: dict[str, Any], output_path: str | Path | None) -> Path:
    if output_path is not None:
        report_path = Path(output_path)
    else:
        output_dir = Path(config.get("visualization", {}).get("output_dir", "outputs"))
        report_path = output_dir / "autonomous_benchmark.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    return report_path


def _to_benchmark_run_result(seed: int, result: SystemRunResult) -> BenchmarkRunResult:
    return BenchmarkRunResult(
        seed=seed,
        steps_executed=result.steps_executed,
        intercepted=result.intercepted,
        mean_loop_fps=result.mean_loop_fps,
        final_distance_m=result.final_distance_m,
        total_cost=result.total_cost,
    )


def _write_report(summary: BenchmarkSummary) -> None:
    payload = {
        "runs": summary.runs,
        "base_seed": summary.base_seed,
        "success_rate": summary.success_rate,
        "mean_steps_executed": summary.mean_steps_executed,
        "mean_loop_fps": summary.mean_loop_fps,
        "mean_final_distance_m": summary.mean_final_distance_m,
        "mean_total_cost": summary.mean_total_cost,
        "best_run_seed": summary.best_run_seed,
        "worst_run_seed": summary.worst_run_seed,
        "run_results": [
            {
                "seed": result.seed,
                "steps_executed": result.steps_executed,
                "intercepted": result.intercepted,
                "mean_loop_fps": result.mean_loop_fps,
                "final_distance_m": result.final_distance_m,
                "total_cost": result.total_cost,
            }
            for result in summary.run_results
        ],
    }
    summary.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _mean(values: Any) -> float:
    sequence = list(values)
    if not sequence:
        return 0.0
    return float(sum(sequence) / len(sequence))


__all__ = ["BenchmarkRunResult", "BenchmarkSummary", "run_autonomy_benchmark"]
