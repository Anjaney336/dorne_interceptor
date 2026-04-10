from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'assurance' / 'performance_budgets.yaml'


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y'}


def _to_float(value: str) -> float:
    return float(str(value).strip())


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open('r', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _evaluate_rule(rows: list[dict[str, str]], column: str, rule: str, threshold: float | None) -> tuple[bool, str]:
    if not rows:
        return False, 'no rows'

    if rule == 'all_true':
        values = [_to_bool(r[column]) for r in rows]
        return all(values), f'all_true={all(values)}'

    if rule == 'ratio_true':
        values = [_to_bool(r[column]) for r in rows]
        ratio = sum(values) / len(values)
        ok = ratio >= float(threshold)
        return ok, f'ratio_true={ratio:.3f} threshold={float(threshold):.3f}'

    numeric = [_to_float(r[column]) for r in rows]

    if rule == 'max':
        observed = max(numeric)
        ok = observed <= float(threshold)
        return ok, f'max={observed:.6f} threshold={float(threshold):.6f}'

    if rule == 'min':
        observed = min(numeric)
        ok = observed >= float(threshold)
        return ok, f'min={observed:.6f} threshold={float(threshold):.6f}'

    if rule == 'mean_max':
        observed = mean(numeric)
        ok = observed <= float(threshold)
        return ok, f'mean={observed:.6f} threshold={float(threshold):.6f}'

    raise ValueError(f'Unsupported rule: {rule}')


def main() -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
    budgets = payload.get('budgets', {})
    failures: list[str] = []

    for budget_name, spec in budgets.items():
        csv_path = ROOT / spec['csv_path']
        if not csv_path.exists():
            failures.append(f'[{budget_name}] missing csv: {csv_path}')
            continue
        rows = _read_rows(csv_path)

        for check in spec.get('checks', []):
            col = check['column']
            rule = check['rule']
            threshold = check.get('threshold')

            if rows and col not in rows[0]:
                failures.append(f'[{budget_name}] missing column `{col}` in {csv_path}')
                continue

            ok, details = _evaluate_rule(rows, col, rule, threshold)
            print(f'[{budget_name}] {col} {rule}: {details}')
            if not ok:
                failures.append(f'[{budget_name}] {col} {rule} failed ({details})')

    if failures:
        print('\nPerformance budget gate FAILED:')
        for item in failures:
            print(f'- {item}')
        raise SystemExit(1)

    print('\nPerformance budget gate PASSED.')


if __name__ == '__main__':
    main()
