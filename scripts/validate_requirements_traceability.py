from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = ROOT / 'assurance' / 'requirements_traceability.yaml'


def _expand_globs(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        out.extend(ROOT.glob(pattern))
    return sorted(set(out))


def main() -> None:
    payload = yaml.safe_load(TRACE_PATH.read_text(encoding='utf-8'))
    requirements = payload.get('requirements', [])

    seen_ids: set[str] = set()
    failures: list[str] = []

    for req in requirements:
        req_id = req.get('id', '').strip()
        if not req_id:
            failures.append('requirement with missing id')
            continue
        if req_id in seen_ids:
            failures.append(f'duplicate requirement id: {req_id}')
            continue
        seen_ids.add(req_id)

        modules = _expand_globs(req.get('module_globs', []))
        tests = _expand_globs(req.get('test_globs', []))

        if not modules:
            failures.append(f'{req_id} has no matched modules')
        if not tests:
            failures.append(f'{req_id} has no matched tests')

        print(f"{req_id}: modules={len(modules)} tests={len(tests)}")

    if failures:
        print('\nTraceability gate FAILED:')
        for item in failures:
            print(f'- {item}')
        raise SystemExit(1)

    print('\nTraceability gate PASSED.')


if __name__ == '__main__':
    main()
