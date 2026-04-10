from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def validate_manifest_file(path: Path, top_keys: list[str]) -> dict:
    assert_true(path.exists(), f'Missing manifest: {path}')
    payload = load_json(path)
    for key in top_keys:
        assert_true(key in payload, f'Manifest {path} missing key: {key}')
    return payload


def validate_summary_entry(entry: dict, require_exists: bool = True) -> None:
    for key in ('path', 'file_count', 'total_bytes', 'extensions'):
        assert_true(key in entry, f'Manifest entry missing key `{key}`: {entry}')
    rel = Path(entry['path'])
    abs_path = ROOT / rel
    if require_exists:
        assert_true(abs_path.exists(), f'Required path does not exist: {entry["path"]}')
        assert_true(entry['file_count'] > 0, f'Expected non-empty path: {entry["path"]}')


def main() -> None:
    datasets_manifest = validate_manifest_file(
        ROOT / 'datasets' / 'manifest.json',
        ['full_local_datasets', 'published_samples', 'notes'],
    )
    results_manifest = validate_manifest_file(
        ROOT / 'results' / 'manifest.json',
        ['published_results', 'local_outputs_snapshot', 'notes'],
    )

    for entry in datasets_manifest['published_samples']:
        validate_summary_entry(entry, require_exists=True)

    for entry in results_manifest['published_results']:
        validate_summary_entry(entry, require_exists=True)

    # Local-only paths may be absent in CI, but shape must be valid.
    for entry in datasets_manifest['full_local_datasets']:
        validate_summary_entry(entry, require_exists=False)

    validate_summary_entry(results_manifest['local_outputs_snapshot'], require_exists=False)

    print('Manifest validation passed.')


if __name__ == '__main__':
    main()
