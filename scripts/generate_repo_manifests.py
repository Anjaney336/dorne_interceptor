from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]


def summarize_dir(path: Path) -> dict:
    if not path.exists():
        return {
            'path': str(path.relative_to(ROOT)).replace('\\', '/'),
            'exists': False,
            'file_count': 0,
            'total_bytes': 0,
            'extensions': {},
        }
    files = [p for p in path.rglob('*') if p.is_file()]
    exts = Counter((p.suffix.lower() or '<none>') for p in files)
    return {
        'path': str(path.relative_to(ROOT)).replace('\\', '/'),
        'exists': True,
        'file_count': len(files),
        'total_bytes': sum(p.stat().st_size for p in files),
        'extensions': dict(sorted(exts.items())),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def main() -> None:
    datasets_manifest = {
        'full_local_datasets': [
            summarize_dir(ROOT / 'data' / 'combined_target_yolo'),
            summarize_dir(ROOT / 'data' / 'visdrone_yolo'),
        ],
        'published_samples': [
            summarize_dir(ROOT / 'datasets' / 'samples' / 'combined_target_yolo'),
            summarize_dir(ROOT / 'datasets' / 'samples' / 'visdrone_yolo'),
        ],
        'notes': [
            'Full datasets are kept in local data/ and ignored in Git to keep repository push-safe.',
            'Published samples mirror structure and can be used for quick sanity checks.',
        ],
    }

    results_manifest = {
        'published_results': [
            summarize_dir(ROOT / 'results' / 'graphs'),
            summarize_dir(ROOT / 'results' / 'metrics'),
            summarize_dir(ROOT / 'results' / 'reports'),
            summarize_dir(ROOT / 'results' / 'samples'),
        ],
        'local_outputs_snapshot': summarize_dir(ROOT / 'outputs'),
        'notes': [
            'Full raw outputs remain in local outputs/ and are not tracked due size.',
            'Representative graphs, metrics and report artifacts are versioned under results/.',
        ],
    }

    write_json(ROOT / 'datasets' / 'manifest.json', datasets_manifest)
    write_json(ROOT / 'results' / 'manifest.json', results_manifest)


if __name__ == '__main__':
    main()
