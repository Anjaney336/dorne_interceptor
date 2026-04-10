from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script_name: str) -> None:
    script = ROOT / 'scripts' / script_name
    print(f'Running {script_name}...')
    subprocess.run([sys.executable, str(script)], check=True)


def main() -> None:
    run('validate_repo_manifests.py')
    run('validate_requirements_traceability.py')
    run('check_performance_budgets.py')
    print('All assurance gates PASSED.')


if __name__ == '__main__':
    main()
