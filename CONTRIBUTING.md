# Contributing Guide

Thanks for contributing to `dorne_interceptor`.

## Local Setup
1. Create and activate a virtual environment.
2. Install dependencies.
3. Run lint, manifest checks, and tests before opening a PR.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest ruff
ruff check src scripts tests --select E9,F63,F7,F82
python scripts/validate_repo_manifests.py
pytest -q tests/test_smoke.py tests/test_airsim_connection.py tests/test_constraints.py
```

## Branch and PR Rules
1. Create feature branches from `main`.
2. Keep PRs focused and small.
3. Add or update tests when behavior changes.
4. Update manifests when dataset samples or curated results change.

## Data and Output Policy
- Do not commit full `data/` or `outputs/` trees.
- Commit only curated publication artifacts under:
  - `datasets/samples/`
  - `results/graphs/`
  - `results/metrics/`
  - `results/reports/`
  - `results/samples/`
- Regenerate manifests after changing curated artifacts:

```powershell
python scripts/generate_repo_manifests.py
python scripts/validate_repo_manifests.py
```

## Commit Style
Use concise, imperative commit messages.
Examples:
- `Fix EKF spoofing gate regression`
- `Add day9 benchmark report assets`
- `Update dataset sample manifest`
