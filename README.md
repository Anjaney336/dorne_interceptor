# AI Drone Interceptor

[![CI](https://github.com/Anjaney336/dorne_interceptor/actions/workflows/ci.yml/badge.svg)](https://github.com/Anjaney336/dorne_interceptor/actions/workflows/ci.yml)
[![Release Artifacts](https://github.com/Anjaney336/dorne_interceptor/actions/workflows/release-artifacts.yml/badge.svg)](https://github.com/Anjaney336/dorne_interceptor/actions/workflows/release-artifacts.yml)

Research-grade autonomous interception stack with simulation, anti-spoofing navigation, AirSim mission replay, dashboard telemetry, and validation tooling.

## Modules
- `src/drone_interceptor/perception`: YOLO-based detection + inference utilities
- `src/drone_interceptor/tracking`: target tracking and association logic
- `src/drone_interceptor/navigation`: EKF, drift models, anti-spoofing logic
- `src/drone_interceptor/control`: guidance and controller constraints
- `src/drone_interceptor/simulation`: AirSim adapters, scenarios, telemetry API
- `src/drone_interceptor/backend`: mission/spoof services and run registry
- `src/drone_interceptor/validation`: day-wise validation and benchmark runners

## Repository Data Policy
- Full local datasets are stored under `data/` (ignored in Git due size).
- Full local runtime artifacts are stored under `outputs/` (ignored in Git due size).
- Publishable, versioned assets are stored in:
  - `datasets/samples/`: representative dataset slices used by this project
  - `datasets/manifest.json`: full-vs-published dataset inventory
  - `results/graphs/`, `results/metrics/`, `results/reports/`: curated evidence artifacts
  - `results/manifest.json`: output coverage and inventory

## Quick Start
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/run_pipeline.py --config configs/default.yaml
```

## Validation Commands
```powershell
python scripts/run_day4_validation.py --project-root .
python scripts/run_day9_execution.py --project-root .
python scripts/run_day10_execution.py --project-root .
python scripts/run_combined_target_validation.py
```

## CI/CD and Release
- Continuous integration: `.github/workflows/ci.yml`
  - Lint (runtime-critical checks)
  - Manifest validation
  - Requirements traceability validation
  - Performance budget gate validation
  - Pytest smoke suite
- Artifact release pipeline: `.github/workflows/release-artifacts.yml`
  - Generates zipped/tarred dataset/result bundles
  - Publishes release assets on `v*` tags or manual dispatch
  - Optional Git LFS status attachment

## Professional Artifacts Included
- Dataset samples with labels for `combined_target_yolo` and `visdrone_yolo`
- Sovereign report figures and PDF dossier
- Key mission metrics CSVs and benchmark outputs
- Graph snapshots from day-wise execution and final demos

## Rebuild Manifests
```powershell
python scripts/generate_repo_manifests.py
python scripts/validate_repo_manifests.py
```

## Contributing and Planning
- Contributor guide: `CONTRIBUTING.md`
- Issue templates: `.github/ISSUE_TEMPLATE/`
- PR template: `.github/pull_request_template.md`
- Project board setup: `docs/GITHUB_PROJECT_SETUP.md`
- Advanced assurance architecture: `docs/ADVANCED_ASSURANCE_ARCHITECTURE.md`
- Safety package: `assurance/safety/`
