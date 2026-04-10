# Advanced Systems Assurance Architecture

This layer upgrades the project workflow toward high-end robotics program discipline without changing runtime map rendering or live analytics behavior.

## Assurance Overlay
- Safety case artifacts: `assurance/safety/`
- Requirements traceability source: `assurance/requirements_traceability.yaml`
- Performance budget policy: `assurance/performance_budgets.yaml`
- Gate scripts:
  - `scripts/validate_repo_manifests.py`
  - `scripts/validate_requirements_traceability.py`
  - `scripts/check_performance_budgets.py`
  - `scripts/run_assurance_gates.py`

## Execution Model
1. Developer/test workflow validates logic and artifacts locally.
2. CI executes lint, smoke tests, manifest checks, traceability checks, budget checks.
3. Release workflow bundles validated datasets/results into immutable release assets.
4. Optional Git LFS is available for future heavy-binary versioning.

## Determinism and Auditability
- Seeded replay/benchmark outputs are tracked in curated results artifacts.
- Manifest inventory captures local-vs-published data coverage.
- Requirements map explicitly binds requirement IDs to code and tests.
- Performance budgets enforce mission-level limits on selected KPIs.

## Safety and Mission Assurance
- Hazard log and FMEA provide failure-mode visibility.
- Fault tree and STPA-lite establish control/action-level safety constraints.
- Concise assurance case links claims to evidence artifacts.

## Operator-Facing Components
- Overview maps: unchanged.
- Live analytics dashboards: unchanged.
- Assurance layer is additive and non-intrusive.
