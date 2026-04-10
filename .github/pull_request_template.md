## Summary
- What changed and why?

## Validation
- [ ] `ruff check src scripts tests --select E9,F63,F7,F82`
- [ ] `python scripts/validate_repo_manifests.py`
- [ ] `pytest -q tests/test_smoke.py tests/test_airsim_connection.py tests/test_constraints.py`

## Data/Results Impact
- [ ] No curated datasets/results changed
- [ ] Curated artifacts changed and manifests were regenerated

## Checklist
- [ ] Linked issue (if applicable)
- [ ] Tests updated (if behavior changed)
- [ ] Docs updated (if needed)
