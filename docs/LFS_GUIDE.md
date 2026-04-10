# Optional Git LFS Guide

Use Git LFS if you later decide to version large binaries (weights, raw videos, full datasets).

## Install
- Windows: `git lfs install`
- Verify: `git lfs version`

## Suggested Patterns
```powershell
git lfs track "*.pt"
git lfs track "data/**"
git lfs track "outputs/**/*.mp4"
git lfs track "outputs/**/*.avi"
```

## Commit Tracking Rules
```powershell
git add .gitattributes
git commit -m "Configure git-lfs tracking rules"
```

## Migrate Existing History (only if required)
```powershell
git lfs migrate import --include="*.pt"
```

Run migration only on branches you are ready to rewrite.
