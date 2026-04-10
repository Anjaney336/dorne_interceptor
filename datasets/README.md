# Datasets

This repository publishes representative dataset samples and a machine-readable inventory.

## Included In Git
- `datasets/samples/combined_target_yolo/`:
  - subset of train/val images and matching YOLO labels
  - `dataset.yaml` snapshot
- `datasets/samples/visdrone_yolo/`:
  - subset of train/val images and matching YOLO labels
  - `dataset.yaml` snapshot
- `datasets/manifest.json`:
  - full local dataset counts/sizes
  - published sample counts/sizes

## Full Dataset Location (Local Only)
- `data/combined_target_yolo/`
- `data/visdrone_yolo/`

These remain local and are intentionally not committed due repository size constraints.

## Refresh Inventory
```powershell
python scripts/generate_repo_manifests.py
```
