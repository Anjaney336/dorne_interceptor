from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml


if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[3]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


VISDRONE_CLASS_NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
    "others",
)

DEFAULT_DATASET_ROOT = Path("data/visdrone_yolo")
DEFAULT_MODEL_NAME = "yolov10s.pt"
DEFAULT_RUN_NAME = "visdrone_yolov10s_brain"
DEFAULT_RAW_TRAIN_ROOT = Path("data/visdrone/VisDrone2019-DET-train")
DEFAULT_RAW_VAL_ROOT = Path("data/visdrone/VisDrone2019-DET-val")


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path, project_root: str | Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()

    root = Path(project_root) if project_root is not None else resolve_project_root()
    return (root / candidate).resolve()


def validate_dataset_root(dataset_root: str | Path) -> Path:
    root = resolve_project_path(dataset_root)
    required_directories = (
        root / "images" / "train",
        root / "images" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
    )

    missing = [directory for directory in required_directories if not directory.exists()]
    if missing:
        missing_text = ", ".join(str(directory) for directory in missing)
        raise FileNotFoundError(
            "YOLO dataset layout is incomplete. Expected images/labels train/val directories under "
            f"{root}. Missing: {missing_text}"
        )

    return root


def ensure_dataset_root(dataset_root: str | Path = DEFAULT_DATASET_ROOT) -> Path:
    try:
        return validate_dataset_root(dataset_root)
    except FileNotFoundError:
        root = resolve_project_path(dataset_root)
        _auto_convert_visdrone_dataset(output_root=root)
        return validate_dataset_root(root)


def create_dataset_yaml(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    output_path: str | Path | None = None,
) -> Path:
    root = ensure_dataset_root(dataset_root)
    yaml_path = resolve_project_path(output_path or (root / "dataset.yaml"))
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_config: dict[str, Any] = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "names": list(VISDRONE_CLASS_NAMES),
    }

    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_config, handle, sort_keys=False)

    return yaml_path


def _auto_convert_visdrone_dataset(output_root: Path) -> None:
    from drone_interceptor.datasets.visdrone import convert_visdrone_dataset, resolve_dataset_root

    project_root = resolve_project_root()
    train_root = resolve_dataset_root(DEFAULT_RAW_TRAIN_ROOT, project_root=project_root)
    val_root = resolve_dataset_root(DEFAULT_RAW_VAL_ROOT, project_root=project_root)
    convert_visdrone_dataset(train_root=train_root, val_root=val_root, output_root=output_root)


def _import_yolo() -> type:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is not installed in the active environment. "
            "Install requirements.txt before running training."
        ) from exc

    return YOLO


def _resolve_training_artifact(run_dir: Path) -> Path:
    candidates = (
        run_dir / "weights" / "best.pt",
        run_dir / "weights" / "last.pt",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No trained checkpoint found in {run_dir}")


def train_model(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    model_name: str = DEFAULT_MODEL_NAME,
    imgsz: int = 640,
    epochs: int = 10,
    batch: int = -1,
    fraction: float = 1.0,
    device: str | None = None,
    run_name: str = DEFAULT_RUN_NAME,
) -> Path:
    dataset_yaml = create_dataset_yaml(dataset_root=dataset_root)
    project_root = resolve_project_root()
    models_root = resolve_project_path("models", project_root=project_root)
    models_root.mkdir(parents=True, exist_ok=True)

    YOLO = _import_yolo()
    model = YOLO(model_name)

    train_kwargs: dict[str, Any] = {
        "data": str(dataset_yaml),
        "imgsz": imgsz,
        "epochs": epochs,
        "batch": batch,
        "fraction": fraction,
        "project": str(models_root),
        "name": run_name,
        "exist_ok": True,
    }
    if device:
        train_kwargs["device"] = device

    model.train(**train_kwargs)

    run_dir = models_root / run_name
    trained_checkpoint = _resolve_training_artifact(run_dir)
    exported_checkpoint = models_root / f"{run_name}.pt"
    shutil.copy2(trained_checkpoint, exported_checkpoint)
    return exported_checkpoint


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an Ultralytics YOLOv10 model on VisDrone YOLO data.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the converted YOLO dataset root.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Ultralytics model checkpoint to initialize from.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size. Use -1 for Ultralytics auto batch sizing.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Optional training-set fraction for smoke tests. Keep 1.0 for full training.",
    )
    parser.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cpu or 0.")
    parser.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help="Training run name under models/.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    dataset_yaml = create_dataset_yaml(dataset_root=args.dataset_root)
    checkpoint = train_model(
        dataset_root=args.dataset_root,
        model_name=args.model,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        fraction=args.fraction,
        device=args.device,
        run_name=args.run_name,
    )
    print(f"dataset_yaml={dataset_yaml}")
    print(f"checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
