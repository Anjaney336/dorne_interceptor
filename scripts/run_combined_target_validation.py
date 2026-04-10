from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _import_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is not installed in the active environment. Install requirements before running validation."
        ) from exc
    return YOLO


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def _normalize_yolo_lines(path: Path, force_class_zero: bool = True) -> list[str]:
    lines_out: list[str] = []
    raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue
        if w <= 0.0 or h <= 0.0:
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            continue
        if force_class_zero:
            cls = 0
        lines_out.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines_out


def build_combined_dataset(
    *,
    project_root: Path,
    archive_dataset_dir: Path,
    visdrone_dataset_dir: Path,
    output_dataset_dir: Path,
) -> dict[str, int]:
    for rel in ("images/train", "images/val", "labels/train", "labels/val"):
        (output_dataset_dir / rel).mkdir(parents=True, exist_ok=True)

    counts = {
        "vis_train": 0,
        "vis_val": 0,
        "arc_train": 0,
        "arc_val": 0,
    }

    for split in ("train", "val"):
        src_img_dir = visdrone_dataset_dir / "images" / split
        src_lbl_dir = visdrone_dataset_dir / "labels" / split
        dst_img_dir = output_dataset_dir / "images" / split
        dst_lbl_dir = output_dataset_dir / "labels" / split
        for image_path in sorted(src_img_dir.glob("*")):
            if not image_path.is_file():
                continue
            label_path = src_lbl_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            labels = _normalize_yolo_lines(label_path, force_class_zero=True)
            if not labels:
                continue
            out_stem = f"vis_{image_path.stem}"
            _safe_link_or_copy(image_path, dst_img_dir / f"{out_stem}{image_path.suffix.lower()}")
            (dst_lbl_dir / f"{out_stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
            counts[f"vis_{split}"] += 1

    for image_path in sorted(archive_dataset_dir.glob("*.jpg")):
        label_path = archive_dataset_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        labels = _normalize_yolo_lines(label_path, force_class_zero=True)
        if not labels:
            continue
        token = hashlib.sha1(image_path.stem.encode("utf-8")).hexdigest()
        split = "val" if int(token[:8], 16) % 10 == 0 else "train"
        dst_img_dir = output_dataset_dir / "images" / split
        dst_lbl_dir = output_dataset_dir / "labels" / split
        out_stem = f"arc_{image_path.stem}"
        _safe_link_or_copy(image_path, dst_img_dir / f"{out_stem}{image_path.suffix.lower()}")
        (dst_lbl_dir / f"{out_stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        counts[f"arc_{split}"] += 1

    dataset_yaml = output_dataset_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dataset_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  - drone",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    counts["train_images"] = len(list((output_dataset_dir / "images" / "train").glob("*")))
    counts["val_images"] = len(list((output_dataset_dir / "images" / "val").glob("*")))
    counts["train_labels"] = len(list((output_dataset_dir / "labels" / "train").glob("*.txt")))
    counts["val_labels"] = len(list((output_dataset_dir / "labels" / "val").glob("*.txt")))
    return counts


def _validate_model(weights: Path, dataset_yaml: Path) -> dict[str, float]:
    YOLO = _import_yolo()
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(dataset_yaml),
        split="val",
        imgsz=640,
        batch=8,
        device="cpu",
        workers=0,
        verbose=False,
        plots=False,
    )
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def _presence_hit_rate(weights: Path, image_paths: list[Path], conf: float = 0.25) -> float:
    YOLO = _import_yolo()
    model = YOLO(str(weights))
    hits = 0
    for image_path in image_paths:
        result = model(str(image_path), imgsz=640, conf=float(conf), verbose=False)[0]
        if result.boxes is not None and len(result.boxes) > 0:
            hits += 1
    return float(hits / max(len(image_paths), 1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge archive + current YOLO data and validate detector models.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--archive-dataset-dir",
        type=Path,
        default=Path("archive (1)/drone_dataset_yolo/dataset_txt"),
    )
    parser.add_argument("--visdrone-dataset-dir", type=Path, default=Path("data/visdrone_yolo"))
    parser.add_argument("--output-dataset-dir", type=Path, default=Path("data/combined_target_yolo"))
    parser.add_argument("--output-report", type=Path, default=Path("outputs/combined_dataset_validation_report.json"))
    parser.add_argument(
        "--weights",
        nargs="+",
        default=[
            "models/combined_target_yolov10s_smoke/weights/best.pt",
            "yolov10s.pt",
            "yolov10n.pt",
            "models/visdrone_yolov10n_smoke.pt",
        ],
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    archive_dataset_dir = (root / args.archive_dataset_dir).resolve()
    visdrone_dataset_dir = (root / args.visdrone_dataset_dir).resolve()
    output_dataset_dir = (root / args.output_dataset_dir).resolve()
    report_path = (root / args.output_report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    counts = build_combined_dataset(
        project_root=root,
        archive_dataset_dir=archive_dataset_dir,
        visdrone_dataset_dir=visdrone_dataset_dir,
        output_dataset_dir=output_dataset_dir,
    )
    dataset_yaml = output_dataset_dir / "dataset.yaml"

    validation_metrics: dict[str, dict[str, float]] = {}
    best_model = None
    best_map50 = -1.0
    for raw_weight in args.weights:
        weights_path = (root / raw_weight).resolve()
        if not weights_path.exists():
            continue
        metrics = _validate_model(weights_path, dataset_yaml)
        validation_metrics[weights_path.as_posix()] = metrics
        if metrics["map50"] > best_map50:
            best_map50 = metrics["map50"]
            best_model = weights_path

    archive_val_samples = sorted((output_dataset_dir / "images" / "val").glob("arc_*.jpg"))[:120]
    sample_presence: dict[str, float] = {}
    for weight_key in (best_model.as_posix(),) if best_model is not None else tuple():
        sample_presence[weight_key] = _presence_hit_rate(Path(weight_key), archive_val_samples, conf=0.25)

    report = {
        "dataset_root": output_dataset_dir.as_posix(),
        "dataset_split_counts": counts,
        "validation_metrics": validation_metrics,
        "selected_best_model_for_now": best_model.as_posix() if best_model is not None else None,
        "validation_gate": {
            "passed": bool(best_map50 >= 0.45),
            "reason": (
                "Best model reached required threshold."
                if best_map50 >= 0.45
                else f"Best map50={best_map50:.6f} is below production threshold 0.45."
            ),
        },
        "archive_sample_presence_test": {
            "sample_count": len(archive_val_samples),
            "confidence_threshold": 0.25,
            "hit_rate_by_model": sample_presence,
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report={report_path}")
    if best_model is not None:
        print(f"best_model={best_model}")
        print(f"best_map50={best_map50:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
