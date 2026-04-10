from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]

if __package__ in (None, ""):
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from drone_interceptor.config import load_config
from drone_interceptor.datasets.visdrone import (
    build_drone_dataset_plan,
    load_yolo_dataset_class_names,
    load_yolo_labels,
    summarize_yolo_dataset,
)
from drone_interceptor.perception.detector import DetectionBox, TargetDetector, benchmark_detection_sets


@dataclass(frozen=True)
class DetectorBenchmarkArtifacts:
    summary_json: Path


def run_detector_benchmark(
    project_root: str | Path,
    dataset_root: str | Path | None = None,
    limit: int = 12,
) -> DetectorBenchmarkArtifacts:
    root = Path(project_root).resolve()
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(root / "configs" / "default.yaml")
    detector = TargetDetector(config)
    resolved_dataset_root = Path(dataset_root) if dataset_root is not None else (root / "data" / "visdrone_yolo")
    dataset_summary = summarize_yolo_dataset(resolved_dataset_root)
    dataset_class_names = load_yolo_dataset_class_names(resolved_dataset_root)
    mission_target_classes = [item.category for item in build_drone_dataset_plan() if item.required]

    images_dir = resolved_dataset_root / "images" / "val"
    labels_dir = resolved_dataset_root / "labels" / "val"
    image_paths = sorted([path for path in images_dir.glob("*") if path.is_file()])[:limit]
    ground_truth_sets: list[list[DetectionBox]] = []
    prediction_sets: list[list[DetectionBox]] = []
    inference_times_s: list[float] = []
    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        with Image.open(image_path) as image:
            image_rgb = np.asarray(image.convert("RGB"))
            image_width, image_height = image.size
        start = time.perf_counter()
        try:
            detection = detector.detect({"image": image_rgb})
            targets = detection.metadata.get("targets", [])
        except RuntimeError:
            detection = None
            targets = []
        inference_times_s.append(max(time.perf_counter() - start, 1e-6))
        predictions = [
            DetectionBox(
                x1=float(target["bbox_xyxy"][0]),
                y1=float(target["bbox_xyxy"][1]),
                x2=float(target["bbox_xyxy"][2]),
                y2=float(target["bbox_xyxy"][3]),
                confidence=float(target.get("confidence", 0.0 if detection is None else detection.confidence)),
                class_id=int(target.get("class_id", 0)),
            )
            for target in targets
            if "bbox_xyxy" in target
        ]
        ground_truth = [
            DetectionBox(
                x1=float(box.left),
                y1=float(box.top),
                x2=float(box.right),
                y2=float(box.bottom),
                confidence=1.0,
                class_id=int(class_id),
            )
            for class_id, box in load_yolo_labels(label_path, image_width=image_width, image_height=image_height)
        ]
        ground_truth_sets.append(ground_truth)
        prediction_sets.append(predictions)

    benchmark = benchmark_detection_sets(
        ground_truth_sets=ground_truth_sets,
        prediction_sets=prediction_sets,
        inference_times_s=inference_times_s,
        device_label=f"{platform.system()}::{platform.processor() or 'unknown_cpu'}",
    )
    training_validation_benchmark = _load_training_validation_benchmark(detector._resolve_model_path(config.get("perception", {}).get("model_path")))
    domain_overlap = sorted({name for name in dataset_class_names if name.lower() in _canonical_domain_names(mission_target_classes)})
    domain_aligned = bool(domain_overlap)
    if not domain_aligned:
        effective_benchmark = {
            "sample_count": int(len(ground_truth_sets)),
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
            "mean_iou": None,
            "fps": float(benchmark.fps),
            "device_label": str(benchmark.device_label),
        }
        effective_source = "blocked_by_domain_mismatch"
    else:
        effective_benchmark = training_validation_benchmark or benchmark.__dict__
        effective_source = "training_validation_artifact" if training_validation_benchmark is not None else "host_inference"
    payload = {
        "dataset_summary": dataset_summary,
        "domain_analysis": {
            "dataset_classes": dataset_class_names,
            "mission_target_requirements": mission_target_classes,
            "domain_aligned": domain_aligned,
            "class_overlap": domain_overlap,
            "note": (
                "The available YOLO dataset is not a drone-specific benchmark set."
                if not domain_aligned
                else "The dataset has at least partial overlap with the mission target classes."
            ),
        },
        "host_benchmark": benchmark.__dict__,
        "training_validation_benchmark": training_validation_benchmark,
        "effective_benchmark": effective_benchmark,
        "effective_benchmark_source": effective_source,
        "note": "Host FPS was measured on the current machine. Edge-device evidence still requires running this benchmark on the deployment hardware.",
    }
    summary_json = outputs_dir / "detector_benchmark.json"
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return DetectorBenchmarkArtifacts(summary_json=summary_json)


def _load_training_validation_benchmark(model_path: str | Path) -> dict[str, float | str] | None:
    path = Path(model_path) if not isinstance(model_path, Path) else model_path
    candidates = [
        path.parent / "results.csv",
        path.parent.parent / "results.csv",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        with candidate.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows:
            continue
        last = rows[-1]
        return {
            "sample_count": int(float(last.get("epoch", 0.0))),
            "precision": float(last.get("metrics/precision(B)", 0.0)),
            "recall": float(last.get("metrics/recall(B)", 0.0)),
            "map50": float(last.get("metrics/mAP50(B)", 0.0)),
            "map50_95": float(last.get("metrics/mAP50-95(B)", 0.0)),
            "mean_iou": 0.0,
            "fps": 0.0,
            "device_label": "training_validation_artifact",
        }
    return None


def _canonical_domain_names(names: list[str]) -> set[str]:
    canonical: set[str] = set()
    for name in names:
        lowered = str(name).lower()
        canonical.add(lowered)
        canonical.update(lowered.replace("_", " ").split())
    canonical.update({"drone", "uav", "quadcopter", "hexacopter", "fixed wing"})
    return canonical


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run detector benchmark on the validation split.")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args(argv)
    artifacts = run_detector_benchmark(
        project_root=args.project_root,
        dataset_root=args.dataset_root,
        limit=args.limit,
    )
    print(f"detector_benchmark={artifacts.summary_json}")
    return 0


__all__ = ["DetectorBenchmarkArtifacts", "run_detector_benchmark"]


if __name__ == "__main__":
    raise SystemExit(main())
