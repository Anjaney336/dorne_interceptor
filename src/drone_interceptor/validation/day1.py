from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[3]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from drone_interceptor.datasets.visdrone import load_yolo_labels, visualize_yolo_labels
from drone_interceptor.perception.infer import _import_cv2, _import_yolo, annotate_frame, resolve_model_path
from drone_interceptor.tracking.tracker import DeepSortTracker, draw_tracked_objects


LOGGER = logging.getLogger("day1_validation")


@dataclass(frozen=True)
class ValidationStatus:
    name: str
    passed: bool
    details: str


@dataclass(frozen=True)
class Day1Artifacts:
    dataset_visualizations: list[Path]
    detection_visualization: Path
    tracking_visualization: Path
    trajectory_plot: Path
    log_file: Path


def run_day1_validation(
    project_root: str | Path,
    sample_count: int = 3,
    random_seed: int = 7,
) -> tuple[list[ValidationStatus], Day1Artifacts]:
    root = Path(project_root).resolve()
    outputs_dir = root / "outputs"
    logs_dir = root / "logs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = root / "data" / "visdrone_yolo"
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"

    dataset_status, sample_paths = validate_dataset(
        images_root=images_root,
        labels_root=labels_root,
        outputs_dir=outputs_dir,
        sample_count=sample_count,
        random_seed=random_seed,
    )

    detection_status, detection_image_path, detection_source_image = validate_detection(
        image_paths=_build_detection_candidates(sample_paths=sample_paths, dataset_root=dataset_root),
        outputs_dir=outputs_dir,
    )

    tracking_status, tracking_output_path = validate_tracking(
        image_path=detection_source_image,
        outputs_dir=outputs_dir,
    )

    simulation_status, trajectory_plot_path = validate_simulation(
        outputs_dir=outputs_dir,
        random_seed=random_seed,
    )

    integration_status = validate_integration(
        statuses=[dataset_status, detection_status, tracking_status, simulation_status],
        required_outputs=[detection_image_path, tracking_output_path, trajectory_plot_path],
    )

    statuses = [
        dataset_status,
        detection_status,
        tracking_status,
        simulation_status,
        integration_status,
    ]
    artifacts = Day1Artifacts(
        dataset_visualizations=[sample.output_path for sample in sample_paths],
        detection_visualization=detection_image_path,
        tracking_visualization=tracking_output_path,
        trajectory_plot=trajectory_plot_path,
        log_file=logs_dir / "day1_validation.log",
    )
    return statuses, artifacts


@dataclass(frozen=True)
class DatasetSample:
    image_path: Path
    label_path: Path
    output_path: Path


def validate_dataset(
    images_root: Path,
    labels_root: Path,
    outputs_dir: Path,
    sample_count: int,
    random_seed: int,
) -> tuple[ValidationStatus, list[DatasetSample]]:
    if not images_root.exists() or not labels_root.exists():
        return ValidationStatus("Dataset", False, "Missing data/visdrone_yolo/images or labels"), []

    image_paths = sorted(
        [
            path
            for path in images_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )
    if len(image_paths) < sample_count:
        return ValidationStatus("Dataset", False, "Not enough images for random sampling"), []

    rng = random.Random(random_seed)
    sampled_images = rng.sample(image_paths, sample_count)

    sample_artifacts: list[DatasetSample] = []
    for index, image_path in enumerate(sampled_images, start=1):
        split = image_path.parent.name
        label_path = labels_root / split / f"{image_path.stem}.txt"
        if not label_path.exists():
            return ValidationStatus("Dataset", False, f"Missing label for {image_path.name}"), []

        with Image.open(image_path) as image:
            image_width, image_height = image.size
            if image_width <= 0 or image_height <= 0:
                return ValidationStatus("Dataset", False, f"Failed to load image {image_path.name}"), []

        boxes = load_yolo_labels(label_path=label_path, image_width=image_width, image_height=image_height)
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            _, x_center, y_center, width, height = raw_line.split()
            normalized_values = [float(x_center), float(y_center), float(width), float(height)]
            if not all(0.0 < value < 1.0 for value in normalized_values):
                return ValidationStatus("Dataset", False, f"Invalid normalized bbox in {label_path.name}"), []

        output_path = outputs_dir / f"day1_dataset_sample_{index}.png"
        visualize_yolo_labels(image_path=image_path, label_path=label_path, output_path=output_path)
        LOGGER.info(
            "dataset_sample index=%s image=%s label=%s boxes=%s",
            index,
            image_path.name,
            label_path.name,
            len(boxes),
        )
        sample_artifacts.append(DatasetSample(image_path=image_path, label_path=label_path, output_path=output_path))

    return ValidationStatus("Dataset", True, f"Validated {sample_count} random samples"), sample_artifacts


def validate_detection(image_paths: list[Path], outputs_dir: Path) -> tuple[ValidationStatus, Path, Path]:
    YOLO = _import_yolo()
    cv2 = _import_cv2()
    output_path = outputs_dir / "day1_detection.png"

    model_candidates: list[str | Path] = [resolve_model_path(None), "yolov10s.pt"]
    seen_models: set[str] = set()
    best_attempt: tuple[int, np.ndarray, Any, Path, str | Path] | None = None

    for model_path in model_candidates:
        model_key = str(model_path)
        if model_key in seen_models:
            continue
        seen_models.add(model_key)
        model = YOLO(str(model_path))

        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue

            result = model(frame, imgsz=640, conf=0.25, verbose=False)[0]
            boxes = getattr(result, "boxes", None)
            detection_count = 0 if boxes is None else int(boxes.xyxy.cpu().numpy().shape[0])
            if best_attempt is None or detection_count > best_attempt[0]:
                best_attempt = (detection_count, frame, result, image_path, model_path)
            if detection_count > 0:
                annotated = annotate_frame(frame=frame, result=result, fps=0.0)
                cv2.imwrite(str(output_path), annotated)
                LOGGER.info(
                    "detection_validation image=%s size=%sx%s detections=%s success=%s model=%s",
                    image_path.name,
                    frame.shape[1],
                    frame.shape[0],
                    detection_count,
                    True,
                    model_path,
                )
                return (
                    ValidationStatus(
                        "Detection",
                        True,
                        f"detections={detection_count} image_size={frame.shape[1]}x{frame.shape[0]}",
                    ),
                    output_path,
                    image_path,
                )

    if best_attempt is None:
        return ValidationStatus("Detection", False, "No readable candidate images"), output_path, image_paths[0]

    detection_count, frame, result, image_path, model_path = best_attempt
    annotated = annotate_frame(frame=frame, result=result, fps=0.0)
    cv2.imwrite(str(output_path), annotated)
    LOGGER.info(
        "detection_validation image=%s size=%sx%s detections=%s success=%s model=%s",
        image_path.name,
        frame.shape[1],
        frame.shape[0],
        detection_count,
        detection_count > 0,
        model_path,
    )
    return (
        ValidationStatus("Detection", False, f"detections={detection_count} image_size={frame.shape[1]}x{frame.shape[0]}"),
        output_path,
        image_path,
    )


def validate_tracking(image_path: Path, outputs_dir: Path) -> tuple[ValidationStatus, Path]:
    YOLO = _import_yolo()
    cv2 = _import_cv2()

    model = YOLO("yolov10s.pt")
    base_frame = cv2.imread(str(image_path))
    if base_frame is None:
        return ValidationStatus("Tracking", False, f"Failed to read image {image_path.name}"), outputs_dir / "day1_tracking.png"

    initial_result = model(base_frame, imgsz=640, conf=0.25, verbose=False)[0]
    boxes = getattr(initial_result, "boxes", None)
    if boxes is None or boxes.xyxy.cpu().numpy().size == 0:
        return ValidationStatus("Tracking", False, "YOLO produced no detections for tracking validation"), outputs_dir / "day1_tracking.png"

    tracker = DeepSortTracker(history_length=12)
    tracked_sequences: list[list[Any]] = []
    frames: list[np.ndarray] = []
    for frame_index in range(6):
        dx = frame_index * 4
        dy = frame_index * 2
        shifted_frame = _shift_frame(base_frame, dx=dx, dy=dy)
        shifted_result = _shift_result(initial_result, dx=dx, dy=dy)
        tracked_objects = tracker.update_from_yolo(result=shifted_result, frame=shifted_frame)
        tracked_sequences.append(tracked_objects)
        frames.append(shifted_frame)

    first_non_empty = next((items for items in tracked_sequences if items), [])
    if not first_non_empty:
        return ValidationStatus("Tracking", False, "DeepSORT did not confirm any tracks"), outputs_dir / "day1_tracking.png"

    primary_id = first_non_empty[0].track_id
    continuity_points = []
    for tracked_objects in tracked_sequences:
        for obj in tracked_objects:
            if obj.track_id == primary_id:
                continuity_points.append(obj.centroid)
                break

    if len(continuity_points) < 2:
        return ValidationStatus("Tracking", False, "Track continuity was not maintained"), outputs_dir / "day1_tracking.png"

    final_frame = draw_tracked_objects(frames[-1], tracked_sequences[-1], fps=0.0)
    output_path = outputs_dir / "day1_tracking.png"
    cv2.imwrite(str(output_path), final_frame)
    LOGGER.info(
        "tracking_validation track_id=%s points=%s active_tracks=%s continuity=%s",
        primary_id,
        len(continuity_points),
        len(tracked_sequences[-1]),
        True,
    )
    return ValidationStatus("Tracking", True, f"primary_track={primary_id} points={len(continuity_points)}"), output_path


def validate_simulation(outputs_dir: Path, random_seed: int) -> tuple[ValidationStatus, Path]:
    rng = np.random.default_rng(random_seed)
    dt = 0.1
    steps = 120
    interceptor_max_accel = 4.0
    interceptor_max_speed = 35.0
    intercept_distance = 12.0

    target_positions = np.zeros((steps + 1, 2), dtype=float)
    target_velocities = np.zeros((steps + 1, 2), dtype=float)
    interceptor_positions = np.zeros((steps + 1, 2), dtype=float)
    interceptor_velocities = np.zeros((steps + 1, 2), dtype=float)

    target_positions[0] = np.array([320.0, 180.0], dtype=float)
    target_velocities[0] = np.array([-6.0, 3.0], dtype=float)
    interceptor_positions[0] = np.array([0.0, 0.0], dtype=float)
    interceptor_velocities[0] = np.array([0.0, 0.0], dtype=float)

    intercept_step: int | None = None
    last_step = 0
    for step in range(1, steps + 1):
        target_accel = rng.normal(0.0, 1.5, size=2)
        target_positions[step] = (
            target_positions[step - 1] + target_velocities[step - 1] * dt + 0.5 * target_accel * (dt**2)
        )
        target_velocities[step] = target_velocities[step - 1] + target_accel * dt

        direction = target_positions[step - 1] - interceptor_positions[step - 1]
        distance = np.linalg.norm(direction)
        desired_accel = np.zeros(2, dtype=float) if distance <= 1e-9 else (direction / distance) * interceptor_max_accel

        interceptor_positions[step] = (
            interceptor_positions[step - 1]
            + interceptor_velocities[step - 1] * dt
            + 0.5 * desired_accel * (dt**2)
        )
        interceptor_velocities[step] = interceptor_velocities[step - 1] + desired_accel * dt

        speed = np.linalg.norm(interceptor_velocities[step])
        if speed > interceptor_max_speed and speed > 0.0:
            interceptor_velocities[step] = (interceptor_velocities[step] / speed) * interceptor_max_speed

        last_step = step
        if np.linalg.norm(target_positions[step] - interceptor_positions[step]) <= intercept_distance:
            intercept_step = step
            break

    target_positions = target_positions[: last_step + 1]
    target_velocities = target_velocities[: last_step + 1]
    interceptor_positions = interceptor_positions[: last_step + 1]
    interceptor_velocities = interceptor_velocities[: last_step + 1]

    output_path = outputs_dir / "day1_trajectory.png"
    _plot_day1_trajectories(
        target_positions=target_positions,
        interceptor_positions=interceptor_positions,
        output_path=output_path,
    )
    LOGGER.info(
        "simulation_validation steps=%s intercepted=%s final_distance=%.3f",
        last_step,
        intercept_step is not None,
        float(np.linalg.norm(target_positions[-1] - interceptor_positions[-1])),
    )
    return ValidationStatus(
        "Simulation",
        True,
        (
            f"positions={len(target_positions)} "
            f"velocities={len(target_velocities)} "
            f"intercepted={intercept_step is not None}"
        ),
    ), output_path


def validate_integration(statuses: list[ValidationStatus], required_outputs: list[Path]) -> ValidationStatus:
    if not all(status.passed for status in statuses):
        return ValidationStatus("Visualization", False, "Upstream stage failed")
    missing = [path for path in required_outputs if not path.exists()]
    if missing:
        return ValidationStatus("Visualization", False, f"Missing outputs: {', '.join(path.name for path in missing)}")
    return ValidationStatus("Visualization", True, "Full pipeline executed without crashes")


def _build_detection_candidates(sample_paths: list[DatasetSample], dataset_root: Path) -> list[Path]:
    candidates = [sample.image_path for sample in sample_paths]
    known_candidate = dataset_root / "images" / "val" / "0000001_02999_d_0000005.jpg"
    if known_candidate.exists() and known_candidate not in candidates:
        candidates.append(known_candidate)
    return candidates


def setup_logging(log_path: Path) -> None:
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def print_report(statuses: list[ValidationStatus]) -> None:
    print("DAY 1 VALIDATION REPORT:")
    for status in statuses:
        outcome = "PASS" if status.passed else "FAIL"
        line = f"- {status.name}: {outcome}"
        print(line)
        LOGGER.info("%s %s %s", status.name, outcome, status.details)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Day 1 validation for the drone interceptor pipeline.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root directory.")
    parser.add_argument("--sample-count", type=int, default=3, help="Number of random dataset samples to validate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for repeatable validation.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    log_path = project_root / "logs" / "day1_validation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_path)

    statuses, artifacts = run_day1_validation(
        project_root=project_root,
        sample_count=args.sample_count,
        random_seed=args.seed,
    )
    LOGGER.info(
        "artifacts dataset_visualizations=%s detection=%s tracking=%s trajectory=%s",
        [str(path) for path in artifacts.dataset_visualizations],
        artifacts.detection_visualization,
        artifacts.tracking_visualization,
        artifacts.trajectory_plot,
    )
    print_report(statuses)


def _plot_day1_trajectories(
    target_positions: np.ndarray,
    interceptor_positions: np.ndarray,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(target_positions[:, 0], target_positions[:, 1], label="Target", color="tab:red", linewidth=2)
    axis.plot(
        interceptor_positions[:, 0],
        interceptor_positions[:, 1],
        label="Interceptor",
        color="tab:blue",
        linewidth=2,
    )
    axis.scatter(target_positions[0, 0], target_positions[0, 1], color="tab:red", marker="o", s=50)
    axis.scatter(interceptor_positions[0, 0], interceptor_positions[0, 1], color="tab:blue", marker="o", s=50)
    axis.scatter(target_positions[-1, 0], target_positions[-1, 1], color="tab:red", marker="x", s=70)
    axis.scatter(interceptor_positions[-1, 0], interceptor_positions[-1, 1], color="tab:blue", marker="x", s=70)
    axis.set_title("Day 1 Trajectory Validation")
    axis.set_xlabel("X Position (m)")
    axis.set_ylabel("Y Position (m)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    axis.axis("equal")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _shift_frame(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    cv2 = _import_cv2()
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))


class _TensorLike:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def cpu(self) -> "_TensorLike":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


def _shift_result(result: Any, dx: int, dy: int) -> Any:
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy().copy()
    xyxy[:, [0, 2]] += dx
    xyxy[:, [1, 3]] += dy
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=_TensorLike(xyxy),
            conf=_TensorLike(boxes.conf.cpu().numpy().copy()),
            cls=_TensorLike(boxes.cls.cpu().numpy().copy()),
        ),
        names=result.names,
    )


__all__ = ["run_day1_validation"]


if __name__ == "__main__":
    main()
