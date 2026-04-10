from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class BoundingBox:
    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    def clip(self, image_width: int, image_height: int) -> BoundingBox | None:
        left = min(max(self.left, 0.0), float(image_width))
        top = min(max(self.top, 0.0), float(image_height))
        right = min(max(self.right, 0.0), float(image_width))
        bottom = min(max(self.bottom, 0.0), float(image_height))

        width = right - left
        height = bottom - top
        if width <= 0.0 or height <= 0.0:
            return None

        return BoundingBox(left=left, top=top, width=width, height=height)


@dataclass(frozen=True)
class VisDroneObject:
    bbox: BoundingBox
    score: float
    class_id: int
    truncation: int
    occlusion: int


@dataclass(frozen=True)
class YoloObject:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_line(self) -> str:
        return (
            f"{self.class_id} "
            f"{self.x_center:.6f} "
            f"{self.y_center:.6f} "
            f"{self.width:.6f} "
            f"{self.height:.6f}"
        )


@dataclass(frozen=True)
class SplitConversionSummary:
    split: str
    images_processed: int
    labels_written: int
    objects_written: int
    ignored_objects: int
    clipped_objects: int


@dataclass(frozen=True)
class DatasetRequirement:
    category: str
    required: bool
    rationale: str


def parse_visdrone_annotation_line(line: str) -> VisDroneObject:
    values = [part.strip() for part in line.split(",")]
    while values and values[-1] == "":
        values.pop()
    if len(values) != 8:
        raise ValueError(
            "VisDrone annotations must contain 8 comma-separated values: "
            "bbox_left, bbox_top, width, height, score, class, truncation, occlusion."
        )

    bbox_left, bbox_top, width, height, score, class_id, truncation, occlusion = values
    return VisDroneObject(
        bbox=BoundingBox(
            left=float(bbox_left),
            top=float(bbox_top),
            width=float(width),
            height=float(height),
        ),
        score=float(score),
        class_id=int(class_id),
        truncation=int(truncation),
        occlusion=int(occlusion),
    )


def load_visdrone_annotations(annotation_path: str | Path) -> list[VisDroneObject]:
    path = Path(annotation_path)
    objects: list[VisDroneObject] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            objects.append(parse_visdrone_annotation_line(line))

    return objects


def convert_object_to_yolo(
    annotation: VisDroneObject,
    image_width: int,
    image_height: int,
) -> tuple[YoloObject | None, bool]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive for YOLO normalization.")

    if annotation.class_id == 0:
        return None, False

    clipped_bbox = annotation.bbox.clip(image_width=image_width, image_height=image_height)
    if clipped_bbox is None:
        return None, False

    was_clipped = clipped_bbox != annotation.bbox
    x_center = (clipped_bbox.left + (clipped_bbox.width / 2.0)) / float(image_width)
    y_center = (clipped_bbox.top + (clipped_bbox.height / 2.0)) / float(image_height)
    width = clipped_bbox.width / float(image_width)
    height = clipped_bbox.height / float(image_height)

    yolo_object = YoloObject(
        class_id=annotation.class_id - 1,
        x_center=min(max(x_center, 0.0), 1.0),
        y_center=min(max(y_center, 0.0), 1.0),
        width=min(max(width, 0.0), 1.0),
        height=min(max(height, 0.0), 1.0),
    )
    return yolo_object, was_clipped


def load_yolo_labels(
    label_path: str | Path,
    image_width: int,
    image_height: int,
) -> list[tuple[int, BoundingBox]]:
    boxes: list[tuple[int, BoundingBox]] = []
    path = Path(label_path)

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"YOLO label must have 5 space-separated values: {path}")

            class_id = int(parts[0])
            x_center = float(parts[1]) * image_width
            y_center = float(parts[2]) * image_height
            width = float(parts[3]) * image_width
            height = float(parts[4]) * image_height

            boxes.append(
                (
                    class_id,
                    BoundingBox(
                        left=x_center - (width / 2.0),
                        top=y_center - (height / 2.0),
                        width=width,
                        height=height,
                    ),
                )
            )

    return boxes


def _find_image_path(images_dir: Path, stem: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No image found for annotation stem '{stem}' in {images_dir}")


def _ensure_split_dirs(output_root: Path, split: str) -> tuple[Path, Path]:
    images_dir = output_root / "images" / split
    labels_dir = output_root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def convert_annotation_file(
    annotation_path: str | Path,
    image_path: str | Path,
    output_label_path: str | Path,
) -> tuple[int, int, int]:
    annotation_file = Path(annotation_path)
    image_file = Path(image_path)
    label_file = Path(output_label_path)

    with Image.open(image_file) as image:
        image_width, image_height = image.size

    yolo_objects: list[YoloObject] = []
    ignored_objects = 0
    clipped_objects = 0

    for annotation in load_visdrone_annotations(annotation_file):
        yolo_object, was_clipped = convert_object_to_yolo(
            annotation=annotation,
            image_width=image_width,
            image_height=image_height,
        )
        if yolo_object is None:
            ignored_objects += 1
            continue

        if was_clipped:
            clipped_objects += 1
        yolo_objects.append(yolo_object)

    label_file.parent.mkdir(parents=True, exist_ok=True)
    label_file.write_text(
        "\n".join(object_.to_line() for object_ in yolo_objects),
        encoding="utf-8",
    )

    return len(yolo_objects), ignored_objects, clipped_objects


def convert_visdrone_split(
    dataset_root: str | Path,
    output_root: str | Path,
    split: str,
) -> SplitConversionSummary:
    input_root = Path(dataset_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Split root not found: {input_root}")

    annotations_dir = input_root / "annotations"
    images_dir = input_root / "images"
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    output_images_dir, output_labels_dir = _ensure_split_dirs(Path(output_root), split)

    annotation_paths = sorted(annotations_dir.glob("*.txt"))
    if not annotation_paths:
        raise FileNotFoundError(f"No annotation files found in {annotations_dir}")

    images_processed = 0
    labels_written = 0
    objects_written = 0
    ignored_objects = 0
    clipped_objects = 0

    for annotation_path in annotation_paths:
        image_path = _find_image_path(images_dir=images_dir, stem=annotation_path.stem)
        output_image_path = output_images_dir / image_path.name
        output_label_path = output_labels_dir / f"{annotation_path.stem}.txt"

        shutil.copy2(image_path, output_image_path)
        kept_objects, skipped_objects, clipped = convert_annotation_file(
            annotation_path=annotation_path,
            image_path=image_path,
            output_label_path=output_label_path,
        )

        images_processed += 1
        labels_written += 1
        objects_written += kept_objects
        ignored_objects += skipped_objects
        clipped_objects += clipped

    return SplitConversionSummary(
        split=split,
        images_processed=images_processed,
        labels_written=labels_written,
        objects_written=objects_written,
        ignored_objects=ignored_objects,
        clipped_objects=clipped_objects,
    )


def convert_visdrone_dataset(
    train_root: str | Path,
    val_root: str | Path,
    output_root: str | Path,
) -> list[SplitConversionSummary]:
    summaries = [
        convert_visdrone_split(dataset_root=train_root, output_root=output_root, split="train"),
        convert_visdrone_split(dataset_root=val_root, output_root=output_root, split="val"),
    ]
    return summaries


def build_drone_dataset_plan() -> list[DatasetRequirement]:
    return [
        DatasetRequirement("quadcopters", True, "Primary competition target class."),
        DatasetRequirement("hexacopters", True, "Larger multirotor profile with different silhouette."),
        DatasetRequirement("fixed_wing", True, "Hard-negative separation from non-hovering aircraft-like targets."),
        DatasetRequirement("racing_drones", True, "Small, fast, low-visibility profiles."),
        DatasetRequirement("small_silhouettes", True, "Long-range detection robustness."),
        DatasetRequirement("partial_occlusion", True, "Urban and tree-line operational realism."),
        DatasetRequirement("night", True, "Low-light deployment robustness."),
        DatasetRequirement("backlight", True, "Sun-facing camera failure resistance."),
        DatasetRequirement("urban_clutter", True, "False-positive suppression in dense scenes."),
        DatasetRequirement("birds", True, "Hard negative set."),
        DatasetRequirement("poles", True, "Hard negative set."),
        DatasetRequirement("rooftops", True, "Hard negative set."),
        DatasetRequirement("kites", True, "Hard negative set."),
        DatasetRequirement("aircraft", True, "Hard negative set."),
    ]


def summarize_yolo_dataset(dataset_root: str | Path) -> dict[str, object]:
    root = Path(dataset_root)
    class_names = load_yolo_dataset_class_names(root)
    labels_root = root / "labels"
    images_root = root / "images"
    split_summaries: dict[str, dict[str, int]] = {}
    class_histogram: dict[int, int] = {}
    for split_dir in sorted(labels_root.glob("*")):
        if not split_dir.is_dir():
            continue
        image_count = 0
        label_count = 0
        object_count = 0
        for image_path in (images_root / split_dir.name).glob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_count += 1
        for label_path in split_dir.glob("*.txt"):
            label_count += 1
            with label_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id = int(parts[0])
                    class_histogram[class_id] = class_histogram.get(class_id, 0) + 1
                    object_count += 1
        split_summaries[split_dir.name] = {
            "images": image_count,
            "labels": label_count,
            "objects": object_count,
        }
    return {
        "dataset_root": str(root),
        "splits": split_summaries,
        "class_names": class_names,
        "class_histogram": {str(key): value for key, value in sorted(class_histogram.items())},
        "requirements": [requirement.__dict__ for requirement in build_drone_dataset_plan()],
    }


def load_yolo_dataset_class_names(dataset_root: str | Path) -> list[str]:
    root = Path(dataset_root)
    dataset_yaml = root / "dataset.yaml"
    if not dataset_yaml.exists():
        return []
    class_names: list[str] = []
    lines = dataset_yaml.read_text(encoding="utf-8").splitlines()
    in_names = False
    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if line.startswith("- "):
                class_names.append(stripped[2:].strip())
                continue
            if not raw_line.startswith(" ") and not raw_line.startswith("\t"):
                break
    return class_names


def visualize_yolo_labels(
    image_path: str | Path,
    label_path: str | Path,
    output_path: str | Path | None = None,
    class_names: Sequence[str] | None = None,
    line_width: int = 2,
) -> Image.Image:
    image_file = Path(image_path)
    label_file = Path(label_path)

    with Image.open(image_file) as source_image:
        image = source_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    boxes = load_yolo_labels(label_file, image_width=image.width, image_height=image.height)

    for class_id, bbox in boxes:
        color = _color_for_class(class_id)
        draw.rectangle(
            [(bbox.left, bbox.top), (bbox.right, bbox.bottom)],
            outline=color,
            width=line_width,
        )
        label = _label_for_class(class_id, class_names=class_names)
        text_origin = (bbox.left + 2.0, max(bbox.top - 14.0, 0.0))
        draw.text(text_origin, label, fill=color)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)

    return image


def create_visualizations(
    output_root: str | Path,
    split: str,
    limit: int,
) -> list[Path]:
    if limit <= 0:
        return []

    root = Path(output_root)
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    visualizations_dir = root / "visualizations" / split
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    created_paths: list[Path] = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        output_path = visualizations_dir / image_path.name
        visualize_yolo_labels(image_path=image_path, label_path=label_path, output_path=output_path)
        created_paths.append(output_path)

        if len(created_paths) >= limit:
            break

    return created_paths


def resolve_dataset_root(dataset_root: str | Path, project_root: str | Path) -> Path:
    requested = Path(dataset_root)
    root = Path(project_root)

    candidates = []
    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.append(root / requested)
        candidates.append(root / requested.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not resolve dataset root from {dataset_root}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert VisDrone DET annotations to YOLO format.")
    parser.add_argument(
        "--train-root",
        type=Path,
        default=Path("data/visdrone/VisDrone2019-DET-train"),
        help="Path to the VisDrone training split.",
    )
    parser.add_argument(
        "--val-root",
        type=Path,
        default=Path("data/visdrone/VisDrone2019-DET-val"),
        help="Path to the VisDrone validation split.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/visdrone_yolo"),
        help="Output directory for YOLO-formatted images and labels.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=3,
        help="Number of visualization previews to save for each split.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[3]
    train_root = resolve_dataset_root(args.train_root, project_root=project_root)
    val_root = resolve_dataset_root(args.val_root, project_root=project_root)
    output_root = args.output_root.resolve() if args.output_root.is_absolute() else (project_root / args.output_root).resolve()

    summaries = convert_visdrone_dataset(
        train_root=train_root,
        val_root=val_root,
        output_root=output_root,
    )

    for split in ("train", "val"):
        create_visualizations(output_root=output_root, split=split, limit=args.preview_count)

    for summary in summaries:
        print(
            f"[{summary.split}] images={summary.images_processed} "
            f"labels={summary.labels_written} "
            f"objects={summary.objects_written} "
            f"ignored={summary.ignored_objects} "
            f"clipped={summary.clipped_objects}"
        )


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    colors = (
        (231, 76, 60),
        (46, 204, 113),
        (52, 152, 219),
        (241, 196, 15),
        (155, 89, 182),
        (26, 188, 156),
        (230, 126, 34),
        (149, 165, 166),
        (52, 73, 94),
        (243, 156, 18),
        (192, 57, 43),
    )
    return colors[class_id % len(colors)]


def _label_for_class(class_id: int, class_names: Sequence[str] | None) -> str:
    if class_names is None or class_id >= len(class_names):
        return f"class_{class_id}"
    return class_names[class_id]


__all__ = [
    "DatasetRequirement",
    "SplitConversionSummary",
    "VisDroneObject",
    "YoloObject",
    "build_drone_dataset_plan",
    "convert_annotation_file",
    "convert_object_to_yolo",
    "convert_visdrone_dataset",
    "convert_visdrone_split",
    "create_visualizations",
    "load_visdrone_annotations",
    "load_yolo_dataset_class_names",
    "load_yolo_labels",
    "parse_visdrone_annotation_line",
    "resolve_dataset_root",
    "summarize_yolo_dataset",
    "visualize_yolo_labels",
]
