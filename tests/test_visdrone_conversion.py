from pathlib import Path
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.datasets.visdrone import (
    convert_object_to_yolo,
    convert_visdrone_dataset,
    load_yolo_labels,
    parse_visdrone_annotation_line,
    visualize_yolo_labels,
)


def test_convert_object_to_yolo_normalizes_and_reindexes() -> None:
    annotation = parse_visdrone_annotation_line("10,20,40,20,1,4,0,0")

    yolo_object, was_clipped = convert_object_to_yolo(
        annotation=annotation,
        image_width=200,
        image_height=100,
    )

    assert yolo_object is not None
    assert was_clipped is False
    assert yolo_object.class_id == 3
    assert yolo_object.x_center == 0.15
    assert yolo_object.y_center == 0.30
    assert yolo_object.width == 0.20
    assert yolo_object.height == 0.20


def test_convert_object_to_yolo_ignores_invalid_class_zero() -> None:
    annotation = parse_visdrone_annotation_line("10,20,40,20,0,0,0,0")

    yolo_object, was_clipped = convert_object_to_yolo(
        annotation=annotation,
        image_width=200,
        image_height=100,
    )

    assert yolo_object is None
    assert was_clipped is False


def test_parse_visdrone_annotation_line_allows_trailing_commas() -> None:
    annotation = parse_visdrone_annotation_line("440,541,271,152,1,6,0,0,")

    assert annotation.class_id == 6
    assert annotation.bbox.left == 440.0
    assert annotation.bbox.height == 152.0


def test_convert_visdrone_dataset_creates_split_layout_and_labels(tmp_path: Path) -> None:
    train_root = tmp_path / "data" / "visdrone" / "VisDrone2019-DET-train"
    val_root = tmp_path / "data" / "visdrone" / "VisDrone2019-DET-val"
    output_root = tmp_path / "data" / "visdrone_yolo"

    for split_root in (train_root, val_root):
        (split_root / "images").mkdir(parents=True)
        (split_root / "annotations").mkdir(parents=True)

    Image.new("RGB", (100, 50), color="black").save(train_root / "images" / "sample_train.jpg")
    Image.new("RGB", (80, 40), color="black").save(val_root / "images" / "sample_val.jpg")

    (train_root / "annotations" / "sample_train.txt").write_text(
        "\n".join(
            [
                "10,5,20,10,1,4,0,0",
                "0,0,10,10,0,0,0,0",
                "90,10,20,20,1,2,0,0",
            ]
        ),
        encoding="utf-8",
    )
    (val_root / "annotations" / "sample_val.txt").write_text(
        "5,5,10,10,1,1,0,0",
        encoding="utf-8",
    )

    summaries = convert_visdrone_dataset(
        train_root=train_root,
        val_root=val_root,
        output_root=output_root,
    )

    assert [summary.split for summary in summaries] == ["train", "val"]
    assert (output_root / "images" / "train" / "sample_train.jpg").exists()
    assert (output_root / "images" / "val" / "sample_val.jpg").exists()
    assert (output_root / "labels" / "train" / "sample_train.txt").exists()
    assert (output_root / "labels" / "val" / "sample_val.txt").exists()

    train_labels = (output_root / "labels" / "train" / "sample_train.txt").read_text(encoding="utf-8").splitlines()
    val_labels = (output_root / "labels" / "val" / "sample_val.txt").read_text(encoding="utf-8").splitlines()

    assert train_labels == [
        "3 0.200000 0.200000 0.200000 0.200000",
        "1 0.950000 0.400000 0.100000 0.400000",
    ]
    assert val_labels == ["0 0.125000 0.250000 0.125000 0.250000"]

    train_summary = summaries[0]
    assert train_summary.images_processed == 1
    assert train_summary.objects_written == 2
    assert train_summary.ignored_objects == 1
    assert train_summary.clipped_objects == 1


def test_visualize_yolo_labels_draws_output_image(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    label_path = tmp_path / "frame.txt"
    output_path = tmp_path / "frame_vis.jpg"

    Image.new("RGB", (40, 20), color="black").save(image_path)
    label_path.write_text("0 0.500000 0.500000 0.500000 0.500000", encoding="utf-8")

    image = visualize_yolo_labels(image_path=image_path, label_path=label_path, output_path=output_path)
    boxes = load_yolo_labels(label_path=label_path, image_width=40, image_height=20)

    assert image.size == (40, 20)
    assert output_path.exists()
    assert len(boxes) == 1
