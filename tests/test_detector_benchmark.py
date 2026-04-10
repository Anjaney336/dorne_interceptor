from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.datasets.visdrone import build_drone_dataset_plan, load_yolo_dataset_class_names, summarize_yolo_dataset
from drone_interceptor.perception.detector import DetectionBox, benchmark_detection_sets


def test_drone_dataset_plan_contains_required_competition_classes() -> None:
    plan = build_drone_dataset_plan()
    categories = {item.category for item in plan}
    assert "quadcopters" in categories
    assert "hexacopters" in categories
    assert "birds" in categories
    assert "urban_clutter" in categories


def test_benchmark_detection_sets_scores_matches() -> None:
    summary = benchmark_detection_sets(
        ground_truth_sets=[
            [DetectionBox(0, 0, 10, 10, class_id=0)],
            [DetectionBox(5, 5, 15, 15, class_id=1)],
        ],
        prediction_sets=[
            [DetectionBox(0, 0, 10, 10, confidence=0.9, class_id=0)],
            [DetectionBox(5, 5, 15, 15, confidence=0.8, class_id=1)],
        ],
        inference_times_s=[0.05, 0.05],
        device_label="test",
    )
    assert summary.precision == 1.0
    assert summary.recall == 1.0
    assert summary.map50 == 1.0
    assert summary.fps > 0.0


def test_summarize_yolo_dataset_reports_histogram() -> None:
    summary = summarize_yolo_dataset(ROOT / "data" / "visdrone_yolo")
    assert "splits" in summary
    assert "class_names" in summary
    assert "requirements" in summary


def test_load_yolo_dataset_class_names_reads_dataset_yaml() -> None:
    names = load_yolo_dataset_class_names(ROOT / "data" / "visdrone_yolo")
    assert "car" in names
