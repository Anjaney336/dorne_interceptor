from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.tracking.tracker import TrajectoryHistory, yolo_result_to_deepsort_detections


class _FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


def test_yolo_result_to_deepsort_detections_converts_boxes() -> None:
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=_FakeTensor(np.array([[10.0, 20.0, 40.0, 70.0]], dtype=float)),
            conf=_FakeTensor(np.array([0.8], dtype=float)),
            cls=_FakeTensor(np.array([3], dtype=float)),
            __len__=lambda self: 1,
        ),
        names={3: "car"},
    )

    detections = yolo_result_to_deepsort_detections(result)

    assert detections == [([10.0, 20.0, 30.0, 50.0], 0.8, "car")]


def test_trajectory_history_prunes_stale_tracks() -> None:
    history = TrajectoryHistory(max_length=3)
    history.append("1", (10, 10))
    history.append("1", (12, 12))
    history.append("2", (20, 20))

    history.prune({"1"})

    assert history.get("1") == [(10, 10), (12, 12)]
    assert history.get("2") == []
