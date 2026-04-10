from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.perception.detector import TargetDetector


def _base_config() -> dict:
    return {
        "perception": {
            "detector_backend": "ultralytics",
            "model_path": "models/does_not_need_to_exist.pt",
            "confidence_threshold": 0.4,
            "inference_imgsz": 640,
            "image_width": 1280,
            "image_height": 720,
        }
    }


def test_detector_uses_synthetic_fallback_without_image() -> None:
    detector = TargetDetector(_base_config())
    observation = {
        "target_position": np.array([10.0, 20.0, 30.0]),
    }

    detection = detector.detect(observation)

    assert np.allclose(detection.position, np.array([10.0, 20.0, 30.0]))
    assert detection.metadata["backend"] == "synthetic"
    assert detection.metadata["position_space"] == "world"


def test_detector_extracts_frame_when_available() -> None:
    detector = TargetDetector(_base_config())
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    extracted = detector._extract_image({"frame": image})

    assert extracted is not None
    assert extracted.shape == (32, 32, 3)
