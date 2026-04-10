from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIR = ROOT / "data" / "visdrone_yolo" / "images" / "train"
LABEL_DIR = ROOT / "data" / "visdrone_yolo" / "labels" / "train"


def draw_real_sensor_panel(
    frame: np.ndarray,
    panel: tuple[int, int, int, int],
    step_index: int,
    title: str = "Real Sensor Frames",
    subtitle: str = "VisDrone aerial imagery",
) -> None:
    samples = _load_sensor_samples()
    x0, y0, x1, y1 = panel
    cv2.rectangle(frame, (x0, y0), (x1, y1), (18, 18, 18), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (210, 210, 210), 1)
    cv2.putText(frame, title, (x0 + 12, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (236, 236, 236), 1, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (x0 + 12, y0 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (176, 176, 176), 1, cv2.LINE_AA)

    if not samples:
        cv2.putText(frame, "dataset imagery unavailable", (x0 + 14, y0 + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
        return

    thumb_width = max((x1 - x0 - 30) // 2, 64)
    thumb_height = max(y1 - y0 - 66, 64)
    for slot in range(2):
        sample = samples[(step_index + slot) % len(samples)]
        thumb = cv2.resize(sample, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        tx0 = x0 + 10 + slot * (thumb_width + 10)
        ty0 = y0 + 54
        tx1 = tx0 + thumb_width
        ty1 = ty0 + thumb_height
        frame[ty0:ty1, tx0:tx1] = thumb
        cv2.rectangle(frame, (tx0, ty0), (tx1, ty1), (228, 228, 228), 1)


@lru_cache(maxsize=1)
def _load_sensor_samples() -> tuple[np.ndarray, ...]:
    samples: list[np.ndarray] = []
    if not IMAGE_DIR.exists():
        return ()

    for image_path in sorted(IMAGE_DIR.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        annotated = _annotate_frame(image=image, label_path=label_path)
        samples.append(annotated)
        if len(samples) >= 12:
            break
    return tuple(samples)


def _annotate_frame(image: np.ndarray, label_path: Path) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines()[:12]:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, cx, cy, w, h = (float(value) for value in parts)
            box_w = max(int(w * width), 2)
            box_h = max(int(h * height), 2)
            center_x = int(cx * width)
            center_y = int(cy * height)
            x0 = max(center_x - box_w // 2, 0)
            y0 = max(center_y - box_h // 2, 0)
            x1 = min(center_x + box_w // 2, width - 1)
            y1 = min(center_y + box_h // 2, height - 1)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (80, 255, 170), 1, cv2.LINE_AA)
    cv2.rectangle(annotated, (0, 0), (width - 1, height - 1), (236, 236, 236), 1)
    cv2.putText(annotated, "REAL SENSOR FRAME", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (248, 248, 248), 2, cv2.LINE_AA)
    cv2.putText(annotated, "VisDrone dataset", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
    return annotated


__all__ = ["draw_real_sensor_panel"]
