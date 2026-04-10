from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUT_VIDEO = OUTPUTS / "final_project_walkthrough.mp4"
OUT_NOTE = OUTPUTS / "final_project_walkthrough.md"


def _draw_multiline(
    frame: np.ndarray,
    lines: Iterable[str],
    x: int,
    y: int,
    line_height: int = 34,
    scale: float = 0.8,
    color: tuple[int, int, int] = (230, 245, 255),
    thickness: int = 2,
) -> None:
    yy = y
    for line in lines:
        cv2.putText(frame, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        yy += line_height


def _create_slide(title: str, bullets: list[str], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        alpha = row / max(height - 1, 1)
        frame[row, :, :] = (
            int(8 + 10 * alpha),
            int(14 + 16 * alpha),
            int(22 + 28 * alpha),
        )
    cv2.rectangle(frame, (24, 24), (width - 24, height - 24), (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, title, (48, 84), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 240, 255), 2, cv2.LINE_AA)
    text_lines = [f"- {bullet}" for bullet in bullets]
    _draw_multiline(frame, text_lines, x=56, y=140, line_height=38, scale=0.76, color=(232, 245, 255), thickness=2)
    return frame


def _append_frame(writer: cv2.VideoWriter, frame: np.ndarray, frames: int) -> None:
    for _ in range(max(frames, 1)):
        writer.write(frame)


def build_walkthrough_video() -> tuple[Path, Path]:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    width, height = 1280, 720
    fps = 24
    writer = cv2.VideoWriter(
        str(OUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter for walkthrough output.")

    slides = [
        (
            "Drone Interceptor Final Walkthrough",
            [
                "This recording summarizes the final dashboard implementation.",
                "It covers all tabs, live workflow behavior, and backend service metrics.",
                "Build focus: low-latency inputs, complete telemetry, and readable analytics.",
            ],
        ),
        (
            "Overview Tab",
            [
                "Live Simulation Panel renders trajectory and active stage progression.",
                "Scenario Results table shows per-target outputs from backend mission workflow.",
                "Mission Code Console mirrors running logic snippets during replay.",
            ],
        ),
        (
            "VisDrone Analysis Suite",
            [
                "Training Analysis: precision/recall/mAP and train-vs-val losses.",
                "Dataset Analysis: split volumes, class distribution, weighted interceptor priority.",
                "Equations Panel: interception, RMSE, packet-loss, power, energy, and kill formulas.",
            ],
        ),
        (
            "3D Mission + Map Tabs",
            [
                "3D Mission View shows target/interceptor/spoof paths and stage overlays.",
                "Map tab provides tactical geospatial state and telemetry overlay values.",
                "Swarm map and stream sections visualize multi-UAV mission replay context.",
            ],
        ),
        (
            "Live Analytics Tab",
            [
                "Timeseries and comparison panels track distance, cost, velocity, and effort.",
                "Mission Trends use stored history with readiness/success/RMSE/risk metrics.",
                "Validation plots compare raw vs EKF for success and miss-distance behavior.",
            ],
        ),
        (
            "Architecture & Hardware Tab",
            [
                "Contains mission equations, ROS2 roles, spoof toggle semantics, and hardware matrix.",
                "Command readiness and quality/deployment gates remain visible and explicit.",
                "SolidWorks and judge sweep sections remain available for system-level review.",
            ],
        ),
    ]

    for title, bullets in slides:
        slide = _create_slide(title=title, bullets=bullets, size=(width, height))
        _append_frame(writer, slide, frames=int(3.8 * fps))

    mission_mp4 = OUTPUTS / "mission_final.mp4"
    if mission_mp4.exists():
        cap = cv2.VideoCapture(str(mission_mp4))
        if cap.isOpened():
            sample_count = 0
            max_frames = int(8 * fps)
            while sample_count < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                cv2.putText(
                    frame,
                    "Simulation Replay Segment",
                    (34, 46),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 245, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)
                sample_count += 1
            cap.release()

    writer.release()
    OUT_NOTE.write_text(
        "\n".join(
            [
                "# Final Project Walkthrough",
                "",
                f"- Video: `{OUT_VIDEO}`",
                "- Includes all tab explanations and a simulation replay segment.",
                "- Generated by `scripts/generate_final_walkthrough.py`.",
            ]
        ),
        encoding="utf-8",
    )
    return OUT_VIDEO, OUT_NOTE


if __name__ == "__main__":
    video_path, note_path = build_walkthrough_video()
    print(f"Generated walkthrough video: {video_path}")
    print(f"Generated walkthrough note: {note_path}")
