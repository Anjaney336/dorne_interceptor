from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class VideoWriters:
    primary: cv2.VideoWriter
    primary_path: Path
    fps: float
    compatibility: cv2.VideoWriter | None
    compatibility_path: Path | None
    _released: bool = False

    def __enter__(self) -> "VideoWriters":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.release()

    def write(self, frame: np.ndarray) -> None:
        self.primary.write(frame)
        if self.compatibility is not None:
            self.compatibility.write(frame)

    def release(self) -> None:
        if self._released:
            return
        self.primary.release()
        if self.compatibility is not None:
            self.compatibility.release()
        self._released = True
        _transcode_browser_ready(
            output_path=self.primary_path,
            fps=self.fps,
            source_path=self.compatibility_path if self.compatibility_path is not None and self.compatibility_path.exists() else self.primary_path,
        )


def build_video_writers(output: str | Path, fps: float, frame_size: tuple[int, int]) -> VideoWriters:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    primary = _open_writer(
        output=output_path,
        fps=fps,
        frame_size=frame_size,
        codecs=("mp4v", "avc1", "H264", "X264"),
    )

    compatibility_path = output_path.with_suffix(".avi")
    compatibility = _open_writer(
        output=compatibility_path,
        fps=fps,
        frame_size=frame_size,
        codecs=("MJPG",),
        strict=False,
    )
    return VideoWriters(
        primary=primary,
        primary_path=output_path,
        fps=float(fps),
        compatibility=compatibility,
        compatibility_path=compatibility_path if compatibility is not None else None,
    )


def _open_writer(
    output: Path,
    fps: float,
    frame_size: tuple[int, int],
    codecs: tuple[str, ...],
    strict: bool = True,
) -> cv2.VideoWriter | None:
    for codec in codecs:
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*codec),
            float(fps),
            frame_size,
        )
        if writer.isOpened():
            return writer
        writer.release()
    if strict:
        raise RuntimeError(f"OpenCV could not open a video writer for {output}.")
    return None


def _transcode_browser_ready(output_path: Path, fps: float, source_path: Path | None = None) -> None:
    if output_path.suffix.lower() != ".mp4":
        return
    ffmpeg_path = _resolve_ffmpeg_path()
    if ffmpeg_path is None or not output_path.exists():
        return
    source = output_path if source_path is None else Path(source_path)
    if not source.exists():
        return

    browser_ready_path = output_path.with_name(f"{output_path.stem}.browser_ready{output_path.suffix}")
    command = [
        ffmpeg_path,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        "-r",
        f"{float(fps):.2f}",
        str(browser_ready_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        if browser_ready_path.exists():
            browser_ready_path.unlink(missing_ok=True)
        return

    for _ in range(5):
        try:
            browser_ready_path.replace(output_path)
            return
        except PermissionError:
            time.sleep(0.2)
    browser_ready_path.unlink(missing_ok=True)


def normalize_video_artifact(output_path: str | Path, fps: float | None = None) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".mp4" or not path.exists():
        return path
    resolved_fps = float(fps) if fps is not None else _probe_fps(path)
    compatibility_path = path.with_suffix(".avi")
    _transcode_browser_ready(
        output_path=path,
        fps=resolved_fps,
        source_path=compatibility_path if compatibility_path.exists() else path,
    )
    return path


def _resolve_ffmpeg_path() -> str | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is not None:
        return ffmpeg_path
    try:
        import imageio_ffmpeg  # type: ignore
    except Exception:
        return None
    try:
        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None


def _probe_fps(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    return fps if fps > 0.0 else 20.0


__all__ = ["VideoWriters", "build_video_writers", "normalize_video_artifact"]
