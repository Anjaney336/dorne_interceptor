from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[3]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from drone_interceptor.perception.train import (
    DEFAULT_MODEL_NAME,
    DEFAULT_RUN_NAME,
    resolve_project_path,
    resolve_project_root,
)
from drone_interceptor.tracking.tracker import DeepSortTracker, draw_tracked_objects


WINDOW_NAME = "YOLOv10 Inference"


def _import_yolo() -> type:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is not installed in the active environment. "
            "Install requirements.txt before running inference."
        ) from exc

    return YOLO


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is not installed in the active environment. "
            "Install requirements.txt before running inference."
        ) from exc

    return cv2


def resolve_model_path(model_path: str | Path | None = None) -> str | Path:
    project_root = resolve_project_root()
    models_root = resolve_project_path("models", project_root=project_root)

    if model_path is not None:
        requested = Path(model_path)
        candidates = [requested]
        if not requested.is_absolute():
            candidates.extend(
                [
                    project_root / requested,
                    models_root / requested,
                ]
            )

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    default_checkpoint = models_root / f"{DEFAULT_RUN_NAME}.pt"
    if default_checkpoint.exists():
        return default_checkpoint.resolve()

    best_candidates = sorted(models_root.rglob("best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if best_candidates:
        return best_candidates[0].resolve()

    any_checkpoint = sorted(models_root.rglob("*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if any_checkpoint:
        return any_checkpoint[0].resolve()

    return DEFAULT_MODEL_NAME


def annotate_frame(frame: Any, result: Any, fps: float) -> Any:
    cv2 = _import_cv2()
    names = result.names

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for box, confidence, class_id in zip(xyxy, confs, classes, strict=False):
            x1, y1, x2, y2 = [int(value) for value in box]
            color = _color_for_class(class_id)
            label_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            label = f"{label_name} {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def run_image_inference(
    model_path: str | Path | None,
    image_path: str | Path,
    conf: float = 0.25,
    imgsz: int = 640,
    track: bool = False,
    history_length: int = 30,
    output_path: str | Path | None = None,
    show: bool = True,
) -> Path | None:
    YOLO = _import_yolo()
    cv2 = _import_cv2()

    resolved_model_path = resolve_model_path(model_path)
    resolved_image_path = resolve_project_path(image_path)
    if not resolved_image_path.exists():
        raise FileNotFoundError(f"Image not found: {resolved_image_path}")

    frame = cv2.imread(str(resolved_image_path))
    if frame is None:
        raise ValueError(f"Failed to load image: {resolved_image_path}")

    model = YOLO(str(resolved_model_path))
    start_time = time.perf_counter()
    result = model(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    elapsed = time.perf_counter() - start_time
    fps = 1.0 / elapsed if elapsed > 0.0 else 0.0

    if track:
        tracker = DeepSortTracker(history_length=history_length)
        tracked_objects = tracker.update_from_yolo(result=result, frame=frame)
        annotated = draw_tracked_objects(frame=frame, tracked_objects=tracked_objects, fps=fps)
    else:
        annotated = annotate_frame(frame=frame, result=result, fps=fps)
    saved_path: Path | None = None

    if output_path is not None:
        saved_path = resolve_project_path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(saved_path), annotated)

    if show:
        cv2.imshow(WINDOW_NAME, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return saved_path


def run_stream_inference(
    model_path: str | Path | None,
    source: int | str,
    conf: float = 0.25,
    imgsz: int = 640,
    track: bool = False,
    history_length: int = 30,
    max_frames: int | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> Path | None:
    YOLO = _import_yolo()
    cv2 = _import_cv2()

    resolved_model_path = resolve_model_path(model_path)
    model = YOLO(str(resolved_model_path))
    capture = cv2.VideoCapture(source)
    tracker = DeepSortTracker(history_length=history_length) if track else None

    if not capture.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")

    fps = 0.0
    frame_count = 0
    writer: Any | None = None
    resolved_output_path: Path | None = resolve_project_path(output_path) if output_path is not None else None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            start_time = time.perf_counter()
            result = model(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
            if tracker is not None:
                tracked_objects = tracker.update_from_yolo(result=result, frame=frame)
            elapsed = time.perf_counter() - start_time
            instant_fps = 1.0 / elapsed if elapsed > 0.0 else 0.0
            fps = instant_fps if fps == 0.0 else (0.9 * fps) + (0.1 * instant_fps)

            if tracker is not None:
                annotated = draw_tracked_objects(frame=frame, tracked_objects=tracked_objects, fps=fps)
            else:
                annotated = annotate_frame(frame=frame, result=result, fps=fps)

            if resolved_output_path is not None and writer is None:
                resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
                source_fps = capture.get(cv2.CAP_PROP_FPS)
                source_fps = source_fps if source_fps and source_fps > 0 else 30.0
                writer = cv2.VideoWriter(
                    str(resolved_output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    source_fps,
                    (annotated.shape[1], annotated.shape[0]),
                )

            if writer is not None:
                writer.write(annotated)

            if show:
                cv2.imshow(WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    return resolved_output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLOv10 inference on an image or webcam stream.")
    parser.add_argument("--model", type=Path, default=None, help="Path to a trained model checkpoint.")
    parser.add_argument("--image", type=Path, default=None, help="Path to an input image.")
    parser.add_argument("--video", type=Path, default=None, help="Path to an input video.")
    parser.add_argument("--webcam-index", type=int, default=None, help="Webcam index for live inference.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--track", action="store_true", help="Enable DeepSORT tracking on YOLO detections.")
    parser.add_argument(
        "--history-length",
        type=int,
        default=30,
        help="Number of center points to keep for each track motion path.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for webcam testing.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save annotated image output.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable image display in image mode. Webcam mode always displays frames.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    selected_sources = [args.image is not None, args.video is not None, args.webcam_index is not None]
    if sum(bool(selected) for selected in selected_sources) != 1:
        raise ValueError("Use exactly one source: --image, --video, or --webcam-index.")

    if args.image is not None:
        saved_path = run_image_inference(
            model_path=args.model,
            image_path=args.image,
            conf=args.conf,
            imgsz=args.imgsz,
            track=args.track,
            history_length=args.history_length,
            output_path=args.output,
            show=not args.no_show,
        )
        if saved_path is not None:
            print(f"saved_output={saved_path}")
        return

    stream_source: int | str = args.webcam_index if args.webcam_index is not None else str(resolve_project_path(args.video))
    saved_path = run_stream_inference(
        model_path=args.model,
        source=stream_source,
        conf=args.conf,
        imgsz=args.imgsz,
        track=args.track,
        history_length=args.history_length,
        max_frames=args.max_frames,
        output_path=args.output,
        show=not args.no_show,
    )
    if saved_path is not None:
        print(f"saved_output={saved_path}")


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


if __name__ == "__main__":
    main()
