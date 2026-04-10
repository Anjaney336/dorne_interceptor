"""
scripts/augment_spoof_data.py
------------------------------
Data augmentation utilities for the Spoof Injection pipeline.

Generates adversarial / augmented training data across three modalities:
  - Image  : Gaussian noise, JPEG compression artifacts, brightness jitter
  - Audio  : Pitch-shift ±2 semitones, speed change ±10% (stub — requires librosa)
  - Video  : Frame-drop simulation, temporal blur, per-frame noise

Usage
-----
    python scripts/augment_spoof_data.py --modality image --input path/to/img.png --output out/
    python scripts/augment_spoof_data.py --modality video --input path/to/frames/ --output out/
    python scripts/augment_spoof_data.py --modality audio --input path/to/audio.wav --output out/

All functions can also be imported directly:
    from scripts.augment_spoof_data import augment_image, augment_video_frames
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image augmentation
# ---------------------------------------------------------------------------

def augment_image(
    image: np.ndarray,
    *,
    noise_std: float = 0.04,
    jpeg_quality: int | None = 45,
    brightness_shift: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply adversarial augmentations to an image array (H×W×C uint8).

    Parameters
    ----------
    image:
        Input image as numpy uint8 array.
    noise_std:
        Gaussian noise strength in [0,1] normalised pixel space.
    jpeg_quality:
        If set, round-trip encode/decode as JPEG at this quality (1–95).
        Simulates JPEG compression artifacts seen in adversarial data.
    brightness_shift:
        Additive brightness offset in [0,1] space. Use small values (e.g. 0.05).
    rng:
        Optional numpy random Generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()

    img = image.astype(np.float32)
    in_255_scale = img.max() > 1.0
    if in_255_scale:
        img /= 255.0

    # ── Gaussian adversarial noise ────────────────────────────────────────
    noise = rng.normal(0.0, noise_std, size=img.shape).astype(np.float32)
    img = np.clip(img + noise, 0.0, 1.0)

    # ── Brightness jitter ─────────────────────────────────────────────────
    if brightness_shift != 0.0:
        img = np.clip(img + float(brightness_shift), 0.0, 1.0)

    if in_255_scale:
        img = (img * 255.0).astype(np.uint8)
    else:
        img = img.astype(image.dtype)

    # ── JPEG compression artifacts ────────────────────────────────────────
    if jpeg_quality is not None:
        try:
            import cv2
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
            _, encoded = cv2.imencode(".jpg", img, encode_param)
            img = cv2.imdecode(np.frombuffer(encoded.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        except ImportError:
            logger.warning("cv2 not available — skipping JPEG compression augmentation")
        except Exception as exc:
            logger.warning("JPEG augmentation failed: %s", exc)

    return img


def augment_image_batch(
    images: list[np.ndarray],
    noise_std: float = 0.04,
    jpeg_quality: int | None = 45,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Augment a batch of images (list of H×W×C uint8 arrays)."""
    if rng is None:
        rng = np.random.default_rng()
    return [augment_image(img, noise_std=noise_std, jpeg_quality=jpeg_quality, rng=rng) for img in images]


# ---------------------------------------------------------------------------
# Audio augmentation (stub — full implementation requires librosa)
# ---------------------------------------------------------------------------

def augment_audio(
    audio: bytes | np.ndarray,
    *,
    pitch_semitones: float = 2.0,
    speed_factor: float = 1.0,
) -> bytes | np.ndarray:
    """Audio augmentation stub.

    Phase 1: Returns audio unchanged with a log notice.
    Phase 2: Wire in ``librosa.effects.pitch_shift`` or a TTS engine.

    Parameters
    ----------
    audio:
        Raw audio bytes (WAV) or float32 numpy array.
    pitch_semitones:
        Pitch shift in semitones (±). Not yet applied — requires librosa.
    speed_factor:
        Speed multiplier (e.g. 1.1 = 10% faster). Not yet applied.
    """
    logger.info(
        "augment_audio (stub): pitch_semitones=%.1f, speed_factor=%.2f — returning original. "
        "Wire in librosa.effects.pitch_shift for Phase 2.",
        pitch_semitones, speed_factor,
    )
    # Phase 2 would call:
    #   import librosa
    #   y, sr = librosa.load(io.BytesIO(audio), sr=None)
    #   y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_semitones)
    #   y_speed = librosa.effects.time_stretch(y_shifted, rate=speed_factor)
    return audio


# ---------------------------------------------------------------------------
# Video augmentation
# ---------------------------------------------------------------------------

def augment_video_frames(
    frames: list[np.ndarray],
    *,
    drop_rate: float = 0.08,
    noise_std: float = 0.025,
    temporal_blur_weight: float = 0.2,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Apply video-level spoof augmentations to a sequence of frames.

    Strategy:
    1. Randomly zero-out (drop) ``drop_rate`` fraction of frames to simulate
       packet loss / frame corruption.
    2. Apply per-frame Gaussian noise to surviving frames.
    3. Apply temporal blur: blend each frame with its predecessor (motion artifact).

    Parameters
    ----------
    frames:
        List of H×W×C uint8 numpy arrays.
    drop_rate:
        Fraction of frames to drop (zero-out).  0 = keep all, 1 = drop all.
    noise_std:
        Gaussian noise strength in normalised [0,1] pixel space.
    temporal_blur_weight:
        How much of the previous frame to mix in (0 = none, 1 = full ghosting).
    rng:
        Optional RNG for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(frames)
    drop_mask = rng.random(n) < drop_rate
    augmented: list[np.ndarray] = []
    prev_frame: np.ndarray | None = None

    for idx, frame in enumerate(frames):
        if drop_mask[idx]:
            # Zero-out = dropped/lost frame
            out = np.zeros_like(frame)
        else:
            out = augment_image(frame, noise_std=noise_std, jpeg_quality=None, rng=rng)
            if temporal_blur_weight > 0.0 and prev_frame is not None:
                # Temporal blur: mix with previous frame
                out = (
                    (1.0 - temporal_blur_weight) * out.astype(np.float32)
                    + temporal_blur_weight * prev_frame.astype(np.float32)
                ).clip(0, 255).astype(np.uint8)
        augmented.append(out)
        if not drop_mask[idx]:
            prev_frame = out

    return augmented


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def load_frames_from_dir(directory: Path) -> list[np.ndarray]:
    """Load all PNG/JPG images from *directory* sorted by name."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("cv2 is required to load frames. Install opencv-python.")
    frames = []
    for path in sorted(directory.glob("*.png")) + sorted(directory.glob("*.jpg")):
        img = cv2.imread(str(path))
        if img is not None:
            frames.append(img)
    return frames


def save_frames_to_dir(frames: list[np.ndarray], directory: Path, prefix: str = "aug") -> None:
    """Save a list of frame arrays to *directory* as PNG files."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("cv2 is required to save frames. Install opencv-python.")
    directory.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        cv2.imwrite(str(directory / f"{prefix}_{idx:05d}.png"), frame)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spoof injection data augmentation script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--modality", choices=["image", "audio", "video"], required=True)
    parser.add_argument("--input", type=Path, required=True, help="Input file or directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--noise-std", type=float, default=0.04)
    parser.add_argument("--drop-rate", type=float, default=0.08)
    parser.add_argument("--jpeg-quality", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)

    if args.modality == "image":
        try:
            import cv2
            img = cv2.imread(str(args.input))
            if img is None:
                logger.error("Could not load image: %s", args.input)
                sys.exit(1)
            aug = augment_image(img, noise_std=args.noise_std, jpeg_quality=args.jpeg_quality, rng=rng)
            out_path = args.output / f"aug_{args.input.name}"
            cv2.imwrite(str(out_path), aug)
            logger.info("Saved augmented image → %s", out_path)
        except ImportError:
            logger.error("cv2 required for image augmentation. pip install opencv-python")
            sys.exit(1)

    elif args.modality == "video":
        frames = load_frames_from_dir(args.input)
        if not frames:
            logger.error("No frames found in %s", args.input)
            sys.exit(1)
        aug_frames = augment_video_frames(frames, drop_rate=args.drop_rate, noise_std=args.noise_std, rng=rng)
        save_frames_to_dir(aug_frames, args.output)
        logger.info("Saved %d augmented frames → %s", len(aug_frames), args.output)

    elif args.modality == "audio":
        audio = args.input.read_bytes()
        aug = augment_audio(audio)
        out_path = args.output / f"aug_{args.input.name}"
        out_path.write_bytes(aug if isinstance(aug, bytes) else bytes(aug))
        logger.info("Audio stub — saved original → %s", out_path)


if __name__ == "__main__":
    main()
