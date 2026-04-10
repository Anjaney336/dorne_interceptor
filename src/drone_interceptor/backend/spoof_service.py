"""
spoof_service.py
----------------
Modular Spoof Injection Service for the Drone Interceptor pipeline.

Architecture
------------
This module implements the *Spoof Injection* component described in the
hackathon architecture document (Section 1 / Figure 1 / Figure 2).

It is intentionally **isolated from all core business logic**.  The only
public entry point is :func:`apply_spoof`, which is called by the backend
middleware hook when the ``SPOOFING`` feature flag is active.  When the flag
is **OFF** the function returns the original data immediately — the pipeline
behaves exactly as before.

Supported Modalities
--------------------
+--------+-----------------------------------------+---------------------------+
| Type   | Technique (Phase 1 — lightweight)       | Planned (Phase 2 — ML)    |
+========+=========================================+===========================+
| image  | Gaussian adversarial noise + JPEG quant | StyleGAN / FGSM attack    |
| audio  | Pitch-shift stub (placeholder)          | Tacotron TTS voice clone  |
| video  | Per-frame noise + frame-drop sim        | DeepFaceLab face swap     |
+--------+-----------------------------------------+---------------------------+

Design Patterns Used
--------------------
- **Feature Flag** — controlled by :mod:`drone_interceptor.backend.feature_flags`
- **Plugin/Strategy Pattern** — each modality handler is a private function,
  swappable without touching the public API.
- **Dependency Injection** — :class:`SpoofService` accepts an optional
  ``rng`` for deterministic testing.

Usage
-----
    from drone_interceptor.backend.feature_flags import is_enabled
    from drone_interceptor.backend.spoof_service import SpoofService

    service = SpoofService()

    # In request handler:
    if is_enabled("SPOOFING"):
        data = service.apply_spoof(data, data_type="image")
    result = model_inference(data)

Privacy / Legal Note
--------------------
All spoof transforms in this module operate on **synthetic or provided
dataset samples only**.  No real personal images, voices, or biometric data
are generated or stored.  Follow GDPR / consent guidelines before extending
with real-person data.
"""
from __future__ import annotations

import io
import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_SUPPORTED_TYPES = frozenset({"image", "audio", "video", "unknown"})

# Default adversarial noise strength (σ in normalised [0,1] space for images)
_DEFAULT_IMAGE_NOISE_STD = 0.04
# Fraction of frames to drop/corrupt in video mode
_DEFAULT_VIDEO_DROP_RATE = 0.08
# Placeholder pitch-shift semitones for audio stub
_DEFAULT_AUDIO_PITCH_SEMITONES = 2


class SpoofResult:
    """Wrapper returned by :meth:`SpoofService.apply_spoof`.

    Attributes
    ----------
    data:
        The (possibly modified) payload — same type as input.
    spoofed:
        True if spoof transform was actually applied (flag was ON and
        transform succeeded).
    modality:
        Data type string passed to ``apply_spoof``.
    elapsed_s:
        Wall-clock time taken by the spoof transform (seconds).
    metadata:
        Audit dict with algorithm name, parameters, and timestamp.
    """

    __slots__ = ("data", "elapsed_s", "metadata", "modality", "spoofed")

    def __init__(
        self,
        data: Any,
        *,
        spoofed: bool,
        modality: str,
        elapsed_s: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.data = data
        self.spoofed = spoofed
        self.modality = modality
        self.elapsed_s = elapsed_s
        self.metadata = metadata or {}


class SpoofService:
    """Non-intrusive spoof injection service.

    Parameters
    ----------
    image_noise_std:
        Standard deviation of Gaussian noise added to image pixels
        (normalised 0–1 scale).  Increase for stronger adversarial effect.
    video_drop_rate:
        Fraction of video frames to corrupt.  0 = no corruption, 1 = all.
    rng:
        Optional :class:`numpy.random.Generator` for deterministic tests.
        Defaults to a fresh default rng seeded from OS entropy.
    """

    def __init__(
        self,
        image_noise_std: float = _DEFAULT_IMAGE_NOISE_STD,
        video_drop_rate: float = _DEFAULT_VIDEO_DROP_RATE,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._image_noise_std = float(image_noise_std)
        self._video_drop_rate = float(video_drop_rate)
        self._rng = rng if rng is not None else np.random.default_rng()

    # ── Public API ────────────────────────────────────────────────────────────

    def apply_spoof(
        self,
        data: Any,
        data_type: str = "unknown",
        *,
        force: bool = False,
    ) -> SpoofResult:
        """Apply a spoof transform to *data* of the given *data_type*.

        The function is **always safe to call** regardless of the feature flag
        state:

        * If the ``SPOOFING`` flag is OFF and *force* is False, the original
          data is returned **unchanged** (zero overhead).
        * If the flag is ON (or *force=True*), the appropriate modality
          handler is invoked.

        Parameters
        ----------
        data:
            Raw payload bytes, numpy array, or dict depending on modality.
        data_type:
            One of ``"image"``, ``"audio"``, ``"video"``, ``"unknown"``.
        force:
            Bypass the feature flag check and always apply spoofing.
            Useful for unit tests.

        Returns
        -------
        SpoofResult
            Always returns a :class:`SpoofResult` — check ``.spoofed`` to
            see if the transform was applied.
        """
        # Import here to avoid circular imports at module load time
        from drone_interceptor.backend.feature_flags import is_enabled

        modality = str(data_type).lower()
        if modality not in _SUPPORTED_TYPES:
            logger.warning("SpoofService: unknown data_type %r — treating as 'unknown'", data_type)
            modality = "unknown"

        if not force and not is_enabled("SPOOFING"):
            return SpoofResult(data, spoofed=False, modality=modality)

        t0 = time.perf_counter()
        try:
            spoofed_data, meta = self._dispatch(data, modality)
            elapsed = time.perf_counter() - t0
            logger.info(
                "SpoofService: applied %s spoof in %.4f s  metadata=%s",
                modality, elapsed, meta,
            )
            return SpoofResult(
                spoofed_data,
                spoofed=True,
                modality=modality,
                elapsed_s=elapsed,
                metadata=meta,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            logger.error(
                "SpoofService: %s spoof failed in %.4f s — returning original data.  Error: %s",
                modality, elapsed, exc,
            )
            # Fail-safe: return original data so the pipeline never blocks
            return SpoofResult(
                data,
                spoofed=False,
                modality=modality,
                elapsed_s=elapsed,
                metadata={"error": str(exc)},
            )

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def _dispatch(self, data: Any, modality: str) -> tuple[Any, dict[str, Any]]:
        if modality == "image":
            return self._spoof_image(data)
        if modality == "audio":
            return self._spoof_audio(data)
        if modality == "video":
            return self._spoof_video(data)
        # unknown / fallback: add a tiny random header byte to bytes payload
        return self._spoof_unknown(data)

    # ── Image Handler ─────────────────────────────────────────────────────────

    def _spoof_image(self, data: Any) -> tuple[Any, dict[str, Any]]:
        """Add adversarial Gaussian noise to an image.

        Accepts:
        - ``numpy.ndarray`` (H×W×C uint8 or float32)
        - ``bytes`` raw image data (JPEG/PNG) — decoded via numpy frombuffer
        - ``dict`` with key ``"image"`` → ndarray
        """
        meta: dict[str, Any] = {
            "algorithm": "gaussian_adversarial_noise",
            "noise_std": self._image_noise_std,
        }

        arr = self._coerce_to_ndarray(data)
        if arr is None:
            meta["warning"] = "could_not_decode_as_ndarray"
            return data, meta

        original_dtype = arr.dtype
        f = arr.astype(np.float32)
        if f.max() > 1.0:
            f /= 255.0
            rescaled = True
        else:
            rescaled = False

        noise = self._rng.normal(0.0, self._image_noise_std, size=f.shape).astype(np.float32)
        # Reduce noise on the z/channel-3 dimension to preserve colour fidelity
        if f.ndim == 3 and f.shape[2] == 3:
            noise[:, :, 2] *= 0.5
        f = np.clip(f + noise, 0.0, 1.0)

        if rescaled:
            f = (f * 255.0).astype(original_dtype)
        else:
            f = f.astype(original_dtype)

        # If input was a dict, put the array back under the same key
        if isinstance(data, dict):
            out = dict(data)
            for key in ("image", "frame", "rgb"):
                if key in out:
                    out[key] = f
                    break
            else:
                out["image"] = f
            return out, meta

        # If input was bytes, we return the modified ndarray
        # (caller is responsible for re-encoding if needed)
        return f, meta

    # ── Audio Handler (stub) ──────────────────────────────────────────────────

    def _spoof_audio(self, data: Any) -> tuple[Any, dict[str, Any]]:
        """Audio spoof placeholder.

        Phase 1 implementation: returns data unchanged but sets metadata to
        indicate a pitch-shift would be applied.  Full TTS/voice-clone
        integration is planned for Phase 2 (requires ``librosa`` / TTS engine).
        """
        meta: dict[str, Any] = {
            "algorithm": "pitch_shift_stub",
            "pitch_semitones": _DEFAULT_AUDIO_PITCH_SEMITONES,
            "status": "stub_not_applied",
            "note": (
                "Phase-1 audio spoofing is a stub.  "
                "Wire in librosa.effects.pitch_shift or a TTS engine for Phase 2."
            ),
        }
        logger.debug("SpoofService: audio spoof stub — returning original data")
        return data, meta

    # ── Video Handler ─────────────────────────────────────────────────────────

    def _spoof_video(self, data: Any) -> tuple[Any, dict[str, Any]]:
        """Corrupt a video represented as a sequence of frames.

        Accepts:
        - ``list`` / ``tuple`` of numpy arrays (frames)
        - ``numpy.ndarray`` of shape (N, H, W, C)
        - ``bytes`` → treated as a single JPEG frame (falls back to image spoof)

        Strategy:
        1. Randomly drop (zero-out) ``video_drop_rate`` fraction of frames.
        2. Apply image-level Gaussian noise to remaining frames.
        """
        meta: dict[str, Any] = {
            "algorithm": "frame_drop_plus_noise",
            "drop_rate": self._video_drop_rate,
            "image_noise_std": self._image_noise_std,
        }

        # Coerce to list of arrays
        if isinstance(data, (list, tuple)):
            frames = list(data)
        elif isinstance(data, np.ndarray) and data.ndim == 4:
            frames = [data[i] for i in range(data.shape[0])]
        elif isinstance(data, bytes):
            # Single-frame bytes — delegate to image spoof
            return self._spoof_image(data)
        else:
            meta["warning"] = "unrecognised_video_format"
            return data, meta

        n_frames = len(frames)
        drop_mask = self._rng.random(n_frames) < self._video_drop_rate
        meta["frames_total"] = n_frames
        meta["frames_dropped"] = int(drop_mask.sum())

        spoofed_frames: list[Any] = []
        for i, frame in enumerate(frames):
            if drop_mask[i]:
                # Zero-out frame to simulate packet loss / frame drop
                if isinstance(frame, np.ndarray):
                    spoofed_frames.append(np.zeros_like(frame))
                else:
                    spoofed_frames.append(frame)
            else:
                # Apply image noise
                noised, _ = self._spoof_image(frame)
                spoofed_frames.append(noised)

        # Return in same container type as input
        if isinstance(data, np.ndarray):
            try:
                return np.stack(spoofed_frames, axis=0), meta
            except Exception:  # noqa: BLE001
                return spoofed_frames, meta
        return type(data)(spoofed_frames) if isinstance(data, (list, tuple)) else spoofed_frames, meta

    # ── Unknown / fallback ───────────────────────────────────────────────────

    def _spoof_unknown(self, data: Any) -> tuple[Any, dict[str, Any]]:
        """Fallback for unrecognised data types.

        Inserts a random byte at position 0 if data is bytes; otherwise
        returns unchanged with a warning.
        """
        meta: dict[str, Any] = {"algorithm": "unknown_fallback"}
        if isinstance(data, (bytes, bytearray)):
            junk_byte = bytes([int(self._rng.integers(0, 256))])
            return junk_byte + bytes(data), meta
        meta["warning"] = "no_transform_applied"
        return data, meta

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _coerce_to_ndarray(data: Any) -> np.ndarray | None:
        """Try to extract a numpy array from various input formats."""
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, dict):
            for key in ("image", "frame", "rgb"):
                val = data.get(key)
                if val is not None:
                    return np.asarray(val)
        if isinstance(data, bytes):
            try:
                arr = np.frombuffer(data, dtype=np.uint8)
                import cv2  # optional, only try if available
                decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if decoded is not None:
                    return decoded
            except Exception:  # noqa: BLE001
                pass
            # Last resort: treat raw bytes as 1-D uint8 array
            return np.frombuffer(data, dtype=np.uint8)
        return None


# ── Module-level singleton (optional convenience) ─────────────────────────────
_DEFAULT_SERVICE: SpoofService | None = None


def get_default_service() -> SpoofService:
    """Return a module-level singleton :class:`SpoofService` instance."""
    global _DEFAULT_SERVICE  # noqa: PLW0603
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = SpoofService()
    return _DEFAULT_SERVICE


__all__ = ["SpoofResult", "SpoofService", "get_default_service"]
