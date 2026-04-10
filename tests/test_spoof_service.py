"""
tests/test_spoof_service.py
----------------------------
Unit tests for :class:`SpoofService` and :mod:`feature_flags` integration.

Run with:
    pytest tests/test_spoof_service.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.backend.feature_flags import is_enabled, reset_flags, set_flag
from drone_interceptor.backend.spoof_service import SpoofResult, SpoofService


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_feature_flags():
    """Ensure feature flags are clean before and after every test."""
    reset_flags()
    yield
    reset_flags()


@pytest.fixture()
def service() -> SpoofService:
    """Return a deterministic SpoofService instance."""
    return SpoofService(rng=np.random.default_rng(42))


@pytest.fixture()
def sample_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture()
def sample_frames(sample_image) -> list[np.ndarray]:
    return [sample_image.copy() for _ in range(8)]


# ── Flag OFF — passthrough ────────────────────────────────────────────────────

class TestSpoofFlagOff:
    def test_image_returns_original_data_unchanged(self, service, sample_image):
        """When SPOOFING flag is OFF, apply_spoof must return data untouched."""
        result = service.apply_spoof(sample_image, data_type="image")
        assert isinstance(result, SpoofResult)
        assert result.spoofed is False
        assert np.array_equal(result.data, sample_image), "Data must be byte-identical when flag is OFF"

    def test_video_returns_original_data_unchanged(self, service, sample_frames):
        result = service.apply_spoof(sample_frames, data_type="video")
        assert result.spoofed is False
        for original, returned in zip(sample_frames, result.data):
            assert np.array_equal(original, returned)

    def test_audio_returns_original_data_unchanged(self, service):
        audio = b"\x00\x01\x02\x03"
        result = service.apply_spoof(audio, data_type="audio")
        assert result.spoofed is False
        assert result.data == audio

    def test_unknown_type_returns_original_data_unchanged(self, service):
        payload = b"binary_blob"
        result = service.apply_spoof(payload, data_type="unknown")
        assert result.spoofed is False

    def test_flag_off_zero_elapsed(self, service, sample_image):
        """Passthrough should take negligible time (< 5 ms)."""
        result = service.apply_spoof(sample_image, data_type="image")
        assert result.elapsed_s < 0.005


# ── Flag ON — transform applied ───────────────────────────────────────────────

class TestSpoofFlagOn:
    def test_image_data_modified_when_flag_on(self, service, sample_image):
        set_flag("SPOOFING", True)
        result = service.apply_spoof(sample_image, data_type="image")
        assert result.spoofed is True
        assert result.modality == "image"
        # At least one pixel must differ from the original
        assert not np.array_equal(result.data, sample_image), "Spoofed image must differ from original"

    def test_image_shape_preserved(self, service, sample_image):
        set_flag("SPOOFING", True)
        result = service.apply_spoof(sample_image, data_type="image")
        assert result.data.shape == sample_image.shape

    def test_image_dtype_preserved(self, service, sample_image):
        set_flag("SPOOFING", True)
        result = service.apply_spoof(sample_image, data_type="image")
        assert result.data.dtype == sample_image.dtype

    def test_image_pixel_values_in_valid_range(self, service, sample_image):
        set_flag("SPOOFING", True)
        result = service.apply_spoof(sample_image, data_type="image")
        assert int(result.data.min()) >= 0
        assert int(result.data.max()) <= 255

    def test_video_data_modified_when_flag_on(self, service, sample_frames):
        set_flag("SPOOFING", True)
        result = service.apply_spoof(sample_frames, data_type="video")
        assert result.spoofed is True
        assert result.modality == "video"
        # At least one frame must differ
        any_changed = any(
            not np.array_equal(orig, spoof)
            for orig, spoof in zip(sample_frames, result.data)
        )
        assert any_changed, "At least one video frame must be modified when flag is ON"

    def test_video_frame_count_preserved(self, service, sample_frames):
        set_flag("SPOOFING", True)
        result = service.apply_spoof(sample_frames, data_type="video")
        assert len(result.data) == len(sample_frames)

    def test_metadata_contains_algorithm(self, service, sample_image):
        set_flag("SPOOFING", True)
        result = service.apply_spoof(sample_image, data_type="image")
        assert "algorithm" in result.metadata

    def test_elapsed_time_recorded(self, service, sample_image):
        set_flag("SPOOFING", True)
        result = service.apply_spoof(sample_image, data_type="image")
        assert result.elapsed_s >= 0.0


# ── Force bypass flag check ───────────────────────────────────────────────────

class TestSpoofForce:
    def test_force_true_applies_spoof_even_when_flag_off(self, service, sample_image):
        """force=True must apply transform regardless of feature flag state."""
        assert not is_enabled("SPOOFING")
        result = service.apply_spoof(sample_image, data_type="image", force=True)
        assert result.spoofed is True

    def test_force_false_respects_flag_off(self, service, sample_image):
        result = service.apply_spoof(sample_image, data_type="image", force=False)
        assert result.spoofed is False


# ── Invalid data_type handling ────────────────────────────────────────────────

class TestSpoofInvalidType:
    def test_invalid_type_is_normalised_to_unknown(self, service):
        """An unrecognised data_type must not raise — it falls back to 'unknown'."""
        set_flag("SPOOFING", True)
        result = service.apply_spoof(b"test", data_type="laser_scan")
        # Should NOT raise; modality falls back gracefully
        assert result.modality == "unknown"

    def test_none_type_normalised_gracefully(self, service):
        set_flag("SPOOFING", True)
        result = service.apply_spoof(b"test", data_type="none")
        assert isinstance(result, SpoofResult)


# ── Dict input (sensor fusion payload) ───────────────────────────────────────

class TestSpoofDictInput:
    def test_dict_with_image_key_is_spoofed(self, service, sample_image):
        set_flag("SPOOFING", True)
        payload = {"image": sample_image, "timestamp": 1234.5}
        result = service.apply_spoof(payload, data_type="image")
        assert result.spoofed is True
        assert "image" in result.data
        assert not np.array_equal(result.data["image"], sample_image)

    def test_dict_preserves_other_keys(self, service, sample_image):
        set_flag("SPOOFING", True)
        payload = {"image": sample_image, "sensor_id": "cam_0", "timestamp": 99.9}
        result = service.apply_spoof(payload, data_type="image")
        assert result.data.get("sensor_id") == "cam_0"
        assert result.data.get("timestamp") == 99.9


# ── Feature flag runtime override ─────────────────────────────────────────────

class TestFeatureFlags:
    def test_flag_default_is_false(self):
        assert is_enabled("SPOOFING") is False

    def test_set_flag_true_enables(self):
        set_flag("SPOOFING", True)
        assert is_enabled("SPOOFING") is True

    def test_reset_clears_override(self):
        set_flag("SPOOFING", True)
        reset_flags()
        assert is_enabled("SPOOFING") is False

    def test_flag_case_insensitive(self):
        set_flag("spoofing", True)
        assert is_enabled("SPOOFING") is True

    def test_unknown_flag_returns_false(self):
        assert is_enabled("NO_SUCH_FLAG") is False
