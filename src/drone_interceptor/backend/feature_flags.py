"""
feature_flags.py
----------------
Lightweight, environment-variable-backed feature flag store.

Usage
-----
    from drone_interceptor.backend.feature_flags import is_enabled, set_flag, reset_flags

    # Check at runtime
    if is_enabled("SPOOFING"):
        data = spoof_service.apply_spoof(data, "image")

    # Override programmatically (for tests / admin API)
    set_flag("SPOOFING", True)

    # Reset all runtime overrides (restore env-var defaults)
    reset_flags()

Environment Variables
---------------------
Each flag ``<NAME>`` maps to ``DRONE_FEATURE_<NAME>`` (uppercased).
Values "1", "true", "yes", "on" are truthy; anything else is falsy.
Unset variables default to **False** (safe-off principle).
"""
from __future__ import annotations

import os
import threading
from typing import Any

# ── Runtime override table (takes precedence over env vars) ───────────────────
_LOCK: threading.Lock = threading.Lock()
_OVERRIDES: dict[str, bool] = {}


def is_enabled(name: str) -> bool:
    """Return True if the feature flag *name* is active.

    Precedence (highest → lowest):
    1. Runtime override set via :func:`set_flag`.
    2. Environment variable ``DRONE_FEATURE_<NAME>``.
    3. Default: **False**.
    """
    key = name.upper()
    with _LOCK:
        if key in _OVERRIDES:
            return bool(_OVERRIDES[key])
    env_value = os.environ.get(f"DRONE_FEATURE_{key}", "").strip().lower()
    return env_value in {"1", "true", "yes", "on"}


def set_flag(name: str, value: bool) -> None:
    """Set a runtime override for feature flag *name*.

    This overrides the environment variable for the lifetime of the process
    (or until :func:`reset_flags` is called).
    """
    with _LOCK:
        _OVERRIDES[name.upper()] = bool(value)


def reset_flags() -> None:
    """Clear all runtime overrides; env-var values take effect again."""
    with _LOCK:
        _OVERRIDES.clear()


def all_flags() -> dict[str, Any]:
    """Return a snapshot of all known flags and their effective values.

    Scans environment for any ``DRONE_FEATURE_*`` vars and merges with
    runtime overrides.
    """
    discovered: dict[str, bool] = {}
    for key, value in os.environ.items():
        if key.startswith("DRONE_FEATURE_"):
            flag_name = key[len("DRONE_FEATURE_"):]
            discovered[flag_name] = value.strip().lower() in {"1", "true", "yes", "on"}
    with _LOCK:
        discovered.update(_OVERRIDES)
    return dict(discovered)


__all__ = ["all_flags", "is_enabled", "reset_flags", "set_flag"]
