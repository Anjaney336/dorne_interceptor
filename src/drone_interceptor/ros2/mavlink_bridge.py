from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any


LOGGER = logging.getLogger(__name__)


_SEVERITY_MAP = {
    "EMERGENCY": 0,
    "ALERT": 1,
    "CRITICAL": 2,
    "ERROR": 3,
    "WARNING": 4,
    "NOTICE": 5,
    "INFO": 6,
    "DEBUG": 7,
}


@dataclass(frozen=True, slots=True)
class StatusTextEvent:
    text: str
    severity: str
    severity_code: int
    sent: bool
    timestamp: float
    transport: str


class MavlinkBridge:
    """Send MAVLink STATUSTEXT events to GCS when a MAVLink endpoint is available."""

    def __init__(
        self,
        connection_uri: str = "udpout:127.0.0.1:14550",
        source_system: int = 251,
        source_component: int = 190,
    ) -> None:
        self._connection_uri = str(connection_uri)
        self._source_system = int(source_system)
        self._source_component = int(source_component)
        self._conn: Any | None = None
        self._mavutil: Any | None = None
        self._ready = False
        self._init_failed = False

    def _ensure_connection(self) -> bool:
        if self._ready and self._conn is not None:
            return True
        if self._init_failed:
            return False
        try:
            from pymavlink import mavutil  # type: ignore
        except Exception:
            self._init_failed = True
            return False
        try:
            conn = mavutil.mavlink_connection(
                self._connection_uri,
                source_system=self._source_system,
                source_component=self._source_component,
            )
            self._conn = conn
            self._mavutil = mavutil
            self._ready = True
            return True
        except Exception as exc:  # pragma: no cover - transport depends on host
            LOGGER.warning("MAVLink bridge unavailable: %s", exc)
            self._init_failed = True
            return False

    def send_statustext(self, text: str, severity: str = "INFO") -> StatusTextEvent:
        normalized = str(severity).strip().upper()
        severity_code = int(_SEVERITY_MAP.get(normalized, _SEVERITY_MAP["INFO"]))
        payload = str(text).strip()[:120] or "STATUS"
        timestamp = time.time()

        if self._ensure_connection() and self._conn is not None:
            try:
                self._conn.mav.statustext_send(severity_code, payload.encode("utf-8"))
                return StatusTextEvent(
                    text=payload,
                    severity=normalized,
                    severity_code=severity_code,
                    sent=True,
                    timestamp=timestamp,
                    transport="mavlink",
                )
            except Exception as exc:  # pragma: no cover - transport depends on host
                LOGGER.warning("MAVLink STATUSTEXT send failed: %s", exc)

        LOGGER.info("[STATUSTEXT:%s] %s", normalized, payload)
        return StatusTextEvent(
            text=payload,
            severity=normalized,
            severity_code=severity_code,
            sent=False,
            timestamp=timestamp,
            transport="logger",
        )


__all__ = ["MavlinkBridge", "StatusTextEvent"]
