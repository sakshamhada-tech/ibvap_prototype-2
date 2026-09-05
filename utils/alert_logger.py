"""Durable, rate-limited and size-bounded alert logging."""

from __future__ import annotations

import csv
import io
import logging
import os
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone

from utils.log_rotation import exceeds_limit, rotate_file

LOGGER = logging.getLogger(__name__)
AlertCallback = Callable[[dict], None]


class AlertLogger:
    """Write alerts to CSV and optionally publish them to a callback.

    Cooldowns use caller-supplied source time, so recorded-video behavior does
    not vary with processing speed. Callback failures are isolated from the
    analytics pipeline after the alert has been persisted.
    """

    HEADER = [
        "timestamp_utc",
        "source_time_seconds",
        "alert_type",
        "track_id",
        "details",
        "frame_number",
    ]

    def __init__(
        self,
        csv_path: str,
        on_alert: AlertCallback | None = None,
        cooldown_seconds: float = 1.0,
        state_ttl_seconds: float = 300.0,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ):
        self.csv_path = csv_path
        self.on_alert = on_alert
        self.cooldown_seconds = cooldown_seconds
        self.state_ttl_seconds = state_ttl_seconds
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.total_logged = 0
        self._lock = threading.Lock()
        self._closed = False
        self._recent_alerts: dict[tuple[str, object], float] = {}

        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        self._open_file()

    def _open_file(self) -> None:
        needs_header = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
        # The handle intentionally stays open until close() so every event can
        # be synchronously flushed without reopening the file each frame.
        self._fh = open(  # noqa: SIM115
            self.csv_path, mode="a", newline="", encoding="utf-8"
        )
        self._writer = csv.writer(self._fh)
        if needs_header:
            self._writer.writerow(self.HEADER)
            self._fh.flush()

    def log(
        self,
        alert_type: str,
        track_id,
        details: str,
        frame_number: int,
        *,
        source_time: float | None = None,
    ) -> bool:
        """Persist an alert; return ``False`` when its cooldown suppresses it."""
        event_time = time.monotonic() if source_time is None else float(source_time)
        key = (alert_type, track_id)

        with self._lock:
            if self._closed:
                raise RuntimeError("cannot log to a closed AlertLogger")
            self._prune_state(event_time)
            last_logged = self._recent_alerts.get(key)
            if last_logged is not None and 0 <= event_time - last_logged < self.cooldown_seconds:
                return False
            self._recent_alerts[key] = event_time

            timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            alert = {
                "timestamp": timestamp,
                "source_time_seconds": round(event_time, 3),
                "alert_type": alert_type,
                "track_id": track_id,
                "details": details,
                "frame_number": frame_number,
            }
            row = [
                timestamp,
                alert["source_time_seconds"],
                alert_type,
                track_id,
                details,
                frame_number,
            ]
            self._rotate_if_needed(row)
            self._writer.writerow(row)
            self._fh.flush()
            self.total_logged += 1

        LOGGER.warning(
            "alert type=%s track=%s frame=%s",
            alert_type,
            track_id,
            frame_number,
        )
        if self.on_alert:
            try:
                self.on_alert(alert)
            except Exception:
                LOGGER.exception("alert callback failed")
        return True

    def _rotate_if_needed(self, row: list) -> None:
        buffer = io.StringIO(newline="")
        csv.writer(buffer).writerow(row)
        pending_bytes = len(buffer.getvalue().encode("utf-8"))
        if exceeds_limit(self.csv_path, pending_bytes, self.max_bytes):
            self._fh.close()
            rotate_file(self.csv_path, self.backup_count)
            self._open_file()

    def _prune_state(self, event_time: float) -> None:
        cutoff = event_time - self.state_ttl_seconds
        stale_keys = [
            key
            for key, last_logged in self._recent_alerts.items()
            if last_logged < cutoff or last_logged > event_time
        ]
        for key in stale_keys:
            del self._recent_alerts[key]

    def reset_cooldowns(self) -> None:
        with self._lock:
            self._recent_alerts.clear()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._fh.close()
                self._closed = True
