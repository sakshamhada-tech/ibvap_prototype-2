"""
Alert Logger
------------
Central place where every module (fence, loitering, night mode, ...)
reports an event. Writes a row to a CSV log, prints a console banner, and
optionally invokes a callback (used by the web dashboard to push the alert
to the browser in real time over a websocket).
"""

import csv
import os
from datetime import datetime


class AlertLogger:
    def __init__(self, csv_path: str, on_alert=None):
        """
        on_alert: optional callable(alert_dict) invoked every time a new
        (non-suppressed) alert is actually written. Keeps this class usable
        both standalone (CLI demo) and wired into the dashboard.
        """
        self.csv_path = csv_path
        self.on_alert = on_alert
        self.total_logged = 0

        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

        self._file_exists = os.path.exists(csv_path)
        self._fh = open(csv_path, mode="a", newline="")
        self._writer = csv.writer(self._fh)

        if not self._file_exists:
            self._writer.writerow(
                ["timestamp", "alert_type", "track_id", "details", "frame_number"]
            )
            self._fh.flush()

        # avoid spamming the same alert every single frame
        self._recent_alerts = {}  # (alert_type, track_id) -> last_frame_logged
        self._cooldown_frames = 30  # ~1 second at 30fps

    def log(self, alert_type: str, track_id, details: str, frame_number: int):
        """Returns True if the alert was actually recorded, False if suppressed."""
        key = (alert_type, track_id)
        last_frame = self._recent_alerts.get(key, -9999)
        if frame_number - last_frame < self._cooldown_frames:
            return False  # suppress duplicate/rapid-fire alerts for same track+type
        self._recent_alerts[key] = frame_number

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._writer.writerow([timestamp, alert_type, track_id, details, frame_number])
        self._fh.flush()
        self.total_logged += 1

        print(f"[ALERT] {timestamp} | {alert_type:<20} | track={track_id} | {details}")

        if self.on_alert:
            self.on_alert({
                "timestamp": timestamp,
                "alert_type": alert_type,
                "track_id": track_id,
                "details": details,
                "frame_number": frame_number,
            })

        return True

    def close(self):
        self._fh.close()
