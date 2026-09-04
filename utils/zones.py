"""
Zones & Geometry
----------------
Small, dependency-free helpers for:
  - Detecting when a moving track crosses the virtual fence line
  - Detecting loitering (a track that stays roughly in place too long)

Kept separate from detector/main so the geometry logic is easy to unit
test or swap out (e.g. for a polygon fence instead of a line) later.
"""

import time
from collections import defaultdict


def _ccw(a, b, c):
    """Orientation test used for segment-intersection (standard algorithm)."""
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1, p2, p3, p4):
    """True if line segment p1->p2 intersects segment p3->p4."""
    return _ccw(p1, p3, p4) != _ccw(p2, p3, p4) and _ccw(p1, p2, p3) != _ccw(p1, p2, p4)


class VirtualFence:
    """
    Tracks each object's previous centroid and checks whether the movement
    from previous -> current centroid crosses the configured fence line.
    """

    def __init__(self, line):
        self.line_p1, self.line_p2 = line
        self._last_centroid = {}  # track_id -> (x, y)

    def check_crossing(self, track_id, centroid):
        crossed = False
        prev = self._last_centroid.get(track_id)
        if prev is not None:
            crossed = segments_intersect(prev, centroid, self.line_p1, self.line_p2)
        self._last_centroid[track_id] = centroid
        return crossed


class LoiteringDetector:
    """
    Flags a track as "loitering" if its centroid stays within a small
    radius of where it was `loitering_seconds` ago.
    """

    def __init__(self, loitering_seconds: float, movement_threshold_px: float):
        self.loitering_seconds = loitering_seconds
        self.movement_threshold_px = movement_threshold_px
        # track_id -> list of (timestamp, centroid)
        self._history = defaultdict(list)
        # track_id -> timestamp this track was first observed (independent of
        # history trimming below - this is what tells us whether enough time
        # has actually elapsed to judge loitering)
        self._first_seen = {}
        self._already_flagged = set()

    def update(self, track_id, centroid):
        now = time.time()
        if track_id not in self._first_seen:
            self._first_seen[track_id] = now

        hist = self._history[track_id]
        hist.append((now, centroid))

        # drop entries older than the loitering window (keep it bounded)
        cutoff = now - self.loitering_seconds
        while hist and hist[0][0] < cutoff:
            hist.pop(0)

        # not enough elapsed time since we first saw this track -> can't judge yet
        if now - self._first_seen[track_id] < self.loitering_seconds:
            return False

        old_centroid = hist[0][1]
        dx = centroid[0] - old_centroid[0]
        dy = centroid[1] - old_centroid[1]
        drift = (dx ** 2 + dy ** 2) ** 0.5

        is_loitering = drift < self.movement_threshold_px
        if is_loitering and track_id not in self._already_flagged:
            self._already_flagged.add(track_id)
            return True

        if not is_loitering and track_id in self._already_flagged:
            self._already_flagged.discard(track_id)

        return False
