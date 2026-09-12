"""Stateful virtual-fence and loitering analytics.

All time values are supplied by the caller. Recorded video can therefore use
media time while live cameras use a monotonic clock, making results independent
of inference speed.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

Point = tuple[float, float]
Box = tuple[float, float, float, float]
_EPSILON = 1e-9


def _orientation(a: Point, b: Point, c: Point) -> float:
    """Signed cross product of AB and AC."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, point: Point) -> bool:
    return (
        min(a[0], b[0]) - _EPSILON <= point[0] <= max(a[0], b[0]) + _EPSILON
        and min(a[1], b[1]) - _EPSILON <= point[1] <= max(a[1], b[1]) + _EPSILON
        and abs(_orientation(a, b, point)) <= _EPSILON
    )


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Return whether two closed line segments intersect, including touching."""
    o1 = _orientation(p1, p2, p3)
    o2 = _orientation(p1, p2, p4)
    o3 = _orientation(p3, p4, p1)
    o4 = _orientation(p3, p4, p2)

    if ((o1 > _EPSILON and o2 < -_EPSILON) or (o1 < -_EPSILON and o2 > _EPSILON)) and (
        (o3 > _EPSILON and o4 < -_EPSILON) or (o3 < -_EPSILON and o4 > _EPSILON)
    ):
        return True

    return (
        (abs(o1) <= _EPSILON and _on_segment(p1, p2, p3))
        or (abs(o2) <= _EPSILON and _on_segment(p1, p2, p4))
        or (abs(o3) <= _EPSILON and _on_segment(p3, p4, p1))
        or (abs(o4) <= _EPSILON and _on_segment(p3, p4, p2))
    )


def point_in_polygon(point: Point, polygon: tuple[Point, ...] | None) -> bool:
    """Return whether a point is inside/on an optional polygon; ``None`` means anywhere."""
    if polygon is None:
        return True
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _on_segment(previous, current, point):
            return True
        if (current[1] > point[1]) != (previous[1] > point[1]):
            intersection_x = (previous[0] - current[0]) * (point[1] - current[1]) / (
                previous[1] - current[1]
            ) + current[0]
            if point[0] < intersection_x:
                inside = not inside
        previous = current
    return inside


def segment_intersects_box(
    start: Point,
    end: Point,
    box: Box,
    margin_pixels: float = 0.0,
) -> bool:
    """Return whether a segment touches an axis-aligned box, including its margin."""
    raw_x1, raw_y1, raw_x2, raw_y2 = box
    left = min(raw_x1, raw_x2) - margin_pixels
    right = max(raw_x1, raw_x2) + margin_pixels
    top = min(raw_y1, raw_y2) - margin_pixels
    bottom = max(raw_y1, raw_y2) + margin_pixels

    def inside(point: Point) -> bool:
        return (
            left - _EPSILON <= point[0] <= right + _EPSILON
            and top - _EPSILON <= point[1] <= bottom + _EPSILON
        )

    if inside(start) or inside(end):
        return True
    top_left = (left, top)
    top_right = (right, top)
    bottom_right = (right, bottom)
    bottom_left = (left, bottom)
    return any(
        segments_intersect(start, end, edge_start, edge_end)
        for edge_start, edge_end in (
            (top_left, top_right),
            (top_right, bottom_right),
            (bottom_right, bottom_left),
            (bottom_left, top_left),
        )
    )


class VirtualFence:
    """Detect tracked boxes contacting a fence and points moving across it."""

    def __init__(
        self,
        line: tuple[Point, Point],
        stale_after_seconds: float = 3.0,
        contact_margin_pixels: float = 0.0,
    ):
        self.line_p1, self.line_p2 = line
        self.stale_after_seconds = stale_after_seconds
        self.contact_margin_pixels = contact_margin_pixels
        self._last_observation: dict[int, tuple[Point, float, bool]] = {}

    def check_crossing(
        self,
        track_id: int,
        centroid: Point,
        timestamp: float,
        bounds: Box | None = None,
    ) -> bool:
        """Return true once when a tracked box contacts or moves across the fence.

        Bounding-box contact matches the visible overlays and catches a person
        whose feet or body reach the line before their centroid does. Centroid
        movement remains a fallback when fast motion skips over the line between
        frames. A contact episode is emitted only once, including when tracking
        first becomes available while the object is already on the fence.
        """
        touching = bounds is not None and segment_intersects_box(
            self.line_p1,
            self.line_p2,
            bounds,
            self.contact_margin_pixels,
        )
        crossed = bool(touching and track_id not in self._last_observation)
        previous = self._last_observation.get(track_id)
        if previous is not None:
            previous_centroid, previous_timestamp, previously_touching = previous
            gap = timestamp - previous_timestamp
            if 0 <= gap <= self.stale_after_seconds:
                entered_contact = touching and not previously_touching
                movement_crossed = False
                if not touching and not previously_touching and previous_centroid != centroid:
                    previous_side = _orientation(self.line_p1, self.line_p2, previous_centroid)
                    current_side = _orientation(self.line_p1, self.line_p2, centroid)
                    previous_on_fence = _on_segment(self.line_p1, self.line_p2, previous_centroid)
                    current_on_fence = _on_segment(self.line_p1, self.line_p2, centroid)
                    changed_side = previous_side * current_side < -_EPSILON
                    arrived_on_fence = current_on_fence and not previous_on_fence
                    movement_crossed = segments_intersect(
                        previous_centroid,
                        centroid,
                        self.line_p1,
                        self.line_p2,
                    ) and (changed_side or arrived_on_fence)
                crossed = entered_contact or movement_crossed
            else:
                # Treat an observation after a discontinuity as a new track
                # for contact purposes, but never infer motion across the gap.
                crossed = touching
        self._last_observation[track_id] = (centroid, timestamp, touching)
        return crossed

    def expire(self, timestamp: float) -> None:
        cutoff = timestamp - self.stale_after_seconds
        stale_ids = [
            track_id
            for track_id, (_, last_seen, _) in self._last_observation.items()
            if last_seen < cutoff or last_seen > timestamp
        ]
        for track_id in stale_ids:
            del self._last_observation[track_id]

    def is_touching(self, track_id: int) -> bool:
        observation = self._last_observation.get(track_id)
        return bool(observation and observation[2])

    def reset(self) -> None:
        self._last_observation.clear()

    @property
    def tracked_count(self) -> int:
        return len(self._last_observation)


@dataclass(frozen=True)
class LoiteringState:
    """Current loitering state and whether this frame began a new event."""

    active: bool
    started: bool


class LoiteringDetector:
    """Detect continuous presence inside a bounded area for a time window.

    A large observation gap resets a track. Movement is measured as the
    diagonal of the bounding box containing every centroid in the time window,
    rather than only comparing its endpoints.
    """

    def __init__(
        self,
        loitering_seconds: float,
        movement_threshold_px: float,
        max_observation_gap_seconds: float = 1.5,
        stale_after_seconds: float = 3.0,
    ):
        self.loitering_seconds = loitering_seconds
        self.movement_threshold_px = movement_threshold_px
        self.max_observation_gap_seconds = max_observation_gap_seconds
        self.stale_after_seconds = stale_after_seconds
        self._history: dict[int, deque[tuple[float, Point]]] = defaultdict(deque)
        self._last_seen: dict[int, float] = {}
        self._active: set[int] = set()

    def update(
        self,
        track_id: int,
        centroid: Point,
        timestamp: float,
        *,
        in_scope: bool = True,
    ) -> LoiteringState:
        if not in_scope:
            self._reset_track(track_id)
            return LoiteringState(active=False, started=False)
        previous_seen = self._last_seen.get(track_id)
        if (
            previous_seen is None
            or timestamp < previous_seen
            or timestamp - previous_seen > self.max_observation_gap_seconds
        ):
            self._reset_track(track_id)

        self._last_seen[track_id] = timestamp
        history = self._history[track_id]
        history.append((timestamp, centroid))

        cutoff = timestamp - self.loitering_seconds
        # Preserve the latest sample at or before the cutoff so the window's
        # full observed duration can be measured accurately.
        while len(history) > 1 and history[1][0] <= cutoff:
            history.popleft()

        observed_duration = timestamp - history[0][0]
        points = [point for _, point in history]
        width = max(point[0] for point in points) - min(point[0] for point in points)
        height = max(point[1] for point in points) - min(point[1] for point in points)
        spread = (width**2 + height**2) ** 0.5
        active = observed_duration >= self.loitering_seconds and spread < self.movement_threshold_px
        was_active = track_id in self._active

        if active:
            self._active.add(track_id)
        else:
            self._active.discard(track_id)

        return LoiteringState(active=active, started=active and not was_active)

    def expire(self, timestamp: float) -> None:
        cutoff = timestamp - self.stale_after_seconds
        stale_ids = [
            track_id
            for track_id, last_seen in self._last_seen.items()
            if last_seen < cutoff or last_seen > timestamp
        ]
        for track_id in stale_ids:
            self._reset_track(track_id)

    def _reset_track(self, track_id: int) -> None:
        self._history.pop(track_id, None)
        self._last_seen.pop(track_id, None)
        self._active.discard(track_id)

    def diagnostics(self, track_id: int, timestamp: float, *, in_scope: bool = True) -> dict:
        """Explain whether tracked-person context is ready, accumulating, or blocked."""
        history = self._history.get(track_id)
        if not in_scope:
            return {"track_id": track_id, "status": "blocked", "reason": "outside_loitering_scope"}
        if not history:
            return {"track_id": track_id, "status": "blocked", "reason": "no_track_history"}
        points = [point for _, point in history]
        width = max(point[0] for point in points) - min(point[0] for point in points)
        height = max(point[1] for point in points) - min(point[1] for point in points)
        spread = (width**2 + height**2) ** 0.5
        duration = max(0.0, timestamp - history[0][0])
        if track_id in self._active:
            status, reason = "active", None
        elif spread >= self.movement_threshold_px:
            status, reason = "blocked", "movement_exceeds_loitering_threshold"
        else:
            status, reason = "accumulating", "minimum_observation_time_not_reached"
        return {
            "track_id": track_id,
            "status": status,
            "reason": reason,
            "observed_seconds": round(duration, 3),
            "required_seconds": self.loitering_seconds,
            "movement_spread_px": round(spread, 3),
            "movement_threshold_px": self.movement_threshold_px,
        }

    def history(self, track_id: int) -> tuple[tuple[float, Point], ...]:
        """Expose immutable trajectory samples for cheap derived motion signals."""
        return tuple(self._history.get(track_id, ()))

    def reset(self) -> None:
        self._history.clear()
        self._last_seen.clear()
        self._active.clear()

    @property
    def tracked_count(self) -> int:
        return len(self._last_seen)
