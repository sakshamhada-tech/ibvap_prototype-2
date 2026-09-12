"""Lightweight coordinated-group approach signals from tracked centroids.

This module deliberately uses image-plane geometry rather than a learned model.
It can flag a compact group whose members move in correlated directions toward
one configured fence segment, but perspective and track-ID errors can still
produce false positives or false negatives. Results require human review.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass

Point = tuple[float, float]
TrackPoint = tuple[int, Point]


@dataclass(frozen=True)
class GroupApproachEvent:
    """A newly qualifying coordinated approach episode."""

    track_ids: tuple[int, ...]
    centroid: Point
    heading: Point


def point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    """Return the shortest image-plane distance from a point to a segment."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 0:
        return math.dist(point, start)
    projection = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
    projection = max(0.0, min(1.0, projection))
    nearest = (start[0] + projection * dx, start[1] + projection * dy)
    return math.dist(point, nearest)


class GroupApproachDetector:
    """Track compact, correlated person groups approaching one fence segment."""

    def __init__(
        self,
        *,
        fence_line: tuple[Point, Point],
        minimum_size: int,
        proximity_pixels: float,
        window_seconds: float,
        minimum_speed_pixels_per_second: float,
        heading_similarity: float,
        minimum_fence_progress_pixels: float,
        stale_after_seconds: float,
    ):
        self.fence_line = fence_line
        self.minimum_size = minimum_size
        self.proximity_pixels = proximity_pixels
        self.window_seconds = window_seconds
        self.minimum_speed_pixels_per_second = minimum_speed_pixels_per_second
        self.heading_similarity = heading_similarity
        self.minimum_fence_progress_pixels = minimum_fence_progress_pixels
        self.stale_after_seconds = stale_after_seconds
        self._history: dict[int, deque[tuple[float, Point]]] = defaultdict(deque)
        self._last_seen: dict[int, float] = {}
        self._active_groups: set[frozenset[int]] = set()
        self._grouped_track_ids: frozenset[int] = frozenset()
        self._approach_scores: dict[int, float] = {}

    def update(
        self, tracks: Iterable[TrackPoint], timestamp: float
    ) -> tuple[GroupApproachEvent, ...]:
        """Update visible people and return only newly qualifying group episodes."""
        visible = dict(tracks)
        cutoff = timestamp - self.window_seconds
        for track_id, point in visible.items():
            previous = self._last_seen.get(track_id)
            if previous is not None and (
                timestamp < previous or timestamp - previous > self.stale_after_seconds
            ):
                self._drop_track(track_id)
            history = self._history[track_id]
            history.append((timestamp, point))
            while len(history) > 1 and history[1][0] <= cutoff:
                history.popleft()
            self._last_seen[track_id] = timestamp

        present_groups: set[frozenset[int]] = set()
        qualifying_groups: set[frozenset[int]] = set()
        grouped_ids: set[int] = set()
        approach_scores = {
            track_id: score
            for track_id in visible
            if (score := self._individual_approach_score(track_id)) > 0
        }
        events = []
        for cluster in self._clusters(visible):
            if len(cluster) < self.minimum_size:
                continue
            group_key = frozenset(cluster)
            present_groups.add(group_key)
            qualification = self._qualify(cluster)
            if qualification is None:
                continue
            heading, member_scores = qualification
            qualifying_groups.add(group_key)
            grouped_ids.update(cluster)
            approach_scores.update(member_scores)
            if group_key not in self._active_groups:
                points = [visible[track_id] for track_id in cluster]
                events.append(
                    GroupApproachEvent(
                        track_ids=tuple(sorted(cluster)),
                        centroid=(
                            sum(point[0] for point in points) / len(points),
                            sum(point[1] for point in points) / len(points),
                        ),
                        heading=heading,
                    )
                )

        # Keep an alerted episode latched while that compact group remains
        # visible, even if one noisy frame fails the heading test.
        self._active_groups = {
            group for group in self._active_groups if group in present_groups
        } | qualifying_groups
        self._grouped_track_ids = frozenset(grouped_ids)
        self._approach_scores = approach_scores
        self.expire(timestamp)
        return tuple(events)

    def _clusters(self, visible: dict[int, Point]) -> list[set[int]]:
        remaining = set(visible)
        clusters = []
        while remaining:
            seed = remaining.pop()
            cluster = {seed}
            frontier = [seed]
            while frontier:
                current = frontier.pop()
                neighbors = {
                    candidate
                    for candidate in remaining
                    if math.dist(visible[current], visible[candidate]) <= self.proximity_pixels
                }
                remaining.difference_update(neighbors)
                cluster.update(neighbors)
                frontier.extend(neighbors)
            clusters.append(cluster)
        return clusters

    def _individual_approach_score(self, track_id: int) -> float:
        history = self._history[track_id]
        if len(history) < 2:
            return 0.0
        first_time, first = history[0]
        last_time, last = history[-1]
        duration = last_time - first_time
        if duration <= 0:
            return 0.0
        distance = math.dist(first, last)
        if distance / duration < self.minimum_speed_pixels_per_second:
            return 0.0
        progress = point_to_segment_distance(first, *self.fence_line) - point_to_segment_distance(
            last, *self.fence_line
        )
        if progress <= 0:
            return 0.0
        return min(1.0, progress / max(self.proximity_pixels, 1.0))

    def _qualify(self, cluster: set[int]) -> tuple[Point, dict[int, float]] | None:
        headings = {}
        progress = {}
        for track_id in cluster:
            history = self._history[track_id]
            if len(history) < 2:
                return None
            first_time, first = history[0]
            last_time, last = history[-1]
            duration = last_time - first_time
            if duration <= 0:
                return None
            dx, dy = last[0] - first[0], last[1] - first[1]
            distance = math.hypot(dx, dy)
            speed = distance / duration
            if speed < self.minimum_speed_pixels_per_second:
                return None
            headings[track_id] = (dx / distance, dy / distance)
            fence_progress = point_to_segment_distance(
                first, *self.fence_line
            ) - point_to_segment_distance(last, *self.fence_line)
            if fence_progress < self.minimum_fence_progress_pixels:
                return None
            progress[track_id] = min(1.0, fence_progress / max(self.proximity_pixels, 1.0))

        mean_x = sum(heading[0] for heading in headings.values()) / len(headings)
        mean_y = sum(heading[1] for heading in headings.values()) / len(headings)
        coherence = math.hypot(mean_x, mean_y)
        if coherence < self.heading_similarity:
            return None
        mean_heading = (mean_x / coherence, mean_y / coherence)
        member_scores = {
            track_id: max(
                0.0,
                min(
                    1.0,
                    (
                        (heading[0] * mean_heading[0] + heading[1] * mean_heading[1])
                        + progress[track_id]
                    )
                    / 2,
                ),
            )
            for track_id, heading in headings.items()
        }
        return mean_heading, member_scores

    def approach_score(self, track_id: int) -> float:
        return self._approach_scores.get(track_id, 0.0)

    @property
    def grouped_track_ids(self) -> frozenset[int]:
        return self._grouped_track_ids

    def history(self, track_id: int) -> tuple[tuple[float, Point], ...]:
        return tuple(self._history.get(track_id, ()))

    def expire(self, timestamp: float) -> None:
        cutoff = timestamp - self.stale_after_seconds
        stale = [
            track_id
            for track_id, last_seen in self._last_seen.items()
            if last_seen < cutoff or last_seen > timestamp
        ]
        for track_id in stale:
            self._drop_track(track_id)
        if stale:
            stale_set = set(stale)
            self._active_groups = {
                group for group in self._active_groups if group.isdisjoint(stale_set)
            }
            self._grouped_track_ids = frozenset(self._grouped_track_ids - stale_set)

    def _drop_track(self, track_id: int) -> None:
        self._history.pop(track_id, None)
        self._last_seen.pop(track_id, None)
        self._approach_scores.pop(track_id, None)

    def reset(self) -> None:
        self._history.clear()
        self._last_seen.clear()
        self._active_groups.clear()
        self._grouped_track_ids = frozenset()
        self._approach_scores.clear()

    @property
    def tracked_count(self) -> int:
        return len(self._last_seen)
