"""Explainable, config-driven composite risk scoring.

The scorer is a transparent weighted heuristic because this repository has no
labelled border-risk dataset from which to train or validate a predictive
model. A threshold crossing is a review signal, never an automated response.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class RiskSignals:
    fence_contact: float = 0.0
    fence_proximity: float = 0.0
    loitering: float = 0.0
    motion_signature: float = 0.0
    low_posture: float = 0.0
    night: float = 0.0
    fence_approach: float = 0.0
    group_membership: float = 0.0


@dataclass(frozen=True)
class RiskAssessment:
    score: float
    crossed: bool
    contributions: tuple[tuple[str, float], ...]

    def details(self) -> str:
        reasons = ", ".join(f"{name}={value:.2f}" for name, value in self.contributions)
        return f"score={self.score:.2f}; signals: {reasons or 'none'}; human review required"


class RiskScorer:
    """Own per-track threshold state and emit once per elevated episode."""

    def __init__(
        self,
        *,
        threshold: float,
        weights: dict[str, float],
        stale_after_seconds: float,
    ):
        signal_names = {field.name for field in fields(RiskSignals)}
        unknown = set(weights) - signal_names
        if unknown:
            raise ValueError(f"unknown risk weights: {', '.join(sorted(unknown))}")
        self.threshold = threshold
        self.weights = {name: weights.get(name, 0.0) for name in signal_names}
        self.stale_after_seconds = stale_after_seconds
        self._active: set[int] = set()
        self._last_seen: dict[int, float] = {}
        self._scores: dict[int, float] = {}

    def update(self, track_id: int, signals: RiskSignals, timestamp: float) -> RiskAssessment:
        contributions = []
        total = 0.0
        for signal_field in fields(signals):
            name = signal_field.name
            value = max(0.0, min(1.0, float(getattr(signals, name))))
            contribution = value * self.weights[name]
            total += contribution
            if contribution > 0:
                contributions.append((name, contribution))
        score = max(0.0, min(1.0, total))
        was_active = track_id in self._active
        elevated = score >= self.threshold
        if elevated:
            self._active.add(track_id)
        else:
            self._active.discard(track_id)
        self._last_seen[track_id] = timestamp
        self._scores[track_id] = score
        contributions.sort(key=lambda item: (-item[1], item[0]))
        return RiskAssessment(score, elevated and not was_active, tuple(contributions))

    def score_for(self, track_id: int) -> float:
        return self._scores.get(track_id, 0.0)

    def expire(self, timestamp: float) -> None:
        cutoff = timestamp - self.stale_after_seconds
        stale = [
            track_id
            for track_id, seen in self._last_seen.items()
            if seen < cutoff or seen > timestamp
        ]
        for track_id in stale:
            self._last_seen.pop(track_id, None)
            self._scores.pop(track_id, None)
            self._active.discard(track_id)

    def reset(self) -> None:
        self._active.clear()
        self._last_seen.clear()
        self._scores.clear()

    @property
    def tracked_count(self) -> int:
        return len(self._last_seen)
