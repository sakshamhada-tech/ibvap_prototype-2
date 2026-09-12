"""Heuristic motion-signature scoring from short centroid trajectories.

This signal is intentionally modest: image-plane oscillation cannot reliably
separate people, animals, vegetation, and rigid objects at border distances.
It is unvalidated on operational footage and must not replace thermal sensing,
a trained classifier, or human review.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

Point = tuple[float, float]
TimedPoint = tuple[float, Point]


@dataclass(frozen=True)
class MotionSignature:
    """A confidence-scored heuristic signal, not a hard object classifier."""

    category: str
    score: float
    confidence: float
    net_displacement: float
    zero_crossing_ratio: float


class MotionSignatureAnalyzer:
    def __init__(
        self,
        *,
        window_seconds: float,
        minimum_samples: int,
        static_path_threshold_pixels: float,
        vegetation_max_net_displacement_pixels: float,
        high_frequency_zero_crossing_ratio: float,
        displacement_scale_pixels: float,
        oscillation_scale_pixels: float,
        static_score: float,
        vegetation_score: float,
        gait_score: float,
        rigid_score: float,
        ambiguous_score: float,
    ):
        self.window_seconds = window_seconds
        self.minimum_samples = minimum_samples
        self.static_path_threshold_pixels = static_path_threshold_pixels
        self.vegetation_max_net_displacement_pixels = vegetation_max_net_displacement_pixels
        self.high_frequency_zero_crossing_ratio = high_frequency_zero_crossing_ratio
        self.displacement_scale_pixels = displacement_scale_pixels
        self.oscillation_scale_pixels = oscillation_scale_pixels
        self.static_score = static_score
        self.vegetation_score = vegetation_score
        self.gait_score = gait_score
        self.rigid_score = rigid_score
        self.ambiguous_score = ambiguous_score

    def analyze(
        self, samples: Sequence[TimedPoint], timestamp: float | None = None
    ) -> MotionSignature:
        """Score purposeful motion from a recent trajectory.

        A high score means the trajectory is more consistent with net,
        potentially purposeful movement. It does not establish whether the
        source is a human or an animal.
        """
        if timestamp is None and samples:
            timestamp = samples[-1][0]
        if timestamp is not None:
            cutoff = timestamp - self.window_seconds
            samples = tuple(sample for sample in samples if sample[0] >= cutoff)
        if len(samples) < 2:
            return MotionSignature("insufficient", 0.0, 0.0, 0.0, 0.0)

        deltas = [
            (current[1][0] - previous[1][0], current[1][1] - previous[1][1])
            for previous, current in zip(samples, samples[1:], strict=False)
        ]
        path_length = sum(math.hypot(dx, dy) for dx, dy in deltas)
        net_displacement = math.dist(samples[0][1], samples[-1][1])
        vertical = [dy for _dx, dy in deltas]
        signs = [1 if value > 0 else -1 for value in vertical if abs(value) > 1e-9]
        crossings = sum(first != second for first, second in zip(signs, signs[1:], strict=False))
        zero_crossing_ratio = crossings / max(1, len(signs) - 1)
        confidence = min(1.0, len(samples) / self.minimum_samples)

        if path_length < self.static_path_threshold_pixels:
            return MotionSignature(
                "static",
                self.static_score * confidence,
                confidence,
                net_displacement,
                zero_crossing_ratio,
            )

        mean_vertical = sum(vertical) / len(vertical)
        oscillation = sum(abs(value - mean_vertical) for value in vertical) / len(vertical)
        oscillation_strength = min(1.0, oscillation / self.oscillation_scale_pixels)
        displacement_strength = min(1.0, net_displacement / self.displacement_scale_pixels)

        if (
            net_displacement <= self.vegetation_max_net_displacement_pixels
            and zero_crossing_ratio >= self.high_frequency_zero_crossing_ratio
        ):
            category = "oscillating_no_net_motion"
            base_score = self.vegetation_score
        elif (
            net_displacement > self.vegetation_max_net_displacement_pixels
            and oscillation_strength > 0
        ):
            category = "directional_rhythmic_motion"
            base_score = self.gait_score
        elif net_displacement > self.vegetation_max_net_displacement_pixels:
            category = "directional_rigid_motion"
            base_score = self.rigid_score
        else:
            category = "ambiguous_motion"
            base_score = self.ambiguous_score

        score = base_score * confidence * max(displacement_strength, oscillation_strength)
        return MotionSignature(
            category,
            max(0.0, min(1.0, score)),
            confidence,
            net_displacement,
            zero_crossing_ratio,
        )
