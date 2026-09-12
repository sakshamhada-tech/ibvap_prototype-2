"""Regional ANPR normalization, validation, and temporal consensus.

This module deliberately contains no CV/model dependencies so the decision logic
can be tested independently from optional plate-detector and OCR runtimes.
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

SUPPORTED_REGIONS = (
    "IN",
    "BD",
    "NP",
    "PK",
    "BT",
    "LK",
    "MM",
    "CN",
    "AF",
    "MV",
    "GENERIC",
)

_INDIAN_STATE_CODES = frozenset(
    {
        "AN",
        "AP",
        "AR",
        "AS",
        "BR",
        "CG",
        "CH",
        "DD",
        "DL",
        "DN",
        "GA",
        "GJ",
        "HP",
        "HR",
        "JH",
        "JK",
        "KA",
        "KL",
        "LA",
        "LD",
        "MH",
        "ML",
        "MN",
        "MP",
        "MZ",
        "NL",
        "OD",
        "OR",  # Legacy Odisha code remains visible on older vehicles.
        "PB",
        "PY",
        "RJ",
        "SK",
        "TN",
        "TR",
        "TS",
        "UK",
        "UP",
        "WB",
    }
)


@dataclass(frozen=True)
class PlateObservation:
    """One quality-gated detector/OCR result from a tracked vehicle."""

    text: str
    bbox: tuple[int, int, int, int]
    region: str
    ocr_confidence: float
    detector_confidence: float
    sharpness: float

    @property
    def quality(self) -> float:
        """Bounded confidence used for candidate ranking and temporal voting."""
        sharpness_score = min(max(self.sharpness, 0.0) / 200.0, 1.0)
        return min(
            max(
                0.70 * self.ocr_confidence
                + 0.20 * self.detector_confidence
                + 0.10 * sharpness_score,
                0.0,
            ),
            1.0,
        )


@dataclass(frozen=True)
class PlateConsensus:
    """Stable per-track read; ``newly_stable`` is true only once per text."""

    text: str
    bbox: tuple[int, int, int, int]
    region: str
    confidence: float
    observations: int
    newly_stable: bool


@dataclass(frozen=True)
class _TimedObservation:
    timestamp: float
    observation: PlateObservation


def normalize_plate_text(text: str) -> str:
    """Normalize OCR output while retaining South-Asian non-Latin scripts."""
    normalized = unicodedata.normalize("NFKC", text).upper()
    result = []
    for character in normalized:
        category = unicodedata.category(character)
        if category == "Nd":
            try:
                result.append(str(unicodedata.digit(character)))
            except (TypeError, ValueError):
                result.append(character)
        elif category.startswith("L"):
            result.append(character)
        elif category.startswith("M") and result:
            # Preserve dependent vowels/diacritics used by Indic scripts, but
            # never retain an unattached mark introduced by OCR noise.
            previous_category = unicodedata.category(result[-1])
            if previous_category.startswith(("L", "M")):
                result.append(character)
    return "".join(result)


def combine_ocr_results(results) -> tuple[str, float] | None:
    """Order EasyOCR tokens by text rows and return normalized text/confidence."""
    tokens = []
    for result in results or ():
        if len(result) < 3 or not str(result[1]).strip():
            continue
        quadrilateral, text, confidence = result[:3]
        try:
            points = [(float(point[0]), float(point[1])) for point in quadrilateral]
            center_x = sum(point[0] for point in points) / len(points)
            center_y = sum(point[1] for point in points) / len(points)
            token_height = max(point[1] for point in points) - min(point[1] for point in points)
            confidence = float(confidence)
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            continue
        normalized_text = normalize_plate_text(str(text))
        if not normalized_text or not math.isfinite(confidence):
            continue
        tokens.append((center_y, center_x, max(token_height, 1.0), normalized_text, confidence))
    if not tokens:
        return None

    lines: list[list[tuple[float, float, float, str, float]]] = []
    for token in sorted(tokens, key=lambda item: item[0]):
        for line in lines:
            line_center = sum(item[0] for item in line) / len(line)
            line_height = sum(item[2] for item in line) / len(line)
            if abs(token[0] - line_center) <= 0.6 * max(token[2], line_height):
                line.append(token)
                break
        else:
            lines.append([token])

    lines.sort(key=lambda line: sum(item[0] for item in line) / len(line))
    ordered = [token for line in lines for token in sorted(line, key=lambda item: item[1])]
    text = "".join(token[3] for token in ordered)
    weights = [max(len(token[3]), 1) for token in ordered]
    confidence = sum(
        token[4] * weight for token, weight in zip(ordered, weights, strict=True)
    ) / sum(weights)
    return text, confidence


def _contains_range(text: str, start: int, end: int) -> bool:
    return any(start <= ord(character) <= end for character in text)


def _has_letter_and_digit(text: str) -> bool:
    return any(character.isalpha() for character in text) and any(
        character.isdigit() for character in text
    )


def _ascii_letters(text: str) -> bool:
    return text.isascii() and text.isalpha()


def _valid_indian_plate(text: str) -> bool:
    if len(text) == 10 and text[:2].isdigit() and text[2:4] == "BH":
        # Bharat-series registration: 22BH1234AA.
        return text[4:8].isdigit() and _ascii_letters(text[8:])
    if "CD" in text or "CC" in text or "UN" in text:
        # Diplomatic formats vary in separator placement; separators are removed.
        return 5 <= len(text) <= 12 and text.isascii() and _has_letter_and_digit(text)
    if len(text) < 8 or len(text) > 13 or text[:2] not in _INDIAN_STATE_CODES:
        return False
    remainder = text[2:]
    district_digits = 0
    while district_digits < len(remainder) and remainder[district_digits].isdigit():
        district_digits += 1
    if district_digits not in {1, 2}:
        return False
    series_and_number = remainder[district_digits:]
    series_letters = 0
    while series_letters < len(series_and_number) and series_and_number[series_letters].isalpha():
        series_letters += 1
    return (
        series_letters in {1, 2, 3}
        and _ascii_letters(series_and_number[:series_letters])
        and series_and_number[series_letters:].isdigit()
        and len(series_and_number[series_letters:]) == 4
    )


def plate_matches_region(text: str, region: str) -> bool:
    """Apply conservative structural/script checks without correcting OCR text."""
    region = region.upper()
    if region == "IN":
        return _valid_indian_plate(text)
    if region == "BD":
        return (
            6 <= len(text) <= 20
            and _contains_range(text, 0x0980, 0x09FF)
            and any(character.isdigit() for character in text)
        )
    if region == "NP":
        return (
            5 <= len(text) <= 20
            and _contains_range(text, 0x0900, 0x097F)
            and any(character.isdigit() for character in text)
        )
    if region == "MM":
        return (
            5 <= len(text) <= 20
            and _contains_range(text, 0x1000, 0x109F)
            and any(character.isdigit() for character in text)
        )
    if region == "CN":
        return (
            6 <= len(text) <= 12
            and _contains_range(text, 0x4E00, 0x9FFF)
            and _has_letter_and_digit(text)
        )
    if region == "BT":
        return (
            7 <= len(text) <= 10
            and text.isascii()
            and text[:2] in {"BP", "BT"}
            and _has_letter_and_digit(text)
        )
    if region == "LK":
        prefix = text[:-4]
        return len(prefix) in {2, 3, 5} and _ascii_letters(prefix) and text[-4:].isdigit()
    if region == "PK":
        # Provincial formats vary substantially; accept a bounded mixed-script read.
        return 4 <= len(text) <= 12 and _has_letter_and_digit(text)
    if region == "AF":
        return (
            5 <= len(text) <= 20
            and _contains_range(text, 0x0600, 0x06FF)
            and any(character.isdigit() for character in text)
        )
    if region == "MV":
        prefix = text[:-4]
        return 5 <= len(text) <= 7 and _ascii_letters(prefix) and text[-4:].isdigit()
    if region == "GENERIC":
        return 4 <= len(text) <= 16 and _has_letter_and_digit(text)
    raise ValueError(f"unsupported ANPR region {region!r}")


def identify_plate_region(text: str, regions: tuple[str, ...]) -> str | None:
    """Return the first configured region whose structural checks accept text."""
    return next((region for region in regions if plate_matches_region(text, region)), None)


class PlateConsensusTracker:
    """Require repeated, weighted agreement before emitting an ANPR event."""

    def __init__(
        self,
        *,
        min_reads: int,
        window_seconds: float,
        min_weight_ratio: float,
        min_ocr_confidence: float,
        max_observations: int,
        state_ttl_seconds: float,
    ):
        if min_reads < 1 or max_observations < min_reads:
            raise ValueError("ANPR consensus requires max_observations >= min_reads >= 1")
        if not 0 <= min_weight_ratio <= 1 or not 0 <= min_ocr_confidence <= 1:
            raise ValueError("ANPR consensus ratios and confidences must be between zero and one")
        if window_seconds <= 0 or state_ttl_seconds <= 0:
            raise ValueError("ANPR consensus windows and state TTLs must be positive")
        self.min_reads = min_reads
        self.window_seconds = window_seconds
        self.min_weight_ratio = min_weight_ratio
        self.min_ocr_confidence = min_ocr_confidence
        self.max_observations = max_observations
        self.state_ttl_seconds = state_ttl_seconds
        self._observations: dict[int, deque[_TimedObservation]] = defaultdict(
            lambda: deque(maxlen=max_observations)
        )
        self._last_seen: dict[int, float] = {}
        self._emitted_texts: dict[int, set[str]] = defaultdict(set)

    @property
    def tracked_count(self) -> int:
        return len(self._observations)

    def add(
        self, track_id: int, observation: PlateObservation, timestamp: float
    ) -> PlateConsensus | None:
        if observation.ocr_confidence < self.min_ocr_confidence:
            return None
        samples = self._observations[track_id]
        samples.append(_TimedObservation(timestamp, observation))
        self._last_seen[track_id] = timestamp
        self._prune_track(track_id, timestamp)

        counts = Counter(sample.observation.text for sample in samples)
        weights: dict[str, float] = defaultdict(float)
        for sample in samples:
            weights[sample.observation.text] += max(sample.observation.quality, 1e-6)
        winner = max(weights, key=lambda text: (weights[text], counts[text], text))
        winner_count = counts[winner]
        total_weight = sum(weights.values())
        weight_ratio = weights[winner] / total_weight if total_weight else 0.0
        if winner_count < self.min_reads or weight_ratio < self.min_weight_ratio:
            return None

        candidates = [sample.observation for sample in samples if sample.observation.text == winner]
        best = max(candidates, key=lambda item: item.quality)
        newly_stable = winner not in self._emitted_texts[track_id]
        if newly_stable:
            self._emitted_texts[track_id].add(winner)
        return PlateConsensus(
            text=winner,
            bbox=best.bbox,
            region=best.region,
            confidence=weight_ratio,
            observations=winner_count,
            newly_stable=newly_stable,
        )

    def expire(self, timestamp: float) -> set[int]:
        """Discard stale tracks and return their IDs so callers can clear overlays."""
        expired = set()
        for track_id, last_seen in list(self._last_seen.items()):
            self._prune_track(track_id, timestamp)
            if timestamp - last_seen > self.state_ttl_seconds:
                self._discard(track_id)
                expired.add(track_id)
        return expired

    def reset(self) -> None:
        self._observations.clear()
        self._last_seen.clear()
        self._emitted_texts.clear()

    def _prune_track(self, track_id: int, timestamp: float) -> None:
        cutoff = timestamp - self.window_seconds
        samples = self._observations.get(track_id)
        if samples is None:
            return
        while samples and samples[0].timestamp < cutoff:
            samples.popleft()

    def _discard(self, track_id: int) -> None:
        self._observations.pop(track_id, None)
        self._last_seen.pop(track_id, None)
        self._emitted_texts.pop(track_id, None)
