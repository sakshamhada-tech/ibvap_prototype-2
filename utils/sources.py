"""Helpers for classifying sources and deriving deterministic media time."""

from __future__ import annotations

import math
from urllib.parse import urlparse

NETWORK_SCHEMES = {"rtsp", "rtsps", "http", "https"}


def is_network_source(source) -> bool:
    if not isinstance(source, str):
        return False
    return urlparse(source).scheme.lower() in NETWORK_SCHEMES


def is_file_source(source) -> bool:
    return isinstance(source, str) and not is_network_source(source)


def describe_source(source) -> str:
    """Return a log-safe source description with URL credentials removed."""
    if not is_network_source(source):
        return str(source)
    parsed = urlparse(source)
    hostname = parsed.hostname or "unknown-host"
    host = f"[{hostname}]" if ":" in hostname else hostname
    if parsed.port:
        host += f":{parsed.port}"
    return parsed._replace(netloc=host).geturl()


def normalize_fps(value: float, fallback: float = 25.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    return value if math.isfinite(value) and value > 0 else fallback


def recorded_source_time(position_msec: float, frame_number: int, fps: float) -> float:
    """Use media position where available, otherwise derive it from frame rate."""
    try:
        seconds = float(position_msec) / 1000.0
    except (TypeError, ValueError):
        seconds = -1.0
    if math.isfinite(seconds) and seconds > 0:
        return seconds
    return max(0, frame_number - 1) / normalize_fps(fps)
