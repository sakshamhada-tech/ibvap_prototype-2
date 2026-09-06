"""Environment-backed configuration for the IBVAP prototype.

Every setting can be supplied through an ``IBVAP_*`` environment variable.
Secrets deliberately have no source-controlled default; call
:func:`validate_server_security` before starting the web application.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false, got {raw!r}")


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.getenv(name)
    value = default if raw is None else float(raw)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return value


def _env_csv(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(part.strip() for part in os.getenv(name, default).split(",") if part.strip())


def _resolve_path(raw: str) -> str:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path.resolve())


def _parse_video_source(raw: str):
    raw = raw.strip()
    if re.fullmatch(r"\d+", raw):
        return int(raw)
    parsed = urlparse(raw)
    if parsed.scheme.lower() in {"rtsp", "rtsps", "http", "https"}:
        return raw
    return _resolve_path(raw)


def _parse_fence(raw: str) -> tuple[tuple[int, int], tuple[int, int]]:
    try:
        x1, y1, x2, y2 = (int(value.strip()) for value in raw.split(","))
    except (TypeError, ValueError) as exc:
        raise ValueError("IBVAP_VIRTUAL_FENCE must be x1,y1,x2,y2") from exc
    if (x1, y1) == (x2, y2):
        raise ValueError("IBVAP_VIRTUAL_FENCE endpoints must differ")
    return (x1, y1), (x2, y2)


# Input and model
VIDEO_SOURCE = _parse_video_source(os.getenv("IBVAP_VIDEO_SOURCE", "0"))
_model_setting = os.getenv("IBVAP_YOLO_MODEL", "yolov8n.pt")
_local_model = Path(_model_setting).expanduser()
if not _local_model.is_absolute():
    _local_model = BASE_DIR / _local_model
YOLO_MODEL_PATH = str(_local_model.resolve()) if _local_model.exists() else _model_setting
_default_model_hash = (
    "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
    if _model_setting == "yolov8n.pt"
    else ""
)
YOLO_MODEL_SHA256 = os.getenv("IBVAP_YOLO_MODEL_SHA256", _default_model_hash).lower()
if YOLO_MODEL_SHA256 and not re.fullmatch(r"[0-9a-f]{64}", YOLO_MODEL_SHA256):
    raise ValueError("IBVAP_YOLO_MODEL_SHA256 must be a 64-character hexadecimal digest")
CONFIDENCE_THRESHOLD = _env_float("IBVAP_CONFIDENCE_THRESHOLD", 0.4, minimum=0.0)
if CONFIDENCE_THRESHOLD > 1.0:
    raise ValueError("IBVAP_CONFIDENCE_THRESHOLD must be <= 1")
PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Analytics
VIRTUAL_FENCE_LINE = _parse_fence(os.getenv("IBVAP_VIRTUAL_FENCE", "50,400,1200,400"))
VIRTUAL_FENCE_CONTACT_MARGIN_PX = _env_float(
    "IBVAP_VIRTUAL_FENCE_CONTACT_MARGIN_PX", 2.0, minimum=0.0
)
LOITERING_SECONDS = _env_float("IBVAP_LOITERING_SECONDS", 15.0, minimum=0.1)
LOITERING_MOVEMENT_THRESHOLD_PX = _env_float(
    "IBVAP_LOITERING_MOVEMENT_THRESHOLD_PX", 60.0, minimum=1.0
)
TRACK_STALE_SECONDS = _env_float("IBVAP_TRACK_STALE_SECONDS", 3.0, minimum=0.1)
TRACK_MAX_OBSERVATION_GAP_SECONDS = _env_float(
    "IBVAP_TRACK_MAX_OBSERVATION_GAP_SECONDS", 1.5, minimum=0.01
)
NIGHT_BRIGHTNESS_THRESHOLD = _env_float("IBVAP_NIGHT_BRIGHTNESS_THRESHOLD", 60.0, minimum=0.0)
if NIGHT_BRIGHTNESS_THRESHOLD > 255.0:
    raise ValueError("IBVAP_NIGHT_BRIGHTNESS_THRESHOLD must be <= 255")
NIGHT_BRIGHTNESS_HYSTERESIS = _env_float("IBVAP_NIGHT_BRIGHTNESS_HYSTERESIS", 5.0, minimum=0.0)
NIGHT_BRIGHTNESS_SMOOTHING_ALPHA = _env_float(
    "IBVAP_NIGHT_BRIGHTNESS_SMOOTHING_ALPHA", 0.15, minimum=0.01
)
if NIGHT_BRIGHTNESS_SMOOTHING_ALPHA > 1.0:
    raise ValueError("IBVAP_NIGHT_BRIGHTNESS_SMOOTHING_ALPHA must be <= 1")
NIGHT_BRIGHTNESS_ROI_MARGIN = _env_float("IBVAP_NIGHT_BRIGHTNESS_ROI_MARGIN", 0.05, minimum=0.0)
if NIGHT_BRIGHTNESS_ROI_MARGIN >= 0.5:
    raise ValueError("IBVAP_NIGHT_BRIGHTNESS_ROI_MARGIN must be < 0.5")
# SCRFD source-face review is opt-in because it requires separately licensed
# model weights. It never modifies the primary video frame.
ENABLE_FACE_ENHANCEMENT = _env_bool("IBVAP_ENABLE_FACE_ENHANCEMENT", False)
# Accurate face review uses SCRFD source crops. GFPGAN is a separate,
# generative preview and is deliberately off unless an operator opts in.
ENABLE_GFPGAN_RESTORATION = _env_bool("IBVAP_ENABLE_GFPGAN_RESTORATION", False)
FACE_ENHANCEMENT_INTERVAL_FRAMES = _env_int("IBVAP_FACE_ENHANCEMENT_INTERVAL_FRAMES", 15, minimum=1)
FACE_ENHANCEMENT_MAX_FACES = _env_int("IBVAP_FACE_ENHANCEMENT_MAX_FACES", 4, minimum=1)
FACE_ENHANCEMENT_REFRESH_SECONDS = _env_float(
    "IBVAP_FACE_ENHANCEMENT_REFRESH_SECONDS", 2.0, minimum=0.1
)
FACE_ENHANCEMENT_CACHE_TTL_SECONDS = _env_float(
    "IBVAP_FACE_ENHANCEMENT_CACHE_TTL_SECONDS", 10.0, minimum=1.0
)
FACE_ENHANCEMENT_JPEG_QUALITY = _env_int("IBVAP_FACE_ENHANCEMENT_JPEG_QUALITY", 90, minimum=1)
if FACE_ENHANCEMENT_JPEG_QUALITY > 100:
    raise ValueError("IBVAP_FACE_ENHANCEMENT_JPEG_QUALITY must be <= 100")
SCRFD_CONFIDENCE_THRESHOLD = _env_float("IBVAP_SCRFD_CONFIDENCE_THRESHOLD", 0.6, minimum=0.0)
if SCRFD_CONFIDENCE_THRESHOLD > 1.0:
    raise ValueError("IBVAP_SCRFD_CONFIDENCE_THRESHOLD must be <= 1")
SCRFD_INPUT_SIZE = _env_int("IBVAP_SCRFD_INPUT_SIZE", 640, minimum=128)
if SCRFD_INPUT_SIZE % 32:
    raise ValueError("IBVAP_SCRFD_INPUT_SIZE must be divisible by 32")
SCRFD_MIN_FACE_SIZE_PX = _env_int("IBVAP_SCRFD_MIN_FACE_SIZE_PX", 32, minimum=1)
GFPGAN_MIN_SOURCE_FACE_SIZE_PX = _env_int("IBVAP_GFPGAN_MIN_SOURCE_FACE_SIZE_PX", 96, minimum=32)
if GFPGAN_MIN_SOURCE_FACE_SIZE_PX < SCRFD_MIN_FACE_SIZE_PX:
    raise ValueError("IBVAP_GFPGAN_MIN_SOURCE_FACE_SIZE_PX must be >= IBVAP_SCRFD_MIN_FACE_SIZE_PX")
GFPGAN_MIN_SOURCE_SHARPNESS = _env_float("IBVAP_GFPGAN_MIN_SOURCE_SHARPNESS", 50.0, minimum=0.0)
GFPGAN_BLEND_WEIGHT = _env_float("IBVAP_GFPGAN_BLEND_WEIGHT", 0.10, minimum=0.0)
if GFPGAN_BLEND_WEIGHT > 0.5:
    raise ValueError("IBVAP_GFPGAN_BLEND_WEIGHT must be <= 0.5")
SCRFD_MODEL_PATH = _resolve_path(os.getenv("IBVAP_SCRFD_MODEL_PATH", "models/scrfd_2.5g.onnx"))
GFPGAN_MODEL_PATH = _resolve_path(os.getenv("IBVAP_GFPGAN_MODEL_PATH", "models/GFPGANv1.4.pth"))
GFPGAN_MODEL_SHA256 = os.getenv(
    "IBVAP_GFPGAN_MODEL_SHA256",
    "e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad",
).lower()
if not re.fullmatch(r"[0-9a-f]{64}", GFPGAN_MODEL_SHA256):
    raise ValueError("IBVAP_GFPGAN_MODEL_SHA256 must be a 64-character hexadecimal digest")
FACE_ENHANCEMENT_DEVICE = os.getenv("IBVAP_FACE_ENHANCEMENT_DEVICE", "auto").strip().lower()
if FACE_ENHANCEMENT_DEVICE not in {"auto", "cpu", "mps", "cuda"}:
    raise ValueError("IBVAP_FACE_ENHANCEMENT_DEVICE must be auto, cpu, mps, or cuda")
# ANPR is opt-in: EasyOCR is heavy and may download separate weights.
ENABLE_ANPR = _env_bool("IBVAP_ENABLE_ANPR", False)
ANPR_INTERVAL_FRAMES = _env_int("IBVAP_ANPR_INTERVAL_FRAMES", 15, minimum=1)
ANPR_MIN_OCR_CONFIDENCE = _env_float("IBVAP_ANPR_MIN_OCR_CONFIDENCE", 0.35, minimum=0.0)
if ANPR_MIN_OCR_CONFIDENCE > 1.0:
    raise ValueError("IBVAP_ANPR_MIN_OCR_CONFIDENCE must be <= 1")

# Alerts and output
ALERT_COOLDOWN_SECONDS = _env_float("IBVAP_ALERT_COOLDOWN_SECONDS", 1.0, minimum=0.0)
ALERT_STATE_TTL_SECONDS = _env_float("IBVAP_ALERT_STATE_TTL_SECONDS", 300.0, minimum=1.0)
RECENT_ALERT_DISPLAY_SECONDS = _env_float("IBVAP_RECENT_ALERT_DISPLAY_SECONDS", 8.0, minimum=0.1)
MAX_RECENT_ALERTS = _env_int("IBVAP_MAX_RECENT_ALERTS", 20, minimum=1)
SAVE_ANNOTATED_VIDEO = _env_bool("IBVAP_SAVE_ANNOTATED_VIDEO", True)
ANNOTATED_VIDEO_PATH = _resolve_path(
    os.getenv("IBVAP_ANNOTATED_VIDEO_PATH", "output/annotated_output.mp4")
)
ALERT_LOG_CSV = _resolve_path(os.getenv("IBVAP_ALERT_LOG_CSV", "logs/alerts.csv"))
AUDIT_LOG_JSONL = _resolve_path(os.getenv("IBVAP_AUDIT_LOG_JSONL", "logs/audit.jsonl"))
ALERT_LOG_MAX_BYTES = _env_int("IBVAP_ALERT_LOG_MAX_BYTES", 10 * 1024 * 1024, minimum=1024)
AUDIT_LOG_MAX_BYTES = _env_int("IBVAP_AUDIT_LOG_MAX_BYTES", 10 * 1024 * 1024, minimum=1024)
LOG_BACKUP_COUNT = _env_int("IBVAP_LOG_BACKUP_COUNT", 5, minimum=1)
SHOW_LIVE_WINDOW = _env_bool("IBVAP_SHOW_LIVE_WINDOW", True)

# Web service and camera lifecycle
SERVER_HOST = os.getenv("IBVAP_SERVER_HOST", "0.0.0.0")
SERVER_PORT = _env_int("IBVAP_SERVER_PORT", 8000, minimum=1)
if SERVER_PORT > 65535:
    raise ValueError("IBVAP_SERVER_PORT must be <= 65535")
CAMERA_RECONNECT_SECONDS = _env_float("IBVAP_CAMERA_RECONNECT_SECONDS", 2.0, minimum=0.1)
CAMERA_OPEN_TIMEOUT_MILLISECONDS = _env_int(
    "IBVAP_CAMERA_OPEN_TIMEOUT_MILLISECONDS", 5000, minimum=100
)
CAMERA_READ_TIMEOUT_MILLISECONDS = _env_int(
    "IBVAP_CAMERA_READ_TIMEOUT_MILLISECONDS", 5000, minimum=100
)
FILE_PLAYBACK_REALTIME = _env_bool("IBVAP_FILE_PLAYBACK_REALTIME", True)
STATS_PUSH_INTERVAL_SECONDS = _env_float("IBVAP_STATS_PUSH_INTERVAL_SECONDS", 0.4, minimum=0.05)
FRAME_STALE_AFTER_SECONDS = _env_float("IBVAP_FRAME_STALE_AFTER_SECONDS", 10.0, minimum=0.1)
MAX_STREAM_CLIENTS = _env_int("IBVAP_MAX_STREAM_CLIENTS", 8, minimum=1)
MAX_WEBSOCKET_CLIENTS = _env_int("IBVAP_MAX_WEBSOCKET_CLIENTS", 16, minimum=1)
WEBSOCKET_ATTEMPTS_PER_MINUTE = _env_int("IBVAP_WEBSOCKET_ATTEMPTS_PER_MINUTE", 60, minimum=1)
ALERT_SUBSCRIBER_QUEUE_SIZE = _env_int("IBVAP_ALERT_SUBSCRIBER_QUEUE_SIZE", 100, minimum=1)
ALERT_HISTORY_SIZE = _env_int("IBVAP_ALERT_HISTORY_SIZE", 100, minimum=1)

# Dashboard authentication. Values are read only from the environment.
DASHBOARD_USERNAME = os.getenv("IBVAP_DASHBOARD_USERNAME", "operator")
DASHBOARD_PASSWORD = os.getenv("IBVAP_DASHBOARD_PASSWORD", "")
SESSION_SECRET = os.getenv("IBVAP_SESSION_SECRET", "")
SESSION_TTL_SECONDS = _env_int("IBVAP_SESSION_TTL_SECONDS", 8 * 60 * 60, minimum=60)
SESSION_COOKIE_NAME = os.getenv("IBVAP_SESSION_COOKIE_NAME", "ibvap_session")
LOGIN_ATTEMPTS_PER_MINUTE = _env_int("IBVAP_LOGIN_ATTEMPTS_PER_MINUTE", 5, minimum=1)
LOGIN_RATE_LIMIT_MAX_CLIENTS = _env_int("IBVAP_LOGIN_RATE_LIMIT_MAX_CLIENTS", 10_000, minimum=100)
ALLOWED_HOSTS = _env_csv("IBVAP_ALLOWED_HOSTS", "localhost,127.0.0.1")
ALLOWED_ORIGINS = _env_csv("IBVAP_ALLOWED_ORIGINS")
REQUIRE_HTTPS = _env_bool("IBVAP_REQUIRE_HTTPS", False)
HTTPS_BEHIND_PROXY = _env_bool("IBVAP_HTTPS_BEHIND_PROXY", False)
COOKIE_SECURE = _env_bool("IBVAP_COOKIE_SECURE", False)
TLS_CERTFILE = os.getenv("IBVAP_TLS_CERTFILE") or None
TLS_KEYFILE = os.getenv("IBVAP_TLS_KEYFILE") or None
FORWARDED_ALLOW_IPS = os.getenv("IBVAP_FORWARDED_ALLOW_IPS", "127.0.0.1")


def validate_server_security() -> None:
    """Fail closed when the dashboard is started without viable credentials."""
    errors = []
    if len(DASHBOARD_USERNAME) < 3:
        errors.append("IBVAP_DASHBOARD_USERNAME must contain at least 3 characters")
    if len(DASHBOARD_PASSWORD) < 12:
        errors.append("IBVAP_DASHBOARD_PASSWORD must contain at least 12 characters")
    if len(SESSION_SECRET) < 32:
        errors.append("IBVAP_SESSION_SECRET must contain at least 32 characters")
    if bool(TLS_CERTFILE) != bool(TLS_KEYFILE):
        errors.append("IBVAP_TLS_CERTFILE and IBVAP_TLS_KEYFILE must be set together")
    if REQUIRE_HTTPS and not (TLS_CERTFILE and TLS_KEYFILE) and not HTTPS_BEHIND_PROXY:
        errors.append("HTTPS requires TLS files or IBVAP_HTTPS_BEHIND_PROXY=true")
    if HTTPS_BEHIND_PROXY and FORWARDED_ALLOW_IPS in {"", "*"}:
        errors.append("IBVAP_FORWARDED_ALLOW_IPS must name trusted proxy addresses")
    if not ALLOWED_HOSTS:
        errors.append("IBVAP_ALLOWED_HOSTS must contain at least one trusted hostname")
    if errors:
        raise RuntimeError("Unsafe dashboard configuration:\n- " + "\n- ".join(errors))
