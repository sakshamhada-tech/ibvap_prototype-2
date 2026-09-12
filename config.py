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


def _parse_optional_polygon(raw: str) -> tuple[tuple[int, int], ...] | None:
    """Parse ``x,y;x,y;...`` or the explicit whole-frame value ``anywhere``."""
    if raw.strip().lower() == "anywhere":
        return None
    try:
        points = tuple(
            tuple(int(value.strip()) for value in pair.split(",")) for pair in raw.split(";")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("IBVAP_LOITERING_ZONE must be 'anywhere' or x,y;x,y;x,y") from exc
    if len(points) < 3 or any(len(point) != 2 for point in points):
        raise ValueError("IBVAP_LOITERING_ZONE polygon must contain at least three x,y points")
    if len(set(points)) < 3:
        raise ValueError("IBVAP_LOITERING_ZONE polygon must contain three distinct points")
    return points


def _validate_sha256(name: str, value: str, *, required: bool = False) -> str:
    normalized = value.strip().lower()
    if required and not normalized:
        raise ValueError(f"{name} must be configured when its feature is enabled")
    if normalized and not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{name} must be a 64-character hexadecimal digest")
    return normalized


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
YOLO_TRACKER = os.getenv("IBVAP_YOLO_TRACKER", "bytetrack.yaml").strip()
if YOLO_TRACKER != "bytetrack.yaml":
    raise ValueError("IBVAP_YOLO_TRACKER currently supports only bytetrack.yaml")
# CPU avoids native Metal failures on macOS. MPS remains an explicit operator choice.
YOLO_DEVICE = os.getenv("IBVAP_YOLO_DEVICE", "cpu").strip().lower()
if YOLO_DEVICE not in {"cpu", "mps", "cuda", "cuda:0"}:
    raise ValueError("IBVAP_YOLO_DEVICE must be cpu, mps, cuda, or cuda:0")
YOLO_INPUT_SIZE = _env_int("IBVAP_YOLO_INPUT_SIZE", 640, minimum=32)
if YOLO_INPUT_SIZE > 4096 or YOLO_INPUT_SIZE % 32:
    raise ValueError("IBVAP_YOLO_INPUT_SIZE must be divisible by 32 and <= 4096")
CONFIDENCE_THRESHOLD = _env_float("IBVAP_CONFIDENCE_THRESHOLD", 0.4, minimum=0.0)
if CONFIDENCE_THRESHOLD > 1.0:
    raise ValueError("IBVAP_CONFIDENCE_THRESHOLD must be <= 1")
ENABLE_TILED_INFERENCE = _env_bool("IBVAP_ENABLE_TILED_INFERENCE", False)
TILE_SIZE = _env_int("IBVAP_TILE_SIZE", 640, minimum=128)
TILE_OVERLAP = _env_float("IBVAP_TILE_OVERLAP", 0.20, minimum=0.0)
if TILE_OVERLAP >= 0.9:
    raise ValueError("IBVAP_TILE_OVERLAP must be < 0.9")
TILED_INFERENCE_INTERVAL_FRAMES = _env_int("IBVAP_TILED_INFERENCE_INTERVAL_FRAMES", 30, minimum=1)
TILED_INFERENCE_MAX_TILES_PER_CYCLE = _env_int(
    "IBVAP_TILED_INFERENCE_MAX_TILES_PER_CYCLE", 4, minimum=1
)
if TILED_INFERENCE_MAX_TILES_PER_CYCLE > 64:
    raise ValueError("IBVAP_TILED_INFERENCE_MAX_TILES_PER_CYCLE must be <= 64")
TILED_INFERENCE_CONFIDENCE_THRESHOLD = _env_float(
    "IBVAP_TILED_INFERENCE_CONFIDENCE_THRESHOLD", 0.5, minimum=0.0
)
if TILED_INFERENCE_CONFIDENCE_THRESHOLD > 1.0:
    raise ValueError("IBVAP_TILED_INFERENCE_CONFIDENCE_THRESHOLD must be <= 1")
TILED_INFERENCE_NMS_IOU = _env_float("IBVAP_TILED_INFERENCE_NMS_IOU", 0.5, minimum=0.0)
if TILED_INFERENCE_NMS_IOU > 1.0:
    raise ValueError("IBVAP_TILED_INFERENCE_NMS_IOU must be <= 1")
TILED_INFERENCE_DEVICE = os.getenv("IBVAP_TILED_INFERENCE_DEVICE", "cpu").strip().lower()
if TILED_INFERENCE_DEVICE not in {"cpu", "mps", "cuda", "cuda:0"}:
    raise ValueError("IBVAP_TILED_INFERENCE_DEVICE must be cpu, mps, cuda, or cuda:0")
TILED_INFERENCE_QUEUE_POLL_SECONDS = _env_float(
    "IBVAP_TILED_INFERENCE_QUEUE_POLL_SECONDS", 0.2, minimum=0.01
)
if TILED_INFERENCE_QUEUE_POLL_SECONDS > 10.0:
    raise ValueError("IBVAP_TILED_INFERENCE_QUEUE_POLL_SECONDS must be <= 10")
TILED_INFERENCE_SHUTDOWN_TIMEOUT_SECONDS = _env_float(
    "IBVAP_TILED_INFERENCE_SHUTDOWN_TIMEOUT_SECONDS", 10.0, minimum=0.1
)
if TILED_INFERENCE_SHUTDOWN_TIMEOUT_SECONDS > 120.0:
    raise ValueError("IBVAP_TILED_INFERENCE_SHUTDOWN_TIMEOUT_SECONDS must be <= 120")

# Native focus reuses the reviewed object model on untouched source-frame crops.
# The demo enables its bounded CPU worker by default; operators can explicitly disable it.
ENABLE_NATIVE_FOCUS = _env_bool("IBVAP_ENABLE_NATIVE_FOCUS", True)
NATIVE_FOCUS_INTERVAL_FRAMES = _env_int("IBVAP_NATIVE_FOCUS_INTERVAL_FRAMES", 5, minimum=1)
if NATIVE_FOCUS_INTERVAL_FRAMES > 10_000:
    raise ValueError("IBVAP_NATIVE_FOCUS_INTERVAL_FRAMES must be <= 10000")
NATIVE_FOCUS_MODEL_INPUT_SIZE = _env_int("IBVAP_NATIVE_FOCUS_MODEL_INPUT_SIZE", 640, minimum=32)
NATIVE_FOCUS_TRACK_INPUT_SIZE = _env_int("IBVAP_NATIVE_FOCUS_TRACK_INPUT_SIZE", 1280, minimum=32)
for _focus_size_name, _focus_size in (
    ("IBVAP_NATIVE_FOCUS_MODEL_INPUT_SIZE", NATIVE_FOCUS_MODEL_INPUT_SIZE),
    ("IBVAP_NATIVE_FOCUS_TRACK_INPUT_SIZE", NATIVE_FOCUS_TRACK_INPUT_SIZE),
):
    if _focus_size > 4096 or _focus_size % 32:
        raise ValueError(f"{_focus_size_name} must be divisible by 32 and <= 4096")
if ENABLE_NATIVE_FOCUS and NATIVE_FOCUS_TRACK_INPUT_SIZE <= YOLO_INPUT_SIZE:
    raise ValueError(
        "IBVAP_NATIVE_FOCUS_TRACK_INPUT_SIZE must exceed IBVAP_YOLO_INPUT_SIZE when enabled"
    )
NATIVE_FOCUS_MAX_REGIONS = _env_int("IBVAP_NATIVE_FOCUS_MAX_REGIONS", 2, minimum=1)
if NATIVE_FOCUS_MAX_REGIONS > 16:
    raise ValueError("IBVAP_NATIVE_FOCUS_MAX_REGIONS must be <= 16")
NATIVE_FOCUS_PADDING_RATIO = _env_float("IBVAP_NATIVE_FOCUS_PADDING_RATIO", 0.5, minimum=0.0)
if NATIVE_FOCUS_PADDING_RATIO > 4.0:
    raise ValueError("IBVAP_NATIVE_FOCUS_PADDING_RATIO must be <= 4")
NATIVE_FOCUS_MIN_CROP_SIZE_PX = _env_int("IBVAP_NATIVE_FOCUS_MIN_CROP_SIZE_PX", 192, minimum=32)
if NATIVE_FOCUS_MIN_CROP_SIZE_PX > 8192:
    raise ValueError("IBVAP_NATIVE_FOCUS_MIN_CROP_SIZE_PX must be <= 8192")
NATIVE_FOCUS_CONF_THRESHOLD = _env_float("IBVAP_NATIVE_FOCUS_CONF_THRESHOLD", 0.25, minimum=0.0)
if NATIVE_FOCUS_CONF_THRESHOLD > 1.0:
    raise ValueError("IBVAP_NATIVE_FOCUS_CONF_THRESHOLD must be <= 1")
NATIVE_FOCUS_PROPOSAL_IOU = _env_float("IBVAP_NATIVE_FOCUS_PROPOSAL_IOU", 0.05, minimum=0.0)
if NATIVE_FOCUS_PROPOSAL_IOU > 1.0:
    raise ValueError("IBVAP_NATIVE_FOCUS_PROPOSAL_IOU must be <= 1")
NATIVE_FOCUS_CONFIRMATION_FRAMES = _env_int("IBVAP_NATIVE_FOCUS_CONFIRMATION_FRAMES", 2, minimum=2)
if NATIVE_FOCUS_CONFIRMATION_FRAMES > 30:
    raise ValueError("IBVAP_NATIVE_FOCUS_CONFIRMATION_FRAMES must be <= 30")
NATIVE_FOCUS_CONFIRMATION_IOU = _env_float("IBVAP_NATIVE_FOCUS_CONFIRMATION_IOU", 0.15, minimum=0.0)
if NATIVE_FOCUS_CONFIRMATION_IOU > 1.0:
    raise ValueError("IBVAP_NATIVE_FOCUS_CONFIRMATION_IOU must be <= 1")
NATIVE_FOCUS_MAX_GAP_FRAMES = _env_int("IBVAP_NATIVE_FOCUS_MAX_GAP_FRAMES", 30, minimum=1)
if NATIVE_FOCUS_MAX_GAP_FRAMES > 10_000:
    raise ValueError("IBVAP_NATIVE_FOCUS_MAX_GAP_FRAMES must be <= 10000")
NATIVE_FOCUS_TRACK_BOOST_FRAMES = _env_int("IBVAP_NATIVE_FOCUS_TRACK_BOOST_FRAMES", 30, minimum=1)
if NATIVE_FOCUS_TRACK_BOOST_FRAMES > 300:
    raise ValueError("IBVAP_NATIVE_FOCUS_TRACK_BOOST_FRAMES must be <= 300")
NATIVE_FOCUS_QUEUE_POLL_SECONDS = _env_float(
    "IBVAP_NATIVE_FOCUS_QUEUE_POLL_SECONDS", 0.2, minimum=0.01
)
if NATIVE_FOCUS_QUEUE_POLL_SECONDS > 10.0:
    raise ValueError("IBVAP_NATIVE_FOCUS_QUEUE_POLL_SECONDS must be <= 10")
NATIVE_FOCUS_SHUTDOWN_TIMEOUT_SECONDS = _env_float(
    "IBVAP_NATIVE_FOCUS_SHUTDOWN_TIMEOUT_SECONDS", 10.0, minimum=0.1
)
if NATIVE_FOCUS_SHUTDOWN_TIMEOUT_SECONDS > 120.0:
    raise ValueError("IBVAP_NATIVE_FOCUS_SHUTDOWN_TIMEOUT_SECONDS must be <= 120")
NATIVE_FOCUS_RESULT_MAX_AGE_FRAMES = _env_int(
    "IBVAP_NATIVE_FOCUS_RESULT_MAX_AGE_FRAMES", 60, minimum=1
)
if NATIVE_FOCUS_RESULT_MAX_AGE_FRAMES > 10_000:
    raise ValueError("IBVAP_NATIVE_FOCUS_RESULT_MAX_AGE_FRAMES must be <= 10000")
NATIVE_FOCUS_ACQUISITION_IOU = _env_float("IBVAP_NATIVE_FOCUS_ACQUISITION_IOU", 0.10, minimum=0.0)
if NATIVE_FOCUS_ACQUISITION_IOU > 1.0:
    raise ValueError("IBVAP_NATIVE_FOCUS_ACQUISITION_IOU must be <= 1")
NATIVE_FOCUS_JPEG_QUALITY = _env_int("IBVAP_NATIVE_FOCUS_JPEG_QUALITY", 90, minimum=1)
if NATIVE_FOCUS_JPEG_QUALITY > 100:
    raise ValueError("IBVAP_NATIVE_FOCUS_JPEG_QUALITY must be <= 100")
NATIVE_FOCUS_DEVICE = os.getenv("IBVAP_NATIVE_FOCUS_DEVICE", "cpu").strip().lower()
if NATIVE_FOCUS_DEVICE not in {"auto", "cpu", "mps", "cuda", "cuda:0"}:
    raise ValueError("IBVAP_NATIVE_FOCUS_DEVICE must be auto, cpu, mps, cuda, or cuda:0")

# Controlled evaluation bounds and scoring. These are dataset/reporting settings,
# not claims about operational range or universal model performance.
EVALUATION_IOU_THRESHOLD = _env_float("IBVAP_EVALUATION_IOU_THRESHOLD", 0.50, minimum=0.0)
if EVALUATION_IOU_THRESHOLD > 1.0:
    raise ValueError("IBVAP_EVALUATION_IOU_THRESHOLD must be <= 1")
EVALUATION_LOW_FPS_PERCENTILE = _env_float("IBVAP_EVALUATION_LOW_FPS_PERCENTILE", 10.0, minimum=0.1)
if EVALUATION_LOW_FPS_PERCENTILE > 50.0:
    raise ValueError("IBVAP_EVALUATION_LOW_FPS_PERCENTILE must be <= 50")
EVALUATION_MAX_FRAMES = _env_int("IBVAP_EVALUATION_MAX_FRAMES", 10_000, minimum=1)
if EVALUATION_MAX_FRAMES > 1_000_000:
    raise ValueError("IBVAP_EVALUATION_MAX_FRAMES must be <= 1000000")
EVALUATION_RUNTIME_WARMUP_FRAMES = _env_int("IBVAP_EVALUATION_RUNTIME_WARMUP_FRAMES", 5, minimum=0)
if EVALUATION_RUNTIME_WARMUP_FRAMES >= EVALUATION_MAX_FRAMES:
    raise ValueError("IBVAP_EVALUATION_RUNTIME_WARMUP_FRAMES must be less than max frames")
EVALUATION_MAX_OBJECTS_PER_FRAME = _env_int(
    "IBVAP_EVALUATION_MAX_OBJECTS_PER_FRAME", 1_000, minimum=1
)
if EVALUATION_MAX_OBJECTS_PER_FRAME > 10_000:
    raise ValueError("IBVAP_EVALUATION_MAX_OBJECTS_PER_FRAME must be <= 10000")
EVALUATION_MAX_ALERTS = _env_int("IBVAP_EVALUATION_MAX_ALERTS", 100_000, minimum=1)
if EVALUATION_MAX_ALERTS > 1_000_000:
    raise ValueError("IBVAP_EVALUATION_MAX_ALERTS must be <= 1000000")
EVALUATION_MAX_JSON_BYTES = _env_int(
    "IBVAP_EVALUATION_MAX_JSON_BYTES", 100 * 1024 * 1024, minimum=1024
)
if EVALUATION_MAX_JSON_BYTES > 1024 * 1024 * 1024:
    raise ValueError("IBVAP_EVALUATION_MAX_JSON_BYTES must be <= 1073741824")
EVALUATION_HARDWARE_LABEL = os.getenv("IBVAP_EVALUATION_HARDWARE_LABEL", "unrecorded").strip()
EVALUATION_BUILD_LABEL = os.getenv("IBVAP_EVALUATION_BUILD_LABEL", "unrecorded").strip()
for _evaluation_label_name, _evaluation_label in (
    ("IBVAP_EVALUATION_HARDWARE_LABEL", EVALUATION_HARDWARE_LABEL),
    ("IBVAP_EVALUATION_BUILD_LABEL", EVALUATION_BUILD_LABEL),
):
    if not _evaluation_label or len(_evaluation_label) > 128:
        raise ValueError(f"{_evaluation_label_name} must contain 1..128 characters")
    if any(not _character.isprintable() for _character in _evaluation_label):
        raise ValueError(f"{_evaluation_label_name} must contain only printable characters")

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
LOITERING_ZONE = _parse_optional_polygon(os.getenv("IBVAP_LOITERING_ZONE", "anywhere"))
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

# Classical-CV context signals are opt-in and retain only short-lived track state.
ENABLE_GROUP_APPROACH = _env_bool("IBVAP_ENABLE_GROUP_APPROACH", False)
GROUP_MIN_SIZE = _env_int("IBVAP_GROUP_MIN_SIZE", 3, minimum=2)
GROUP_PROXIMITY_PX = _env_float("IBVAP_GROUP_PROXIMITY_PX", 160.0, minimum=1.0)
GROUP_APPROACH_WINDOW_SECONDS = _env_float("IBVAP_GROUP_APPROACH_WINDOW_SECONDS", 3.0, minimum=0.1)
GROUP_MIN_SPEED_PX_PER_SECOND = _env_float("IBVAP_GROUP_MIN_SPEED_PX_PER_SECOND", 5.0, minimum=0.0)
GROUP_HEADING_SIMILARITY = _env_float("IBVAP_GROUP_HEADING_SIMILARITY", 0.75, minimum=0.0)
if GROUP_HEADING_SIMILARITY > 1.0:
    raise ValueError("IBVAP_GROUP_HEADING_SIMILARITY must be <= 1")
GROUP_MIN_FENCE_PROGRESS_PX = _env_float("IBVAP_GROUP_MIN_FENCE_PROGRESS_PX", 12.0, minimum=0.0)

ENABLE_MOTION_SIGNATURE = _env_bool("IBVAP_ENABLE_MOTION_SIGNATURE", False)
MOTION_SIGNATURE_WINDOW_SECONDS = _env_float(
    "IBVAP_MOTION_SIGNATURE_WINDOW_SECONDS", 3.0, minimum=0.1
)
if ENABLE_MOTION_SIGNATURE and MOTION_SIGNATURE_WINDOW_SECONDS > LOITERING_SECONDS:
    raise ValueError(
        "IBVAP_MOTION_SIGNATURE_WINDOW_SECONDS must not exceed IBVAP_LOITERING_SECONDS "
        "because motion scoring reuses loitering trajectory history"
    )
MOTION_SIGNATURE_MIN_SAMPLES = _env_int("IBVAP_MOTION_SIGNATURE_MIN_SAMPLES", 8, minimum=3)
MOTION_SIGNATURE_STATIC_PATH_PX = _env_float(
    "IBVAP_MOTION_SIGNATURE_STATIC_PATH_PX", 3.0, minimum=0.0
)
MOTION_SIGNATURE_VEGETATION_MAX_NET_PX = _env_float(
    "IBVAP_MOTION_SIGNATURE_VEGETATION_MAX_NET_PX", 8.0, minimum=0.0
)
MOTION_SIGNATURE_HIGH_FREQUENCY_RATIO = _env_float(
    "IBVAP_MOTION_SIGNATURE_HIGH_FREQUENCY_RATIO", 0.5, minimum=0.0
)
if MOTION_SIGNATURE_HIGH_FREQUENCY_RATIO > 1.0:
    raise ValueError("IBVAP_MOTION_SIGNATURE_HIGH_FREQUENCY_RATIO must be <= 1")
MOTION_SIGNATURE_DISPLACEMENT_SCALE_PX = _env_float(
    "IBVAP_MOTION_SIGNATURE_DISPLACEMENT_SCALE_PX", 40.0, minimum=0.1
)
MOTION_SIGNATURE_OSCILLATION_SCALE_PX = _env_float(
    "IBVAP_MOTION_SIGNATURE_OSCILLATION_SCALE_PX", 4.0, minimum=0.1
)
MOTION_SIGNATURE_STATIC_SCORE = _env_float("IBVAP_MOTION_SIGNATURE_STATIC_SCORE", 0.0, minimum=0.0)
MOTION_SIGNATURE_VEGETATION_SCORE = _env_float(
    "IBVAP_MOTION_SIGNATURE_VEGETATION_SCORE", 0.05, minimum=0.0
)
MOTION_SIGNATURE_GAIT_SCORE = _env_float("IBVAP_MOTION_SIGNATURE_GAIT_SCORE", 0.8, minimum=0.0)
MOTION_SIGNATURE_RIGID_SCORE = _env_float("IBVAP_MOTION_SIGNATURE_RIGID_SCORE", 0.4, minimum=0.0)
MOTION_SIGNATURE_AMBIGUOUS_SCORE = _env_float(
    "IBVAP_MOTION_SIGNATURE_AMBIGUOUS_SCORE", 0.2, minimum=0.0
)
for _motion_score_name, _motion_score in (
    ("IBVAP_MOTION_SIGNATURE_STATIC_SCORE", MOTION_SIGNATURE_STATIC_SCORE),
    ("IBVAP_MOTION_SIGNATURE_VEGETATION_SCORE", MOTION_SIGNATURE_VEGETATION_SCORE),
    ("IBVAP_MOTION_SIGNATURE_GAIT_SCORE", MOTION_SIGNATURE_GAIT_SCORE),
    ("IBVAP_MOTION_SIGNATURE_RIGID_SCORE", MOTION_SIGNATURE_RIGID_SCORE),
    ("IBVAP_MOTION_SIGNATURE_AMBIGUOUS_SCORE", MOTION_SIGNATURE_AMBIGUOUS_SCORE),
):
    if _motion_score > 1.0:
        raise ValueError(f"{_motion_score_name} must be <= 1")

ENABLE_CONTEXTUAL_RISK = _env_bool("IBVAP_ENABLE_CONTEXTUAL_RISK", False)
RISK_ALERT_THRESHOLD = _env_float("IBVAP_RISK_ALERT_THRESHOLD", 0.65, minimum=0.0)
if RISK_ALERT_THRESHOLD > 1.0:
    raise ValueError("IBVAP_RISK_ALERT_THRESHOLD must be <= 1")
RISK_FENCE_PROXIMITY_PX = _env_float("IBVAP_RISK_FENCE_PROXIMITY_PX", 180.0, minimum=1.0)
RISK_WEIGHT_FENCE_CONTACT = _env_float("IBVAP_RISK_WEIGHT_FENCE_CONTACT", 0.35, minimum=0.0)
RISK_WEIGHT_FENCE_PROXIMITY = _env_float("IBVAP_RISK_WEIGHT_FENCE_PROXIMITY", 0.15, minimum=0.0)
RISK_WEIGHT_LOITERING = _env_float("IBVAP_RISK_WEIGHT_LOITERING", 0.20, minimum=0.0)
RISK_WEIGHT_MOTION_SIGNATURE = _env_float("IBVAP_RISK_WEIGHT_MOTION_SIGNATURE", 0.10, minimum=0.0)
RISK_WEIGHT_LOW_POSTURE = _env_float("IBVAP_RISK_WEIGHT_LOW_POSTURE", 0.20, minimum=0.0)
RISK_WEIGHT_NIGHT = _env_float("IBVAP_RISK_WEIGHT_NIGHT", 0.10, minimum=0.0)
RISK_WEIGHT_FENCE_APPROACH = _env_float("IBVAP_RISK_WEIGHT_FENCE_APPROACH", 0.20, minimum=0.0)
RISK_WEIGHT_GROUP_MEMBERSHIP = _env_float("IBVAP_RISK_WEIGHT_GROUP_MEMBERSHIP", 0.20, minimum=0.0)
for _risk_weight_name, _risk_weight in (
    ("IBVAP_RISK_WEIGHT_FENCE_CONTACT", RISK_WEIGHT_FENCE_CONTACT),
    ("IBVAP_RISK_WEIGHT_FENCE_PROXIMITY", RISK_WEIGHT_FENCE_PROXIMITY),
    ("IBVAP_RISK_WEIGHT_LOITERING", RISK_WEIGHT_LOITERING),
    ("IBVAP_RISK_WEIGHT_MOTION_SIGNATURE", RISK_WEIGHT_MOTION_SIGNATURE),
    ("IBVAP_RISK_WEIGHT_LOW_POSTURE", RISK_WEIGHT_LOW_POSTURE),
    ("IBVAP_RISK_WEIGHT_NIGHT", RISK_WEIGHT_NIGHT),
    ("IBVAP_RISK_WEIGHT_FENCE_APPROACH", RISK_WEIGHT_FENCE_APPROACH),
    ("IBVAP_RISK_WEIGHT_GROUP_MEMBERSHIP", RISK_WEIGHT_GROUP_MEMBERSHIP),
):
    if _risk_weight > 1.0:
        raise ValueError(f"{_risk_weight_name} must be <= 1")

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
# CPU is the safe default: ONNX Runtime's CoreML provider can terminate a macOS
# process inside Metal, which Python exception isolation cannot recover from.
SCRFD_EXECUTION_PROVIDER = os.getenv("IBVAP_SCRFD_EXECUTION_PROVIDER", "cpu").strip().lower()
if SCRFD_EXECUTION_PROVIDER not in {"auto", "cpu", "cuda", "coreml"}:
    raise ValueError("IBVAP_SCRFD_EXECUTION_PROVIDER must be auto, cpu, cuda, or coreml")
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
FACE_ENHANCEMENT_DEVICE = os.getenv("IBVAP_FACE_ENHANCEMENT_DEVICE", "cpu").strip().lower()
if FACE_ENHANCEMENT_DEVICE not in {"auto", "cpu", "mps", "cuda"}:
    raise ValueError("IBVAP_FACE_ENHANCEMENT_DEVICE must be auto, cpu, mps, or cuda")

# Pose analysis is isolated and disabled until an operator provides reviewed weights.
ENABLE_POSTURE_ANALYSIS = _env_bool("IBVAP_ENABLE_POSTURE_ANALYSIS", False)
POSTURE_MODEL_PATH = _resolve_path(os.getenv("IBVAP_POSTURE_MODEL_PATH", "models/yolov8n-pose.pt"))
POSTURE_MODEL_SHA256 = _validate_sha256(
    "IBVAP_POSTURE_MODEL_SHA256", os.getenv("IBVAP_POSTURE_MODEL_SHA256", "")
)
POSTURE_CONFIDENCE_THRESHOLD = _env_float("IBVAP_POSTURE_CONFIDENCE_THRESHOLD", 0.60, minimum=0.0)
if POSTURE_CONFIDENCE_THRESHOLD > 1.0:
    raise ValueError("IBVAP_POSTURE_CONFIDENCE_THRESHOLD must be <= 1")
POSTURE_INTERVAL_FRAMES = _env_int("IBVAP_POSTURE_INTERVAL_FRAMES", 15, minimum=1)
POSTURE_MAX_PEOPLE = _env_int("IBVAP_POSTURE_MAX_PEOPLE", 4, minimum=1)
POSTURE_MIN_PERSON_SIZE_PX = _env_int("IBVAP_POSTURE_MIN_PERSON_SIZE_PX", 96, minimum=1)
POSTURE_LOW_ASPECT_RATIO = _env_float("IBVAP_POSTURE_LOW_ASPECT_RATIO", 0.9, minimum=0.1)
POSTURE_MAX_VERTICAL_SPREAD_RATIO = _env_float(
    "IBVAP_POSTURE_MAX_VERTICAL_SPREAD_RATIO", 0.55, minimum=0.0
)
if POSTURE_MAX_VERTICAL_SPREAD_RATIO > 1.0:
    raise ValueError("IBVAP_POSTURE_MAX_VERTICAL_SPREAD_RATIO must be <= 1")
POSTURE_MIN_VISIBLE_KEYPOINTS = _env_int("IBVAP_POSTURE_MIN_VISIBLE_KEYPOINTS", 3, minimum=1)
POSTURE_MIN_KEYPOINT_CONFIDENCE = _env_float(
    "IBVAP_POSTURE_MIN_KEYPOINT_CONFIDENCE", 0.40, minimum=0.0
)
if POSTURE_MIN_KEYPOINT_CONFIDENCE > 1.0:
    raise ValueError("IBVAP_POSTURE_MIN_KEYPOINT_CONFIDENCE must be <= 1")
POSTURE_SUSTAIN_SECONDS = _env_float("IBVAP_POSTURE_SUSTAIN_SECONDS", 1.5, minimum=0.1)
POSTURE_STATE_TTL_SECONDS = _env_float("IBVAP_POSTURE_STATE_TTL_SECONDS", 3.0, minimum=0.1)
POSTURE_RESULT_MAX_AGE_FRAMES = _env_int("IBVAP_POSTURE_RESULT_MAX_AGE_FRAMES", 45, minimum=1)
POSTURE_DEVICE = os.getenv("IBVAP_POSTURE_DEVICE", "cpu").strip().lower()
if POSTURE_DEVICE not in {"auto", "cpu", "mps", "cuda"}:
    raise ValueError("IBVAP_POSTURE_DEVICE must be auto, cpu, mps, or cuda")

# ANPR is opt-in: it requires separately reviewed Indian plate-detector weights
# plus EasyOCR and its separately governed recognition weights.
ENABLE_ANPR = _env_bool("IBVAP_ENABLE_ANPR", False)
ANPR_DEBUG = _env_bool("IBVAP_ANPR_DEBUG", False)
ANPR_INTERVAL_FRAMES = _env_int("IBVAP_ANPR_INTERVAL_FRAMES", 10, minimum=1)
_anpr_model_setting = os.getenv("IBVAP_ANPR_PLATE_MODEL_PATH", "models/indian_plate_yolov8n.pt")
ANPR_PLATE_MODEL_PATH = _resolve_path(_anpr_model_setting)
_default_anpr_model_hash = (
    "897ad71e4c94cb69aabdb6ba8fbd4bbecfad239fde6bc386ae04c650411787b4"
    if _anpr_model_setting == "models/indian_plate_yolov8n.pt"
    else ""
)
ANPR_PLATE_MODEL_SHA256 = os.getenv(
    "IBVAP_ANPR_PLATE_MODEL_SHA256", _default_anpr_model_hash
).lower()
if ANPR_PLATE_MODEL_SHA256 and not re.fullmatch(r"[0-9a-f]{64}", ANPR_PLATE_MODEL_SHA256):
    raise ValueError("IBVAP_ANPR_PLATE_MODEL_SHA256 must be a 64-character hexadecimal digest")
ANPR_PLATE_DETECTION_CONFIDENCE = _env_float(
    "IBVAP_ANPR_PLATE_DETECTION_CONFIDENCE", 0.25, minimum=0.0
)
if ANPR_PLATE_DETECTION_CONFIDENCE > 1.0:
    raise ValueError("IBVAP_ANPR_PLATE_DETECTION_CONFIDENCE must be <= 1")
ANPR_PLATE_INPUT_SIZE = _env_int("IBVAP_ANPR_PLATE_INPUT_SIZE", 640, minimum=128)
if ANPR_PLATE_INPUT_SIZE % 32:
    raise ValueError("IBVAP_ANPR_PLATE_INPUT_SIZE must be divisible by 32")
ANPR_MAX_PLATE_CANDIDATES = _env_int("IBVAP_ANPR_MAX_PLATE_CANDIDATES", 3, minimum=1)
ANPR_MIN_PLATE_WIDTH_PX = _env_int("IBVAP_ANPR_MIN_PLATE_WIDTH_PX", 80, minimum=1)
ANPR_MIN_PLATE_SHARPNESS = _env_float("IBVAP_ANPR_MIN_PLATE_SHARPNESS", 20.0, minimum=0.0)
ANPR_OCR_TARGET_HEIGHT_PX = _env_int("IBVAP_ANPR_OCR_TARGET_HEIGHT_PX", 64, minimum=16)
ANPR_OCR_LANGUAGES = tuple(
    language.lower() for language in _env_csv("IBVAP_ANPR_OCR_LANGUAGES", "en")
)
if not ANPR_OCR_LANGUAGES:
    raise ValueError("IBVAP_ANPR_OCR_LANGUAGES must contain at least one EasyOCR language code")
ANPR_REGIONS = tuple(region.upper() for region in _env_csv("IBVAP_ANPR_REGIONS", "IN"))
_supported_anpr_regions = {
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
}
_unknown_anpr_regions = set(ANPR_REGIONS) - _supported_anpr_regions
if not ANPR_REGIONS or _unknown_anpr_regions:
    raise ValueError(
        "IBVAP_ANPR_REGIONS must contain supported values; unknown: "
        + ", ".join(sorted(_unknown_anpr_regions))
    )
ANPR_MIN_OCR_CONFIDENCE = _env_float("IBVAP_ANPR_MIN_OCR_CONFIDENCE", 0.50, minimum=0.0)
if ANPR_MIN_OCR_CONFIDENCE > 1.0:
    raise ValueError("IBVAP_ANPR_MIN_OCR_CONFIDENCE must be <= 1")
ANPR_MIN_CONSENSUS_READS = _env_int("IBVAP_ANPR_MIN_CONSENSUS_READS", 3, minimum=1)
ANPR_CONSENSUS_WINDOW_SECONDS = _env_float("IBVAP_ANPR_CONSENSUS_WINDOW_SECONDS", 3.0, minimum=0.1)
ANPR_MIN_CONSENSUS_WEIGHT_RATIO = _env_float(
    "IBVAP_ANPR_MIN_CONSENSUS_WEIGHT_RATIO", 0.60, minimum=0.0
)
if ANPR_MIN_CONSENSUS_WEIGHT_RATIO > 1.0:
    raise ValueError("IBVAP_ANPR_MIN_CONSENSUS_WEIGHT_RATIO must be <= 1")
ANPR_MAX_OBSERVATIONS_PER_TRACK = _env_int("IBVAP_ANPR_MAX_OBSERVATIONS_PER_TRACK", 12, minimum=1)
if ANPR_MAX_OBSERVATIONS_PER_TRACK < ANPR_MIN_CONSENSUS_READS:
    raise ValueError("IBVAP_ANPR_MAX_OBSERVATIONS_PER_TRACK must be >= consensus reads")
ANPR_STATE_TTL_SECONDS = _env_float("IBVAP_ANPR_STATE_TTL_SECONDS", 5.0, minimum=0.1)

# Alerts and output
ALERT_COOLDOWN_SECONDS = _env_float("IBVAP_ALERT_COOLDOWN_SECONDS", 1.0, minimum=0.0)
ALERT_STATE_TTL_SECONDS = _env_float("IBVAP_ALERT_STATE_TTL_SECONDS", 300.0, minimum=1.0)
RECENT_ALERT_DISPLAY_SECONDS = _env_float("IBVAP_RECENT_ALERT_DISPLAY_SECONDS", 8.0, minimum=0.1)
MAX_RECENT_ALERTS = _env_int("IBVAP_MAX_RECENT_ALERTS", 20, minimum=1)
ENABLE_AUDIBLE_ALARMS = _env_bool("IBVAP_ENABLE_AUDIBLE_ALARMS", True)
AUDIBLE_ALARM_EVENTS = tuple(
    event.upper()
    for event in _env_csv(
        "IBVAP_AUDIBLE_ALARM_EVENTS",
        "VIRTUAL_FENCE_INTRUSION,SUSPICIOUS_LOITERING",
    )
)
_known_alarm_events = {
    "VIRTUAL_FENCE_INTRUSION",
    "SUSPICIOUS_LOITERING",
    "GROUP_APPROACH",
    "CONTEXTUAL_RISK",
    "NIGHT_MOVEMENT",
    "ANPR_READ",
}
_unknown_alarm_events = set(AUDIBLE_ALARM_EVENTS) - _known_alarm_events
if _unknown_alarm_events:
    raise ValueError(
        "IBVAP_AUDIBLE_ALARM_EVENTS contains unsupported values: "
        + ", ".join(sorted(_unknown_alarm_events))
    )
AUDIBLE_ALARM_VOLUME = _env_float("IBVAP_AUDIBLE_ALARM_VOLUME", 0.75, minimum=0.0)
if AUDIBLE_ALARM_VOLUME > 1.0:
    raise ValueError("IBVAP_AUDIBLE_ALARM_VOLUME must be <= 1")
AUDIBLE_ALARM_COOLDOWN_SECONDS = _env_float(
    "IBVAP_AUDIBLE_ALARM_COOLDOWN_SECONDS", 5.0, minimum=0.0
)
AUDIBLE_ALARM_QUEUE_SIZE = _env_int("IBVAP_AUDIBLE_ALARM_QUEUE_SIZE", 8, minimum=1)
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
