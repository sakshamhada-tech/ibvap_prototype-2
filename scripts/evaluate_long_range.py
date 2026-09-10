"""Collect and compare controlled wide-only versus native-focus evaluation traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from utils.evaluation import compare_runs, render_markdown_report, validate_trace  # noqa: E402
from utils.sources import (  # noqa: E402
    describe_source,
    is_file_source,
    is_network_source,
    normalize_fps,
    recorded_source_time,
)
from utils.timestamps import current_wall_times  # noqa: E402

LOGGER = logging.getLogger(__name__)


def load_bounded_json(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"JSON input does not exist: {resolved}")
    size = resolved.stat().st_size
    if size > config.EVALUATION_MAX_JSON_BYTES:
        raise ValueError(
            f"JSON input exceeds IBVAP_EVALUATION_MAX_JSON_BYTES={config.EVALUATION_MAX_JSON_BYTES}"
        )
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse JSON input {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must contain an object: {resolved}")
    return payload


def write_output(path: Path, content: str, *, overwrite: bool) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists() and not overwrite:
        raise ValueError(f"refusing to overwrite {destination}; pass --overwrite deliberately")
    encoded = content.encode("utf-8")
    if len(encoded) > config.EVALUATION_MAX_JSON_BYTES:
        raise ValueError(
            f"evaluation output exceeds IBVAP_EVALUATION_MAX_JSON_BYTES={config.EVALUATION_MAX_JSON_BYTES}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def validate_output_paths(paths: list[Path], *, overwrite: bool) -> None:
    resolved = [path.expanduser().resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("evaluation output paths must be distinct")
    existing = [path for path in resolved if path.exists()]
    if existing and not overwrite:
        raise ValueError(f"refusing to overwrite {existing[0]}; pass --overwrite deliberately")


def validate_collection_mode(mode: str) -> None:
    if mode == "wide-only" and (config.ENABLE_TILED_INFERENCE or config.ENABLE_NATIVE_FOCUS):
        raise ValueError(
            "wide-only collection requires IBVAP_ENABLE_TILED_INFERENCE=false and "
            "IBVAP_ENABLE_NATIVE_FOCUS=false"
        )
    if mode == "focus-enabled" and not (
        config.ENABLE_TILED_INFERENCE and config.ENABLE_NATIVE_FOCUS
    ):
        raise ValueError(
            "focus-enabled collection requires IBVAP_ENABLE_TILED_INFERENCE=true and "
            "IBVAP_ENABLE_NATIVE_FOCUS=true"
        )


def _trace_settings() -> dict:
    """Record fair-comparison inputs without persisting paths, URLs, or secrets."""
    return {
        "shared": {
            "build_label": config.EVALUATION_BUILD_LABEL,
            "hardware_label": config.EVALUATION_HARDWARE_LABEL,
            "model_filename": Path(config.YOLO_MODEL_PATH).name,
            "model_sha256": config.YOLO_MODEL_SHA256,
            "tracker": config.YOLO_TRACKER,
            "wide_input_size": config.YOLO_INPUT_SIZE,
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "virtual_fence": config.VIRTUAL_FENCE_LINE,
            "fence_contact_margin_px": config.VIRTUAL_FENCE_CONTACT_MARGIN_PX,
            "loitering_seconds": config.LOITERING_SECONDS,
            "loitering_movement_threshold_px": config.LOITERING_MOVEMENT_THRESHOLD_PX,
            "loitering_zone": config.LOITERING_ZONE,
            "track_max_observation_gap_seconds": config.TRACK_MAX_OBSERVATION_GAP_SECONDS,
            "track_stale_seconds": config.TRACK_STALE_SECONDS,
            "night_brightness_threshold": config.NIGHT_BRIGHTNESS_THRESHOLD,
            "night_brightness_hysteresis": config.NIGHT_BRIGHTNESS_HYSTERESIS,
            "night_brightness_smoothing_alpha": config.NIGHT_BRIGHTNESS_SMOOTHING_ALPHA,
            "night_brightness_roi_margin": config.NIGHT_BRIGHTNESS_ROI_MARGIN,
            "alert_cooldown_seconds": config.ALERT_COOLDOWN_SECONDS,
            "group_approach_enabled": config.ENABLE_GROUP_APPROACH,
            "group_min_size": config.GROUP_MIN_SIZE,
            "group_proximity_px": config.GROUP_PROXIMITY_PX,
            "group_approach_window_seconds": config.GROUP_APPROACH_WINDOW_SECONDS,
            "group_min_speed_px_per_second": config.GROUP_MIN_SPEED_PX_PER_SECOND,
            "group_heading_similarity": config.GROUP_HEADING_SIMILARITY,
            "group_min_fence_progress_px": config.GROUP_MIN_FENCE_PROGRESS_PX,
            "motion_signature_enabled": config.ENABLE_MOTION_SIGNATURE,
            "motion_signature_window_seconds": config.MOTION_SIGNATURE_WINDOW_SECONDS,
            "motion_signature_min_samples": config.MOTION_SIGNATURE_MIN_SAMPLES,
            "motion_signature_static_path_px": config.MOTION_SIGNATURE_STATIC_PATH_PX,
            "motion_signature_vegetation_max_net_px": config.MOTION_SIGNATURE_VEGETATION_MAX_NET_PX,
            "motion_signature_high_frequency_ratio": config.MOTION_SIGNATURE_HIGH_FREQUENCY_RATIO,
            "motion_signature_displacement_scale_px": config.MOTION_SIGNATURE_DISPLACEMENT_SCALE_PX,
            "motion_signature_oscillation_scale_px": config.MOTION_SIGNATURE_OSCILLATION_SCALE_PX,
            "motion_signature_scores": {
                "static": config.MOTION_SIGNATURE_STATIC_SCORE,
                "vegetation": config.MOTION_SIGNATURE_VEGETATION_SCORE,
                "gait": config.MOTION_SIGNATURE_GAIT_SCORE,
                "rigid": config.MOTION_SIGNATURE_RIGID_SCORE,
                "ambiguous": config.MOTION_SIGNATURE_AMBIGUOUS_SCORE,
            },
            "contextual_risk_enabled": config.ENABLE_CONTEXTUAL_RISK,
            "risk_alert_threshold": config.RISK_ALERT_THRESHOLD,
            "risk_fence_proximity_px": config.RISK_FENCE_PROXIMITY_PX,
            "risk_weights": {
                "fence_contact": config.RISK_WEIGHT_FENCE_CONTACT,
                "fence_proximity": config.RISK_WEIGHT_FENCE_PROXIMITY,
                "loitering": config.RISK_WEIGHT_LOITERING,
                "motion_signature": config.RISK_WEIGHT_MOTION_SIGNATURE,
                "low_posture": config.RISK_WEIGHT_LOW_POSTURE,
                "night": config.RISK_WEIGHT_NIGHT,
                "fence_approach": config.RISK_WEIGHT_FENCE_APPROACH,
                "group_membership": config.RISK_WEIGHT_GROUP_MEMBERSHIP,
            },
            "posture_analysis_enabled": config.ENABLE_POSTURE_ANALYSIS,
            "posture_model_filename": Path(config.POSTURE_MODEL_PATH).name,
            "posture_model_sha256": config.POSTURE_MODEL_SHA256,
            "posture_confidence_threshold": config.POSTURE_CONFIDENCE_THRESHOLD,
            "posture_interval_frames": config.POSTURE_INTERVAL_FRAMES,
            "posture_min_person_size_px": config.POSTURE_MIN_PERSON_SIZE_PX,
            "posture_low_aspect_ratio": config.POSTURE_LOW_ASPECT_RATIO,
            "posture_max_vertical_spread_ratio": config.POSTURE_MAX_VERTICAL_SPREAD_RATIO,
            "posture_min_visible_keypoints": config.POSTURE_MIN_VISIBLE_KEYPOINTS,
            "posture_min_keypoint_confidence": config.POSTURE_MIN_KEYPOINT_CONFIDENCE,
            "posture_sustain_seconds": config.POSTURE_SUSTAIN_SECONDS,
            "posture_state_ttl_seconds": config.POSTURE_STATE_TTL_SECONDS,
            "posture_result_max_age_frames": config.POSTURE_RESULT_MAX_AGE_FRAMES,
            "posture_device": config.POSTURE_DEVICE,
            "anpr_enabled": config.ENABLE_ANPR,
            "anpr_model_filename": Path(config.ANPR_PLATE_MODEL_PATH).name,
            "anpr_model_sha256": config.ANPR_PLATE_MODEL_SHA256,
            "anpr_interval_frames": config.ANPR_INTERVAL_FRAMES,
            "anpr_plate_confidence": config.ANPR_PLATE_DETECTION_CONFIDENCE,
            "anpr_plate_input_size": config.ANPR_PLATE_INPUT_SIZE,
            "anpr_min_plate_width_px": config.ANPR_MIN_PLATE_WIDTH_PX,
            "anpr_min_plate_sharpness": config.ANPR_MIN_PLATE_SHARPNESS,
            "anpr_min_ocr_confidence": config.ANPR_MIN_OCR_CONFIDENCE,
            "anpr_regions": config.ANPR_REGIONS,
            "anpr_consensus_reads": config.ANPR_MIN_CONSENSUS_READS,
            "anpr_consensus_window_seconds": config.ANPR_CONSENSUS_WINDOW_SECONDS,
            "anpr_consensus_weight_ratio": config.ANPR_MIN_CONSENSUS_WEIGHT_RATIO,
        },
        "mode_specific": {
            "tiled_inference": config.ENABLE_TILED_INFERENCE,
            "tile_size": config.TILE_SIZE,
            "tile_overlap": config.TILE_OVERLAP,
            "tile_interval_frames": config.TILED_INFERENCE_INTERVAL_FRAMES,
            "tile_nms_iou": config.TILED_INFERENCE_NMS_IOU,
            "native_focus": config.ENABLE_NATIVE_FOCUS,
            "focus_model_input_size": config.NATIVE_FOCUS_MODEL_INPUT_SIZE,
            "focus_track_input_size": config.NATIVE_FOCUS_TRACK_INPUT_SIZE,
            "focus_interval_frames": config.NATIVE_FOCUS_INTERVAL_FRAMES,
            "focus_max_regions": config.NATIVE_FOCUS_MAX_REGIONS,
            "focus_padding_ratio": config.NATIVE_FOCUS_PADDING_RATIO,
            "focus_min_crop_size_px": config.NATIVE_FOCUS_MIN_CROP_SIZE_PX,
            "focus_confidence_threshold": config.NATIVE_FOCUS_CONF_THRESHOLD,
            "focus_proposal_iou": config.NATIVE_FOCUS_PROPOSAL_IOU,
            "focus_confirmation_frames": config.NATIVE_FOCUS_CONFIRMATION_FRAMES,
            "focus_confirmation_iou": config.NATIVE_FOCUS_CONFIRMATION_IOU,
            "focus_max_gap_frames": config.NATIVE_FOCUS_MAX_GAP_FRAMES,
            "focus_track_boost_frames": config.NATIVE_FOCUS_TRACK_BOOST_FRAMES,
            "focus_result_max_age_frames": config.NATIVE_FOCUS_RESULT_MAX_AGE_FRAMES,
            "focus_acquisition_iou": config.NATIVE_FOCUS_ACQUISITION_IOU,
            "focus_device": config.NATIVE_FOCUS_DEVICE,
        },
    }


def _safe_alert(alert: dict) -> dict:
    """Retain evaluation fields without copying plate text or free-form details."""
    return {
        "alert_type": str(alert.get("alert_type", "UNKNOWN")),
        "frame_number": int(alert.get("frame_number", 0)),
        "track_id": alert.get("track_id"),
        "source_time_seconds": alert.get("source_time_seconds"),
    }


def collect_trace(
    *,
    source_setting: str | None,
    source_id: str,
    mode: str,
) -> dict:
    """Run one bounded configuration and return a metadata-only trace."""
    validate_collection_mode(mode)
    if (
        not source_id.strip()
        or len(source_id) > 128
        or any(not char.isprintable() for char in source_id)
    ):
        raise ValueError("--source-id must contain 1..128 printable characters")
    if "://" in source_id or "/" in source_id or "\\" in source_id or "@" in source_id:
        raise ValueError("--source-id must not contain a path, URL, or account identifier")
    source = (
        config.VIDEO_SOURCE
        if source_setting is None
        else config._parse_video_source(source_setting)
    )
    if is_file_source(source) and not Path(source).is_file():
        raise ValueError(f"video source does not exist: {source}")

    try:
        import cv2

        from pipeline import VideoPipeline
    except ImportError as exc:
        raise RuntimeError(f"evaluation runtime dependency is missing: {exc}") from exc

    capture = (
        cv2.VideoCapture(
            source,
            cv2.CAP_ANY,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                config.CAMERA_OPEN_TIMEOUT_MILLISECONDS,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                config.CAMERA_READ_TIMEOUT_MILLISECONDS,
            ],
        )
        if is_network_source(source)
        else cv2.VideoCapture(source)
    )
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open {describe_source(source)}")
    source_fps = normalize_fps(capture.get(cv2.CAP_PROP_FPS))
    source_is_file = is_file_source(source)
    observations: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []
    dropped_alerts = 0

    def on_frame(observation: dict) -> None:
        observations.append(observation)

    def on_alert(alert: dict) -> None:
        nonlocal dropped_alerts
        if len(alerts) >= config.EVALUATION_MAX_ALERTS:
            dropped_alerts += 1
            return
        alerts.append(_safe_alert(alert))

    try:
        pipeline = VideoPipeline(
            on_alert=on_alert,
            on_evaluation_frame=on_frame,
            show_overlays=False,
        )
    except Exception:
        capture.release()
        raise

    frame_number = 0
    reached_eof = False
    started = time.perf_counter()
    try:
        while frame_number < config.EVALUATION_MAX_FRAMES:
            ok, frame = capture.read()
            if not ok:
                reached_eof = True
                break
            frame_number += 1
            source_time = (
                recorded_source_time(capture.get(cv2.CAP_PROP_POS_MSEC), frame_number, source_fps)
                if source_is_file
                else time.monotonic()
            )
            source_frame_sha256 = hashlib.sha256(frame.tobytes()).hexdigest()
            observation_count = len(observations)
            pipeline.process_frame(frame, source_time=source_time)
            if len(observations) != observation_count + 1:
                raise RuntimeError(
                    "evaluation callback lost frame metadata; refuse to produce an incomplete trace"
                )
            if observations[-1].get("frame_number") != frame_number:
                raise RuntimeError("pipeline and collector frame numbers diverged")
            observations[-1]["source_frame_sha256"] = source_frame_sha256
    finally:
        capture.release()
        pipeline.close()
    elapsed = time.perf_counter() - started
    if not observations:
        raise RuntimeError("evaluation source produced no processable frames")
    if len(observations) != frame_number:
        raise RuntimeError(
            "evaluation callback lost frame metadata; refuse to produce an incomplete trace"
        )

    first = observations[0]
    local_time, utc_time = current_wall_times()
    trace = {
        "schema_version": 1,
        "source_id": source_id,
        "source_fps": source_fps,
        "source_width": first["source_width"],
        "source_height": first["source_height"],
        "mode": mode,
        "generated_system_local": local_time,
        "generated_utc": utc_time,
        "runtime_warmup_frames": min(
            config.EVALUATION_RUNTIME_WARMUP_FRAMES,
            max(0, len(observations) - 1),
        ),
        "settings": _trace_settings(),
        "run": {
            "frames_processed": frame_number,
            "elapsed_seconds": elapsed,
            "reached_eof": reached_eof,
            "frame_limit": config.EVALUATION_MAX_FRAMES,
            "dropped_alerts": dropped_alerts,
        },
        "frames": observations,
        "alerts": alerts,
    }
    validate_trace(
        trace,
        max_frames=config.EVALUATION_MAX_FRAMES,
        max_objects_per_frame=config.EVALUATION_MAX_OBJECTS_PER_FRAME,
        max_alerts=config.EVALUATION_MAX_ALERTS,
    )
    return trace


def build_report(annotations_path: Path, wide_path: Path, focus_path: Path) -> dict:
    annotations = load_bounded_json(annotations_path)
    wide_trace = load_bounded_json(wide_path)
    focus_trace = load_bounded_json(focus_path)
    return compare_runs(
        annotations,
        wide_trace,
        focus_trace,
        iou_threshold=config.EVALUATION_IOU_THRESHOLD,
        low_fps_percentile=config.EVALUATION_LOW_FPS_PERCENTILE,
        max_frames=config.EVALUATION_MAX_FRAMES,
        max_objects_per_frame=config.EVALUATION_MAX_OBJECTS_PER_FRAME,
        max_alerts=config.EVALUATION_MAX_ALERTS,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="collect one bounded runtime trace")
    collect.add_argument("--mode", choices=("wide-only", "focus-enabled"), required=True)
    collect.add_argument(
        "--source", help="video path/camera index/URL; defaults to IBVAP_VIDEO_SOURCE"
    )
    collect.add_argument(
        "--source-id",
        required=True,
        help="non-sensitive ID shared with annotations; camera URL/path is not stored",
    )
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--overwrite", action="store_true")

    report = subparsers.add_parser("report", help="score identical wide and focus traces")
    report.add_argument("--annotations", type=Path, required=True)
    report.add_argument("--wide-trace", type=Path, required=True)
    report.add_argument("--focus-trace", type=Path, required=True)
    report.add_argument("--json-output", type=Path, required=True)
    report.add_argument("--markdown-output", type=Path, required=True)
    report.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "collect":
            validate_output_paths([args.output], overwrite=args.overwrite)
            trace = collect_trace(
                source_setting=args.source,
                source_id=args.source_id,
                mode=args.mode,
            )
            destination = write_output(
                args.output,
                json.dumps(trace, indent=2, sort_keys=True) + "\n",
                overwrite=args.overwrite,
            )
            print(f"Trace written: {destination}")
            print(
                f"Frames: {trace['run']['frames_processed']} · mode: {trace['mode']} · "
                f"source pixels: {trace['source_width']}x{trace['source_height']}"
            )
            return 0

        validate_output_paths([args.json_output, args.markdown_output], overwrite=args.overwrite)
        report = build_report(args.annotations, args.wide_trace, args.focus_trace)
        json_destination = write_output(
            args.json_output,
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            overwrite=args.overwrite,
        )
        markdown_destination = write_output(
            args.markdown_output,
            render_markdown_report(report),
            overwrite=args.overwrite,
        )
        print(f"JSON report: {json_destination}")
        print(f"Markdown report: {markdown_destination}")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
