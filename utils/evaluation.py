"""Pure, deterministic scoring for controlled long-range IBVAP evaluations.

The evaluator consumes human-authored ground truth and metadata-only run traces.
It does not load model weights or video frames, so unit tests and CI remain
independent of OpenCV, PyTorch, and accelerator hardware.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from utils.timestamps import current_wall_times

SCHEMA_VERSION = 1
TARGET_HEIGHT_BANDS = ("<=24 px", "25-48 px", "49-96 px", ">96 px")
SUPPORTED_CLASSES = {"person", "car", "motorcycle", "bus", "truck"}
SUPPORTED_LIGHTING = {"daylight", "low-light"}
SUPPORTED_MODES = {"wide-only", "focus-enabled"}


def target_pixel_height_band(height_pixels: int | float) -> str:
    """Return the fixed dataset reporting band for source-pixel target height."""
    if height_pixels <= 24:
        return "<=24 px"
    if height_pixels <= 48:
        return "25-48 px"
    if height_pixels <= 96:
        return "49-96 px"
    return ">96 px"


def bbox_iou(
    first: list[float] | tuple[float, ...], second: list[float] | tuple[float, ...]
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _require_mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _require_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} must be >= {minimum:g}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label} must be <= {maximum:g}")
    return number


def _require_integer(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    number = _require_number(value, label, minimum=minimum, maximum=maximum)
    if not number.is_integer():
        raise ValueError(f"{label} must be an integer")
    return int(number)


def _require_source_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError(f"{label} must be a non-empty string of at most 128 characters")
    if any(not character.isprintable() for character in value):
        raise ValueError(f"{label} must contain only printable characters")
    return value


def _require_dataset_id(value: Any, label: str) -> str:
    source_id = _require_source_id(value, label)
    if "://" in source_id or "/" in source_id or "\\" in source_id or "@" in source_id:
        raise ValueError(f"{label} must not contain a path, URL, or account identifier")
    return source_id


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_bbox(value: Any, label: str) -> tuple[float, float, float, float]:
    coordinates = _require_list(value, label)
    if len(coordinates) != 4:
        raise ValueError(f"{label} must contain x1,y1,x2,y2")
    box = tuple(
        _require_number(item, f"{label}[{index}]") for index, item in enumerate(coordinates)
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{label} must have positive width and height")
    return box


def validate_annotations(
    annotations: dict,
    *,
    max_frames: int,
    max_objects_per_frame: int,
) -> None:
    root = _require_mapping(annotations, "annotations")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"annotations.schema_version must be {SCHEMA_VERSION}")
    _require_dataset_id(root.get("source_id"), "annotations.source_id")
    _require_number(root.get("source_fps"), "annotations.source_fps", minimum=0.001, maximum=1000)
    source_width = _require_integer(
        root.get("source_width"), "annotations.source_width", minimum=1, maximum=65_536
    )
    source_height = _require_integer(
        root.get("source_height"), "annotations.source_height", minimum=1, maximum=65_536
    )
    if not isinstance(root.get("exhaustive_frames"), bool):
        raise ValueError("annotations.exhaustive_frames must be true or false")
    if not isinstance(root.get("events_exhaustive"), bool):
        raise ValueError("annotations.events_exhaustive must be true or false")

    frames = _require_list(root.get("frames"), "annotations.frames")
    if not frames or len(frames) > max_frames:
        raise ValueError(f"annotations.frames must contain 1..{max_frames} entries")
    seen_frames: set[int] = set()
    object_classes: dict[str, str] = {}
    object_frames: dict[str, set[int]] = defaultdict(set)
    for frame_index, raw_frame in enumerate(frames):
        frame = _require_mapping(raw_frame, f"annotations.frames[{frame_index}]")
        frame_number = _require_integer(
            frame.get("frame_number"),
            f"annotations.frames[{frame_index}].frame_number",
            minimum=1,
        )
        if frame_number in seen_frames:
            raise ValueError(f"duplicate annotated frame_number {frame_number}")
        seen_frames.add(frame_number)
        lighting = frame.get("lighting")
        if lighting not in SUPPORTED_LIGHTING:
            raise ValueError(
                f"annotations.frames[{frame_index}].lighting must be daylight or low-light"
            )
        objects = _require_list(frame.get("objects"), f"annotations.frames[{frame_index}].objects")
        if len(objects) > max_objects_per_frame:
            raise ValueError(
                f"annotations.frames[{frame_index}].objects exceeds {max_objects_per_frame}"
            )
        seen_objects: set[str] = set()
        for object_index, raw_object in enumerate(objects):
            item_label = f"annotations.frames[{frame_index}].objects[{object_index}]"
            item = _require_mapping(raw_object, item_label)
            object_id = _require_source_id(item.get("object_id"), f"{item_label}.object_id")
            if object_id in seen_objects:
                raise ValueError(f"duplicate object_id {object_id!r} in frame {frame_number}")
            seen_objects.add(object_id)
            class_name = item.get("class_name")
            if class_name not in SUPPORTED_CLASSES:
                raise ValueError(f"{item_label}.class_name is unsupported")
            prior_class = object_classes.setdefault(object_id, class_name)
            if prior_class != class_name:
                raise ValueError(f"object_id {object_id!r} changes class across frames")
            object_frames[object_id].add(frame_number)
            bbox = _require_bbox(item.get("bbox"), f"{item_label}.bbox")
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > source_width or bbox[3] > source_height:
                raise ValueError(f"{item_label}.bbox must stay inside source dimensions")
    if root["exhaustive_frames"]:
        ordered_frames = sorted(seen_frames)
        expected_count = ordered_frames[-1] - ordered_frames[0] + 1
        if len(ordered_frames) != expected_count:
            raise ValueError("exhaustive annotations must contain every frame in their range")

    events = _require_list(root.get("events", []), "annotations.events")
    if len(events) > max_frames:
        raise ValueError(f"annotations.events exceeds {max_frames}")
    seen_event_ids: set[str] = set()
    event_windows: dict[str, list[tuple[int, int]]] = defaultdict(list)
    first_annotated_frame, last_annotated_frame = min(seen_frames), max(seen_frames)
    for event_index, raw_event in enumerate(events):
        item_label = f"annotations.events[{event_index}]"
        event = _require_mapping(raw_event, item_label)
        event_id = _require_source_id(event.get("event_id"), f"{item_label}.event_id")
        if event_id in seen_event_ids:
            raise ValueError(f"duplicate event_id {event_id!r}")
        seen_event_ids.add(event_id)
        event_type = _require_source_id(event.get("event_type"), f"{item_label}.event_type")
        if not isinstance(event.get("expected"), bool):
            raise ValueError(f"{item_label}.expected must be true or false")
        eligible = _require_integer(
            event.get("eligible_frame"), f"{item_label}.eligible_frame", minimum=1
        )
        end = _require_integer(event.get("end_frame"), f"{item_label}.end_frame", minimum=1)
        if end < eligible:
            raise ValueError(f"{item_label}.end_frame must be >= eligible_frame")
        if eligible < first_annotated_frame or end > last_annotated_frame:
            raise ValueError(f"{item_label} must stay inside the annotated frame range")
        event_windows[event_type].append((eligible, end))
        if "scenario" in event:
            _require_source_id(event["scenario"], f"{item_label}.scenario")
        target_ids = _require_list(
            event.get("target_object_ids", []), f"{item_label}.target_object_ids"
        )
        normalized_target_ids = [
            _require_source_id(target_id, f"{item_label}.target_object_ids[{target_index}]")
            for target_index, target_id in enumerate(target_ids)
        ]
        if len(set(normalized_target_ids)) != len(normalized_target_ids):
            raise ValueError(f"{item_label}.target_object_ids must be unique")
        unknown_targets = set(normalized_target_ids) - object_classes.keys()
        if unknown_targets:
            raise ValueError(
                f"{item_label}.target_object_ids reference unknown objects: {sorted(unknown_targets)}"
            )
        for target_id in normalized_target_ids:
            if not any(
                eligible <= frame_number <= end for frame_number in object_frames[target_id]
            ):
                raise ValueError(
                    f"{item_label}.target_object_ids includes {target_id!r}, which is absent from its window"
                )
    for event_type, windows in event_windows.items():
        ordered_windows = sorted(windows)
        if any(
            current[0] <= previous[1]
            for previous, current in zip(ordered_windows, ordered_windows[1:], strict=False)
        ):
            raise ValueError(f"{event_type} event windows must not overlap")


def validate_trace(
    trace: dict,
    *,
    max_frames: int,
    max_objects_per_frame: int,
    max_alerts: int,
) -> None:
    root = _require_mapping(trace, "trace")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"trace.schema_version must be {SCHEMA_VERSION}")
    _require_dataset_id(root.get("source_id"), "trace.source_id")
    _require_number(root.get("source_fps"), "trace.source_fps", minimum=0.001, maximum=1000)
    source_width = _require_integer(
        root.get("source_width"), "trace.source_width", minimum=1, maximum=65_536
    )
    source_height = _require_integer(
        root.get("source_height"), "trace.source_height", minimum=1, maximum=65_536
    )
    if root.get("mode") not in SUPPORTED_MODES:
        raise ValueError("trace.mode must be wide-only or focus-enabled")
    settings = _require_mapping(root.get("settings"), "trace.settings")
    shared_settings = _require_mapping(settings.get("shared"), "trace.settings.shared")
    for key in ("build_label", "hardware_label", "tracker"):
        _require_source_id(shared_settings.get(key), f"trace.settings.shared.{key}")
    if shared_settings["tracker"] != "bytetrack.yaml":
        raise ValueError("trace.settings.shared.tracker must be bytetrack.yaml")
    _require_sha256(shared_settings.get("model_sha256"), "trace.settings.shared.model_sha256")
    mode_settings = _require_mapping(settings.get("mode_specific"), "trace.settings.mode_specific")
    expected_mode_enabled = root["mode"] == "focus-enabled"
    if (
        mode_settings.get("tiled_inference") is not expected_mode_enabled
        or mode_settings.get("native_focus") is not expected_mode_enabled
    ):
        raise ValueError("trace mode does not match tiled/native-focus settings")
    warmup_frames = _require_integer(
        root.get("runtime_warmup_frames", 0),
        "trace.runtime_warmup_frames",
        minimum=0,
        maximum=max_frames,
    )
    run = _require_mapping(root.get("run"), "trace.run")
    frames_processed = _require_integer(
        run.get("frames_processed"), "trace.run.frames_processed", maximum=max_frames
    )
    _require_number(run.get("elapsed_seconds"), "trace.run.elapsed_seconds", minimum=0.000001)
    dropped_alerts = _require_integer(
        run.get("dropped_alerts", 0),
        "trace.run.dropped_alerts",
        maximum=max_alerts,
    )
    if dropped_alerts:
        raise ValueError("trace.run.dropped_alerts must be zero for complete event scoring")

    frames = _require_list(root.get("frames"), "trace.frames")
    if not frames or len(frames) > max_frames:
        raise ValueError(f"trace.frames must contain 1..{max_frames} entries")
    if frames_processed != len(frames):
        raise ValueError("trace.run.frames_processed must equal the number of trace frames")
    if warmup_frames >= len(frames):
        raise ValueError("trace.runtime_warmup_frames must leave at least one runtime sample")
    seen_frames: set[int] = set()
    for frame_index, raw_frame in enumerate(frames):
        item_label = f"trace.frames[{frame_index}]"
        frame = _require_mapping(raw_frame, item_label)
        frame_number = _require_integer(
            frame.get("frame_number"), f"{item_label}.frame_number", minimum=1
        )
        if frame_number in seen_frames:
            raise ValueError(f"duplicate trace frame_number {frame_number}")
        seen_frames.add(frame_number)
        if frame.get("source_width") != source_width or frame.get("source_height") != source_height:
            raise ValueError(f"{item_label} source dimensions must match the trace root")
        _require_sha256(frame.get("source_frame_sha256"), f"{item_label}.source_frame_sha256")
        _require_number(
            frame.get("source_time_seconds"),
            f"{item_label}.source_time_seconds",
            minimum=0,
        )
        if frame.get("lighting") not in SUPPORTED_LIGHTING:
            raise ValueError(f"{item_label}.lighting must be daylight or low-light")
        _require_number(
            frame.get("processing_seconds"), f"{item_label}.processing_seconds", minimum=0.000000001
        )
        detections = _require_list(frame.get("detections"), f"{item_label}.detections")
        if len(detections) > max_objects_per_frame:
            raise ValueError(f"{item_label}.detections exceeds {max_objects_per_frame}")
        for detection_index, raw_detection in enumerate(detections):
            detection_label = f"{item_label}.detections[{detection_index}]"
            detection = _require_mapping(raw_detection, detection_label)
            if detection.get("class_name") not in SUPPORTED_CLASSES:
                raise ValueError(f"{detection_label}.class_name is unsupported")
            bbox = _require_bbox(detection.get("bbox"), f"{detection_label}.bbox")
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > source_width or bbox[3] > source_height:
                raise ValueError(f"{detection_label}.bbox must stay inside source dimensions")
            _require_number(
                detection.get("confidence"),
                f"{detection_label}.confidence",
                minimum=0,
                maximum=1,
            )
            track_id = detection.get("track_id")
            if track_id is not None:
                _require_integer(track_id, f"{detection_label}.track_id", minimum=0)
            detection_source = detection.get("source")
            if detection_source not in {"full-frame", "tile"}:
                raise ValueError(f"{detection_label}.source must be full-frame or tile")
            if detection_source == "tile" and track_id is not None:
                raise ValueError(f"{detection_label}.track_id must be null for tile proposals")

    alerts = _require_list(root.get("alerts", []), "trace.alerts")
    if len(alerts) > max_alerts:
        raise ValueError(f"trace.alerts exceeds {max_alerts}")
    for alert_index, raw_alert in enumerate(alerts):
        item_label = f"trace.alerts[{alert_index}]"
        alert = _require_mapping(raw_alert, item_label)
        _require_source_id(alert.get("alert_type"), f"{item_label}.alert_type")
        alert_frame = _require_integer(
            alert.get("frame_number"), f"{item_label}.frame_number", minimum=1
        )
        if alert_frame not in seen_frames:
            raise ValueError(f"{item_label}.frame_number must reference a trace frame")
        track_id = alert.get("track_id")
        if track_id is not None:
            _require_integer(track_id, f"{item_label}.track_id", minimum=0)
        source_time = alert.get("source_time_seconds")
        if source_time is not None:
            _require_number(source_time, f"{item_label}.source_time_seconds", minimum=0)


def _greedy_matches(
    ground_truth: list[dict],
    predictions: list[dict],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidates = []
    for truth_index, truth in enumerate(ground_truth):
        for prediction_index, prediction in enumerate(predictions):
            if truth["class_name"] != prediction["class_name"]:
                continue
            overlap = bbox_iou(truth["bbox"], prediction["bbox"])
            if overlap >= iou_threshold:
                candidates.append((overlap, truth_index, prediction_index))
    matches = []
    matched_truth: set[int] = set()
    matched_predictions: set[int] = set()
    for overlap, truth_index, prediction_index in sorted(
        candidates, key=lambda candidate: (-candidate[0], candidate[1], candidate[2])
    ):
        if truth_index in matched_truth or prediction_index in matched_predictions:
            continue
        matched_truth.add(truth_index)
        matched_predictions.add(prediction_index)
        matches.append((truth_index, prediction_index, overlap))
    return matches, matched_truth, matched_predictions


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _detection_metrics(
    annotations: dict,
    trace_frames: dict[int, dict],
    iou_threshold: float,
    prediction_filter: Callable[[dict], bool],
) -> tuple[dict, dict[str, list[tuple[int, dict]]]]:
    totals = {"ground_truth": 0, "matched": 0, "false_detections": 0}
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"ground_truth": 0, "matched": 0})
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"ground_truth": 0, "matched": 0})
    by_band: dict[str, dict[str, int]] = defaultdict(lambda: {"ground_truth": 0, "matched": 0})
    by_lighting: dict[str, dict[str, int]] = defaultdict(lambda: {"ground_truth": 0, "matched": 0})
    by_class_band_lighting: dict[str, dict[str, int]] = defaultdict(
        lambda: {"ground_truth": 0, "matched": 0}
    )
    by_category_band_lighting: dict[str, dict[str, int]] = defaultdict(
        lambda: {"ground_truth": 0, "matched": 0}
    )
    object_matches: dict[str, list[tuple[int, dict]]] = defaultdict(list)

    for annotated_frame in annotations["frames"]:
        frame_number = annotated_frame["frame_number"]
        truth = annotated_frame["objects"]
        predictions = [
            prediction
            for prediction in trace_frames[frame_number]["detections"]
            if prediction_filter(prediction)
        ]
        matches, _matched_truth, matched_predictions = _greedy_matches(
            truth, predictions, iou_threshold
        )
        matched_truth_indexes = {truth_index for truth_index, _prediction_index, _iou in matches}
        totals["ground_truth"] += len(truth)
        totals["matched"] += len(matches)
        totals["false_detections"] += len(predictions) - len(matched_predictions)
        for truth_index, item in enumerate(truth):
            band = target_pixel_height_band(item["bbox"][3] - item["bbox"][1])
            class_name = item["class_name"]
            category = "person" if class_name == "person" else "vehicle"
            lighting = annotated_frame["lighting"]
            compound_key = f"{class_name}|{band}|{lighting}"
            category_compound_key = f"{category}|{band}|{lighting}"
            for bucket in (
                by_class[class_name],
                by_category[category],
                by_band[band],
                by_lighting[lighting],
                by_class_band_lighting[compound_key],
                by_category_band_lighting[category_compound_key],
            ):
                bucket["ground_truth"] += 1
                if truth_index in matched_truth_indexes:
                    bucket["matched"] += 1
        for truth_index, prediction_index, _overlap in matches:
            object_matches[truth[truth_index]["object_id"]].append(
                (frame_number, predictions[prediction_index])
            )

    def finish(bucket: dict[str, int]) -> dict:
        return {**bucket, "recall": _ratio(bucket["matched"], bucket["ground_truth"])}

    duration_minutes = None
    false_per_minute = None
    if annotations["exhaustive_frames"]:
        frame_numbers = [frame["frame_number"] for frame in annotations["frames"]]
        duration_minutes = (
            (max(frame_numbers) - min(frame_numbers) + 1) / annotations["source_fps"] / 60
        )
        false_per_minute = round(totals["false_detections"] / duration_minutes, 4)
    metrics = {
        "overall": finish({"ground_truth": totals["ground_truth"], "matched": totals["matched"]}),
        "false_detections": {
            "count": totals["false_detections"],
            "evaluated_duration_minutes": (
                None if duration_minutes is None else round(duration_minutes, 4)
            ),
            "per_minute": false_per_minute,
        },
        "by_class": {key: finish(value) for key, value in sorted(by_class.items())},
        "by_category": {
            category: finish(by_category[category])
            if category in by_category
            else finish({"ground_truth": 0, "matched": 0})
            for category in ("person", "vehicle")
        },
        "by_height_band": {
            band: finish(by_band[band])
            if band in by_band
            else finish({"ground_truth": 0, "matched": 0})
            for band in TARGET_HEIGHT_BANDS
        },
        "by_lighting": {
            key: finish(by_lighting[key])
            if key in by_lighting
            else finish({"ground_truth": 0, "matched": 0})
            for key in sorted(SUPPORTED_LIGHTING)
        },
        "by_class_height_lighting": {
            key: finish(value) for key, value in sorted(by_class_band_lighting.items())
        },
        "by_category_height_lighting": {
            key: finish(value) for key, value in sorted(by_category_band_lighting.items())
        },
    }
    return metrics, object_matches


def _tracking_metrics(
    detected_matches: dict[str, list[tuple[int, dict]]],
    authoritative_matches: dict[str, list[tuple[int, dict]]],
    fps: float,
) -> dict:
    detected_observations = sum(len(matches) for matches in detected_matches.values())
    authoritative_observations = sum(len(matches) for matches in authoritative_matches.values())
    detected_targets = set(detected_matches)
    acquired_targets = set(authoritative_matches)
    by_object = {}
    acquisition_seconds: list[float] = []
    total_switches = 0
    longest_frames = 0
    for object_id in sorted(detected_targets):
        detected_entries = sorted(detected_matches[object_id])
        entries = sorted(authoritative_matches.get(object_id, []))
        switches = 0
        longest = 0
        current = 0
        previous_frame = None
        previous_track = None
        observed_track_ids = []
        for frame_number, prediction in entries:
            track_id = prediction.get("track_id")
            if track_id is None:
                continue
            observed_track_ids.append(track_id)
            if previous_track is not None and track_id != previous_track:
                switches += 1
            if (
                previous_frame is not None
                and frame_number == previous_frame + 1
                and track_id == previous_track
            ):
                current += 1
            else:
                current = 1
            longest = max(longest, current)
            previous_frame = frame_number
            previous_track = track_id
        total_switches += switches
        longest_frames = max(longest_frames, longest)
        acquisition_frames = entries[0][0] - detected_entries[0][0] if entries else None
        acquisition_time = (
            round(acquisition_frames / fps, 4) if acquisition_frames is not None else None
        )
        if acquisition_time is not None:
            acquisition_seconds.append(acquisition_time)
        by_object[object_id] = {
            "detected_observations": len(detected_entries),
            "authoritative_observations": len(entries),
            "acquired": bool(entries),
            "acquisition_frames": acquisition_frames,
            "acquisition_seconds": acquisition_time,
            "id_switches": switches,
            "distinct_track_ids": len(set(observed_track_ids)),
            "longest_continuous_frames": longest,
            "longest_continuous_seconds": round(longest / fps, 4),
        }
    return {
        "detected_targets": len(detected_targets),
        "acquired_targets": len(acquired_targets),
        "target_acquisition_rate": _ratio(len(acquired_targets), len(detected_targets)),
        "median_acquisition_seconds": (
            round(statistics.median(acquisition_seconds), 4) if acquisition_seconds else None
        ),
        "detected_observations": detected_observations,
        "authoritative_observations": authoritative_observations,
        "observation_acquisition_rate": _ratio(authoritative_observations, detected_observations),
        "id_switches": total_switches,
        "longest_continuous_frames": longest_frames,
        "longest_continuous_seconds": round(longest_frames / fps, 4),
        "by_object": by_object,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _runtime_metrics(trace: dict, low_fps_percentile: float) -> dict:
    warmup = trace.get("runtime_warmup_frames", 0)
    ordered_frames = sorted(trace["frames"], key=lambda frame: frame["frame_number"])
    samples = [1 / frame["processing_seconds"] for frame in ordered_frames[warmup:]]
    elapsed = trace["run"]["elapsed_seconds"]
    processed = trace["run"]["frames_processed"]
    return {
        "warmup_frames_excluded": warmup,
        "samples": len(samples),
        "median_fps": round(statistics.median(samples), 4) if samples else None,
        "low_fps_percentile": low_fps_percentile,
        "low_percentile_fps": (
            None if not samples else round(_percentile(samples, low_fps_percentile) or 0.0, 4)
        ),
        "wall_throughput_fps": round(processed / elapsed, 4),
        "elapsed_seconds": round(elapsed, 4),
    }


def _event_metrics(annotations: dict, trace: dict) -> dict:
    annotated_events = annotations.get("events", [])
    trace_alerts = trace.get("alerts", [])
    by_type: dict[str, dict] = {}
    event_types = {event["event_type"] for event in annotated_events}
    if annotations["events_exhaustive"]:
        event_types.update(alert["alert_type"] for alert in trace_alerts)
    for event_type in sorted(event_types):
        events = [event for event in annotated_events if event["event_type"] == event_type]
        alerts = [alert for alert in trace_alerts if alert["alert_type"] == event_type]
        matched_alert_indexes: set[int] = set()
        matched_positive = 0
        latencies = []
        scenarios: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"expected_positive": 0, "expected_negative": 0, "detected": 0}
        )
        for event in sorted(events, key=lambda item: item["eligible_frame"]):
            matching_indexes = [
                index
                for index, alert in enumerate(alerts)
                if index not in matched_alert_indexes
                and event["eligible_frame"] <= alert["frame_number"] <= event["end_frame"]
            ]
            detected = bool(matching_indexes)
            scenario = event.get("scenario", "unspecified")
            scenarios[scenario][
                "expected_positive" if event["expected"] else "expected_negative"
            ] += 1
            scenarios[scenario]["detected"] += int(detected)
            if event["expected"] and detected:
                alert_index = min(matching_indexes, key=lambda index: alerts[index]["frame_number"])
                matched_alert_indexes.add(alert_index)
                matched_positive += 1
                latency_frames = alerts[alert_index]["frame_number"] - event["eligible_frame"]
                latencies.append(latency_frames / annotations["source_fps"])

        positive_count = sum(event["expected"] for event in events)
        false_alerts = len(alerts) - matched_positive if annotations["events_exhaustive"] else None
        precision = (
            _ratio(matched_positive, matched_positive + false_alerts)
            if false_alerts is not None
            else None
        )
        by_type[event_type] = {
            "expected_positive": positive_count,
            "matched_positive": matched_positive,
            "false_alerts": false_alerts,
            "precision": precision,
            "recall": _ratio(matched_positive, positive_count),
            "median_alert_latency_seconds": (
                round(statistics.median(latencies), 4) if latencies else None
            ),
            "scenarios": {key: value for key, value in sorted(scenarios.items())},
        }
    return by_type


def evaluate_run(
    annotations: dict,
    trace: dict,
    *,
    iou_threshold: float,
    low_fps_percentile: float,
) -> dict:
    if annotations["source_id"] != trace["source_id"]:
        raise ValueError("annotation and trace source_id values do not match")
    if not math.isclose(annotations["source_fps"], trace["source_fps"], rel_tol=0, abs_tol=0.001):
        raise ValueError("annotation and trace source_fps values do not match")
    trace_frames = {frame["frame_number"]: frame for frame in trace["frames"]}
    missing = sorted(
        frame["frame_number"]
        for frame in annotations["frames"]
        if frame["frame_number"] not in trace_frames
    )
    if missing:
        raise ValueError(f"trace is missing {len(missing)} annotated frame(s), first={missing[0]}")

    detections, detected_matches = _detection_metrics(
        annotations, trace_frames, iou_threshold, lambda _prediction: True
    )
    authoritative, authoritative_matches = _detection_metrics(
        annotations,
        trace_frames,
        iou_threshold,
        lambda prediction: (
            prediction.get("track_id") is not None and prediction.get("source") == "full-frame"
        ),
    )
    return {
        "mode": trace["mode"],
        "detection": detections,
        "authoritative_detection": authoritative,
        "tracking": _tracking_metrics(
            detected_matches, authoritative_matches, annotations["source_fps"]
        ),
        "runtime": _runtime_metrics(trace, low_fps_percentile),
        "events": _event_metrics(annotations, trace),
        "settings": trace.get("settings", {}),
    }


def compare_runs(
    annotations: dict,
    wide_trace: dict,
    focus_trace: dict,
    *,
    iou_threshold: float,
    low_fps_percentile: float,
    max_frames: int,
    max_objects_per_frame: int,
    max_alerts: int,
) -> dict:
    """Validate and compare wide-only and focus-enabled traces on identical labels."""
    validate_annotations(
        annotations,
        max_frames=max_frames,
        max_objects_per_frame=max_objects_per_frame,
    )
    for trace in (wide_trace, focus_trace):
        validate_trace(
            trace,
            max_frames=max_frames,
            max_objects_per_frame=max_objects_per_frame,
            max_alerts=max_alerts,
        )
    if wide_trace["mode"] != "wide-only" or focus_trace["mode"] != "focus-enabled":
        raise ValueError("comparison requires wide-only then focus-enabled traces")
    if wide_trace["settings"]["shared"] != focus_trace["settings"]["shared"]:
        raise ValueError("wide-only and focus-enabled shared settings must match exactly")
    if (
        annotations["source_width"],
        annotations["source_height"],
    ) != (wide_trace["source_width"], wide_trace["source_height"]) or (
        annotations["source_width"],
        annotations["source_height"],
    ) != (focus_trace["source_width"], focus_trace["source_height"]):
        raise ValueError("annotation and trace source dimensions do not match")
    wide_frames = {frame["frame_number"] for frame in wide_trace["frames"]}
    focus_frames = {frame["frame_number"] for frame in focus_trace["frames"]}
    if wide_frames != focus_frames:
        raise ValueError("wide-only and focus-enabled traces must contain identical frame numbers")
    annotated_frames = {frame["frame_number"] for frame in annotations["frames"]}
    if annotations["exhaustive_frames"] and wide_frames != annotated_frames:
        raise ValueError("exhaustive annotations must cover every compared trace frame")
    wide_frame_hashes = {
        frame["frame_number"]: frame["source_frame_sha256"] for frame in wide_trace["frames"]
    }
    focus_frame_hashes = {
        frame["frame_number"]: frame["source_frame_sha256"] for frame in focus_trace["frames"]
    }
    if wide_frame_hashes != focus_frame_hashes:
        raise ValueError(
            "wide-only and focus-enabled traces must contain identical decoded source frames"
        )

    wide = evaluate_run(
        annotations,
        wide_trace,
        iou_threshold=iou_threshold,
        low_fps_percentile=low_fps_percentile,
    )
    focus = evaluate_run(
        annotations,
        focus_trace,
        iou_threshold=iou_threshold,
        low_fps_percentile=low_fps_percentile,
    )

    def delta(focus_value: float | None, wide_value: float | None) -> float | None:
        if focus_value is None or wide_value is None:
            return None
        return round(focus_value - wide_value, 4)

    by_band = {}
    for band in TARGET_HEIGHT_BANDS:
        wide_detection = wide["detection"]["by_height_band"][band]["recall"]
        focus_detection = focus["detection"]["by_height_band"][band]["recall"]
        wide_authoritative = wide["authoritative_detection"]["by_height_band"][band]["recall"]
        focus_authoritative = focus["authoritative_detection"]["by_height_band"][band]["recall"]
        by_band[band] = {
            "wide_detection_recall": wide_detection,
            "focus_detection_recall": focus_detection,
            "detection_recall_delta": delta(focus_detection, wide_detection),
            "wide_authoritative_recall": wide_authoritative,
            "focus_authoritative_recall": focus_authoritative,
            "authoritative_recall_delta": delta(focus_authoritative, wide_authoritative),
        }

    local_time, utc_time = current_wall_times()
    shared_settings = wide_trace["settings"]["shared"]
    limitations = [
        "These measurements apply only to the labelled source and configuration recorded in each trace.",
        "Digital focus preserves source pixels but cannot create detail absent from the camera signal.",
        "Event matching is clip-window based and does not claim identity or cross-camera re-identification.",
        "Weather, sensor, terrain, compression, and camera-motion conditions require separate validation.",
    ]
    if "unrecorded" in {
        shared_settings.get("build_label"),
        shared_settings.get("hardware_label"),
    }:
        limitations.insert(
            0,
            "Build or hardware metadata is unrecorded; this run is not ready for publication.",
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_system_local": local_time,
        "generated_utc": utc_time,
        "source_id": annotations["source_id"],
        "build_label": shared_settings["build_label"],
        "hardware_label": shared_settings["hardware_label"],
        "iou_threshold": iou_threshold,
        "low_fps_percentile": low_fps_percentile,
        "runs": {"wide-only": wide, "focus-enabled": focus},
        "comparison": {
            "detection_recall_delta": delta(
                focus["detection"]["overall"]["recall"],
                wide["detection"]["overall"]["recall"],
            ),
            "authoritative_recall_delta": delta(
                focus["authoritative_detection"]["overall"]["recall"],
                wide["authoritative_detection"]["overall"]["recall"],
            ),
            "target_acquisition_rate_delta": delta(
                focus["tracking"]["target_acquisition_rate"],
                wide["tracking"]["target_acquisition_rate"],
            ),
            "median_fps_delta": delta(
                focus["runtime"]["median_fps"], wide["runtime"]["median_fps"]
            ),
            "by_height_band": by_band,
        },
        "limitations": limitations,
    }


def render_markdown_report(report: dict) -> str:
    """Render a concise SIH-ready table without turning results into universal claims."""

    def value(raw: Any, *, percent: bool = False) -> str:
        if raw is None:
            return "—"
        if percent:
            return f"{raw * 100:.1f}%"
        if isinstance(raw, float):
            return f"{raw:.2f}"
        return str(raw)

    def text(raw: Any) -> str:
        return (
            str(raw)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace("|", "\\|")
            .replace("`", "\\`")
        )

    lines = [
        f"# Astra I controlled evaluation — {text(report['source_id'])}",
        "",
        f"Generated (system local): `{text(report['generated_system_local'])}`  ",
        f"Build label: `{text(report['build_label'])}`  ",
        f"Hardware label: `{text(report['hardware_label'])}`  ",
        f"IoU match threshold: `{report['iou_threshold']}`",
        "",
        "> Controlled labelled-footage result only. This is not a field-range or all-weather capability claim.",
        "",
        "## Baseline versus focus summary",
        "",
        "| Metric | Wide only | Focus enabled |",
        "|---|---:|---:|",
    ]
    wide = report["runs"]["wide-only"]
    focus = report["runs"]["focus-enabled"]
    rows = (
        (
            "Detection recall",
            wide["detection"]["overall"]["recall"],
            focus["detection"]["overall"]["recall"],
            True,
        ),
        (
            "Authoritative tracked recall",
            wide["authoritative_detection"]["overall"]["recall"],
            focus["authoritative_detection"]["overall"]["recall"],
            True,
        ),
        (
            "Target acquisition rate",
            wide["tracking"]["target_acquisition_rate"],
            focus["tracking"]["target_acquisition_rate"],
            True,
        ),
        (
            "Median acquisition time (s)",
            wide["tracking"]["median_acquisition_seconds"],
            focus["tracking"]["median_acquisition_seconds"],
            False,
        ),
        (
            "False detections",
            wide["detection"]["false_detections"]["count"],
            focus["detection"]["false_detections"]["count"],
            False,
        ),
        (
            "False detections/min",
            wide["detection"]["false_detections"]["per_minute"],
            focus["detection"]["false_detections"]["per_minute"],
            False,
        ),
        ("ID switches", wide["tracking"]["id_switches"], focus["tracking"]["id_switches"], False),
        (
            "Longest continuous track (s)",
            wide["tracking"]["longest_continuous_seconds"],
            focus["tracking"]["longest_continuous_seconds"],
            False,
        ),
        (
            "Median processing FPS",
            wide["runtime"]["median_fps"],
            focus["runtime"]["median_fps"],
            False,
        ),
        (
            f"P{report['low_fps_percentile']:g} processing FPS",
            wide["runtime"]["low_percentile_fps"],
            focus["runtime"]["low_percentile_fps"],
            False,
        ),
        (
            "Wall throughput FPS",
            wide["runtime"]["wall_throughput_fps"],
            focus["runtime"]["wall_throughput_fps"],
            False,
        ),
    )
    lines.extend(
        f"| {label} | {value(wide_value, percent=percent)} | {value(focus_value, percent=percent)} |"
        for label, wide_value, focus_value, percent in rows
    )
    lines.extend(
        [
            "",
            "## Source target-height bands",
            "",
            "| Source target height | Wide detection | Focus detection | Wide authoritative | Focus authoritative | Authoritative delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for band, metrics in report["comparison"]["by_height_band"].items():
        lines.append(
            "| "
            + " | ".join(
                (
                    band,
                    value(metrics["wide_detection_recall"], percent=True),
                    value(metrics["focus_detection_recall"], percent=True),
                    value(metrics["wide_authoritative_recall"], percent=True),
                    value(metrics["focus_authoritative_recall"], percent=True),
                    value(metrics["authoritative_recall_delta"], percent=True),
                )
            )
            + " |"
        )

    compound_keys = set(wide["detection"]["by_category_height_lighting"]) | set(
        focus["detection"]["by_category_height_lighting"]
    )
    compound_keys = sorted(
        compound_keys,
        key=lambda key: (
            ("person", "vehicle").index(key.split("|", 2)[0]),
            TARGET_HEIGHT_BANDS.index(key.split("|", 2)[1]),
            ("daylight", "low-light").index(key.split("|", 2)[2]),
        ),
    )
    if compound_keys:
        lines.extend(
            [
                "",
                "## Person/vehicle, target-height, and lighting breakdown",
                "",
                "| Category | Height band | Lighting | Wide detection | Focus detection | Wide authoritative | Focus authoritative |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for key in compound_keys:
            category, band, lighting = key.split("|", 2)
            wide_detection = wide["detection"]["by_category_height_lighting"].get(key, {})
            focus_detection = focus["detection"]["by_category_height_lighting"].get(key, {})
            wide_authoritative = wide["authoritative_detection"]["by_category_height_lighting"].get(
                key, {}
            )
            focus_authoritative = focus["authoritative_detection"][
                "by_category_height_lighting"
            ].get(key, {})
            lines.append(
                f"| {text(category)} | {band} | {text(lighting)} | "
                f"{value(wide_detection.get('recall'), percent=True)} | "
                f"{value(focus_detection.get('recall'), percent=True)} | "
                f"{value(wide_authoritative.get('recall'), percent=True)} | "
                f"{value(focus_authoritative.get('recall'), percent=True)} |"
            )

    event_types = sorted(set(wide["events"]) | set(focus["events"]))
    if event_types:
        lines.extend(
            [
                "",
                "## Labelled event windows",
                "",
                "| Event | Wide precision | Wide recall | Focus precision | Focus recall | Wide median latency (s) | Focus median latency (s) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for event_type in event_types:
            wide_event = wide["events"].get(event_type, {})
            focus_event = focus["events"].get(event_type, {})
            lines.append(
                f"| {text(event_type)} | {value(wide_event.get('precision'), percent=True)} | "
                f"{value(wide_event.get('recall'), percent=True)} | "
                f"{value(focus_event.get('precision'), percent=True)} | "
                f"{value(focus_event.get('recall'), percent=True)} | "
                f"{value(wide_event.get('median_alert_latency_seconds'))} | "
                f"{value(focus_event.get('median_alert_latency_seconds'))} |"
            )
        lines.extend(
            [
                "",
                "### Event scenario breakdown",
                "",
                "| Event | Scenario | Expected positive | Expected negative | Wide detected windows | Focus detected windows |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for event_type in event_types:
            wide_scenarios = wide["events"].get(event_type, {}).get("scenarios", {})
            focus_scenarios = focus["events"].get(event_type, {}).get("scenarios", {})
            for scenario in sorted(set(wide_scenarios) | set(focus_scenarios)):
                wide_scenario = wide_scenarios.get(scenario, {})
                focus_scenario = focus_scenarios.get(scenario, {})
                reference = wide_scenario or focus_scenario
                lines.append(
                    f"| {text(event_type)} | {text(scenario)} | "
                    f"{value(reference.get('expected_positive'))} | "
                    f"{value(reference.get('expected_negative'))} | "
                    f"{value(wide_scenario.get('detected'))} | "
                    f"{value(focus_scenario.get('detected'))} |"
                )

    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.extend(
        [
            "",
            "Detection recall includes matching full-frame and supplemental tile observations. "
            "Authoritative recall includes only full-frame detections carrying a ByteTrack ID.",
            "",
        ]
    )
    return "\n".join(lines)
