"""YOLO tracking plus optional face and regional number-plate detection."""

from __future__ import annotations

import hashlib
import logging
import queue
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

import config
from focus_analysis import NativeFocusService
from utils.anpr import PlateObservation, combine_ocr_results, identify_plate_region

LOGGER = logging.getLogger(__name__)
_LATIN_ANPR_REGIONS = {"IN", "BT", "LK", "MV"}
_LATIN_OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class TiledInferenceWorker:
    """Run supplemental tiled inference outside the live-frame critical path."""

    def __init__(
        self,
        process: Callable[[Any, Any], list[dict]],
        on_result: Callable[[Any, list[dict], int], None],
    ) -> None:
        self._process = process
        self._on_result = on_result
        self._jobs: queue.Queue[tuple[Any, int, int]] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._accepting_jobs = False
        self._generation = 0
        self._snapshot: dict[str, Any] = {
            "status": "starting",
            "mode": "background_drop_on_busy",
            "device": config.TILED_INFERENCE_DEVICE,
            "queue_capacity": 1,
            "submitted": 0,
            "completed": 0,
            "dropped_busy": 0,
            "errors": 0,
            "last_frame_number": None,
            "last_proposals": 0,
            "last_inference_seconds": None,
            "message": None,
        }

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="ibvap-tiled-inference",
            daemon=True,
        )
        self._thread.start()

    def submit(self, frame, frame_number: int) -> bool:
        if not self._accepting_jobs or self._stop_event.is_set():
            return False
        try:
            frame_copy = frame.copy()
        except Exception as exc:
            self._record_error(f"tile frame copy failed: {exc}")
            return False
        with self._lock:
            generation = self._generation
        try:
            self._jobs.put_nowait((frame_copy, frame_number, generation))
        except queue.Full:
            with self._lock:
                self._snapshot["dropped_busy"] += 1
            return False
        with self._lock:
            self._snapshot["submitted"] += 1
        return True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._snapshot)

    def reset(self) -> None:
        with self._lock:
            self._generation += 1
            self._snapshot.update(
                {
                    "last_frame_number": None,
                    "last_proposals": 0,
                    "last_inference_seconds": None,
                    "message": None,
                }
            )
        while True:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        self._accepting_jobs = False
        self._stop_event.set()
        while True:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                break
        if self._thread is not None:
            self._thread.join(timeout=config.TILED_INFERENCE_SHUTDOWN_TIMEOUT_SECONDS)
        if self._thread is not None and self._thread.is_alive():
            LOGGER.error("tiled inference worker did not stop before the shutdown timeout")

    def _run(self) -> None:
        try:
            # Loading inside the worker keeps optional model setup off the capture thread.
            model = YOLO(config.YOLO_MODEL_PATH)
        except Exception as exc:
            LOGGER.exception("tiled detector initialization failed; full-frame tracking continues")
            self._record_error(f"tiled detector unavailable: {exc}", status="unavailable")
            return
        self._accepting_jobs = True
        with self._lock:
            self._snapshot["status"] = "ready"
        LOGGER.info(
            "tiled inference ready in bounded background mode on device=%s",
            config.TILED_INFERENCE_DEVICE,
        )
        while not self._stop_event.is_set():
            try:
                frame, frame_number, generation = self._jobs.get(
                    timeout=config.TILED_INFERENCE_QUEUE_POLL_SECONDS
                )
            except queue.Empty:
                continue
            started = time.perf_counter()
            try:
                detections = self._process(model, frame)
            except Exception as exc:
                LOGGER.exception("tiled inference failed; full-frame tracking continues")
                self._record_error(f"tiled inference failed: {exc}")
                continue
            elapsed = time.perf_counter() - started
            with self._lock:
                if generation != self._generation:
                    continue
                self._snapshot.update(
                    {
                        "status": "ready",
                        "completed": self._snapshot["completed"] + 1,
                        "last_frame_number": frame_number,
                        "last_proposals": len(detections),
                        "last_inference_seconds": round(elapsed, 4),
                        "message": None,
                    }
                )
            try:
                self._on_result(frame, detections, frame_number)
            except Exception:
                LOGGER.exception("tiled result callback failed; full-frame tracking continues")

    def _record_error(self, message: str, *, status: str = "error") -> None:
        with self._lock:
            self._snapshot.update(
                {
                    "status": status,
                    "errors": self._snapshot["errors"] + 1,
                    "message": message,
                }
            )


class Detector:
    @staticmethod
    def _new_anpr_metrics() -> dict:
        return {
            "status": "enabled" if config.ENABLE_ANPR else "disabled",
            "attempts": 0,
            "detector_candidates": 0,
            "rejected_small": 0,
            "rejected_blurry": 0,
            "ocr_no_text": 0,
            "ocr_low_confidence": 0,
            "region_rejected": 0,
            "valid_observations": 0,
            "inference_errors": 0,
            "last_result": "waiting" if config.ENABLE_ANPR else "disabled",
        }

    def __init__(
        self,
        on_focus_snapshot: Callable[[dict[str, Any]], None] | None = None,
        *,
        frame_aligned_tiling: bool = False,
    ):
        self._anpr_metrics = self._new_anpr_metrics()
        self._on_focus_snapshot = on_focus_snapshot
        self._focus_lock = threading.RLock()
        self._focus_snapshot: dict[str, Any] = {
            "status": "disabled" if not config.ENABLE_NATIVE_FOCUS else "starting",
            "focus": None,
        }
        self._focus_boost_remaining = 0
        self._acquired_focus_episodes: dict[int, int] = {}
        self._detection_diagnostics: dict[str, Any] = {
            "tracker": config.YOLO_TRACKER,
            "configured_input_size": config.YOLO_INPUT_SIZE,
            "active_input_size": config.YOLO_INPUT_SIZE,
            "tracking_boost_active": False,
            "tracked_full_frame": 0,
            "untracked_full_frame": 0,
            "untracked_tile": 0,
            "total": 0,
        }
        self._verify_model_file(
            config.YOLO_MODEL_PATH,
            config.YOLO_MODEL_SHA256,
            "object detector",
            required=False,
        )
        LOGGER.info(
            "loading YOLO model %s on device=%s", config.YOLO_MODEL_PATH, config.YOLO_DEVICE
        )
        self.model = YOLO(config.YOLO_MODEL_PATH)
        self.tile_model = None
        self.tile_worker: TiledInferenceWorker | None = None
        self._tile_cursor = 0
        self._frame_aligned_tiling = bool(frame_aligned_tiling)
        self._tiled_snapshot: dict[str, Any] = {
            "status": "disabled" if not config.ENABLE_TILED_INFERENCE else "starting",
            "mode": "frame_aligned" if self._frame_aligned_tiling else "background_drop_on_busy",
            "device": config.TILED_INFERENCE_DEVICE,
            "last_frame_number": None,
            "last_proposals": 0,
            "message": None,
        }
        if config.ENABLE_TILED_INFERENCE and self._frame_aligned_tiling:
            try:
                # Evaluation requires proposals from the exact source frame. A
                # separate predictor protects ByteTrack state while preserving parity.
                self.tile_model = YOLO(config.YOLO_MODEL_PATH)
                self._tiled_snapshot["status"] = "ready"
                LOGGER.info(
                    "tiled inference ready in frame-aligned evaluation mode on device=%s",
                    config.TILED_INFERENCE_DEVICE,
                )
            except Exception as exc:
                LOGGER.exception(
                    "tiled detector initialization failed; full-frame tracking continues"
                )
                self._tiled_snapshot.update(status="unavailable", message=str(exc))
        elif config.ENABLE_TILED_INFERENCE:
            self.tile_worker = TiledInferenceWorker(
                self._process_tiled_job,
                self._handle_tiled_result,
            )
        self._frame_number = 0
        self._tiled_error_active = False
        self.focus_service = NativeFocusService(on_snapshot=self._handle_focus_snapshot)
        self.plate_model = None
        self.ocr_reader = None
        self._anpr_inference_error_active = False
        if config.ENABLE_ANPR:
            self._verify_model_file(
                config.ANPR_PLATE_MODEL_PATH,
                config.ANPR_PLATE_MODEL_SHA256,
                "ANPR plate detector",
                required=True,
            )
            LOGGER.info(
                "loading regional YOLO plate detector %s on device=%s",
                config.ANPR_PLATE_MODEL_PATH,
                config.YOLO_DEVICE,
            )
            self.plate_model = YOLO(config.ANPR_PLATE_MODEL_PATH)
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError(
                    "ANPR is enabled but EasyOCR is not installed; install requirements-anpr.lock"
                ) from exc
            LOGGER.info(
                "loading EasyOCR languages %s for ANPR; first use may download OCR weights",
                ",".join(config.ANPR_OCR_LANGUAGES),
            )
            self.ocr_reader = easyocr.Reader(
                list(config.ANPR_OCR_LANGUAGES),
                gpu=False,
                verbose=False,
            )
            self._anpr_metrics["status"] = "ready"
        self.focus_service.start()
        if self.tile_worker is not None:
            self.tile_worker.start()

    @staticmethod
    def _verify_model_file(
        path_setting: str, expected_hash: str, label: str, *, required: bool
    ) -> None:
        model_path = Path(path_setting)
        if not model_path.is_file():
            if required:
                raise RuntimeError(
                    f"{label} is missing at {model_path}; run "
                    "python scripts/download_anpr_model.py to review its terms"
                )
            return
        if not expected_hash:
            if required:
                raise RuntimeError(f"{label} requires a configured SHA-256 digest: {model_path}")
            LOGGER.warning("%s checksum verification is disabled for %s", label, model_path)
            return
        checksum = hashlib.sha256()
        with model_path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                checksum.update(chunk)
        if checksum.hexdigest() != expected_hash:
            raise RuntimeError(
                f"{label} checksum mismatch for {model_path}; refuse to load unverified weights"
            )

    def track_frame(self, frame) -> list[dict]:
        """Run authoritative ByteTrack, then optional proposal/focus work.

        Only full-frame detections with IDs from the explicitly configured tracker
        are authoritative. Tile and focus observations retain source coordinates
        but never receive synthetic persistent IDs.
        """
        self._frame_number += 1
        focus_lock = getattr(self, "_focus_lock", None)
        if focus_lock is None:  # Supports lightweight detector test doubles created with __new__.
            boost_remaining = getattr(self, "_focus_boost_remaining", 0)
            if boost_remaining > 0:
                self._focus_boost_remaining = boost_remaining - 1
        else:
            with focus_lock:
                cutoff = self._frame_number - config.NATIVE_FOCUS_MAX_GAP_FRAMES
                self._acquired_focus_episodes = {
                    episode_id: acquired_frame
                    for episode_id, acquired_frame in self._acquired_focus_episodes.items()
                    if acquired_frame >= cutoff
                }
                boost_remaining = self._focus_boost_remaining
                if boost_remaining > 0:
                    self._focus_boost_remaining = boost_remaining - 1
        boost_active = boost_remaining > 0
        input_size = (
            config.NATIVE_FOCUS_TRACK_INPUT_SIZE if boost_active else config.YOLO_INPUT_SIZE
        )

        results = self.model.track(
            frame,
            persist=True,
            tracker=config.YOLO_TRACKER,
            imgsz=input_size,
            device=config.YOLO_DEVICE,
            conf=config.CONFIDENCE_THRESHOLD,
            classes=[config.PERSON_CLASS_ID, *config.VEHICLE_CLASS_IDS],
            verbose=False,
        )
        detections = self._result_detections(results[0] if results else None, include_tracks=True)
        tile_due = self._frame_number % config.TILED_INFERENCE_INTERVAL_FRAMES == 0
        if self.tile_model is not None and tile_due:
            try:
                tiled = self._tiled_detections(frame)
            except Exception:
                if not self._tiled_error_active:
                    LOGGER.exception("tiled inference failed; full-frame tracking will continue")
                self._tiled_error_active = True
                self._update_tiled_snapshot(
                    status="error",
                    message="tiled inference failed; full-frame tracking continues",
                )
            else:
                self._tiled_error_active = False
                self._update_tiled_snapshot(
                    status="ready",
                    last_frame_number=self._frame_number,
                    last_proposals=len(tiled),
                    message=None,
                )
                detections = self._merge_supplemental(detections, tiled)
        elif getattr(self, "tile_worker", None) is not None and tile_due:
            # The one-slot worker drops instead of stalling the live feed.
            self.tile_worker.submit(frame, self._frame_number)

        self._record_detection_diagnostics(detections, input_size, boost_active)
        self._match_authoritative_acquisition(detections)
        focus_service = getattr(self, "focus_service", None)
        if (
            focus_service is not None
            and self._frame_number % config.NATIVE_FOCUS_INTERVAL_FRAMES == 0
        ):
            candidates = [detection for detection in detections if detection["track_id"] is None]
            focus_service.submit(frame, candidates, self._frame_number)
        return detections

    def _update_tiled_snapshot(self, **values: Any) -> None:
        snapshot = getattr(self, "_tiled_snapshot", None)
        if snapshot is None:  # Supports lightweight detector test doubles.
            snapshot = {}
            self._tiled_snapshot = snapshot
        snapshot.update(values)

    def _process_tiled_job(self, model, frame) -> list[dict]:
        return self._tiled_detections(
            frame,
            model=model,
            device=config.TILED_INFERENCE_DEVICE,
        )

    def _handle_tiled_result(self, frame, detections: list[dict], frame_number: int) -> None:
        """Route exact-frame tile proposals to focus without overlaying stale boxes."""
        if not detections:
            return
        focus_service = getattr(self, "focus_service", None)
        if focus_service is not None:
            focus_service.submit(frame, detections, frame_number)

    @staticmethod
    def tile_bounds(
        frame_width: int,
        frame_height: int,
        tile_size: int,
        overlap: float,
    ) -> tuple[tuple[int, int, int, int], ...]:
        """Cover a frame with overlapping edge-aligned tiles."""
        step = max(1, int(tile_size * (1.0 - overlap)))

        def starts(length: int) -> list[int]:
            if length <= tile_size:
                return [0]
            values = list(range(0, max(1, length - tile_size + 1), step))
            final = length - tile_size
            if values[-1] != final:
                values.append(final)
            return values

        return tuple(
            (x, y, min(frame_width, x + tile_size), min(frame_height, y + tile_size))
            for y in starts(frame_height)
            for x in starts(frame_width)
        )

    def _tiled_detections(self, frame, *, model=None, device: str | None = None) -> list[dict]:
        height, width = frame.shape[:2]
        bounds = self.tile_bounds(width, height, config.TILE_SIZE, config.TILE_OVERLAP)
        if bounds == ((0, 0, width, height),):
            # A full-frame "tile" has no magnification benefit and only repeats
            # the authoritative detector, producing duplicate untracked boxes.
            return []
        maximum = config.TILED_INFERENCE_MAX_TILES_PER_CYCLE
        bound_count = len(bounds)
        if bound_count > maximum:
            start = getattr(self, "_tile_cursor", 0) % bound_count
            bounds = tuple(bounds[(start + offset) % bound_count] for offset in range(maximum))
            self._tile_cursor = (start + maximum) % bound_count
        predictor = model if model is not None else self.tile_model
        if predictor is None:
            return []
        selected_device = device or config.TILED_INFERENCE_DEVICE
        detections = []
        for left, top, right, bottom in bounds:
            tile = frame[top:bottom, left:right]
            results = predictor.predict(
                tile,
                conf=config.TILED_INFERENCE_CONFIDENCE_THRESHOLD,
                device=selected_device,
                classes=[config.PERSON_CLASS_ID, *config.VEHICLE_CLASS_IDS],
                verbose=False,
            )
            mapped = self._result_detections(results[0] if results else None, include_tracks=False)
            for detection in mapped:
                x1, y1, x2, y2 = detection["bbox"]
                detection["bbox"] = (x1 + left, y1 + top, x2 + left, y2 + top)
                detection["centroid"] = (
                    detection["centroid"][0] + left,
                    detection["centroid"][1] + top,
                )
                detections.append(detection)
        return self._nms(detections, config.TILED_INFERENCE_NMS_IOU)

    @staticmethod
    def _result_detections(result, *, include_tracks: bool) -> list[dict]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        coordinates = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        raw_ids = getattr(boxes, "id", None) if include_tracks else None
        track_ids = raw_ids.cpu().numpy() if raw_ids is not None else [None] * len(coordinates)
        detections = []
        for box, raw_track_id, raw_class_id, confidence in zip(
            coordinates, track_ids, class_ids, confidences, strict=True
        ):
            x1, y1, x2, y2 = (float(value) for value in box)
            class_id = int(raw_class_id)
            detections.append(
                {
                    "track_id": None if raw_track_id is None else int(raw_track_id),
                    "class_id": class_id,
                    "class_name": (
                        "person"
                        if class_id == config.PERSON_CLASS_ID
                        else config.VEHICLE_CLASS_IDS.get(class_id, f"class_{class_id}")
                    ),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "centroid": ((x1 + x2) / 2, (y1 + y2) / 2),
                    "conf": float(confidence),
                    "source": "full-frame" if include_tracks else "tile",
                }
            )
        return detections

    @classmethod
    def _nms(cls, detections: list[dict], threshold: float) -> list[dict]:
        kept = []
        for candidate in sorted(detections, key=lambda item: item["conf"], reverse=True):
            if any(
                candidate["class_id"] == existing["class_id"]
                and cls._iou(candidate["bbox"], existing["bbox"]) > threshold
                for existing in kept
            ):
                continue
            kept.append(candidate)
        return kept

    @classmethod
    def _merge_supplemental(cls, tracked: list[dict], tiled: list[dict]) -> list[dict]:
        merged = list(tracked)
        for candidate in tiled:
            if any(
                candidate["class_id"] == existing["class_id"]
                and cls._iou(candidate["bbox"], existing["bbox"]) > config.TILED_INFERENCE_NMS_IOU
                for existing in merged
            ):
                continue
            merged.append(candidate)
        return merged

    @staticmethod
    def _iou(first, second) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    def _record_detection_diagnostics(
        self, detections: list[dict], input_size: int, boost_active: bool
    ) -> None:
        tracked = [item for item in detections if item["track_id"] is not None]
        untracked_full = [
            item
            for item in detections
            if item["track_id"] is None and item["source"] == "full-frame"
        ]
        untracked_tiles = [item for item in detections if item["source"] == "tile"]
        tracked_people = sum(item["class_id"] == config.PERSON_CLASS_ID for item in tracked)
        untracked_people = sum(
            item["class_id"] == config.PERSON_CLASS_ID
            for item in (*untracked_full, *untracked_tiles)
        )
        if tracked_people:
            context_blocker = None
        elif untracked_people:
            context_blocker = "authoritative_tracker_not_acquired"
        else:
            context_blocker = "no_person_detection"
        self._detection_diagnostics = {
            "tracker": config.YOLO_TRACKER,
            "configured_input_size": config.YOLO_INPUT_SIZE,
            "active_input_size": input_size,
            "tracking_boost_active": boost_active,
            "tracking_boost_frames_remaining": getattr(self, "_focus_boost_remaining", 0),
            "tracked_full_frame": len(tracked),
            "untracked_full_frame": len(untracked_full),
            "untracked_tile": len(untracked_tiles),
            "tracked_people": tracked_people,
            "untracked_people": untracked_people,
            "total": len(detections),
            "context_blocker": context_blocker,
            "provenance": {
                "authoritative": "full-frame ByteTrack only",
                "supplemental": "tile/focus candidates have no persistent ID",
            },
        }

    def _handle_focus_snapshot(self, snapshot: dict[str, Any]) -> None:
        focus_lock = getattr(self, "_focus_lock", None)
        with focus_lock if focus_lock is not None else nullcontext():
            payload = dict(snapshot)
            focus = dict(payload["focus"]) if payload.get("focus") else None
            payload["focus"] = focus
            if focus:
                episode_id = focus.get("candidate_episode_id")
                if episode_id in getattr(self, "_acquired_focus_episodes", {}):
                    focus["acquisition_state"] = "tracker_acquired"
                result_age = self._frame_number - int(payload.get("frame_number") or 0)
                if focus.get("newly_confirmed"):
                    if result_age <= config.NATIVE_FOCUS_RESULT_MAX_AGE_FRAMES:
                        self._focus_boost_remaining = max(
                            getattr(self, "_focus_boost_remaining", 0),
                            config.NATIVE_FOCUS_TRACK_BOOST_FRAMES,
                        )
                    else:
                        focus["acquisition_state"] = "expired_before_tracker_request"
            self._focus_snapshot = payload
            published = self.focus_snapshot()
        if self._on_focus_snapshot is not None:
            try:
                self._on_focus_snapshot(published)
            except Exception:
                LOGGER.exception("Focus snapshot callback failed; tracking continues")

    def _match_authoritative_acquisition(self, detections: list[dict]) -> None:
        focus_lock = getattr(self, "_focus_lock", None)
        with focus_lock if focus_lock is not None else nullcontext():
            snapshot = getattr(self, "_focus_snapshot", {})
            focus = snapshot.get("focus")
            if not focus or not focus.get("confirmed"):
                return
            episode_id = focus.get("candidate_episode_id")
            if not isinstance(episode_id, int):
                return
            if episode_id in getattr(self, "_acquired_focus_episodes", {}):
                return
            matches = [
                detection
                for detection in detections
                if detection["track_id"] is not None
                and detection["source"] == "full-frame"
                and detection["class_id"] == focus["class_id"]
                and self._iou(detection["bbox"], focus["bbox"])
                >= config.NATIVE_FOCUS_ACQUISITION_IOU
            ]
            if not matches:
                return
            acquired = max(matches, key=lambda item: self._iou(item["bbox"], focus["bbox"]))
            self._acquired_focus_episodes[episode_id] = self._frame_number
            focus["acquisition_state"] = "tracker_acquired"
            focus["authoritative_track_id"] = acquired["track_id"]
            self._focus_boost_remaining = 0
            published = self.focus_snapshot()
        if self._on_focus_snapshot is not None:
            try:
                self._on_focus_snapshot(published)
            except Exception:
                LOGGER.exception("Focus acquisition callback failed; tracking continues")

    def detection_diagnostics(self) -> dict[str, Any]:
        diagnostics = dict(getattr(self, "_detection_diagnostics", {}))
        diagnostics["focus"] = self.focus_snapshot(include_image=False)
        tile_worker = getattr(self, "tile_worker", None)
        diagnostics["tiled_inference"] = (
            tile_worker.snapshot()
            if tile_worker is not None
            else dict(
                getattr(
                    self,
                    "_tiled_snapshot",
                    {
                        "status": "disabled",
                        "mode": "disabled",
                        "last_proposals": 0,
                    },
                )
            )
        )
        return diagnostics

    def focus_snapshot(self, *, include_image: bool = True) -> dict[str, Any]:
        focus_lock = getattr(self, "_focus_lock", None)
        with focus_lock if focus_lock is not None else nullcontext():
            payload = dict(getattr(self, "_focus_snapshot", {}))
            focus = dict(payload["focus"]) if payload.get("focus") else None
            if focus is not None and not include_image:
                focus.pop("image_jpeg", None)
            payload["focus"] = focus
            return payload

    def reset_tracking(self) -> None:
        """Clear ByteTrack and ephemeral focus state after a source discontinuity."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()) or ():
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()
        with self._focus_lock:
            self._focus_boost_remaining = 0
            self._acquired_focus_episodes.clear()
        focus_service = getattr(self, "focus_service", None)
        if focus_service is not None:
            focus_service.reset()
        tile_worker = getattr(self, "tile_worker", None)
        if tile_worker is not None:
            tile_worker.reset()
        self._tile_cursor = 0

    def close(self) -> None:
        tile_worker = getattr(self, "tile_worker", None)
        if tile_worker is not None:
            tile_worker.close()
        focus_service = getattr(self, "focus_service", None)
        if focus_service is not None:
            focus_service.close()

    def anpr_diagnostics(self) -> dict:
        """Return privacy-safe counters explaining ANPR abstention."""
        metrics = dict(self._anpr_metrics)
        metrics.update(
            {
                "regions": list(config.ANPR_REGIONS),
                "minimum_plate_width_px": config.ANPR_MIN_PLATE_WIDTH_PX,
                "minimum_sharpness": config.ANPR_MIN_PLATE_SHARPNESS,
                "minimum_ocr_confidence": config.ANPR_MIN_OCR_CONFIDENCE,
                "consensus_reads": config.ANPR_MIN_CONSENSUS_READS,
            }
        )
        return metrics

    def _set_anpr_result(self, result: str) -> None:
        self._anpr_metrics["last_result"] = result
        if config.ANPR_DEBUG:
            LOGGER.info(
                "ANPR diagnostic: result=%s attempts=%s candidates=%s valid=%s",
                result,
                self._anpr_metrics["attempts"],
                self._anpr_metrics["detector_candidates"],
                self._anpr_metrics["valid_observations"],
            )

    def detect_plate_observation(self, frame, vehicle_bbox) -> PlateObservation | None:
        """Detect and OCR the strongest valid regional plate in one vehicle crop."""
        if self.plate_model is None or self.ocr_reader is None:
            return None
        self._anpr_metrics["attempts"] += 1
        vehicle_crop, offset = self._crop(frame, vehicle_bbox)
        if vehicle_crop is None:
            self._set_anpr_result("invalid_vehicle_crop")
            return None

        try:
            result = self.plate_model.predict(
                vehicle_crop,
                conf=config.ANPR_PLATE_DETECTION_CONFIDENCE,
                imgsz=config.ANPR_PLATE_INPUT_SIZE,
                device=config.YOLO_DEVICE,
                verbose=False,
            )[0]
            observations = self._plate_observations(vehicle_crop, offset, result)
        except Exception:
            self._anpr_metrics["inference_errors"] += 1
            self._set_anpr_result("inference_error")
            if not self._anpr_inference_error_active:
                LOGGER.exception("regional ANPR inference failed; primary analytics will continue")
                self._anpr_inference_error_active = True
            return None

        self._anpr_inference_error_active = False
        if not observations:
            return None
        observation = max(observations, key=lambda item: item.quality)
        self._anpr_metrics["valid_observations"] += 1
        self._set_anpr_result("valid_observation")
        return observation

    def detect_plate(self, frame, vehicle_bbox):
        """Backward-compatible ``(text, bbox)`` wrapper around regional ANPR."""
        observation = self.detect_plate_observation(frame, vehicle_bbox)
        if observation is None:
            return None, None
        return observation.text, observation.bbox

    def _plate_observations(self, vehicle_crop, offset, result) -> list[PlateObservation]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            self._set_anpr_result("no_plate_candidate")
            return []
        coordinates = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        candidates = sorted(
            zip(coordinates, confidences, strict=True),
            key=lambda candidate: float(candidate[1]),
            reverse=True,
        )[: config.ANPR_MAX_PLATE_CANDIDATES]
        self._anpr_metrics["detector_candidates"] += len(candidates)
        if not candidates:
            self._set_anpr_result("no_plate_candidate")

        observations = []
        crop_height, crop_width = vehicle_crop.shape[:2]
        x_offset, y_offset = offset
        for box, detector_confidence in candidates:
            raw_x1, raw_y1, raw_x2, raw_y2 = map(float, box)
            x1 = max(0, min(crop_width, int(raw_x1)))
            y1 = max(0, min(crop_height, int(raw_y1)))
            x2 = max(0, min(crop_width, int(raw_x2)))
            y2 = max(0, min(crop_height, int(raw_y2)))
            if x2 - x1 < config.ANPR_MIN_PLATE_WIDTH_PX or y2 <= y1:
                self._anpr_metrics["rejected_small"] += 1
                self._set_anpr_result("plate_too_small")
                continue

            pad_x = max(2, int((x2 - x1) * 0.04))
            pad_y = max(2, int((y2 - y1) * 0.10))
            crop_x1, crop_y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            crop_x2, crop_y2 = min(crop_width, x2 + pad_x), min(crop_height, y2 + pad_y)
            plate_crop = vehicle_crop[crop_y1:crop_y2, crop_x1:crop_x2]
            if plate_crop.size == 0:
                self._set_anpr_result("empty_plate_crop")
                continue

            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if sharpness < config.ANPR_MIN_PLATE_SHARPNESS:
                self._anpr_metrics["rejected_blurry"] += 1
                self._set_anpr_result("plate_too_blurry")
                continue
            read = self._read_plate_variants(plate_crop)
            if read is None:
                continue
            text, ocr_confidence, region = read
            observations.append(
                PlateObservation(
                    text=text,
                    bbox=(x_offset + x1, y_offset + y1, x_offset + x2, y_offset + y2),
                    region=region,
                    ocr_confidence=ocr_confidence,
                    detector_confidence=float(detector_confidence),
                    sharpness=sharpness,
                )
            )
        return observations

    def _read_plate_variants(self, plate_crop) -> tuple[str, float, str] | None:
        best = None
        saw_text = False
        saw_valid_region = False
        allowlist = (
            _LATIN_OCR_ALLOWLIST if set(config.ANPR_REGIONS).issubset(_LATIN_ANPR_REGIONS) else None
        )
        for image in self._plate_variants(plate_crop):
            kwargs = {"detail": 1, "paragraph": False}
            if allowlist:
                kwargs["allowlist"] = allowlist
            results = self.ocr_reader.readtext(image, **kwargs)
            candidate = combine_ocr_results(results)
            if candidate is None:
                continue
            saw_text = True
            text, confidence = candidate
            region = identify_plate_region(text, config.ANPR_REGIONS)
            if config.ANPR_DEBUG:
                LOGGER.info(
                    "ANPR OCR diagnostic: text=%r confidence=%.3f matched_region=%s",
                    text,
                    confidence,
                    region,
                )
            if region is None:
                continue
            saw_valid_region = True
            if confidence < config.ANPR_MIN_OCR_CONFIDENCE:
                continue
            if best is None or confidence > best[1]:
                best = (text, confidence, region)

        if best is not None:
            return best
        if not saw_text:
            reason = "ocr_no_text"
        elif not saw_valid_region:
            reason = "region_rejected"
        else:
            reason = "ocr_low_confidence"
        self._anpr_metrics[reason] += 1
        self._set_anpr_result(reason)
        return None

    @classmethod
    def _plate_variants(cls, plate_crop) -> tuple:
        crops = [plate_crop]
        rectified = cls._rectify_plate(plate_crop)
        if rectified is not None:
            crops.append(rectified)

        variants = []
        for crop in crops:
            height = crop.shape[0]
            scale = max(1.0, config.ANPR_OCR_TARGET_HEIGHT_PX / max(height, 1))
            if scale > 1.0:
                crop = cv2.resize(
                    crop,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            variants.extend((crop, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)))
        return tuple(variants)

    @staticmethod
    def _rectify_plate(plate_crop):
        """Return a conservative four-corner perspective correction when detectable."""
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        image_area = plate_crop.shape[0] * plate_crop.shape[1]
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            if cv2.contourArea(contour) < image_area * 0.2:
                break
            perimeter = cv2.arcLength(contour, True)
            corners = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            if len(corners) != 4 or not cv2.isContourConvex(corners):
                continue

            points = corners.reshape(4, 2).astype(np.float32)
            coordinate_sum = points.sum(axis=1)
            coordinate_difference = np.diff(points, axis=1).reshape(-1)
            ordered = np.zeros((4, 2), dtype=np.float32)
            ordered[0] = points[np.argmin(coordinate_sum)]  # top-left
            ordered[2] = points[np.argmax(coordinate_sum)]  # bottom-right
            ordered[1] = points[np.argmin(coordinate_difference)]  # top-right
            ordered[3] = points[np.argmax(coordinate_difference)]  # bottom-left

            top_left, top_right, bottom_right, bottom_left = ordered
            width = int(
                max(
                    np.linalg.norm(bottom_right - bottom_left),
                    np.linalg.norm(top_right - top_left),
                )
            )
            height = int(
                max(
                    np.linalg.norm(top_right - bottom_right),
                    np.linalg.norm(top_left - bottom_left),
                )
            )
            aspect_ratio = width / max(height, 1)
            if width < 16 or height < 8 or not 1.0 <= aspect_ratio <= 10.0:
                continue

            target = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype=np.float32,
            )
            transform = cv2.getPerspectiveTransform(ordered, target)
            return cv2.warpPerspective(plate_crop, transform, (width, height))
        return None

    @staticmethod
    def _crop(frame, bbox):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None, None
        crop = frame[y1:y2, x1:x2]
        return (crop, (x1, y1)) if crop.size else (None, None)
