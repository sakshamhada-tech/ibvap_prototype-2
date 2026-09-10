"""Optional native-resolution focus analysis for small, untracked candidates.

This module deliberately does not create persistent identities. It re-runs the existing
reviewed object model on padded crops copied from the untouched source frame, confirms
candidate episodes over time, and reports evidence to the authoritative full-frame
tracker. Browser enlargement of the returned JPEG is presentation only; no pixels are
synthesized here.
"""

from __future__ import annotations

import hashlib
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config
from utils.evaluation import target_pixel_height_band

LOGGER = logging.getLogger(__name__)


@dataclass
class _CandidateEpisode:
    episode_id: int
    class_id: int
    bbox: tuple[int, int, int, int]
    observations: int
    last_frame: int


class FocusConfirmationTracker:
    """Confirm spatial candidate episodes without assigning object/track identities."""

    def __init__(self, required_observations: int, iou_threshold: float, max_gap_frames: int):
        self.required_observations = required_observations
        self.iou_threshold = iou_threshold
        self.max_gap_frames = max_gap_frames
        self._episodes: dict[int, _CandidateEpisode] = {}
        self._next_episode_id = 1

    @staticmethod
    def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    def update(
        self, detections: Iterable[dict[str, Any]], frame_number: int
    ) -> list[dict[str, Any]]:
        self.expire(frame_number)
        matched_episodes: set[int] = set()
        confirmed: list[dict[str, Any]] = []
        for detection in sorted(detections, key=lambda item: item["conf"], reverse=True):
            best: _CandidateEpisode | None = None
            best_iou = self.iou_threshold
            for episode in self._episodes.values():
                if (
                    episode.episode_id in matched_episodes
                    or episode.class_id != detection["class_id"]
                ):
                    continue
                overlap = self._iou(episode.bbox, detection["bbox"])
                if overlap >= best_iou:
                    best = episode
                    best_iou = overlap

            if best is None:
                best = _CandidateEpisode(
                    episode_id=self._next_episode_id,
                    class_id=detection["class_id"],
                    bbox=detection["bbox"],
                    observations=1,
                    last_frame=frame_number,
                )
                self._episodes[best.episode_id] = best
                self._next_episode_id += 1
            else:
                best.bbox = detection["bbox"]
                best.observations += 1
                best.last_frame = frame_number

            matched_episodes.add(best.episode_id)
            item = dict(detection)
            item.update(
                {
                    "candidate_episode_id": best.episode_id,
                    "confirmation_count": best.observations,
                    "confirmation_required": self.required_observations,
                    "confirmed": best.observations >= self.required_observations,
                    "newly_confirmed": best.observations == self.required_observations,
                }
            )
            confirmed.append(item)
        return confirmed

    def expire(self, frame_number: int) -> None:
        self._episodes = {
            episode_id: episode
            for episode_id, episode in self._episodes.items()
            if frame_number - episode.last_frame <= self.max_gap_frames
        }

    def reset(self) -> None:
        self._episodes.clear()
        self._next_episode_id = 1

    @property
    def active_count(self) -> int:
        return len(self._episodes)


def expand_focus_bbox(
    bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    padding_ratio: float,
    minimum_crop_size: int,
) -> tuple[int, int, int, int]:
    """Return a clamped, padded source-coordinate crop around a candidate."""

    x1, y1, x2, y2 = bbox
    target_width = max(1, x2 - x1)
    target_height = max(1, y2 - y1)
    crop_width = min(
        frame_width, max(minimum_crop_size, round(target_width * (1 + 2 * padding_ratio)))
    )
    crop_height = min(
        frame_height, max(minimum_crop_size, round(target_height * (1 + 2 * padding_ratio)))
    )
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    left = max(0, min(frame_width - crop_width, round(center_x - crop_width / 2)))
    top = max(0, min(frame_height - crop_height, round(center_y - crop_height / 2)))
    return left, top, left + crop_width, top + crop_height


def focused_detection_from_result(
    result: Any,
    crop_bbox: tuple[int, int, int, int],
    candidate: dict[str, Any],
    minimum_proposal_iou: float = 0.0,
) -> dict[str, Any] | None:
    """Select the best matching model result and remap it to source coordinates."""

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    left, top, _right, _bottom = crop_bbox
    matches = []
    for box, class_id, confidence in zip(xyxy, class_ids, confidences, strict=True):
        if int(class_id) != int(candidate["class_id"]):
            continue
        x1, y1, x2, y2 = map(int, box)
        source_bbox = (left + x1, top + y1, left + x2, top + y2)
        proposal_iou = FocusConfirmationTracker._iou(source_bbox, candidate["bbox"])
        if proposal_iou < minimum_proposal_iou:
            continue
        matches.append(
            {
                "class_id": int(class_id),
                "class_name": candidate["class_name"],
                "conf": float(confidence),
                "bbox": source_bbox,
                "proposal_iou": proposal_iou,
                "source_candidate": candidate.get("source", "unknown"),
                "track_id": None,
            }
        )
    return max(matches, key=lambda item: (item["proposal_iou"], item["conf"]), default=None)


class NativeFocusService:
    """Bounded one-slot worker for source-pixel ROI re-inference."""

    _SENTINEL = object()

    def __init__(self, on_snapshot: Callable[[dict[str, Any]], None] | None = None):
        self.enabled = config.ENABLE_NATIVE_FOCUS
        self.on_snapshot = on_snapshot
        self._jobs: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._generation = 0
        self._model = None
        self._device = "cpu"
        self._accepting_jobs = False
        self._tracker = FocusConfirmationTracker(
            config.NATIVE_FOCUS_CONFIRMATION_FRAMES,
            config.NATIVE_FOCUS_CONFIRMATION_IOU,
            config.NATIVE_FOCUS_MAX_GAP_FRAMES,
        )
        self._snapshot: dict[str, Any] = {
            "status": "disabled" if not self.enabled else "starting",
            "message": None,
            "frame_number": None,
            "updated_monotonic": None,
            "submitted": 0,
            "processed": 0,
            "dropped_busy": 0,
            "errors": 0,
            "focus": None,
        }

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="ibvap-native-focus", daemon=True)
        self._thread.start()

    def submit(self, frame: Any, candidates: list[dict[str, Any]], frame_number: int) -> bool:
        if not self.enabled or not candidates or not self._accepting_jobs:
            return False
        if self._jobs.full():
            self._mutate_snapshot("dropped_busy", 1)
            return False
        try:
            source_frame = frame.copy()
            job = (
                self._generation,
                frame_number,
                source_frame,
                tuple(dict(item) for item in candidates),
            )
            self._jobs.put_nowait(job)
        except queue.Full:
            self._mutate_snapshot("dropped_busy", 1)
            return False
        except Exception as exc:  # pragma: no cover - defensive boundary for camera arrays
            self._record_error(f"focus frame copy failed: {exc}")
            return False
        self._mutate_snapshot("submitted", 1)
        return True

    def reset(self) -> None:
        with self._lock:
            self._generation += 1
            self._tracker.reset()
            self._snapshot.update(
                {
                    "status": "ready" if self._accepting_jobs else self._snapshot["status"],
                    "message": None,
                    "frame_number": None,
                    "updated_monotonic": None,
                    "focus": None,
                }
            )
        self._discard_pending_job()
        self._publish()

    def close(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._accepting_jobs = False
        self._discard_pending_job()
        with suppress(queue.Full):  # queue was cleared immediately above
            self._jobs.put_nowait(self._SENTINEL)
        self._thread.join(timeout=config.NATIVE_FOCUS_SHUTDOWN_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            self._record_error(
                "native focus worker did not stop before the shutdown timeout",
                status="shutdown_timeout",
            )
            return
        self._thread = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            payload = dict(self._snapshot)
            payload["focus"] = dict(payload["focus"]) if payload.get("focus") else None
            payload["active_candidate_episodes"] = self._tracker.active_count
            payload["device"] = self._device
            return payload

    def _run(self) -> None:
        try:
            self._device = self._select_device(config.NATIVE_FOCUS_DEVICE)
            self._model = self._load_model()
            self._accepting_jobs = True
            LOGGER.info("native focus ready on device=%s", self._device)
            self._set_status("ready", None)
        except Exception as exc:
            LOGGER.exception("Native focus initialization failed")
            self._record_error(f"native focus unavailable: {exc}", status="unavailable")
            return

        while not self._stop_event.is_set():
            try:
                job = self._jobs.get(timeout=config.NATIVE_FOCUS_QUEUE_POLL_SECONDS)
            except queue.Empty:
                continue
            if job is self._SENTINEL:
                break
            generation, frame_number, frame, candidates = job
            if generation != self._generation:
                continue
            try:
                focus = self._process(frame, candidates, frame_number, generation=generation)
                if generation != self._generation:
                    continue
                with self._lock:
                    self._snapshot.update(
                        {
                            "status": "ready",
                            "message": None,
                            "frame_number": frame_number,
                            "updated_monotonic": time.monotonic(),
                            "processed": self._snapshot["processed"] + 1,
                            "focus": focus,
                        }
                    )
                self._publish()
            except Exception as exc:
                LOGGER.exception("Native focus job failed")
                self._record_error(f"native focus job failed: {exc}")

    def _process(
        self,
        frame: Any,
        candidates: Iterable[dict[str, Any]],
        frame_number: int,
        *,
        generation: int | None = None,
    ) -> dict[str, Any] | None:
        frame_height, frame_width = frame.shape[:2]
        detections: list[dict[str, Any]] = []
        crops: dict[tuple[int, int, int, int], Any] = {}
        ordered = sorted(candidates, key=lambda item: item["conf"], reverse=True)
        for candidate in ordered[: config.NATIVE_FOCUS_MAX_REGIONS]:
            crop_bbox = expand_focus_bbox(
                candidate["bbox"],
                frame_width,
                frame_height,
                config.NATIVE_FOCUS_PADDING_RATIO,
                config.NATIVE_FOCUS_MIN_CROP_SIZE_PX,
            )
            left, top, right, bottom = crop_bbox
            crop = frame[top:bottom, left:right]
            if getattr(crop, "size", 0) == 0:
                continue
            kwargs: dict[str, Any] = {
                "verbose": False,
                "conf": config.NATIVE_FOCUS_CONF_THRESHOLD,
                "imgsz": config.NATIVE_FOCUS_MODEL_INPUT_SIZE,
                "classes": [int(candidate["class_id"])],
            }
            if self._device:
                kwargs["device"] = self._device
            results = self._model.predict(crop, **kwargs)
            detection = (
                focused_detection_from_result(
                    results[0],
                    crop_bbox,
                    candidate,
                    config.NATIVE_FOCUS_PROPOSAL_IOU,
                )
                if results
                else None
            )
            if detection is not None:
                detection["crop_bbox"] = crop_bbox
                detections.append(detection)
                crops[crop_bbox] = crop

        with self._lock:
            if generation is not None and generation != self._generation:
                return None
            observations = self._tracker.update(detections, frame_number)
        if not observations:
            return None
        focus = max(
            observations,
            key=lambda item: (item["newly_confirmed"], item["confirmed"], item["conf"]),
        )
        x1, y1, x2, y2 = focus["bbox"]
        target_height = max(0, y2 - y1)
        crop_bbox = focus.pop("crop_bbox")
        focus.update(
            {
                "source_frame_width": frame_width,
                "source_frame_height": frame_height,
                "source_target_width": max(0, x2 - x1),
                "source_target_height": target_height,
                "source_target_height_band": target_pixel_height_band(target_height),
                "source_crop_width": crop_bbox[2] - crop_bbox[0],
                "source_crop_height": crop_bbox[3] - crop_bbox[1],
                "model_input_size": config.NATIVE_FOCUS_MODEL_INPUT_SIZE,
                "acquisition_state": "tracker_requested" if focus["confirmed"] else "confirming",
                "image_jpeg": self._encode_source_crop(crops[crop_bbox]),
            }
        )
        return focus

    @staticmethod
    def _load_model() -> Any:
        model_path = Path(config.YOLO_MODEL_PATH)
        if not model_path.is_file():
            raise RuntimeError(f"reviewed object model is missing: {model_path}")
        if not config.YOLO_MODEL_SHA256:
            raise RuntimeError("native focus requires IBVAP_YOLO_MODEL_SHA256")
        actual = hashlib.sha256(model_path.read_bytes()).hexdigest().lower()
        if actual != config.YOLO_MODEL_SHA256:
            raise RuntimeError("reviewed object model checksum mismatch")
        from ultralytics import YOLO

        return YOLO(str(model_path))

    @staticmethod
    def _select_device(requested: str) -> str:
        if requested == "cpu":
            return "cpu"
        try:
            import torch
        except Exception:
            LOGGER.warning("Torch device probing failed; native focus will use CPU")
            return "cpu"
        cuda_available = bool(torch.cuda.is_available())
        mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        if requested in {"cuda", "cuda:0"}:
            if cuda_available:
                return "cuda:0"
            LOGGER.warning("CUDA was requested but is unavailable; native focus will use CPU")
            return "cpu"
        if requested == "mps":
            if mps_available:
                return "mps"
            LOGGER.warning("MPS was requested but is unavailable; native focus will use CPU")
            return "cpu"
        if cuda_available:
            return "cuda:0"
        if mps_available:
            LOGGER.info(
                "MPS is available but native-focus auto mode uses CPU for macOS stability; "
                "request mps explicitly to opt in"
            )
        return "cpu"

    @staticmethod
    def _encode_source_crop(crop: Any) -> bytes | None:
        import cv2

        success, encoded = cv2.imencode(
            ".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), config.NATIVE_FOCUS_JPEG_QUALITY]
        )
        return encoded.tobytes() if success else None

    def _set_status(self, status: str, message: str | None) -> None:
        with self._lock:
            self._snapshot.update({"status": status, "message": message})
        self._publish()

    def _record_error(self, message: str, *, status: str = "degraded") -> None:
        with self._lock:
            self._snapshot.update(
                {
                    "status": status,
                    "message": message,
                    "errors": self._snapshot["errors"] + 1,
                }
            )
        self._publish()

    def _mutate_snapshot(self, key: str, increment: int) -> None:
        with self._lock:
            self._snapshot[key] += increment

    def _publish(self) -> None:
        if self.on_snapshot is None:
            return
        try:
            self.on_snapshot(self.snapshot())
        except Exception:
            LOGGER.exception("Native focus status callback failed")

    def _discard_pending_job(self) -> None:
        while True:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                return
