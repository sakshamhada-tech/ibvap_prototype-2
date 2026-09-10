"""Failure-isolated pose-based low-posture review signals.

Low-posture classification is meaningful only when a person occupies enough
pixels for stable keypoints. It can confuse crawling with falls, sitting,
bending, foreshortening, and occlusion; every result is a human-review signal.
"""

from __future__ import annotations

import hashlib
import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)
PersonBounds = tuple[int, tuple[int, int, int, int]]
SnapshotCallback = Callable[[dict], None]


class PostureModelsUnavailable(RuntimeError):
    """Raised internally when optional pose dependencies/weights are unavailable."""


@dataclass(frozen=True)
class _PostureJob:
    frame: object
    people: tuple[PersonBounds, ...]
    frame_number: int
    source_time: float
    generation: int


class PostureStateTracker:
    """Require a low pose to persist before publishing a sustained signal."""

    def __init__(self, sustain_seconds: float, stale_after_seconds: float):
        self.sustain_seconds = sustain_seconds
        self.stale_after_seconds = stale_after_seconds
        self._low_since: dict[int, float] = {}
        self._last_seen: dict[int, float] = {}

    def update(self, track_id: int, low: bool, timestamp: float) -> bool:
        previous = self._last_seen.get(track_id)
        if previous is not None and (
            timestamp < previous or timestamp - previous > self.stale_after_seconds
        ):
            self._drop(track_id)
        self._last_seen[track_id] = timestamp
        if not low:
            self._low_since.pop(track_id, None)
            return False
        self._low_since.setdefault(track_id, timestamp)
        return timestamp - self._low_since[track_id] >= self.sustain_seconds

    def expire(self, timestamp: float) -> None:
        cutoff = timestamp - self.stale_after_seconds
        for track_id, last_seen in tuple(self._last_seen.items()):
            if last_seen < cutoff or last_seen > timestamp:
                self._drop(track_id)

    def _drop(self, track_id: int) -> None:
        self._last_seen.pop(track_id, None)
        self._low_since.pop(track_id, None)

    def reset(self) -> None:
        self._last_seen.clear()
        self._low_since.clear()

    @property
    def tracked_count(self) -> int:
        return len(self._last_seen)


def classify_low_posture(
    bbox: tuple[float, float, float, float],
    keypoints: Sequence[Sequence[float]],
    keypoint_confidences: Sequence[float],
    *,
    minimum_visible_keypoints: int,
    minimum_keypoint_confidence: float,
    minimum_aspect_ratio: float,
    maximum_vertical_spread_ratio: float,
) -> tuple[bool, float]:
    """Return a conservative low-posture flag and geometric evidence score."""
    left, top, right, bottom = bbox
    width, height = max(1.0, right - left), max(1.0, bottom - top)
    visible_y = [
        float(point[1])
        for point, confidence in zip(keypoints, keypoint_confidences, strict=False)
        if len(point) >= 2 and float(confidence) >= minimum_keypoint_confidence
    ]
    if len(visible_y) < minimum_visible_keypoints:
        return False, 0.0
    aspect_ratio = width / height
    vertical_spread_ratio = (max(visible_y) - min(visible_y)) / height
    aspect_evidence = min(1.0, aspect_ratio / minimum_aspect_ratio)
    spread_evidence = max(0.0, 1.0 - vertical_spread_ratio)
    evidence = min(aspect_evidence, spread_evidence)
    low = (
        aspect_ratio >= minimum_aspect_ratio
        and vertical_spread_ratio <= maximum_vertical_spread_ratio
    )
    return low, evidence if low else 0.0


class PostureAnalysisService:
    """Run optional pose inference off-thread with a bounded single-job queue."""

    def __init__(
        self,
        *,
        model_path: str,
        model_sha256: str,
        confidence_threshold: float,
        minimum_person_size_px: int,
        maximum_people: int,
        minimum_visible_keypoints: int,
        minimum_keypoint_confidence: float,
        minimum_aspect_ratio: float,
        maximum_vertical_spread_ratio: float,
        sustain_seconds: float,
        state_ttl_seconds: float,
        device: str,
        on_snapshot: SnapshotCallback,
    ):
        self.model_path = model_path
        self.model_sha256 = model_sha256
        self.confidence_threshold = confidence_threshold
        self.minimum_person_size_px = minimum_person_size_px
        self.maximum_people = maximum_people
        self.minimum_visible_keypoints = minimum_visible_keypoints
        self.minimum_keypoint_confidence = minimum_keypoint_confidence
        self.minimum_aspect_ratio = minimum_aspect_ratio
        self.maximum_vertical_spread_ratio = maximum_vertical_spread_ratio
        self.device_setting = device
        self.on_snapshot = on_snapshot
        self.state = PostureStateTracker(sustain_seconds, state_ttl_seconds)
        self._jobs: queue.Queue[_PostureJob] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._accepting_jobs = threading.Event()
        self._thread: threading.Thread | None = None
        self._generation_lock = threading.Lock()
        self._generation = 0
        self._model = None
        self._device = "cpu"
        self._status = "loading"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._publish("loading", (), None)
        self._thread = threading.Thread(target=self._run, name="ibvap-posture", daemon=True)
        self._thread.start()

    def submit(
        self,
        frame,
        people: Sequence[PersonBounds],
        frame_number: int,
        source_time: float,
    ) -> bool:
        if not self._accepting_jobs.is_set() or self._stop_event.is_set():
            return False
        eligible = tuple(
            (track_id, bounds)
            for track_id, bounds in people
            if min(bounds[2] - bounds[0], bounds[3] - bounds[1]) >= self.minimum_person_size_px
        )[: self.maximum_people]
        if not eligible:
            return False
        with self._generation_lock:
            generation = self._generation
        try:
            self._jobs.put_nowait(
                _PostureJob(frame.copy(), eligible, frame_number, source_time, generation)
            )
        except queue.Full:
            return False
        return True

    def reset(self) -> None:
        with self._generation_lock:
            self._generation += 1
        self.state.reset()
        while True:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                break
        self._publish("ready" if self._accepting_jobs.is_set() else self._status, (), None)

    def close(self) -> None:
        self._accepting_jobs.clear()
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        if self._thread and self._thread.is_alive():
            LOGGER.error("posture worker did not stop before timeout")

    def _run(self) -> None:
        try:
            self._load_model()
        except PostureModelsUnavailable as exc:
            LOGGER.error("posture analysis unavailable: %s", exc)
            self._publish("unavailable", (), None, str(exc))
            return
        except Exception:
            LOGGER.exception("posture analysis initialization failed")
            self._publish(
                "error", (), None, "posture initialization failed; core analytics continue"
            )
            return
        self._accepting_jobs.set()
        LOGGER.info("posture analysis ready on device=%s", self._device)
        self._publish("ready", (), None)
        while not self._stop_event.is_set():
            try:
                job = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                signals = self._process(job)
            except Exception:
                LOGGER.exception("posture analysis failed for frame %s", job.frame_number)
                self._publish("error", (), job.frame_number, "posture inference failed")
                continue
            with self._generation_lock:
                current = job.generation == self._generation
            if current:
                self._publish("ready", signals, job.frame_number)

    def _load_model(self) -> None:
        path = Path(self.model_path)
        if not path.is_file():
            raise PostureModelsUnavailable(f"pose model is missing at {path}")
        if not self.model_sha256:
            raise PostureModelsUnavailable("pose model requires IBVAP_POSTURE_MODEL_SHA256")
        if _sha256(path) != self.model_sha256:
            raise PostureModelsUnavailable("pose model checksum mismatch")
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise PostureModelsUnavailable("install the core Ultralytics/PyTorch runtime") from exc
        self._device = _select_device(torch, self.device_setting)
        self._model = YOLO(str(path))

    def _process(self, job: _PostureJob) -> list[dict]:
        signals = []
        for track_id, bounds in job.people:
            left, top, right, bottom = bounds
            crop = job.frame[top:bottom, left:right]
            if getattr(crop, "size", 0) == 0:
                continue
            result = self._model.predict(
                crop,
                conf=self.confidence_threshold,
                classes=[0],
                device=self._device,
                verbose=False,
            )[0]
            boxes = getattr(result, "boxes", None)
            keypoints = getattr(result, "keypoints", None)
            if boxes is None or keypoints is None or len(boxes) == 0:
                self.state.update(track_id, False, job.source_time)
                continue
            confidences = boxes.conf.cpu().numpy()
            index = max(range(len(confidences)), key=lambda item: float(confidences[item]))
            pose_box = tuple(float(value) for value in boxes.xyxy.cpu().numpy()[index])
            points = keypoints.xy.cpu().numpy()[index]
            point_confidences = keypoints.conf.cpu().numpy()[index]
            low, evidence = classify_low_posture(
                pose_box,
                points,
                point_confidences,
                minimum_visible_keypoints=self.minimum_visible_keypoints,
                minimum_keypoint_confidence=self.minimum_keypoint_confidence,
                minimum_aspect_ratio=self.minimum_aspect_ratio,
                maximum_vertical_spread_ratio=self.maximum_vertical_spread_ratio,
            )
            sustained = self.state.update(track_id, low, job.source_time)
            signals.append(
                {
                    "track_id": track_id,
                    "low_posture": low,
                    "sustained": sustained,
                    "confidence": round(float(confidences[index]) * evidence, 3),
                }
            )
        self.state.expire(job.source_time)
        return signals

    def _publish(
        self,
        status: str,
        signals: Sequence[dict],
        frame_number: int | None,
        message: str | None = None,
    ) -> None:
        self._status = status
        snapshot = {
            "status": status,
            "signals": list(signals),
            "frame_number": frame_number,
            "updated_monotonic": time.monotonic(),
            "message": message,
        }
        try:
            self.on_snapshot(snapshot)
        except Exception:
            LOGGER.exception("posture snapshot callback failed")


def _select_device(torch, setting: str) -> str:
    if setting == "cpu":
        return "cpu"
    if setting == "cuda":
        if not torch.cuda.is_available():
            raise PostureModelsUnavailable("CUDA was requested but is unavailable")
        return "cuda"
    if setting == "mps":
        if not torch.backends.mps.is_available():
            raise PostureModelsUnavailable("MPS was requested but is unavailable")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    # Background MPS inference has produced unrecoverable native Metal assertions
    # when combined with other macOS model workers. MPS therefore remains opt-in.
    return "cpu"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model:
        for chunk in iter(lambda: model.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
