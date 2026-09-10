"""Failure-isolated optional firearm detection for human review.

Typical firearms occupy very few pixels in wide-area border footage. Even a
high-confidence model result can be wrong; this module never authorizes an
automated response and deliberately defaults to disabled.
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
SnapshotCallback = Callable[[dict], None]


class FirearmModelUnavailable(RuntimeError):
    """Raised internally when optional firearm weights/runtime are unavailable."""


@dataclass(frozen=True)
class _FirearmJob:
    frame: object
    frame_number: int
    generation: int


class FirearmDetectionService:
    """Run a separate fine-tuned YOLO model on a bounded background worker."""

    def __init__(
        self,
        *,
        model_path: str,
        model_sha256: str,
        confidence_threshold: float,
        device: str,
        on_snapshot: SnapshotCallback,
    ):
        self.model_path = model_path
        self.model_sha256 = model_sha256
        self.confidence_threshold = confidence_threshold
        self.device_setting = device
        self.on_snapshot = on_snapshot
        self._jobs: queue.Queue[_FirearmJob] = queue.Queue(maxsize=1)
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
        self._thread = threading.Thread(target=self._run, name="ibvap-firearm", daemon=True)
        self._thread.start()

    def submit(self, frame, frame_number: int) -> bool:
        if not self._accepting_jobs.is_set() or self._stop_event.is_set():
            return False
        with self._generation_lock:
            generation = self._generation
        try:
            self._jobs.put_nowait(_FirearmJob(frame.copy(), frame_number, generation))
        except queue.Full:
            return False
        return True

    def reset(self) -> None:
        with self._generation_lock:
            self._generation += 1
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
            LOGGER.error("firearm worker did not stop before timeout")

    def _run(self) -> None:
        try:
            self._load_model()
        except FirearmModelUnavailable as exc:
            LOGGER.error("firearm detection unavailable: %s", exc)
            self._publish("unavailable", (), None, str(exc))
            return
        except Exception:
            LOGGER.exception("firearm detector initialization failed")
            self._publish(
                "error", (), None, "firearm initialization failed; core analytics continue"
            )
            return
        self._accepting_jobs.set()
        self._publish("ready", (), None)
        while not self._stop_event.is_set():
            try:
                job = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                detections = self._process(job)
            except Exception:
                LOGGER.exception("firearm inference failed for frame %s", job.frame_number)
                self._publish("error", (), job.frame_number, "firearm inference failed")
                continue
            with self._generation_lock:
                current = job.generation == self._generation
            if current:
                self._publish("ready", detections, job.frame_number)

    def _load_model(self) -> None:
        path = Path(self.model_path)
        if not path.is_file():
            raise FirearmModelUnavailable(f"firearm model is missing at {path}")
        if not self.model_sha256:
            raise FirearmModelUnavailable("firearm model requires IBVAP_FIREARM_MODEL_SHA256")
        if _sha256(path) != self.model_sha256:
            raise FirearmModelUnavailable("firearm model checksum mismatch")
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise FirearmModelUnavailable("install the core Ultralytics/PyTorch runtime") from exc
        self._device = _select_device(torch, self.device_setting)
        self._model = YOLO(str(path))

    def _process(self, job: _FirearmJob) -> list[dict]:
        result = self._model.predict(
            job.frame,
            conf=self.confidence_threshold,
            device=self._device,
            verbose=False,
        )[0]
        return parse_firearm_result(result, self.confidence_threshold)

    def _publish(
        self,
        status: str,
        detections: Sequence[dict],
        frame_number: int | None,
        message: str | None = None,
    ) -> None:
        self._status = status
        snapshot = {
            "status": status,
            "detections": list(detections),
            "frame_number": frame_number,
            "updated_monotonic": time.monotonic(),
            "message": message,
        }
        try:
            self.on_snapshot(snapshot)
        except Exception:
            LOGGER.exception("firearm snapshot callback failed")


def parse_firearm_result(result, threshold: float) -> list[dict]:
    """Convert a YOLO result while enforcing the conservative threshold again."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    coordinates = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    detections = []
    for box, confidence in zip(coordinates, confidences, strict=True):
        confidence = float(confidence)
        if confidence < threshold:
            continue
        detections.append(
            {
                "bbox": tuple(int(value) for value in box),
                "confidence": confidence,
            }
        )
    return detections


def _select_device(torch, setting: str) -> str:
    if setting == "cpu":
        return "cpu"
    if setting == "cuda":
        if not torch.cuda.is_available():
            raise FirearmModelUnavailable("CUDA was requested but is unavailable")
        return "cuda"
    if setting == "mps":
        if not torch.backends.mps.is_available():
            raise FirearmModelUnavailable("MPS was requested but is unavailable")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model:
        for chunk in iter(lambda: model.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
