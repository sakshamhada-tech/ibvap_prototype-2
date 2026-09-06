"""Failure-isolated SCRFD source extraction and optional GFPGAN preview.

The worker never mutates the source video frame. It emits camera-pixel and
aligned JPEG crops for an authenticated dashboard panel. Generative restoration
is opt-in, bounded, and dropped when work cannot keep up with capture.
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
_GFPGAN_TEMPLATE = (
    (192.98138, 239.94708),
    (318.90277, 240.19360),
    (256.63416, 314.01935),
    (201.26117, 371.41043),
    (313.08905, 371.15118),
)


class FaceModelsUnavailable(RuntimeError):
    """Raised when optional dependencies or model files are unavailable."""


@dataclass(frozen=True)
class _FaceJob:
    frame: object
    people: tuple[PersonBounds, ...]
    frame_number: int
    generation: int


class FaceEnhancementService:
    """Run SCRFD extraction and optional GFPGAN off-thread with a one-job queue."""

    def __init__(
        self,
        *,
        scrfd_model_path: str,
        gfpgan_model_path: str,
        gfpgan_model_sha256: str,
        enable_restoration: bool,
        confidence_threshold: float,
        input_size: int,
        min_face_size_px: int,
        restoration_min_face_size_px: int,
        restoration_min_sharpness: float,
        restoration_blend_weight: float,
        max_faces: int,
        refresh_seconds: float,
        cache_ttl_seconds: float,
        jpeg_quality: int,
        device: str,
        on_snapshot: SnapshotCallback,
    ):
        self.scrfd_model_path = scrfd_model_path
        self.gfpgan_model_path = gfpgan_model_path
        self.gfpgan_model_sha256 = gfpgan_model_sha256
        self.enable_restoration = enable_restoration
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.min_face_size_px = min_face_size_px
        self.restoration_min_face_size_px = restoration_min_face_size_px
        self.restoration_min_sharpness = restoration_min_sharpness
        self.restoration_blend_weight = restoration_blend_weight
        self.max_faces = max_faces
        self.refresh_seconds = refresh_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self.jpeg_quality = jpeg_quality
        self.device_setting = device
        self.on_snapshot = on_snapshot

        self._jobs: queue.Queue[_FaceJob] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._accepting_jobs = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = "loading"
        self._generation_lock = threading.Lock()
        self._generation = 0
        self._cache: dict[tuple[int, int], tuple[float, dict]] = {}
        self._scrfd = None
        self._gfpgan = None
        self._torch = None
        self._np = None
        self._cv2 = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._publish("loading", (), frame_number=None)
        self._thread = threading.Thread(
            target=self._run,
            name="ibvap-face-enhancement",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        frame,
        people: Sequence[PersonBounds],
        frame_number: int,
    ) -> bool:
        """Queue the latest raw frame or drop it when the worker is busy."""
        if not self._accepting_jobs.is_set() or self._stop_event.is_set():
            return False
        if not people:
            self.reset()
            return True
        with self._generation_lock:
            generation = self._generation
        try:
            self._jobs.put_nowait(
                _FaceJob(
                    frame=frame.copy(),
                    people=tuple(people),
                    frame_number=frame_number,
                    generation=generation,
                )
            )
        except queue.Full:
            return False
        return True

    def reset(self) -> None:
        """Invalidate queued work and cached track thumbnails after discontinuity."""
        with self._generation_lock:
            self._generation += 1
        while True:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                break
        status = "ready" if self._accepting_jobs.is_set() else self._status
        self._publish(status, (), frame_number=None)

    def close(self) -> None:
        self._accepting_jobs.clear()
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        if self._thread and self._thread.is_alive():
            LOGGER.error("face-enhancement worker did not stop before timeout")

    def _run(self) -> None:
        try:
            self._load_models()
        except FaceModelsUnavailable as exc:
            LOGGER.error("face enhancement unavailable: %s", exc)
            self._publish(
                "unavailable",
                (),
                frame_number=None,
                message="Face models or optional dependencies are not installed.",
            )
            return
        except Exception:
            LOGGER.exception("face enhancement initialization failed")
            self._publish(
                "error",
                (),
                frame_number=None,
                message="Face enhancement initialization failed; check server logs.",
            )
            return

        self._accepting_jobs.set()
        self._publish("ready", (), frame_number=None)
        while not self._stop_event.is_set():
            try:
                job = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                faces = self._process(job)
            except Exception:
                LOGGER.exception("face enhancement failed for frame %s", job.frame_number)
                self._publish(
                    "error",
                    (),
                    frame_number=job.frame_number,
                    message="Face enhancement failed; the primary pipeline is still running.",
                )
            else:
                with self._generation_lock:
                    is_current = job.generation == self._generation
                if is_current:
                    self._publish("ready", faces, frame_number=job.frame_number)

    def _load_models(self) -> None:
        scrfd_path = Path(self.scrfd_model_path)
        if not scrfd_path.is_file():
            raise FaceModelsUnavailable(
                "run scripts/download_face_models.py after reviewing the model licenses"
            )
        _verify_sidecar_hash(scrfd_path)

        try:
            import cv2
            import numpy as np
            import onnxruntime
            from insightface.model_zoo import get_model
        except ImportError as exc:
            raise FaceModelsUnavailable(
                "install requirements-face.lock in the active environment"
            ) from exc

        available_providers = set(onnxruntime.get_available_providers())
        providers = [
            provider
            for provider in (
                "CoreMLExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            )
            if provider in available_providers
        ]
        if self.device_setting == "cpu":
            providers = ["CPUExecutionProvider"]
        self._scrfd = get_model(str(scrfd_path), providers=providers)
        if self._scrfd is None or not getattr(self._scrfd, "use_kps", False):
            raise FaceModelsUnavailable("the configured SCRFD model must provide five landmarks")
        self._scrfd.prepare(
            ctx_id=0,
            input_size=(self.input_size, self.input_size),
            det_thresh=self.confidence_threshold,
        )

        self._cv2 = cv2
        self._np = np
        if not self.enable_restoration:
            LOGGER.info("face extraction ready: SCRFD providers=%s; GFPGAN disabled", providers)
            return

        gfpgan_path = Path(self.gfpgan_model_path)
        if not gfpgan_path.is_file():
            raise FaceModelsUnavailable(
                "GFPGAN preview enabled but its reviewed model file is unavailable"
            )
        _verify_hash(gfpgan_path, self.gfpgan_model_sha256)
        try:
            import torch
        except ImportError as exc:
            raise FaceModelsUnavailable("GFPGAN preview requires requirements-face.lock") from exc
        from third_party.gfpgan_arch import GFPGANv1Clean

        device = _select_torch_device(torch, self.device_setting)
        restorer = GFPGANv1Clean(
            out_size=512,
            num_style_feat=512,
            channel_multiplier=2,
            decoder_load_path=None,
            fix_decoder=False,
            num_mlp=8,
            input_is_latent=True,
            different_w=True,
            narrow=1,
            sft_half=True,
        )
        checkpoint = torch.load(
            str(gfpgan_path),
            map_location="cpu",
            weights_only=True,
        )
        parameters = checkpoint.get("params_ema") or checkpoint.get("params")
        if not isinstance(parameters, dict):
            raise FaceModelsUnavailable("GFPGAN checkpoint contains no model parameters")
        restorer.load_state_dict(parameters, strict=True)
        restorer.eval().to(device)

        self._torch = torch
        self._gfpgan = restorer
        self._device = device
        LOGGER.info(
            "face extraction ready: SCRFD providers=%s; experimental GFPGAN device=%s",
            providers,
            device,
        )

    def _process(self, job: _FaceJob) -> list[dict]:
        detections, landmarks = self._scrfd.detect(
            job.frame,
            input_size=(self.input_size, self.input_size),
            max_num=0,
        )
        if landmarks is None:
            return []

        candidates = []
        for detection, keypoints in zip(detections, landmarks, strict=True):
            left, top, right, bottom, confidence = (float(value) for value in detection[:5])
            source_size = min(right - left, bottom - top)
            if confidence < self.confidence_threshold or source_size < self.min_face_size_px:
                continue
            face_box = (left, top, right, bottom)
            track_id = _containing_track(face_box, job.people)
            if track_id is None:
                continue
            candidates.append(
                (
                    confidence,
                    (right - left) * (bottom - top),
                    track_id,
                    keypoints,
                    face_box,
                    source_size,
                )
            )
        candidates.sort(reverse=True, key=lambda candidate: (candidate[0], candidate[1]))

        now = time.monotonic()
        faces = []
        for confidence, _, track_id, keypoints, face_box, source_size in candidates[
            : self.max_faces
        ]:
            cache_key = (job.generation, track_id)
            cached = self._cache.get(cache_key)
            if cached is not None and now - cached[0] < self.refresh_seconds:
                face = dict(cached[1])
            else:
                try:
                    source_crop = _extract_source_face(job.frame, face_box)
                    source_jpeg = self._encode_jpeg(source_crop)
                    sharpness = _source_sharpness(job.frame, face_box, self._cv2)
                except Exception:
                    LOGGER.exception("could not extract source face for track %s", track_id)
                    continue
                if source_jpeg is None:
                    continue

                aligned = None
                aligned_jpeg = None
                review_jpeg = None
                reason = None
                restoration_status = "disabled"
                try:
                    aligned = _align_face(job.frame, keypoints, self._cv2, self._np)
                    aligned_jpeg = self._encode_jpeg(aligned)
                    if aligned_jpeg is None:
                        reason = "alignment_encoding_failed"
                except Exception:
                    LOGGER.exception("could not align face for track %s", track_id)
                    reason = "alignment_failed"

                if self.enable_restoration and aligned is not None and aligned_jpeg is not None:
                    reason = _restoration_rejection_reason(
                        source_size,
                        sharpness,
                        self.restoration_min_face_size_px,
                        self.restoration_min_sharpness,
                    )
                    restoration_status = "rejected" if reason is not None else "blended"
                    if reason is None:
                        try:
                            restored = self._restore(aligned)
                            review = self._cv2.addWeighted(
                                aligned,
                                1.0 - self.restoration_blend_weight,
                                restored,
                                self.restoration_blend_weight,
                                0.0,
                            )
                            review_jpeg = self._encode_jpeg(review)
                            if review_jpeg is None:
                                reason = "restoration_encoding_failed"
                                restoration_status = "rejected"
                        except Exception:
                            LOGGER.exception("GFPGAN restoration failed for track %s", track_id)
                            reason = "restoration_failed"
                            restoration_status = "rejected"
                elif reason is not None:
                    restoration_status = "rejected"

                face = {
                    "track_id": track_id,
                    "confidence": round(confidence, 3),
                    "source_face_size_px": round(source_size, 1),
                    "source_sharpness": round(sharpness, 1),
                    "restoration_status": restoration_status,
                    "quality_reason": reason,
                    "blend_weight": (
                        self.restoration_blend_weight if review_jpeg is not None else None
                    ),
                    "source_jpeg": source_jpeg,
                    "aligned_jpeg": aligned_jpeg,
                    "jpeg": review_jpeg,
                }
                self._cache[cache_key] = (now, face)
            faces.append(face)

        cutoff = now - self.cache_ttl_seconds
        for cache_key, (updated_at, _) in tuple(self._cache.items()):
            if updated_at < cutoff or cache_key[0] != job.generation:
                del self._cache[cache_key]
        return faces

    def _encode_jpeg(self, image) -> bytes | None:
        encoded, buffer = self._cv2.imencode(
            ".jpg",
            image,
            [int(self._cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        return buffer.tobytes() if encoded else None

    def _restore(self, aligned):
        rgb = self._np.ascontiguousarray(aligned[:, :, ::-1])
        tensor = self._torch.from_numpy(rgb.transpose(2, 0, 1)).float().div(255.0)
        tensor = tensor.sub(0.5).div(0.5).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            output = self._gfpgan(
                tensor,
                return_rgb=False,
                randomize_noise=False,
            )[0]
        output = output.squeeze(0).detach().float().cpu().clamp_(-1, 1)
        image = output.add(1).div(2).mul(255).round().byte().numpy().transpose(1, 2, 0)
        return self._np.ascontiguousarray(image[:, :, ::-1])

    def _publish(
        self,
        status: str,
        faces: Sequence[dict],
        *,
        frame_number: int | None,
        message: str | None = None,
    ) -> None:
        self._status = status
        snapshot = {
            "status": status,
            "frame_number": frame_number,
            "updated_monotonic": time.monotonic(),
            "message": message,
            "faces": list(faces),
        }
        try:
            self.on_snapshot(snapshot)
        except Exception:
            LOGGER.exception("face snapshot callback failed")


def _select_torch_device(torch, setting: str):
    if setting == "cpu":
        return torch.device("cpu")
    if setting == "cuda":
        if not torch.cuda.is_available():
            raise FaceModelsUnavailable("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if setting == "mps":
        if not torch.backends.mps.is_available():
            raise FaceModelsUnavailable("MPS was requested but is unavailable")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _verify_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise FaceModelsUnavailable(
            f"checksum mismatch for {path.name}; expected {expected}, got {actual}"
        )


def _verify_sidecar_hash(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        LOGGER.warning("no checksum sidecar found for %s", path.name)
        return
    expected = sidecar.read_text(encoding="ascii").strip().split()[0].lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise FaceModelsUnavailable(f"invalid checksum sidecar for {path.name}")
    _verify_hash(path, expected)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_source_face(
    frame,
    face_box: tuple[float, float, float, float],
    padding_ratio: float = 0.15,
):
    """Copy detected camera pixels with modest context and no generative processing."""
    height, width = frame.shape[:2]
    face_width = max(1.0, face_box[2] - face_box[0])
    face_height = max(1.0, face_box[3] - face_box[1])
    padding_x = face_width * padding_ratio
    padding_y = face_height * padding_ratio
    left = max(0, min(width, int(face_box[0] - padding_x)))
    top = max(0, min(height, int(face_box[1] - padding_y)))
    right = max(0, min(width, int(face_box[2] + padding_x + 0.999)))
    bottom = max(0, min(height, int(face_box[3] + padding_y + 0.999)))
    return frame[top:bottom, left:right].copy()


def _source_sharpness(frame, face_box: tuple[float, float, float, float], cv2) -> float:
    """Return variance-of-Laplacian sharpness from the detected source pixels."""
    height, width = frame.shape[:2]
    left = max(0, min(width, int(face_box[0])))
    top = max(0, min(height, int(face_box[1])))
    right = max(0, min(width, int(face_box[2] + 0.999)))
    bottom = max(0, min(height, int(face_box[3] + 0.999)))
    if right <= left or bottom <= top:
        return 0.0
    source_crop = frame[top:bottom, left:right]
    if len(source_crop.shape) == 3:
        source_crop = cv2.cvtColor(source_crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(source_crop, cv2.CV_64F).var())


def _restoration_rejection_reason(
    source_size: float,
    sharpness: float,
    minimum_size: int,
    minimum_sharpness: float,
) -> str | None:
    if source_size < minimum_size:
        return "source_face_too_small"
    if sharpness < minimum_sharpness:
        return "source_face_too_blurry"
    return None


def _containing_track(
    face_box: tuple[float, float, float, float],
    people: Sequence[PersonBounds],
) -> int | None:
    center_x = (face_box[0] + face_box[2]) / 2
    center_y = (face_box[1] + face_box[3]) / 2
    containing = [
        (max(1, (box[2] - box[0]) * (box[3] - box[1])), track_id)
        for track_id, box in people
        if box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3]
    ]
    return min(containing)[1] if containing else None


def _align_face(frame, landmarks, cv2, np):
    source = np.asarray(landmarks, dtype=np.float32)
    destination = np.asarray(_GFPGAN_TEMPLATE, dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(source, destination, method=cv2.LMEDS)
    if matrix is None:
        raise RuntimeError("could not align SCRFD facial landmarks")
    return cv2.warpAffine(
        frame,
        matrix,
        (512, 512),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
