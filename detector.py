"""YOLO tracking plus optional face and regional number-plate detection."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import config
from utils.anpr import PlateObservation, combine_ocr_results, identify_plate_region

LOGGER = logging.getLogger(__name__)
_LATIN_ANPR_REGIONS = {"IN", "BT", "LK", "MV"}
_LATIN_OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


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

    def __init__(self):
        self._anpr_metrics = self._new_anpr_metrics()
        self._verify_model_file(
            config.YOLO_MODEL_PATH,
            config.YOLO_MODEL_SHA256,
            "object detector",
            required=False,
        )
        LOGGER.info("loading YOLO model %s", config.YOLO_MODEL_PATH)
        self.model = YOLO(config.YOLO_MODEL_PATH)
        self.tile_model = None
        if config.ENABLE_TILED_INFERENCE:
            try:
                # A separate predictor prevents tile predictions from mutating
                # the callbacks/state owned by the full-frame tracker.
                self.tile_model = YOLO(config.YOLO_MODEL_PATH)
            except Exception:
                LOGGER.exception(
                    "tiled detector initialization failed; full-frame tracking continues"
                )
        self._frame_number = 0
        self._tiled_error_active = False
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
            LOGGER.info("loading regional YOLO plate detector %s", config.ANPR_PLATE_MODEL_PATH)
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
        """Track the full frame and optionally add cadence-limited tiled detections.

        Full-frame tracking always runs, preserving ByteTrack continuity. Tiled
        detections are supplemental and remain untracked until the normal
        detector subsequently acquires them; this limitation is preferable to
        inventing track identities across independent tile predictions.
        """
        self._frame_number += 1
        results = self.model.track(
            frame,
            persist=True,
            conf=config.CONFIDENCE_THRESHOLD,
            classes=[config.PERSON_CLASS_ID, *config.VEHICLE_CLASS_IDS],
            verbose=False,
        )
        detections = self._result_detections(results[0] if results else None, include_tracks=True)
        if (
            self.tile_model is not None
            and self._frame_number % config.TILED_INFERENCE_INTERVAL_FRAMES == 0
        ):
            try:
                tiled = self._tiled_detections(frame)
            except Exception:
                if not self._tiled_error_active:
                    LOGGER.exception("tiled inference failed; full-frame tracking will continue")
                self._tiled_error_active = True
            else:
                self._tiled_error_active = False
                detections = self._merge_supplemental(detections, tiled)
        return detections

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

    def _tiled_detections(self, frame) -> list[dict]:
        height, width = frame.shape[:2]
        detections = []
        for left, top, right, bottom in self.tile_bounds(
            width, height, config.TILE_SIZE, config.TILE_OVERLAP
        ):
            tile = frame[top:bottom, left:right]
            results = self.tile_model.predict(
                tile,
                conf=config.CONFIDENCE_THRESHOLD,
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

    def reset_tracking(self) -> None:
        """Clear ByteTrack state after a source discontinuity."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()) or ():
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

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
