"""YOLO tracking plus optional face and number-plate detection."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import cv2
from ultralytics import YOLO

import config

LOGGER = logging.getLogger(__name__)


class Detector:
    def __init__(self):
        self._verify_model()
        LOGGER.info("loading YOLO model %s", config.YOLO_MODEL_PATH)
        self.model = YOLO(config.YOLO_MODEL_PATH)

        self.face_cascade = None
        if config.ENABLE_FACE_DETECTION:
            self.face_cascade = self._load_cascade("haarcascade_frontalface_default.xml")

        self.plate_cascade = None
        self.ocr_reader = None
        if config.ENABLE_ANPR:
            self.plate_cascade = self._load_cascade("haarcascade_russian_plate_number.xml")
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError(
                    "ANPR is enabled but EasyOCR is not installed; install the anpr dependency group"
                ) from exc
            LOGGER.info("loading EasyOCR for ANPR; first use may download OCR weights")
            self.ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    @staticmethod
    def _verify_model() -> None:
        model_path = Path(config.YOLO_MODEL_PATH)
        if not config.YOLO_MODEL_SHA256 or not model_path.is_file():
            return
        checksum = hashlib.sha256()
        with model_path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                checksum.update(chunk)
        if checksum.hexdigest() != config.YOLO_MODEL_SHA256:
            raise RuntimeError(
                f"model checksum mismatch for {model_path}; refuse to load unverified weights"
            )

    @staticmethod
    def _load_cascade(filename: str):
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + filename)
        if cascade.empty():
            raise RuntimeError(f"failed to load OpenCV cascade {filename}")
        return cascade

    def track_frame(self, frame) -> list[dict]:
        """Return relevant YOLO detections, including temporarily untracked boxes."""
        results = self.model.track(
            frame,
            persist=True,
            conf=config.CONFIDENCE_THRESHOLD,
            classes=[config.PERSON_CLASS_ID, *config.VEHICLE_CLASS_IDS],
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        coordinates = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        track_ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(coordinates)

        detections = []
        for box, raw_track_id, raw_class_id, confidence in zip(
            coordinates, track_ids, class_ids, confidences, strict=True
        ):
            x1, y1, x2, y2 = box
            class_id = int(raw_class_id)
            class_name = (
                "person"
                if class_id == config.PERSON_CLASS_ID
                else config.VEHICLE_CLASS_IDS.get(class_id, f"class_{class_id}")
            )
            detections.append(
                {
                    "track_id": None if raw_track_id is None else int(raw_track_id),
                    "class_id": class_id,
                    "class_name": class_name,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "centroid": ((x1 + x2) / 2, (y1 + y2) / 2),
                    "conf": float(confidence),
                }
            )
        return detections

    def reset_tracking(self) -> None:
        """Clear ByteTrack state after a source discontinuity or reconnect."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()) or ():
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    def detect_faces(self, frame, person_bbox) -> list[tuple[int, int, int, int]]:
        if self.face_cascade is None:
            return []
        crop, offset = self._crop(frame, person_bbox)
        if crop is None:
            return []
        x_offset, y_offset = offset
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return [
            (
                x_offset + int(x),
                y_offset + int(y),
                x_offset + int(x + width),
                y_offset + int(y + height),
            )
            for x, y, width, height in faces
        ]

    def detect_plate(self, frame, vehicle_bbox):
        """Return ``(text, full_frame_bbox)`` for the best OCR plate candidate."""
        if self.plate_cascade is None or self.ocr_reader is None:
            return None, None
        vehicle_crop, offset = self._crop(frame, vehicle_bbox)
        if vehicle_crop is None:
            return None, None
        x_offset, y_offset = offset

        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        plates = self.plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        if len(plates) == 0:
            return None, None

        x, y, width, height = max(plates, key=lambda plate: plate[2] * plate[3])
        plate_crop = vehicle_crop[y : y + height, x : x + width]
        if plate_crop.size == 0:
            return None, None
        ocr_results = self.ocr_reader.readtext(plate_crop)
        if not ocr_results:
            return None, None

        best = max(ocr_results, key=lambda result: result[2])
        raw_text, ocr_confidence = best[1], float(best[2])
        plate_text = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()
        if not plate_text or ocr_confidence < config.ANPR_MIN_OCR_CONFIDENCE:
            return None, None

        bbox = (
            x_offset + int(x),
            y_offset + int(y),
            x_offset + int(x + width),
            y_offset + int(y + height),
        )
        return plate_text, bbox

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
