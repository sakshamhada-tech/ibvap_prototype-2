"""
Detector
--------
Wraps YOLOv8 (via ultralytics) for combined object detection + tracking,
plus a lightweight face detector and ANPR (plate localization + OCR).

Why YOLOv8 + built-in tracker:
  - Single model handles person + vehicle detection (COCO classes)
  - `model.track(...)` gives each object a persistent ID across frames
    for free (ByteTrack under the hood) - which the virtual fence and
    loitering logic both depend on.

ANPR approach:
  - Plate localization uses OpenCV's bundled Haar cascade
    (haarcascade_russian_plate_number.xml) - ships with opencv-python,
    so no extra model download is needed for this step. It's a general
    rectangular-plate detector, not India-specific, but works fine for
    a demo despite the filename.
  - Text recognition uses EasyOCR, loaded lazily (only if ANPR is
    enabled) since it pulls in a ~100MB model on first run and would
    otherwise slow down startup for people not using this feature.
"""

import re

import cv2
from ultralytics import YOLO

import config


class Detector:
    def __init__(self):
        print(f"[INIT] Loading YOLO model: {config.YOLO_MODEL_PATH}")
        self.model = YOLO(config.YOLO_MODEL_PATH)

        self.face_cascade = None
        if config.ENABLE_FACE_DETECTION:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("[INIT] Face detector (Haar cascade) loaded")

        self.plate_cascade = None
        self.ocr_reader = None
        if config.ENABLE_ANPR:
            plate_cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
            self.plate_cascade = cv2.CascadeClassifier(plate_cascade_path)

            print("[INIT] Loading OCR engine for ANPR (first run downloads model weights)...")
            import easyocr  # imported lazily - heavy dependency, only needed here
            self.ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            print("[INIT] ANPR ready (plate cascade + EasyOCR)")

    def track_frame(self, frame):
        """
        Runs detection + tracking on a single frame.

        Returns a list of dicts:
            {track_id, class_id, class_name, bbox (x1,y1,x2,y2), centroid, conf}
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=config.CONFIDENCE_THRESHOLD,
            classes=[config.PERSON_CLASS_ID, *config.VEHICLE_CLASS_IDS.keys()],
            verbose=False,
        )

        detections = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for box, track_id, cls, conf in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.id.cpu().numpy(),
                boxes.cls.cpu().numpy(),
                boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = box
                cls_id = int(cls)
                class_name = (
                    "person" if cls_id == config.PERSON_CLASS_ID
                    else config.VEHICLE_CLASS_IDS.get(cls_id, f"class_{cls_id}")
                )
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)

                detections.append({
                    "track_id": int(track_id),
                    "class_id": cls_id,
                    "class_name": class_name,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "centroid": centroid,
                    "conf": float(conf),
                })

        return detections

    def detect_faces(self, frame, person_bbox):
        """Runs face detection cropped to a person's bounding box (cheap + focused)."""
        if self.face_cascade is None:
            return []

        x1, y1, x2, y2 = person_bbox
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # convert face boxes back to full-frame coordinates
        return [(x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh) for (fx, fy, fw, fh) in faces]

    def detect_plate(self, frame, vehicle_bbox):
        """
        ANPR: locate a plate-shaped region inside the vehicle's bounding box,
        then run OCR on it.

        Returns (plate_text, plate_bbox_in_full_frame_coords) or (None, None)
        if no plate was confidently found/read. Off by default
        (config.ENABLE_ANPR = False) - flip it on once you've accepted the
        one-time EasyOCR model download.
        """
        if self.plate_cascade is None or self.ocr_reader is None:
            return None, None

        vx1, vy1, vx2, vy2 = vehicle_bbox
        vx1, vy1 = max(0, vx1), max(0, vy1)
        vehicle_crop = frame[vy1:vy2, vx1:vx2]
        if vehicle_crop.size == 0:
            return None, None

        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        plates = self.plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        if len(plates) == 0:
            return None, None

        # take the largest candidate region (most likely to actually be the plate)
        px, py, pw, ph = max(plates, key=lambda p: p[2] * p[3])
        plate_crop = vehicle_crop[py:py + ph, px:px + pw]
        if plate_crop.size == 0:
            return None, None

        ocr_results = self.ocr_reader.readtext(plate_crop)
        if not ocr_results:
            return None, None

        # pick the highest-confidence text read, clean it up (plates are
        # alphanumeric - strip anything OCR mis-reads as punctuation/noise)
        best = max(ocr_results, key=lambda r: r[2])
        raw_text, ocr_conf = best[1], best[2]
        plate_text = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()

        if not plate_text or ocr_conf < 0.35:
            return None, None

        full_frame_bbox = (vx1 + px, vy1 + py, vx1 + px + pw, vy1 + py + ph)
        return plate_text, full_frame_bbox