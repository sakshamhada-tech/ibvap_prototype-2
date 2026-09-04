"""
Video Pipeline
--------------
The actual analytics pipeline: detection -> virtual fence -> loitering ->
night detection -> face detection -> overlay drawing -> alert logging.

This is the ONE place that logic lives. Both main.py (CLI/OpenCV-window
demo) and server.py (web dashboard) drive this same class, so there is
no duplicated - and therefore no divergent, re-buggable - analytics code
between the two.
"""

import time

import cv2
import numpy as np

import config
from detector import Detector
from utils.alert_logger import AlertLogger
from utils.zones import VirtualFence, LoiteringDetector


COLOR_PERSON = (0, 200, 0)
COLOR_VEHICLE = (255, 140, 0)
COLOR_FACE = (0, 255, 255)
COLOR_PLATE = (255, 0, 255)
COLOR_FENCE = (0, 0, 255)
COLOR_ALERT_BANNER = (0, 0, 255)


def frame_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def draw_detection(frame, det, is_loitering):
    x1, y1, x2, y2 = det["bbox"]
    color = COLOR_PERSON if det["class_name"] == "person" else COLOR_VEHICLE
    label = f'{det["class_name"]} #{det["track_id"]} {det["conf"]:.2f}'
    if is_loitering:
        label += " [LOITERING]"
        color = (0, 165, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_faces(frame, faces):
    for (x1, y1, x2, y2) in faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_FACE, 1)


def draw_plate(frame, plate_bbox, plate_text):
    x1, y1, x2, y2 = plate_bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PLATE, 2)
    cv2.putText(frame, plate_text, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PLATE, 2)


def draw_fence(frame, line):
    cv2.line(frame, line[0], line[1], COLOR_FENCE, 2)
    cv2.putText(frame, "VIRTUAL FENCE", (line[0][0], line[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_FENCE, 2)


def draw_status_bar(frame, is_night, active_alerts, frame_number, fps):
    h, w = frame.shape[:2]
    mode_text = "NIGHT MODE" if is_night else "DAY MODE"
    mode_color = (255, 200, 0) if is_night else (0, 255, 0)

    cv2.rectangle(frame, (0, 0), (w, 34), (30, 30, 30), -1)
    cv2.putText(frame, f"IBVAP | {mode_text} | frame {frame_number} | ~{fps:.1f} fps",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

    if active_alerts:
        banner = " | ".join(active_alerts[-3:])  # show latest few
        cv2.rectangle(frame, (0, h - 30), (w, h), COLOR_ALERT_BANNER, -1)
        cv2.putText(frame, banner, (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


class VideoPipeline:
    """
    One instance = one camera/video source's full analytics pipeline.
    Call process_frame(frame) once per frame; it returns the annotated
    frame and updates self.stats in place.
    """

    def __init__(self, on_alert=None, show_overlays=True):
        self.detector = Detector()
        self.fence = VirtualFence(config.VIRTUAL_FENCE_LINE)
        self.loiter_detector = LoiteringDetector(
            config.LOITERING_SECONDS, config.LOITERING_MOVEMENT_THRESHOLD_PX
        )
        self.alert_logger = AlertLogger(config.ALERT_LOG_CSV, on_alert=on_alert)
        self.show_overlays = show_overlays

        self.frame_number = 0
        self.t_start = time.time()
        self.recent_alert_labels = []

        self.stats = {
            "people": 0,
            "vehicles": 0,
            "alerts_total": 0,
            "night_mode": False,
            "frame_number": 0,
            "fps": 0.0,
            "uptime_seconds": 0.0,
        }

    def process_frame(self, frame):
        self.frame_number += 1
        brightness = frame_brightness(frame)
        is_night = brightness < config.NIGHT_BRIGHTNESS_THRESHOLD

        detections = self.detector.track_frame(frame)
        people_count = 0
        vehicle_count = 0

        for det in detections:
            track_id = det["track_id"]

            if det["class_name"] == "person":
                people_count += 1
            else:
                vehicle_count += 1

            # -- Virtual fence intrusion --
            if self.fence.check_crossing(track_id, det["centroid"]):
                tag = " (night)" if is_night else ""
                logged = self.alert_logger.log(
                    "VIRTUAL_FENCE_INTRUSION", track_id, det["class_name"] + tag,
                    self.frame_number,
                )
                if logged:
                    self.recent_alert_labels.append(
                        "INTRUSION: " + det["class_name"] + tag
                    )

            # -- Loitering (people only) --
            is_loitering = False
            if det["class_name"] == "person":
                is_loitering = self.loiter_detector.update(track_id, det["centroid"])
                if is_loitering:
                    logged = self.alert_logger.log(
                        "SUSPICIOUS_LOITERING", track_id,
                        f"stationary > {config.LOITERING_SECONDS}s", self.frame_number,
                    )
                    if logged:
                        self.recent_alert_labels.append(f"LOITERING: person #{track_id}")

            # -- Night-time movement flag --
            if is_night:
                self.alert_logger.log(
                    "NIGHT_MOVEMENT", track_id,
                    f'{det["class_name"]} detected in low light '
                    f'(brightness={brightness:.0f})', self.frame_number,
                )

            # -- Face detection (people only) --
            faces = []
            if det["class_name"] == "person":
                faces = self.detector.detect_faces(frame, det["bbox"])

            # -- ANPR (vehicles only, off by default) --
            plate_text, plate_bbox = (None, None)
            if config.ENABLE_ANPR and det["class_name"] != "person":
                plate_text, plate_bbox = self.detector.detect_plate(frame, det["bbox"])
                if plate_text:
                    logged = self.alert_logger.log(
                        "ANPR_READ", track_id, plate_text, self.frame_number
                    )
                    if logged:
                        self.recent_alert_labels.append(f"PLATE: {plate_text}")

            if self.show_overlays:
                draw_detection(frame, det, is_loitering)
                draw_faces(frame, faces)
                if plate_bbox is not None:
                    draw_plate(frame, plate_bbox, plate_text)
                cx, cy = int(det["centroid"][0]), int(det["centroid"][1])
                cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)

        elapsed = time.time() - self.t_start
        fps = self.frame_number / elapsed if elapsed > 0 else 0.0

        self.stats["people"] = people_count
        self.stats["vehicles"] = vehicle_count
        self.stats["alerts_total"] = self.alert_logger.total_logged
        self.stats["night_mode"] = is_night
        self.stats["frame_number"] = self.frame_number
        self.stats["fps"] = fps
        self.stats["uptime_seconds"] = elapsed

        if self.show_overlays:
            draw_fence(frame, config.VIRTUAL_FENCE_LINE)
            draw_status_bar(frame, is_night, self.recent_alert_labels,
                             self.frame_number, fps)

        return frame

    def close(self):
        self.alert_logger.close()
