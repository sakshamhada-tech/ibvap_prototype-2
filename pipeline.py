"""Shared analytics pipeline used by both CLI and web entry points."""

from __future__ import annotations

import logging
import math
import time
from collections import deque

import cv2
import numpy as np

import config
from detector import Detector
from face_enhancement import FaceEnhancementService
from utils.alert_logger import AlertLogger
from utils.zones import LoiteringDetector, LoiteringState, VirtualFence

LOGGER = logging.getLogger(__name__)
COLOR_PERSON = (0, 200, 0)
COLOR_VEHICLE = (0, 140, 255)
COLOR_PLATE = (255, 0, 255)
COLOR_FENCE = (0, 0, 255)
COLOR_ALERT_BANNER = (0, 0, 255)


def frame_brightness(frame) -> float:
    """Measure a central ROI so letterboxing and edge glare have less influence."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    margin = config.NIGHT_BRIGHTNESS_ROI_MARGIN
    margin_x, margin_y = int(width * margin), int(height * margin)
    if margin_x or margin_y:
        gray = gray[margin_y : height - margin_y, margin_x : width - margin_x]
    return float(np.mean(gray))


def draw_detection(frame, detection: dict, is_loitering: bool) -> None:
    x1, y1, x2, y2 = detection["bbox"]
    color = COLOR_PERSON if detection["class_name"] == "person" else COLOR_VEHICLE
    track_label = f"#{detection['track_id']}" if detection["track_id"] is not None else "untracked"
    label = f"{detection['class_name']} {track_label} {detection['conf']:.2f}"
    if is_loitering:
        label += " [LOITERING]"
        color = (0, 165, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(0, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )


def draw_plate(frame, plate_bbox, plate_text: str) -> None:
    x1, y1, x2, y2 = plate_bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PLATE, 2)
    cv2.putText(
        frame,
        plate_text,
        (x1, max(0, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_PLATE,
        2,
    )


def draw_fence(frame, line) -> None:
    cv2.line(frame, line[0], line[1], COLOR_FENCE, 2)
    cv2.putText(
        frame,
        "VIRTUAL FENCE",
        (line[0][0], max(15, line[0][1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_FENCE,
        2,
    )


def draw_status_bar(frame, is_night: bool, active_alerts, frame_number: int, fps: float) -> None:
    height, width = frame.shape[:2]
    mode_text = "NIGHT MODE" if is_night else "DAY MODE"
    mode_color = (255, 200, 0) if is_night else (0, 255, 0)
    cv2.rectangle(frame, (0, 0), (width, 34), (30, 30, 30), -1)
    cv2.putText(
        frame,
        f"IBVAP | {mode_text} | frame {frame_number} | {fps:.1f} fps",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        mode_color,
        2,
    )
    if active_alerts:
        banner = " | ".join(active_alerts[-3:])
        cv2.rectangle(frame, (0, height - 30), (width, height), COLOR_ALERT_BANNER, -1)
        cv2.putText(
            frame,
            banner,
            (10, height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )


class VideoPipeline:
    """One camera's detection, event generation, annotation, and statistics."""

    def __init__(
        self,
        on_alert=None,
        on_face_snapshot=None,
        show_overlays: bool = True,
    ):
        self.detector = Detector()
        self.face_enhancer = None
        if config.ENABLE_FACE_ENHANCEMENT and on_face_snapshot is not None:
            self.face_enhancer = FaceEnhancementService(
                scrfd_model_path=config.SCRFD_MODEL_PATH,
                gfpgan_model_path=config.GFPGAN_MODEL_PATH,
                gfpgan_model_sha256=config.GFPGAN_MODEL_SHA256,
                confidence_threshold=config.SCRFD_CONFIDENCE_THRESHOLD,
                input_size=config.SCRFD_INPUT_SIZE,
                min_face_size_px=config.SCRFD_MIN_FACE_SIZE_PX,
                max_faces=config.FACE_ENHANCEMENT_MAX_FACES,
                refresh_seconds=config.FACE_ENHANCEMENT_REFRESH_SECONDS,
                cache_ttl_seconds=config.FACE_ENHANCEMENT_CACHE_TTL_SECONDS,
                jpeg_quality=config.FACE_ENHANCEMENT_JPEG_QUALITY,
                device=config.FACE_ENHANCEMENT_DEVICE,
                on_snapshot=on_face_snapshot,
            )
            self.face_enhancer.start()
        self.fence = VirtualFence(
            config.VIRTUAL_FENCE_LINE,
            stale_after_seconds=config.TRACK_STALE_SECONDS,
            contact_margin_pixels=config.VIRTUAL_FENCE_CONTACT_MARGIN_PX,
        )
        self.loiter_detector = LoiteringDetector(
            config.LOITERING_SECONDS,
            config.LOITERING_MOVEMENT_THRESHOLD_PX,
            max_observation_gap_seconds=config.TRACK_MAX_OBSERVATION_GAP_SECONDS,
            stale_after_seconds=config.TRACK_STALE_SECONDS,
        )
        self.alert_logger = AlertLogger(
            config.ALERT_LOG_CSV,
            on_alert=on_alert,
            cooldown_seconds=config.ALERT_COOLDOWN_SECONDS,
            state_ttl_seconds=config.ALERT_STATE_TTL_SECONDS,
            max_bytes=config.ALERT_LOG_MAX_BYTES,
            backup_count=config.LOG_BACKUP_COUNT,
        )
        self.show_overlays = show_overlays
        self.frame_number = 0
        self._started_monotonic = time.monotonic()
        self._last_source_time: float | None = None
        self._smoothed_brightness: float | None = None
        self._night_mode = False
        self._fence_validated = False
        self._fps_samples = deque(maxlen=120)
        self._recent_alerts = deque(maxlen=config.MAX_RECENT_ALERTS)
        self.stats = {
            "people": 0,
            "vehicles": 0,
            "alerts_total": 0,
            "night_mode": False,
            "frame_number": 0,
            "fps": 0.0,
            "uptime_seconds": 0.0,
            "source_status": "starting",
        }

    def process_frame(self, frame, *, source_time: float | None = None):
        """Process one frame using media time (recording) or monotonic time (live)."""
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("process_frame requires a non-empty image")
        timestamp = time.monotonic() if source_time is None else float(source_time)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("source_time must be a finite, non-negative number")
        if not self._fence_validated:
            self._validate_fence(frame)
        if self._last_source_time is not None and timestamp < self._last_source_time:
            self.reset_tracking()
        self._last_source_time = timestamp
        self.frame_number += 1

        measured_brightness = frame_brightness(frame)
        alpha = config.NIGHT_BRIGHTNESS_SMOOTHING_ALPHA
        self._smoothed_brightness = (
            measured_brightness
            if self._smoothed_brightness is None
            else alpha * measured_brightness + (1 - alpha) * self._smoothed_brightness
        )
        if self._night_mode:
            self._night_mode = self._smoothed_brightness < (
                config.NIGHT_BRIGHTNESS_THRESHOLD + config.NIGHT_BRIGHTNESS_HYSTERESIS
            )
        else:
            self._night_mode = self._smoothed_brightness < config.NIGHT_BRIGHTNESS_THRESHOLD
        brightness = self._smoothed_brightness
        is_night = self._night_mode
        detections = self.detector.track_frame(frame)
        people_count = sum(det["class_name"] == "person" for det in detections)
        vehicle_count = len(detections) - people_count
        if (
            self.face_enhancer is not None
            and self.frame_number % config.FACE_ENHANCEMENT_INTERVAL_FRAMES == 0
        ):
            people = [
                (detection["track_id"], detection["bbox"])
                for detection in detections
                if detection["class_name"] == "person" and detection["track_id"] is not None
            ]
            self.face_enhancer.submit(frame, people, self.frame_number)

        for detection in detections:
            track_id = detection["track_id"]
            class_name = detection["class_name"]
            loitering = LoiteringState(active=False, started=False)

            if track_id is not None:
                if self.fence.check_crossing(
                    track_id,
                    detection["centroid"],
                    timestamp,
                    bounds=detection["bbox"],
                ):
                    night_tag = " (night)" if is_night else ""
                    if self.alert_logger.log(
                        "VIRTUAL_FENCE_INTRUSION",
                        track_id,
                        class_name + night_tag,
                        self.frame_number,
                        source_time=timestamp,
                    ):
                        self._remember_alert(f"INTRUSION: {class_name}{night_tag}")

                if class_name == "person":
                    loitering = self.loiter_detector.update(
                        track_id, detection["centroid"], timestamp
                    )
                    if loitering.started and self.alert_logger.log(
                        "SUSPICIOUS_LOITERING",
                        track_id,
                        f"stationary > {config.LOITERING_SECONDS:g}s",
                        self.frame_number,
                        source_time=timestamp,
                    ):
                        self._remember_alert(f"LOITERING: person #{track_id}")

                if is_night and self.alert_logger.log(
                    "NIGHT_MOVEMENT",
                    track_id,
                    f"{class_name} detected in low light (brightness={brightness:.0f})",
                    self.frame_number,
                    source_time=timestamp,
                ):
                    self._remember_alert(f"NIGHT MOVEMENT: {class_name} #{track_id}")

            plate_text, plate_bbox = None, None
            if (
                config.ENABLE_ANPR
                and class_name != "person"
                and self._scheduled(track_id, config.ANPR_INTERVAL_FRAMES)
            ):
                plate_text, plate_bbox = self.detector.detect_plate(frame, detection["bbox"])
                if (
                    plate_text
                    and track_id is not None
                    and self.alert_logger.log(
                        "ANPR_READ",
                        track_id,
                        plate_text,
                        self.frame_number,
                        source_time=timestamp,
                    )
                ):
                    self._remember_alert(f"PLATE: {plate_text}")

            if self.show_overlays:
                draw_detection(frame, detection, loitering.active)
                if plate_bbox is not None:
                    draw_plate(frame, plate_bbox, plate_text)
                center_x, center_y = map(int, detection["centroid"])
                cv2.circle(frame, (center_x, center_y), 3, (255, 255, 255), -1)

        self.fence.expire(timestamp)
        self.loiter_detector.expire(timestamp)
        now = time.monotonic()
        self._fps_samples.append(now)
        fps = self._recent_fps()
        elapsed = now - self._started_monotonic
        self._expire_alert_labels(now)

        self.stats.update(
            {
                "people": people_count,
                "vehicles": vehicle_count,
                "alerts_total": self.alert_logger.total_logged,
                "night_mode": is_night,
                "brightness": round(brightness, 1),
                "frame_number": self.frame_number,
                "fps": fps,
                "uptime_seconds": elapsed,
                "source_status": "running",
            }
        )
        if self.show_overlays:
            draw_fence(frame, config.VIRTUAL_FENCE_LINE)
            draw_status_bar(
                frame,
                is_night,
                [label for _, label in self._recent_alerts],
                self.frame_number,
                fps,
            )
        return frame

    def _validate_fence(self, frame) -> None:
        height, width = frame.shape[:2]
        self._fence_validated = True
        if not all(0 <= x < width and 0 <= y < height for x, y in config.VIRTUAL_FENCE_LINE):
            LOGGER.warning(
                "virtual fence %s falls outside source dimensions %sx%s",
                config.VIRTUAL_FENCE_LINE,
                width,
                height,
            )

    def _scheduled(self, track_id: int | None, interval: int) -> bool:
        offset = 0 if track_id is None else track_id
        return (self.frame_number + offset) % interval == 0

    def _remember_alert(self, label: str) -> None:
        self._recent_alerts.append((time.monotonic(), label))

    def _expire_alert_labels(self, now: float) -> None:
        cutoff = now - config.RECENT_ALERT_DISPLAY_SECONDS
        while self._recent_alerts and self._recent_alerts[0][0] < cutoff:
            self._recent_alerts.popleft()

    def _recent_fps(self) -> float:
        if len(self._fps_samples) < 2:
            return 0.0
        duration = self._fps_samples[-1] - self._fps_samples[0]
        return (len(self._fps_samples) - 1) / duration if duration > 0 else 0.0

    def reset_tracking(self) -> None:
        """Clear source-specific state after seek, loop, or reconnect."""
        self.detector.reset_tracking()
        if self.face_enhancer is not None:
            self.face_enhancer.reset()
        self.fence.reset()
        self.loiter_detector.reset()
        self.alert_logger.reset_cooldowns()
        self._recent_alerts.clear()
        self._fps_samples.clear()
        self._last_source_time = None
        self._smoothed_brightness = None
        self._night_mode = False
        self._fence_validated = False

    def close(self) -> None:
        if self.face_enhancer is not None:
            self.face_enhancer.close()
        self.alert_logger.close()
