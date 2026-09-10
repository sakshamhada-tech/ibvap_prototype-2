"""Shared analytics pipeline used by both CLI and web entry points."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque

import cv2
import numpy as np

import config
from detector import Detector
from face_enhancement import FaceEnhancementService
from posture_analysis import PostureAnalysisService
from utils.alert_logger import AlertLogger
from utils.anpr import PlateConsensusTracker
from utils.group_behavior import GroupApproachDetector, point_to_segment_distance
from utils.motion_signature import MotionSignatureAnalyzer
from utils.risk_scoring import RiskScorer, RiskSignals
from utils.zones import LoiteringDetector, LoiteringState, VirtualFence, point_in_polygon

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


def should_draw_detection(detection: dict) -> bool:
    """Keep supplemental tile proposals out of the authoritative live overlay."""
    return detection.get("source", "full-frame") != "tile"


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
        on_focus_snapshot=None,
        on_evaluation_frame=None,
        show_overlays: bool = True,
    ):
        self.detector = Detector(
            on_focus_snapshot=on_focus_snapshot,
            frame_aligned_tiling=on_evaluation_frame is not None,
        )
        self._on_evaluation_frame = on_evaluation_frame
        self.face_enhancer = None
        if config.ENABLE_FACE_ENHANCEMENT and on_face_snapshot is not None:
            self.face_enhancer = FaceEnhancementService(
                scrfd_model_path=config.SCRFD_MODEL_PATH,
                gfpgan_model_path=config.GFPGAN_MODEL_PATH,
                gfpgan_model_sha256=config.GFPGAN_MODEL_SHA256,
                enable_restoration=config.ENABLE_GFPGAN_RESTORATION,
                confidence_threshold=config.SCRFD_CONFIDENCE_THRESHOLD,
                input_size=config.SCRFD_INPUT_SIZE,
                execution_provider=config.SCRFD_EXECUTION_PROVIDER,
                min_face_size_px=config.SCRFD_MIN_FACE_SIZE_PX,
                restoration_min_face_size_px=config.GFPGAN_MIN_SOURCE_FACE_SIZE_PX,
                restoration_min_sharpness=config.GFPGAN_MIN_SOURCE_SHARPNESS,
                restoration_blend_weight=config.GFPGAN_BLEND_WEIGHT,
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
        self.group_detector = (
            GroupApproachDetector(
                fence_line=config.VIRTUAL_FENCE_LINE,
                minimum_size=config.GROUP_MIN_SIZE,
                proximity_pixels=config.GROUP_PROXIMITY_PX,
                window_seconds=config.GROUP_APPROACH_WINDOW_SECONDS,
                minimum_speed_pixels_per_second=config.GROUP_MIN_SPEED_PX_PER_SECOND,
                heading_similarity=config.GROUP_HEADING_SIMILARITY,
                minimum_fence_progress_pixels=config.GROUP_MIN_FENCE_PROGRESS_PX,
                stale_after_seconds=config.TRACK_STALE_SECONDS,
            )
            if config.ENABLE_GROUP_APPROACH or config.ENABLE_CONTEXTUAL_RISK
            else None
        )
        self.motion_analyzer = (
            MotionSignatureAnalyzer(
                window_seconds=config.MOTION_SIGNATURE_WINDOW_SECONDS,
                minimum_samples=config.MOTION_SIGNATURE_MIN_SAMPLES,
                static_path_threshold_pixels=config.MOTION_SIGNATURE_STATIC_PATH_PX,
                vegetation_max_net_displacement_pixels=(
                    config.MOTION_SIGNATURE_VEGETATION_MAX_NET_PX
                ),
                high_frequency_zero_crossing_ratio=(config.MOTION_SIGNATURE_HIGH_FREQUENCY_RATIO),
                displacement_scale_pixels=config.MOTION_SIGNATURE_DISPLACEMENT_SCALE_PX,
                oscillation_scale_pixels=config.MOTION_SIGNATURE_OSCILLATION_SCALE_PX,
                static_score=config.MOTION_SIGNATURE_STATIC_SCORE,
                vegetation_score=config.MOTION_SIGNATURE_VEGETATION_SCORE,
                gait_score=config.MOTION_SIGNATURE_GAIT_SCORE,
                rigid_score=config.MOTION_SIGNATURE_RIGID_SCORE,
                ambiguous_score=config.MOTION_SIGNATURE_AMBIGUOUS_SCORE,
            )
            if config.ENABLE_MOTION_SIGNATURE
            else None
        )
        self.risk_scorer = (
            RiskScorer(
                threshold=config.RISK_ALERT_THRESHOLD,
                weights={
                    "fence_contact": config.RISK_WEIGHT_FENCE_CONTACT,
                    "fence_proximity": config.RISK_WEIGHT_FENCE_PROXIMITY,
                    "loitering": config.RISK_WEIGHT_LOITERING,
                    "motion_signature": config.RISK_WEIGHT_MOTION_SIGNATURE,
                    "low_posture": config.RISK_WEIGHT_LOW_POSTURE,
                    "night": config.RISK_WEIGHT_NIGHT,
                    "fence_approach": config.RISK_WEIGHT_FENCE_APPROACH,
                    "group_membership": config.RISK_WEIGHT_GROUP_MEMBERSHIP,
                },
                stale_after_seconds=config.TRACK_STALE_SECONDS,
            )
            if config.ENABLE_CONTEXTUAL_RISK
            else None
        )
        self._posture_lock = threading.Lock()
        self._posture_snapshot = {
            "status": "disabled" if not config.ENABLE_POSTURE_ANALYSIS else "loading",
            "signals": [],
            "frame_number": None,
            "message": None,
        }
        self.posture_service = None
        if config.ENABLE_POSTURE_ANALYSIS:
            self.posture_service = PostureAnalysisService(
                model_path=config.POSTURE_MODEL_PATH,
                model_sha256=config.POSTURE_MODEL_SHA256,
                confidence_threshold=config.POSTURE_CONFIDENCE_THRESHOLD,
                minimum_person_size_px=config.POSTURE_MIN_PERSON_SIZE_PX,
                maximum_people=config.POSTURE_MAX_PEOPLE,
                minimum_visible_keypoints=config.POSTURE_MIN_VISIBLE_KEYPOINTS,
                minimum_keypoint_confidence=config.POSTURE_MIN_KEYPOINT_CONFIDENCE,
                minimum_aspect_ratio=config.POSTURE_LOW_ASPECT_RATIO,
                maximum_vertical_spread_ratio=config.POSTURE_MAX_VERTICAL_SPREAD_RATIO,
                sustain_seconds=config.POSTURE_SUSTAIN_SECONDS,
                state_ttl_seconds=config.POSTURE_STATE_TTL_SECONDS,
                device=config.POSTURE_DEVICE,
                on_snapshot=self._update_posture_snapshot,
            )
            self.posture_service.start()
        self.anpr_consensus = PlateConsensusTracker(
            min_reads=config.ANPR_MIN_CONSENSUS_READS,
            window_seconds=config.ANPR_CONSENSUS_WINDOW_SECONDS,
            min_weight_ratio=config.ANPR_MIN_CONSENSUS_WEIGHT_RATIO,
            min_ocr_confidence=config.ANPR_MIN_OCR_CONFIDENCE,
            state_ttl_seconds=config.ANPR_STATE_TTL_SECONDS,
            max_observations=config.ANPR_MAX_OBSERVATIONS_PER_TRACK,
        )
        self._stable_plates = {}
        self._anpr_stable_reads = 0
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
            "detection": self._detector_diagnostics(),
            "context_readiness": {
                "status": "blocked",
                "reason": "no_person_detection",
                "people": [],
            },
            "anpr": {
                **self.detector.anpr_diagnostics(),
                "consensus_tracks": 0,
                "stable_reads": 0,
            },
            "group_approach_tracks": 0,
            "contextual_risk_tracks": 0,
            "posture_status": self._posture_snapshot["status"],
        }

    def process_frame(self, frame, *, source_time: float | None = None):
        """Process one frame using media time (recording) or monotonic time (live)."""
        processing_started = time.perf_counter()
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
        people = [
            (detection["track_id"], detection["bbox"])
            for detection in detections
            if detection["class_name"] == "person" and detection["track_id"] is not None
        ]
        person_centroids = [
            (detection["track_id"], detection["centroid"])
            for detection in detections
            if detection["class_name"] == "person" and detection["track_id"] is not None
        ]
        group_events = (
            self.group_detector.update(person_centroids, timestamp) if self.group_detector else ()
        )
        if config.ENABLE_GROUP_APPROACH:
            for event in group_events:
                group_id = min(event.track_ids)
                if self.alert_logger.log(
                    "GROUP_APPROACH",
                    group_id,
                    "coordinated fence approach by tracks "
                    + ",".join(str(track_id) for track_id in event.track_ids),
                    self.frame_number,
                    source_time=timestamp,
                ):
                    self._remember_alert(f"GROUP APPROACH: {len(event.track_ids)} people")
        if (
            self.posture_service is not None
            and self.frame_number % config.POSTURE_INTERVAL_FRAMES == 0
        ):
            self.posture_service.submit(frame, people, self.frame_number, timestamp)
        posture_signals = self._current_posture_signals()
        if (
            self.face_enhancer is not None
            and self.frame_number % config.FACE_ENHANCEMENT_INTERVAL_FRAMES == 0
        ):
            self.face_enhancer.submit(frame, people, self.frame_number)

        context_people = []
        for detection in detections:
            track_id = detection["track_id"]
            class_name = detection["class_name"]
            loitering = LoiteringState(active=False, started=False)
            fence_contact = False
            if class_name == "person" and track_id is None:
                context_people.append(
                    {
                        "track_id": None,
                        "status": "blocked",
                        "reason": "authoritative_tracker_not_acquired",
                        "source": detection.get("source", "unknown"),
                    }
                )

            if track_id is not None:
                fence_contact = self.fence.check_crossing(
                    track_id,
                    detection["centroid"],
                    timestamp,
                    bounds=detection["bbox"],
                )
                if fence_contact:
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
                    in_loitering_scope = point_in_polygon(
                        detection["centroid"], config.LOITERING_ZONE
                    )
                    loitering = self.loiter_detector.update(
                        track_id,
                        detection["centroid"],
                        timestamp,
                        in_scope=in_loitering_scope,
                    )
                    context_people.append(
                        self.loiter_detector.diagnostics(
                            track_id,
                            timestamp,
                            in_scope=in_loitering_scope,
                        )
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

                if self.risk_scorer is not None:
                    motion_score = 0.0
                    if class_name == "person" and self.motion_analyzer is not None:
                        motion_score = self.motion_analyzer.analyze(
                            self.loiter_detector.history(track_id), timestamp
                        ).score
                    fence_distance = point_to_segment_distance(
                        detection["centroid"], *config.VIRTUAL_FENCE_LINE
                    )
                    fence_proximity = max(
                        0.0, 1.0 - fence_distance / config.RISK_FENCE_PROXIMITY_PX
                    )
                    assessment = self.risk_scorer.update(
                        track_id,
                        RiskSignals(
                            fence_contact=float(self.fence.is_touching(track_id)),
                            fence_proximity=fence_proximity,
                            loitering=float(loitering.active),
                            motion_signature=motion_score,
                            low_posture=posture_signals.get(track_id, 0.0),
                            night=float(is_night),
                            fence_approach=(
                                self.group_detector.approach_score(track_id)
                                if self.group_detector
                                else 0.0
                            ),
                            group_membership=float(
                                self.group_detector is not None
                                and track_id in self.group_detector.grouped_track_ids
                            ),
                        ),
                        timestamp,
                    )
                    if assessment.crossed and self.alert_logger.log(
                        "CONTEXTUAL_RISK",
                        track_id,
                        assessment.details(),
                        self.frame_number,
                        source_time=timestamp,
                    ):
                        self._remember_alert(
                            f"CONTEXTUAL RISK: {class_name} #{track_id} {assessment.score:.2f}"
                        )

            plate_text, plate_bbox = None, None
            if (
                config.ENABLE_ANPR
                and class_name != "person"
                and track_id is not None
                and self._scheduled(track_id, config.ANPR_INTERVAL_FRAMES)
            ):
                observation = self.detector.detect_plate_observation(frame, detection["bbox"])
                if observation is not None:
                    consensus = self.anpr_consensus.add(track_id, observation, timestamp)
                    if consensus is not None:
                        self._stable_plates[track_id] = (
                            consensus.text,
                            self._relative_bbox(observation.bbox, detection["bbox"]),
                        )
                        if consensus.newly_stable:
                            self._anpr_stable_reads += 1
                        if consensus.newly_stable and self.alert_logger.log(
                            "ANPR_READ",
                            track_id,
                            f"{consensus.text} ({consensus.region})",
                            self.frame_number,
                            source_time=timestamp,
                        ):
                            self._remember_alert(f"PLATE: {consensus.text}")
                    elif track_id in self._stable_plates:
                        stable_text, _ = self._stable_plates[track_id]
                        self._stable_plates[track_id] = (
                            stable_text,
                            self._relative_bbox(observation.bbox, detection["bbox"]),
                        )
            if track_id is not None and track_id in self._stable_plates:
                plate_text, relative_bbox = self._stable_plates[track_id]
                plate_bbox = self._project_bbox(relative_bbox, detection["bbox"])

            # Tile detections are supplemental review proposals, not authoritative
            # tracks. Rendering them as normal green boxes is misleading.
            if self.show_overlays and should_draw_detection(detection):
                draw_detection(frame, detection, loitering.active)
                if plate_bbox is not None:
                    draw_plate(frame, plate_bbox, plate_text)
                center_x, center_y = map(int, detection["centroid"])
                cv2.circle(frame, (center_x, center_y), 3, (255, 255, 255), -1)

        self.fence.expire(timestamp)
        self.loiter_detector.expire(timestamp)
        if self.group_detector is not None:
            self.group_detector.expire(timestamp)
        if self.risk_scorer is not None:
            self.risk_scorer.expire(timestamp)
        for expired_track_id in self.anpr_consensus.expire(timestamp):
            self._stable_plates.pop(expired_track_id, None)
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
                "detection": self._detector_diagnostics(),
                "context_readiness": {
                    "status": (
                        "ready"
                        if any(item["track_id"] is not None for item in context_people)
                        else "blocked"
                    ),
                    "reason": (
                        None
                        if any(item["track_id"] is not None for item in context_people)
                        else (
                            "authoritative_tracker_not_acquired"
                            if context_people
                            else "no_person_detection"
                        )
                    ),
                    "people": context_people,
                },
                "anpr": {
                    **self.detector.anpr_diagnostics(),
                    "consensus_tracks": self.anpr_consensus.tracked_count,
                    "stable_reads": self._anpr_stable_reads,
                },
                "group_approach_tracks": (
                    self.group_detector.tracked_count if self.group_detector else 0
                ),
                "contextual_risk_tracks": (
                    self.risk_scorer.tracked_count if self.risk_scorer else 0
                ),
                "posture_status": self._posture_snapshot.get("status", "disabled"),
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
        self._publish_evaluation_frame(
            frame,
            detections,
            timestamp,
            is_night,
            time.perf_counter() - processing_started,
        )
        return frame

    def _publish_evaluation_frame(
        self,
        frame,
        detections: list[dict],
        source_time: float,
        is_night: bool,
        processing_seconds: float,
    ) -> None:
        if self._on_evaluation_frame is None:
            return
        height, width = frame.shape[:2]
        observation = {
            "frame_number": self.frame_number,
            "source_time_seconds": source_time,
            "processing_seconds": processing_seconds,
            "lighting": "low-light" if is_night else "daylight",
            "source_width": width,
            "source_height": height,
            "detections": [
                {
                    "class_name": detection["class_name"],
                    "bbox": list(detection["bbox"]),
                    "confidence": detection["conf"],
                    "track_id": detection["track_id"],
                    "source": detection.get("source", "full-frame"),
                }
                for detection in detections
            ],
            "detection_diagnostics": self._detector_diagnostics(),
            "context_readiness": self.stats["context_readiness"],
        }
        try:
            self._on_evaluation_frame(observation)
        except Exception:
            LOGGER.exception("Evaluation frame callback failed; primary analytics continue")

    def _detector_diagnostics(self) -> dict:
        snapshot = getattr(self.detector, "detection_diagnostics", None)
        if callable(snapshot):
            return snapshot()
        return {
            "tracker": config.YOLO_TRACKER,
            "configured_input_size": config.YOLO_INPUT_SIZE,
            "context_blocker": None,
        }

    def _update_posture_snapshot(self, snapshot: dict) -> None:
        with self._posture_lock:
            self._posture_snapshot = dict(snapshot)

    def _current_posture_signals(self) -> dict[int, float]:
        with self._posture_lock:
            snapshot_frame = self._posture_snapshot.get("frame_number")
            signals = list(self._posture_snapshot.get("signals", ()))
        if (
            not isinstance(snapshot_frame, int)
            or self.frame_number - snapshot_frame > config.POSTURE_RESULT_MAX_AGE_FRAMES
        ):
            return {}
        return {
            int(signal["track_id"]): float(signal.get("confidence", 0.0))
            for signal in signals
            if signal.get("sustained")
        }

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

    @staticmethod
    def _relative_bbox(plate_bbox, vehicle_bbox) -> tuple[float, float, float, float]:
        vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2 = vehicle_bbox
        width = max(vehicle_x2 - vehicle_x1, 1)
        height = max(vehicle_y2 - vehicle_y1, 1)
        plate_x1, plate_y1, plate_x2, plate_y2 = plate_bbox
        return (
            (plate_x1 - vehicle_x1) / width,
            (plate_y1 - vehicle_y1) / height,
            (plate_x2 - vehicle_x1) / width,
            (plate_y2 - vehicle_y1) / height,
        )

    @staticmethod
    def _project_bbox(relative_bbox, vehicle_bbox) -> tuple[int, int, int, int]:
        vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2 = vehicle_bbox
        width = max(vehicle_x2 - vehicle_x1, 1)
        height = max(vehicle_y2 - vehicle_y1, 1)
        rx1, ry1, rx2, ry2 = relative_bbox
        return (
            int(vehicle_x1 + rx1 * width),
            int(vehicle_y1 + ry1 * height),
            int(vehicle_x1 + rx2 * width),
            int(vehicle_y1 + ry2 * height),
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
        if self.posture_service is not None:
            self.posture_service.reset()
        self.fence.reset()
        self.loiter_detector.reset()
        if self.group_detector is not None:
            self.group_detector.reset()
        if self.risk_scorer is not None:
            self.risk_scorer.reset()
        self.anpr_consensus.reset()
        self._stable_plates.clear()
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
        if self.posture_service is not None:
            self.posture_service.close()
        close_detector = getattr(self.detector, "close", None)
        if callable(close_detector):
            close_detector()
        self.alert_logger.close()
