import hashlib
import sys
import types

import pytest

# The analytics orchestration is tested with lightweight detector/CV doubles;
# CI does not need to download PyTorch merely to exercise state transitions.
fake_cv2 = types.ModuleType("cv2")
fake_cv2.COLOR_BGR2GRAY = 1
fake_cv2.FONT_HERSHEY_SIMPLEX = 1
fake_cv2.cvtColor = lambda frame, _mode: frame
fake_cv2.rectangle = lambda *args, **kwargs: None
fake_cv2.putText = lambda *args, **kwargs: None
fake_cv2.line = lambda *args, **kwargs: None
fake_cv2.circle = lambda *args, **kwargs: None
fake_cv2.data = types.SimpleNamespace(haarcascades="")
sys.modules.setdefault("cv2", fake_cv2)

fake_numpy = types.ModuleType("numpy")
fake_numpy.mean = lambda _value: 100.0
sys.modules.setdefault("numpy", fake_numpy)

fake_ultralytics = types.ModuleType("ultralytics")
fake_ultralytics.YOLO = object
sys.modules.setdefault("ultralytics", fake_ultralytics)

import config  # noqa: E402
import pipeline  # noqa: E402
from detector import Detector as ActualDetector  # noqa: E402
from utils.anpr import PlateObservation  # noqa: E402


class FakeFrame:
    size = 1
    shape = (100, 100, 3)


class FakeFaceEnhancementService:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.submissions = []
        self.reset_calls = 0
        self.closed = False

    def start(self):
        self.started = True

    def submit(self, frame, people, frame_number):
        self.submissions.append((frame, people, frame_number))
        return True

    def reset(self):
        self.reset_calls += 1

    def close(self):
        self.closed = True


class FakeDetector:
    def __init__(self):
        self.reset_calls = 0
        self.bbox = (0, 0, 20, 20)

    def track_frame(self, _frame):
        x1, y1, x2, y2 = self.bbox
        return [
            {
                "track_id": 1,
                "class_id": 0,
                "class_name": "person",
                "bbox": self.bbox,
                "centroid": ((x1 + x2) / 2, (y1 + y2) / 2),
                "conf": 0.9,
            }
        ]

    def detect_plate(self, _frame, _bbox):
        raise AssertionError("ANPR is disabled")

    def anpr_diagnostics(self):
        return {
            "status": "ready" if config.ENABLE_ANPR else "disabled",
            "attempts": 0,
            "valid_observations": 0,
            "last_result": "waiting" if config.ENABLE_ANPR else "disabled",
        }

    def reset_tracking(self):
        self.reset_calls += 1


class FakeVehicleDetector(FakeDetector):
    def __init__(self):
        super().__init__()
        self.observations = []

    def track_frame(self, _frame):
        x1, y1, x2, y2 = self.bbox
        return [
            {
                "track_id": 5,
                "class_id": 2,
                "class_name": "car",
                "bbox": self.bbox,
                "centroid": ((x1 + x2) / 2, (y1 + y2) / 2),
                "conf": 0.9,
            }
        ]

    def detect_plate_observation(self, _frame, _bbox):
        return self.observations.pop(0) if self.observations else None


def configure_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "Detector", FakeDetector)
    monkeypatch.setattr(pipeline, "frame_brightness", lambda _frame: 100.0)
    monkeypatch.setattr(config, "ALERT_LOG_CSV", str(tmp_path / "alerts.csv"))
    monkeypatch.setattr(config, "VIRTUAL_FENCE_LINE", ((0, 50), (99, 50)))
    monkeypatch.setattr(config, "LOITERING_SECONDS", 2)
    monkeypatch.setattr(config, "LOITERING_MOVEMENT_THRESHOLD_PX", 5)
    monkeypatch.setattr(config, "TRACK_MAX_OBSERVATION_GAP_SECONDS", 1.1)
    monkeypatch.setattr(config, "TRACK_STALE_SECONDS", 2)
    monkeypatch.setattr(config, "ENABLE_ANPR", False)
    monkeypatch.setattr(config, "ENABLE_FACE_ENHANCEMENT", False)


def test_pipeline_uses_source_time_and_persists_active_loitering(monkeypatch, tmp_path):
    configure_pipeline(monkeypatch, tmp_path)
    alerts = []
    video_pipeline = pipeline.VideoPipeline(on_alert=alerts.append, show_overlays=False)
    frame = FakeFrame()

    video_pipeline.process_frame(frame, source_time=0)
    video_pipeline.process_frame(frame, source_time=1)
    video_pipeline.process_frame(frame, source_time=2)
    video_pipeline.process_frame(frame, source_time=3)

    loitering = [alert for alert in alerts if alert["alert_type"] == "SUSPICIOUS_LOITERING"]
    assert len(loitering) == 1
    assert loitering[0]["source_time_seconds"] == 2.0
    assert video_pipeline.loiter_detector.update(1, (10, 10), 3.5).active
    video_pipeline.close()


def test_pipeline_submits_raw_people_to_optional_face_worker(monkeypatch, tmp_path):
    configure_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "ENABLE_FACE_ENHANCEMENT", True)
    monkeypatch.setattr(config, "FACE_ENHANCEMENT_INTERVAL_FRAMES", 1)
    monkeypatch.setattr(pipeline, "FaceEnhancementService", FakeFaceEnhancementService)
    snapshots = []
    video_pipeline = pipeline.VideoPipeline(
        on_face_snapshot=snapshots.append,
        show_overlays=False,
    )
    frame = FakeFrame()

    video_pipeline.process_frame(frame, source_time=0)
    assert video_pipeline.face_enhancer.started
    assert video_pipeline.face_enhancer.submissions == [(frame, [(1, (0, 0, 20, 20))], 1)]
    video_pipeline.reset_tracking()
    assert video_pipeline.face_enhancer.reset_calls == 1
    video_pipeline.close()
    assert video_pipeline.face_enhancer.closed


def test_pipeline_flags_person_box_contact_before_centroid_crosses(monkeypatch, tmp_path):
    configure_pipeline(monkeypatch, tmp_path)
    alerts = []
    video_pipeline = pipeline.VideoPipeline(on_alert=alerts.append, show_overlays=False)
    frame = FakeFrame()

    video_pipeline.process_frame(frame, source_time=0)
    video_pipeline.detector.bbox = (0, 0, 20, 50)
    video_pipeline.process_frame(frame, source_time=1)
    video_pipeline.process_frame(frame, source_time=2)

    intrusions = [alert for alert in alerts if alert["alert_type"] == "VIRTUAL_FENCE_INTRUSION"]
    assert len(intrusions) == 1
    assert intrusions[0]["source_time_seconds"] == 1.0
    assert video_pipeline.detector.bbox[3] == 50
    assert (video_pipeline.detector.bbox[1] + video_pipeline.detector.bbox[3]) / 2 < 50
    video_pipeline.close()


def test_pipeline_resets_tracking_when_source_time_moves_backwards(monkeypatch, tmp_path):
    configure_pipeline(monkeypatch, tmp_path)
    video_pipeline = pipeline.VideoPipeline(show_overlays=False)
    frame = FakeFrame()
    video_pipeline.process_frame(frame, source_time=5)
    video_pipeline.process_frame(frame, source_time=1)
    assert video_pipeline.detector.reset_calls == 1
    assert video_pipeline.loiter_detector.tracked_count == 1
    video_pipeline.close()


def test_pipeline_emits_only_stable_tracked_anpr_reads_and_resets_state(monkeypatch, tmp_path):
    configure_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "Detector", FakeVehicleDetector)
    monkeypatch.setattr(config, "ENABLE_ANPR", True)
    monkeypatch.setattr(config, "ANPR_INTERVAL_FRAMES", 1)
    monkeypatch.setattr(config, "ANPR_MIN_CONSENSUS_READS", 3)
    monkeypatch.setattr(config, "ANPR_CONSENSUS_WINDOW_SECONDS", 3)
    monkeypatch.setattr(config, "ANPR_MIN_CONSENSUS_WEIGHT_RATIO", 0.6)
    monkeypatch.setattr(config, "ANPR_MIN_OCR_CONFIDENCE", 0.5)
    monkeypatch.setattr(config, "ANPR_MAX_OBSERVATIONS_PER_TRACK", 6)
    monkeypatch.setattr(config, "ANPR_STATE_TTL_SECONDS", 5)
    alerts = []
    video_pipeline = pipeline.VideoPipeline(on_alert=alerts.append, show_overlays=False)
    plate = PlateObservation("KA01AB1234", (4, 10, 18, 16), "IN", 0.9, 0.8, 100)
    video_pipeline.detector.observations = [plate, plate, plate, plate]

    for timestamp in (0, 0.5):
        video_pipeline.process_frame(FakeFrame(), source_time=timestamp)
    assert not [alert for alert in alerts if alert["alert_type"] == "ANPR_READ"]

    video_pipeline.process_frame(FakeFrame(), source_time=1)
    video_pipeline.process_frame(FakeFrame(), source_time=1.5)
    anpr_alerts = [alert for alert in alerts if alert["alert_type"] == "ANPR_READ"]
    assert len(anpr_alerts) == 1
    assert anpr_alerts[0]["details"] == "KA01AB1234 (IN)"
    assert video_pipeline._stable_plates[5][0] == "KA01AB1234"
    assert video_pipeline.stats["anpr"]["stable_reads"] == 1

    video_pipeline.reset_tracking()
    assert video_pipeline.anpr_consensus.tracked_count == 0
    assert video_pipeline._stable_plates == {}
    video_pipeline.close()


def test_relative_plate_bbox_follows_a_moving_vehicle():
    relative = pipeline.VideoPipeline._relative_bbox((20, 30, 40, 40), (10, 10, 50, 50))
    assert pipeline.VideoPipeline._project_bbox(relative, (30, 20, 110, 100)) == (50, 60, 90, 80)


def test_anpr_model_file_requires_presence_and_matching_sha256(tmp_path):
    model_path = tmp_path / "plate.pt"
    with pytest.raises(RuntimeError, match="missing"):
        ActualDetector._verify_model_file(model_path, "0" * 64, "ANPR", required=True)

    model_path.write_bytes(b"reviewed checkpoint")
    with pytest.raises(RuntimeError, match="requires a configured SHA-256"):
        ActualDetector._verify_model_file(model_path, "", "ANPR", required=True)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        ActualDetector._verify_model_file(model_path, "0" * 64, "ANPR", required=True)

    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    ActualDetector._verify_model_file(model_path, digest, "ANPR", required=True)


def test_anpr_diagnostics_distinguish_region_confidence_and_empty_ocr(monkeypatch):
    class Reader:
        def __init__(self, results):
            self.results = results

        def readtext(self, _image, **_kwargs):
            return self.results

    box = [[0, 0], [20, 0], [20, 10], [0, 10]]
    detector = ActualDetector.__new__(ActualDetector)
    detector._anpr_metrics = ActualDetector._new_anpr_metrics()
    monkeypatch.setattr(ActualDetector, "_plate_variants", lambda _self, _crop: (object(),))
    monkeypatch.setattr(config, "ANPR_REGIONS", ("IN",))
    monkeypatch.setattr(config, "ANPR_MIN_OCR_CONFIDENCE", 0.5)

    detector.ocr_reader = Reader([])
    assert detector._read_plate_variants(object()) is None
    detector.ocr_reader = Reader([(box, "NOT-A-PLATE", 0.9)])
    assert detector._read_plate_variants(object()) is None
    detector.ocr_reader = Reader([(box, "KA01AB1234", 0.2)])
    assert detector._read_plate_variants(object()) is None

    metrics = detector.anpr_diagnostics()
    assert metrics["ocr_no_text"] == 1
    assert metrics["region_rejected"] == 1
    assert metrics["ocr_low_confidence"] == 1
    assert metrics["last_result"] == "ocr_low_confidence"


def test_loitering_is_wide_area_and_not_fence_gated(monkeypatch, tmp_path):
    configure_pipeline(monkeypatch, tmp_path)
    alerts = []
    video_pipeline = pipeline.VideoPipeline(on_alert=alerts.append, show_overlays=False)
    frame = FakeFrame()

    for timestamp in (0, 1, 2, 3):
        video_pipeline.process_frame(frame, source_time=timestamp)

    alert_types = [alert["alert_type"] for alert in alerts]
    assert "SUSPICIOUS_LOITERING" in alert_types
    assert "VIRTUAL_FENCE_INTRUSION" not in alert_types
    video_pipeline.close()


class FakeGroupDetector(FakeDetector):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def track_frame(self, _frame):
        top = self.calls * 15
        self.calls += 1
        return [
            {
                "track_id": track_id,
                "class_id": 0,
                "class_name": "person",
                "bbox": (left, top, left + 10, top + 10),
                "centroid": (left + 5, top + 5),
                "conf": 0.9,
            }
            for track_id, left in ((1, 10), (2, 25), (3, 40))
        ]


class FakeFirearmDetector(FakeDetector):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def track_frame(self, _frame):
        self.calls += 1
        return [
            {
                "track_id": 9,
                "class_id": 0,
                "class_name": "person",
                "bbox": (0, 0, 30, 40),
                "centroid": (15, 20),
                "conf": 0.9,
            }
        ]

    def firearm_snapshot(self):
        return {
            "status": "ready",
            "frame_number": self.calls,
            "detections": [{"bbox": (10, 10, 20, 20), "confidence": 0.91}],
        }


def test_pipeline_emits_group_approach_once(monkeypatch, tmp_path):
    configure_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "Detector", FakeGroupDetector)
    monkeypatch.setattr(config, "ENABLE_GROUP_APPROACH", True)
    monkeypatch.setattr(config, "GROUP_MIN_SIZE", 3)
    monkeypatch.setattr(config, "GROUP_PROXIMITY_PX", 50)
    monkeypatch.setattr(config, "GROUP_APPROACH_WINDOW_SECONDS", 3)
    monkeypatch.setattr(config, "GROUP_MIN_SPEED_PX_PER_SECOND", 2)
    monkeypatch.setattr(config, "GROUP_HEADING_SIMILARITY", 0.8)
    monkeypatch.setattr(config, "GROUP_MIN_FENCE_PROGRESS_PX", 5)
    alerts = []
    video_pipeline = pipeline.VideoPipeline(on_alert=alerts.append, show_overlays=False)

    for timestamp in (0, 1, 2):
        video_pipeline.process_frame(FakeFrame(), source_time=timestamp)

    group_alerts = [alert for alert in alerts if alert["alert_type"] == "GROUP_APPROACH"]
    assert len(group_alerts) == 1
    assert "tracks 1,2,3" in group_alerts[0]["details"]
    video_pipeline.close()


def test_pipeline_emits_conservative_firearm_review_once(monkeypatch, tmp_path):
    configure_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "Detector", FakeFirearmDetector)
    alerts = []
    video_pipeline = pipeline.VideoPipeline(on_alert=alerts.append, show_overlays=False)

    video_pipeline.process_frame(FakeFrame(), source_time=0)
    video_pipeline.process_frame(FakeFrame(), source_time=1)

    firearm_alerts = [alert for alert in alerts if alert["alert_type"] == "FIREARM_DETECTED"]
    assert len(firearm_alerts) == 1
    assert "human review only" in firearm_alerts[0]["details"]
    video_pipeline.close()


def test_pipeline_contextual_risk_includes_explainable_signals(monkeypatch, tmp_path):
    configure_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "Detector", FakeGroupDetector)
    monkeypatch.setattr(config, "ENABLE_CONTEXTUAL_RISK", True)
    monkeypatch.setattr(config, "RISK_ALERT_THRESHOLD", 0.3)
    monkeypatch.setattr(config, "GROUP_MIN_SIZE", 3)
    monkeypatch.setattr(config, "GROUP_PROXIMITY_PX", 50)
    monkeypatch.setattr(config, "GROUP_APPROACH_WINDOW_SECONDS", 3)
    monkeypatch.setattr(config, "GROUP_MIN_SPEED_PX_PER_SECOND", 2)
    monkeypatch.setattr(config, "GROUP_HEADING_SIMILARITY", 0.8)
    monkeypatch.setattr(config, "GROUP_MIN_FENCE_PROGRESS_PX", 5)
    alerts = []
    video_pipeline = pipeline.VideoPipeline(on_alert=alerts.append, show_overlays=False)

    video_pipeline.process_frame(FakeFrame(), source_time=0)
    video_pipeline.process_frame(FakeFrame(), source_time=1)

    contextual = [alert for alert in alerts if alert["alert_type"] == "CONTEXTUAL_RISK"]
    assert contextual
    assert "signals:" in contextual[0]["details"]
    assert "group_membership" in contextual[0]["details"]
    video_pipeline.close()
