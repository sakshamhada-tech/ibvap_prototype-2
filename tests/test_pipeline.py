import sys
import types

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


class FakeFrame:
    size = 1
    shape = (100, 100, 3)


class FakeDetector:
    def __init__(self):
        self.face_calls = 0
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

    def detect_faces(self, _frame, _bbox):
        self.face_calls += 1
        return []

    def detect_plate(self, _frame, _bbox):
        raise AssertionError("ANPR is disabled")

    def reset_tracking(self):
        self.reset_calls += 1


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
    monkeypatch.setattr(config, "FACE_DETECTION_INTERVAL_FRAMES", 3)


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
    assert video_pipeline.detector.face_calls == 1
    video_pipeline.close()


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
