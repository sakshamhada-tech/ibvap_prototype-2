import sys
import types
from types import SimpleNamespace

fake_cv2 = types.ModuleType("cv2")
fake_numpy = types.ModuleType("numpy")
fake_ultralytics = types.ModuleType("ultralytics")
fake_ultralytics.YOLO = object
sys.modules.setdefault("cv2", fake_cv2)
sys.modules.setdefault("numpy", fake_numpy)
sys.modules.setdefault("ultralytics", fake_ultralytics)

import config  # noqa: E402
from detector import Detector  # noqa: E402


class Values:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class Frame:
    shape = (100, 150, 3)
    size = 1

    def __getitem__(self, _item):
        return self


class TileModel:
    def predict(self, _tile, **_kwargs):
        boxes = SimpleNamespace(
            xyxy=Values(((10, 10, 20, 20),)),
            cls=Values((0,)),
            conf=Values((0.9,)),
            id=None,
        )
        return [SimpleNamespace(boxes=boxes)]


def test_tile_grid_covers_edges_with_overlap():
    bounds = Detector.tile_bounds(150, 100, 100, 0.5)
    assert bounds == ((0, 0, 100, 100), (50, 0, 150, 100))


def test_tiled_inference_remaps_tile_coordinates(monkeypatch):
    detector = Detector.__new__(Detector)
    detector.tile_model = TileModel()
    monkeypatch.setattr(config, "TILE_SIZE", 100)
    monkeypatch.setattr(config, "TILE_OVERLAP", 0.5)
    monkeypatch.setattr(config, "TILED_INFERENCE_NMS_IOU", 0.5)
    detections = detector._tiled_detections(Frame())
    assert [detection["bbox"] for detection in detections] == [
        (10, 10, 20, 20),
        (60, 10, 70, 20),
    ]


def test_tiled_failure_preserves_full_frame_tracking(monkeypatch):
    boxes = SimpleNamespace(
        xyxy=Values(((1, 2, 11, 22),)),
        cls=Values((0,)),
        conf=Values((0.9,)),
        id=Values((7,)),
    )

    tracking_calls = []

    class TrackingModel:
        def track(self, _frame, **kwargs):
            tracking_calls.append(kwargs)
            return [SimpleNamespace(boxes=boxes)]

    class FailingTileModel:
        def predict(self, _tile, **_kwargs):
            raise RuntimeError("optional tile failure")

    detector = Detector.__new__(Detector)
    detector.model = TrackingModel()
    detector.tile_model = FailingTileModel()
    detector._frame_number = 0
    detector._tiled_error_active = False
    monkeypatch.setattr(config, "TILED_INFERENCE_INTERVAL_FRAMES", 1)
    detections = detector.track_frame(Frame())
    assert detections[0]["track_id"] == 7
    assert detections[0]["source"] == "full-frame"
    assert tracking_calls[0]["tracker"] == "bytetrack.yaml"
    assert tracking_calls[0]["imgsz"] == config.YOLO_INPUT_SIZE
    assert tracking_calls[0]["device"] == "cpu"


def test_confirmed_focus_requests_bounded_tracker_boost_and_only_tracker_assigns_id():
    boxes = SimpleNamespace(
        xyxy=Values(((11, 12, 31, 42),)),
        cls=Values((0,)),
        conf=Values((0.85,)),
        id=Values((9,)),
    )
    tracking_calls = []

    class TrackingModel:
        def track(self, _frame, **kwargs):
            tracking_calls.append(kwargs)
            return [SimpleNamespace(boxes=boxes)]

    snapshots = []
    detector = Detector.__new__(Detector)
    detector.model = TrackingModel()
    detector.tile_model = None
    detector.focus_service = None
    detector._frame_number = 10
    detector._focus_boost_remaining = 0
    detector._acquired_focus_episodes = {}
    detector._on_focus_snapshot = snapshots.append
    detector._focus_snapshot = {"status": "ready", "focus": None}
    detector._handle_focus_snapshot(
        {
            "status": "ready",
            "frame_number": 10,
            "focus": {
                "candidate_episode_id": 2,
                "class_id": 0,
                "class_name": "person",
                "bbox": (10, 10, 30, 40),
                "conf": 0.8,
                "confirmed": True,
                "newly_confirmed": True,
                "track_id": None,
                "acquisition_state": "tracker_requested",
            },
        }
    )

    detections = detector.track_frame(Frame())

    assert tracking_calls[0]["imgsz"] == config.NATIVE_FOCUS_TRACK_INPUT_SIZE
    assert tracking_calls[0]["tracker"] == "bytetrack.yaml"
    assert tracking_calls[0]["device"] == "cpu"
    assert detections[0]["track_id"] == 9
    assert detector.focus_snapshot()["focus"]["authoritative_track_id"] == 9
    assert detector.focus_snapshot()["focus"]["acquisition_state"] == "tracker_acquired"
    assert detector.detection_diagnostics()["tracking_boost_active"] is True
    assert detector._focus_boost_remaining == 0
    assert snapshots[-1]["focus"]["track_id"] is None


def test_detector_diagnostics_explain_untracked_context_blocker_and_provenance():
    detector = Detector.__new__(Detector)
    detector._focus_boost_remaining = 0
    detector._focus_snapshot = {"status": "disabled", "focus": None}
    detector._record_detection_diagnostics(
        [
            {"track_id": None, "class_id": 0, "source": "full-frame"},
            {"track_id": None, "class_id": 0, "source": "tile"},
        ],
        640,
        False,
    )

    diagnostics = detector.detection_diagnostics()
    assert diagnostics["tracked_full_frame"] == 0
    assert diagnostics["untracked_full_frame"] == 1
    assert diagnostics["untracked_tile"] == 1
    assert diagnostics["context_blocker"] == "authoritative_tracker_not_acquired"
    assert diagnostics["provenance"]["authoritative"] == "full-frame ByteTrack only"


def test_nms_merges_duplicate_tile_boundary_detections():
    first = {
        "class_id": 0,
        "conf": 0.9,
        "bbox": (10, 10, 30, 30),
    }
    duplicate = {
        "class_id": 0,
        "conf": 0.8,
        "bbox": (11, 11, 31, 31),
    }
    assert Detector._nms([duplicate, first], 0.5) == [first]
