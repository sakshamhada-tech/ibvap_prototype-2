import sys
import threading
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
import detector as detector_module  # noqa: E402
from detector import Detector, TiledInferenceWorker  # noqa: E402


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

    def copy(self):
        return self

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


def test_tiled_inference_skips_duplicate_full_frame_tile(monkeypatch):
    class SmallFrame(Frame):
        shape = (100, 100, 3)

    class UnexpectedModel:
        def predict(self, *_args, **_kwargs):
            raise AssertionError("a full-frame tile has no magnification benefit")

    detector = Detector.__new__(Detector)
    detector.tile_model = UnexpectedModel()
    monkeypatch.setattr(config, "TILE_SIZE", 100)

    assert detector._tiled_detections(SmallFrame()) == []


def test_tiled_inference_rotates_a_bounded_tile_subset(monkeypatch):
    class WideFrame(Frame):
        shape = (100, 350, 3)

    detector = Detector.__new__(Detector)
    detector.tile_model = TileModel()
    detector._tile_cursor = 0
    monkeypatch.setattr(config, "TILE_SIZE", 100)
    monkeypatch.setattr(config, "TILE_OVERLAP", 0.0)
    monkeypatch.setattr(config, "TILED_INFERENCE_MAX_TILES_PER_CYCLE", 2)

    first = detector._tiled_detections(WideFrame())
    second = detector._tiled_detections(WideFrame())

    assert [item["bbox"] for item in first] == [(10, 10, 20, 20), (110, 10, 120, 20)]
    assert [item["bbox"] for item in second] == [(210, 10, 220, 20), (260, 10, 270, 20)]


def test_tiled_worker_queue_is_single_slot_and_drops_when_busy():
    worker = TiledInferenceWorker(lambda _model, _frame: [], lambda *_args: None)
    worker._accepting_jobs = True

    assert worker.submit(Frame(), 1) is True
    assert worker.submit(Frame(), 2) is False
    assert worker.snapshot()["dropped_busy"] == 1
    worker.close()


def test_tiled_worker_processes_exact_frame_off_the_caller_thread(monkeypatch):
    model_ready = threading.Event()
    result_ready = threading.Event()
    callbacks = []

    class Model:
        def __init__(self, _path):
            model_ready.set()

    def on_result(frame, detections, frame_number):
        callbacks.append((frame, detections, frame_number, threading.current_thread().name))
        result_ready.set()

    monkeypatch.setattr(detector_module, "YOLO", Model)
    worker = TiledInferenceWorker(
        lambda _model, _frame: [{"source": "tile"}],
        on_result,
    )
    worker.start()
    assert model_ready.wait(1)
    worker._accepting_jobs = True
    frame = Frame()
    assert worker.submit(frame, 8)
    assert result_ready.wait(1)

    assert callbacks == [(frame, [{"source": "tile"}], 8, "ibvap-tiled-inference")]
    assert worker.snapshot()["completed"] == 1
    worker.close()


def test_tiled_worker_initialization_failure_is_status_only(monkeypatch):
    def fail_model_load(_path):
        raise RuntimeError("fake tile model unavailable")

    monkeypatch.setattr(detector_module, "YOLO", fail_model_load)
    worker = TiledInferenceWorker(lambda _model, _frame: [], lambda *_args: None)
    worker.start()
    worker._thread.join(timeout=1)

    assert worker.snapshot()["status"] == "unavailable"
    assert worker.snapshot()["errors"] == 1
    assert "fake tile model unavailable" in worker.snapshot()["message"]
    worker.close()


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


def test_live_tiling_submits_background_job_without_merging_green_boxes(monkeypatch):
    boxes = SimpleNamespace(
        xyxy=Values(((1, 2, 11, 22),)),
        cls=Values((0,)),
        conf=Values((0.9,)),
        id=Values((7,)),
    )

    class TrackingModel:
        def track(self, _frame, **_kwargs):
            return [SimpleNamespace(boxes=boxes)]

    class TileWorker:
        def __init__(self):
            self.submissions = []

        def submit(self, frame, frame_number):
            self.submissions.append((frame, frame_number))
            return True

        def snapshot(self):
            return {"status": "ready", "last_proposals": 3}

    detector = Detector.__new__(Detector)
    detector.model = TrackingModel()
    detector.tile_model = None
    detector.tile_worker = TileWorker()
    detector.focus_service = None
    detector._frame_number = 0
    detector._focus_boost_remaining = 0
    detector._focus_snapshot = {"status": "disabled", "focus": None}
    monkeypatch.setattr(config, "TILED_INFERENCE_INTERVAL_FRAMES", 1)

    frame = Frame()
    detections = detector.track_frame(frame)

    assert detector.tile_worker.submissions == [(frame, 1)]
    assert len(detections) == 1
    assert detections[0]["source"] == "full-frame"
    assert detections[0]["track_id"] == 7
    assert detector.detection_diagnostics()["tiled_inference"]["last_proposals"] == 3


def test_background_tile_results_feed_focus_without_entering_primary_detections():
    class FocusService:
        def __init__(self):
            self.submissions = []

        def submit(self, frame, candidates, frame_number):
            self.submissions.append((frame, candidates, frame_number))
            return True

    detector = Detector.__new__(Detector)
    detector.focus_service = FocusService()
    frame = Frame()
    candidates = [{"source": "tile", "track_id": None}]

    detector._handle_tiled_result(frame, candidates, 12)

    assert detector.focus_service.submissions == [(frame, candidates, 12)]


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
