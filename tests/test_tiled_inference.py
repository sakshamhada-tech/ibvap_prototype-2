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

    class TrackingModel:
        def track(self, _frame, **_kwargs):
            return [SimpleNamespace(boxes=boxes)]

    class FailingTileModel:
        def predict(self, _tile, **_kwargs):
            raise RuntimeError("optional tile failure")

    detector = Detector.__new__(Detector)
    detector.model = TrackingModel()
    detector.tile_model = FailingTileModel()
    detector.firearm_service = None
    detector._frame_number = 0
    detector._tiled_error_active = False
    monkeypatch.setattr(config, "TILED_INFERENCE_INTERVAL_FRAMES", 1)
    detections = detector.track_frame(Frame())
    assert detections[0]["track_id"] == 7
    assert detections[0]["source"] == "full-frame"


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
