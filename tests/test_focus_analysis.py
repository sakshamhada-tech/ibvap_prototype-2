import sys
from types import SimpleNamespace

import config
from focus_analysis import (
    FocusConfirmationTracker,
    NativeFocusService,
    expand_focus_bbox,
    focused_detection_from_result,
    target_pixel_height_band,
)


class Values:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class Boxes:
    def __init__(self, boxes, class_ids, confidences):
        self.xyxy = Values(boxes)
        self.cls = Values(class_ids)
        self.conf = Values(confidences)

    def __len__(self):
        return len(self.xyxy.values)


class Frame:
    shape = (100, 200, 3)
    size = 1

    def copy(self):
        return self

    def __getitem__(self, _item):
        return self


class FocusModel:
    def predict(self, _crop, **_kwargs):
        boxes = Boxes(((10, 20, 30, 60),), (0,), (0.91,))
        return [SimpleNamespace(boxes=boxes)]


def test_focus_crop_padding_clamps_to_untouched_source_frame():
    assert expand_focus_bbox((0, 0, 20, 30), 200, 100, 0.5, 80) == (0, 0, 80, 80)
    assert expand_focus_bbox((180, 80, 200, 100), 200, 100, 0.5, 80) == (120, 20, 200, 100)


def test_target_pixel_height_bands_match_the_redesign_evaluation_protocol():
    assert [target_pixel_height_band(height) for height in (24, 25, 48, 49, 96, 97)] == [
        "<=24 px",
        "25-48 px",
        "25-48 px",
        "49-96 px",
        "49-96 px",
        ">96 px",
    ]


def test_focus_confirmation_uses_ephemeral_episodes_not_track_ids():
    tracker = FocusConfirmationTracker(required_observations=2, iou_threshold=0.2, max_gap_frames=3)
    candidate = {"class_id": 0, "conf": 0.8, "bbox": (10, 10, 20, 30), "track_id": None}

    first = tracker.update([candidate], 1)[0]
    second = tracker.update([{**candidate, "bbox": (11, 10, 21, 30)}], 2)[0]

    assert first["confirmed"] is False
    assert second["confirmed"] is True
    assert second["newly_confirmed"] is True
    assert second["track_id"] is None
    assert second["candidate_episode_id"] == first["candidate_episode_id"]
    tracker.expire(6)
    assert tracker.active_count == 0
    tracker.reset()
    assert tracker.active_count == 0


def test_focus_result_remaps_crop_detection_to_source_coordinates():
    result = SimpleNamespace(boxes=Boxes(((5, 6, 15, 26),), (0,), (0.75,)))
    candidate = {
        "class_id": 0,
        "class_name": "person",
        "source": "tile",
        "source_authoritative_track_id": 17,
        "bbox": (100, 200, 120, 230),
    }

    detection = focused_detection_from_result(result, (100, 200, 300, 400), candidate)

    assert detection["bbox"] == (105, 206, 115, 226)
    assert detection["source_candidate"] == "tile"
    assert detection["source_authoritative_track_id"] == 17
    assert detection["track_id"] is None
    assert focused_detection_from_result(result, (300, 300, 500, 500), candidate, 0.05) is None


def test_native_focus_process_confirms_source_crop_without_inventing_identity(monkeypatch):
    monkeypatch.setattr(config, "NATIVE_FOCUS_MAX_REGIONS", 1)
    monkeypatch.setattr(config, "NATIVE_FOCUS_PADDING_RATIO", 0.5)
    monkeypatch.setattr(config, "NATIVE_FOCUS_MIN_CROP_SIZE_PX", 40)
    service = NativeFocusService()
    service._model = FocusModel()
    monkeypatch.setattr(service, "_encode_source_crop", lambda _crop: b"source-jpeg")
    candidate = {
        "class_id": 0,
        "class_name": "person",
        "conf": 0.4,
        "bbox": (50, 20, 70, 60),
        "source": "tile",
        "track_id": None,
    }

    first = service._process(Frame(), [candidate], 1)
    second = service._process(Frame(), [candidate], 2)

    assert first["confirmed"] is False
    assert second["confirmed"] is True
    assert second["acquisition_state"] == "tracker_requested"
    assert second["track_id"] is None
    assert second["source_target_width"] == 20
    assert second["source_target_height"] == 40
    assert second["source_target_height_band"] == "25-48 px"
    assert second["image_jpeg"] == b"source-jpeg"


def test_native_focus_auto_device_does_not_implicitly_use_mps(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert NativeFocusService._select_device("auto") == "cpu"
    assert NativeFocusService._select_device("mps") == "mps"


def test_native_focus_initialization_failure_is_status_only(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_NATIVE_FOCUS", True)
    snapshots = []
    service = NativeFocusService(on_snapshot=snapshots.append)

    def fail_model_load():
        raise RuntimeError("fake model unavailable")

    monkeypatch.setattr(service, "_load_model", fail_model_load)
    service.start()
    service._thread.join(timeout=1)

    assert service.snapshot()["status"] == "unavailable"
    assert service.snapshot()["errors"] == 1
    assert "fake model unavailable" in service.snapshot()["message"]
    assert snapshots[-1]["status"] == "unavailable"
    service.close()


def test_native_focus_queue_is_single_slot_and_drops_when_busy(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_NATIVE_FOCUS", True)
    service = NativeFocusService()
    service._accepting_jobs = True
    candidate = [{"class_id": 0, "class_name": "person", "conf": 0.5, "bbox": (1, 1, 5, 8)}]

    assert service.submit(Frame(), candidate, 1) is True
    assert service.submit(Frame(), candidate, 2) is False
    assert service.snapshot()["dropped_busy"] == 1
