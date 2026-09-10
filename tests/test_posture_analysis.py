import threading
from types import SimpleNamespace

import posture_analysis
from posture_analysis import PostureAnalysisService, classify_low_posture


class CopyableFrame:
    def copy(self):
        return self


def make_service(tmp_path, callback):
    return PostureAnalysisService(
        model_path=str(tmp_path / "pose.pt"),
        model_sha256="0" * 64,
        confidence_threshold=0.6,
        minimum_person_size_px=50,
        maximum_people=3,
        minimum_visible_keypoints=3,
        minimum_keypoint_confidence=0.4,
        minimum_aspect_ratio=0.9,
        maximum_vertical_spread_ratio=0.55,
        sustain_seconds=1,
        state_ttl_seconds=3,
        device="cpu",
        on_snapshot=callback,
    )


def test_standing_and_prone_geometry_are_distinguished():
    standing, standing_score = classify_low_posture(
        (0, 0, 50, 150),
        ((25, 10), (20, 70), (25, 140)),
        (0.9, 0.9, 0.9),
        minimum_visible_keypoints=3,
        minimum_keypoint_confidence=0.4,
        minimum_aspect_ratio=0.9,
        maximum_vertical_spread_ratio=0.55,
    )
    prone, prone_score = classify_low_posture(
        (0, 0, 150, 60),
        ((10, 25), (70, 30), (140, 35)),
        (0.9, 0.9, 0.9),
        minimum_visible_keypoints=3,
        minimum_keypoint_confidence=0.4,
        minimum_aspect_ratio=0.9,
        maximum_vertical_spread_ratio=0.55,
    )
    assert not standing and standing_score == 0
    assert prone and prone_score > 0


def test_posture_auto_device_does_not_implicitly_use_mps():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )

    assert posture_analysis._select_device(fake_torch, "auto") == "cpu"
    assert posture_analysis._select_device(fake_torch, "mps") == "mps"


def test_missing_pose_model_is_failure_isolated(tmp_path):
    snapshots = []
    unavailable = threading.Event()

    def callback(snapshot):
        snapshots.append(snapshot)
        if snapshot["status"] == "unavailable":
            unavailable.set()

    service = make_service(tmp_path, callback)
    service.start()
    assert unavailable.wait(2)
    assert not service.submit(CopyableFrame(), [(1, (0, 0, 100, 100))], 1, 0)
    service.close()
    assert [snapshot["status"] for snapshot in snapshots] == ["loading", "unavailable"]


def test_low_posture_must_be_sustained():
    from posture_analysis import PostureStateTracker

    tracker = PostureStateTracker(sustain_seconds=1, stale_after_seconds=3)
    assert not tracker.update(5, True, 0)
    assert not tracker.update(5, False, 0.5)
    assert not tracker.update(5, True, 1)
    assert tracker.update(5, True, 2)
    tracker.reset()
    assert tracker.tracked_count == 0
