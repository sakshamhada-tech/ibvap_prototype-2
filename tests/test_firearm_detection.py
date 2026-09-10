import threading
from types import SimpleNamespace

from firearm_detection import FirearmDetectionService, parse_firearm_result


class Values:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class CopyableFrame:
    def copy(self):
        return self


def test_firearm_result_enforces_conservative_threshold():
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=Values(((1, 2, 10, 20), (30, 40, 50, 60))),
            conf=Values((0.74, 0.91)),
        )
    )
    assert parse_firearm_result(result, 0.8) == [{"bbox": (30, 40, 50, 60), "confidence": 0.91}]


def test_missing_firearm_model_is_failure_isolated(tmp_path):
    snapshots = []
    unavailable = threading.Event()

    def callback(snapshot):
        snapshots.append(snapshot)
        if snapshot["status"] == "unavailable":
            unavailable.set()

    service = FirearmDetectionService(
        model_path=str(tmp_path / "firearm.pt"),
        model_sha256="0" * 64,
        confidence_threshold=0.8,
        device="cpu",
        on_snapshot=callback,
    )
    service.start()
    assert unavailable.wait(2)
    assert not service.submit(CopyableFrame(), 1)
    service.close()
    assert [snapshot["status"] for snapshot in snapshots] == ["loading", "unavailable"]
