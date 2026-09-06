import hashlib
import threading
from pathlib import Path

import face_enhancement
from face_enhancement import FaceEnhancementService


class CopyableFrame:
    def copy(self):
        return self


def make_service(tmp_path, callback):
    return FaceEnhancementService(
        scrfd_model_path=str(tmp_path / "scrfd.onnx"),
        gfpgan_model_path=str(tmp_path / "gfpgan.pth"),
        gfpgan_model_sha256="0" * 64,
        confidence_threshold=0.6,
        input_size=640,
        min_face_size_px=24,
        max_faces=4,
        refresh_seconds=2,
        cache_ttl_seconds=10,
        jpeg_quality=90,
        device="cpu",
        on_snapshot=callback,
    )


def test_missing_face_models_fail_without_stopping_caller(tmp_path):
    snapshots = []
    unavailable = threading.Event()

    def callback(snapshot):
        snapshots.append(snapshot)
        if snapshot["status"] == "unavailable":
            unavailable.set()

    service = make_service(tmp_path, callback)
    service.start()
    assert unavailable.wait(2)
    service.close()
    assert [snapshot["status"] for snapshot in snapshots] == ["loading", "unavailable"]
    assert not service.submit(CopyableFrame(), [(1, (0, 0, 10, 10))], 1)


def test_worker_publishes_result_and_resets_queued_state(monkeypatch, tmp_path):
    snapshots = []
    result_ready = threading.Event()

    def callback(snapshot):
        snapshots.append(snapshot)
        if snapshot["status"] == "ready" and snapshot["faces"]:
            result_ready.set()

    service = make_service(tmp_path, callback)
    monkeypatch.setattr(service, "_load_models", lambda: None)
    monkeypatch.setattr(
        service,
        "_process",
        lambda job: [{"track_id": job.people[0][0], "confidence": 0.9, "jpeg": b"jpeg"}],
    )
    service.start()
    assert service._accepting_jobs.wait(2)
    assert service.submit(CopyableFrame(), [(7, (0, 0, 10, 10))], 12)
    assert result_ready.wait(2)
    assert snapshots[-1]["faces"][0]["track_id"] == 7

    service.reset()
    assert snapshots[-1]["faces"] == []
    service.close()


def test_face_to_person_assignment_prefers_smallest_containing_box():
    people = [(1, (0, 0, 100, 100)), (2, (20, 20, 60, 60)), (3, (200, 200, 300, 300))]
    assert face_enhancement._containing_track((30, 30, 50, 50), people) == 2
    assert face_enhancement._containing_track((120, 120, 140, 140), people) is None


def test_model_checksum_sidecar_is_enforced(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"trusted model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    Path(f"{model}.sha256").write_text(f"{digest}  {model.name}\n", encoding="ascii")
    face_enhancement._verify_sidecar_hash(model)

    model.write_bytes(b"tampered")
    try:
        face_enhancement._verify_sidecar_hash(model)
    except face_enhancement.FaceModelsUnavailable as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("tampered model was accepted")
