import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace

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
        enable_restoration=False,
        confidence_threshold=0.6,
        input_size=640,
        execution_provider="cpu",
        min_face_size_px=32,
        restoration_min_face_size_px=80,
        restoration_min_sharpness=30,
        restoration_blend_weight=0.25,
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


def test_scrfd_provider_selection_keeps_coreml_explicit():
    mac_providers = {"CoreMLExecutionProvider", "CPUExecutionProvider"}
    assert face_enhancement._select_scrfd_providers(mac_providers, "auto") == [
        "CPUExecutionProvider"
    ]
    assert face_enhancement._select_scrfd_providers(mac_providers, "coreml") == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    cuda_providers = {"CUDAExecutionProvider", "CPUExecutionProvider"}
    assert face_enhancement._select_scrfd_providers(cuda_providers, "auto") == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    try:
        face_enhancement._select_scrfd_providers(mac_providers, "cuda")
    except face_enhancement.FaceModelsUnavailable as exc:
        assert "unavailable" in str(exc)
    else:
        raise AssertionError("unavailable SCRFD provider was accepted")


def test_gfpgan_auto_device_does_not_implicitly_use_mps():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        device=lambda value: value,
    )

    assert face_enhancement._select_torch_device(fake_torch, "auto") == "cpu"
    assert face_enhancement._select_torch_device(fake_torch, "mps") == "mps"


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


def _prepare_process_service(monkeypatch, tmp_path, detection, *, sharpness):
    service = make_service(tmp_path, lambda _snapshot: None)
    service.enable_restoration = True
    service._scrfd = SimpleNamespace(detect=lambda *_args, **_kwargs: ([detection], [[(0, 0)] * 5]))
    source_crop = object()
    aligned = object()
    monkeypatch.setattr(face_enhancement, "_extract_source_face", lambda *_args: source_crop)
    monkeypatch.setattr(face_enhancement, "_align_face", lambda *_args: aligned)
    monkeypatch.setattr(face_enhancement, "_source_sharpness", lambda *_args: sharpness)
    service._cv2 = SimpleNamespace(
        IMWRITE_JPEG_QUALITY=95,
        imencode=lambda *_args: (True, SimpleNamespace(tobytes=lambda: b"jpeg")),
        addWeighted=lambda source, source_weight, restored, restored_weight, gamma: (
            source,
            source_weight,
            restored,
            restored_weight,
            gamma,
        ),
    )
    service._np = object()
    return service


def test_tiny_source_face_is_source_only_and_never_restored(monkeypatch, tmp_path):
    service = _prepare_process_service(
        monkeypatch,
        tmp_path,
        [10, 10, 70, 70, 0.9],
        sharpness=100,
    )
    monkeypatch.setattr(
        service,
        "_restore",
        lambda _aligned: (_ for _ in ()).throw(AssertionError("must not restore")),
    )
    job = SimpleNamespace(frame=object(), people=((7, (0, 0, 100, 100)),), generation=0)

    faces = service._process(job)

    assert faces[0]["source_jpeg"] == b"jpeg"
    assert faces[0]["aligned_jpeg"] == b"jpeg"
    assert faces[0]["jpeg"] is None
    assert faces[0]["restoration_status"] == "rejected"
    assert faces[0]["quality_reason"] == "source_face_too_small"


def test_identity_first_mode_extracts_source_without_running_gfpgan(monkeypatch, tmp_path):
    service = _prepare_process_service(
        monkeypatch,
        tmp_path,
        [10, 10, 110, 110, 0.95],
        sharpness=100,
    )
    service.enable_restoration = False
    monkeypatch.setattr(
        service,
        "_restore",
        lambda _aligned: (_ for _ in ()).throw(AssertionError("must not restore")),
    )
    job = SimpleNamespace(frame=object(), people=((8, (0, 0, 150, 150)),), generation=0)

    faces = service._process(job)

    assert faces[0]["source_jpeg"] == b"jpeg"
    assert faces[0]["aligned_jpeg"] == b"jpeg"
    assert faces[0]["jpeg"] is None
    assert faces[0]["restoration_status"] == "disabled"
    assert faces[0]["quality_reason"] is None


def test_restoration_quality_gate_rejects_blurry_sources():
    assert (
        face_enhancement._restoration_rejection_reason(100, 10, 80, 30) == "source_face_too_blurry"
    )
    assert face_enhancement._restoration_rejection_reason(100, 50, 80, 30) is None


def test_adequate_source_face_uses_conservative_identity_blend(monkeypatch, tmp_path):
    service = _prepare_process_service(
        monkeypatch,
        tmp_path,
        [10, 10, 110, 110, 0.95],
        sharpness=100,
    )
    restored = object()
    monkeypatch.setattr(service, "_restore", lambda _aligned: restored)
    job = SimpleNamespace(frame=object(), people=((8, (0, 0, 150, 150)),), generation=0)

    faces = service._process(job)

    assert faces[0]["source_jpeg"] == b"jpeg"
    assert faces[0]["aligned_jpeg"] == b"jpeg"
    assert faces[0]["jpeg"] == b"jpeg"
    assert faces[0]["restoration_status"] == "blended"
    assert faces[0]["quality_reason"] is None
    assert faces[0]["blend_weight"] == 0.25
