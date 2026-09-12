import hashlib
import zipfile

import pytest

from scripts import download_face_models


def test_downloader_requires_explicit_model_license_acceptance(capsys):
    assert download_face_models.main([]) == 2
    output = capsys.readouterr().out
    assert "non-commercial research use" in output
    assert "--accept-model-licenses" in output


def test_downloader_rejects_non_https_source():
    with pytest.raises(RuntimeError, match="HTTPS"):
        download_face_models._download("http://example.test/model", "0" * 64, 1)


def test_scrfd_archive_extracts_only_expected_verified_member(monkeypatch, tmp_path):
    archive = tmp_path / "models.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("bundle/det_2.5g.onnx", b"scrfd model")
        bundle.writestr("bundle/unrelated.onnx", b"do not extract")

    monkeypatch.setattr(download_face_models, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(download_face_models, "_download", lambda *_args: archive)
    source = {
        "name": "SCRFD",
        "archive_url": "https://example.test/models.zip",
        "archive_sha256": "unused",
        "archive_size_bytes": 1,
        "archive_member_basename": "det_2.5g.onnx",
        "destination": "scrfd.onnx",
    }
    download_face_models._install_scrfd(source)

    model = tmp_path / "scrfd.onnx"
    assert model.read_bytes() == b"scrfd model"
    digest = hashlib.sha256(b"scrfd model").hexdigest()
    assert (tmp_path / "scrfd.onnx.sha256").read_text(encoding="ascii").startswith(digest)
    assert not (tmp_path / "unrelated.onnx").exists()
