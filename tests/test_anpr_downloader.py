import hashlib
import io
import json

import pytest

from scripts import download_anpr_model


def test_model_metadata_matches_the_checksum_pinned_downloader():
    metadata_path = download_anpr_model.ROOT / "models" / "ANPR_MODEL_SOURCES.json"
    model = json.loads(metadata_path.read_text(encoding="utf-8"))["models"][0]
    assert model["download_url"] == download_anpr_model.SOURCE_URL
    assert model["sha256"] == download_anpr_model.EXPECTED_SHA256
    assert model["bytes"] == download_anpr_model.EXPECTED_SIZE
    assert model["approval"] == "research-evaluation-only"


def test_downloader_requires_explicit_risk_acceptance_before_network(monkeypatch, capsys):
    monkeypatch.setattr(download_anpr_model.sys, "argv", ["download_anpr_model.py"])
    monkeypatch.setattr(
        download_anpr_model.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("network must not be used without acceptance"),
    )
    assert download_anpr_model.main() == 2
    assert "No download was attempted" in capsys.readouterr().err


def test_download_accepts_only_the_pinned_size_and_checksum(monkeypatch, tmp_path):
    payload = b"verified model bytes"
    monkeypatch.setattr(download_anpr_model, "EXPECTED_SIZE", len(payload))
    monkeypatch.setattr(download_anpr_model, "EXPECTED_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(
        download_anpr_model.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(payload),
    )
    destination = tmp_path / "model.pt"
    download_anpr_model.download(destination)
    assert destination.read_bytes() == payload
    assert not (tmp_path / "model.pt.part").exists()

    monkeypatch.setattr(
        download_anpr_model.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(b"wrong"),
    )
    rejected = tmp_path / "rejected.pt"
    with pytest.raises(RuntimeError, match="failed"):
        download_anpr_model.download(rejected)
    assert not rejected.exists()
    assert not (tmp_path / "rejected.pt.part").exists()
