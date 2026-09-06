#!/usr/bin/env python3
"""Download and verify the optional SCRFD and GFPGAN model files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
SOURCES_PATH = MODEL_DIR / "FACE_MODEL_SOURCES.json"
CHUNK_SIZE = 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, expected_hash: str, expected_size: int) -> Path:
    if urlparse(url).scheme != "https":
        raise RuntimeError("model downloads must use HTTPS")
    request = urllib.request.Request(url, headers={"User-Agent": "IBVAP-model-installer/1"})
    temporary = tempfile.NamedTemporaryFile(  # noqa: SIM115 - atomically moved after checks
        prefix="ibvap-model-", delete=False, dir=MODEL_DIR
    )
    temporary_path = Path(temporary.name)
    digest = hashlib.sha256()
    downloaded = 0
    try:
        with temporary, urllib.request.urlopen(request, timeout=60) as response:  # nosec B310
            while chunk := response.read(CHUNK_SIZE):
                downloaded += len(chunk)
                if downloaded > expected_size:
                    raise RuntimeError(f"download from {url} exceeded its expected size")
                digest.update(chunk)
                temporary.write(chunk)
        if downloaded != expected_size:
            raise RuntimeError(
                f"download size mismatch for {url}: expected {expected_size}, got {downloaded}"
            )
        actual_hash = digest.hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"download checksum mismatch for {url}: expected {expected_hash}, got {actual_hash}"
            )
        return temporary_path
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _install_scrfd(source: dict) -> None:
    destination = MODEL_DIR / source["destination"]
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    if destination.is_file() and sidecar.is_file():
        expected = sidecar.read_text(encoding="ascii").strip().split()[0]
        if _sha256(destination) == expected:
            print(f"[OK] {source['name']}: already installed")
            return

    archive = _download(
        source["archive_url"],
        source["archive_sha256"],
        source["archive_size_bytes"],
    )
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with zipfile.ZipFile(archive) as bundle:
            matches = [
                member
                for member in bundle.infolist()
                if Path(member.filename).name == source["archive_member_basename"]
            ]
            if len(matches) != 1:
                raise RuntimeError("verified InsightFace archive does not contain one SCRFD model")
            member = matches[0]
            if member.file_size > 64 * 1024 * 1024:
                raise RuntimeError("SCRFD archive member is unexpectedly large")
            with bundle.open(member) as model_source, partial.open("wb") as model_destination:
                shutil.copyfileobj(model_source, model_destination, length=CHUNK_SIZE)
        model_hash = _sha256(partial)
        os.replace(partial, destination)
        sidecar.write_text(f"{model_hash}  {destination.name}\n", encoding="ascii")
    finally:
        archive.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    print(f"[OK] {source['name']}: {destination}")


def _install_gfpgan(source: dict) -> None:
    destination = MODEL_DIR / source["destination"]
    if destination.is_file() and _sha256(destination) == source["sha256"]:
        print(f"[OK] {source['name']}: already installed")
        return
    downloaded = _download(source["url"], source["sha256"], source["size_bytes"])
    os.replace(downloaded, destination)
    print(f"[OK] {source['name']}: {destination}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accept-model-licenses",
        action="store_true",
        help="confirm that you reviewed and accept the upstream model-license terms",
    )
    args = parser.parse_args(argv)
    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    if not args.accept_model_licenses:
        print("Model files were not downloaded. Review these terms first:\n")
        for source in sources.values():
            print(f"- {source['name']}: {source['license_note']}")
        print("\nIf those terms fit your use, rerun with --accept-model-licenses.")
        return 2

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _install_scrfd(sources["scrfd"])
    _install_gfpgan(sources["gfpgan"])
    print("Face models installed. They are ignored by Git and will not be committed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
