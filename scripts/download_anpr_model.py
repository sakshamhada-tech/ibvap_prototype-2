"""Explicitly acquire the checksum-pinned research ANPR detector candidate."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = ROOT / "models" / "indian_plate_yolov8n.pt"
SOURCE_URL = (
    "https://raw.githubusercontent.com/vipulcj/"
    "AUTOMATIC-LICENSE-PLATE-RECOGNITION-FOR-INDIAN-VEHICLES-/"
    "1ec0e751a3d3ce1e4b81f3d809e7c97089c7b834/INPD_more_accuracy_n.pt"
)
EXPECTED_SHA256 = "897ad71e4c94cb69aabdb6ba8fbd4bbecfad239fde6bc386ae04c650411787b4"
EXPECTED_SIZE = 6_230_819
RISK_NOTICE = """
This optional checkpoint is a research/evaluation candidate, not a production-
approved model. Its training-data provenance, held-out South-Asian benchmark,
and separate checkpoint licence are undocumented. Loading it through
Ultralytics also requires review of Ultralytics' AGPL/commercial terms. You are
responsible for confirming that your intended use and data handling are lawful.
See models/ANPR_MODEL_SOURCES.json and THIRD_PARTY_NOTICES.md.
""".strip()


def sha256_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "IBVAP-model-fetch/1"})
    checksum = hashlib.sha256()
    downloaded = 0
    try:
        # The URL is a source-controlled HTTPS constant, never operator input.
        with (
            urllib.request.urlopen(request, timeout=60) as response,  # nosec B310
            temporary.open("xb") as target,
        ):
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
                checksum.update(chunk)
                downloaded += len(chunk)
                if downloaded > EXPECTED_SIZE:
                    raise RuntimeError("download exceeded the pinned checkpoint size")
        if downloaded != EXPECTED_SIZE or checksum.hexdigest() != EXPECTED_SHA256:
            raise RuntimeError(
                "downloaded checkpoint failed its pinned size/SHA-256 verification; refusing it"
            )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, epilog=RISK_NOTICE)
    parser.add_argument("--output", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--accept-research-model-risks",
        action="store_true",
        help="confirm that you reviewed and accept the printed licensing/data/model risks",
    )
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()

    print(RISK_NOTICE, file=sys.stderr)
    if not args.accept_research_model_risks:
        print(
            "\nNo download was attempted. Re-run with --accept-research-model-risks "
            "only after review.",
            file=sys.stderr,
        )
        return 2

    if destination.is_file() and sha256_file(destination) == EXPECTED_SHA256:
        print(f"Verified existing checkpoint: {destination}")
        return 0
    if destination.exists():
        print(f"Refusing to overwrite unverified path: {destination}", file=sys.stderr)
        return 1

    print(f"Downloading the pinned candidate to {destination}")
    download(destination)
    print(f"Verified SHA-256 {EXPECTED_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
