"""Verify that the active interpreter can import IBVAP's runtime dependencies."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import platform
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = (
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("ultralytics", "ultralytics"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
)
FACE_DEPENDENCIES = (
    ("onnxruntime", "onnxruntime"),
    ("insightface", "insightface"),
)
ANPR_DEPENDENCIES = (("easyocr", "easyocr"),)


def _sha256_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main(include_face: bool = False, include_anpr: bool = False) -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Interpreter: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    failures = []
    dependencies = DEPENDENCIES
    if (include_face or include_anpr) and str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if include_face:
        dependencies += FACE_DEPENDENCIES
    if include_anpr:
        dependencies += ANPR_DEPENDENCIES

    for module_name, distribution_name in dependencies:
        try:
            importlib.import_module(module_name)
            installed_version = version(distribution_name)
        except Exception as exc:
            failures.append((module_name, distribution_name, exc))
            print(f"[MISSING] {module_name} ({distribution_name}): {exc}")
        else:
            print(f"[OK] {module_name}: {installed_version}")

    if include_face:
        import config

        for name, model_path in (
            ("SCRFD", config.SCRFD_MODEL_PATH),
            ("GFPGAN", config.GFPGAN_MODEL_PATH),
        ):
            if Path(model_path).is_file():
                print(f"[OK] {name} model: {model_path}")
            else:
                error = FileNotFoundError(model_path)
                failures.append((name, "face model", error))
                print(f"[MISSING] {name} model: {model_path}")

    if include_anpr:
        import config

        model_path = Path(config.ANPR_PLATE_MODEL_PATH)
        if not model_path.is_file():
            failures.append(("ANPR", "plate model", FileNotFoundError(model_path)))
            print(f"[MISSING] ANPR plate model: {model_path}")
        elif not config.ANPR_PLATE_MODEL_SHA256:
            failures.append(("ANPR", "plate model", ValueError("missing configured checksum")))
            print(f"[INVALID] ANPR plate model has no configured SHA-256: {model_path}")
        elif _sha256_file(model_path) != config.ANPR_PLATE_MODEL_SHA256:
            failures.append(("ANPR", "plate model", ValueError("checksum mismatch")))
            print(f"[INVALID] ANPR plate model checksum: {model_path}")
        else:
            print(f"[OK] ANPR plate model: {model_path}")

    if failures:
        print("\nThe active interpreter is incomplete.")
        locks = ["requirements.lock"]
        if include_face:
            locks.append("requirements-face.lock")
        if include_anpr:
            locks.append("requirements-anpr.lock")
        lock_arguments = " ".join(f"-r {lock}" for lock in locks)
        print(f"Reinstall with: {sys.executable} -m pip install {lock_arguments}")
        if include_face:
            print("Install reviewed model files with: python scripts/download_face_models.py")
        if include_anpr:
            print("Review/acquire ANPR weights with: python scripts/download_anpr_model.py")
        print("Always run the app with the same interpreter shown above.")
        return 1

    print("\nRuntime dependency check passed.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face", action="store_true", help="also check SCRFD/GFPGAN support")
    parser.add_argument("--anpr", action="store_true", help="also check regional ANPR support")
    arguments = parser.parse_args()
    raise SystemExit(main(include_face=arguments.face, include_anpr=arguments.anpr))
