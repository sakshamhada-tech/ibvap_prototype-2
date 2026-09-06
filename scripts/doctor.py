"""Verify that the active interpreter can import IBVAP's runtime dependencies."""

from __future__ import annotations

import argparse
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


def main(include_face: bool = False) -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Interpreter: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    failures = []
    dependencies = DEPENDENCIES
    if include_face:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        dependencies += FACE_DEPENDENCIES

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

    if failures:
        print("\nThe active interpreter is incomplete.")
        lock = "requirements-face.lock" if include_face else "requirements.lock"
        print(f"Reinstall with: {sys.executable} -m pip install -r {lock}")
        if include_face:
            print("Install reviewed model files with: python scripts/download_face_models.py")
        print("Always run the app with the same interpreter shown above.")
        return 1

    print("\nRuntime dependency check passed.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face", action="store_true", help="also check SCRFD/GFPGAN support")
    raise SystemExit(main(include_face=parser.parse_args().face))
