"""Verify that the active interpreter can import IBVAP's runtime dependencies."""

from __future__ import annotations

import importlib
import platform
import sys
from importlib.metadata import version

DEPENDENCIES = (
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("ultralytics", "ultralytics"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
)


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Interpreter: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    failures = []

    for module_name, distribution_name in DEPENDENCIES:
        try:
            importlib.import_module(module_name)
            installed_version = version(distribution_name)
        except Exception as exc:
            failures.append((module_name, distribution_name, exc))
            print(f"[MISSING] {module_name} ({distribution_name}): {exc}")
        else:
            print(f"[OK] {module_name}: {installed_version}")

    if failures:
        print("\nThe active interpreter is incomplete.")
        print(f"Reinstall with: {sys.executable} -m pip install -r requirements.lock")
        print("Always run the app with the same interpreter shown above.")
        return 1

    print("\nRuntime dependency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
