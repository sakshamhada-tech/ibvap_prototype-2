# Third-party notices

IBVAP depends on third-party software and includes a pretrained model. The
repository's proprietary license does **not** replace their licenses.

## Bundled YOLOv8 Nano weights

- File: `yolov8n.pt`
- SHA-256: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`
- Upstream project: [Ultralytics](https://github.com/ultralytics/ultralytics)
- Model metadata identifies Ultralytics and the AGPL-3.0 license.

Review Ultralytics' current software and model licensing terms before
redistributing this repository or using it in a network service or commercial
product. Obtain an appropriate commercial license where required. This notice
is not legal advice.

## Runtime dependencies

The principal dependencies are Ultralytics, OpenCV, NumPy, FastAPI, Uvicorn,
and optionally EasyOCR. Their full transitive dependency trees and license
texts are installed by the package manager and remain governed by their own
terms. Run a software-composition/license scan as part of each release because
transitive dependencies can change.
