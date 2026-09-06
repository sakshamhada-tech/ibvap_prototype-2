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

## Optional SCRFD and GFPGAN models

The optional enhanced-face panel uses SCRFD code through InsightFace and the
GFPGAN v1.4 restoration model. The minimal GFPGAN clean inference architecture
under `third_party/gfpgan_arch/` is derived from TencentARC/GFPGAN 1.3.8, and
its Apache-2.0 license is retained there. Its small initialization helper is
derived from XPixelGroup/BasicSR 1.4.2 under Apache-2.0; BasicSR itself is not a
runtime dependency. The large weights are not included in this repository.
Their reviewed download locations and checksums are recorded in
`models/FACE_MODEL_SOURCES.json`.

- SCRFD/InsightFace code is MIT/Apache-2.0, but InsightFace states that its
  provided pretrained models are limited to **non-commercial research use**.
  Obtain separately licensed SCRFD-compatible weights for other uses.
- GFPGAN is published under Apache-2.0. Preserve its upstream attribution and
  license when redistributing the model or software.
- GFPGAN output is generative. It can invent plausible details and must not be
  represented as an authentic or forensically reliable reconstruction.

## Runtime dependencies

The principal dependencies are Ultralytics, OpenCV, NumPy, FastAPI, Uvicorn,
and optionally EasyOCR, InsightFace, and ONNX Runtime. Their full transitive
dependency trees and license texts are installed by the package manager and
remain governed by their own terms. Run a software-composition/license scan as
part of each release because transitive dependencies can change.
