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

## Optional Indian plate-detector candidate

The regional ANPR path can load an external Indian YOLOv8 detector checkpoint.
No ANPR checkpoint is distributed in this repository. The reviewed candidate,
its pinned source commit, byte size, SHA-256, and unresolved issues are recorded
in `models/ANPR_MODEL_SOURCES.json`. Its downloader requires explicit
`--accept-research-model-risks` acknowledgement before making a network request.

The source repository includes Apache-2.0 text, but no separate checkpoint
licence/model card, training-data provenance, credible held-out deployment
benchmark, or proprietary-use grant was found. The checkpoint also runs through
Ultralytics, whose AGPL-3.0/commercial terms require separate review. Treat the
candidate as **research/evaluation only** until legal, data-governance, accuracy,
and security review are complete. EasyOCR code and its separately downloaded OCR
weights remain governed by their own upstream terms. This notice is not legal
advice.

## Optional SCRFD and GFPGAN models

The optional identity-first face-review panel uses SCRFD code through InsightFace.
GFPGAN v1.4 is available only as an explicitly enabled experimental preview.
The minimal GFPGAN clean inference architecture
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

## Operator-supplied pose and firearm checkpoints

The optional posture and firearm workers do not include or download model
weights. Any operator-supplied Ultralytics-compatible checkpoint retains its
own code, dataset, and weight terms. Before use, document its source and
SHA-256, verify redistribution and deployment rights, assess training-data
provenance and bias, and validate false-positive/false-negative behavior on
representative labelled footage. Firearm results are human-review flags only.
See `models/CONTEXT_MODEL_REQUIREMENTS.json`.

## Runtime dependencies

The principal dependencies are Ultralytics, OpenCV, NumPy, FastAPI, Uvicorn,
and optionally EasyOCR, InsightFace, and ONNX Runtime. Their full transitive
dependency trees and license texts are installed by the package manager and
remain governed by their own terms. Run a software-composition/license scan as
part of each release because transitive dependencies can change.
