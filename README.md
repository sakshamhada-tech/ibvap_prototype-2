# IBVAP Prototype

Single-camera intelligent video analytics for existing CCTV, video files,
webcams, and RTSP/HTTP camera streams.

> **Scope:** This is an engineering prototype, not a certified border-security
> system. Validate it with representative labelled footage and complete a
> security, privacy, licensing, and operational review before real deployment.

## Capabilities

| Capability | Implementation |
|---|---|
| Person and vehicle detection | YOLOv8 COCO classes |
| Short-lived object track IDs | Ultralytics ByteTrack integration |
| Virtual fence | Visible-box contact and robust crossing with stale-track expiry |
| Loitering | Continuous observation window using source/media time |
| Night movement | Smoothed central-ROI brightness with hysteresis |
| Face review panel | Aligned source beside optional conservative SCRFD/GFPGAN blend |
| ANPR | Optional Haar plate localization + throttled EasyOCR |
| Alerts | UTC/source timestamps, cooldowns, CSV, bounded live broadcast |
| Dashboard | Authenticated FastAPI, MJPEG feed, enhanced faces, WebSocket alerts/stats |
| Source resilience | Camera reconnect, health checks, file-loop state reset |

SCRFD face localization and GFPGAN restoration are **not** face recognition or
watchlist matching. Restored faces are generative visualizations, can contain
invented details, and are not forensic evidence. ANPR is a best-effort demo;
the bundled Russian-plate Haar cascade is not optimized for Indian plates.

## Architecture

```text
Camera / recording
       │
       ▼
OpenCV capture ── media/monotonic timestamp
       │
       ▼
YOLOv8 + ByteTrack
       │
       ├── virtual-fence state
       ├── continuous loitering window
       ├── night classification
       ├── optional throttled plate OCR
       └── bounded raw-frame handoff → SCRFD → GFPGAN worker
                                           │
                                           └── enhanced JPEG thumbnails
       │
       ▼
VideoPipeline
       ├── AlertLogger → CSV
       ├── main.py → window and/or MP4
       └── server.py → MJPEG + MP4 + face-review panel + per-client WebSocket queues
```

Recorded-video analytics use media timestamps, so alerts do not change merely
because one computer processes the file faster than another. Live sources use
a monotonic clock. Track state expires after gaps and resets on file loops or
camera reconnects.

## Requirements

- Python 3.11–3.13; Python 3.11 is the reference version
- Linux, Windows, or Apple Silicon with macOS 14 or newer
- Sufficient disk/RAM for PyTorch and Ultralytics
- A webcam, recording, or supported camera URL
- TLS reverse proxy or certificate/key for non-local dashboard use

Current security-patched PyTorch releases no longer publish Intel macOS wheels,
so Intel Macs are not supported by the locked environment. Do not work around
this by silently downgrading PyTorch to an older vulnerable release.

The repository intentionally retains `yolov8n.pt` for offline model startup.
Its checksum and provenance metadata are in `models/MODEL_MANIFEST.json` and
its licensing notice is in `THIRD_PARTY_NOTICES.md`.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python scripts/doctor.py
```

The lock is generated as a universal requirements file: Linux-only CUDA
packages are guarded by platform markers and are skipped on macOS and Windows.
Use the `-r` flag—`pip install requirements.lock` does not install a
requirements file. Always run the app with the same `python` interpreter used
for installation.

ANPR is disabled by default. Install and opt in only when needed:

```bash
python -m pip install -r requirements-anpr.lock
export IBVAP_ENABLE_ANPR=true
```

EasyOCR may download separate weights on first use. The base system does not
need those weights.

### Optional SCRFD + GFPGAN enhanced-face panel

The enhancement stack is separate so it cannot destabilize a base installation.
Install it only after the normal runtime passes `doctor.py`:

```bash
python -m pip install -r requirements-face.lock
python scripts/download_face_models.py
```

The downloader first prints the upstream model terms and exits without changing
anything. InsightFace-provided pretrained models are restricted to
non-commercial research use. After reviewing the terms, either provide your own
licensed SCRFD-compatible model with `IBVAP_SCRFD_MODEL_PATH`, or explicitly run:

```bash
python scripts/download_face_models.py --accept-model-licenses
python scripts/doctor.py --face
export IBVAP_ENABLE_FACE_ENHANCEMENT=true
```

The verified model files are placed under `models/` and ignored by Git. SCRFD
and GFPGAN run on a bounded background worker; capture, object detection,
tracking, alerts, and the raw annotated stream continue if that worker is slow
or fails. GFPGAN never replaces pixels in the primary stream or recording. Its
results appear only in the authenticated **Face source / AI review** panel.
The panel always keeps the aligned source beside any result. GFPGAN is skipped
when the detected source face is below 80 pixels or fails the sharpness gate;
that entry is clearly marked source-only/rejected. For accepted inputs, the
default review image is 75% aligned source and 25% GFPGAN output rather than a
fully synthetic restoration. These gates reduce identity drift; they cannot
create authentic details that the camera never captured. Tiny, blurred,
compressed, occluded, or poorly exposed faces remain unsuitable for identity
assessment.

### macOS: `No module named 'cv2'`

The import is named `cv2`, but its package is named `opencv-python`. Recreate
the environment if a previous platform-specific lock install stopped partway:

```bash
deactivate 2>/dev/null || true
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python scripts/doctor.py
python -c "import cv2; print(cv2.__version__)"
```

Confirm that `which python` points to `.venv/bin/python`. If installation
succeeds but another terminal still cannot import `cv2`, that terminal is
using a different interpreter or does not have the virtual environment active.

## Configuration

Configuration is read from `IBVAP_*` environment variables. The application
does not automatically load `.env`; use your shell, service manager, secret
manager, or Docker Compose. See `.env.example` for a starting point.

Common analytics settings:

| Variable | Default | Meaning |
|---|---:|---|
| `IBVAP_VIDEO_SOURCE` | `0` | Camera index, file path, RTSP(S), or HTTP(S) URL |
| `IBVAP_CONFIDENCE_THRESHOLD` | `0.4` | YOLO confidence cutoff |
| `IBVAP_VIRTUAL_FENCE` | `50,400,1200,400` | Fence endpoints in source pixels |
| `IBVAP_VIRTUAL_FENCE_CONTACT_MARGIN_PX` | `2` | Contact allowance around the visible fence |
| `IBVAP_LOITERING_SECONDS` | `15` | Continuous dwell duration |
| `IBVAP_LOITERING_MOVEMENT_THRESHOLD_PX` | `60` | Maximum window spread |
| `IBVAP_TRACK_MAX_OBSERVATION_GAP_SECONDS` | `1.5` | Gap that resets loitering history |
| `IBVAP_TRACK_STALE_SECONDS` | `3` | State-retention period for missing tracks |
| `IBVAP_ENABLE_FACE_ENHANCEMENT` | `false` | Enable optional SCRFD + GFPGAN panel |
| `IBVAP_FACE_ENHANCEMENT_INTERVAL_FRAMES` | `15` | Raw-frame submission cadence |
| `IBVAP_FACE_ENHANCEMENT_MAX_FACES` | `4` | Maximum enhanced faces per update |
| `IBVAP_FACE_ENHANCEMENT_DEVICE` | `auto` | GFPGAN device: auto, CPU, MPS, or CUDA |
| `IBVAP_SCRFD_CONFIDENCE_THRESHOLD` | `0.6` | Minimum SCRFD detection confidence |
| `IBVAP_SCRFD_MIN_FACE_SIZE_PX` | `32` | Smallest face retained as an aligned source preview |
| `IBVAP_GFPGAN_MIN_SOURCE_FACE_SIZE_PX` | `80` | Minimum source-face dimension allowed into GFPGAN |
| `IBVAP_GFPGAN_MIN_SOURCE_SHARPNESS` | `30` | Minimum source-crop variance-of-Laplacian sharpness |
| `IBVAP_GFPGAN_BLEND_WEIGHT` | `0.25` | GFPGAN share of review image; constrained to 0–0.5 |
| `IBVAP_ENABLE_ANPR` | `false` | Enable optional plate OCR |
| `IBVAP_ANPR_INTERVAL_FRAMES` | `15` | Plate/OCR processing cadence |

A virtual-fence intrusion starts when a tracked object's visible bounding box
first contacts the fence (including the small configurable margin). Centroid
motion is also checked so fast movement that skips over the line between frames
is not missed.

Relative file and output paths resolve from the repository directory, not the
process's current directory. Camera URLs are never stored in source code; pass
them through a protected environment or secret manager.

## Process a recording or camera locally

```bash
export IBVAP_VIDEO_SOURCE=sample_data/demo.mp4
export IBVAP_SHOW_LIVE_WINDOW=true
python main.py
```

For a headless system:

```bash
export IBVAP_SHOW_LIVE_WINDOW=false
python main.py
```

By default, generated files are written to `output/annotated_output.mp4` and
`logs/alerts.csv`. Both directories are ignored by Git. Press `q` to stop the
window, or use Ctrl+C.

## Run the authenticated dashboard

The server fails closed unless a password of at least 12 characters and a
session secret of at least 32 characters are supplied.

```bash
export IBVAP_DASHBOARD_USERNAME=operator
export IBVAP_DASHBOARD_PASSWORD="$(python -c 'import secrets; print(secrets.token_urlsafe(24))')"
export IBVAP_SESSION_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export IBVAP_VIDEO_SOURCE=0
python server.py
```

Open `http://localhost:8000/login`. Plain HTTP is suitable only for loopback
development. `localhost`, `127.0.0.1`, and `::1` are treated as equivalent
local origins when the port matches; other origins remain restricted. The
credentials above are newly generated each shell session; store controlled
deployment values in a secret manager.

`server.py` honors `IBVAP_SAVE_ANNOTATED_VIDEO` and writes annotated frames to
`IBVAP_ANNOTATED_VIDEO_PATH` (`output/annotated_output.mp4` by
default), independently of connected browser clients. Stop the server
gracefully (Ctrl+C or SIGTERM) so OpenCV releases the writer and finalizes the
MP4 container metadata; a running or forcibly killed process can leave the file
unplayable or apparently empty. Recording-open/write failures are reported as
`recording_status=failed` by `/api/stats` but do not stop capture, analytics, or
streaming.

Alert CSV output is separate and defaults to `logs/alerts.csv`. To keep it next
to the MP4 instead, set:

```bash
export IBVAP_ALERT_LOG_CSV=output/alerts.csv
```

Endpoints:

- `GET /health/live` — capture-thread liveness, no source details
- `GET /health/ready` — whether a recent frame is available
- `GET /stream` — authenticated MJPEG stream
- `GET /api/stats` — authenticated current statistics
- `GET /api/faces` — authenticated source/review status, quality, and image references
- `GET /api/faces/{index}?variant=source|review` — authenticated face JPEG variant
- `WS /ws` — authenticated, same-origin alerts and statistics

Each browser receives its own bounded alert queue plus bounded recent history.
Stream and WebSocket connection counts are capped. Login attempts are limited
per client address.

### HTTPS

For direct TLS:

```bash
export IBVAP_TLS_CERTFILE=/secure/path/tls.crt
export IBVAP_TLS_KEYFILE=/secure/path/tls.key
export IBVAP_REQUIRE_HTTPS=true
export IBVAP_COOKIE_SECURE=true
python server.py
```

For a reverse proxy, terminate TLS there, set `IBVAP_HTTPS_BEHIND_PROXY=true`,
and set `IBVAP_FORWARDED_ALLOW_IPS` to the proxy's exact address. Configure
`IBVAP_ALLOWED_HOSTS` with public hostnames and `IBVAP_ALLOWED_ORIGINS` with
the exact public HTTPS origins. The default host allowlist contains only
`localhost` and `127.0.0.1`. Do not trust all forwarded addresses on an exposed
host. See `SECURITY.md` for the deployment checklist.

Run exactly one application worker: capture and in-memory security state are
process-local. Horizontal scaling requires an external session store, rate
limiter, pub/sub service, and camera-work coordinator.

## Docker

```bash
export IBVAP_DASHBOARD_PASSWORD="$(python -c 'import secrets; print(secrets.token_urlsafe(24))')"
export IBVAP_SESSION_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
docker compose up --build
```

The default Compose port is bound to loopback. Configure camera device access
or an RTSP URL explicitly for the deployment host.

## Testing and quality checks

Install lightweight developer tools without the ML runtime:

```bash
python -m pip install -r requirements-dev.lock
make check
```

`make check` runs formatting checks, linting, unit tests with coverage, a Bandit
source scan, and audits all dependency locks. `make lock` regenerates universal
platform-aware locks with `uv`. Critical geometry, loitering continuity, alert
cooldown, bounded broadcast, authentication, origin, rate-limit, and source-time
behavior have unit tests. CI also verifies that the pinned OpenCV package
provides the `cv2` import on macOS.

## Operational limitations

- YOLO, SCRFD, plate Haar cascades, brightness thresholds, and OCR can produce
  false positives and false negatives. Human review remains necessary.
- GFPGAN synthesizes plausible detail rather than recovering ground truth. No
  enhancement model can reconstruct identity-accurate detail absent from a
  tiny, blurred, compressed, occluded, or poorly exposed source. Keep raw
  source footage and never use enhanced thumbnails as identity evidence.
- Smoothed central-region brightness is still a coarse day/night classifier
  and should be calibrated per camera.
- A pixel-coordinate fence must be calibrated whenever resolution or camera
  geometry changes.
- CSV/JSONL files are local append-only logs, not a tamper-evident evidence
  store. Production deployments need protected centralized storage, retention,
  backup, and integrity controls.
- The bundled account model provides one operator role. Organizational use
  should delegate authentication and role-based authorization to approved SSO.
- Neither liveness detection nor face recognition is implemented.

## Repository policy and licensing

Generated footage, logs, caches, and local secrets are excluded by `.gitignore`.
Do not commit operational imagery, credentials, faces, number plates, or alert
records.

The original project code is proprietary and all rights are reserved; see
`LICENSE`. Third-party dependencies and the bundled model retain their own
terms; see `THIRD_PARTY_NOTICES.md`. Review those terms before redistribution,
network deployment, or commercial use.
