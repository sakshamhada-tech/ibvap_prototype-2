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
| Persistent object IDs | Ultralytics ByteTrack integration |
| Virtual fence | Robust line-segment crossing with stale-track expiry |
| Loitering | Continuous observation window using source/media time |
| Night movement | Smoothed central-ROI brightness with hysteresis |
| Face localization | OpenCV Haar cascade, throttled per track |
| ANPR | Optional Haar plate localization + throttled EasyOCR |
| Alerts | UTC/source timestamps, cooldowns, CSV, bounded live broadcast |
| Dashboard | Authenticated FastAPI, MJPEG feed, WebSocket alerts/stats |
| Source resilience | Camera reconnect, health checks, file-loop state reset |

Face localization is **not** face recognition or watchlist matching. ANPR is a
best-effort demo implementation; the bundled Russian-plate Haar cascade is not
optimized for Indian plates.

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
       ├── throttled face detection
       └── optional throttled plate OCR
       │
       ▼
VideoPipeline
       ├── AlertLogger → CSV
       ├── main.py → window and/or MP4
       └── server.py → bounded MJPEG + per-client WebSocket queues
```

Recorded-video analytics use media timestamps, so alerts do not change merely
because one computer processes the file faster than another. Live sources use
a monotonic clock. Track state expires after gaps and resets on file loops or
camera reconnects.

## Requirements

- Python 3.10–3.13; Python 3.11 is the reference version
- Sufficient disk/RAM for PyTorch and Ultralytics
- A webcam, recording, or supported camera URL
- TLS reverse proxy or certificate/key for non-local dashboard use

The repository intentionally retains `yolov8n.pt` for offline model startup.
Its checksum and provenance metadata are in `models/MODEL_MANIFEST.json` and
its licensing notice is in `THIRD_PARTY_NOTICES.md`.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock  # reproducible Python 3.11 environment
```

ANPR is disabled by default. Install and opt in only when needed:

```bash
python -m pip install -r requirements-anpr.lock
export IBVAP_ENABLE_ANPR=true
```

EasyOCR may download separate weights on first use. The base system does not
need those weights.

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
| `IBVAP_LOITERING_SECONDS` | `15` | Continuous dwell duration |
| `IBVAP_LOITERING_MOVEMENT_THRESHOLD_PX` | `60` | Maximum window spread |
| `IBVAP_TRACK_MAX_OBSERVATION_GAP_SECONDS` | `1.5` | Gap that resets loitering history |
| `IBVAP_TRACK_STALE_SECONDS` | `3` | State-retention period for missing tracks |
| `IBVAP_ENABLE_FACE_DETECTION` | `true` | Enable face localization |
| `IBVAP_FACE_DETECTION_INTERVAL_FRAMES` | `5` | Face-processing cadence |
| `IBVAP_ENABLE_ANPR` | `false` | Enable optional plate OCR |
| `IBVAP_ANPR_INTERVAL_FRAMES` | `15` | Plate/OCR processing cadence |

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
development. The credentials above are newly generated each shell session;
store controlled deployment values in a secret manager.

Endpoints:

- `GET /health/live` — capture-thread liveness, no source details
- `GET /health/ready` — whether a recent frame is available
- `GET /stream` — authenticated MJPEG stream
- `GET /api/stats` — authenticated current statistics
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
source scan, and dependency auditing when `requirements.lock` is present.
Critical geometry, loitering continuity, alert cooldown, bounded broadcast,
authentication, origin, rate-limit, and source-time behavior have unit tests.
CI performs the same checks on Python 3.11.

## Operational limitations

- YOLO, Haar cascades, brightness thresholds, and OCR can produce false
  positives and false negatives. Human review remains necessary.
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
