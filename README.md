# IBVAP Prototype

IBVAP is a single-camera intelligent video analytics prototype for existing
CCTV feeds, webcams, local recordings, and RTSP/HTTP streams. It combines
object detection and short-lived tracking with virtual-fence, loitering,
low-light movement, optional face-review, and optional ANPR workflows. It can
run as a local OpenCV process or as an authenticated FastAPI dashboard.

> **Scope and safety:** This is an engineering prototype, not a certified
> border-security product, biometric identification system, or forensic tool.
> Validate it on representative labelled footage and complete security,
> privacy, legal, licensing, model-bias, and operational reviews before any
> real deployment. Human review remains mandatory.

## Implementation status at a glance

| Area | Current implementation | Default |
|---|---|---:|
| Video sources | Camera indexes, video files, RTSP/RTSPS, HTTP/HTTPS | Enabled |
| Object detection | YOLOv8 COCO person, car, motorcycle, bus, and truck classes | Enabled |
| Object tracking | Ultralytics tracking with short-lived ByteTrack IDs | Enabled |
| Virtual fence | Visible-box contact plus centroid segment crossing | Enabled |
| Loitering | Continuous person observation with movement and gap limits | Enabled |
| Night movement | Smoothed central-ROI brightness with hysteresis | Enabled |
| Alert persistence | Rotating CSV with local-system, UTC, and source/media timestamps | Enabled |
| Server audible alarms | Distinct bounded tones for intrusion and loitering | Enabled |
| Security audit log | Rotating structured JSONL for dashboard security events | Server mode |
| Dashboard | FastAPI, authenticated MJPEG, statistics, and WebSocket alerts | Server mode |
| Annotated MP4 | `mp4v` recording from both `main.py` and `server.py` | Enabled |
| Face source review | SCRFD detection, original crop, and landmark-aligned source | Optional |
| GFPGAN preview | Deterministic, quality-gated, low-strength generative preview | Disabled |
| ANPR | Vehicle crop, Haar plate localization, and EasyOCR | Disabled |
| Persistent cross-camera identity | Re-identification/watchlist matching | Not implemented |

## Capabilities in detail

### Person and vehicle detection

- Loads the configured Ultralytics YOLO model and verifies the bundled
  `yolov8n.pt` checksum before use.
- Processes only these COCO classes:
  - person
  - car
  - motorcycle
  - bus
  - truck
- Returns bounding boxes, centroids, class labels, confidence scores, and track
  IDs where the tracker has assigned one.
- Draws class, confidence, and short-lived track labels on the annotated frame.
- Keeps temporarily untracked detections visible, but track-dependent analytics
  require a valid track ID.

### Short-lived object tracking

- Uses the tracker integrated with Ultralytics `model.track(..., persist=True)`.
- Tracking state is reset after source discontinuities, file loops, camera
  reconnects, and backwards media-time movement.
- Stale virtual-fence and loitering state expires after a configurable timeout.
- Track IDs are local to one process and one uninterrupted source segment. They
  are not identities and are not stable across restarts or reconnects.

### Virtual-fence intrusion

- Uses a configurable two-point line in source-image pixel coordinates.
- Generates `VIRTUAL_FENCE_INTRUSION` when a tracked object's visible bounding
  box first contacts the line, including a small configurable contact margin.
- Also checks centroid motion segments so sufficiently fast movement that skips
  from one side of the line to the other is not missed.
- Emits once per contact episode, subject to the alert cooldown.
- Displays the configured line and intrusion state on annotated frames.

### Suspicious loitering

- Applies to tracked people, not vehicles.
- Requires a continuous observation window of the configured duration.
- Resets a person's loitering history if the observation gap exceeds the
  configured maximum.
- Measures movement as the diagonal spread of all centroids retained in the
  window, rather than comparing only the first and last point.
- Generates `SUSPICIOUS_LOITERING` when duration and movement conditions first
  become active.

### Night/low-light movement

- Measures mean grayscale brightness in a central region of interest so borders,
  letterboxing, and edge glare have less influence.
- Applies exponential smoothing and hysteresis to reduce mode flicker.
- Generates `NIGHT_MOVEMENT` for tracked people and vehicles while night mode is
  active, subject to per-track/type cooldowns.
- This is brightness classification, not infrared sensing or a learned night
  detector, and it must be calibrated for each camera.

### Alerts and event delivery

The implemented analytics event types are:

| Event | Trigger |
|---|---|
| `VIRTUAL_FENCE_INTRUSION` | Tracked box contacts the fence or centroid motion crosses it |
| `SUSPICIOUS_LOITERING` | A continuously observed person remains within the movement limit |
| `NIGHT_MOVEMENT` | A tracked person or vehicle is detected during low-light mode |
| `ANPR_READ` | Optional ANPR returns a plate string above its confidence threshold |

Alert behavior includes:

- synchronous CSV persistence before an alert callback is invoked;
- immediate file flush after each accepted alert;
- server-system local timestamps with explicit UTC offsets, canonical UTC
  timestamps, and source/media-time seconds;
- cooldowns keyed by alert type and track ID;
- bounded cooldown-state retention;
- bounded recent alert labels on video overlays;
- bounded WebSocket alert history;
- a separate bounded queue for every connected WebSocket client, so one client
  cannot consume another client's events; and
- non-blocking server-device audible alarms for configured event types.

### Server-device audible alarms

In `server.py` mode, accepted analytics alerts can produce a sound from the
physical device running the server. On macOS this uses `/usr/bin/afplay` and the
current default audio output. No sound is sent through the browser.

The built-in patterns are intentionally short, non-verbal, and severity-coded:

| Event | Pattern | Default audible |
|---|---|---:|
| `VIRTUAL_FENCE_INTRUSION` | Rapid alternating high/low critical tone | Yes |
| `SUSPICIOUS_LOITERING` | Slower three-pulse rising caution tone | Yes |
| `NIGHT_MOVEMENT` | Low double pulse | No |
| `ANPR_READ` | Short two-note confirmation | No |

Fence contact needs immediate attention, so it uses the highest-priority and
most urgent pattern. Loitering is cautionary and uses a slower pattern. Night
movement can occur frequently and ANPR is informational, so enabling either by
default would create avoidable alarm fatigue.

Alarm playback:

- starts only after the alert has been persisted;
- runs on a dedicated worker and never blocks capture or analytics;
- uses a bounded priority queue, with fence intrusion ahead of lower-severity
  queued events;
- applies a separate per-event audible cooldown;
- generates its own temporary WAV files, requiring no downloaded sound asset;
- exposes status, last type, played count, dropped count, and error state through
  `/api/stats`; and
- is failure-isolated when the host has no player or audio output.

macOS uses `afplay`; Windows uses `winsound`; Linux uses `paplay` or `aplay` when
available. The server process must have access to an active audio session, and
the operating-system output must not be muted. Audition the patterns before
operational use:

```bash
python -m scripts.test_alarm intrusion --volume 0.75
python -m scripts.test_alarm loitering --volume 0.75
```

The other accepted test names are `night` and `anpr`.

### Authenticated web dashboard

`server.py` provides:

- an authenticated operations dashboard;
- a continuously updated MJPEG annotated stream;
- people, vehicle, alert, FPS, uptime, source-health, face-worker, reconnect,
  recording, and server-alarm status;
- bounded recent alert history and live per-client WebSocket delivery;
- identity-first face source review when enabled;
- liveness and readiness endpoints;
- camera/source reconnect handling; and
- lifecycle-managed annotated MP4 output independent of browser connections.

The API documentation and OpenAPI endpoints are disabled. The dashboard is
intended to run as exactly one application worker because camera ownership,
rate limits, alert queues, and analytics state are process-local.

### Identity-first face review with SCRFD

Face review is optional and isolated from the primary capture/analytics path:

- SCRFD detects faces and supplies five facial landmarks.
- A face is associated only with the smallest tracked person box containing its
  center.
- The authenticated panel shows **Detected source · no AI**, a padded copy of
  the original camera pixels from the SCRFD bounding box.
- It separately shows **Landmark aligned · no AI**, which contains source pixels
  geometrically transformed to GFPGAN's standard pose.
- The source crop is the default face API image and is never replaced by a
  generated portrait.
- Face work is submitted only on a configurable frame cadence.
- A one-job queue drops work rather than blocking capture when face processing
  falls behind.
- Results are capped, cached per short-lived track, expired, and held only in
  bounded in-memory dashboard snapshots.
- Face images are authenticated and never pasted into the MJPEG stream or MP4.
- Missing models, slow processing, alignment errors, and restoration failures
  do not stop object analytics, alerts, streaming, or recording.

SCRFD here performs detection and source extraction only. The prototype does
**not** perform face recognition, ArcFace embedding comparison, watchlist
matching, liveness detection, emotion inference, or persistent identity.

### Experimental GFPGAN preview

GFPGAN is available only as an explicitly enabled experimental visualization:

- disabled by default with `IBVAP_ENABLE_GFPGAN_RESTORATION=false`;
- loaded only when explicitly enabled;
- rejected for source faces below the configured size threshold;
- rejected when variance-of-Laplacian sharpness is too low;
- run with deterministic StyleGAN noise;
- blended at 10% GFPGAN and 90% aligned source by default; and
- constrained so its configurable contribution cannot exceed 50%.

GFPGAN is generative. It optimizes for a plausible face, not identity fidelity,
and can change eyes, nose, mouth, skin, age, or other identity-relevant details
even when a face is visible to a person. It cannot recover authentic information
that was never captured. Never use its output as identity or forensic evidence.

### Optional ANPR

When enabled, ANPR:

- runs only for detected vehicles and only at a configurable cadence;
- crops to the vehicle bounds before plate processing;
- uses OpenCV's bundled `haarcascade_russian_plate_number.xml` to localize a
  candidate plate;
- applies EasyOCR in CPU mode to the largest candidate;
- selects the OCR result with the highest confidence;
- strips non-alphanumeric characters and converts text to uppercase; and
- emits `ANPR_READ` only above the configured OCR confidence threshold.

The bundled plate cascade is not designed specifically for Indian plates. This
feature is a best-effort demonstration, not production ANPR.

### Source timing and lifecycle

- Recorded-video analytics use media timestamps, with a frame/FPS fallback when
  container timestamps are unavailable.
- Live sources use a monotonic clock.
- This makes recorded-video loitering and cooldown results independent of how
  quickly a particular machine processes the file.
- Invalid/non-finite FPS values are normalized to a safe fallback.
- RTSP/HTTP capture uses configurable open and read timeouts.
- `server.py` retries unavailable sources and reports reconnect/health state.
- In server mode, a file loops from frame zero and resets track-dependent state.
- In CLI mode, `main.py` stops at the end of a file.
- Graceful shutdown releases capture, analytics workers, log handles, and the
  MP4 writer.

## Architecture

```text
Camera index / file / RTSP(S) / HTTP(S)
                    │
                    ▼
             OpenCV VideoCapture
                    │
       media time (file) / monotonic time (live)
                    │
                    ▼
             YOLOv8 + tracker
                    │
       ┌────────────┼───────────────┬────────────────┐
       ▼            ▼               ▼                ▼
 virtual fence   loitering     night classifier   optional ANPR
       │            │               │                │
       └────────────┴───────┬───────┴────────────────┘
                            ▼
                     AlertLogger → CSV
                            │
                            ├────────→ bounded AlertBroker
                            │                 │
                            │                 └→ per-client WebSocket queues
                            └────────→ bounded AlarmPlayer → server speakers

Raw frame + tracked person bounds
                    │
                    └→ bounded face worker → SCRFD
                                             ├→ original source crop
                                             ├→ landmark-aligned source
                                             └→ optional GFPGAN preview

Annotated frame
       ├→ main.py: OpenCV window and/or MP4
       └→ server.py: MJPEG + MP4 + statistics dashboard
```

The face worker receives a copy of the raw frame before object overlays are
drawn. Face processing does not alter the primary annotated frame.

## Technology stack

### Core runtime

| Technology | Use |
|---|---|
| Python 3.11 | Reference runtime |
| OpenCV | Capture, image conversion, drawing, JPEG/MP4 encoding, plate cascade |
| Ultralytics YOLOv8 | Person/vehicle detection and tracker integration |
| PyTorch | YOLO runtime and optional GFPGAN inference |
| NumPy | Frame and numerical operations |
| FastAPI / Starlette | Authenticated HTTP and WebSocket application |
| Uvicorn | ASGI server |
| HTML/CSS/JavaScript | Dependency-free dashboard frontend |

### Optional components

| Component | Technology | Notes |
|---|---|---|
| Face source review | SCRFD via InsightFace + ONNX Runtime | Requires separately reviewed model weights |
| Face alignment | SCRFD five landmarks + OpenCV affine transform | Source pixels, but geometry is transformed |
| Generative face preview | Vendored minimal GFPGAN clean inference architecture | Experimental; disabled by default |
| ANPR | OpenCV Haar cascade + EasyOCR | CPU OCR; separate dependency lock |
| Audible alerts | Generated WAV + `afplay`/`winsound`/`paplay`/`aplay` | Plays on the server device, not the browser |
| Acceleration | ONNX Runtime CoreML/CUDA/CPU; PyTorch MPS/CUDA/CPU | Selected from available providers/devices |
| Container deployment | Docker / Docker Compose | Base image includes core runtime only |
| HTTPS | Uvicorn certificate/key or trusted reverse proxy | Required outside loopback development |

The external vulnerable BasicSR runtime is deliberately not installed. Only the
minimal attributed GFPGAN/BasicSR-derived inference code needed by this project
is vendored under `third_party/gfpgan_arch/`.

## Entry points

| Command | Intended use | Dashboard auth | File behavior | MP4 behavior |
|---|---|---:|---|---|
| `python main.py` | Local window or headless batch/camera processing | No | Stops at EOF | Writes when enabled; writer failure stops startup |
| `python server.py` | Authenticated dashboard and long-running capture | Yes | Loops at EOF | Writes independently of clients; failures are isolated |

Do not run `main.py` and `server.py` against the same physical camera at the
same time. Many camera backends permit only one owner. Audible-alarm playback is
currently wired to `server.py`, because it owns the long-running monitored
service lifecycle.

## Requirements

- Python 3.11–3.13; Python 3.11 is the reference and CI version.
- Linux, Windows, or Apple Silicon running macOS 14 or newer.
- Sufficient RAM, disk, and compute for PyTorch and Ultralytics.
- A webcam, recording, or OpenCV-compatible stream.
- A protected secret source for dashboard credentials.
- TLS termination for dashboard access outside local loopback.

Current security-patched PyTorch releases no longer publish Intel macOS wheels,
so Intel Macs are not supported by the locked environment. Do not silently
downgrade PyTorch to an older vulnerable release.

## Installation

### Core runtime

```bash
python3.11 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python scripts/doctor.py
```

Always run the application with the same interpreter used for installation:

```bash
which python
python -c "import cv2; print(cv2.__version__)"
```

The lock is a universal requirements file. Linux-only CUDA packages have
platform markers and are skipped on macOS and Windows. Use `-r`:
`pip install requirements.lock` is not equivalent to
`pip install -r requirements.lock`.

The repository intentionally contains `yolov8n.pt` for offline startup. Its
checksum and provenance are recorded in `models/MODEL_MANIFEST.json`.

### Optional ANPR dependencies

```bash
python -m pip install -r requirements-anpr.lock
export IBVAP_ENABLE_ANPR=true
```

EasyOCR may acquire its own separately governed weights on first use. Review
those terms and network-access implications before enabling ANPR.

### Optional SCRFD face source review

```bash
python -m pip install -r requirements-face.lock
python scripts/download_face_models.py
```

The downloader first prints model terms and exits without changing anything.
InsightFace-provided pretrained models are restricted to non-commercial
research use. After reviewing the terms, either provide appropriately licensed
SCRFD-compatible weights or explicitly accept the reviewed model terms:

```bash
python scripts/download_face_models.py --accept-model-licenses
python scripts/doctor.py --face
export IBVAP_ENABLE_FACE_ENHANCEMENT=true
export IBVAP_ENABLE_GFPGAN_RESTORATION=false
```

The downloader verifies expected archive/model size and SHA-256 metadata before
installation. Downloaded face weights are placed under `models/`, ignored by
Git, and are never fetched silently by the application.

To experiment with GFPGAN despite its identity limitations:

```bash
export IBVAP_ENABLE_GFPGAN_RESTORATION=true
```

## Configuration

All application configuration is read from `IBVAP_*` environment variables at
process startup. Restart the process after changing a value. The application
deliberately does **not** parse `.env` files automatically; use a shell,
macOS Keychain-backed launcher, systemd/launchd, a container secret mechanism,
or another protected process manager. Never commit credentials or camera URLs.

Relative paths are resolved from the repository directory, not the terminal's
current working directory.

### Input and model

| Variable | Default | Meaning |
|---|---:|---|
| `IBVAP_VIDEO_SOURCE` | `0` | Camera index, file path, RTSP(S), or HTTP(S) URL |
| `IBVAP_YOLO_MODEL` | `yolov8n.pt` | Local model path or Ultralytics model setting |
| `IBVAP_YOLO_MODEL_SHA256` | Bundled model digest | Expected hash; empty disables verification for a custom model |
| `IBVAP_CONFIDENCE_THRESHOLD` | `0.4` | YOLO detection cutoff, range 0–1 |
| `IBVAP_LOG_LEVEL` | `INFO` | Python log level used by both entry points |

### Core analytics

| Variable | Default | Meaning |
|---|---:|---|
| `IBVAP_VIRTUAL_FENCE` | `50,400,1200,400` | Fence endpoints `x1,y1,x2,y2` in source pixels |
| `IBVAP_VIRTUAL_FENCE_CONTACT_MARGIN_PX` | `2` | Visible-box/fence contact allowance |
| `IBVAP_LOITERING_SECONDS` | `15` | Required continuous dwell duration |
| `IBVAP_LOITERING_MOVEMENT_THRESHOLD_PX` | `60` | Maximum centroid-window diagonal spread |
| `IBVAP_TRACK_MAX_OBSERVATION_GAP_SECONDS` | `1.5` | Gap that resets loitering continuity |
| `IBVAP_TRACK_STALE_SECONDS` | `3` | Retention period for missing track state |
| `IBVAP_NIGHT_BRIGHTNESS_THRESHOLD` | `60` | Smoothed grayscale threshold entering night mode |
| `IBVAP_NIGHT_BRIGHTNESS_HYSTERESIS` | `5` | Additional brightness needed to leave night mode |
| `IBVAP_NIGHT_BRIGHTNESS_SMOOTHING_ALPHA` | `0.15` | Exponential smoothing coefficient |
| `IBVAP_NIGHT_BRIGHTNESS_ROI_MARGIN` | `0.05` | Fraction removed from each brightness-ROI edge |

### Face source review and GFPGAN

| Variable | Default | Meaning |
|---|---:|---|
| `IBVAP_ENABLE_FACE_ENHANCEMENT` | `false` | Enable SCRFD source-face review |
| `IBVAP_ENABLE_GFPGAN_RESTORATION` | `false` | Enable experimental generative preview |
| `IBVAP_FACE_ENHANCEMENT_INTERVAL_FRAMES` | `15` | Raw-frame submission cadence |
| `IBVAP_FACE_ENHANCEMENT_MAX_FACES` | `4` | Maximum retained faces per update |
| `IBVAP_FACE_ENHANCEMENT_REFRESH_SECONDS` | `2` | Minimum cache refresh period per track |
| `IBVAP_FACE_ENHANCEMENT_CACHE_TTL_SECONDS` | `10` | Face-cache retention period |
| `IBVAP_FACE_ENHANCEMENT_JPEG_QUALITY` | `90` | Dashboard face JPEG quality, range 1–100 |
| `IBVAP_FACE_ENHANCEMENT_DEVICE` | `auto` | GFPGAN device: `auto`, `cpu`, `mps`, or `cuda` |
| `IBVAP_SCRFD_CONFIDENCE_THRESHOLD` | `0.6` | SCRFD confidence cutoff |
| `IBVAP_SCRFD_INPUT_SIZE` | `640` | SCRFD input side; must be divisible by 32 |
| `IBVAP_SCRFD_MIN_FACE_SIZE_PX` | `32` | Smallest face retained as source review |
| `IBVAP_SCRFD_MODEL_PATH` | `models/scrfd_2.5g.onnx` | SCRFD ONNX file |
| `IBVAP_GFPGAN_MODEL_PATH` | `models/GFPGANv1.4.pth` | GFPGAN checkpoint |
| `IBVAP_GFPGAN_MODEL_SHA256` | Reviewed v1.4 digest | Required checkpoint digest |
| `IBVAP_GFPGAN_MIN_SOURCE_FACE_SIZE_PX` | `96` | Minimum source-face dimension for GFPGAN |
| `IBVAP_GFPGAN_MIN_SOURCE_SHARPNESS` | `50` | Minimum variance-of-Laplacian sharpness |
| `IBVAP_GFPGAN_BLEND_WEIGHT` | `0.10` | GFPGAN contribution, constrained to 0–0.5 |

### ANPR

| Variable | Default | Meaning |
|---|---:|---|
| `IBVAP_ENABLE_ANPR` | `false` | Enable plate localization and OCR |
| `IBVAP_ANPR_INTERVAL_FRAMES` | `15` | Per-vehicle OCR cadence |
| `IBVAP_ANPR_MIN_OCR_CONFIDENCE` | `0.35` | Minimum accepted EasyOCR confidence |

### Alerts, logs, and output

| Variable | Default | Meaning |
|---|---:|---|
| `IBVAP_ALERT_COOLDOWN_SECONDS` | `1` | Minimum repeat interval per alert type/track |
| `IBVAP_ALERT_STATE_TTL_SECONDS` | `300` | Cooldown-state retention |
| `IBVAP_RECENT_ALERT_DISPLAY_SECONDS` | `8` | Annotated-frame banner retention |
| `IBVAP_MAX_RECENT_ALERTS` | `20` | Maximum recent overlay labels |
| `IBVAP_ENABLE_AUDIBLE_ALARMS` | `true` | Enable server-device alarm playback |
| `IBVAP_AUDIBLE_ALARM_EVENTS` | `VIRTUAL_FENCE_INTRUSION,SUSPICIOUS_LOITERING` | Comma-separated supported event names |
| `IBVAP_AUDIBLE_ALARM_VOLUME` | `0.75` | macOS playback multiplier, range 0–1 |
| `IBVAP_AUDIBLE_ALARM_COOLDOWN_SECONDS` | `5` | Minimum interval per audible event type |
| `IBVAP_AUDIBLE_ALARM_QUEUE_SIZE` | `8` | Maximum queued alarm patterns |
| `IBVAP_SAVE_ANNOTATED_VIDEO` | `true` | Enable annotated MP4 output |
| `IBVAP_ANNOTATED_VIDEO_PATH` | `output/annotated_output.mp4` | MP4 destination |
| `IBVAP_ALERT_LOG_CSV` | `logs/alerts.csv` | Analytics event CSV |
| `IBVAP_AUDIT_LOG_JSONL` | `logs/audit.jsonl` | Dashboard security audit log |
| `IBVAP_ALERT_LOG_MAX_BYTES` | `10485760` | Alert CSV rotation threshold |
| `IBVAP_AUDIT_LOG_MAX_BYTES` | `10485760` | Audit JSONL rotation threshold |
| `IBVAP_LOG_BACKUP_COUNT` | `5` | Number of rotated log backups |
| `IBVAP_SHOW_LIVE_WINDOW` | `true` | Show `main.py` OpenCV window when display is available |

### Capture and web-service lifecycle

| Variable | Default | Meaning |
|---|---:|---|
| `IBVAP_SERVER_HOST` | `0.0.0.0` | Uvicorn bind address; host-header rules still apply |
| `IBVAP_SERVER_PORT` | `8000` | Uvicorn port |
| `IBVAP_CAMERA_RECONNECT_SECONDS` | `2` | Source reconnect delay |
| `IBVAP_CAMERA_OPEN_TIMEOUT_MILLISECONDS` | `5000` | Network-source open timeout |
| `IBVAP_CAMERA_READ_TIMEOUT_MILLISECONDS` | `5000` | Network-source read timeout |
| `IBVAP_FILE_PLAYBACK_REALTIME` | `true` | Pace looping files in server mode to source FPS |
| `IBVAP_STATS_PUSH_INTERVAL_SECONDS` | `0.4` | WebSocket statistics interval |
| `IBVAP_FRAME_STALE_AFTER_SECONDS` | `10` | Maximum frame age for readiness |
| `IBVAP_MAX_STREAM_CLIENTS` | `8` | Concurrent MJPEG connection cap |
| `IBVAP_MAX_WEBSOCKET_CLIENTS` | `16` | Concurrent WebSocket cap |
| `IBVAP_WEBSOCKET_ATTEMPTS_PER_MINUTE` | `60` | WebSocket attempt limit per client address |
| `IBVAP_ALERT_SUBSCRIBER_QUEUE_SIZE` | `100` | Per-client alert queue capacity |
| `IBVAP_ALERT_HISTORY_SIZE` | `100` | Recent alerts replayed to new clients |

### Authentication and transport security

| Variable | Default | Meaning |
|---|---:|---|
| `IBVAP_DASHBOARD_USERNAME` | `operator` | Single prototype operator username |
| `IBVAP_DASHBOARD_PASSWORD` | none | Required; minimum 12 characters |
| `IBVAP_SESSION_SECRET` | none | Required HMAC secret; minimum 32 characters |
| `IBVAP_SESSION_TTL_SECONDS` | `28800` | Signed session lifetime |
| `IBVAP_SESSION_COOKIE_NAME` | `ibvap_session` | Session cookie name |
| `IBVAP_LOGIN_ATTEMPTS_PER_MINUTE` | `5` | Login attempt cap per client address |
| `IBVAP_LOGIN_RATE_LIMIT_MAX_CLIENTS` | `10000` | Maximum in-memory limiter keys |
| `IBVAP_ALLOWED_HOSTS` | `localhost,127.0.0.1` | Exact/wildcard host-header allowlist |
| `IBVAP_ALLOWED_ORIGINS` | empty | Additional exact allowed browser origins |
| `IBVAP_REQUIRE_HTTPS` | `false` | Reject non-health HTTP requests without HTTPS |
| `IBVAP_HTTPS_BEHIND_PROXY` | `false` | Declare trusted proxy TLS termination |
| `IBVAP_COOKIE_SECURE` | `false` | Always mark the session cookie Secure |
| `IBVAP_TLS_CERTFILE` | none | Direct-TLS certificate path |
| `IBVAP_TLS_KEYFILE` | none | Direct-TLS private-key path |
| `IBVAP_FORWARDED_ALLOW_IPS` | `127.0.0.1` | Addresses trusted to supply forwarded headers |

## Running locally with `main.py`

Process a recording:

```bash
export IBVAP_VIDEO_SOURCE=sample_data/demo.mp4
export IBVAP_SHOW_LIVE_WINDOW=true
python main.py
```

For headless processing:

```bash
export IBVAP_SHOW_LIVE_WINDOW=false
python main.py
```

Press `q` in the OpenCV window or Ctrl+C in the terminal. `main.py` exits at the
end of a recording.

## Running the authenticated dashboard

The server fails closed unless the username is at least 3 characters, the
password is at least 12 characters, the session secret is at least 32
characters, host rules are non-empty, and HTTPS settings are internally
consistent.

```bash
export IBVAP_DASHBOARD_USERNAME=operator
export IBVAP_DASHBOARD_PASSWORD="use-a-protected-password-of-at-least-12-characters"
export IBVAP_SESSION_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export IBVAP_VIDEO_SOURCE=0
python server.py
```

Open `http://localhost:8000/login`. Plain HTTP is suitable only for loopback
development. Store stable credentials in macOS Keychain, a secret manager, or a
protected service definition instead of adding them to Git. A persistent session
secret also prevents all sessions from changing whenever the process restarts.

`localhost`, `127.0.0.1`, and `::1` are treated as equivalent local origins when
the effective port matches. Other hosts and origins remain restricted. The
header clock and alert times are rendered in the browser's system timezone;
alert instants arrive as UTC and are converted with the browser's local clock.

### Dashboard authentication and security controls

- One environment-configured operator account.
- Constant-time username/password comparison.
- HMAC-SHA-256 signed session cookies with issued-at and expiry timestamps.
- `HttpOnly` and `SameSite=Strict`; `Secure` under HTTPS or when forced.
- Host-header allowlisting.
- Origin/Fetch-Metadata checks on login and logout.
- Same-origin, authenticated WebSocket validation.
- Per-client login and WebSocket attempt limits.
- Global MJPEG and WebSocket connection caps.
- 8 KiB login-body limit.
- CSP, `nosniff`, frame denial, no-referrer, permissions, no-store, and HSTS
  headers where applicable.
- Optional mandatory HTTPS and direct/reverse-proxy TLS configurations.
- Structured security event logging to `audit.jsonl`.

The password is supplied to the process; the application does not maintain a
user database or persist a password hash. Logout deletes the browser cookie,
but there is no centralized token revocation list. Rotate the session secret to
invalidate every outstanding session.

### API and streaming endpoints

| Endpoint | Authentication | Purpose |
|---|---:|---|
| `GET /health/live` | No session | Capture-thread liveness; host checks still apply |
| `GET /health/ready` | No session | Recent-frame readiness and service status |
| `GET /login` | No session | Login page |
| `POST /login` | Credentials | Create signed session cookie |
| `POST /logout` | Session where present | Delete session cookie |
| `GET /` | Required | Dashboard HTML |
| `GET /stream` | Required | Annotated MJPEG stream |
| `GET /api/stats` | Required | Current analytics/capture/recording statistics |
| `GET /api/faces` | Required | Face worker state and versioned image references |
| `GET /api/faces/{index}?variant=source` | Required | Original detected camera-pixel crop |
| `GET /api/faces/{index}?variant=aligned` | Required | Landmark-aligned source crop |
| `GET /api/faces/{index}?variant=review` | Required | Experimental GFPGAN image, when available |
| `WS /ws` | Required | Per-client live alerts and periodic statistics |

## Output files and retention

### Annotated MP4

Both entry points honor `IBVAP_SAVE_ANNOTATED_VIDEO` and
`IBVAP_ANNOTATED_VIDEO_PATH`. The default is:

```text
output/annotated_output.mp4
```

The recording contains visual overlays but no audio. It uses OpenCV's `mp4v`
writer. In server mode, recording begins after the first successfully processed
frame and does not depend on a connected browser. Writer open/write errors set
`recording_status=failed` but do not stop streaming or analytics.

Stop the process gracefully with Ctrl+C or SIGTERM so OpenCV can release the
writer and finalize MP4 container metadata. A running or forcibly killed process
may leave the file apparently empty or unplayable.

### Analytics alert CSV

The default analytics log is:

```text
logs/alerts.csv
```

It is created with a header when `VideoPipeline` successfully initializes. Data
rows appear only when an alert occurs. Columns are:

```text
timestamp_local,timestamp_utc,source_time_seconds,alert_type,track_id,details,frame_number
```

`timestamp_local` follows the server host's configured system timezone and
always includes its UTC offset. `timestamp_utc` records the same instant in UTC.
`source_time_seconds` remains the media timeline for files or a monotonic clock
for live analytics; it is not a wall-clock value. When upgrading an existing
installation, a CSV with the older header is preserved as `alerts.csv.1` and a
new file is opened with the current schema.

To place it beside the MP4:

```bash
export IBVAP_ALERT_LOG_CSV=output/alerts.csv
```

### Dashboard security audit JSONL

The separate server security log is:

```text
logs/audit.jsonl
```

It records login success/failure, rejected origins, rate limits, logout, MJPEG
connections, and WebSocket connection lifecycle as structured JSON lines. Each
record contains `timestamp_local` with the server's UTC offset and
`timestamp_utc` for canonical correlation. `audit.jsonl` does not replace
`alerts.csv`: the former is a security access log;
the latter contains video analytics events.

Both logs rotate by size with numbered backups. They are local operational logs,
not immutable or tamper-evident evidence.

### Face images

Face source/aligned/review JPEGs are kept in bounded process memory for the
authenticated panel. The application does not write them as individual files.
Screenshots, browser caches outside application control, and the original video
must still be governed by an appropriate retention policy.

Generated logs, output video, downloaded face weights, caches, bytecode, and
local secrets are excluded from Git. The bundled `yolov8n.pt` remains tracked
intentionally.

## Troubleshooting

### `No module named 'cv2'`

The import is named `cv2`, but the package is `opencv-python`. Confirm that the
terminal running IBVAP uses the environment where the lock was installed:

```bash
which python
python -c "import cv2; print(cv2.__version__)"
python scripts/doctor.py
```

If those commands use the wrong interpreter, activate `.venv` or run the entry
point explicitly with `.venv/bin/python`. If a platform-specific installation
failed partway, recreate the environment and reinstall `requirements.lock`
rather than downgrading individual packages.

### Webcam does not open on macOS

- Close applications that may already own the camera.
- In **System Settings → Privacy & Security → Camera**, allow the terminal or
  launcher used to start Python.
- Confirm `IBVAP_VIDEO_SOURCE=0`; try another integer only when the machine has
  multiple capture devices.
- Do not run `main.py` and `server.py` against the same camera simultaneously.

### `audit.jsonl` exists but `alerts.csv` does not

These files are independent. `audit.jsonl` is written by dashboard security
events. `alerts.csv` is created only after `VideoPipeline` initializes
successfully, and receives data rows only when an analytics event occurs. Check
the server terminal for model or pipeline initialization errors and print the
resolved paths with:

```bash
.venv/bin/python -c 'import config; print(config.ALERT_LOG_CSV); print(config.AUDIT_LOG_JSONL)'
```

### Dashboard or log time looks incorrect

The dashboard converts each UTC event instant into the browser's system
timezone. Confirm the browser/OS date, time, and timezone settings. Logs include
both `timestamp_local` (server system timezone with offset) and `timestamp_utc`.
If the browser runs on a different machine or timezone than the server, its
displayed local time can intentionally differ from the server-local log column;
the UTC values still identify the same instant.

### Annotated MP4 appears empty or is missing

- Confirm `IBVAP_SAVE_ANNOTATED_VIDEO=true`.
- Check `recording_status` on the dashboard or `/api/stats`.
- Recording begins only after a frame is successfully processed.
- Inspect the terminal for writer/codec/path errors.
- Stop with Ctrl+C or SIGTERM before opening the final file so MP4 metadata is
  finalized.

### Server alarm does not sound

- Run `python -m scripts.test_alarm intrusion --volume 0.75` on the server host.
- On macOS, verify the intended output under **System Settings → Sound** and
  confirm the Mac is not muted.
- Confirm `IBVAP_ENABLE_AUDIBLE_ALARMS=true` and inspect `alarm_status` through
  the dashboard or `/api/stats`.
- Alarms play from the server machine, not from a remote browser.
- Headless service accounts and containers may not have access to a desktop
  audio session.

### Face appears inaccurate

Use **Detected source · no AI** as the authoritative crop. It contains the
camera pixels SCRFD detected. Landmark alignment can distort geometry when
landmarks are weak. Keep `IBVAP_ENABLE_GFPGAN_RESTORATION=false`; GFPGAN and
other generative restoration tools cannot guarantee identity fidelity. Improve
camera distance, focus, lighting, exposure, and source resolution instead of
using a stronger generative blend.

## HTTPS deployment

For direct Uvicorn TLS:

```bash
export IBVAP_TLS_CERTFILE=/secure/path/tls.crt
export IBVAP_TLS_KEYFILE=/secure/path/tls.key
export IBVAP_REQUIRE_HTTPS=true
export IBVAP_COOKIE_SECURE=true
python server.py
```

For a reverse proxy:

- terminate TLS at the trusted proxy;
- set `IBVAP_HTTPS_BEHIND_PROXY=true`;
- set `IBVAP_FORWARDED_ALLOW_IPS` to only the proxy's exact address(es), never
  `*`;
- configure `IBVAP_ALLOWED_HOSTS` with approved public hostnames;
- configure `IBVAP_ALLOWED_ORIGINS` with exact public HTTPS origins;
- set `IBVAP_REQUIRE_HTTPS=true` and `IBVAP_COOKIE_SECURE=true`; and
- restrict network access with a firewall or VPN.

See [SECURITY.md](SECURITY.md) for the deployment checklist.

## Docker

```bash
export IBVAP_DASHBOARD_PASSWORD="$(python -c 'import secrets; print(secrets.token_urlsafe(24))')"
export IBVAP_SESSION_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
docker compose up --build
```

The default Compose port binds to `127.0.0.1:8000`, runs as a non-root user,
enables `no-new-privileges`, persists `logs/` and `output/`, and mounts
`sample_data/` read-only. Configure camera-device access or an RTSP URL
explicitly for the deployment host.

The base Docker image installs only `requirements.lock`. ANPR and face extras,
face model mounts, hardware acceleration, production TLS, and external secret
management require an explicitly reviewed image/deployment extension. Container
audio is not routed to the host by default, so Compose disables audible alarms
unless explicitly overridden. Enabling them requires deliberately designed and
tested secure host audio access.

## Testing, reproducibility, and repository hygiene

Install lightweight development tooling:

```bash
python -m pip install -r requirements-dev.lock
make check
```

`make check` performs:

1. Ruff formatting checks;
2. Ruff linting;
3. unit tests with branch coverage;
4. a Bandit static security scan; and
5. vulnerability audits of runtime, ANPR, face, and development locks.

The current suite contains 74 unit tests covering geometry, fence contact,
loitering continuity, source-time behavior, alert cooldown/rotation, bounded
broadcast, authentication, sessions, host/origin controls, rate limits,
source parsing, server recording failure isolation, audible-alarm patterns and
fan-out, face quality gates, and source/review API access. The configured 85%
coverage threshold and reported
coverage apply to `utils` and `security`; they are not a claim of whole-system
or ML-model coverage.

CI runs on Python 3.11, repeats formatting/lint/test/security/audit checks,
dry-resolves core and face locks for Apple Silicon macOS 14, and separately
installs pinned OpenCV on a macOS 14 runner to verify that `import cv2` works.

Dependency inputs and universal pinned locks are separated:

| Purpose | Input | Lock |
|---|---|---|
| Core runtime | `requirements.txt` | `requirements.lock` |
| ANPR | `requirements-anpr.txt` | `requirements-anpr.lock` |
| Face review | `requirements-face.txt` | `requirements-face.lock` |
| Development | `requirements-dev.txt` | `requirements-dev.lock` |

Use `make lock` to regenerate locks with the recorded command. Review resulting
diffs and rerun `make check` before accepting dependency changes.

Automated tests use fakes for heavy camera/model behavior. CI does not prove
real-world detection accuracy, camera-driver compatibility, OCR quality, or
GFPGAN visual fidelity. Those require representative hardware and labelled
footage.

## Security and privacy properties

Implemented safeguards include:

- fail-closed dashboard credential and TLS configuration validation;
- no source-controlled dashboard password/session secret;
- signed expiring `HttpOnly` sessions;
- host, origin, HTTPS, and WebSocket validation;
- bounded login/connection/queue state;
- security headers and disabled API docs;
- rotating alert and audit logs;
- checksum verification for bundled YOLO and reviewed face model acquisition;
- explicit model-license acceptance before face model download;
- ignored generated data and model weights; and
- a non-root, no-new-privileges default container.

These controls do not replace network segmentation, OS hardening, encrypted
storage, access review, centralized monitoring, incident response, or an
independent penetration test.

## Limitations and non-goals

### Accuracy and analytics

- YOLO, ByteTrack, SCRFD, Haar cascades, OCR, and brightness thresholds can all
  produce false positives and false negatives.
- The bundled YOLO model is a general COCO model, not a border-domain model.
- Occlusion, crowding, motion blur, compression, low resolution, weather,
  camera angle, glare, and darkness reduce detection and tracking quality.
- Track-ID switches can duplicate, suppress, or misassociate track-based events.
- The virtual fence uses 2D pixel geometry and must be recalibrated whenever
  camera position, lens, crop, or resolution changes.
- Loitering uses image-plane movement, not world coordinates; perspective means
  one pixel threshold does not represent the same physical distance everywhere.
- Night mode is a coarse brightness heuristic and does not distinguish shadows,
  exposure changes, infrared mode, or actual time of day.
- Alert cooldowns reduce repetition but are not event deduplication across
  cameras, processes, or restarts.

### Faces and identity

- SCRFD source review is face detection/cropping, not identification.
- The original crop is only as detailed as the camera pixels; enlarging a crop
  cannot create authentic missing information.
- Landmark alignment changes geometry and may be poor when landmarks are noisy,
  profile, occluded, or low resolution.
- GFPGAN and alternative restoration/super-resolution generators may invent
  identity details. A visually attractive portrait is not proof of identity.
- No face recognition, watchlists, cross-camera re-identification, persistent
  IDs, liveness detection, anti-spoofing, or biometric database is implemented.

### ANPR

- The Russian-plate Haar cascade is not optimized for Indian plate formats,
  scripts, layouts, fonts, mounting angles, or traffic conditions.
- EasyOCR is CPU-only in the current integration and may be slow.
- OCR output is not jurisdictionally validated and can contain plausible but
  incorrect characters.
- ANPR requires separate dependencies and potentially separately downloaded
  OCR weights.

### Operations and evidence

- Only one video source is owned by one process.
- Horizontal scaling is not implemented. Multiple workers would duplicate
  capture and analytics and would not share limits, queues, or state.
- There is no database, durable alert acknowledgment workflow, case management,
  search interface, centralized telemetry, or remote configuration service.
- CSV and JSONL are local append-only operational files, not tamper-evident logs.
- MP4 output is annotated, uses `mp4v`, has no audio, and may be corrupt after an
  ungraceful shutdown.
- Face thumbnails are in memory only and disappear on restart.
- There is no raw-video archival subsystem; preserve source recordings through
  an independently governed system when required.
- Health endpoints report process/capture readiness, not end-to-end analytics
  accuracy or camera-scene correctness.
- Audible alarms depend on the host audio player, active output device, volume,
  and physical audibility. They are not a substitute for a supervised alarm
  console, redundant notification path, or life-safety signalling system.

### Security and account model

- The dashboard has one shared operator role, no user database, MFA, SSO, RBAC,
  account lockout workflow, or per-user authorization.
- Rate limits, connection counts, capture state, and alert queues are in memory
  and reset on restart.
- Signed sessions cannot be individually revoked server-side before expiry;
  rotate the session secret to invalidate all sessions.
- HTTP is acceptable only for loopback development. Production use needs TLS,
  network controls, protected secrets, and centralized logs.
- Camera credentials in environment variables may be visible to privileged host
  administrators or process-inspection tooling.

### Validation and compliance

- The project has not undergone independent security, privacy, accessibility,
  performance, fairness, or penetration testing.
- It is not certified for evidentiary, biometric, border-control, public-safety,
  or high-availability use.
- Operators are responsible for lawful authority, notice/consent where required,
  data minimization, retention, deletion, disclosure, and subject-rights
  processes.

## Repository policy and licensing

The original IBVAP project code is proprietary and all rights are reserved; see
[LICENSE](LICENSE). Third-party code, dependencies, and models retain their own
terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Important model considerations:

- The bundled Ultralytics YOLO weights identify AGPL-3.0 licensing. Review
  Ultralytics' current terms and obtain an appropriate license where required.
- InsightFace code is open source, but its supplied pretrained model terms limit
  those models to non-commercial research. Obtain separately licensed compatible
  weights for other uses.
- GFPGAN's code/model have their own upstream terms and its output remains
  generative regardless of licensing.
- EasyOCR and any downloaded OCR weights retain their own terms.

Do not commit credentials, camera URLs, operational footage, faces, number
plates, alerts, audit logs, generated video, downloaded weights, or other
sensitive/generated artifacts.
