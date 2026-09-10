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
| Wide-area loitering | Whole-frame person observation by default; optional polygon scope | Enabled |
| Group approach | Proximity clusters plus correlated movement toward one fence segment | Optional, disabled |
| Motion signature | Heuristic centroid-oscillation score for composite risk only | Experimental, disabled |
| Tiled small-object inference | Cadence-limited overlapping tiles plus cross-tile NMS | Optional, disabled |
| Native digital focus | Source-pixel crop confirmation plus bounded tracker-acquisition boost | Experimental, disabled |
| Controlled evaluation | Identical-frame wide/focus traces, pixel-band metrics, tracking/events/runtime report | Offline tool |
| Low-posture review | Failure-isolated pose worker with size and persistence gates | Experimental, disabled |
| Firearm review | No detector, alert, dashboard, or alarm path | Not implemented |
| Contextual risk | Explainable config-weighted per-track signal composition | Experimental, disabled |
| Local VLM scene captioning | Stretch capability intentionally not implemented in this pass | Not implemented |
| Night movement | Smoothed central-ROI brightness with hysteresis | Enabled |
| Alert persistence | Rotating CSV with local-system, UTC, and source/media timestamps | Enabled |
| Server audible alarms | Distinct bounded tones for intrusion and loitering | Enabled |
| Security audit log | Rotating structured JSONL for dashboard security events | Server mode |
| Dashboard | FastAPI, authenticated MJPEG, statistics, and WebSocket alerts | Server mode |
| Annotated MP4 | `mp4v` recording from both `main.py` and `server.py` | Enabled |
| Face source review | SCRFD detection, original crop, and landmark-aligned source | Optional |
| GFPGAN preview | Deterministic, quality-gated, low-strength generative preview | Disabled |
| Regional ANPR | Tracked vehicle, external YOLO plate detector, EasyOCR, validation, temporal consensus | Disabled |
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

- Calls Ultralytics with explicit `tracker="bytetrack.yaml"`, `persist=True`,
  and a validated configurable detector input size rather than relying on
  Ultralytics tracker/input defaults.
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

### Wide-area suspicious loitering

- Applies to tracked people, not vehicles, anywhere in the frame by default;
  it has no implicit fence-distance gate.
- `IBVAP_LOITERING_ZONE=anywhere` retains whole-frame behavior. A polygon such
  as `100,100;1100,100;1100,650;100,650` can explicitly scope it later.
- Requires a continuous observation window of the configured duration.
- Resets a person's loitering history after an excessive observation gap or
  when the track leaves an explicitly configured polygon.
- Measures movement as the diagonal spread of all centroids retained in the
  window, rather than comparing only the first and last point.
- Generates `SUSPICIOUS_LOITERING` when duration and movement conditions first
  become active.

### Group/coordinated approach

The optional `GroupApproachDetector` uses no additional ML model. It clusters
currently visible person tracks by image-plane proximity, calculates velocity
from short centroid histories, and requires a minimum-size cluster whose
headings are correlated and whose members make measurable progress toward the
configured fence segment. `GROUP_APPROACH` is emitted once per qualifying group
episode and can also contribute to contextual risk.

This is perspective-sensitive image geometry, not proof that people are acting
in coordination. Track switches, crowd overlap, camera vibration, and an
uncalibrated fence can all affect it. It defaults off and requires labelled
scene-specific tuning.

### Motion-signature discrimination signal

The optional motion-signature analyzer reuses loitering trajectory history (and
therefore the same optional loitering polygon scope) and examines net
displacement, vertical-displacement sign changes, and oscillation magnitude. It
returns a bounded score with rough diagnostic categories for
static, high-frequency/no-net motion, directional rhythmic motion, and rigid
directional motion. It never emits a standalone alert; only the contextual
risk scorer can consume it.

This heuristic is unvalidated on real border footage and cannot reliably
separate humans, animals, and vegetation—especially for tiny blobs. It is not a
substitute for thermal sensing, calibrated multi-sensor fusion, or a trained
and independently validated classifier.

### SAHI-style tiled inference

Optional tiled inference keeps normal full-frame tracking active, then on a
configurable cadence runs the same COCO detector on overlapping full-resolution
tiles. Tile detections are mapped back to frame coordinates and class-aware NMS
merges duplicates at tile boundaries. Supplemental detections remain untracked
until normal full-frame tracking acquires them; the implementation does not
invent persistent IDs from independent tile predictions.

Tiling can improve recall for some small objects but multiplies inference work,
can add false positives, and has no real-time guarantee. It defaults off.

### Native source-pixel focus and tracker acquisition

The optional native-focus bridge accepts untracked full-frame or tile proposals
on a configurable cadence. A failure-isolated worker keeps at most one pending
job, copies the untouched source frame, pads each selected proposal in source
coordinates, and re-runs the same checksum-verified COCO model on that crop.
Repeated overlapping observations form a short-lived **candidate episode**, not
an object identity or persistent track ID.

After temporal confirmation, the worker requests a bounded run of larger-input
full-frame inference. The normal, explicitly configured ByteTrack instance must
still acquire the object before fence, loitering, group, posture, ANPR, or risk
state can use it. The dashboard displays the crop as **DIGITAL FOCUS — SOURCE
PIXELS ONLY**, including original target dimensions, confidence, confirmation
progress, provenance, and acquisition state. No super-resolution or generative
enhancement is used. Browser enlargement does not create sensor detail.

The feature is disabled by default because it owns a second model instance and
can materially reduce throughput. A one-slot drop-on-busy queue, CPU/MPS/CUDA
device selection, visible worker status, model checksum verification, and
exception isolation keep it from blocking or stopping wide tracking.

### Controlled baseline-versus-focus evaluation

`scripts/evaluate_long_range.py` collects bounded, metadata-only traces for the
same source in `wide-only` and `focus-enabled` configurations, then validates
and compares them against human-authored source-coordinate annotations. The
report separates all matching detector proposals from authoritative full-frame
detections carrying ByteTrack IDs. It reports recall by person/vehicle category
(and detailed class), daylight versus low-light, and the fixed `<=24 px`, `25-48 px`,
`49-96 px`, and `>96 px` target
height bands.

The same report includes unmatched detections per minute for exhaustively
labelled intervals, target acquisition, longest continuous track, ID switches,
median/low-percentile/wall FPS, event-window precision and recall, event latency,
and focus-minus-baseline deltas. One-way decoded-frame SHA-256 fingerprints enforce
input parity while generated traces omit frame pixels, camera URLs, credentials, plate
text, and free-form alert details. See
[`evaluation/README.md`](evaluation/README.md) and its annotation template.
These measurements describe only the labelled source and recorded settings;
they do not establish field range, weather robustness, or universal accuracy.
The repository currently publishes no representative-footage result because no
consented labelled 1080p/4K evaluation set has been supplied here.

### Pose-based low-posture review

The optional `PostureAnalysisService` owns a one-slot worker queue and a
separately supplied, checksum-verified YOLO pose checkpoint. Only sufficiently
large tracked person crops are submitted. A conservative combination of box
aspect ratio, visible-keypoint vertical spread, and sustained duration produces
a `LOW_POSTURE` signal for contextual scoring. Bending, sitting, falls,
foreshortening, and occlusion can resemble crawling; no standalone response is
authorized.

Missing weights, missing dependencies, queue pressure, or inference exceptions
publish an unavailable/error status and never stop capture, primary detection,
alerts, recording, or streaming. MPS is selected on supported Apple Silicon,
then CUDA, with clean CPU fallback.

### Explainable contextual risk scoring

The optional `RiskScorer` combines available per-track signals: fence contact
and proximity, loitering, motion signature, sustained low posture, night mode,
heading toward the fence, and coordinated-group membership. Every weight and
the alert threshold are environment-configurable.
The bounded weighted sum has no trained parameters because no labelled risk
dataset is available.

A low individual signal does not automatically produce contextual escalation.
Only a threshold crossing emits `CONTEXTUAL_RISK`, once per elevated episode,
and its CSV/dashboard details list the contributing weighted signals. Existing
individual event types remain for backward compatibility and operational
clarity. Contextual risk is a transparent triage aid for human review—not a
probability, intent inference, threat determination, or automated-action input.

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
| `GROUP_APPROACH` | Optional compact person group moves coherently toward the fence |
| `CONTEXTUAL_RISK` | Optional weighted signal combination crosses its review threshold |
| `NIGHT_MOVEMENT` | A tracked person or vehicle is detected during low-light mode |
| `ANPR_READ` | Optional ANPR obtains a repeated, quality-weighted, region-valid read for one vehicle track |

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
| `CONTEXTUAL_RISK` | Three-step elevated review tone | No |
| `GROUP_APPROACH` | Four-step coordinated-approach tone | No |
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

The other accepted test names are `group`, `risk`, `night`, and `anpr`.

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

### Optional regional ANPR

ANPR is disabled by default and has no effect on the other analytics unless it
is explicitly enabled. The India-first workflow then:

1. uses existing vehicle track IDs to stagger expensive OCR attempts;
2. crops each scheduled tracked vehicle;
3. localizes up to a bounded number of plate candidates with a separately
   supplied, SHA-256-verified YOLO checkpoint;
4. rejects plate crops below the configured minimum pixel width;
5. rejects crops below a variance-of-Laplacian sharpness gate, optionally
   rectifies a confidently found four-corner contour, pads/upscales the crop,
   and tries source-color and CLAHE grayscale variants;
6. orders EasyOCR tokens geometrically and combines token confidence by text
   length;
7. normalizes Unicode without deleting Bengali, Devanagari, Burmese, or Chinese
   letters, and maps Unicode decimal digits to ASCII digits;
8. rejects results that do not match a configured regional structure/script;
9. confidence-weights repeated reads within a bounded per-track time window; and
10. abstains until one exact normalized text has enough reads and vote share.

Only a newly stable result emits `ANPR_READ`; unstable one-frame guesses are not
shown as accepted plate text or written as ANPR alerts. The stable state is
short-lived, expires with its vehicle track, and resets on source reconnect,
loop, seek, or backwards source time. Plate/OCR exceptions are caught and logged
without stopping person/vehicle analytics, alerts, recording, alarms, or the
dashboard.

The implemented validator profiles are `IN`, `BD`, `NP`, `PK`, `BT`, `LK`,
`MM`, `CN`, `AF`, `MV`, and permissive fallback `GENERIC`. These are routing and
abstention guards—not proof of country, authenticity, or correctness. `IN` is
the only default profile. Bangladesh requires Bengali OCR; Nepal may require
Devanagari; Myanmar requires Burmese; China requires Chinese; and Afghanistan
may require Arabic-script OCR. Configure corresponding
EasyOCR languages only after testing their compatibility. No single detector or
OCR checkpoint in this project has proven universal accuracy across those
countries, scripts, plate layouts, and camera conditions.

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
       ├────────────┴───────┬───────┴────────────────┤
       ▼                    ▼                        ▼
 group/motion signals   contextual scorer    async posture signals
       │                    │                        │
       └────────────────────┴────────────┬───────────┘
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
| OpenCV | Capture, preprocessing, drawing, JPEG/MP4 encoding, and video writing |
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
| ANPR | External regional YOLO checkpoint + OpenCV preprocessing + EasyOCR + tracked consensus | Disabled by default; separate dependency lock and weights |
| Tiled detection | Separate Ultralytics predictor over overlapping tiles | Disabled; synchronous and FPS-expensive |
| Native digital focus | Reviewed object model on source crops in a one-slot worker | Disabled; source evidence only, no synthetic track IDs |
| Controlled evaluation | JSON annotations/traces plus deterministic Python scoring | No extra model or dependency |
| Posture review | Operator-supplied Ultralytics pose checkpoint in a bounded worker | Disabled; uses core runtime plus external weights |
| Context signals | Classical centroid geometry plus transparent weighted scoring | Disabled; unvalidated scene heuristics |
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
| `python scripts/diagnose_anpr.py` | Explain ANPR acceptance/abstention | No | Stops at frame bound/EOF | Optional explicit diagnostic output |
| `python scripts/evaluate_long_range.py` | Collect/compare labelled wide and focus traces | No | Bounded by evaluation frame limit | Metadata-only JSON/Markdown |

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

### Optional regional ANPR

Install the isolated OCR dependency set first:

```bash
python -m pip install -r requirements-anpr.lock
```

The application runtime never downloads plate-detector weights. The reviewed Indian
checkpoint is an unapproved research candidate with unresolved training-data,
checkpoint-licence, benchmark, and Ultralytics-licensing questions. Inspect
`models/ANPR_MODEL_SOURCES.json` and `THIRD_PARTY_NOTICES.md`. Running the
acquisition script without acceptance prints the risk notice and makes no
network request:

```bash
python scripts/download_anpr_model.py
```

That terms-only invocation intentionally makes no network request and exits with
status 2. After independent review, run these comment-free commands in the same
terminal (interactive zsh does not always treat pasted `#` lines as comments):

```bash
python scripts/download_anpr_model.py --accept-research-model-risks
python scripts/doctor.py --anpr
export IBVAP_ENABLE_ANPR=true
```

The downloader pins the source commit and accepts the file only if both its
6,230,819-byte size and SHA-256 match. It installs to the ignored path
`models/indian_plate_yolov8n.pt`; do not commit it. For a separately reviewed
checkpoint, configure its local path and mandatory expected digest instead.

EasyOCR may acquire separately governed recognition weights on first use. That
behavior is announced in the server log but is controlled by EasyOCR, not this
application. Pre-provision and govern its cache when runtime network access is
not acceptable. OCR runs in CPU mode in this prototype.

For a complete local-file setup, activate the same environment and run all of
these in one terminal before starting `server.py`:

```bash
source .venv/bin/activate
export IBVAP_VIDEO_SOURCE="$PWD/demo1.mp4"
export IBVAP_ENABLE_ANPR=true
export IBVAP_ANPR_REGIONS=IN
export IBVAP_ANPR_OCR_LANGUAGES=en
python scripts/doctor.py --anpr
python scripts/diagnose_anpr.py --source "$PWD/demo1.mp4" --max-frames 300 --output output/anpr_diagnostic.mp4
python server.py
```

The bounded diagnostic prints privacy-safe counters and a recommended next step.
The dashboard also shows `last result · valid observations/attempts`; hover it
for candidate/rejection counters. To inspect normalized OCR text during a
controlled test, set `IBVAP_ANPR_DEBUG=true` and restart. Debug logs can contain
number plates, so turn this back off after diagnosis.

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
| `IBVAP_YOLO_TRACKER` | `bytetrack.yaml` | Explicit authoritative tracker; only ByteTrack is currently accepted |
| `IBVAP_YOLO_INPUT_SIZE` | `640` | Full-frame detector side length, divisible by 32, maximum 4096 |
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
| `IBVAP_LOITERING_ZONE` | `anywhere` | Whole frame or polygon `x,y;x,y;x,y` |

### Context-aware detection

| Variable | Default | Meaning |
|---|---:|---|
| `IBVAP_ENABLE_GROUP_APPROACH` | `false` | Enable classical coordinated-group signal/alert |
| `IBVAP_GROUP_MIN_SIZE` | `3` | Minimum people in a candidate group |
| `IBVAP_GROUP_PROXIMITY_PX` | `160` | Image-plane clustering distance |
| `IBVAP_GROUP_APPROACH_WINDOW_SECONDS` | `3` | Heading/velocity history window |
| `IBVAP_GROUP_MIN_SPEED_PX_PER_SECOND` | `5` | Reject stationary group members |
| `IBVAP_GROUP_HEADING_SIMILARITY` | `0.75` | Minimum mean heading coherence, range 0–1 |
| `IBVAP_GROUP_MIN_FENCE_PROGRESS_PX` | `12` | Required reduction in fence distance |
| `IBVAP_ENABLE_MOTION_SIGNATURE` | `false` | Enable unvalidated trajectory heuristic |
| `IBVAP_MOTION_SIGNATURE_WINDOW_SECONDS` | `3` | Trajectory-analysis window |
| `IBVAP_MOTION_SIGNATURE_MIN_SAMPLES` | `8` | Samples needed for full signal confidence |
| `IBVAP_ENABLE_CONTEXTUAL_RISK` | `false` | Enable weighted contextual review alerts |
| `IBVAP_RISK_ALERT_THRESHOLD` | `0.65` | Bounded score threshold, range 0–1 |
| `IBVAP_RISK_FENCE_PROXIMITY_PX` | `180` | Distance at which fence proximity begins contributing |
| `IBVAP_RISK_WEIGHT_*` | varies | Individual transparent signal weights; see `.env.example` |

Motion path/net-displacement, oscillation scales, category scores, and every
risk weight are also configurable in `.env.example`; they are scene-tuning
parameters rather than validated universal values.

### Tiled inference

| Variable | Default | Meaning |
|---|---:|---|
| `IBVAP_ENABLE_TILED_INFERENCE` | `false` | Enable cadence-limited supplemental tile inference |
| `IBVAP_TILE_SIZE` | `640` | Square tile side in source pixels |
| `IBVAP_TILE_OVERLAP` | `0.20` | Fractional overlap, range 0–0.9 |
| `IBVAP_TILED_INFERENCE_INTERVAL_FRAMES` | `30` | Expensive tiled-inference cadence |
| `IBVAP_TILED_INFERENCE_NMS_IOU` | `0.50` | Cross-tile class-aware NMS threshold |

### Native digital focus

| Variable | Default | Meaning |
|---|---:|---|
| `IBVAP_ENABLE_NATIVE_FOCUS` | `false` | Enable the isolated source-crop confirmation bridge |
| `IBVAP_NATIVE_FOCUS_INTERVAL_FRAMES` | `5` | Candidate submission cadence |
| `IBVAP_NATIVE_FOCUS_MODEL_INPUT_SIZE` | `640` | ROI model input side, divisible by 32 |
| `IBVAP_NATIVE_FOCUS_TRACK_INPUT_SIZE` | `1280` | Temporary authoritative full-frame input side |
| `IBVAP_NATIVE_FOCUS_MAX_REGIONS` | `2` | Maximum highest-confidence proposals per worker job |
| `IBVAP_NATIVE_FOCUS_PADDING_RATIO` | `0.5` | Context padding around each source candidate |
| `IBVAP_NATIVE_FOCUS_CONFIRMATION_FRAMES` | `2` | Spatial observations required before tracker request |
| `IBVAP_NATIVE_FOCUS_TRACK_BOOST_FRAMES` | `30` | Maximum larger-input full-frame tracking frames per request |
| `IBVAP_NATIVE_FOCUS_DEVICE` | `auto` | `auto`, `cpu`, `mps`, `cuda`, or `cuda:0` |

Confidence, crop-size, IoU, result-age, JPEG, queue-poll, and shutdown controls
are validated settings listed in `.env.example`.

### Controlled evaluation

| Variable | Default | Meaning |
|---|---:|---|
| `IBVAP_EVALUATION_IOU_THRESHOLD` | `0.50` | Ground-truth/prediction match threshold |
| `IBVAP_EVALUATION_LOW_FPS_PERCENTILE` | `10` | Low processing-FPS percentile reported with median FPS |
| `IBVAP_EVALUATION_MAX_FRAMES` | `10000` | Hard collection and input frame bound |
| `IBVAP_EVALUATION_RUNTIME_WARMUP_FRAMES` | `5` | Initial processing samples excluded from FPS percentiles |
| `IBVAP_EVALUATION_MAX_OBJECTS_PER_FRAME` | `1000` | Annotation/trace object-count bound |
| `IBVAP_EVALUATION_MAX_ALERTS` | `100000` | Trace alert-count bound |
| `IBVAP_EVALUATION_MAX_JSON_BYTES` | `104857600` | Input and generated JSON/report size bound |
| `IBVAP_EVALUATION_HARDWARE_LABEL` | `unrecorded` | Non-sensitive hardware description required for a publishable run |
| `IBVAP_EVALUATION_BUILD_LABEL` | `unrecorded` | Commit/release identifier required for a publishable run |

The annotation schema, controlled two-run commands, event windows, metric
interpretation, and safe reporting rules are in [`evaluation/README.md`](evaluation/README.md).

### Low-posture review

| Variable | Default | Meaning |
|---|---:|---|
| `IBVAP_ENABLE_POSTURE_ANALYSIS` | `false` | Enable isolated pose worker |
| `IBVAP_POSTURE_MODEL_PATH` | `models/yolov8n-pose.pt` | Operator-supplied pose checkpoint |
| `IBVAP_POSTURE_MODEL_SHA256` | empty | Mandatory reviewed digest when enabled |
| `IBVAP_POSTURE_CONFIDENCE_THRESHOLD` | `0.60` | Pose confidence gate |
| `IBVAP_POSTURE_INTERVAL_FRAMES` | `15` | Pose submission cadence |
| `IBVAP_POSTURE_MAX_PEOPLE` | `4` | Maximum person crops per job |
| `IBVAP_POSTURE_MIN_PERSON_SIZE_PX` | `96` | Minimum box dimension for meaningful keypoints |
| `IBVAP_POSTURE_SUSTAIN_SECONDS` | `1.5` | Required continuous low-pose duration |
| `IBVAP_POSTURE_DEVICE` | `auto` | `auto`, `cpu`, `mps`, or `cuda` |

Additional posture geometry and state-TTL controls are listed in
`.env.example`. Missing optional models produce an unavailable status rather
than failing the primary pipeline.

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
| `IBVAP_ENABLE_ANPR` | `false` | Enable optional tracked regional ANPR |
| `IBVAP_ANPR_DEBUG` | `false` | Log detailed rejection reasons and OCR text; sensitive, diagnostic use only |
| `IBVAP_ANPR_PLATE_MODEL_PATH` | `models/indian_plate_yolov8n.pt` | External YOLO plate-detector checkpoint |
| `IBVAP_ANPR_PLATE_MODEL_SHA256` | Reviewed candidate digest | Mandatory expected checkpoint SHA-256; set explicitly for a custom path |
| `IBVAP_ANPR_PLATE_DETECTION_CONFIDENCE` | `0.25` | Plate-localization confidence, range 0–1 |
| `IBVAP_ANPR_PLATE_INPUT_SIZE` | `640` | Plate YOLO input side, divisible by 32 |
| `IBVAP_ANPR_MAX_PLATE_CANDIDATES` | `3` | Maximum plate boxes OCR'd per scheduled vehicle |
| `IBVAP_ANPR_INTERVAL_FRAMES` | `10` | Track-staggered OCR cadence |
| `IBVAP_ANPR_MIN_PLATE_WIDTH_PX` | `80` | Reject smaller detected plate crops |
| `IBVAP_ANPR_MIN_PLATE_SHARPNESS` | `20` | Minimum variance-of-Laplacian crop sharpness |
| `IBVAP_ANPR_OCR_TARGET_HEIGHT_PX` | `64` | Upscale short plate crops to this height |
| `IBVAP_ANPR_OCR_LANGUAGES` | `en` | Comma-separated EasyOCR language codes |
| `IBVAP_ANPR_REGIONS` | `IN` | Ordered profiles: `IN,BD,NP,PK,BT,LK,MM,CN,AF,MV,GENERIC` |
| `IBVAP_ANPR_MIN_OCR_CONFIDENCE` | `0.50` | Minimum accepted EasyOCR confidence |
| `IBVAP_ANPR_MIN_CONSENSUS_READS` | `3` | Matching reads required before stability |
| `IBVAP_ANPR_CONSENSUS_WINDOW_SECONDS` | `3` | Source-time voting window |
| `IBVAP_ANPR_MIN_CONSENSUS_WEIGHT_RATIO` | `0.60` | Winner's minimum quality-weighted vote share |
| `IBVAP_ANPR_MAX_OBSERVATIONS_PER_TRACK` | `12` | Per-track memory bound |
| `IBVAP_ANPR_STATE_TTL_SECONDS` | `5` | Idle ANPR track-state expiry |

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

### Enabling context features for a controlled demo

All new context features are disabled unless explicitly selected. The
model-free signals can be enabled together after calibrating the fence and
reviewing the defaults against labelled footage:

```bash
export IBVAP_ENABLE_GROUP_APPROACH=true
export IBVAP_ENABLE_MOTION_SIGNATURE=true
export IBVAP_ENABLE_CONTEXTUAL_RISK=true
```

Tiling is independent and expensive:

```bash
export IBVAP_ENABLE_TILED_INFERENCE=true
```

Posture review requires operator-supplied weights and an exact hash; there is
intentionally no automatic downloader:

```bash
export IBVAP_ENABLE_POSTURE_ANALYSIS=true
export IBVAP_POSTURE_MODEL_PATH=/reviewed/models/pose.pt
export IBVAP_POSTURE_MODEL_SHA256="REPLACE_WITH_EXACT_64_CHARACTER_HEX_DIGEST"
```

Replace the digest placeholder before starting the process. If the model is missing,
unverified, or incompatible, its worker reports `unavailable`/`error` while the
primary detector and dashboard continue. Firearm detection is not implemented.

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
| `GET /api/focus` | Required | Native-focus status, source provenance, and acquisition metadata |
| `GET /api/focus/image` | Required | Versioned untouched source-crop JPEG |
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

Generated logs, output video, downloaded face/ANPR weights, caches, bytecode,
and local secrets are excluded from Git. The core `yolov8n.pt` remains tracked
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

### ANPR does not start or does not emit a plate

If `scripts/download_anpr_model.py` is missing or `doctor.py` does not recognize
`--anpr`, the checkout predates the regional ANPR implementation. Update to a
revision containing the feature before retrying; installing EasyOCR alone does
not add repository scripts or options.

Run `python scripts/doctor.py --anpr`. Startup intentionally fails when ANPR is
enabled but the detector checkpoint is absent, has no configured SHA-256, or
fails checksum verification. Then run the bounded diagnostic against the same
source:

```bash
python scripts/diagnose_anpr.py --source "$PWD/demo1.mp4" --max-frames 300 --output output/anpr_diagnostic.mp4
```

Interpret `last_result` as follows:

| Result | Meaning |
|---|---|
| `waiting` with zero attempts | No tracked supported vehicle has reached ANPR yet |
| `no_plate_candidate` | The optional plate YOLO found no box in scheduled vehicle crops |
| `plate_too_small` / `plate_too_blurry` | A quality gate rejected the crop |
| `ocr_no_text` / `ocr_low_confidence` | EasyOCR failed or remained below threshold |
| `region_rejected` | Text did not match any configured regional profile |
| `valid_observation` but zero stable reads | OCR varied and did not reach temporal consensus |

If attempts remain zero, isolate plate localization from vehicle tracking with:

```bash
python scripts/diagnose_anpr.py --source "$PWD/demo1.mp4" --max-frames 300 --full-frame --output output/anpr_full_frame_diagnostic.mp4
```

`--full-frame` is diagnostic only; normal alerts remain track-gated. The
diagnostic recommends one temporary tuning change for the observed failure.
Test on labelled footage and tune one threshold at a time; do not bypass model
checksum verification, and restore conservative validation/consensus settings
after debugging.

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

The current suite contains 148 unit/integration tests covering geometry, fence
contact, loitering continuity, source-time behavior, alert cooldown/rotation,
bounded broadcast, authentication, sessions, host/origin controls, rate limits,
source parsing, server recording failure isolation, audible-alarm patterns,
face quality gates, regional plate normalization/validation, OCR token ordering,
weighted track consensus, ANPR pipeline gating/reset, wide-area loitering,
group-approach episodes, motion-signature score direction, tile remapping/NMS,
posture persistence, contextual-risk crossings, checksum enforcement, and
licence-gated model acquisition. The configured 85% coverage threshold and
reported coverage apply to `utils` and `security`; they are not a claim of
whole-system or ML-model coverage.

CI runs on Python 3.11, repeats formatting/lint/test/security/audit checks,
dry-resolves core, ANPR, and face locks for Apple Silicon macOS 14, and separately
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
- checksum verification for bundled YOLO and configured face/ANPR models;
- explicit risk/licence acknowledgement before reviewed face or ANPR acquisition;
- ignored generated data and model weights; and
- a non-root, no-new-privileges default container.

These controls do not replace network segmentation, OS hardening, encrypted
storage, access review, centralized monitoring, incident response, or an
independent penetration test.

## Limitations and non-goals

### Accuracy and analytics

- YOLO, ByteTrack, SCRFD, OCR, regional validators, consensus thresholds, and
  brightness thresholds can all produce false positives and false negatives.
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
- Motion signatures cannot provide reliable 100 m+ human/animal/vegetation
  discrimination without appropriate optics, thermal input, labelled data, and
  validated sensor fusion; the implemented score is only a best-effort aid.
- Group approach is image-plane clustering and heading correlation, not intent
  recognition or proof of coordination.
- Tiled and native-focus inference may improve small-object recall but can
  substantially reduce FPS, increase false positives, and have no real-time
  guarantee on underpowered hardware.
- Digital focus cannot recover detail absent from the source sensor. ROI
  inference can preserve source pixels discarded by whole-frame resizing, but
  confirmation is not identity and contextual analytics still require ByteTrack.
- Evaluation metrics are valid only for the exact labelled source, commit,
  settings, and hardware in the trace. Event scoring uses labelled clip windows,
  not identity association; sparse labels cannot support false detections/minute.
- Low-posture geometry is not reliable for tiny, occluded, or distant people and
  cannot distinguish crawling from every fall, bend, or seated posture.
- Firearm detection, related alerts, dashboard metadata, and alarm playback are
  not implemented.
- Contextual risk is a tunable weighted heuristic, not a calibrated probability,
  intent classifier, or authorization for automated action.
- No weather robustness for fog, dust, snow, or severe rain is claimed; those
  conditions require appropriate cameras, sensors, enclosures, and validation.
- Local VLM scene captioning is not implemented in this pass. No cloud API is
  called, and no caption can independently trigger an alert.

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

- The reviewed Indian detector is a research candidate, not a production-
  approved model; its training-data provenance and held-out accuracy are
  unresolved.
- The candidate localizes plates only. EasyOCR performs recognition in CPU mode
  and may be inaccurate or slow, especially for small, blurred, angled,
  two-line, stylized, or non-Latin plates.
- Structural/script profiles reject some implausible strings but do not verify
  a registration, determine nationality, correct OCR, or guarantee authenticity.
- Exact-text temporal consensus suppresses unstable guesses but can still settle
  on the same repeated wrong text. Track-ID switches can split or misassociate
  votes.
- Support profiles do not imply validated model support. India plus all
  neighboring-country scripts require separately benchmarked detector/OCR
  combinations; none is supplied or claimed here.
- ANPR requires external detector weights, separate dependencies, and EasyOCR's
  separately governed recognition weights. Runtime footage and plate strings
  remain sensitive personal/operational data.

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
- The reviewed Indian plate checkpoint remains research/evaluation-only here;
  its source-repository licence does not resolve checkpoint, training-data, or
  Ultralytics commercial-use questions.
- EasyOCR and any downloaded OCR weights retain their own terms.

Do not commit credentials, camera URLs, operational footage, faces, number
plates, alerts, audit logs, generated video, downloaded weights, or other
sensitive/generated artifacts.
