# IBVAP Prototype
### AI-Based Intelligent Video Analytics Platform for Border Surveillance
*(runs on ordinary CCTV/video — no proprietary FRS/ANPR hardware needed)*

A working single-camera prototype that combines, on one video stream:

| Feature | Status |
|---|---|
| Human detection & tracking | ✅ implemented |
| Vehicle detection & classification (car/motorcycle/bus/truck) | ✅ implemented |
| Face detection | ✅ implemented |
| Virtual fence intrusion detection | ✅ implemented |
| Suspicious activity (loitering) detection | ✅ implemented |
| Night-time movement detection | ✅ implemented |
| Real-time alert generation + event logging (CSV) | ✅ implemented |
| ANPR (license plate OCR) | 🔶 stub/extension point — see below |

---

## 1. Setup

```bash
cd ibvap_prototype
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run of `main.py` will auto-download the YOLOv8-nano weights
(`yolov8n.pt`, ~6MB) via the `ultralytics` package — this needs internet
**once**; after that it runs fully offline.

## 2. Get a demo video

You need *some* footage of people/vehicles moving through a scene — this
doesn't have to be real border footage. Good free options for a demo:
- Any phone recording of a street/gate/parking lot from a fixed angle
- Public sample surveillance clips (search "CCTV sample footage" /
  "pedestrian tracking dataset video")
- Your own webcam, live (set `VIDEO_SOURCE = 0` in `config.py`)

Put the file at `sample_data/demo.mp4`, or point `config.VIDEO_SOURCE` at
wherever your file lives.

## 3. Run it

```bash
python3 main.py
```

A window pops up showing:
- Green boxes = people, orange boxes = vehicles, each with a persistent ID
- Yellow boxes = detected faces
- A red virtual fence line
- A status bar (day/night mode, frame count, fps)
- A red alert banner at the bottom when something fires

Press `q` to stop early. On exit you get:
- `output/annotated_output.mp4` — the full annotated recording
- `logs/alerts.csv` — every alert with timestamp, type, track ID, detail

## 4. Run the web dashboard instead

For presenting to judges, the browser dashboard is the better option than
the OpenCV window — it looks like an actual product, and multiple people
can watch it on a projector or their own laptops on the same network.

```bash
python3 server.py
```

Then open **http://localhost:8000**. You'll see:
- The live annotated feed, streamed to the browser
- A scrolling alert log that updates in real time via websocket
- Live stats: people/vehicles currently tracked, total alerts, uptime, fps
- A day/night mode indicator that switches automatically

It uses the exact same `pipeline.py` analytics as `main.py` — same
detection, same fence/loitering/night logic, same alert log CSV. The
dashboard is just a different "front end" for the same brain, which is
itself a good talking point: the analytics engine is decoupled from
the display, so it could equally feed a command-center UI later.

To let other people on the same network/wifi view it (e.g. on their
phones during the demo), find your machine's local IP (`ipconfig getifaddr en0`
on macOS) and share `http://<that-ip>:8000` instead of localhost.

## 5. Tune it for your footage

Open `config.py` — every threshold lives there:
- `VIRTUAL_FENCE_LINE` — **you must adjust this** to match your video's
  resolution/geometry. Run once, note where you want the line, update
  the coordinates, run again.
- `LOITERING_SECONDS` / `LOITERING_MOVEMENT_THRESHOLD_PX` — how "stuck"
  someone must be before it's flagged
- `NIGHT_BRIGHTNESS_THRESHOLD` — average pixel brightness cutoff
- `CONFIDENCE_THRESHOLD` — detection sensitivity

---

## Architecture (for your pitch deck)

```
IP Camera / video file
        │  (RTSP or file, standard OpenCV VideoCapture)
        ▼
  Detector (YOLOv8 + ByteTrack)  ──► persistent track IDs for people & vehicles
        │
        ├──► Face detector (Haar cascade, cropped to person boxes)
        ├──► Virtual fence (line-crossing test on track centroids)
        ├──► Loitering detector (centroid drift over a time window)
        ├──► Night detector (mean frame brightness)
        └──► [extension] ANPR (plate detector + OCR on vehicle crops)
        │
        ▼
   Alert Logger ──► CSV log (+ console) ──► [extension] push to dashboard/command-center
        │
        ▼
   Annotated video output (for review / evidence)
```

This same pipeline is what you'd deploy **at the edge**, next to (or at)
each Border Out Post — it doesn't need internet to run; only alerts
(tiny payloads) need to go upstream to a command center, not raw video.
That directly addresses the "remote border area, limited connectivity"
constraint in the problem statement.

## Why this design choice-by-choice (talking points for judges)

- **YOLOv8 + built-in tracker, not separate detection+tracking models**:
  one model, persistent IDs "for free," fast enough for CPU inference —
  important since edge boxes at remote posts won't all have GPUs.
- **Face detection only inside person boxes**, not the whole frame:
  far cheaper, and avoids false positives from background clutter.
  (This is the "detection" half of FRS — matching detected faces against
  a watchlist database is the natural next module, kept separate since
  it needs a face-embedding + database layer.)
- **Virtual fence as a line-crossing test on track centroids**, not
  per-frame zone checks: correctly catches fast crossings between frames
  and avoids re-alerting every frame someone stands still inside a zone.
- **Cooldown in the alert logger**: without it, a single loitering person
  would generate hundreds of alerts per minute. Real systems need this.
- **Everything reads from `config.py`**: shows the judges this isn't a
  one-off script — it's parameterized for different cameras/sites,
  which is the "scalable across many BOPs" requirement.
- **Dashboard is a thin layer over the same pipeline, not a separate
  implementation**: `main.py` and `server.py` both drive `pipeline.py`.
  This means the analytics logic is tested once and used everywhere —
  and it's an honest architecture point: a real deployment would want
  exactly this separation between "the AI engine" and "whatever screen
  displays it" (a projector, a command-center wall, a phone app).

## Extension roadmap (mention these as "next steps" in your pitch)

1. **ANPR**: wire `detector.detect_plate()` to a plate-detector model +
   EasyOCR/PaddleOCR on vehicle crops.
2. **FRS (watchlist matching)**: add face embeddings (e.g. `face_recognition`
   or `insightface`) and compare against a stored watchlist database —
   detection is already in place, this adds the recognition layer.
3. **Multi-camera**: run one `Detector` + pipeline instance per RTSP
   stream (multiprocessing), aggregate alerts centrally.
4. **Dashboard**: a simple web UI (FastAPI + websockets) tailing
   `logs/alerts.csv` and streaming the annotated frames — turns this from
   a script into the "platform" the problem statement asks for.
5. **Command & control integration**: alerts.csv → REST webhook / MQTT
   publish, so this plugs into whatever C2 system border forces already use.
