"""
IBVAP Web Dashboard
--------------------
Wraps the same VideoPipeline used by main.py in a small FastAPI server:
  - GET  /        the dashboard page
  - GET  /stream  MJPEG live annotated video feed (works in a plain <img> tag)
  - WS   /ws      pushes each new alert + periodic stats to the browser

The pipeline runs in a background thread (video capture + YOLO inference
are blocking calls) and hands frames/alerts/stats to the async web layer
through simple thread-safe shared state. No analytics logic lives here -
that's all in pipeline.py, unchanged from the CLI demo.

Run:
    python3 server.py
Then open:
    http://localhost:8000
"""

import asyncio
import queue
import threading
import time

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
from pipeline import VideoPipeline

app = FastAPI()
app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")

# --- shared state between the background capture thread and the async web layer ---
_frame_lock = threading.Lock()
_latest_jpeg = None

_stats_lock = threading.Lock()
_latest_stats = {}

_alert_queue: "queue.Queue" = queue.Queue()


def _on_alert(alert: dict):
    _alert_queue.put(alert)


def _capture_loop():
    global _latest_jpeg

    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {config.VIDEO_SOURCE}")
        return

    print(f"[INIT] Dashboard capture loop started on source: {config.VIDEO_SOURCE}")
    pipeline = VideoPipeline(on_alert=_on_alert)

    is_file_source = isinstance(config.VIDEO_SOURCE, str) and not config.VIDEO_SOURCE.startswith("rtsp")

    while True:
        ok, frame = cap.read()
        if not ok:
            if is_file_source:
                # loop the demo video so the dashboard keeps running continuously
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("[INFO] Stream ended / unreadable frame. Stopping capture loop.")
            break

        annotated = pipeline.process_frame(frame)

        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with _frame_lock:
                _latest_jpeg = buf.tobytes()

        with _stats_lock:
            _latest_stats.update(pipeline.stats)

        time.sleep(0.001)  # yield briefly; keeps thread responsive without capping fps meaningfully

    pipeline.close()


@app.on_event("startup")
def _on_startup():
    thread = threading.Thread(target=_capture_loop, daemon=True)
    thread.start()


def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        with _frame_lock:
            frame = _latest_jpeg
        if frame is not None:
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.03)  # ~30fps cap on the stream to browsers


@app.get("/stream")
def stream():
    return StreamingResponse(
        _mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # drain any alerts that have arrived since the last tick
            while True:
                try:
                    alert = _alert_queue.get_nowait()
                except queue.Empty:
                    break
                await websocket.send_json({"type": "alert", "data": alert})

            with _stats_lock:
                stats_copy = dict(_latest_stats)
            await websocket.send_json({"type": "stats", "data": stats_copy})

            await asyncio.sleep(0.4)
    except WebSocketDisconnect:
        pass


@app.get("/", response_class=HTMLResponse)
def index():
    with open("dashboard/static/index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
