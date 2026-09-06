"""Authenticated FastAPI dashboard and resilient background capture service."""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

import config
from security import (
    AuditLogger,
    AuthManager,
    SlidingWindowRateLimiter,
    is_allowed_host,
    is_allowed_origin,
)
from utils.broker import AlertBroker, ConnectionLimiter
from utils.sources import (
    describe_source,
    is_file_source,
    is_network_source,
    normalize_fps,
    recorded_source_time,
)

LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "dashboard" / "static"


class CaptureService:
    """Own capture, inference, current-frame state, and source reconnection."""

    def __init__(self, broker: AlertBroker):
        self.broker = broker
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_frame_monotonic: float | None = None
        self._stats_lock = threading.Lock()
        self._stats: dict = {}
        self._faces_lock = threading.Lock()
        self._face_version = 0
        self._face_images: tuple[bytes, ...] = ()
        self._face_image_versions: dict[int, tuple[bytes, ...]] = {0: ()}
        self._face_metadata: tuple[dict, ...] = ()
        self._face_status = "loading" if config.ENABLE_FACE_ENHANCEMENT else "disabled"
        self._face_message: str | None = None
        self._face_frame_number: int | None = None
        self._face_updated_monotonic: float | None = None
        self._status_lock = threading.Lock()
        self._status = "stopped"
        self._last_error: str | None = None
        self._reconnects = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="ibvap-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=15)
        if self._thread and self._thread.is_alive():
            LOGGER.error("capture worker did not stop before the shutdown timeout")
            self._set_status("shutdown_timeout", "capture worker is still blocked")
        else:
            self._set_status("stopped")

    @staticmethod
    def _open_capture(cv2):
        if is_network_source(config.VIDEO_SOURCE):
            return cv2.VideoCapture(
                config.VIDEO_SOURCE,
                cv2.CAP_ANY,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    config.CAMERA_OPEN_TIMEOUT_MILLISECONDS,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    config.CAMERA_READ_TIMEOUT_MILLISECONDS,
                ],
            )
        return cv2.VideoCapture(config.VIDEO_SOURCE)

    def _run(self) -> None:
        pipeline = None
        try:
            # Heavy CV dependencies are loaded inside the worker so importing
            # the ASGI app for tooling does not initialize a model or camera.
            import cv2

            from pipeline import VideoPipeline

            self._set_status("initializing")
            pipeline = VideoPipeline(
                on_alert=self.broker.publish,
                on_face_snapshot=self._update_face_snapshot,
            )
            while not self._stop_event.is_set():
                capture = self._open_capture(cv2)
                if not capture.isOpened():
                    capture.release()
                    self._clear_frame()
                    self._set_status("reconnecting", "could not open video source")
                    if self._stop_event.wait(config.CAMERA_RECONNECT_SECONDS):
                        break
                    self._reconnects += 1
                    continue

                source_is_file = is_file_source(config.VIDEO_SOURCE)
                source_fps = normalize_fps(capture.get(cv2.CAP_PROP_FPS))
                frame_period = 1.0 / source_fps
                source_frame_number = 0
                frames_since_open = 0
                consecutive_errors = 0
                self._set_status("running")
                LOGGER.info("capture opened: %s", describe_source(config.VIDEO_SOURCE))

                try:
                    while not self._stop_event.is_set():
                        iteration_started = time.monotonic()
                        ok, frame = capture.read()
                        if not ok:
                            if (
                                source_is_file
                                and frames_since_open > 0
                                and capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ):
                                pipeline.reset_tracking()
                                source_frame_number = 0
                                frames_since_open = 0
                                LOGGER.info("recording looped; tracking state reset")
                                continue
                            self._clear_frame()
                            self._set_status("reconnecting", "video stream became unavailable")
                            break

                        source_frame_number += 1
                        frames_since_open += 1
                        source_time = (
                            recorded_source_time(
                                capture.get(cv2.CAP_PROP_POS_MSEC),
                                source_frame_number,
                                source_fps,
                            )
                            if source_is_file
                            else time.monotonic()
                        )
                        try:
                            annotated = pipeline.process_frame(frame, source_time=source_time)
                            encoded, buffer = cv2.imencode(
                                ".jpg",
                                annotated,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                            )
                            if encoded:
                                now = time.monotonic()
                                with self._frame_lock:
                                    self._latest_jpeg = buffer.tobytes()
                                    self._latest_frame_monotonic = now
                            with self._stats_lock:
                                self._stats = dict(pipeline.stats)
                            consecutive_errors = 0
                            self._set_status("running")
                        except Exception as exc:
                            consecutive_errors += 1
                            LOGGER.exception("frame processing failed")
                            self._set_status("degraded", str(exc))
                            if consecutive_errors >= 5:
                                break

                        if source_is_file and config.FILE_PLAYBACK_REALTIME:
                            remaining = frame_period - (time.monotonic() - iteration_started)
                            if remaining > 0:
                                self._stop_event.wait(remaining)
                finally:
                    capture.release()

                if not self._stop_event.is_set():
                    pipeline.reset_tracking()
                    self._reconnects += 1
                    self._stop_event.wait(config.CAMERA_RECONNECT_SECONDS)
        except Exception as exc:
            LOGGER.exception("capture service failed")
            self._clear_frame()
            self._set_status("failed", str(exc))
        finally:
            if pipeline is not None:
                pipeline.close()
            if self._status != "failed":
                self._set_status("stopped")

    def _set_status(self, status: str, error: str | None = None) -> None:
        with self._status_lock:
            self._status = status
            self._last_error = error

    def _clear_frame(self) -> None:
        with self._frame_lock:
            self._latest_jpeg = None
            self._latest_frame_monotonic = None

    def latest_jpeg(self) -> bytes | None:
        with self._frame_lock:
            return self._latest_jpeg

    def stats(self) -> dict:
        with self._stats_lock:
            stats = dict(self._stats)
        stats.update(self.health())
        with self._faces_lock:
            stats["face_enhancement_status"] = self._face_status
            stats["enhanced_face_count"] = len(self._face_images)
        return stats

    def _update_face_snapshot(self, snapshot: dict) -> None:
        faces = snapshot.get("faces", ())
        images = tuple(face["jpeg"] for face in faces if isinstance(face.get("jpeg"), bytes))
        metadata = tuple(
            {
                "track_id": face.get("track_id"),
                "confidence": face.get("confidence"),
            }
            for face in faces
            if isinstance(face.get("jpeg"), bytes)
        )
        with self._faces_lock:
            self._face_version += 1
            self._face_images = images
            self._face_image_versions[self._face_version] = images
            while len(self._face_image_versions) > 3:
                del self._face_image_versions[min(self._face_image_versions)]
            self._face_metadata = metadata
            self._face_status = str(snapshot.get("status", "error"))
            self._face_message = snapshot.get("message")
            self._face_frame_number = snapshot.get("frame_number")
            self._face_updated_monotonic = snapshot.get("updated_monotonic")

    def face_summary(self) -> dict:
        with self._faces_lock:
            updated_age = (
                None
                if self._face_updated_monotonic is None
                else max(0.0, time.monotonic() - self._face_updated_monotonic)
            )
            version = self._face_version
            return {
                "status": self._face_status,
                "message": self._face_message,
                "frame_number": self._face_frame_number,
                "age_seconds": None if updated_age is None else round(updated_age, 2),
                "version": version,
                "faces": [
                    {
                        **metadata,
                        "image_url": f"/api/faces/{index}?version={version}",
                    }
                    for index, metadata in enumerate(self._face_metadata)
                ],
                "disclaimer": "AI-enhanced visualization; not forensic evidence.",
            }

    def face_image(self, index: int, version: int | None = None) -> bytes | None:
        with self._faces_lock:
            images = (
                self._face_images if version is None else self._face_image_versions.get(version, ())
            )
            if index < 0 or index >= len(images):
                return None
            return images[index]

    def health(self) -> dict:
        now = time.monotonic()
        with self._frame_lock:
            frame_age = (
                None
                if self._latest_frame_monotonic is None
                else max(0.0, now - self._latest_frame_monotonic)
            )
        with self._status_lock:
            status = self._status
            has_error = self._last_error is not None
        alive = bool(self._thread and self._thread.is_alive())
        ready = (
            alive
            and status == "running"
            and frame_age is not None
            and frame_age <= config.FRAME_STALE_AFTER_SECONDS
        )
        return {
            "service_status": status,
            "capture_thread_alive": alive,
            "ready": ready,
            "frame_age_seconds": None if frame_age is None else round(frame_age, 2),
            "reconnects": self._reconnects,
            "has_error": has_error,
        }


def _client_ip(connection) -> str | None:
    return connection.client.host if connection.client else None


def _is_https(connection) -> bool:
    return connection.url.scheme in {"https", "wss"}


broker = AlertBroker(
    queue_size=config.ALERT_SUBSCRIBER_QUEUE_SIZE,
    history_size=config.ALERT_HISTORY_SIZE,
)
capture_service = CaptureService(broker)
auth = AuthManager(
    config.DASHBOARD_USERNAME,
    config.DASHBOARD_PASSWORD,
    config.SESSION_SECRET,
    config.SESSION_TTL_SECONDS,
)
audit = AuditLogger(
    config.AUDIT_LOG_JSONL,
    max_bytes=config.AUDIT_LOG_MAX_BYTES,
    backup_count=config.LOG_BACKUP_COUNT,
)
login_limiter = SlidingWindowRateLimiter(
    config.LOGIN_ATTEMPTS_PER_MINUTE,
    max_keys=config.LOGIN_RATE_LIMIT_MAX_CLIENTS,
)
stream_limiter = ConnectionLimiter(config.MAX_STREAM_CLIENTS)
websocket_limiter = ConnectionLimiter(config.MAX_WEBSOCKET_CLIENTS)
websocket_attempt_limiter = SlidingWindowRateLimiter(
    config.WEBSOCKET_ATTEMPTS_PER_MINUTE,
    max_keys=config.LOGIN_RATE_LIMIT_MAX_CLIENTS,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.validate_server_security()
    capture_service.start()
    try:
        yield
    finally:
        capture_service.stop()


app = FastAPI(
    title="IBVAP Dashboard",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    if not is_allowed_host(request.headers.get("host", ""), config.ALLOWED_HOSTS):
        response = JSONResponse({"detail": "host not allowed"}, status_code=400)
    elif (
        config.REQUIRE_HTTPS
        and not _is_https(request)
        and not request.url.path.startswith("/health/")
    ):
        response = JSONResponse({"detail": "HTTPS is required"}, status_code=426)
    else:
        response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data:; style-src 'self'; "
        "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; "
        "base-uri 'none'; form-action 'self'"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Cache-Control"] = "no-store"
    if _is_https(request):
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response


def _operator(request: Request) -> dict | None:
    return auth.verify_session(request.cookies.get(config.SESSION_COOKIE_NAME))


def _http_origin_allowed(request: Request) -> bool:
    origin = request.headers.get("origin")
    if origin and is_allowed_origin(
        origin,
        request.headers.get("host", ""),
        config.ALLOWED_ORIGINS,
    ):
        return True

    # Fetch Metadata headers are browser-controlled and cannot be set by
    # cross-origin JavaScript. They provide a secure fallback for privacy
    # configurations that serialize a local form's Origin as "null" or pass
    # through a development proxy that rewrites Host. A real cross-site form
    # submission is still rejected as "cross-site" (or "same-site").
    fetch_site_header = request.headers.get("sec-fetch-site")
    fetch_site = fetch_site_header.lower() if fetch_site_header else fetch_site_header
    if fetch_site in {"none", "same-origin"}:
        return True

    # Non-browser clients may omit both headers. They do not carry a victim's
    # browser-managed Strict session cookie and therefore cannot perform CSRF.
    return origin is None and fetch_site is None


@app.get("/login")
def login_page(request: Request):
    if _operator(request):
        return RedirectResponse("/", status_code=303)
    return FileResponse(STATIC_DIR / "login.html")


@app.post("/login")
async def login(request: Request):
    client_ip = _client_ip(request) or "unknown"
    if not _http_origin_allowed(request):
        origin_details = {
            "host": request.headers.get("host", "")[:256],
            "origin": request.headers.get("origin", "")[:512],
            "sec_fetch_site": request.headers.get("sec-fetch-site", "")[:64],
        }
        audit.log(
            "login_origin_rejected",
            success=False,
            client_ip=client_ip,
            details=origin_details,
        )
        LOGGER.warning("login origin rejected: %r", origin_details)
        return JSONResponse({"detail": "origin not allowed"}, status_code=403)
    if not login_limiter.allow(client_ip):
        audit.log("login_rate_limited", success=False, client_ip=client_ip)
        return RedirectResponse("/login?error=rate", status_code=303)

    body = await request.body()
    if len(body) > 8192:
        return JSONResponse({"detail": "request too large"}, status_code=413)
    try:
        fields = parse_qs(body.decode("utf-8"), keep_blank_values=True)
    except UnicodeDecodeError:
        fields = {}
    username = fields.get("username", [""])[0]
    password = fields.get("password", [""])[0]
    if not auth.authenticate(username, password):
        audit.log(
            "login",
            success=False,
            client_ip=client_ip,
            actor=username[:128] or None,
        )
        return RedirectResponse("/login?error=credentials", status_code=303)

    token = auth.create_session()
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        config.SESSION_COOKIE_NAME,
        token,
        max_age=config.SESSION_TTL_SECONDS,
        httponly=True,
        secure=config.COOKIE_SECURE or _is_https(request),
        samesite="strict",
        path="/",
    )
    audit.log("login", success=True, client_ip=client_ip, actor=username)
    return response


@app.post("/logout")
def logout(request: Request):
    operator = _operator(request)
    if not _http_origin_allowed(request):
        audit.log(
            "logout_origin_rejected",
            success=False,
            client_ip=_client_ip(request),
            actor=operator.get("sub") if operator else None,
        )
        return JSONResponse({"detail": "origin not allowed"}, status_code=403)
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(config.SESSION_COOKIE_NAME, path="/")
    audit.log(
        "logout",
        success=operator is not None,
        client_ip=_client_ip(request),
        actor=operator.get("sub") if operator else None,
    )
    return response


@app.get("/health/live")
def liveness():
    health = capture_service.health()
    return JSONResponse(
        {"status": "ok" if health["capture_thread_alive"] else "failed"},
        status_code=200 if health["capture_thread_alive"] else 503,
    )


@app.get("/health/ready")
def readiness():
    health = capture_service.health()
    return JSONResponse(
        {
            "status": "ready" if health["ready"] else "not_ready",
            "service_status": health["service_status"],
        },
        status_code=200 if health["ready"] else 503,
    )


@app.get("/")
def index(request: Request):
    if not _operator(request):
        return RedirectResponse("/login", status_code=303)
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/stats")
def api_stats(request: Request):
    if not _operator(request):
        return JSONResponse({"detail": "authentication required"}, status_code=401)
    return capture_service.stats()


@app.get("/api/faces")
def api_faces(request: Request):
    if not _operator(request):
        return JSONResponse({"detail": "authentication required"}, status_code=401)
    return capture_service.face_summary()


@app.get("/api/faces/{index}")
def api_face_image(index: int, request: Request, version: int | None = None):
    if not _operator(request):
        return JSONResponse({"detail": "authentication required"}, status_code=401)
    image = capture_service.face_image(index, version)
    if image is None:
        return JSONResponse({"detail": "enhanced face not found"}, status_code=404)
    return Response(content=image, media_type="image/jpeg")


@app.get("/stream")
def stream(request: Request):
    operator = _operator(request)
    if not operator:
        return JSONResponse({"detail": "authentication required"}, status_code=401)
    if not stream_limiter.acquire():
        return JSONResponse({"detail": "stream capacity reached"}, status_code=503)

    client_ip = _client_ip(request)
    audit.log("stream_connect", success=True, client_ip=client_ip, actor=operator["sub"])

    def generate():
        boundary = b"--frame"
        try:
            while True:
                frame = capture_service.latest_jpeg()
                if frame is not None:
                    yield (
                        boundary
                        + b"\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                        + frame
                        + b"\r\n"
                    )
                time.sleep(0.03)
        finally:
            stream_limiter.release()
            audit.log(
                "stream_disconnect",
                success=True,
                client_ip=client_ip,
                actor=operator["sub"],
            )

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_ip = _client_ip(websocket)
    if not websocket_attempt_limiter.allow(client_ip or "unknown"):
        await websocket.close(code=1013)
        return
    operator = auth.verify_session(websocket.cookies.get(config.SESSION_COOKIE_NAME))
    host = websocket.headers.get("host", "")
    host_ok = is_allowed_host(host, config.ALLOWED_HOSTS)
    origin_ok = is_allowed_origin(
        websocket.headers.get("origin"),
        host,
        config.ALLOWED_ORIGINS,
    )
    https_ok = not config.REQUIRE_HTTPS or _is_https(websocket)
    if not operator or not host_ok or not origin_ok or not https_ok:
        audit.log(
            "websocket_connect",
            success=False,
            client_ip=client_ip,
            actor=operator.get("sub") if operator else None,
            details={"host_allowed": host_ok, "origin_allowed": origin_ok, "https": https_ok},
        )
        await websocket.close(code=4403)
        return
    if not websocket_limiter.acquire():
        await websocket.close(code=1013)
        return

    subscriber, history = broker.subscribe()
    await websocket.accept()
    audit.log(
        "websocket_connect",
        success=True,
        client_ip=client_ip,
        actor=operator["sub"],
    )
    try:
        for alert in history:
            await websocket.send_json({"type": "alert", "data": alert})
        while True:
            while True:
                try:
                    alert = subscriber.get_nowait()
                except queue.Empty:
                    break
                await websocket.send_json({"type": "alert", "data": alert})
            await websocket.send_json({"type": "stats", "data": capture_service.stats()})
            await asyncio.sleep(config.STATS_PUSH_INTERVAL_SECONDS)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        broker.unsubscribe(subscriber)
        websocket_limiter.release()
        audit.log(
            "websocket_disconnect",
            success=True,
            client_ip=client_ip,
            actor=operator["sub"],
        )


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=os.getenv("IBVAP_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config.validate_server_security()
    uvicorn.run(
        app,
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        proxy_headers=True,
        forwarded_allow_ips=config.FORWARDED_ALLOW_IPS,
        ssl_certfile=config.TLS_CERTFILE,
        ssl_keyfile=config.TLS_KEYFILE,
    )
