import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import server
from security import AuditLogger, AuthManager, SlidingWindowRateLimiter
from utils.broker import AlertBroker, ConnectionLimiter


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(server.config, "validate_server_security", lambda: None)
    monkeypatch.setattr(server.config, "ALLOWED_HOSTS", ("testserver", "localhost", "127.0.0.1"))
    monkeypatch.setattr(server.capture_service, "start", lambda: None)
    monkeypatch.setattr(server.capture_service, "stop", lambda: None)
    monkeypatch.setattr(server.alarm_player, "start", lambda: None)
    monkeypatch.setattr(server.alarm_player, "close", lambda: None)
    monkeypatch.setattr(
        server,
        "auth",
        AuthManager("operator", "a sufficiently long password", "s" * 32, 3600),
    )
    monkeypatch.setattr(server, "audit", AuditLogger(str(tmp_path / "audit.jsonl")))
    monkeypatch.setattr(server, "login_limiter", SlidingWindowRateLimiter(5))
    monkeypatch.setattr(server, "broker", AlertBroker(queue_size=5, history_size=5))
    monkeypatch.setattr(server, "stream_limiter", ConnectionLimiter(2))
    monkeypatch.setattr(server, "websocket_limiter", ConnectionLimiter(2))
    monkeypatch.setattr(server, "websocket_attempt_limiter", SlidingWindowRateLimiter(10))
    with TestClient(server.app) as test_client:
        yield test_client


def sign_in(client):
    return client.post(
        "/login",
        data={"username": "operator", "password": "a sufficiently long password"},
        headers={"origin": "http://testserver"},
        follow_redirects=False,
    )


def test_protected_routes_and_security_headers(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"
    assert response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert client.get("/api/stats").status_code == 401
    assert client.get("/api/faces").status_code == 401
    assert client.get("/api/faces/0").status_code == 401
    assert client.get("/api/focus").status_code == 401
    assert client.get("/api/focus/image").status_code == 401
    assert client.get("/", headers={"host": "evil.example"}).status_code == 400


def test_login_accepts_equivalent_loopback_origin(client):
    response = client.post(
        "/login",
        data={"username": "operator", "password": "a sufficiently long password"},
        headers={"host": "127.0.0.1:8000", "origin": "http://localhost:8000"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/"


def test_login_accepts_browser_same_origin_metadata_fallback(client):
    response = client.post(
        "/login",
        data={"username": "operator", "password": "a sufficiently long password"},
        headers={"origin": "null", "sec-fetch-site": "same-origin"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/"


def test_login_rejects_cross_origin_and_bad_credentials(client):
    cross_origin = client.post(
        "/login",
        data={"username": "operator", "password": "a sufficiently long password"},
        headers={"origin": "https://evil.example", "sec-fetch-site": "cross-site"},
    )
    assert cross_origin.status_code == 403

    rejected = client.post(
        "/login",
        data={"username": "operator", "password": "wrong"},
        headers={"origin": "http://testserver"},
        follow_redirects=False,
    )
    assert rejected.status_code == 303
    assert rejected.headers["location"] == "/login?error=credentials"


def test_authenticated_session_access_and_logout(client):
    response = sign_in(client)
    assert response.status_code == 303
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=strict" in response.headers["set-cookie"]
    assert client.get("/").status_code == 200
    stats = client.get("/api/stats")
    assert stats.status_code == 200
    assert "recording_status" in stats.json()

    logged_out = client.post("/logout", follow_redirects=False)
    assert logged_out.status_code == 303
    assert client.get("/api/stats").status_code == 401


def test_enhanced_face_panel_api_is_authenticated(client):
    sign_in(client)
    server.capture_service._update_face_snapshot(
        {
            "status": "ready",
            "frame_number": 12,
            "updated_monotonic": 1.0,
            "message": None,
            "faces": [
                {
                    "track_id": 7,
                    "confidence": 0.93,
                    "source_jpeg": b"source-jpeg",
                    "aligned_jpeg": b"aligned-jpeg",
                    "jpeg": b"review-jpeg",
                    "restoration_status": "blended",
                    "quality_reason": None,
                    "blend_weight": 0.25,
                },
                {
                    "track_id": 8,
                    "confidence": 0.91,
                    "source_jpeg": b"rejected-source",
                    "jpeg": None,
                    "restoration_status": "rejected",
                    "quality_reason": "source_face_too_small",
                    "blend_weight": None,
                },
            ],
        }
    )

    summary = client.get("/api/faces")
    assert summary.status_code == 200
    payload = summary.json()
    assert payload["faces"][0]["track_id"] == 7
    assert "not forensic evidence" in payload["disclaimer"]
    source = client.get(payload["faces"][0]["source_image_url"])
    aligned = client.get(payload["faces"][0]["aligned_image_url"])
    review = client.get(payload["faces"][0]["review_image_url"])
    assert source.content == b"source-jpeg"
    assert aligned.content == b"aligned-jpeg"
    assert review.status_code == 200
    assert review.headers["content-type"] == "image/jpeg"
    assert review.content == b"review-jpeg"
    assert payload["faces"][1]["aligned_image_url"] is None
    assert payload["faces"][1]["review_image_url"] is None
    assert client.get(payload["faces"][1]["source_image_url"]).content == b"rejected-source"
    assert client.get("/api/faces/1").content == b"rejected-source"
    assert client.get("/api/faces/1?variant=review").status_code == 404
    assert client.get("/api/faces/0?variant=invalid").status_code == 404
    assert client.get("/api/faces/99").status_code == 404


def test_native_focus_api_is_authenticated_and_preserves_source_provenance(client):
    sign_in(client)
    server.capture_service._update_focus_snapshot(
        {
            "status": "ready",
            "frame_number": 42,
            "updated_monotonic": 1.0,
            "message": None,
            "focus": {
                "candidate_episode_id": 3,
                "class_id": 0,
                "class_name": "person",
                "conf": 0.88,
                "bbox": (10, 20, 30, 68),
                "source_candidate": "tile",
                "confirmation_count": 2,
                "confirmation_required": 2,
                "confirmed": True,
                "source_frame_width": 3840,
                "source_frame_height": 2160,
                "source_target_width": 20,
                "source_target_height": 48,
                "source_target_height_band": "25-48 px",
                "source_crop_width": 192,
                "source_crop_height": 192,
                "model_input_size": 640,
                "acquisition_state": "tracker_requested",
                "authoritative_track_id": None,
                "image_jpeg": b"untouched-source-crop",
            },
        }
    )

    summary = client.get("/api/focus")
    assert summary.status_code == 200
    payload = summary.json()
    assert payload["focus"]["source_candidate"] == "tile"
    assert payload["focus"]["source_target_height"] == 48
    assert payload["focus"]["source_target_height_band"] == "25-48 px"
    assert payload["focus"]["authoritative_track_id"] is None
    assert "do not create sensor detail" in payload["disclaimer"]
    image = client.get(payload["focus"]["image_url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    assert image.content == b"untouched-source-crop"
    assert client.get("/api/focus/image?version=999999").status_code == 404


def test_websocket_requires_origin_and_receives_bounded_history(client):
    sign_in(client)
    server.broker.publish(
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "alert_type": "NIGHT_MOVEMENT",
            "track_id": 1,
            "details": "test",
            "frame_number": 1,
        }
    )
    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as websocket:
        message = websocket.receive_json()
        assert message["type"] == "alert"
        assert message["data"]["track_id"] == 1

    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/ws", headers={"origin": "https://evil.example"}),
    ):
        pass


class _FakeWriter:
    def __init__(self, *, opened=True, fail_write=False):
        self.opened = opened
        self.fail_write = fail_write
        self.frames = []
        self.released = False

    def isOpened(self):
        return self.opened

    def write(self, frame):
        if self.fail_write:
            raise RuntimeError("simulated write failure")
        self.frames.append(frame)

    def release(self):
        self.released = True


def _run_capture_once(monkeypatch, tmp_path, writer):
    service = server.CaptureService(AlertBroker())
    frame = SimpleNamespace(shape=(48, 64, 3))

    class FakeCapture:
        def isOpened(self):
            return True

        def read(self):
            service._stop_event.set()
            return True, frame

        def get(self, _property):
            return 25.0

        def set(self, _property, _value):
            return False

        def release(self):
            self.released = True

    class FakePipeline:
        def __init__(self, **_kwargs):
            self.stats = {"frames_processed": 1}
            self.closed = False

        def process_frame(self, received_frame, *, source_time):
            assert source_time > 0
            return received_frame

        def reset_tracking(self):
            raise AssertionError("tracking reset was not expected")

        def close(self):
            self.closed = True

    writer_calls = []

    def create_writer(*args):
        writer_calls.append(args)
        return writer

    fake_cv2 = SimpleNamespace(
        CAP_PROP_FPS=5,
        CAP_PROP_POS_MSEC=0,
        CAP_PROP_POS_FRAMES=1,
        IMWRITE_JPEG_QUALITY=95,
        VideoCapture=lambda _source: FakeCapture(),
        VideoWriter=create_writer,
        VideoWriter_fourcc=lambda *_codec: 1234,
        imencode=lambda *_args: (True, SimpleNamespace(tobytes=lambda: b"jpeg")),
        resize=lambda received_frame, _size: received_frame,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "pipeline", SimpleNamespace(VideoPipeline=FakePipeline))
    monkeypatch.setattr(server.config, "VIDEO_SOURCE", 0)
    monkeypatch.setattr(server.config, "SAVE_ANNOTATED_VIDEO", True)
    monkeypatch.setattr(
        server.config,
        "ANNOTATED_VIDEO_PATH",
        str(tmp_path / "nested" / "annotated.mp4"),
    )

    service._run()
    return service, frame, writer_calls


def test_capture_service_records_annotated_frames_and_finalizes_on_shutdown(monkeypatch, tmp_path):
    writer = _FakeWriter()
    service, frame, writer_calls = _run_capture_once(monkeypatch, tmp_path, writer)

    assert writer_calls == [(str(tmp_path / "nested" / "annotated.mp4"), 1234, 25.0, (64, 48))]
    assert writer.frames == [frame]
    assert writer.released is True
    assert service.latest_jpeg() == b"jpeg"
    assert service.stats()["recording_status"] == "closed"


def test_capture_service_isolates_writer_open_failure(monkeypatch, tmp_path):
    writer = _FakeWriter(opened=False)
    service, _frame, _writer_calls = _run_capture_once(monkeypatch, tmp_path, writer)

    assert writer.released is True
    assert service.latest_jpeg() == b"jpeg"
    assert service.stats()["frames_processed"] == 1
    assert service.stats()["recording_status"] == "failed"


def test_capture_service_isolates_writer_runtime_failure(monkeypatch, tmp_path):
    writer = _FakeWriter(fail_write=True)
    service, _frame, _writer_calls = _run_capture_once(monkeypatch, tmp_path, writer)

    assert writer.released is True
    assert service.latest_jpeg() == b"jpeg"
    assert service.stats()["frames_processed"] == 1
    assert service.stats()["recording_status"] == "failed"


def test_capture_service_fans_alert_to_dashboard_and_server_alarm():
    received_by_alarm = []
    alarm = SimpleNamespace(
        notify=lambda alert: received_by_alarm.append(dict(alert)),
        snapshot=lambda: {"alarm_status": "ready"},
    )
    alert_broker = AlertBroker(queue_size=2, history_size=2)
    subscriber, _history = alert_broker.subscribe()
    service = server.CaptureService(alert_broker, alarm)
    alert = {"alert_type": "VIRTUAL_FENCE_INTRUSION", "track_id": 4}

    service._handle_alert(alert)

    assert subscriber.get_nowait() == alert
    assert received_by_alarm == [alert]
    assert service.stats()["alarm_status"] == "ready"
