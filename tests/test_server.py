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


def test_login_rejects_cross_origin_and_bad_credentials(client):
    cross_origin = client.post(
        "/login",
        data={"username": "operator", "password": "a sufficiently long password"},
        headers={"origin": "https://evil.example"},
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
    assert client.get("/api/stats").status_code == 200

    logged_out = client.post("/logout", follow_redirects=False)
    assert logged_out.status_code == 303
    assert client.get("/api/stats").status_code == 401


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
