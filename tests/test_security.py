import json

from security import (
    AuditLogger,
    AuthManager,
    SlidingWindowRateLimiter,
    is_allowed_host,
    is_allowed_origin,
)


def make_auth():
    return AuthManager("operator", "a sufficiently long password", "s" * 32, 100)


def test_authentication_and_signed_session_lifecycle():
    auth = make_auth()
    assert auth.authenticate("operator", "a sufficiently long password")
    assert not auth.authenticate("operator", "wrong")
    token = auth.create_session(now=1000)
    session = auth.verify_session(token, now=1050)
    assert session["sub"] == "operator"
    assert session["role"] == "operator"
    assert auth.verify_session(token, now=1100) is None


def test_tampered_or_malformed_sessions_are_rejected():
    auth = make_auth()
    token = auth.create_session(now=1000)
    assert auth.verify_session(token + "x", now=1001) is None
    assert auth.verify_session("not-a-token", now=1001) is None
    assert auth.verify_session(None, now=1001) is None


def test_rate_limiter_uses_sliding_window():
    limiter = SlidingWindowRateLimiter(attempts=2, window_seconds=10)
    assert limiter.allow("address", now=0)
    assert limiter.allow("address", now=1)
    assert not limiter.allow("address", now=2)
    assert limiter.allow("address", now=11)
    assert limiter.allow("other", now=2)

    bounded = SlidingWindowRateLimiter(attempts=1, max_keys=2)
    assert bounded.allow("first", now=0)
    assert bounded.allow("second", now=1)
    assert bounded.allow("third", now=2)
    assert len(bounded._events) == 2


def test_host_validation_supports_exact_and_subdomain_patterns():
    assert is_allowed_host("console.example:8443", ("console.example",))
    assert is_allowed_host("north.ops.example", ("*.ops.example",))
    assert not is_allowed_host("evil.example", ("console.example",))
    assert not is_allowed_host("", ("*",))


def test_websocket_origin_must_match_host_or_allowlist():
    assert is_allowed_origin("https://console.example", "console.example", ())
    assert is_allowed_origin("https://public.example", "internal:8000", ("https://public.example",))
    assert not is_allowed_origin("https://evil.example", "console.example", ())
    assert not is_allowed_origin("https://console.example:8443", "console.example:443", ())
    assert not is_allowed_origin("https://console.example/path", "console.example", ())
    assert not is_allowed_origin(None, "console.example", ())


def test_localhost_origin_aliases_are_equivalent_only_on_the_same_port():
    assert is_allowed_origin("http://localhost:8000", "127.0.0.1:8000", ())
    assert is_allowed_origin("http://127.0.0.1:8000", "localhost:8000", ())
    assert is_allowed_origin("http://[::1]:8000", "localhost:8000", ())
    assert not is_allowed_origin("http://localhost:9000", "127.0.0.1:8000", ())


def test_audit_logger_writes_structured_record_without_secret(tmp_path):
    path = tmp_path / "audit" / "events.jsonl"
    logger = AuditLogger(str(path), max_bytes=250, backup_count=2)
    logger.log("login", success=False, client_ip="127.0.0.1", actor="operator", details={"x": 1})
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["event"] == "login"
    assert record["success"] is False
    assert record["details"] == {"x": 1}
    assert "password" not in record

    logger.log("login", success=True, actor="operator", details={"padding": "x" * 200})
    assert (path.parent / "events.jsonl.1").exists()
