"""Small, dependency-free security primitives used by the dashboard."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from urllib.parse import urlparse

from utils.log_rotation import exceeds_limit, rotate_file


class AuthManager:
    """Authenticate one environment-configured operator and sign sessions."""

    def __init__(
        self,
        username: str,
        password: str,
        session_secret: str,
        ttl_seconds: int,
    ):
        self.username = username
        self._password = password
        self._secret = session_secret.encode("utf-8")
        self.ttl_seconds = ttl_seconds

    def authenticate(self, username: str, password: str) -> bool:
        username_matches = hmac.compare_digest(username, self.username)
        password_matches = hmac.compare_digest(password, self._password)
        return username_matches and password_matches

    def create_session(self, *, now: int | None = None) -> str:
        issued_at = int(time.time() if now is None else now)
        payload = {
            "sub": self.username,
            "role": "operator",
            "iat": issued_at,
            "exp": issued_at + self.ttl_seconds,
        }
        encoded = _base64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        signature = _base64url(
            hmac.new(self._secret, encoded.encode("ascii"), hashlib.sha256).digest()
        )
        return f"{encoded}.{signature}"

    def verify_session(self, token: str | None, *, now: int | None = None) -> dict | None:
        if not token or token.count(".") != 1:
            return None
        encoded, supplied_signature = token.split(".", 1)
        expected_signature = _base64url(
            hmac.new(self._secret, encoded.encode("ascii"), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(supplied_signature, expected_signature):
            return None
        try:
            payload = json.loads(_base64url_decode(encoded))
            current_time = int(time.time() if now is None else now)
            valid = (
                payload.get("sub") == self.username
                and payload.get("role") == "operator"
                and isinstance(payload.get("iat"), int)
                and isinstance(payload.get("exp"), int)
                and payload["iat"] <= current_time + 60
                and current_time < payload["exp"]
            )
        except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return payload if valid else None


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


class SlidingWindowRateLimiter:
    """Thread-safe in-memory attempt limiter, intended for login protection."""

    def __init__(
        self,
        attempts: int,
        window_seconds: float = 60.0,
        max_keys: int = 10_000,
    ):
        self.attempts = attempts
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, *, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        cutoff = current - self.window_seconds
        with self._lock:
            if key not in self._events and len(self._events) >= self.max_keys:
                oldest_key = min(self._events, key=lambda item: self._events[item][-1])
                del self._events[oldest_key]
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.attempts:
                return False
            events.append(current)
            return True


def is_allowed_host(host_header: str, allowed_hosts: tuple[str, ...]) -> bool:
    """Validate an HTTP Host value against exact names or ``*.example`` patterns."""
    try:
        hostname = urlparse(f"//{host_header}").hostname
    except ValueError:
        return False
    if not hostname:
        return False
    hostname = hostname.lower()
    for pattern in allowed_hosts:
        normalized = pattern.lower().rstrip(".")
        if normalized == "*" or hostname == normalized:
            return True
        if normalized.startswith("*.") and hostname.endswith(normalized[1:]):
            return True
    return False


def is_allowed_origin(origin: str | None, host: str, allowed_origins: tuple[str, ...]) -> bool:
    """Allow explicitly configured origins or the request's exact host."""
    if not origin:
        return False
    normalized = origin.rstrip("/")
    if normalized in {entry.rstrip("/") for entry in allowed_origins}:
        return True
    try:
        parsed = urlparse(normalized)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and parsed.netloc.lower() == host.lower()


class AuditLogger:
    """Append security events as structured JSON lines with size rotation."""

    def __init__(
        self,
        path: str,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ):
        self.path = path
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(
        self,
        event: str,
        *,
        success: bool,
        client_ip: str | None = None,
        actor: str | None = None,
        details: dict | None = None,
    ) -> None:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            "success": success,
            "client_ip": client_ip,
            "actor": actor,
            "details": details or {},
        }
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=True) + "\n"
        with self._lock:
            if exceeds_limit(self.path, len(line.encode("utf-8")), self.max_bytes):
                rotate_file(self.path, self.backup_count)
            with open(self.path, "a", encoding="utf-8") as file:
                file.write(line)
