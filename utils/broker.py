"""Thread-safe bounded broadcast primitives for dashboard clients."""

from __future__ import annotations

import queue
import threading
from collections import deque
from contextlib import suppress


class AlertBroker:
    """Fan each alert out to every subscriber without unbounded buffering."""

    def __init__(self, queue_size: int = 100, history_size: int = 100):
        self.queue_size = queue_size
        self._history = deque(maxlen=history_size)
        self._subscribers: set[queue.Queue] = set()
        self._lock = threading.Lock()

    def publish(self, alert: dict) -> None:
        with self._lock:
            self._history.append(dict(alert))
            for subscriber in tuple(self._subscribers):
                if subscriber.full():
                    with suppress(queue.Empty):
                        subscriber.get_nowait()
                subscriber.put_nowait(dict(alert))

    def subscribe(self) -> tuple[queue.Queue, list[dict]]:
        subscriber: queue.Queue = queue.Queue(maxsize=self.queue_size)
        with self._lock:
            self._subscribers.add(subscriber)
            history = [dict(alert) for alert in self._history]
        return subscriber, history

    def unsubscribe(self, subscriber: queue.Queue) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


class ConnectionLimiter:
    """Atomic counter for long-lived HTTP or WebSocket connections."""

    def __init__(self, maximum: int):
        self.maximum = maximum
        self._active = 0
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        with self._lock:
            if self._active >= self.maximum:
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)

    @property
    def active(self) -> int:
        with self._lock:
            return self._active
