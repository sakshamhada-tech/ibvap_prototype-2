"""Non-blocking, severity-coded audible alarms for analytics events."""

from __future__ import annotations

import logging
import math
import platform
import queue
import struct

# Playback uses only fixed absolute executables and application-generated files.
import subprocess  # nosec B404
import tempfile
import threading
import time
import wave
from collections.abc import Callable, Iterable
from itertools import count
from pathlib import Path

LOGGER = logging.getLogger(__name__)
Playback = Callable[[Path, float], None]

SUPPORTED_ALARM_EVENTS = (
    "VIRTUAL_FENCE_INTRUSION",
    "CONTEXTUAL_RISK",
    "GROUP_APPROACH",
    "SUSPICIOUS_LOITERING",
    "NIGHT_MOVEMENT",
    "ANPR_READ",
)
_PRIORITIES = {
    "VIRTUAL_FENCE_INTRUSION": 0,
    "CONTEXTUAL_RISK": 1,
    "GROUP_APPROACH": 1,
    "SUSPICIOUS_LOITERING": 2,
    "NIGHT_MOVEMENT": 3,
    "ANPR_READ": 4,
}
# (frequency_hz, duration_seconds). A zero frequency is silence. Patterns are
# intentionally short and non-verbal to reduce masking and operator fatigue.
_PATTERNS = {
    "VIRTUAL_FENCE_INTRUSION": (
        (1050, 0.16),
        (0, 0.05),
        (720, 0.16),
        (0, 0.05),
        (1050, 0.16),
        (0, 0.05),
        (720, 0.16),
        (0, 0.05),
        (1050, 0.16),
        (0, 0.05),
        (720, 0.16),
    ),
    "CONTEXTUAL_RISK": (
        (740, 0.14),
        (0, 0.07),
        (900, 0.14),
        (0, 0.07),
        (1060, 0.20),
    ),
    "GROUP_APPROACH": (
        (520, 0.16),
        (0, 0.08),
        (620, 0.16),
        (0, 0.08),
        (720, 0.16),
        (0, 0.08),
        (820, 0.20),
    ),
    "SUSPICIOUS_LOITERING": (
        (600, 0.24),
        (0, 0.16),
        (690, 0.24),
        (0, 0.16),
        (780, 0.32),
    ),
    "NIGHT_MOVEMENT": (
        (480, 0.20),
        (0, 0.20),
        (480, 0.20),
    ),
    "ANPR_READ": (
        (820, 0.10),
        (0, 0.05),
        (1040, 0.14),
    ),
}


class AlarmPlayer:
    """Queue analytics alarms without blocking capture or alert persistence."""

    def __init__(
        self,
        *,
        enabled: bool,
        event_types: Iterable[str],
        volume: float,
        cooldown_seconds: float,
        queue_size: int,
        playback: Playback | None = None,
    ):
        self.enabled = enabled
        self.event_types = frozenset(event_types)
        self.volume = volume
        self.cooldown_seconds = cooldown_seconds
        self._playback = playback
        self._queue: queue.PriorityQueue[tuple[int, int, str]] = queue.PriorityQueue(
            maxsize=queue_size
        )
        self._sequence = count()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_queued: dict[str, float] = {}
        self._sounds: dict[str, Path] = {}
        self._temporary_directory: tempfile.TemporaryDirectory | None = None
        self._status = "disabled" if not enabled else "stopped"
        self._last_alarm_type: str | None = None
        self._last_error: str | None = None
        self._played = 0
        self._dropped = 0

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._drain_queue()
        try:
            temporary_directory = tempfile.TemporaryDirectory(prefix="ibvap-alarms-")
            sounds = _create_alarm_files(Path(temporary_directory.name), self.event_types)
            playback = self._playback or _system_playback()
            if playback is None:
                raise RuntimeError("no supported system audio player was found")
        except Exception as exc:
            LOGGER.warning("audible alarms unavailable: %s", exc)
            with self._lock:
                self._status = "unavailable"
                self._last_error = str(exc)
            if "temporary_directory" in locals():
                temporary_directory.cleanup()
            return

        self._temporary_directory = temporary_directory
        self._sounds = sounds
        self._playback = playback
        with self._lock:
            self._status = "ready"
            self._last_error = None
        self._thread = threading.Thread(
            target=self._run,
            name="ibvap-audible-alarm",
            daemon=True,
        )
        self._thread.start()
        LOGGER.info("audible alarms ready for: %s", ", ".join(sorted(self.event_types)))

    def notify(self, alert: dict) -> bool:
        """Queue one configured event, returning immediately in every case."""
        event_type = str(alert.get("alert_type", ""))
        if not self.enabled or event_type not in self.event_types:
            return False
        now = time.monotonic()
        with self._lock:
            if self._status not in {"ready", "playing", "degraded"}:
                return False
            previous = self._last_queued.get(event_type)
            if previous is not None and now - previous < self.cooldown_seconds:
                return False
            try:
                self._queue.put_nowait((_PRIORITIES[event_type], next(self._sequence), event_type))
            except queue.Full:
                self._dropped += 1
                return False
            self._last_queued[event_type] = now
        return True

    def close(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._thread and self._thread.is_alive():
            LOGGER.error("audible-alarm worker did not stop before timeout")
            with self._lock:
                self._status = "shutdown_timeout"
        elif self.enabled:
            with self._lock:
                if self._status != "unavailable":
                    self._status = "stopped"
        self._thread = None
        self._sounds = {}
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "alarm_status": self._status,
                "last_alarm_type": self._last_alarm_type,
                "alarms_played": self._played,
                "alarms_dropped": self._dropped,
                "alarm_has_error": self._last_error is not None,
            }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                _priority, _sequence, event_type = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._stop_event.is_set():
                break
            with self._lock:
                self._status = "playing"
                self._last_alarm_type = event_type
            try:
                self._playback(self._sounds[event_type], self.volume)
            except Exception as exc:
                LOGGER.exception("could not play audible alarm for %s", event_type)
                with self._lock:
                    self._status = "degraded"
                    self._last_error = str(exc)
            else:
                with self._lock:
                    self._played += 1
                    self._status = "ready"
                    self._last_error = None

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return


def _create_alarm_files(directory: Path, event_types: Iterable[str]) -> dict[str, Path]:
    sounds = {}
    for event_type in event_types:
        path = directory / f"{event_type.lower()}.wav"
        _write_tone(path, _PATTERNS[event_type])
        sounds[event_type] = path
    return sounds


def _write_tone(
    path: Path,
    segments: tuple[tuple[int, float], ...],
    *,
    sample_rate: int = 44_100,
) -> None:
    frames = bytearray()
    attack_release_samples = int(sample_rate * 0.01)
    for frequency, duration in segments:
        sample_count = max(1, int(sample_rate * duration))
        for index in range(sample_count):
            if frequency == 0:
                value = 0
            else:
                envelope = min(
                    1.0,
                    index / max(1, attack_release_samples),
                    (sample_count - index - 1) / max(1, attack_release_samples),
                )
                value = int(
                    0.32
                    * envelope
                    * 32_767
                    * math.sin(2 * math.pi * frequency * index / sample_rate)
                )
            frames.extend(struct.pack("<h", value))
    with wave.open(str(path), "wb") as sound_file:
        sound_file.setnchannels(1)
        sound_file.setsampwidth(2)
        sound_file.setframerate(sample_rate)
        sound_file.writeframes(frames)


def _system_playback() -> Playback | None:
    system = platform.system()
    if system == "Darwin" and Path("/usr/bin/afplay").is_file():

        def play_macos(path: Path, volume: float) -> None:
            # Executable and sound file are controlled by this module.
            result = subprocess.run(  # nosec B603
                ["/usr/bin/afplay", "-v", str(volume), str(path)],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            if result.returncode:
                raise RuntimeError(f"afplay exited with status {result.returncode}")

        return play_macos

    if system == "Windows":
        import winsound

        def play_windows(path: Path, _volume: float) -> None:
            winsound.PlaySound(str(path), winsound.SND_FILENAME)

        return play_windows

    for executable in (Path("/usr/bin/paplay"), Path("/usr/bin/aplay")):
        if executable.is_file():

            def play_linux(path: Path, _volume: float, player=executable) -> None:
                # Executable and sound file are controlled by this module.
                result = subprocess.run(  # nosec B603
                    [str(player), str(path)],
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
                if result.returncode:
                    raise RuntimeError(f"{player.name} exited with status {result.returncode}")

            return play_linux
    return None
