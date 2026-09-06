"""Play one server-side alarm pattern for output and volume verification."""

from __future__ import annotations

import argparse
import sys
import time

import config
from alarms import SUPPORTED_ALARM_EVENTS, AlarmPlayer

ALIASES = {
    "intrusion": "VIRTUAL_FENCE_INTRUSION",
    "loitering": "SUSPICIOUS_LOITERING",
    "night": "NIGHT_MOVEMENT",
    "anpr": "ANPR_READ",
}


def _volume(raw: str) -> float:
    value = float(raw)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("volume must be between 0 and 1")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "event",
        choices=tuple(ALIASES),
        help="alarm pattern to audition",
    )
    parser.add_argument(
        "--volume",
        type=_volume,
        default=config.AUDIBLE_ALARM_VOLUME,
        help="playback volume from 0 to 1",
    )
    args = parser.parse_args(argv)
    event_type = ALIASES[args.event]
    if event_type not in SUPPORTED_ALARM_EVENTS:
        parser.error("unsupported alarm event")

    player = AlarmPlayer(
        enabled=True,
        event_types=(event_type,),
        volume=args.volume,
        cooldown_seconds=0,
        queue_size=1,
    )
    player.start()
    try:
        initial = player.snapshot()
        if initial["alarm_status"] != "ready":
            print("Alarm player unavailable on this device.", file=sys.stderr)
            return 2
        if not player.notify({"alert_type": event_type}):
            print("Alarm could not be queued.", file=sys.stderr)
            return 2
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            state = player.snapshot()
            if state["alarms_played"]:
                print(f"Played {event_type} at volume {args.volume:.2f}.")
                return 0
            if state["alarm_has_error"]:
                print("Alarm playback failed; check the selected output device.", file=sys.stderr)
                return 2
            time.sleep(0.05)
        print("Alarm playback timed out.", file=sys.stderr)
        return 2
    finally:
        player.close()


if __name__ == "__main__":
    raise SystemExit(main())
