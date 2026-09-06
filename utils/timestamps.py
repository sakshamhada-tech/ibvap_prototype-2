"""Wall-clock timestamp helpers shared by operational logs."""

from __future__ import annotations

from datetime import datetime, timezone


def current_wall_times(now: datetime | None = None) -> tuple[str, str]:
    """Return system-local and UTC ISO-8601 timestamps for the same instant.

    The local value always includes its UTC offset, avoiding ambiguous naive
    timestamps while matching the timezone configured on the server host.
    """
    local_time = datetime.now().astimezone() if now is None else now
    if local_time.tzinfo is None or local_time.utcoffset() is None:
        raise ValueError("current_wall_times requires an aware datetime")
    utc_time = local_time.astimezone(timezone.utc)
    return (
        local_time.isoformat(timespec="seconds"),
        utc_time.isoformat(timespec="seconds"),
    )
