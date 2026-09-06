from datetime import datetime, timedelta, timezone

import pytest

from utils.timestamps import current_wall_times


def test_wall_times_represent_same_instant_with_explicit_offsets():
    local, utc = current_wall_times(
        datetime(2026, 9, 6, 12, 30, 45, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    )

    assert local == "2026-09-06T12:30:45+05:30"
    assert utc == "2026-09-06T07:00:45+00:00"


def test_wall_times_reject_ambiguous_naive_datetime():
    with pytest.raises(ValueError, match="aware datetime"):
        current_wall_times(datetime(2026, 9, 6, 12, 30, 45))
