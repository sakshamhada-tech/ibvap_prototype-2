import csv
from datetime import datetime

import pytest

from utils.alert_logger import AlertLogger


def test_alert_logger_writes_header_and_uses_source_time_cooldown(tmp_path):
    path = tmp_path / "nested" / "alerts.csv"
    received = []
    logger = AlertLogger(
        str(path), on_alert=received.append, cooldown_seconds=2, state_ttl_seconds=10
    )
    assert logger.log("NIGHT_MOVEMENT", 1, "person", 1, source_time=0)
    assert not logger.log("NIGHT_MOVEMENT", 1, "person", 2, source_time=1)
    assert logger.log("NIGHT_MOVEMENT", 1, "person", 3, source_time=2)
    logger.close()
    logger.close()

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[0]["source_time_seconds"] == "0.0"
    assert rows[1]["frame_number"] == "3"
    assert len(received) == 2
    assert received[0]["timestamp"] == received[0]["timestamp_utc"]
    assert received[0]["timestamp_utc"].endswith("+00:00")
    local_time = datetime.fromisoformat(rows[0]["timestamp_local"])
    utc_time = datetime.fromisoformat(rows[0]["timestamp_utc"])
    assert local_time.utcoffset() is not None
    assert local_time.timestamp() == utc_time.timestamp()


def test_alert_logger_reset_and_backward_time_allow_new_event(tmp_path):
    logger = AlertLogger(str(tmp_path / "alerts.csv"), cooldown_seconds=100)
    assert logger.log("A", 1, "first", 1, source_time=50)
    assert logger.log("A", 1, "replayed", 2, source_time=0)
    assert not logger.log("A", 1, "suppressed", 3, source_time=1)
    logger.reset_cooldowns()
    assert logger.log("A", 1, "after reset", 4, source_time=1)
    logger.close()


def test_callback_failure_does_not_break_persistence(tmp_path):
    def broken_callback(_alert):
        raise RuntimeError("subscriber failed")

    logger = AlertLogger(str(tmp_path / "alerts.csv"), on_alert=broken_callback)
    assert logger.log("A", 1, "saved", 1, source_time=0)
    assert logger.total_logged == 1
    logger.close()


def test_alert_log_rotates_at_size_limit(tmp_path):
    path = tmp_path / "alerts.csv"
    logger = AlertLogger(str(path), cooldown_seconds=0, max_bytes=180, backup_count=2)
    for index in range(4):
        logger.log("LONG_EVENT", index, "x" * 80, index, source_time=index)
    logger.close()
    assert path.exists()
    assert (tmp_path / "alerts.csv.1").exists()
    assert path.read_text(encoding="utf-8").startswith("timestamp_local,timestamp_utc,")


def test_alert_logger_preserves_old_schema_as_rotated_backup(tmp_path):
    path = tmp_path / "alerts.csv"
    old_contents = (
        "timestamp_utc,source_time_seconds,alert_type,track_id,details,frame_number\n"
        "2026-01-01T00:00:00+00:00,0.0,A,1,old,1\n"
    )
    path.write_text(old_contents, encoding="utf-8")

    logger = AlertLogger(str(path), backup_count=2)
    logger.close()

    assert (tmp_path / "alerts.csv.1").read_text(encoding="utf-8") == old_contents
    assert path.read_text(encoding="utf-8").startswith("timestamp_local,timestamp_utc,")


def test_closed_logger_rejects_writes(tmp_path):
    logger = AlertLogger(str(tmp_path / "alerts.csv"))
    logger.close()
    with pytest.raises(RuntimeError, match="closed"):
        logger.log("A", 1, "late", 1)
