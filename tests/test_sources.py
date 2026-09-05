import math

from utils.sources import (
    describe_source,
    is_file_source,
    is_network_source,
    normalize_fps,
    recorded_source_time,
)


def test_source_classification_supports_common_url_schemes():
    for source in (
        "rtsp://camera/stream",
        "RTSPS://camera/stream",
        "https://camera/playlist.m3u8",
        "http://camera/feed",
    ):
        assert is_network_source(source)
        assert not is_file_source(source)
    assert is_file_source("recordings/demo.mp4")
    assert not is_file_source(0)


def test_source_description_removes_camera_credentials():
    description = describe_source("rtsp://admin:secret@[2001:db8::1]:554/live")
    assert description == "rtsp://[2001:db8::1]:554/live"
    assert describe_source("recording.mp4") == "recording.mp4"


def test_fps_normalization_rejects_invalid_values():
    assert normalize_fps(30) == 30
    assert normalize_fps(0) == 25
    assert normalize_fps(float("nan")) == 25
    assert normalize_fps("invalid", fallback=20) == 20


def test_recorded_time_prefers_media_timestamp_then_frame_fallback():
    assert recorded_source_time(1500, 99, 30) == 1.5
    assert math.isclose(recorded_source_time(0, 31, 30), 1.0)
