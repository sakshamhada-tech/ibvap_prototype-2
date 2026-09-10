import os
import subprocess
import sys

import pytest

import config


def test_feature_defaults_are_conservative_and_alarm_is_severity_scoped():
    assert config._env_bool("A_VARIABLE_THAT_IS_NOT_SET", False) is False
    assert config.ENABLE_GFPGAN_RESTORATION is False
    assert config.ENABLE_ANPR is False
    assert config.ANPR_DEBUG is False
    assert config.LOITERING_ZONE is None
    assert config.ENABLE_GROUP_APPROACH is False
    assert config.ENABLE_MOTION_SIGNATURE is False
    assert config.ENABLE_TILED_INFERENCE is False
    assert config.ENABLE_NATIVE_FOCUS is False
    assert config.YOLO_TRACKER == "bytetrack.yaml"
    assert config.YOLO_INPUT_SIZE == 640
    assert config.EVALUATION_IOU_THRESHOLD == 0.5
    assert config.EVALUATION_MAX_FRAMES == 10_000
    assert config.EVALUATION_HARDWARE_LABEL == "unrecorded"
    assert config.EVALUATION_BUILD_LABEL == "unrecorded"
    assert config.SCRFD_EXECUTION_PROVIDER == "cpu"
    assert config.ENABLE_POSTURE_ANALYSIS is False
    assert config.ENABLE_CONTEXTUAL_RISK is False
    assert len(config.ANPR_PLATE_MODEL_SHA256) == 64
    assert config.ANPR_REGIONS == ("IN",)
    assert config.ENABLE_AUDIBLE_ALARMS is True
    assert config.AUDIBLE_ALARM_EVENTS == (
        "VIRTUAL_FENCE_INTRUSION",
        "SUSPICIOUS_LOITERING",
    )


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    (
        ("IBVAP_YOLO_TRACKER", "botsort.yaml", "supports only bytetrack.yaml"),
        ("IBVAP_YOLO_INPUT_SIZE", "641", "must be divisible by 32"),
        ("IBVAP_NATIVE_FOCUS_CONF_THRESHOLD", "1.1", "must be <= 1"),
        ("IBVAP_EVALUATION_LOW_FPS_PERCENTILE", "60", "must be <= 50"),
        ("IBVAP_EVALUATION_BUILD_LABEL", " ", "must contain 1..128"),
        ("IBVAP_SCRFD_EXECUTION_PROVIDER", "metal", "must be auto, cpu, cuda, or coreml"),
    ),
)
def test_detector_and_focus_configuration_rejects_invalid_values(variable, value, message):
    environment = os.environ.copy()
    environment[variable] = value
    result = subprocess.run(
        [sys.executable, "-c", "import config"],
        cwd=config.BASE_DIR,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert message in result.stderr


def test_video_source_parser_handles_camera_url_index_and_relative_file():
    assert config._parse_video_source("2") == 2
    assert config._parse_video_source("RTSPS://camera/live") == "RTSPS://camera/live"
    assert config._parse_video_source("recording.mp4").endswith("/recording.mp4")


def test_fence_validation():
    assert config._parse_fence("1,2,3,4") == ((1, 2), (3, 4))
    with pytest.raises(ValueError, match="must be"):
        config._parse_fence("invalid")
    with pytest.raises(ValueError, match="differ"):
        config._parse_fence("1,2,1,2")


def test_optional_loitering_polygon_parser():
    assert config._parse_optional_polygon("anywhere") is None
    assert config._parse_optional_polygon("0,0;100,0;100,100") == (
        (0, 0),
        (100, 0),
        (100, 100),
    )
    with pytest.raises(ValueError, match="at least three"):
        config._parse_optional_polygon("0,0;1,1")


def test_server_security_fails_closed_and_accepts_valid_settings(monkeypatch):
    monkeypatch.setattr(config, "DASHBOARD_PASSWORD", "")
    monkeypatch.setattr(config, "SESSION_SECRET", "")
    with pytest.raises(RuntimeError, match="Unsafe dashboard configuration"):
        config.validate_server_security()

    monkeypatch.setattr(config, "DASHBOARD_PASSWORD", "long-enough-password")
    monkeypatch.setattr(config, "SESSION_SECRET", "s" * 32)
    monkeypatch.setattr(config, "TLS_CERTFILE", None)
    monkeypatch.setattr(config, "TLS_KEYFILE", None)
    monkeypatch.setattr(config, "REQUIRE_HTTPS", False)
    monkeypatch.setattr(config, "HTTPS_BEHIND_PROXY", False)
    monkeypatch.setattr(config, "ALLOWED_HOSTS", ("localhost",))
    config.validate_server_security()
