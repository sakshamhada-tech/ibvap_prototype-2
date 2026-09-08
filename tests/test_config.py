import pytest

import config


def test_feature_defaults_are_conservative_and_alarm_is_severity_scoped():
    assert config._env_bool("A_VARIABLE_THAT_IS_NOT_SET", False) is False
    assert config.ENABLE_GFPGAN_RESTORATION is False
    assert config.ENABLE_ANPR is False
    assert config.ANPR_DEBUG is False
    assert len(config.ANPR_PLATE_MODEL_SHA256) == 64
    assert config.ANPR_REGIONS == ("IN",)
    assert config.ENABLE_AUDIBLE_ALARMS is True
    assert config.AUDIBLE_ALARM_EVENTS == (
        "VIRTUAL_FENCE_INTRUSION",
        "SUSPICIOUS_LOITERING",
    )


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
