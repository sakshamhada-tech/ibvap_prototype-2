import threading

from alarms import AlarmPlayer


def test_alarm_player_uses_distinct_non_blocking_event_patterns():
    played = []
    completed = threading.Event()

    def playback(path, volume):
        played.append((path.name, path.read_bytes(), volume))
        if len(played) == 2:
            completed.set()

    player = AlarmPlayer(
        enabled=True,
        event_types=("VIRTUAL_FENCE_INTRUSION", "SUSPICIOUS_LOITERING"),
        volume=0.6,
        cooldown_seconds=10,
        queue_size=4,
        playback=playback,
    )
    player.start()

    assert player.notify({"alert_type": "VIRTUAL_FENCE_INTRUSION"})
    assert not player.notify({"alert_type": "VIRTUAL_FENCE_INTRUSION"})
    assert player.notify({"alert_type": "SUSPICIOUS_LOITERING"})
    assert not player.notify({"alert_type": "NIGHT_MOVEMENT"})
    assert completed.wait(2)

    snapshot = player.snapshot()
    assert snapshot["alarms_played"] == 2
    assert snapshot["alarms_dropped"] == 0
    assert {name for name, _data, _volume in played} == {
        "virtual_fence_intrusion.wav",
        "suspicious_loitering.wav",
    }
    assert all(data.startswith(b"RIFF") for _name, data, _volume in played)
    assert played[0][1] != played[1][1]
    assert all(volume == 0.6 for _name, _data, volume in played)

    player.close()
    assert player.snapshot()["alarm_status"] == "stopped"


def test_disabled_alarm_player_never_queues_audio():
    player = AlarmPlayer(
        enabled=False,
        event_types=("VIRTUAL_FENCE_INTRUSION",),
        volume=0.75,
        cooldown_seconds=5,
        queue_size=1,
        playback=lambda _path, _volume: None,
    )

    player.start()
    assert not player.notify({"alert_type": "VIRTUAL_FENCE_INTRUSION"})
    assert player.snapshot()["alarm_status"] == "disabled"
    player.close()
