import pytest

from utils.zones import (
    LoiteringDetector,
    VirtualFence,
    segment_intersects_box,
    segments_intersect,
)


@pytest.mark.parametrize(
    ("movement", "fence", "expected"),
    [
        (((5, -1), (5, 1)), ((0, 0), (10, 0)), True),
        (((5, -1), (5, 0)), ((0, 0), (10, 0)), True),
        (((2, 0), (8, 0)), ((0, 0), (10, 0)), True),
        (((5, -2), (5, -1)), ((0, 0), (10, 0)), False),
        (((20, 0), (30, 0)), ((0, 0), (10, 0)), False),
    ],
)
def test_segment_intersection_includes_boundaries(movement, fence, expected):
    assert segments_intersect(*movement, *fence) is expected


def test_segment_box_contact_includes_configured_visual_margin():
    assert segment_intersects_box((0, 10), (20, 10), (5, 0, 15, 10))
    assert not segment_intersects_box((0, 10), (20, 10), (5, 0, 15, 7))
    assert segment_intersects_box((0, 10), (20, 10), (5, 0, 15, 7), margin_pixels=3)
    assert not segment_intersects_box((0, 10), (20, 10), (30, 0, 40, 20))


def test_virtual_fence_box_contact_fires_once_and_rearms_after_leaving():
    fence = VirtualFence(((0, 10), (20, 10)), contact_margin_pixels=2)
    assert not fence.check_crossing(1, (10, 2), 0, bounds=(5, 0, 15, 4))
    assert fence.check_crossing(1, (10, 4), 1, bounds=(5, 0, 15, 8))
    assert not fence.check_crossing(1, (10, 5), 2, bounds=(5, 0, 15, 10))
    assert not fence.check_crossing(1, (10, 15), 3, bounds=(5, 13, 15, 18))
    assert fence.check_crossing(1, (10, 8), 4, bounds=(5, 6, 15, 10))

    first_seen_on_fence = VirtualFence(((0, 10), (20, 10)))
    assert first_seen_on_fence.check_crossing(2, (10, 10), 0, bounds=(5, 5, 15, 15))
    assert not first_seen_on_fence.check_crossing(2, (10, 10), 1, bounds=(5, 5, 15, 15))


def test_virtual_fence_ignores_crossing_after_observation_gap():
    fence = VirtualFence(((0, 0), (10, 0)), stale_after_seconds=2)
    assert not fence.check_crossing(1, (5, -1), 0)
    assert not fence.check_crossing(1, (5, 1), 3)
    assert fence.check_crossing(1, (5, -1), 4)


def test_virtual_fence_contact_is_not_repeated_while_leaving_or_following_line():
    fence = VirtualFence(((0, 0), (10, 0)), stale_after_seconds=2)
    assert not fence.check_crossing(1, (5, -1), 0)
    assert fence.check_crossing(1, (5, 0), 1)
    assert not fence.check_crossing(1, (5, 1), 2)
    assert not fence.check_crossing(2, (2, 0), 0)
    assert not fence.check_crossing(2, (8, 0), 1)


def test_virtual_fence_expires_and_resets_tracks():
    fence = VirtualFence(((0, 0), (10, 0)), stale_after_seconds=2)
    fence.check_crossing(1, (5, -1), 0)
    fence.check_crossing(2, (6, -1), 2)
    fence.expire(3)
    assert fence.tracked_count == 1
    fence.reset()
    assert fence.tracked_count == 0


def make_loitering_detector():
    return LoiteringDetector(
        loitering_seconds=3,
        movement_threshold_px=5,
        max_observation_gap_seconds=1.1,
        stale_after_seconds=2,
    )


def test_loitering_remains_active_after_transition():
    detector = make_loitering_detector()
    states = [detector.update(7, (0, 0), timestamp) for timestamp in range(5)]
    assert [state.active for state in states] == [False, False, False, True, True]
    assert [state.started for state in states] == [False, False, False, True, False]


def test_loitering_uses_full_window_spread_not_only_endpoints():
    detector = make_loitering_detector()
    detector.update(7, (0, 0), 0)
    detector.update(7, (10, 0), 1)
    detector.update(7, (0, 0), 2)
    state = detector.update(7, (0, 0), 3)
    assert not state.active


def test_loitering_requires_continuous_observations():
    detector = make_loitering_detector()
    detector.update(7, (0, 0), 0)
    state = detector.update(7, (0, 0), 3)
    assert not state.active
    assert detector.tracked_count == 1


def test_loitering_state_expires():
    detector = make_loitering_detector()
    detector.update(7, (0, 0), 0)
    detector.expire(3)
    assert detector.tracked_count == 0
    detector.update(8, (0, 0), 5)
    detector.reset()
    assert detector.tracked_count == 0
