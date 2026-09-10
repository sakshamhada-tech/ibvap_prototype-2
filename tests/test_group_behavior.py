from utils.group_behavior import GroupApproachDetector


def make_detector():
    return GroupApproachDetector(
        fence_line=((0, 100), (200, 100)),
        minimum_size=3,
        proximity_pixels=50,
        window_seconds=3,
        minimum_speed_pixels_per_second=2,
        heading_similarity=0.8,
        minimum_fence_progress_pixels=5,
        stale_after_seconds=3,
    )


def test_converging_group_emits_once_per_episode():
    detector = make_detector()
    initial = [(1, (40, 20)), (2, (55, 20)), (3, (70, 20))]
    approaching = [(1, (42, 35)), (2, (56, 35)), (3, (69, 35))]

    assert detector.update(initial, 0) == ()
    events = detector.update(approaching, 1)
    assert len(events) == 1
    assert events[0].track_ids == (1, 2, 3)
    assert detector.grouped_track_ids == frozenset({1, 2, 3})
    assert all(detector.approach_score(track_id) > 0 for track_id in (1, 2, 3))
    assert detector.update([(1, (44, 50)), (2, (57, 50)), (3, (68, 50))], 2) == ()


def test_individual_fence_approach_signal_does_not_require_a_group():
    detector = make_detector()
    detector.update([(9, (20, 20))], 0)
    assert detector.update([(9, (20, 40))], 1) == ()
    assert detector.approach_score(9) > 0
    assert not detector.grouped_track_ids


def test_independent_or_away_motion_does_not_qualify():
    detector = make_detector()
    detector.update([(1, (40, 50)), (2, (55, 50)), (3, (70, 50))], 0)
    events = detector.update([(1, (25, 40)), (2, (55, 65)), (3, (85, 40))], 1)
    assert events == ()
    assert not detector.grouped_track_ids


def test_stationary_cluster_does_not_qualify_and_state_expires():
    detector = make_detector()
    tracks = [(1, (40, 50)), (2, (55, 50)), (3, (70, 50))]
    assert detector.update(tracks, 0) == ()
    assert detector.update(tracks, 1) == ()
    detector.expire(5)
    assert detector.tracked_count == 0
    detector.reset()
