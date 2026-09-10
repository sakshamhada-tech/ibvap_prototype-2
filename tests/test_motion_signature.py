from utils.motion_signature import MotionSignatureAnalyzer


def analyzer():
    return MotionSignatureAnalyzer(
        window_seconds=10,
        minimum_samples=6,
        static_path_threshold_pixels=2,
        vegetation_max_net_displacement_pixels=5,
        high_frequency_zero_crossing_ratio=0.5,
        displacement_scale_pixels=20,
        oscillation_scale_pixels=2,
        static_score=0,
        vegetation_score=0.05,
        gait_score=0.8,
        rigid_score=0.4,
        ambiguous_score=0.2,
    )


def samples(y_values):
    return tuple((float(index), (0.0, float(y))) for index, y in enumerate(y_values))


def test_motion_signal_direction_is_sane_for_synthetic_trajectories():
    model = analyzer()
    vegetation = model.analyze(samples((0, 3, -3, 3, -3, 0)))
    rhythmic = model.analyze(samples((0, 6, 8, 15, 17, 25)))
    rigid = model.analyze(samples((0, 5, 10, 15, 20, 25)))
    static = model.analyze(samples((0, 0, 0, 0, 0, 0)))

    assert vegetation.category == "oscillating_no_net_motion"
    assert rhythmic.category == "directional_rhythmic_motion"
    assert rigid.category == "directional_rigid_motion"
    assert static.category == "static"
    assert rhythmic.score > rigid.score > vegetation.score >= static.score


def test_short_history_has_low_confidence():
    signal = analyzer().analyze(samples((0, 5)))
    assert signal.confidence < 1
    assert 0 <= signal.score <= 1
