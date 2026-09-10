from utils.risk_scoring import RiskScorer, RiskSignals

WEIGHTS = {
    "fence_contact": 0.4,
    "fence_proximity": 0.1,
    "loitering": 0.2,
    "motion_signature": 0.1,
    "low_posture": 0.2,
    "night": 0.1,
    "fence_approach": 0.2,
    "group_membership": 0.2,
    "firearm": 0.5,
}


def test_composite_risk_crosses_once_and_explains_contributors():
    scorer = RiskScorer(threshold=0.6, weights=WEIGHTS, stale_after_seconds=3)
    low = scorer.update(7, RiskSignals(loitering=1), 0)
    elevated = scorer.update(
        7,
        RiskSignals(loitering=1, fence_proximity=1, fence_approach=1, group_membership=1),
        1,
    )
    repeated = scorer.update(7, RiskSignals(fence_contact=1, firearm=1), 2)

    assert low.score < 0.6 and not low.crossed
    assert elevated.score >= 0.6 and elevated.crossed
    assert "group_membership" in elevated.details()
    assert not repeated.crossed


def test_one_low_weight_hardcoded_style_signal_does_not_raise_contextual_alert():
    scorer = RiskScorer(threshold=0.6, weights=WEIGHTS, stale_after_seconds=3)
    assessment = scorer.update(1, RiskSignals(loitering=1), 0)
    assert assessment.score == 0.2
    assert not assessment.crossed


def test_risk_state_rearms_and_expires():
    scorer = RiskScorer(threshold=0.5, weights=WEIGHTS, stale_after_seconds=2)
    assert scorer.update(2, RiskSignals(firearm=1), 0).crossed
    assert not scorer.update(2, RiskSignals(), 1).crossed
    assert scorer.update(2, RiskSignals(firearm=1), 2).crossed
    scorer.expire(5)
    assert scorer.tracked_count == 0
