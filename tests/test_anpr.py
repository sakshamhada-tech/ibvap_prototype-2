import pytest

from utils.anpr import (
    SUPPORTED_REGIONS,
    PlateConsensusTracker,
    PlateObservation,
    combine_ocr_results,
    identify_plate_region,
    normalize_plate_text,
    plate_matches_region,
)


def observation(
    text="KA01AB1234",
    *,
    region="IN",
    ocr=0.9,
    detector=0.8,
    sharpness=100.0,
    bbox=(10, 20, 80, 45),
):
    return PlateObservation(text, bbox, region, ocr, detector, sharpness)


def tracker(**overrides):
    settings = {
        "min_reads": 3,
        "window_seconds": 3.0,
        "min_weight_ratio": 0.6,
        "min_ocr_confidence": 0.5,
        "max_observations": 12,
        "state_ttl_seconds": 5.0,
    }
    settings.update(overrides)
    return PlateConsensusTracker(**settings)


def test_normalization_preserves_scripts_marks_and_normalizes_decimal_digits():
    assert normalize_plate_text(" ka-01 ab 1234 ") == "KA01AB1234"
    assert normalize_plate_text(" ঢাকা-মেট্রো গ-১২৩৪ ") == "ঢাকামেট্রোগ1234"
    assert normalize_plate_text("्क१२") == "क12"  # unattached leading mark is discarded
    assert normalize_plate_text("A²-1") == "A21"  # NFKC normalization happens first


@pytest.mark.parametrize(
    ("text", "region"),
    [
        ("KA01AB1234", "IN"),
        ("22BH1234AA", "IN"),
        ("77CD1", "IN"),
        ("ঢাকামেট্রোগ1234", "BD"),
        ("बाप1234", "NP"),
        ("ABC123", "PK"),
        ("BP1A1234", "BT"),
        ("CAB1234", "LK"),
        ("WPCAB1234", "LK"),
        ("ကခ1234", "MM"),
        ("京A12345", "CN"),
        ("کابل12345", "AF"),
        ("A1234", "MV"),
        ("ABC123", "GENERIC"),
    ],
)
def test_supported_profiles_accept_representative_structures(text, region):
    assert region in SUPPORTED_REGIONS
    assert plate_matches_region(text, region)


@pytest.mark.parametrize(
    ("text", "region"),
    [
        ("ZZ01AB1234", "IN"),
        ("KA1234", "IN"),
        ("KA01অ1234", "IN"),
        ("DHAKA1234", "BD"),
        ("ABC123", "NP"),
        ("123456", "PK"),
        ("ZZ1A1234", "BT"),
        ("A1234", "LK"),
        ("ABC123", "MM"),
        ("ABC123", "CN"),
        ("ABC123", "AF"),
        ("12345", "MV"),
        ("123456", "GENERIC"),
    ],
)
def test_profiles_abstain_from_invalid_or_wrong_script_reads(text, region):
    assert not plate_matches_region(text, region)


def test_ocr_tokens_are_grouped_into_rows_then_weighted_by_character_count():
    def box(x1, y1, x2, y2):
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    results = [
        (box(40, 1, 60, 11), "01", 0.8),
        (box(1, 3, 25, 13), "ka-", 0.9),
        (box(35, 23, 70, 33), "1234", 0.6),
        (box(0, 21, 25, 31), "ab", 0.7),
        ([], "bad", 0.99),
        (box(0, 50, 10, 60), "", 0.99),
        (box(0, 50, 10, 60), "ignored", float("nan")),
    ]
    text, confidence = combine_ocr_results(results)
    assert text == "KA01AB1234"
    assert abs(confidence - 0.72) < 1e-9
    assert combine_ocr_results([]) is None


def test_unknown_profile_raises_and_profile_order_is_respected():
    with pytest.raises(ValueError, match="unsupported"):
        plate_matches_region("ABC123", "XX")
    assert identify_plate_region("KA01AB1234", ("GENERIC", "IN")) == "GENERIC"
    assert identify_plate_region("NOTAPLATE", ("IN", "BD")) is None


def test_observation_quality_is_bounded_and_uses_crop_sharpness():
    assert observation(ocr=2, detector=2, sharpness=1000).quality == 1.0
    assert observation(ocr=-1, detector=-1, sharpness=-1).quality == 0.0
    assert observation(sharpness=200).quality > observation(sharpness=0).quality


def test_consensus_abstains_then_emits_each_stable_text_once_per_track():
    consensus = tracker()
    assert consensus.add(7, observation(), 0.0) is None
    assert consensus.add(7, observation(), 0.5) is None
    stable = consensus.add(7, observation(), 1.0)
    assert stable is not None
    assert stable.text == "KA01AB1234"
    assert stable.region == "IN"
    assert stable.observations == 3
    assert stable.confidence == 1.0
    assert stable.newly_stable

    repeated = consensus.add(7, observation(), 1.5)
    assert repeated is not None
    assert not repeated.newly_stable


def test_consensus_uses_quality_weighted_share_and_best_candidate_bbox():
    consensus = tracker(min_reads=2, min_weight_ratio=0.65)
    strong = observation(bbox=(1, 2, 30, 12), ocr=0.99, detector=0.99, sharpness=200)
    weaker = observation(bbox=(3, 4, 32, 14), ocr=0.75, detector=0.6, sharpness=10)
    wrong = observation("MH01CD5678", ocr=0.5, detector=0.1, sharpness=0)
    assert consensus.add(2, strong, 0) is None
    assert consensus.add(2, wrong, 0.2) is None
    stable = consensus.add(2, weaker, 0.4)
    assert stable is not None
    assert stable.text == strong.text
    assert stable.bbox == strong.bbox
    assert stable.confidence >= 0.65


def test_consensus_rejects_low_confidence_and_old_reads():
    consensus = tracker(min_reads=2, window_seconds=1)
    assert consensus.add(1, observation(ocr=0.49), 0) is None
    assert consensus.tracked_count == 0
    assert consensus.add(1, observation(), 0) is None
    assert consensus.add(1, observation(), 2) is None
    assert consensus.tracked_count == 1


def test_consensus_expires_and_reset_clears_track_and_emission_state():
    consensus = tracker(min_reads=1, state_ttl_seconds=2)
    assert consensus.add(8, observation(), 1).newly_stable
    assert consensus.expire(2) == set()
    assert consensus.expire(4) == {8}
    assert consensus.tracked_count == 0
    assert consensus.add(8, observation(), 4).newly_stable
    consensus.reset()
    assert consensus.tracked_count == 0
    assert consensus.add(8, observation(), 5).newly_stable


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_reads": 0},
        {"min_reads": 3, "max_observations": 2},
        {"min_weight_ratio": 1.1},
        {"min_ocr_confidence": -0.1},
        {"window_seconds": 0},
        {"state_ttl_seconds": 0},
    ],
)
def test_consensus_configuration_rejects_unsafe_values(overrides):
    with pytest.raises(ValueError):
        tracker(**overrides)
