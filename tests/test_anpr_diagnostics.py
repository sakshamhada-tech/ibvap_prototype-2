from scripts.diagnose_anpr import diagnostic_advice


def test_diagnostic_advice_explains_disabled_untracked_and_model_failures():
    assert "Enable ANPR" in diagnostic_advice({"status": "disabled"})[0]
    assert "No tracked vehicle" in diagnostic_advice({"status": "ready", "attempts": 0})[0]
    assert (
        "found no candidates"
        in diagnostic_advice({"status": "ready", "attempts": 2, "detector_candidates": 0})[0]
    )
    assert (
        "inference exception"
        in diagnostic_advice({"status": "ready", "attempts": 1, "inference_errors": 1})[0]
    )


def test_diagnostic_advice_maps_quality_ocr_region_and_consensus_counters():
    advice = diagnostic_advice(
        {
            "status": "ready",
            "attempts": 8,
            "detector_candidates": 7,
            "rejected_small": 1,
            "rejected_blurry": 1,
            "ocr_no_text": 1,
            "ocr_low_confidence": 1,
            "region_rejected": 1,
            "valid_observations": 2,
            "stable_reads": 0,
        }
    )
    combined = " ".join(advice)
    assert "MIN_PLATE_WIDTH_PX=40" in combined
    assert "MIN_PLATE_SHARPNESS=5" in combined
    assert "EasyOCR returned no text" in combined
    assert "MIN_OCR_CONFIDENCE=0.30" in combined
    assert "IN,GENERIC" in combined
    assert "MIN_CONSENSUS_READS=2" in combined


def test_diagnostic_advice_reports_success_and_generic_fallback():
    assert (
        "stable read"
        in diagnostic_advice(
            {
                "status": "ready",
                "attempts": 2,
                "detector_candidates": 2,
                "valid_observations": 2,
                "stable_reads": 1,
            }
        )[0]
    )
    assert (
        "ANPR_DEBUG"
        in diagnostic_advice({"status": "ready", "attempts": 1, "detector_candidates": 1})[0]
    )
