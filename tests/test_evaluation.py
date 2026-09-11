import copy
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import evaluate_long_range
from utils.evaluation import (
    bbox_iou,
    compare_runs,
    render_markdown_report,
    target_pixel_height_band,
    validate_annotations,
    validate_trace,
)


def annotations(*, exhaustive=True):
    heights = (20, 40, 80, 120)
    return {
        "schema_version": 1,
        "source_id": "controlled-clip-01",
        "source_fps": 2.0,
        "source_width": 200,
        "source_height": 200,
        "exhaustive_frames": exhaustive,
        "events_exhaustive": True,
        "frames": [
            {
                "frame_number": frame_number,
                "lighting": "daylight" if frame_number < 3 else "low-light",
                "objects": [
                    {
                        "object_id": "person-01",
                        "class_name": "person",
                        "bbox": [10, 10, 30, 10 + height],
                    }
                ],
            }
            for frame_number, height in enumerate(heights, start=1)
        ],
        "events": [
            {
                "event_id": "loiter-01",
                "event_type": "SUSPICIOUS_LOITERING",
                "expected": True,
                "eligible_frame": 3,
                "end_frame": 4,
                "scenario": "stationary-person",
            },
            {
                "event_id": "group-away-01",
                "event_type": "GROUP_APPROACH",
                "expected": False,
                "eligible_frame": 1,
                "end_frame": 4,
                "scenario": "away",
            },
        ],
    }


def detection(frame_number, *, track_id=None, source="full-frame", class_name="person"):
    height = (20, 40, 80, 120)[frame_number - 1]
    return {
        "class_name": class_name,
        "bbox": [10, 10, 30, 10 + height],
        "confidence": 0.8,
        "track_id": track_id,
        "source": source,
    }


def trace(mode):
    if mode == "wide-only":
        frame_detections = {
            1: [
                {
                    "class_name": "car",
                    "bbox": [100, 100, 140, 150],
                    "confidence": 0.7,
                    "track_id": None,
                    "source": "full-frame",
                }
            ],
            2: [detection(2)],
            3: [detection(3, track_id=1)],
            4: [detection(4, track_id=2)],
        }
        alerts = [
            {"alert_type": "SUSPICIOUS_LOITERING", "frame_number": 4},
            {"alert_type": "GROUP_APPROACH", "frame_number": 2},
        ]
        seconds = 0.1
    else:
        frame_detections = {
            1: [detection(1, source="tile")],
            2: [detection(2, track_id=5)],
            3: [detection(3, track_id=5)],
            4: [detection(4, track_id=5)],
        }
        alerts = [{"alert_type": "SUSPICIOUS_LOITERING", "frame_number": 3}]
        seconds = 0.2
    return {
        "schema_version": 1,
        "source_id": "controlled-clip-01",
        "source_fps": 2.0,
        "source_width": 200,
        "source_height": 200,
        "mode": mode,
        "runtime_warmup_frames": 0,
        "settings": {
            "shared": {
                "build_label": "commit-test",
                "hardware_label": "fake-cpu",
                "tracker": "bytetrack.yaml",
                "model_sha256": "a" * 64,
            },
            "mode_specific": {
                "tiled_inference": mode == "focus-enabled",
                "native_focus": mode == "focus-enabled",
            },
        },
        "run": {"frames_processed": 4, "elapsed_seconds": seconds * 4},
        "frames": [
            {
                "frame_number": frame_number,
                "source_width": 200,
                "source_height": 200,
                "source_frame_sha256": f"{frame_number:064x}",
                "source_time_seconds": frame_number / 2,
                "lighting": "daylight" if frame_number <= 2 else "low-light",
                "processing_seconds": seconds,
                "detections": frame_detections[frame_number],
            }
            for frame_number in range(1, 5)
        ],
        "alerts": alerts,
    }


def compare():
    return compare_runs(
        annotations(),
        trace("wide-only"),
        trace("focus-enabled"),
        iou_threshold=0.5,
        low_fps_percentile=10,
        max_frames=100,
        max_objects_per_frame=10,
        max_alerts=100,
    )


def test_pixel_bands_and_iou_are_source_coordinate_metrics():
    assert [target_pixel_height_band(value) for value in (24, 25, 48, 49, 96, 97)] == [
        "<=24 px",
        "25-48 px",
        "25-48 px",
        "49-96 px",
        "49-96 px",
        ">96 px",
    ]
    assert bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]) == 25 / 175
    assert bbox_iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0


def test_comparison_reports_focus_value_tracking_runtime_and_false_detections():
    report = compare()
    wide = report["runs"]["wide-only"]
    focus = report["runs"]["focus-enabled"]

    assert wide["detection"]["overall"]["recall"] == 0.75
    assert focus["detection"]["overall"]["recall"] == 1.0
    assert wide["authoritative_detection"]["overall"]["recall"] == 0.5
    assert focus["authoritative_detection"]["overall"]["recall"] == 0.75
    assert report["comparison"]["authoritative_recall_delta"] == 0.25
    assert report["comparison"]["by_height_band"]["25-48 px"]["authoritative_recall_delta"] == 1.0
    assert wide["detection"]["false_detections"]["per_minute"] == 30.0
    assert focus["detection"]["false_detections"]["per_minute"] == 0.0
    assert wide["tracking"]["median_acquisition_seconds"] == 0.5
    assert wide["tracking"]["id_switches"] == 1
    assert wide["tracking"]["longest_continuous_seconds"] == 0.5
    assert focus["tracking"]["longest_continuous_seconds"] == 1.5
    assert wide["runtime"]["median_fps"] == 10.0
    assert focus["runtime"]["median_fps"] == 5.0


def test_person_and_vehicle_categories_are_reported_separately():
    labels = annotations()
    labels["frames"][0]["objects"].append(
        {
            "object_id": "vehicle-01",
            "class_name": "car",
            "bbox": [100, 100, 140, 150],
        }
    )
    report = compare_runs(
        labels,
        trace("wide-only"),
        trace("focus-enabled"),
        iou_threshold=0.5,
        low_fps_percentile=10,
        max_frames=100,
        max_objects_per_frame=10,
        max_alerts=100,
    )

    assert report["runs"]["wide-only"]["detection"]["by_category"]["vehicle"]["recall"] == 1.0
    assert report["runs"]["focus-enabled"]["detection"]["by_category"]["vehicle"]["recall"] == 0.0


def test_event_windows_report_loiter_latency_and_group_negative_false_alert():
    report = compare()
    wide_events = report["runs"]["wide-only"]["events"]
    focus_events = report["runs"]["focus-enabled"]["events"]

    assert wide_events["SUSPICIOUS_LOITERING"]["recall"] == 1.0
    assert wide_events["SUSPICIOUS_LOITERING"]["median_alert_latency_seconds"] == 0.5
    assert focus_events["SUSPICIOUS_LOITERING"]["median_alert_latency_seconds"] == 0.0
    assert wide_events["GROUP_APPROACH"]["false_alerts"] == 1
    assert wide_events["GROUP_APPROACH"]["scenarios"]["away"] == {
        "expected_positive": 0,
        "expected_negative": 1,
        "detected": 1,
    }


def test_markdown_report_is_honest_and_contains_presentation_tables():
    markdown = render_markdown_report(compare())
    assert markdown.startswith("# Astra I controlled evaluation")
    assert "# IBVAP controlled evaluation" not in markdown
    assert "Controlled labelled-footage result only" in markdown
    assert "Authoritative tracked recall" in markdown
    assert "25-48 px" in markdown
    assert "SUSPICIOUS_LOITERING" in markdown
    assert "Person/vehicle, target-height, and lighting breakdown" in markdown
    assert "Event scenario breakdown" in markdown
    assert "away" in markdown
    assert "cannot create detail" in markdown


def test_sparse_annotations_withhold_false_detections_per_minute():
    sparse = annotations(exhaustive=False)
    sparse["frames"] = [sparse["frames"][0], sparse["frames"][2], sparse["frames"][3]]
    report = compare_runs(
        sparse,
        trace("wide-only"),
        trace("focus-enabled"),
        iou_threshold=0.5,
        low_fps_percentile=10,
        max_frames=100,
        max_objects_per_frame=10,
        max_alerts=100,
    )
    assert report["runs"]["wide-only"]["detection"]["false_detections"]["per_minute"] is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.update(schema_version=2), "schema_version"),
        (
            lambda payload: payload["frames"].append(copy.deepcopy(payload["frames"][0])),
            "duplicate annotated frame_number",
        ),
        (
            lambda payload: payload["frames"][0]["objects"][0].update(bbox=[1, 1, 1, 2]),
            "positive width",
        ),
        (
            lambda payload: payload["frames"][0].update(lighting="infrared"),
            "lighting",
        ),
        (
            lambda payload: payload["frames"][3]["objects"][0].update(class_name="car"),
            "changes class",
        ),
        (
            lambda payload: payload["events"][0].update(target_object_ids=["unknown"]),
            "unknown objects",
        ),
    ),
)
def test_annotation_validation_rejects_malformed_or_ambiguous_data(mutation, message):
    payload = annotations()
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        validate_annotations(payload, max_frames=100, max_objects_per_frame=10)


def test_exhaustive_annotations_must_be_contiguous():
    payload = annotations()
    del payload["frames"][1]
    with pytest.raises(ValueError, match="every frame"):
        validate_annotations(payload, max_frames=100, max_objects_per_frame=10)


def test_trace_validation_and_pairing_fail_closed():
    malformed = trace("wide-only")
    malformed["frames"][0]["detections"][0]["confidence"] = 2
    with pytest.raises(ValueError, match="confidence"):
        validate_trace(malformed, max_frames=100, max_objects_per_frame=10, max_alerts=100)

    wrong_source = trace("focus-enabled")
    wrong_source["source_id"] = "different"
    with pytest.raises(ValueError, match="source_id"):
        compare_runs(
            annotations(),
            trace("wide-only"),
            wrong_source,
            iou_threshold=0.5,
            low_fps_percentile=10,
            max_frames=100,
            max_objects_per_frame=10,
            max_alerts=100,
        )

    mismatched_frames = trace("focus-enabled")
    mismatched_frames["frames"][0]["source_frame_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="identical decoded source frames"):
        compare_runs(
            annotations(),
            trace("wide-only"),
            mismatched_frames,
            iou_threshold=0.5,
            low_fps_percentile=10,
            max_frames=100,
            max_objects_per_frame=10,
            max_alerts=100,
        )

    mismatched_settings = trace("focus-enabled")
    mismatched_settings["settings"]["shared"]["model_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="shared settings"):
        compare_runs(
            annotations(),
            trace("wide-only"),
            mismatched_settings,
            iou_threshold=0.5,
            low_fps_percentile=10,
            max_frames=100,
            max_objects_per_frame=10,
            max_alerts=100,
        )


def test_report_cli_reads_bounded_json_and_writes_json_and_markdown(tmp_path):
    annotation_path = tmp_path / "annotations.json"
    wide_path = tmp_path / "wide.json"
    focus_path = tmp_path / "focus.json"
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    annotation_path.write_text(json.dumps(annotations()), encoding="utf-8")
    wide_path.write_text(json.dumps(trace("wide-only")), encoding="utf-8")
    focus_path.write_text(json.dumps(trace("focus-enabled")), encoding="utf-8")

    result = evaluate_long_range.main(
        [
            "report",
            "--annotations",
            str(annotation_path),
            "--wide-trace",
            str(wide_path),
            "--focus-trace",
            str(focus_path),
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert result == 0
    assert json.loads(json_path.read_text())["comparison"]["authoritative_recall_delta"] == 0.25
    assert "Baseline versus focus" in markdown_path.read_text()
    assert (
        evaluate_long_range.main(
            [
                "report",
                "--annotations",
                str(annotation_path),
                "--wide-trace",
                str(wide_path),
                "--focus-trace",
                str(focus_path),
                "--json-output",
                str(json_path),
                "--markdown-output",
                str(markdown_path),
            ]
        )
        == 2
    )


def test_trace_collection_uses_fake_runtime_and_retains_only_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate_long_range.config, "ENABLE_TILED_INFERENCE", False)
    monkeypatch.setattr(evaluate_long_range.config, "ENABLE_NATIVE_FOCUS", False)

    class FakeCapture:
        def __init__(self):
            self.reads = 0
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            self.reads += 1
            if self.reads > 2:
                return False, None
            return True, SimpleNamespace(
                shape=(200, 200, 3),
                size=1,
                tobytes=lambda: bytes([self.reads]),
            )

        def get(self, property_id):
            return 2.0 if property_id == 5 else self.reads * 500.0

        def release(self):
            self.released = True

    class FakePipeline:
        def __init__(self, *, on_alert, on_evaluation_frame, **_kwargs):
            self.on_alert = on_alert
            self.on_evaluation_frame = on_evaluation_frame
            self.frame = 0
            self.closed = False

        def process_frame(self, _frame, *, source_time):
            self.frame += 1
            self.on_evaluation_frame(
                {
                    "frame_number": self.frame,
                    "source_time_seconds": source_time,
                    "processing_seconds": 0.1,
                    "lighting": "daylight",
                    "source_width": 200,
                    "source_height": 200,
                    "detections": [detection(self.frame, track_id=4)],
                    "detection_diagnostics": {"tracker": "bytetrack.yaml"},
                    "context_readiness": {"status": "ready"},
                }
            )
            if self.frame == 2:
                self.on_alert(
                    {
                        "alert_type": "SUSPICIOUS_LOITERING",
                        "frame_number": 2,
                        "track_id": 4,
                        "source_time_seconds": source_time,
                        "details": "must not be retained",
                    }
                )

        def close(self):
            self.closed = True

    fake_cv2 = SimpleNamespace(
        CAP_PROP_FPS=5,
        CAP_PROP_POS_MSEC=0,
        VideoCapture=lambda _source: FakeCapture(),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "pipeline", SimpleNamespace(VideoPipeline=FakePipeline))

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake")
    result = evaluate_long_range.collect_trace(
        source_setting=str(source),
        source_id="controlled-clip-01",
        mode="wide-only",
    )

    assert result["run"]["frames_processed"] == 2
    assert result["run"]["reached_eof"] is True
    assert result["runtime_warmup_frames"] == 1
    assert result["alerts"] == [
        {
            "alert_type": "SUSPICIOUS_LOITERING",
            "frame_number": 2,
            "track_id": 4,
            "source_time_seconds": 1.0,
        }
    ]
    assert "details" not in result["alerts"][0]
    assert len(result["frames"][0]["source_frame_sha256"]) == 64
    assert result["frames"][0]["source_frame_sha256"] != result["frames"][1]["source_frame_sha256"]
    assert "source" not in result


def test_collection_mode_requires_an_honest_baseline_or_complete_focus_path(monkeypatch):
    monkeypatch.setattr(evaluate_long_range.config, "ENABLE_TILED_INFERENCE", True)
    monkeypatch.setattr(evaluate_long_range.config, "ENABLE_NATIVE_FOCUS", False)
    with pytest.raises(ValueError, match="wide-only collection requires"):
        evaluate_long_range.validate_collection_mode("wide-only")
    with pytest.raises(ValueError, match="focus-enabled collection requires"):
        evaluate_long_range.validate_collection_mode("focus-enabled")
