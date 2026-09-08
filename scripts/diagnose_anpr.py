"""Run ANPR on a bounded source sample and explain why it accepts or abstains."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from utils.sources import (  # noqa: E402
    describe_source,
    is_file_source,
    normalize_fps,
    recorded_source_time,
)


def diagnostic_advice(metrics: dict) -> list[str]:
    """Turn privacy-safe ANPR counters into concrete troubleshooting guidance."""
    if metrics.get("status") == "disabled":
        return ["Enable ANPR with IBVAP_ENABLE_ANPR=true before running this command."]
    if metrics.get("inference_errors", 0):
        return [
            "Inspect the preceding ANPR inference exception and verify model/runtime compatibility."
        ]
    if metrics.get("attempts", 0) == 0:
        return [
            "No tracked vehicle reached ANPR. Ensure a complete car, motorcycle, bus, or truck is visible; "
            "then use --full-frame once to test plate localization independently from vehicle tracking."
        ]
    if metrics.get("detector_candidates", 0) == 0:
        return [
            "The plate model found no candidates. First verify the footage is Indian-domain and the plate "
            "is visible; for diagnosis only, try IBVAP_ANPR_PLATE_DETECTION_CONFIDENCE=0.15."
        ]
    advice = []
    if metrics.get("rejected_small", 0):
        advice.append(
            "Some candidates were too small. Improve camera resolution/distance or temporarily try "
            "IBVAP_ANPR_MIN_PLATE_WIDTH_PX=40."
        )
    if metrics.get("rejected_blurry", 0):
        advice.append(
            "Some candidates were blurry. Improve focus/shutter/lighting or temporarily try "
            "IBVAP_ANPR_MIN_PLATE_SHARPNESS=5."
        )
    if metrics.get("ocr_no_text", 0):
        advice.append(
            "EasyOCR returned no text. Confirm its recognition weights were provisioned and inspect plate "
            "size, glare, angle, and script-language configuration."
        )
    if metrics.get("ocr_low_confidence", 0):
        advice.append(
            "Region-valid text was below the OCR threshold. For diagnosis only, try "
            "IBVAP_ANPR_MIN_OCR_CONFIDENCE=0.30."
        )
    if metrics.get("region_rejected", 0):
        advice.append(
            "OCR text failed every configured regional profile. Check IBVAP_ANPR_REGIONS/languages; "
            "IN,GENERIC can reveal format issues during diagnosis but is not a production setting."
        )
    if metrics.get("valid_observations", 0) and not metrics.get("stable_reads", 0):
        advice.append(
            "Valid one-frame reads did not reach exact-text consensus. Try a longer/steadier clip or, for "
            "diagnosis only, IBVAP_ANPR_INTERVAL_FRAMES=5 and IBVAP_ANPR_MIN_CONSENSUS_READS=2."
        )
    if metrics.get("stable_reads", 0):
        advice.append(
            "ANPR produced at least one stable read; check ANPR_READ in the dashboard and CSV."
        )
    return advice or ["Review the reported last_result and run again with IBVAP_ANPR_DEBUG=true."]


def _source_time(capture, cv2, frame_number: int, fps: float, source_is_file: bool) -> float:
    if source_is_file:
        return recorded_source_time(capture.get(cv2.CAP_PROP_POS_MSEC), frame_number, fps)
    return time.monotonic()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", help="video path/camera index/URL; defaults to IBVAP_VIDEO_SOURCE"
    )
    parser.add_argument(
        "--max-frames", type=int, default=300, help="bounded frames to inspect (default: 300)"
    )
    parser.add_argument("--output", type=Path, help="optional annotated diagnostic MP4")
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help="diagnostic-only plate scan that bypasses the tracked-vehicle prerequisite",
    )
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if not config.ENABLE_ANPR:
        print("ANPR is disabled. Run: export IBVAP_ENABLE_ANPR=true")
        return 2

    source = config.VIDEO_SOURCE if args.source is None else config._parse_video_source(args.source)
    if is_file_source(source) and not Path(source).is_file():
        print(f"Video source does not exist: {source}")
        return 2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        import cv2

        from pipeline import VideoPipeline, draw_plate
    except ImportError as exc:
        print(f"ANPR runtime dependency is missing: {exc}")
        return 2

    print(f"Source: {describe_source(source)}")
    print(f"Plate model: {config.ANPR_PLATE_MODEL_PATH}")
    print(f"Regions: {','.join(config.ANPR_REGIONS)}")
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"OpenCV could not open: {describe_source(source)}")
        capture.release()
        return 2

    fps = normalize_fps(capture.get(cv2.CAP_PROP_FPS))
    source_is_file = is_file_source(source)
    writer = None
    frame_number = 0
    anpr_alerts = []
    diagnostic_stable_reads = 0
    try:
        pipeline = VideoPipeline(
            on_alert=lambda alert: (
                anpr_alerts.append(alert) if alert.get("alert_type") == "ANPR_READ" else None
            ),
            show_overlays=args.output is not None,
        )
    except Exception as exc:
        capture.release()
        print(f"ANPR initialization failed: {exc}")
        return 2

    try:
        while frame_number < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            timestamp = _source_time(capture, cv2, frame_number, fps, source_is_file)
            if args.full_frame:
                annotated = frame.copy()
                if frame_number % config.ANPR_INTERVAL_FRAMES == 0:
                    height, width = frame.shape[:2]
                    observation = pipeline.detector.detect_plate_observation(
                        frame, (0, 0, width, height)
                    )
                    if observation is not None:
                        consensus = pipeline.anpr_consensus.add(0, observation, timestamp)
                        if consensus is not None and consensus.newly_stable:
                            diagnostic_stable_reads += 1
                            anpr_alerts.append(
                                {"alert_type": "ANPR_READ", "details": consensus.text}
                            )
                        if args.output is not None:
                            draw_plate(annotated, observation.bbox, f"OBS {observation.text}")
            else:
                annotated = pipeline.process_frame(frame, source_time=timestamp)
            if args.output is not None:
                if writer is None:
                    destination = args.output.expanduser().resolve()
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(destination),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"could not create diagnostic video {destination}")
                writer.write(annotated)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        pipeline.close()

    metrics = (
        {
            **pipeline.detector.anpr_diagnostics(),
            "consensus_tracks": pipeline.anpr_consensus.tracked_count,
            "stable_reads": diagnostic_stable_reads,
        }
        if args.full_frame
        else dict(pipeline.stats.get("anpr", {}))
    )
    metrics["mode"] = "full-frame-diagnostic" if args.full_frame else "tracked-vehicle"
    metrics["frames_inspected"] = frame_number
    metrics["anpr_alerts"] = len(anpr_alerts)
    print("\nANPR diagnostics")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("\nRecommended next step")
    for message in diagnostic_advice(metrics):
        print(f"- {message}")
    if args.output is not None:
        print(f"- Annotated output: {args.output.expanduser().resolve()}")
    return 0 if frame_number and not metrics.get("inference_errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
