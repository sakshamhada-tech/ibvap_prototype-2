"""Command-line/OpenCV entry point for the IBVAP analytics pipeline."""

from __future__ import annotations

import logging
import os
import time

import cv2

import config
from pipeline import VideoPipeline
from utils.sources import (
    describe_source,
    is_file_source,
    is_network_source,
    normalize_fps,
    recorded_source_time,
)

LOGGER = logging.getLogger(__name__)


def _open_capture():
    if is_network_source(config.VIDEO_SOURCE):
        capture = cv2.VideoCapture(
            config.VIDEO_SOURCE,
            cv2.CAP_ANY,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                config.CAMERA_OPEN_TIMEOUT_MILLISECONDS,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                config.CAMERA_READ_TIMEOUT_MILLISECONDS,
            ],
        )
    else:
        capture = cv2.VideoCapture(config.VIDEO_SOURCE)
    if capture.isOpened():
        return capture
    capture.release()
    return None


def _wait_for_reconnect(pipeline: VideoPipeline):
    while True:
        LOGGER.warning(
            "video source unavailable; retrying in %.1fs", config.CAMERA_RECONNECT_SECONDS
        )
        time.sleep(config.CAMERA_RECONNECT_SECONDS)
        capture = _open_capture()
        if capture is not None:
            pipeline.reset_tracking()
            LOGGER.info("video source reconnected")
            return capture


def main() -> None:
    logging.basicConfig(
        level=os.getenv("IBVAP_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if is_file_source(config.VIDEO_SOURCE) and not os.path.isfile(config.VIDEO_SOURCE):
        raise FileNotFoundError(
            f"Video source not found: {config.VIDEO_SOURCE}\n"
            "Set IBVAP_VIDEO_SOURCE to a file, camera index, or supported URL."
        )

    capture = _open_capture()
    if capture is None:
        raise RuntimeError(f"Could not open video source: {describe_source(config.VIDEO_SOURCE)}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = normalize_fps(capture.get(cv2.CAP_PROP_FPS))
    if frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise RuntimeError("Video source reported invalid frame dimensions")

    LOGGER.info("source opened: %sx%s at %.2f fps", frame_width, frame_height, source_fps)
    LOGGER.info("virtual fence: %s", config.VIRTUAL_FENCE_LINE)
    pipeline = VideoPipeline()
    writer = None

    if config.SAVE_ANNOTATED_VIDEO:
        os.makedirs(os.path.dirname(config.ANNOTATED_VIDEO_PATH) or ".", exist_ok=True)
        writer = cv2.VideoWriter(
            config.ANNOTATED_VIDEO_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            capture.release()
            pipeline.close()
            raise RuntimeError(f"Could not open output writer: {config.ANNOTATED_VIDEO_PATH}")

    show_window = config.SHOW_LIVE_WINDOW
    if show_window and os.name != "nt" and not os.getenv("DISPLAY"):
        LOGGER.warning("DISPLAY is unavailable; disabling the OpenCV window")
        show_window = False

    source_is_file = is_file_source(config.VIDEO_SOURCE)
    source_frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if source_is_file:
                    LOGGER.info("end of recording")
                    break
                capture.release()
                capture = _wait_for_reconnect(pipeline)
                source_frame_number = 0
                continue

            source_frame_number += 1
            source_time = (
                recorded_source_time(
                    capture.get(cv2.CAP_PROP_POS_MSEC), source_frame_number, source_fps
                )
                if source_is_file
                else time.monotonic()
            )
            annotated = pipeline.process_frame(frame, source_time=source_time)
            if writer is not None:
                writer.write(annotated)
            if show_window:
                cv2.imshow("IBVAP - Border Surveillance", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    LOGGER.info("q pressed; stopping")
                    break
    except KeyboardInterrupt:
        LOGGER.info("interrupted; stopping")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()
        pipeline.close()

    LOGGER.info("processed %s frames", pipeline.frame_number)
    if writer is not None:
        LOGGER.info("annotated video: %s", config.ANNOTATED_VIDEO_PATH)
    LOGGER.info("alert log: %s", config.ALERT_LOG_CSV)


if __name__ == "__main__":
    main()
