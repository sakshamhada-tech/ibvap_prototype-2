"""
IBVAP Prototype - CLI Demo
---------------------------
AI-Based Intelligent Video Analytics Platform for Border Surveillance
using existing CCTV infrastructure.

This is the terminal/OpenCV-window demo. For the browser dashboard, run
server.py instead - both share the exact same analytics code in pipeline.py.

Run:
    python3 main.py

Edit config.py to point VIDEO_SOURCE at your own video file, a webcam
index (0), or an RTSP camera URL.
"""

import os

import cv2

import config
from pipeline import VideoPipeline


def main():
    if isinstance(config.VIDEO_SOURCE, str) and not config.VIDEO_SOURCE.startswith("rtsp"):
        if not os.path.exists(config.VIDEO_SOURCE):
            raise FileNotFoundError(
                f"Video source not found: {config.VIDEO_SOURCE}\n"
                f"Put a demo video at that path, or change VIDEO_SOURCE in "
                f"config.py to a webcam index (e.g. 0) or an RTSP URL."
            )

    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {config.VIDEO_SOURCE}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    print(f"[INIT] Video source opened: {frame_w}x{frame_h} @ {src_fps:.1f} fps")
    print(f"[INIT] Virtual fence line (pixel coords): {config.VIRTUAL_FENCE_LINE}")
    print("[INIT] Adjust config.VIRTUAL_FENCE_LINE if it doesn't align with your scene.\n")

    pipeline = VideoPipeline()

    writer = None
    if config.SAVE_ANNOTATED_VIDEO:
        os.makedirs(os.path.dirname(config.ANNOTATED_VIDEO_PATH) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(config.ANNOTATED_VIDEO_PATH, fourcc, src_fps,
                                  (frame_w, frame_h))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[INFO] End of stream / cannot read frame. Stopping.")
                break

            annotated = pipeline.process_frame(frame)

            if writer is not None:
                writer.write(annotated)

            if config.SHOW_LIVE_WINDOW:
                cv2.imshow("IBVAP - Border Surveillance Demo", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] 'q' pressed - stopping.")
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        pipeline.close()
        print(f"\n[DONE] Processed {pipeline.frame_number} frames.")
        if config.SAVE_ANNOTATED_VIDEO:
            print(f"[DONE] Annotated video saved to: {config.ANNOTATED_VIDEO_PATH}")
        print(f"[DONE] Alert log saved to: {config.ALERT_LOG_CSV}")


if __name__ == "__main__":
    main()
