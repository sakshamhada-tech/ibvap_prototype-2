"""
IBVAP Prototype - Configuration
--------------------------------
All tunable parameters for the demo live here so you can adapt the system
to a new camera / video without touching the pipeline code.
"""

# ---------------------------------------------------------------------------
# INPUT SOURCE
# ---------------------------------------------------------------------------
# Path to a video file for offline demo, OR an integer (0, 1, ...) for a
# connected webcam, OR an RTSP URL string for a real IP camera, e.g.:
#   "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101"
VIDEO_SOURCE = 0

# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
# Nano model = fastest, good enough for a hackathon demo on CPU.
# Swap to yolov8s.pt / yolov8m.pt for better accuracy if you have a GPU.
YOLO_MODEL_PATH = "yolov8n.pt"

# Detection confidence threshold (0-1). Lower = more detections, more noise.
CONFIDENCE_THRESHOLD = 0.4

# COCO class IDs this system cares about (full list: 80 classes).
# 0 = person, 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ---------------------------------------------------------------------------
# VIRTUAL FENCE
# ---------------------------------------------------------------------------
# Defined as a line segment in pixel coordinates: (x1, y1) -> (x2, y2).
# Anyone/anything whose track crosses this line triggers an intrusion alert.
# NOTE: coordinates depend on your video's resolution - adjust after you
# see the first frame (the demo prints frame size on startup).
VIRTUAL_FENCE_LINE = ((50, 400), (1200, 400))

# ---------------------------------------------------------------------------
# LOITERING / SUSPICIOUS ACTIVITY
# ---------------------------------------------------------------------------
# If a tracked person stays within roughly the same area for longer than
# this many seconds, flag it as suspicious/loitering.
LOITERING_SECONDS = 15
LOITERING_MOVEMENT_THRESHOLD_PX = 60  # max centroid drift still counted as "staying put"

# ---------------------------------------------------------------------------
# NIGHT-TIME DETECTION
# ---------------------------------------------------------------------------
# Average frame brightness (0-255) below this is treated as "night mode".
# In night mode, alerts are tagged separately since low-light detections
# are noisier and worth flagging for human review.
NIGHT_BRIGHTNESS_THRESHOLD = 60

# ---------------------------------------------------------------------------
# FACE DETECTION
# ---------------------------------------------------------------------------
# Only run face detection inside person bounding boxes (cheaper + more
# reliable than scanning the whole frame). Uses OpenCV's built-in Haar
# cascade so no extra model download is required.
ENABLE_FACE_DETECTION = True

# ---------------------------------------------------------------------------
# ANPR (stretch goal - off by default)
# ---------------------------------------------------------------------------
# Full ANPR needs a plate-detector model + OCR (e.g. EasyOCR/PaddleOCR).
# Left as a hook in detector.py - flip this on once you've wired up an
# OCR engine. Kept off by default to keep the base demo fast and dependency
# -light for the hackathon.
ENABLE_ANPR = True

# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
SAVE_ANNOTATED_VIDEO = True
ANNOTATED_VIDEO_PATH = "output/annotated_output.mp4"
ALERT_LOG_CSV = "logs/alerts.csv"

# Show a live OpenCV window while processing (set False on headless servers).
SHOW_LIVE_WINDOW = True
