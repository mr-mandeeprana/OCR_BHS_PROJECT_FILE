from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from app.config.config_loader import Config
from app.camera.camera import Camera
from app.detection.yolo_detector import YOLODetector


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_ID = "camera_1"

MODEL_PATH = (
    PROJECT_ROOT
    / "output"
    / "iata_tag_yolo26n_baseline"
    / "weights"
    / "best.pt"
)

# RTSP YOLO settings
CONFIDENCE = 0.10
IMAGE_SIZE = 832
DEVICE = "cpu"

WINDOW_NAME = "OCR_BHS - YOLO RTSP Test"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("OCR_BHS - YOLO26 RTSP DETECTION TEST")
    print("=" * 80)

    print(f"Project     : {PROJECT_ROOT}")
    print(f"Camera      : {CAMERA_ID}")
    print(f"Model       : {MODEL_PATH}")
    print(f"Confidence  : {CONFIDENCE}")
    print(f"Image Size  : {IMAGE_SIZE}")
    print(f"Device      : {DEVICE}")
    print()

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # LOAD CENTRAL CONFIG
    # --------------------------------------------------------

    config = Config()

    camera_config = config.get_camera(CAMERA_ID)

    if not camera_config:
        raise RuntimeError(
            f"Camera configuration not found: {CAMERA_ID}"
        )

    if not camera_config.get("enabled", False):
        raise RuntimeError(
            f"{CAMERA_ID} is disabled in config.yaml"
        )

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    print("Loading YOLO model...")

    detector = YOLODetector(
        model_path=str(MODEL_PATH),
        confidence=CONFIDENCE,
        image_size=IMAGE_SIZE,
        device=DEVICE,
    )

    print("YOLO model loaded successfully.")
    print()

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    camera = Camera(
        camera_id=CAMERA_ID,
        config=config,
    )

    print("Starting RTSP camera...")

    camera.start()

    print("Waiting for frames...")
    print("Press Q or ESC to stop.")
    print()

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    frame_count = 0
    start_time = time.perf_counter()

    last_detection_time = 0.0
    last_detections = []

    try:

        while True:

            # ------------------------------------------------
            # GET LATEST FRAME
            # ------------------------------------------------

            frame = camera.latest_frame

            if frame is None:
                time.sleep(0.01)
                continue

            frame = frame.copy()

            frame_count += 1

            # ------------------------------------------------
            # YOLO DETECTION
            # ------------------------------------------------

            detections = detector.detect(frame)

            last_detections = detections
            last_detection_time = time.perf_counter()

            # ------------------------------------------------
            # DRAW DETECTIONS
            # ------------------------------------------------

            for index, detection in enumerate(
                sorted(
                    detections,
                    key=lambda d: d.confidence,
                    reverse=True,
                ),
                start=1,
            ):

                x1 = max(0, int(detection.x1))
                y1 = max(0, int(detection.y1))
                x2 = min(frame.shape[1] - 1, int(detection.x2))
                y2 = min(frame.shape[0] - 1, int(detection.y2))

                confidence = detection.confidence

                # Bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3,
                )

                # Center
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                cv2.circle(
                    frame,
                    (cx, cy),
                    5,
                    (0, 0, 255),
                    -1,
                )

                # Label
                label = (
                    f"{detection.class_name} "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                # Detection number
                cv2.putText(
                    frame,
                    f"#{index}",
                    (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            elapsed = time.perf_counter() - start_time

            if elapsed > 0:
                fps = frame_count / elapsed
            else:
                fps = 0.0

            # ------------------------------------------------
            # INFORMATION
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Frame: {frame_count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"YOLO Detections: {len(detections)}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Processing FPS: {fps:.1f}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Conf: {CONFIDENCE:.2f}  Size: {IMAGE_SIZE}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # ------------------------------------------------
            # SHOW FRAME
            # ------------------------------------------------

            cv2.imshow(
                WINDOW_NAME,
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:

        print()
        print("Keyboard interrupt received.")

    finally:

        print()
        print("Stopping camera...")

        camera.stop()

        cv2.destroyAllWindows()

        print("Camera stopped.")
        print()
        print("=" * 80)
        print("YOLO RTSP TEST COMPLETE")
        print("=" * 80)


if __name__ == "__main__":
    main()