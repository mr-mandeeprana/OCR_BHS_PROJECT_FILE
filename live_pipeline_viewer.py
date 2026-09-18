"""
OCR_BHS - SYNCHRONIZED LIVE YOLO + BYTE TRACK VIEWER

Purpose
-------
Display the EXACT frame that was processed by OCRBHSPipeline's
YOLO + ByteTrack detection cycle.

Architecture
------------
Camera -> OCRBHSPipeline -> YOLO -> ByteTrack
                         |
                         +-> synchronized visual state
                                      |
                                      v
                                  this viewer

IMPORTANT
---------
- No second YOLO inference is performed here.
- No second ByteTrack instance is created.
- OCR and barcode workers remain asynchronous.
- The viewer never reads a fresh camera frame for drawing.
- Frame, detections and tracks are synchronized by frame_id.
- This file is a visual/debug entry point only.

Controls
--------
Q / ESC : Stop
P       : Pause / Resume DISPLAY ONLY
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv


# ---------------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------
# OCR_BHS IMPORTS
# ---------------------------------------------------------------------

from app.config.config_loader import Config
from app.camera.camera import Camera
from app.detection.yolo_detector import YOLODetector
from app.tracking.tag_tracker import TagTracker
from app.pipeline.pipeline import OCRBHSPipeline


WINDOW_NAME = "OCR_BHS - Synchronized YOLO + ByteTrack"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def resolve_path(value: str | Path) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def get_enabled_cameras(config: Config) -> dict[str, Any]:
    cameras = config.get_enabled_cameras()

    if isinstance(cameras, dict):
        return cameras

    if isinstance(cameras, list):
        return {
            str(index): value
            for index, value in enumerate(cameras)
        }

    raise TypeError(
        "Unsupported enabled camera configuration: "
        f"{type(cameras).__name__}"
    )


def create_detector(config: Config) -> YOLODetector:
    detection = config.data["detection"]

    model_path = resolve_path(detection["model_path"])

    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}"
        )

    return YOLODetector(
        model_path=str(model_path),
        confidence=float(
            detection.get("confidence", 0.30)
        ),
        image_size=int(
            detection.get("image_size", 640)
        ),
        device=str(
            detection.get("device", "cpu")
        ),
    )


def create_tracker(config: Config) -> TagTracker:
    tracking = config.data.get("tracking", {})

    return TagTracker(
        track_high_thresh=float(
            tracking.get("track_high_thresh", 0.5)
        ),
        track_low_thresh=float(
            tracking.get("track_low_thresh", 0.1)
        ),
        new_track_thresh=float(
            tracking.get("new_track_thresh", 0.6)
        ),
        track_buffer=int(
            tracking.get("track_buffer", 30)
        ),
        match_thresh=float(
            tracking.get("match_thresh", 0.8)
        ),
        fuse_score=bool(
            tracking.get("fuse_score", True)
        ),
        min_hits=int(
            tracking.get("min_hits", 3)
        ),
        prediction_buffer=int(
            tracking.get("prediction_buffer", 5)
        ),
        max_jump_distance=float(
            tracking.get("max_jump_distance", 200.0)
        ),
        min_iou_warning=float(
            tracking.get("min_iou_warning", 0.1)
        ),
        unstable_frame_threshold=int(
            tracking.get("unstable_frame_threshold", 5)
        ),
        min_bbox_area=float(
            tracking.get("min_bbox_area", 100)
        ),
        min_track_confidence=float(
            tracking.get("min_track_confidence", 0.3)
        ),
        frame_rate=30.0,
    )


def get_pipeline_running(
    pipeline: OCRBHSPipeline,
) -> bool:
    value = getattr(
        pipeline,
        "running",
        False,
    )

    if callable(value):
        try:
            value = value()
        except Exception:
            return False

    return bool(value)


# ---------------------------------------------------------------------
# TRACK DRAWING
# ---------------------------------------------------------------------

def draw_track(
    frame,
    track: Any,
) -> None:

    detection = getattr(
        track,
        "detection",
        None,
    )

    if detection is None:
        return

    bbox = getattr(
        detection,
        "bbox",
        None,
    )

    if bbox is None or len(bbox) < 4:
        return

    try:
        x1 = int(float(bbox[0]))
        y1 = int(float(bbox[1]))
        x2 = int(float(bbox[2]))
        y2 = int(float(bbox[3]))
    except (TypeError, ValueError):
        return

    height, width = frame.shape[:2]

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return

    track_id = getattr(
        track,
        "track_id",
        "?",
    )

    confidence = getattr(
        detection,
        "confidence",
        None,
    )

    confirmed = getattr(
        track,
        "is_confirmed",
        True,
    )

    predicted = getattr(
        track,
        "is_predicted",
        False,
    )

    label = f"ID: {track_id}"

    if confidence is not None:
        try:
            label += f" | {float(confidence):.2f}"
        except Exception:
            pass

    if predicted:
        label += " | PREDICTED"
    elif not confirmed:
        label += " | TENTATIVE"

    # Bounding box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3,
    )

    # Center point
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    cv2.circle(
        frame,
        (cx, cy),
        5,
        (0, 255, 255),
        -1,
    )

    # Label
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.60
    thickness = 2

    (text_width, text_height), _ = cv2.getTextSize(
        label,
        font,
        scale,
        thickness,
    )

    label_y = max(
        y1,
        text_height + 12,
    )

    cv2.rectangle(
        frame,
        (
            x1,
            label_y - text_height - 12,
        ),
        (
            x1 + text_width + 12,
            label_y,
        ),
        (0, 255, 0),
        -1,
    )

    cv2.putText(
        frame,
        label,
        (
            x1 + 6,
            label_y - 6,
        ),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------
# OVERLAY
# ---------------------------------------------------------------------

def draw_overlay(
    frame,
    camera_id: str,
    processed_frame_id: int,
    tracks: list[Any],
    detection_count: int,
    pipeline_stats: dict[str, Any],
    display_fps: float,
    paused: bool,
) -> None:

    height, width = frame.shape[:2]

    # Header
    cv2.rectangle(
        frame,
        (0, 0),
        (width, 82),
        (15, 15, 15),
        -1,
    )

    header = (
        f"OCR_BHS | {camera_id} | "
        f"Processed Frame: {processed_frame_id} | "
        f"FPS: {display_fps:.1f}"
    )

    cv2.putText(
        frame,
        header,
        (15, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    status = (
        f"YOLO Detections: {detection_count} | "
        f"ByteTrack Tracks: {len(tracks)}"
    )

    if paused:
        status += " | DISPLAY PAUSED"

    cv2.putText(
        frame,
        status,
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Track boxes
    for track in tracks:
        draw_track(frame, track)

    # Footer
    processed = pipeline_stats.get(
        "frames_processed",
        0,
    )

    detection_runs = pipeline_stats.get(
        "detection_runs",
        0,
    )

    ocr_jobs = pipeline_stats.get(
        "ocr_jobs_submitted",
        0,
    )

    ocr_results = pipeline_stats.get(
        "ocr_results_received",
        0,
    )

    barcode_jobs = pipeline_stats.get(
        "barcode_jobs_submitted",
        0,
    )

    barcode_results = pipeline_stats.get(
        "barcode_results_received",
        0,
    )

    footer = (
        f"Pipeline frames: {processed} | "
        f"YOLO cycles: {detection_runs} | "
        f"OCR: {ocr_results}/{ocr_jobs} | "
        f"Barcode: {barcode_results}/{barcode_jobs}"
    )

    cv2.rectangle(
        frame,
        (0, height - 38),
        (width, height),
        (15, 15, 15),
        -1,
    )

    cv2.putText(
        frame,
        footer,
        (12, height - 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------
# WAITING SCREEN
# ---------------------------------------------------------------------

def create_waiting_frame(
    width: int = 1280,
    height: int = 720,
):
    frame = cv2.UMat(
        height,
        width,
        cv2.CV_8UC3,
    ).get()

    frame[:] = 20

    cv2.putText(
        frame,
        "OCR_BHS",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Waiting for first YOLO + ByteTrack visual state...",
        (50, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "The viewer is synchronized to processed frames.",
        (50, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return frame


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> int:

    print("=" * 72)
    print("OCR_BHS LIVE YOLO + BYTE TRACK VIEWER")
    print("=" * 72)
    print()
    print("SYNCHRONIZED VIEWER")
    print("Frame + YOLO + ByteTrack state come from the SAME pipeline cycle.")
    print("NO second YOLO inference.")
    print("NO second ByteTrack instance.")
    print()
    print("Q / ESC = Stop")
    print("P       = Pause / Resume DISPLAY")
    print()

    # -------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------

    config = Config()

    cameras = get_enabled_cameras(config)

    if not cameras:
        print("[ERROR] No enabled cameras.")
        return 1

    camera_id = next(iter(cameras))
    camera_config = cameras[camera_id]

    print(f"Camera : {camera_id}")
    print(
        f"Source : "
        f"{camera_config.get('video_source', '')}"
    )
    print()

    # -------------------------------------------------------------
    # COMPONENTS
    # -------------------------------------------------------------

    try:
        camera = Camera(
            camera_id,
            config,
        )

        print("[OK] Camera created.")

        detector = create_detector(config)

        print("[OK] YOLO detector created.")

        tracker = create_tracker(config)

        print("[OK] ByteTrack tracker created.")

        pipeline = OCRBHSPipeline(
            camera=camera,
            detector=detector,
            tracker=tracker,
            config=config,
        )

        print("[OK] OCR_BHS pipeline created.")

    except Exception as exc:
        print()
        print(
            "[ERROR] Component creation failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return 2

    # -------------------------------------------------------------
    # START
    # -------------------------------------------------------------

    try:
        pipeline.start()
    except Exception as exc:
        print()
        print(
            "[ERROR] Pipeline start failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return 3

    print()
    print("[OK] Pipeline started.")
    print("[OK] Opening synchronized live processing window.")
    print()

    # -------------------------------------------------------------
    # WINDOW
    # -------------------------------------------------------------

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        1280,
        720,
    )

    # -------------------------------------------------------------
    # DISPLAY STATE
    # -------------------------------------------------------------

    paused = False
    last_visual_frame_id = None

    display_count = 0
    display_start = time.perf_counter()
    display_fps = 0.0

    last_stats_print = time.perf_counter()

    # -------------------------------------------------------------
    # LOOP
    # -------------------------------------------------------------

    try:
        while get_pipeline_running(pipeline):

            key = cv2.waitKey(16) & 0xFF

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                print("[VIEWER] Stop requested.")
                break

            if key in (
                ord("p"),
                ord("P"),
            ):
                paused = not paused

                print(
                    "[VIEWER] "
                    + (
                        "Display paused."
                        if paused
                        else "Display resumed."
                    )
                )

            # -----------------------------------------------------
            # DISPLAY PAUSE ONLY
            # -----------------------------------------------------

            if paused:
                time.sleep(0.03)
                continue

            # -----------------------------------------------------
            # GET SYNCHRONIZED VISUAL STATE
            # -----------------------------------------------------

            try:
                state = pipeline.get_visual_state()
            except AttributeError:
                print(
                    "[VIEWER ERROR] "
                    "OCRBHSPipeline does not expose get_visual_state()."
                )
                break
            except Exception:
                state = {}

            frame = state.get("frame")
            frame_id = state.get("frame_id")

            if frame is None or frame_id is None:
                waiting = create_waiting_frame()
                cv2.imshow(
                    WINDOW_NAME,
                    waiting,
                )
                time.sleep(0.01)
                continue

            # -----------------------------------------------------
            # Only redraw when pipeline produced a NEW processed frame.
            # -----------------------------------------------------

            if frame_id == last_visual_frame_id:
                time.sleep(0.005)
                continue

            last_visual_frame_id = int(frame_id)

            tracks = state.get(
                "tracks",
                [],
            )

            if tracks is None:
                tracks = []

            tracks = list(tracks)

            detection_count = int(
                state.get(
                    "detection_count",
                    0,
                )
                or 0
            )

            # -----------------------------------------------------
            # FPS = synchronized processed-frame display rate.
            # -----------------------------------------------------

            display_count += 1

            elapsed = max(
                time.perf_counter() - display_start,
                0.001,
            )

            display_fps = display_count / elapsed

            # -----------------------------------------------------
            # PIPELINE STATS
            # -----------------------------------------------------

            try:
                stats = pipeline.get_stats()
            except Exception:
                stats = {}

            # -----------------------------------------------------
            # DRAW
            # -----------------------------------------------------

            visual_frame = frame.copy()

            draw_overlay(
                frame=visual_frame,
                camera_id=camera_id,
                processed_frame_id=int(frame_id),
                tracks=tracks,
                detection_count=detection_count,
                pipeline_stats=stats,
                display_fps=display_fps,
                paused=paused,
            )

            cv2.imshow(
                WINDOW_NAME,
                visual_frame,
            )

            # -----------------------------------------------------
            # TERMINAL DIAGNOSTIC
            # -----------------------------------------------------

            now = time.perf_counter()

            if now - last_stats_print >= 5.0:

                print(
                    "[LIVE] "
                    f"visual_frame={frame_id} | "
                    f"processed={stats.get('frames_processed', 0)} | "
                    f"YOLO={stats.get('detection_runs', 0)} | "
                    f"detections={detection_count} | "
                    f"tracks={len(tracks)} | "
                    f"OCR={stats.get('ocr_results_received', 0)} | "
                    f"Barcode={stats.get('barcode_results_received', 0)} | "
                    f"FPS={display_fps:.1f}"
                )

                last_stats_print = now

    except KeyboardInterrupt:
        print("[VIEWER] Keyboard interrupt.")

    except Exception as exc:
        print()
        print(
            "[VIEWER ERROR] "
            f"{type(exc).__name__}: {exc}"
        )

    finally:
        print()
        print("[VIEWER] Stopping OCR_BHS pipeline...")

        try:
            pipeline.stop()
        except Exception as exc:
            print(
                f"[VIEWER] Shutdown warning: {exc}"
            )

        cv2.destroyAllWindows()

        print("[VIEWER] Stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
