from __future__ import annotations

import time
import cv2

from app.config.config_loader import load_config
from app.camera.camera import Camera
from app.detection.yolo_detector import YOLODetector
from app.tracking.tag_tracker import TagTracker
from app.models.detection import Detection as TrackerDetection


def bbox_to_int(bbox):
    """
    Convert any bbox representation to integer x1,y1,x2,y2.
    Avoids dependency on as_xyxy_int being a property or method.
    """
    x1, y1, x2, y2 = bbox

    return (
        int(round(x1)),
        int(round(y1)),
        int(round(x2)),
        int(round(y2)),
    )


def normalize_detection(detection):
    """
    Convert the detector's Detection object into the common
    app.models.detection.Detection format used by TagTracker.
    """

    if isinstance(detection, TrackerDetection):
        return detection

    return TrackerDetection(
        class_id=int(detection.class_id),
        class_name=str(detection.class_name),
        confidence=float(detection.confidence),
        bbox=(
            float(detection.bbox[0]),
            float(detection.bbox[1]),
            float(detection.bbox[2]),
            float(detection.bbox[3]),
        ),
    )


def main() -> None:

    print("=" * 70)
    print(" OCR_BHS - RTSP YOLO26 BYTE TRACKING TEST")
    print("=" * 70)

    # =========================================================
    # 1. LOAD CONFIGURATION
    # =========================================================

    config = load_config()

    enabled_cameras = config.get_enabled_cameras()

    if not enabled_cameras:
        raise RuntimeError(
            "No enabled cameras found in config.yaml"
        )

    # get_enabled_cameras() returns a dictionary.
    # Take the first camera ID safely.
    camera_id = next(iter(enabled_cameras))

    print()
    print(f"Camera              : {camera_id}")
    print(f"Model               : {config.model_path}")
    print(f"Device              : {config.device}")
    print(f"Confidence           : {config.detection_confidence}")
    print(f"Image size           : {config.image_size}")

    # =========================================================
    # 2. INITIALIZE CAMERA
    # =========================================================

    camera = Camera(
        camera_id=camera_id,
        config=config,
    )

    # =========================================================
    # 3. INITIALIZE YOLO
    # =========================================================

    detector = YOLODetector(
        model_path=config.model_path,
        confidence=config.detection_confidence,
        image_size=config.image_size,
        device=config.device,
    )

    # =========================================================
    # 4. INITIALIZE TRACKER
    # =========================================================

    tracker = TagTracker(
        config=config.tracking
    )

    print()
    print("Tracker:")
    print(
        f"  Type                 : "
        f"{tracker.tracker_name}"
    )
    print(
        f"  High threshold       : "
        f"{tracker.track_high_thresh}"
    )
    print(
        f"  Low threshold        : "
        f"{tracker.track_low_thresh}"
    )
    print(
        f"  New track threshold  : "
        f"{tracker.new_track_thresh}"
    )
    print(
        f"  Track buffer         : "
        f"{tracker.track_buffer}"
    )
    print(
        f"  Match threshold      : "
        f"{tracker.match_thresh}"
    )
    print(
        f"  Min hits             : "
        f"{tracker.min_hits}"
    )
    print(
        f"  Prediction buffer    : "
        f"{tracker.prediction_buffer}"
    )
    print(
        f"  Max jump distance    : "
        f"{tracker.max_jump_distance}"
    )
    print(
        f"  Velocity smoothing   : "
        f"{tracker.velocity_smoothing}"
    )
    print(
        f"  History size         : "
        f"{tracker.history_size}"
    )

    # =========================================================
    # 5. START CAMERA
    # =========================================================

    print()
    print("Starting camera...")

    camera.start()

    print("Waiting for first frame...")
    print("Press Q to stop.")
    print()

    frame_id = 0

    processing_start = time.time()

    last_status_time = time.time()

    # =========================================================
    # 6. MAIN LOOP
    # =========================================================

    try:

        while True:

            # -------------------------------------------------
            # GET LATEST CAMERA FRAME
            # -------------------------------------------------

            frame = camera.get_latest_frame()

            if frame is None:
                time.sleep(0.005)
                continue

            frame_id += 1

            # -------------------------------------------------
            # YOLO DETECTION
            # -------------------------------------------------

            raw_detections = detector.detect(frame)

            # -------------------------------------------------
            # NORMALIZE DETECTIONS
            # -------------------------------------------------
            #
            # The existing YOLO detector and tracker currently
            # use different Detection classes.
            #
            # Convert everything into:
            #
            # app.models.detection.Detection
            #
            # before passing detections to TagTracker.
            # -------------------------------------------------

            detections = []

            for detection in raw_detections:

                try:

                    normalized = normalize_detection(
                        detection
                    )

                    detections.append(normalized)

                except Exception as exc:

                    print(
                        f"[Frame {frame_id}] "
                        f"Detection normalization error: {exc}"
                    )

            # -------------------------------------------------
            # TRACKING
            # -------------------------------------------------

            tracks = tracker.update(
                detections=detections,
                frame_id=frame_id,
            )

            # -------------------------------------------------
            # CREATE DISPLAY FRAME
            # -------------------------------------------------

            display = frame.copy()

            # =================================================
            # DRAW RAW YOLO DETECTIONS
            # =================================================

            for detection in detections:

                x1, y1, x2, y2 = bbox_to_int(
                    detection.bbox
                )

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    1,
                )

                detection_label = (
                    f"{detection.class_name} "
                    f"{detection.confidence:.2f}"
                )

                cv2.putText(
                    display,
                    detection_label,
                    (
                        x1,
                        max(20, y1 - 5),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # =================================================
            # DRAW TRACKS
            # =================================================

            for track in tracks:

                # ---------------------------------------------
                # Track detection bbox
                # ---------------------------------------------

                x1, y1, x2, y2 = bbox_to_int(
                    track.detection.bbox
                )

                # ---------------------------------------------
                # Track state
                # ---------------------------------------------

                if track.is_predicted:

                    state = "PREDICTED"

                elif track.is_confirmed:

                    state = "DETECTED"

                else:

                    state = "TENTATIVE"

                # ---------------------------------------------
                # Track ID
                # ---------------------------------------------

                track_id = track.track_id

                # ---------------------------------------------
                # Confidence
                # ---------------------------------------------

                confidence = float(
                    track.confidence
                )

                # ---------------------------------------------
                # Track label
                # ---------------------------------------------

                label = (
                    f"ID {track_id} | "
                    f"{state} | "
                    f"{confidence:.2f}"
                )

                # ---------------------------------------------
                # Track bounding box
                # ---------------------------------------------

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                # ---------------------------------------------
                # Track center
                # ---------------------------------------------

                cx, cy = track.center

                cv2.circle(
                    display,
                    (
                        int(cx),
                        int(cy),
                    ),
                    5,
                    (0, 0, 255),
                    -1,
                )

                # ---------------------------------------------
                # Track label
                # ---------------------------------------------

                cv2.putText(
                    display,
                    label,
                    (
                        x1,
                        max(25, y1 - 10),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # ---------------------------------------------
                # Speed
                # ---------------------------------------------

                speed_text = (
                    f"Speed: "
                    f"{float(track.speed):.1f} px/frame"
                )

                cv2.putText(
                    display,
                    speed_text,
                    (
                        x1,
                        y2 + 20,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                # ---------------------------------------------
                # Missed frames
                # ---------------------------------------------

                if track.missed_frames > 0:

                    missed_text = (
                        f"Missed: "
                        f"{track.missed_frames}"
                    )

                    cv2.putText(
                        display,
                        missed_text,
                        (
                            x1,
                            y2 + 40,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            # =================================================
            # CALCULATE FPS
            # =================================================

            elapsed = (
                time.time() - processing_start
            )

            if elapsed > 0:

                processing_fps = (
                    frame_id / elapsed
                )

            else:

                processing_fps = 0.0

            # =================================================
            # TRACKER STATUS
            # =================================================

            status = tracker.get_status()

            # =================================================
            # DISPLAY SYSTEM STATUS
            # =================================================

            cv2.putText(
                display,
                f"Frame: {frame_id}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                f"FPS: {processing_fps:.1f}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                f"Detections: {len(detections)}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                f"Tracks: {status['active_tracks']}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                (
                    f"Confirmed: "
                    f"{status['confirmed_tracks']}"
                ),
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                (
                    f"Predicted: "
                    f"{status['predicted_tracks']}"
                ),
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # =================================================
            # SHOW FRAME
            # =================================================

            cv2.imshow(
                "OCR_BHS - YOLO26 Tracking",
                display,
            )

            # =================================================
            # PERIODIC TERMINAL STATUS
            # =================================================

            current_time = time.time()

            if current_time - last_status_time >= 5.0:

                print(
                    f"Frame: {frame_id:6d} | "
                    f"FPS: {processing_fps:5.1f} | "
                    f"Detections: {len(detections):2d} | "
                    f"Tracks: "
                    f"{status['active_tracks']:2d} | "
                    f"Confirmed: "
                    f"{status['confirmed_tracks']:2d} | "
                    f"Predicted: "
                    f"{status['predicted_tracks']:2d} | "
                    f"IDs: "
                    f"{status['track_ids']}"
                )

                last_status_time = current_time

            # =================================================
            # KEYBOARD
            # =================================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                print()
                print("Q pressed. Stopping test...")
                break

    except KeyboardInterrupt:

        print()
        print("Keyboard interrupt received.")

    except Exception as exc:

        print()
        print("=" * 70)
        print(" TRACKING TEST ERROR")
        print("=" * 70)
        print(f"Error: {exc}")
        print("=" * 70)

        raise

    finally:

        print()
        print("Stopping camera...")

        try:
            camera.stop()
        except Exception as exc:
            print(f"Camera stop warning: {exc}")

        cv2.destroyAllWindows()

        print()
        print("=" * 70)
        print(" TRACKING TEST COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    main()
