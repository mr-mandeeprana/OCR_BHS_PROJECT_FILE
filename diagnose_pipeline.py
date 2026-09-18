"""
OCR_BHS - DIAGNOSTIC 3: Real pipeline cadence, real Camera thread, real
TagTracker, real YOLODetector -- everything pipeline.py actually uses,
run synchronously in the foreground so we can print exactly what
happens each cycle. OCR/barcode workers are NOT started (keeps this
fast and rules out CPU contention from PaddleOCR entirely).

This replays the exact loop body of OCRBHSPipeline._run_loop by
calling the same private methods, so if this shows tracks, the bug is
purely in _run_loop/threading. If this ALSO shows zero, the bug is in
TagTracker.update() itself.

Run from project root:

    python diagnose_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config.config_loader import load_config  # noqa: E402
from app.camera.camera import Camera  # noqa: E402
from app.detection.yolo_detector import YOLODetector  # noqa: E402
from app.tracking.tag_tracker import TagTracker  # noqa: E402
from app.pipeline.pipeline import OCRBHSPipeline  # noqa: E402


def main() -> int:
    config = load_config()

    detection_cfg = config.data.get("detection", {})
    tracking_cfg = config.data.get("tracking", {})

    model_path = PROJECT_ROOT / detection_cfg["model_path"]

    detector = YOLODetector(
        model_path=str(model_path),
        confidence=float(detection_cfg.get("confidence", 0.30)),
        image_size=int(detection_cfg.get("image_size", 640)),
        device=detection_cfg.get("device", "cpu"),
        classes=detection_cfg.get("classes", [0]),
        max_det=int(detection_cfg.get("max_det", 20)),
        verbose=False,
    )

    tracker = TagTracker(
        track_high_thresh=float(tracking_cfg.get("track_high_thresh", 0.4)),
        track_low_thresh=float(tracking_cfg.get("track_low_thresh", 0.1)),
        new_track_thresh=float(tracking_cfg.get("new_track_thresh", 0.4)),
        track_buffer=int(tracking_cfg.get("track_buffer", 45)),
        match_thresh=float(tracking_cfg.get("match_thresh", 0.85)),
        fuse_score=bool(tracking_cfg.get("fuse_score", True)),
        min_hits=int(tracking_cfg.get("min_hits", 3)),
        prediction_buffer=int(tracking_cfg.get("prediction_buffer", 18)),
        max_jump_distance=float(tracking_cfg.get("max_jump_distance", 250.0)),
        min_iou_warning=float(tracking_cfg.get("min_iou_warning", 0.10)),
        unstable_frame_threshold=int(tracking_cfg.get("unstable_frame_threshold", 3)),
        min_bbox_area=float(tracking_cfg.get("min_bbox_area", 0.0)),
        min_track_confidence=float(tracking_cfg.get("min_track_confidence", 0.25)),
    )

    camera = Camera(camera_id="camera_1", config=config)

    pipeline = OCRBHSPipeline(
        camera=camera,
        detector=detector,
        tracker=tracker,
        config=config,
    )

    print("=" * 72)
    print("DIAGNOSTIC 3: Real pipeline cadence (no OCR/barcode workers)")
    print("=" * 72)
    print(f"detection_interval_seconds = {pipeline.detection_interval_seconds}")
    print(f"tracking_interval_seconds  = {pipeline.tracking_interval_seconds}")
    print(f"drop_old_frames            = {pipeline.drop_old_frames}")
    print()

    started = camera.start()
    if not started:
        print("[FAIL] Camera failed to start.")
        return 1

    frame = camera.wait_for_frame(timeout=10)
    if frame is None:
        print("[FAIL] No frame received within 10s.")
        return 1

    print(f"First frame received: {frame.shape[1]}x{frame.shape[0]}")
    print()

    # ------------------------------------------------------------
    # Replay _run_loop's body manually, synchronously, for a fixed
    # wall-clock duration. No threads, no worker manager.
    # ------------------------------------------------------------

    last_detection_time = 0.0
    last_tracking_time = 0.0
    last_frame_id = None

    run_seconds = 20.0
    start_time = time.perf_counter()

    frames_seen = 0
    duplicate_skips = 0
    tracking_gate_skips = 0
    detection_runs = 0
    total_detections = 0
    total_tracks_nonzero_cycles = 0
    max_tracks_in_one_cycle = 0
    last_print = 0.0

    while time.perf_counter() - start_time < run_seconds:

        latest = pipeline._get_latest_frame()
        if latest is None:
            time.sleep(0.005)
            continue

        frame_id, timestamp, frame = latest
        frames_seen += 1

        if pipeline.drop_old_frames and last_frame_id == frame_id:
            duplicate_skips += 1
            time.sleep(0.002)
            continue

        last_frame_id = frame_id

        now = time.perf_counter()

        if (
            pipeline.tracking_interval_seconds > 0
            and now - last_tracking_time < pipeline.tracking_interval_seconds
        ):
            tracking_gate_skips += 1
            time.sleep(0.001)
            continue

        last_tracking_time = now

        run_detection_this_frame = (
            now - last_detection_time >= pipeline.detection_interval_seconds
        )

        if run_detection_this_frame:
            last_detection_time = now
            detection_runs += 1
            try:
                detections = pipeline._run_detection(frame)
            except Exception as exc:
                print(f"[EXCEPTION in _run_detection] {type(exc).__name__}: {exc}")
                detections = []
        else:
            detections = []

        total_detections += len(detections)

        try:
            tracks = pipeline._update_tracker(frame_id=frame_id, detections=detections)
        except Exception as exc:
            print(f"[EXCEPTION in _update_tracker] {type(exc).__name__}: {exc}")
            tracks = []

        if len(tracks) > 0:
            total_tracks_nonzero_cycles += 1
            max_tracks_in_one_cycle = max(max_tracks_in_one_cycle, len(tracks))

        if len(detections) > 0 or (time.perf_counter() - last_print) > 1.0:
            last_print = time.perf_counter()
            print(
                f"t={time.perf_counter()-start_time:5.1f}s | "
                f"frame_id={frame_id} | "
                f"detections={len(detections)} | "
                f"tracks={len(tracks)}"
            )

    camera.stop()

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"frames_seen                 : {frames_seen}")
    print(f"duplicate_skips             : {duplicate_skips}")
    print(f"tracking_gate_skips         : {tracking_gate_skips}")
    print(f"detection_runs              : {detection_runs}")
    print(f"total_detections (summed)   : {total_detections}")
    print(f"cycles with >=1 track       : {total_tracks_nonzero_cycles}")
    print(f"max tracks in a single cycle: {max_tracks_in_one_cycle}")
    print("=" * 72)

    if detection_runs == 0:
        print("[RESULT] _run_detection was NEVER called. The loop's cadence "
              "gating (drop_old_frames / tracking_interval_seconds / "
              "detection_interval_seconds) is preventing detection from "
              "ever running. Check duplicate_skips vs tracking_gate_skips "
              "above to see which gate is eating every frame.")
    elif total_detections == 0:
        print("[RESULT] _run_detection ran many times but returned 0 "
              "detections every time here, even though the standalone "
              "YOLODetector test found real detections on this same video. "
              "That points to _run_detection/_filter_detections in "
              "pipeline.py discarding valid detections, or a difference "
              "between the frame pipeline.py hands it vs. what the "
              "standalone script read directly.")
    elif total_tracks_nonzero_cycles == 0:
        print("[RESULT] Detections ARE happening (see total_detections "
              "above), but TagTracker.update() NEVER returned a nonzero "
              "track list. The bug is inside tag_tracker.py's update() / "
              "ByteTrack wiring / min_track_confidence / min_hits logic, "
              "not detection.")
    else:
        print("[RESULT] Detections AND tracks are both happening here, "
              "synchronously, outside threads. If your live run still "
              "shows zero, the bug is specifically in pipeline.py's "
              "threading (the _loop_thread never actually running, or "
              "worker_manager.start() blocking/crashing before the loop "
              "thread starts).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())