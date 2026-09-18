"""
OCR_BHS - DIAGNOSTIC 4: Raw BYTETracker.update() output, unfiltered.

Calls the installed Ultralytics BYTETracker exactly the way
tag_tracker.py does (same _ByteTrackInput adapter), but prints the
RAW row values before any bbox/quality interpretation -- so we can
see the actual column layout BYTETracker.update() returns on this
machine's installed ultralytics version.

Run from project root:

    python diagnose_bytetrack_raw.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.detection.yolo_detector import YOLODetector  # noqa: E402
from app.tracking.tag_tracker import _ByteTrackInput  # noqa: E402
from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: E402
from types import SimpleNamespace  # noqa: E402


def main() -> int:
    with open(PROJECT_ROOT / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    detection_cfg = cfg["detection"]
    tracking_cfg = cfg["tracking"]
    video_path = str(PROJECT_ROOT / cfg["cameras"]["camera_1"]["video_source"])
    model_path = str(PROJECT_ROOT / detection_cfg["model_path"])

    detector = YOLODetector(
        model_path=model_path,
        confidence=float(detection_cfg.get("confidence", 0.30)),
        image_size=int(detection_cfg.get("image_size", 640)),
        device=detection_cfg.get("device", "cpu"),
        classes=detection_cfg.get("classes", [0]),
        max_det=int(detection_cfg.get("max_det", 20)),
        verbose=False,
    )

    # Build a raw ultralytics BYTETracker exactly like TagTracker does.
    args = SimpleNamespace(
        track_high_thresh=float(tracking_cfg.get("track_high_thresh", 0.4)),
        track_low_thresh=float(tracking_cfg.get("track_low_thresh", 0.1)),
        new_track_thresh=float(tracking_cfg.get("new_track_thresh", 0.4)),
        track_buffer=int(tracking_cfg.get("track_buffer", 30)),
        match_thresh=float(tracking_cfg.get("match_thresh", 0.8)),
        fuse_score=bool(tracking_cfg.get("fuse_score", True)),
    )

    # Match tag_tracker.py's exact construction call.
    bt = BYTETracker(args)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[FAIL] Could not open video.")
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Walk sequentially near a region we already know has detections
    # (frames 200-820 showed hits in diagnostic 3).
    start_frame = 190
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    printed_raw = 0

    for i in range(400):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        detections = detector.detect(frame)

        if not detections:
            tracked = bt.update(
                _ByteTrackInput(
                    xywh=np.zeros((0, 4), dtype=np.float32),
                    conf=np.zeros((0,), dtype=np.float32),
                    cls=np.zeros((0,), dtype=np.float32),
                )
            )
            continue

        xywh = []
        conf = []
        cls = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            xywh.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1])
            conf.append(d.confidence)
            cls.append(float(d.class_id))

        results = _ByteTrackInput(
            xywh=np.asarray(xywh, dtype=np.float32),
            conf=np.asarray(conf, dtype=np.float32),
            cls=np.asarray(cls, dtype=np.float32),
        )

        tracked = bt.update(results)
        tracked_array = np.asarray(tracked) if tracked is not None else np.empty((0, 0))

        if tracked_array.size > 0 and printed_raw < 8:
            printed_raw += 1
            print(f"--- frame offset {i} ---")
            print(f"input detections: {len(detections)} | "
                  f"YOLO bboxes (x1,y1,x2,y2): {[d.bbox for d in detections]}")
            print(f"tracked_array shape: {tracked_array.shape}")
            for row in tracked_array:
                print(f"  raw row ({len(row)} cols): {row}")
            print()

        if printed_raw >= 8:
            break

    cap.release()

    if printed_raw == 0:
        print("[RESULT] BYTETracker.update() never returned any rows at all, "
              "even directly, with real detections fed in one at a time. "
              "The issue is upstream of row-format -- BYTETracker itself "
              "isn't confirming/returning tracks (check track_high_thresh, "
              "new_track_thresh, track_buffer, min number of frames needed "
              "for a track to be reported).")
    else:
        print("=" * 72)
        print("Compare each 'raw row' above to the input YOLO bbox on the "
              "same line. If row[0:4] look like (x1,y1,x2,y2) close to the "
              "input bbox -> format assumption in tag_tracker.py is correct, "
              "bug is elsewhere. If row[0:4] look like (cx,cy,w,h) instead "
              "(i.e. small width/height-like numbers in positions 2/3, or "
              "row[0]/row[1] look like a center point rather than a corner) "
              "-> tag_tracker.py's row parsing (x1,y1,x2,y2 = row[0:4]) is "
              "wrong for this ultralytics version, and that's the bug.")
        print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())