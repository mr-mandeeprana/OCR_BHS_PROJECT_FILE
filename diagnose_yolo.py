"""
OCR_BHS - DIAGNOSTIC 2: Real YOLODetector class, real config values.

Tests app.detection.yolo_detector.YOLODetector directly (the exact
class pipeline.py uses) with the exact confidence/imgsz/classes from
config.yaml, on the same video, completely outside pipeline.py's
threading/interval logic.

Run from project root:

    python diagnose_yolodetector.py
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


def main() -> int:
    with open(PROJECT_ROOT / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    detection_cfg = cfg["detection"]
    video_path = str(PROJECT_ROOT / cfg["cameras"]["camera_1"]["video_source"])
    model_path = str(PROJECT_ROOT / detection_cfg["model_path"])

    confidence = float(detection_cfg.get("confidence", 0.30))
    image_size = int(detection_cfg.get("image_size", 640))
    device = detection_cfg.get("device", "cpu")
    max_det = int(detection_cfg.get("max_det", 20))
    classes = detection_cfg.get("classes", [0])

    print("=" * 72)
    print("DIAGNOSTIC 2: Real YOLODetector class, real config values")
    print("=" * 72)
    print(f"confidence={confidence} image_size={image_size} device={device} "
          f"classes={classes} max_det={max_det}")
    print()

    detector = YOLODetector(
        model_path=model_path,
        confidence=confidence,
        image_size=image_size,
        device=device,
        classes=classes,
        max_det=max_det,
        verbose=False,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[FAIL] Could not open video.")
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, max(total_frames - 1, 0), num=20, dtype=int)

    total_detections = 0
    frames_with_detection = 0

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Frame {idx:6d} | [FAIL] could not read frame")
            continue

        try:
            detections = detector.detect(frame)
        except Exception as exc:
            print(f"Frame {idx:6d} | [EXCEPTION] {type(exc).__name__}: {exc}")
            continue

        n = len(detections)
        total_detections += n
        if n > 0:
            frames_with_detection += 1
            top = max(d.confidence for d in detections)
            print(f"Frame {idx:6d} | detections={n} | top_conf={top:.3f}")
        else:
            print(f"Frame {idx:6d} | detections=0")

    cap.release()

    print()
    print("=" * 72)
    print(f"Frames with >=1 detection: {frames_with_detection} / {len(sample_indices)}")
    print(f"Total detections: {total_detections}")
    print("=" * 72)

    if total_detections == 0:
        print("[RESULT] YOLODetector wrapper returns ZERO with real config "
              "values, even though raw ultralytics.YOLO found strong boxes "
              "on this same video. The bug is inside YOLODetector's "
              "detect()/_parse_results()/filter logic, or in how main.py's "
              "create_detector() is constructing it "
              "(wrong confidence/classes actually being passed).")
    else:
        print("[RESULT] YOLODetector wrapper works fine standalone with "
              "real config values. The bug is specifically in pipeline.py's "
              "_run_loop threading/interval/tracker wiring, not detection.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())