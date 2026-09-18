from __future__ import annotations

"""
OCR_BHS - Centralized YOLO26 IATA Tag Detector

This module ONLY performs YOLO object detection.

It does NOT:
- create Track IDs
- run ByteTrack
- run OCR
- run barcode detection
- perform tracking

The detector is configured to use the TRAINED IATA_TAG model.

Expected model:
    models/yolo/iata_tag_yolo26n_best.pt

Expected class:
    0 = IATA_TAG
"""

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.models.detection import Detection

# ------------------------------------------------------------------------
# CPU thread contention.
#
# config.yaml enables shared_model=true with up to 4 camera threads all
# calling this detector's predict() concurrently. Left unbounded, every
# camera thread lets PyTorch/OpenBLAS spawn threads across all CPU cores,
# so N camera threads end up fighting each other (and the OCR/barcode
# worker pools) for the same cores instead of running in parallel. This
# mirrors the fix already proven on the FillPac AI project.
# ------------------------------------------------------------------------
torch.set_num_threads(max(1, (os.cpu_count() or 1) // 2))
torch.set_num_interop_threads(2)


class YOLODetector:
    """
    Shared YOLO26 detector for IATA baggage tags.

    One instance can be shared by multiple cameras.

    The trained model is the default. A model path can still be
    explicitly supplied by the pipeline/configuration.
    """

    IATA_TAG_CLASS_ID = 0
    DEFAULT_TRAINED_MODEL = (
        Path("models")
        / "yolo"
        / "iata_tag_yolo26n_best.pt"
    )

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        confidence: float = 0.30,
        image_size: int = 832,
        device: str = "cpu",
        classes: Optional[Sequence[int]] = None,
        max_det: int = 20,
        verbose: bool = False,
    ) -> None:

        self.project_root = Path(__file__).resolve().parents[2]

        # ------------------------------------------------------------
        # ALWAYS prefer the trained IATA model when no path is given.
        # ------------------------------------------------------------
        if model_path is None:
            model_path = self.project_root / self.DEFAULT_TRAINED_MODEL
        else:
            model_path = Path(model_path)
            if not model_path.is_absolute():
                model_path = self.project_root / model_path

        self.model_path = Path(model_path).resolve()

        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.device = str(device)

        # For this project, class 0 is IATA_TAG.
        if classes is None:
            self.classes = [self.IATA_TAG_CLASS_ID]
        else:
            self.classes = [int(x) for x in classes]

        self.max_det = int(max_det)
        self.verbose = bool(verbose)

        self.model: Optional[YOLO] = None
        self.class_names: Dict[int, str] = {}

        # ------------------------------------------------------------
        # Shared-model thread safety.
        #
        # config.yaml's performance.shared_model=true means ONE
        # YOLODetector instance is handed to every enabled camera's
        # pipeline thread (up to performance.max_camera_threads). If
        # two threads call self.model.predict(...) at the same
        # moment, they are mutating the same underlying model/tensor
        # state from two threads at once -- on CPU this can silently
        # corrupt or drop a result rather than raising, which looks
        # exactly like "detections just aren't showing up" with no
        # error anywhere. This lock serializes inference calls across
        # every thread sharing this detector, exactly as the FillPac
        # AI project already does for its shared detector.
        # ------------------------------------------------------------
        self._inference_lock = threading.Lock()

        self._load_model()

    # ================================================================
    # MODEL LOADING
    # ================================================================

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                "TRAINED YOLO model not found.\n"
                f"Expected: {self.model_path}\n\n"
                "Copy the trained best.pt to:\n"
                f"{self.project_root / self.DEFAULT_TRAINED_MODEL}"
            )

        print()
        print("=" * 72)
        print(" OCR_BHS - TRAINED YOLO26 IATA DETECTOR")
        print("=" * 72)
        print(f"Model      : {self.model_path}")
        print(f"Device     : {self.device}")
        print(f"Confidence : {self.confidence}")
        print(f"Image size : {self.image_size}")
        print(f"Classes    : {self.classes}")
        print(f"Max det    : {self.max_det}")
        print()

        # This is the ONLY model loaded by this detector.
        self.model = YOLO(str(self.model_path))

        try:
            names = self.model.names

            if isinstance(names, dict):
                self.class_names = {
                    int(k): str(v)
                    for k, v in names.items()
                }
            elif isinstance(names, list):
                self.class_names = {
                    i: str(name)
                    for i, name in enumerate(names)
                }
        except Exception:
            self.class_names = {}

        print(f"[YOLO] Classes: {self.class_names}")

        # ------------------------------------------------------------
        # Sanity check: warn (don't crash) if class 0's name doesn't
        # look like an IATA tag class. Model class-name strings vary
        # by training run (e.g. "iata_tag", "IATA_TAG", "tag"), so we
        # match loosely instead of requiring one exact string. What
        # actually matters for filtering is the class ID (0), which
        # is enforced everywhere detections/tracks are built.
        # ------------------------------------------------------------
        class_zero = self.get_class_name(self.IATA_TAG_CLASS_ID)
        normalized = class_zero.strip().lower().replace(" ", "_")

        if normalized not in ("iata_tag", "tag", "iata_baggage_tag"):
            print(
                "[YOLO] WARNING: class 0 name from the model is "
                f"'{class_zero}', which doesn't look like an IATA "
                "tag class. Continuing anyway, but double-check "
                f"this is the right model: {self.model_path}"
            )
        else:
            print(f"[YOLO] VERIFIED: class 0 = '{class_zero}'")
        print("[YOLO] Trained IATA model loaded successfully.")
        print("=" * 72)
        print()

    # ================================================================
    # DETECTION
    # ================================================================

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLO detection on one OpenCV BGR frame.

        Returns ONLY class-0 IATA_TAG detections.
        No tracking IDs are generated here.
        """

        self._validate_frame(frame)

        if self.model is None:
            raise RuntimeError("YOLO model is not loaded.")

        # Serialize inference across camera threads sharing this
        # detector (see the lock's docstring in __init__).
        with self._inference_lock:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                imgsz=self.image_size,
                device=self.device,
                classes=self.classes,
                max_det=self.max_det,
                verbose=self.verbose,
            )

        if not results:
            return []

        return self._parse_results(results[0])

    # Backward compatibility
    predict = detect

    # ================================================================
    # RESULT PARSING
    # ================================================================

    def _parse_results(self, result: Any) -> List[Detection]:

        if result is None or result.boxes is None:
            return []

        boxes = result.boxes

        if len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)

        detections: List[Detection] = []

        for bbox, confidence, class_id in zip(
            xyxy,
            confidences,
            class_ids,
        ):

            # --------------------------------------------------------
            # Only trained IATA_TAG class.
            # --------------------------------------------------------
            if int(class_id) != self.IATA_TAG_CLASS_ID:
                continue

            x1, y1, x2, y2 = map(float, bbox)

            # --------------------------------------------------------
            # Basic bbox validation.
            # --------------------------------------------------------
            if x2 <= x1 or y2 <= y1:
                continue

            # Clip coordinates to image-independent valid values.
            detection = Detection(
                class_id=int(class_id),
                class_name=self.get_class_name(class_id),
                confidence=float(confidence),
                bbox=(x1, y1, x2, y2),
            )

            detections.append(detection)

        # Highest-confidence detection first.
        detections.sort(
            key=lambda d: d.confidence,
            reverse=True,
        )

        return detections

    # ================================================================
    # FILTERING
    # ================================================================

    def filter_detections(
        self,
        detections: List[Detection],
        min_confidence: Optional[float] = None,
        allowed_classes: Optional[Sequence[int]] = None,
    ) -> List[Detection]:

        threshold = (
            self.confidence
            if min_confidence is None
            else float(min_confidence)
        )

        allowed = (
            set(int(x) for x in allowed_classes)
            if allowed_classes is not None
            else None
        )

        return [
            detection
            for detection in detections
            if detection.confidence >= threshold
            and (
                allowed is None
                or detection.class_id in allowed
            )
        ]

    # ================================================================
    # ROI FILTER
    # ================================================================

    @staticmethod
    def filter_by_roi(
        detections: List[Detection],
        roi: tuple[int, int, int, int],
        require_center_inside: bool = True,
    ) -> List[Detection]:

        rx1, ry1, rx2, ry2 = roi
        filtered: List[Detection] = []

        for detection in detections:

            if require_center_inside:
                cx, cy = detection.center

                if (
                    rx1 <= cx <= rx2
                    and ry1 <= cy <= ry2
                ):
                    filtered.append(detection)

            else:
                x1, y1, x2, y2 = detection.as_xyxy_int

                if (
                    x2 >= rx1
                    and x1 <= rx2
                    and y2 >= ry1
                    and y1 <= ry2
                ):
                    filtered.append(detection)

        return filtered

    # ================================================================
    # DRAW
    # ================================================================

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_confidence: bool = True,
        show_center: bool = True,
    ) -> np.ndarray:

        output = frame.copy()

        for index, detection in enumerate(
            detections,
            start=1,
        ):

            x1, y1, x2, y2 = detection.as_xyxy_int

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = detection.class_name

            if show_confidence:
                label += f" {detection.confidence:.2f}"

            cv2.putText(
                output,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if show_center:
                cx, cy = detection.center

                cv2.circle(
                    output,
                    (int(cx), int(cy)),
                    4,
                    (0, 0, 255),
                    -1,
                )

        return output

    # ================================================================
    # WARMUP
    # ================================================================

    def warmup(
        self,
        width: int = 1280,
        height: int = 720,
    ) -> None:

        dummy_frame = np.zeros(
            (height, width, 3),
            dtype=np.uint8,
        )

        print("[YOLO] Warming up trained model...")

        self.detect(dummy_frame)

        print("[YOLO] Warmup complete.")

    # ================================================================
    # INFORMATION
    # ================================================================

    def get_class_name(self, class_id: int) -> str:

        return self.class_names.get(
            int(class_id),
            str(class_id),
        )

    def get_model_info(self) -> Dict[str, Any]:

        return {
            "model_path": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "trained_model": True,
            "confidence": self.confidence,
            "image_size": self.image_size,
            "device": self.device,
            "classes_filter": self.classes,
            "max_det": self.max_det,
            "class_names": self.class_names,
            "target_class_id": self.IATA_TAG_CLASS_ID,
            "target_class_name": self.get_class_name(
                self.IATA_TAG_CLASS_ID
            ),
        }

    # ================================================================
    # FRAME VALIDATION
    # ================================================================

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:

        if frame is None:
            raise ValueError("Frame is None.")

        if not isinstance(frame, np.ndarray):
            raise TypeError(
                "Frame must be numpy.ndarray, "
                f"got {type(frame)}"
            )

        if frame.size == 0:
            raise ValueError("Frame is empty.")

        if frame.ndim != 3:
            raise ValueError(
                f"Expected BGR frame with 3 dimensions, "
                f"got shape={frame.shape}"
            )

        if frame.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel BGR frame, "
                f"got {frame.shape[2]} channels"
            )


# ====================================================================
# STANDALONE TEST
# ====================================================================

def main() -> int:

    project_root = Path(__file__).resolve().parents[2]

    print()
    print("=" * 72)
    print(" OCR_BHS - TRAINED YOLO26 DETECTOR TEST")
    print("=" * 72)

    # Explicitly use the trained model.
    model_path = (
        project_root
        / "models"
        / "yolo"
        / "iata_tag_yolo26n_best.pt"
    )

    print(f"Expected trained model:")
    print(f"  {model_path}")

    if not model_path.exists():
        print()
        print("[FAIL] Trained model does not exist.")
        return 1

    try:
        detector = YOLODetector(
            model_path=model_path,
            confidence=0.10,
            image_size=832,
            device="cpu",
            classes=[0],
            max_det=20,
        )
    except Exception as exc:
        print()
        print("[FAIL] Detector initialization failed.")
        print(f"Type : {type(exc).__name__}")
        print(f"Error: {exc}")
        return 1

    print()
    print("MODEL INFORMATION")
    print("-" * 72)

    for key, value in detector.get_model_info().items():
        print(f"{key:20}: {value}")

    # ------------------------------------------------------------
    # Test real image
    # ------------------------------------------------------------

    image_path = (
        project_root
        / "captures"
        / "test_tag_01.jpg"
    )

    if image_path.exists():

        print()
        print("=" * 72)
        print("REAL IATA TAG IMAGE TEST")
        print("=" * 72)

        image = cv2.imread(str(image_path))

        if image is None:
            print("[FAIL] Could not read test image.")
            return 1

        print(
            f"Image: "
            f"{image.shape[1]} x {image.shape[0]}"
        )

        detections = detector.detect(image)

        print(
            f"Detections: {len(detections)}"
        )

        for index, detection in enumerate(
            detections,
            start=1,
        ):
            print(
                f"{index:02d} | "
                f"{detection.class_name} | "
                f"conf={detection.confidence:.4f} | "
                f"bbox={detection.bbox}"
            )

        output = detector.draw_detections(
            image,
            detections,
        )

        output_path = (
            project_root
            / "debug"
            / "yolo_trained_test.jpg"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.imwrite(
            str(output_path),
            output,
        )

        print(f"Visual result: {output_path}")

    else:
        print()
        print(
            f"[INFO] Test image not found:\n"
            f"{image_path}"
        )

    print()
    print("=" * 72)
    print("TRAINED YOLO26 TEST COMPLETE")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())