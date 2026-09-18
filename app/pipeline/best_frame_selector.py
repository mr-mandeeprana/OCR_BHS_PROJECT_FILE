from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.models.detection import Detection
from app.preprocessing.image_quality import (
    ImageQualityChecker,
    ImageQualityResult,
)


@dataclass
class FrameCandidate:
    """
    Best-frame candidate containing ONLY the cropped IATA tag image.
    """

    frame_id: int

    frame: np.ndarray

    detection: Detection

    quality: ImageQualityResult

    score: float

    timestamp: Optional[float] = None


class BestFrameSelector:
    """
    Select the strongest cropped IATA-tag image from multiple
    observations of the same tracked tag.

    The selector NEVER sends the full camera frame downstream.

    Pipeline:

        camera frame
             ↓
        detection bbox
             ↓
        safe tag crop
             ↓
        quality measurement
             ↓
        candidate score
             ↓
        best tag crop
    """

    def __init__(
        self,
        quality_checker: Optional[ImageQualityChecker] = None,
        max_candidates: int = 5,
        crop_padding: float = 0.10,
    ):
        self.quality_checker = (
            quality_checker
            or ImageQualityChecker()
        )

        self.max_candidates = max(
            1,
            int(max_candidates),
        )

        self.crop_padding = max(
            0.0,
            float(crop_padding),
        )

        self._candidates: list[FrameCandidate] = []

    def reset(self) -> None:
        """Remove all stored candidates."""

        self._candidates.clear()

    @staticmethod
    def crop_detection(
        frame: np.ndarray,
        detection: Detection,
        padding: float = 0.10,
    ) -> Optional[np.ndarray]:
        """
        Crop the detected IATA tag from the camera frame.

        A small configurable padding is added around the YOLO bbox
        so OCR does not lose characters at the tag boundary.
        """

        if frame is None or frame.size == 0:
            return None

        if detection is None:
            return None

        height, width = frame.shape[:2]

        if width <= 0 or height <= 0:
            return None

        x1, y1, x2, y2 = detection.bbox

        box_width = max(
            1.0,
            x2 - x1,
        )

        box_height = max(
            1.0,
            y2 - y1,
        )

        pad_x = box_width * padding
        pad_y = box_height * padding

        x1 = int(
            max(
                0,
                np.floor(x1 - pad_x),
            )
        )

        y1 = int(
            max(
                0,
                np.floor(y1 - pad_y),
            )
        )

        x2 = int(
            min(
                width,
                np.ceil(x2 + pad_x),
            )
        )

        y2 = int(
            min(
                height,
                np.ceil(y2 + pad_y),
            )
        )

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:
            return None

        return crop.copy()

    @staticmethod
    def _crop_area_ratio(
        crop: np.ndarray,
    ) -> float:
        """
        Return crop area relative to the original image.

        This method is retained for compatibility but is not used
        as the primary quality metric because candidates are crops.
        """

        if crop is None or crop.size == 0:
            return 0.0

        height, width = crop.shape[:2]

        if width <= 0 or height <= 0:
            return 0.0

        return float(width * height)

    def calculate_score(
        self,
        crop: np.ndarray,
        detection: Detection,
        quality: ImageQualityResult,
    ) -> float:
        """
        Calculate a 0-100 candidate score.

        Quality is the dominant factor.
        Detection confidence is secondary.
        """

        detection_score = (
            max(
                0.0,
                min(
                    1.0,
                    float(detection.confidence),
                ),
            )
            * 100.0
        )

        quality_score = float(
            quality.quality_score
        )

        # Penalize candidates that are too small.
        size_score = min(
            100.0,
            (
                min(
                    crop.shape[:2]
                )
                / 200.0
            ) * 100.0,
        )

        score = (
            0.60 * quality_score
            + 0.25 * detection_score
            + 0.15 * size_score
        )

        return max(
            0.0,
            min(
                100.0,
                score,
            ),
        )

    def add_candidate(
        self,
        frame_id: int,
        frame: np.ndarray,
        detection: Detection,
        timestamp: Optional[float] = None,
    ) -> Optional[FrameCandidate]:
        """
        Crop the detection and add it as a candidate.

        IMPORTANT:
        `frame` is the full camera image.

        The stored FrameCandidate.frame is ONLY the IATA tag crop.
        """

        crop = self.crop_detection(
            frame=frame,
            detection=detection,
            padding=self.crop_padding,
        )

        if crop is None:
            return None

        quality = self.quality_checker.check(
            crop
        )

        score = self.calculate_score(
            crop=crop,
            detection=detection,
            quality=quality,
        )

        candidate = FrameCandidate(
            frame_id=frame_id,
            frame=crop,
            detection=detection,
            quality=quality,
            score=score,
            timestamp=timestamp,
        )

        self._candidates.append(
            candidate
        )

        self._candidates.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        self._candidates = self._candidates[
            : self.max_candidates
        ]

        return candidate

    def get_best(
        self,
    ) -> Optional[FrameCandidate]:

        if not self._candidates:
            return None

        return self._candidates[0]

    def get_candidates(
        self,
    ) -> list[FrameCandidate]:

        return list(
            self._candidates
        )


if __name__ == "__main__":

    print("=" * 72)
    print("BEST FRAME SELECTOR")
    print("=" * 72)

    # Synthetic image
    image = np.full(
        (720, 1280, 3),
        180,
        dtype=np.uint8,
    )

    # Simulated IATA tag
    cv2.rectangle(
        image,
        (400, 250),
        (850, 450),
        (255, 255, 255),
        -1,
    )

    detection = Detection(
        class_id=0,
        confidence=0.92,
        bbox=(
            400.0,
            250.0,
            850.0,
            450.0,
        ),
    )

    selector = BestFrameSelector(
        max_candidates=5,
        crop_padding=0.10,
    )

    candidate = selector.add_candidate(
        frame_id=1,
        frame=image,
        detection=detection,
    )

    if candidate is None:
        print("[FAIL] Candidate was not created.")
        raise SystemExit(1)

    crop_height, crop_width = (
        candidate.frame.shape[:2]
    )

    print()
    print(
        f"Original frame : "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    print(
        f"Tag crop       : "
        f"{crop_width} x {crop_height}"
    )

    print(
        f"Quality score  : "
        f"{candidate.quality.quality_score:.2f}"
    )

    print(
        f"Candidate score: "
        f"{candidate.score:.2f}"
    )

    print()

    if (
        crop_width < image.shape[1]
        and crop_height < image.shape[0]
    ):
        print(
            "[PASS] BestFrameSelector stores "
            "the cropped tag, not the full frame."
        )
    else:
        print(
            "[FAIL] Candidate still contains "
            "the full camera frame."
        )
        raise SystemExit(1)

    print("=" * 72)
    print("BEST FRAME SELECTOR TEST PASSED")
    print("=" * 72)
