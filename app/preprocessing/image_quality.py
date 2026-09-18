from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class ImageQualityResult:
    """
    Quality measurements for an IATA tag image.
    """

    width: int
    height: int

    brightness: float
    contrast: float
    sharpness: float

    quality_score: float

    is_valid: bool

    issues: list[str]


class ImageQualityChecker:
    """
    Analyze the quality of a cropped IATA tag image.

    This class does not perform OCR.

    It provides:
        - brightness
        - contrast
        - sharpness
        - quality score
        - structural validity
        - quality issue information

    It also provides conservative OCR preprocessing.
    """

    def __init__(
        self,
        min_width: int = 100,
        min_height: int = 40,
        min_brightness: float = 35.0,
        max_brightness: float = 225.0,
        min_contrast: float = 20.0,
        min_sharpness: float = 50.0,
        ocr_scale: float = 2.0,
    ) -> None:

        self.min_width = int(
            min_width
        )

        self.min_height = int(
            min_height
        )

        self.min_brightness = float(
            min_brightness
        )

        self.max_brightness = float(
            max_brightness
        )

        self.min_contrast = float(
            min_contrast
        )

        self.min_sharpness = float(
            min_sharpness
        )

        self.ocr_scale = max(
            1.0,
            float(ocr_scale),
        )

    # ==================================================================
    # QUALITY CHECK
    # ==================================================================

    def check(
        self,
        image: Optional[np.ndarray],
    ) -> ImageQualityResult:
        """
        Calculate quality metrics on the original image.
        """

        # --------------------------------------------------------------
        # Empty image
        # --------------------------------------------------------------

        if (
            image is None
            or image.size == 0
        ):

            return ImageQualityResult(
                width=0,
                height=0,
                brightness=0.0,
                contrast=0.0,
                sharpness=0.0,
                quality_score=0.0,
                is_valid=False,
                issues=[
                    "EMPTY_IMAGE"
                ],
            )

        # --------------------------------------------------------------
        # Dimensions
        # --------------------------------------------------------------

        height, width = image.shape[:2]

        # --------------------------------------------------------------
        # Grayscale
        # --------------------------------------------------------------

        if image.ndim == 3:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

        else:

            gray = image

        gray = np.asarray(
            gray,
            dtype=np.uint8,
        )

        # --------------------------------------------------------------
        # Brightness
        # --------------------------------------------------------------

        brightness = float(
            np.mean(gray)
        )

        # --------------------------------------------------------------
        # Contrast
        # --------------------------------------------------------------

        contrast = float(
            np.std(gray)
        )

        # --------------------------------------------------------------
        # Sharpness
        #
        # Variance of Laplacian is a common blur indicator.
        # Higher value = generally sharper image.
        # --------------------------------------------------------------

        sharpness = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

        # --------------------------------------------------------------
        # Issues
        # --------------------------------------------------------------

        issues: list[str] = []

        if (
            brightness
            < self.min_brightness
        ):

            issues.append(
                "LOW_LIGHT"
            )

        if (
            brightness
            > self.max_brightness
        ):

            issues.append(
                "OVEREXPOSURE"
            )

        if (
            contrast
            < self.min_contrast
        ):

            issues.append(
                "LOW_CONTRAST"
            )

        if (
            sharpness
            < self.min_sharpness
        ):

            issues.append(
                "BLUR"
            )

        if (
            width < self.min_width
            or height < self.min_height
        ):

            issues.append(
                "TAG_TOO_SMALL"
            )

        # --------------------------------------------------------------
        # Quality score
        # --------------------------------------------------------------

        score = 100.0

        if "LOW_LIGHT" in issues:
            score -= 25.0

        if "OVEREXPOSURE" in issues:
            score -= 25.0

        if "LOW_CONTRAST" in issues:
            score -= 20.0

        if "BLUR" in issues:
            score -= 25.0

        if "TAG_TOO_SMALL" in issues:
            score -= 20.0

        score = max(
            0.0,
            min(
                100.0,
                score,
            ),
        )

        # --------------------------------------------------------------
        # Structural validity
        # --------------------------------------------------------------

        is_valid = (
            width >= self.min_width
            and height >= self.min_height
        )

        return ImageQualityResult(
            width=width,
            height=height,
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            quality_score=score,
            is_valid=is_valid,
            issues=issues,
        )

    # ==================================================================
    # OCR PREPROCESSING
    # ==================================================================

    def prepare_for_ocr(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Conservative preprocessing for OCR.

        Current strategy:
            original crop
                ↓
            2x upscale
                ↓
            OCR

        We intentionally avoid aggressive thresholding here because
        the OCR engine already performs its own processing/rotations.
        """

        if (
            image is None
            or image.size == 0
        ):

            raise ValueError(
                "Cannot preprocess empty image."
            )

        # --------------------------------------------------------------
        # No scaling requested
        # --------------------------------------------------------------

        if self.ocr_scale == 1.0:

            return image.copy()

        height, width = image.shape[:2]

        target_width = max(
            width + 1,
            int(
                round(
                    width
                    * self.ocr_scale
                )
            ),
        )

        target_height = max(
            height + 1,
            int(
                round(
                    height
                    * self.ocr_scale
                )
            ),
        )

        return cv2.resize(
            image,
            (
                target_width,
                target_height,
            ),
            interpolation=cv2.INTER_CUBIC,
        )


# ======================================================================
# STANDALONE TEST
# ======================================================================

if __name__ == "__main__":

    print("=" * 72)
    print(
        "OCR_BHS IMAGE QUALITY TEST"
    )
    print("=" * 72)

    # --------------------------------------------------------------
    # Create synthetic test image
    # --------------------------------------------------------------

    image = np.full(
        (
            240,
            540,
            3,
        ),
        180,
        dtype=np.uint8,
    )

    cv2.putText(
        image,
        "IATA TEST TAG",
        (
            40,
            125,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (
            20,
            20,
            20,
        ),
        3,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------------
    # Checker
    # --------------------------------------------------------------

    checker = ImageQualityChecker(
        ocr_scale=2.0
    )

    # --------------------------------------------------------------
    # Quality
    # --------------------------------------------------------------

    quality = checker.check(
        image
    )

    print()
    print(
        f"Width       : {quality.width}"
    )

    print(
        f"Height      : {quality.height}"
    )

    print(
        f"Brightness  : "
        f"{quality.brightness:.2f}"
    )

    print(
        f"Contrast    : "
        f"{quality.contrast:.2f}"
    )

    print(
        f"Sharpness   : "
        f"{quality.sharpness:.2f}"
    )

    print(
        f"Score       : "
        f"{quality.quality_score:.2f}"
    )

    print(
        f"Valid       : "
        f"{quality.is_valid}"
    )

    print(
        f"Issues      : "
        f"{quality.issues}"
    )

    # --------------------------------------------------------------
    # OCR preprocessing
    # --------------------------------------------------------------

    prepared = checker.prepare_for_ocr(
        image
    )

    print()
    print(
        f"Original OCR image : "
        f"{image.shape[1]} x "
        f"{image.shape[0]}"
    )

    print(
        f"Prepared OCR image : "
        f"{prepared.shape[1]} x "
        f"{prepared.shape[0]}"
    )

    expected_width = (
        image.shape[1] * 2
    )

    expected_height = (
        image.shape[0] * 2
    )

    if (
        prepared.shape[1]
        != expected_width
        or prepared.shape[0]
        != expected_height
    ):

        print(
            "[FAIL] 2x preprocessing "
            "test failed."
        )

        raise SystemExit(1)

    print()
    print(
        "[PASS] Image quality checker"
    )

    print(
        "[PASS] OCR preprocessing"
    )

    print()
    print("=" * 72)
    print(
        "IMAGE QUALITY TEST PASSED"
    )
    print("=" * 72)