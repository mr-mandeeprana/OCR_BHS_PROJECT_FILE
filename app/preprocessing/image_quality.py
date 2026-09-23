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
    Analyze and preprocess cropped IATA tag images.

    Responsibilities:
        - brightness measurement
        - contrast measurement
        - sharpness measurement
        - quality scoring
        - structural validation
        - OCR-safe preprocessing
        - optional CLAHE
        - optional gamma correction
        - optional bilateral denoising
        - optional mild sharpening
        - controlled upscaling

    This class does NOT perform OCR.
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

        # Enhancement configuration
        enable_clahe: bool = True,
        enable_gamma: bool = True,
        enable_denoise: bool = True,
        enable_sharpening: bool = True,

        clahe_clip_limit: float = 2.0,
        clahe_grid_size: tuple[int, int] = (8, 8),

        bilateral_d: int = 5,
        bilateral_sigma_color: float = 35.0,
        bilateral_sigma_space: float = 35.0,

        sharpen_amount: float = 0.8,
        gamma_dark: float = 1.20,
        gamma_bright: float = 0.85,
    ) -> None:

        self.min_width = int(min_width)
        self.min_height = int(min_height)

        self.min_brightness = float(min_brightness)
        self.max_brightness = float(max_brightness)

        self.min_contrast = float(min_contrast)
        self.min_sharpness = float(min_sharpness)

        self.ocr_scale = max(
            1.0,
            float(ocr_scale),
        )

        # --------------------------------------------------------------
        # Enhancement settings
        # --------------------------------------------------------------

        self.enable_clahe = bool(
            enable_clahe
        )

        self.enable_gamma = bool(
            enable_gamma
        )

        self.enable_denoise = bool(
            enable_denoise
        )

        self.enable_sharpening = bool(
            enable_sharpening
        )

        self.clahe_clip_limit = max(
            0.1,
            float(clahe_clip_limit),
        )

        self.clahe_grid_size = (
            int(clahe_grid_size[0]),
            int(clahe_grid_size[1]),
        )

        self.bilateral_d = max(
            1,
            int(bilateral_d),
        )

        self.bilateral_sigma_color = max(
            1.0,
            float(bilateral_sigma_color),
        )

        self.bilateral_sigma_space = max(
            1.0,
            float(bilateral_sigma_space),
        )

        self.sharpen_amount = max(
            0.0,
            float(sharpen_amount),
        )

        self.gamma_dark = max(
            0.1,
            float(gamma_dark),
        )

        self.gamma_bright = max(
            0.1,
            float(gamma_bright),
        )

    # ==================================================================
    # INTERNAL HELPERS
    # ==================================================================

    @staticmethod
    def _valid_image(
        image: Optional[np.ndarray],
    ) -> bool:

        return (
            image is not None
            and isinstance(image, np.ndarray)
            and image.size > 0
            and image.ndim in (2, 3)
        )

    @staticmethod
    def _to_gray(
        image: np.ndarray,
    ) -> np.ndarray:

        if image.ndim == 2:
            return image

        if image.shape[2] == 1:
            return image[:, :, 0]

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    # ==================================================================
    # QUALITY CHECK
    # ==================================================================

    def check(
        self,
        image: Optional[np.ndarray],
    ) -> ImageQualityResult:
        """
        Calculate quality metrics on an image.

        The quality check itself does not modify the image.
        """

        if not self._valid_image(image):

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

        height, width = image.shape[:2]

        try:

            gray = self._to_gray(image)

            gray = np.asarray(
                gray,
                dtype=np.uint8,
            )

            brightness = float(
                np.mean(gray)
            )

            contrast = float(
                np.std(gray)
            )

            sharpness = float(
                cv2.Laplacian(
                    gray,
                    cv2.CV_64F,
                ).var()
            )

        except Exception:

            return ImageQualityResult(
                width=int(width),
                height=int(height),
                brightness=0.0,
                contrast=0.0,
                sharpness=0.0,
                quality_score=0.0,
                is_valid=False,
                issues=[
                    "QUALITY_ANALYSIS_FAILED"
                ],
            )

        # --------------------------------------------------------------
        # Issues
        # --------------------------------------------------------------

        issues: list[str] = []

        if brightness < self.min_brightness:
            issues.append(
                "LOW_LIGHT"
            )

        if brightness > self.max_brightness:
            issues.append(
                "OVEREXPOSURE"
            )

        if contrast < self.min_contrast:
            issues.append(
                "LOW_CONTRAST"
            )

        if sharpness < self.min_sharpness:
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

        # Small images are structurally invalid.
        # Other quality issues remain usable because
        # preprocessing may improve them.
        is_valid = (
            width >= self.min_width
            and height >= self.min_height
        )

        return ImageQualityResult(
            width=int(width),
            height=int(height),
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            quality_score=score,
            is_valid=is_valid,
            issues=issues,
        )

    # ==================================================================
    # GAMMA CORRECTION
    # ==================================================================

    @staticmethod
    def _gamma_correct(
        image: np.ndarray,
        gamma: float,
    ) -> np.ndarray:
        """
        Apply gamma correction.

        gamma > 1:
            darkens image

        gamma < 1:
            brightens image
        """

        gamma = max(
            0.1,
            float(gamma),
        )

        inverse_gamma = 1.0 / gamma

        table = np.array(
            [
                (
                    (i / 255.0)
                    ** inverse_gamma
                )
                * 255.0
                for i in range(256)
            ],
            dtype=np.uint8,
        )

        return cv2.LUT(
            image,
            table,
        )

    def apply_gamma_if_needed(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply gamma correction only when brightness
        indicates a useful correction is needed.
        """

        if not self.enable_gamma:
            return image.copy()

        if not self._valid_image(image):
            return image

        quality = self.check(image)

        if quality.brightness < self.min_brightness:

            return self._gamma_correct(
                image,
                self.gamma_dark,
            )

        if quality.brightness > self.max_brightness:

            return self._gamma_correct(
                image,
                self.gamma_bright,
            )

        return image.copy()

    # ==================================================================
    # CLAHE
    # ==================================================================

    def apply_clahe(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply CLAHE to luminance only.

        This preserves the original BGR structure instead of
        converting the whole output permanently to grayscale.
        """

        if not self.enable_clahe:
            return image.copy()

        if not self._valid_image(image):
            return image

        try:

            if image.ndim == 2:

                clahe = cv2.createCLAHE(
                    clipLimit=self.clahe_clip_limit,
                    tileGridSize=self.clahe_grid_size,
                )

                return clahe.apply(
                    image
                )

            lab = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2LAB,
            )

            l_channel, a_channel, b_channel = cv2.split(
                lab
            )

            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_grid_size,
            )

            l_channel = clahe.apply(
                l_channel
            )

            enhanced_lab = cv2.merge(
                (
                    l_channel,
                    a_channel,
                    b_channel,
                )
            )

            return cv2.cvtColor(
                enhanced_lab,
                cv2.COLOR_LAB2BGR,
            )

        except Exception:

            return image.copy()

    # ==================================================================
    # BILATERAL DENOISING
    # ==================================================================

    def apply_denoise(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Mild bilateral filtering.

        Designed to reduce camera noise while preserving
        text and barcode edges.
        """

        if not self.enable_denoise:
            return image.copy()

        if not self._valid_image(image):
            return image

        try:

            return cv2.bilateralFilter(
                image,
                self.bilateral_d,
                self.bilateral_sigma_color,
                self.bilateral_sigma_space,
            )

        except Exception:

            return image.copy()

    # ==================================================================
    # SHARPENING
    # ==================================================================

    def apply_sharpening(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Mild unsharp masking.

        Avoids the very aggressive 5x center sharpening kernel.
        """

        if not self.enable_sharpening:
            return image.copy()

        if not self._valid_image(image):
            return image

        try:

            blurred = cv2.GaussianBlur(
                image,
                (0, 0),
                1.0,
            )

            amount = self.sharpen_amount

            sharpened = cv2.addWeighted(
                image,
                1.0 + amount,
                blurred,
                -amount,
                0,
            )

            return sharpened

        except Exception:

            return image.copy()

    # ==================================================================
    # UPSCALING
    # ==================================================================

    def upscale(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Controlled OCR upscaling.
        """

        if not self._valid_image(image):
            raise ValueError(
                "Cannot preprocess empty image."
            )

        if self.ocr_scale <= 1.0:
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

    # ==================================================================
    # OCR PREPROCESSING
    # ==================================================================

    def prepare_for_ocr(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Complete OCR preprocessing pipeline.

        Pipeline:

            input
              ↓
            conditional gamma
              ↓
            CLAHE
              ↓
            bilateral denoise
              ↓
            mild sharpening
              ↓
            2x upscale
        """

        if not self._valid_image(image):
            raise ValueError(
                "Cannot preprocess empty image."
            )

        output = image.copy()

        # --------------------------------------------------------------
        # 1. Gamma
        # --------------------------------------------------------------

        output = self.apply_gamma_if_needed(
            output
        )

        # --------------------------------------------------------------
        # 2. CLAHE
        # --------------------------------------------------------------

        output = self.apply_clahe(
            output
        )

        # --------------------------------------------------------------
        # 3. Denoise
        # --------------------------------------------------------------

        output = self.apply_denoise(
            output
        )

        # --------------------------------------------------------------
        # 4. Sharpen
        # --------------------------------------------------------------

        output = self.apply_sharpening(
            output
        )

        # --------------------------------------------------------------
        # 5. Upscale
        # --------------------------------------------------------------

        output = self.upscale(
            output
        )

        return output

    # ==================================================================
    # OCR VARIANTS
    # ==================================================================

    def get_ocr_candidates(
        self,
        image: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Generate conservative OCR candidates.

        Candidate 1:
            enhanced color/BGR image

        Candidate 2:
            grayscale

        Candidate 3:
            adaptive threshold

        Candidate 4:
            Otsu threshold

        The original enhanced image remains first because
        thresholding can remove useful information.
        """

        enhanced = self.prepare_for_ocr(
            image
        )

        candidates: list[np.ndarray] = [
            enhanced
        ]

        gray = self._to_gray(
            enhanced
        )

        candidates.append(
            gray
        )

        try:

            adaptive = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11,
            )

            candidates.append(
                adaptive
            )

        except Exception:
            pass

        try:

            _, otsu = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY
                + cv2.THRESH_OTSU,
            )

            candidates.append(
                otsu
            )

        except Exception:
            pass

        return candidates

    # ==================================================================
    # DIAGNOSTICS
    # ==================================================================

    def diagnostics(self) -> dict:
        """
        Return current preprocessing configuration.
        """

        return {
            "min_width": self.min_width,
            "min_height": self.min_height,
            "min_brightness": self.min_brightness,
            "max_brightness": self.max_brightness,
            "min_contrast": self.min_contrast,
            "min_sharpness": self.min_sharpness,
            "ocr_scale": self.ocr_scale,
            "enable_clahe": self.enable_clahe,
            "enable_gamma": self.enable_gamma,
            "enable_denoise": self.enable_denoise,
            "enable_sharpening": self.enable_sharpening,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_grid_size": self.clahe_grid_size,
            "bilateral_d": self.bilateral_d,
            "bilateral_sigma_color": self.bilateral_sigma_color,
            "bilateral_sigma_space": self.bilateral_sigma_space,
            "sharpen_amount": self.sharpen_amount,
            "gamma_dark": self.gamma_dark,
            "gamma_bright": self.gamma_bright,
        }


# ======================================================================
# STANDALONE TEST
# ======================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("OCR_BHS IMAGE QUALITY + PREPROCESSING TEST")
    print("=" * 72)

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
        (40, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (20, 20, 20),
        3,
        cv2.LINE_AA,
    )

    checker = ImageQualityChecker(
        ocr_scale=2.0,
        enable_clahe=True,
        enable_gamma=True,
        enable_denoise=True,
        enable_sharpening=True,
    )

    print("\nConfiguration:")

    for key, value in checker.diagnostics().items():
        print(
            f"  {key}: {value}"
        )

    quality = checker.check(
        image
    )

    print("\nOriginal image:")
    print(
        f"  Size       : "
        f"{quality.width} x {quality.height}"
    )

    print(
        f"  Brightness : "
        f"{quality.brightness:.2f}"
    )

    print(
        f"  Contrast   : "
        f"{quality.contrast:.2f}"
    )

    print(
        f"  Sharpness  : "
        f"{quality.sharpness:.2f}"
    )

    print(
        f"  Score      : "
        f"{quality.quality_score:.2f}"
    )

    print(
        f"  Valid      : "
        f"{quality.is_valid}"
    )

    print(
        f"  Issues     : "
        f"{quality.issues}"
    )

    enhanced = checker.prepare_for_ocr(
        image
    )

    print("\nEnhanced image:")
    print(
        f"  Size       : "
        f"{enhanced.shape[1]} x "
        f"{enhanced.shape[0]}"
    )

    print(
        f"  Channels   : "
        f"{enhanced.shape[2] if enhanced.ndim == 3 else 1}"
    )

    candidates = checker.get_ocr_candidates(
        image
    )

    print(
        f"\nOCR candidates: "
        f"{len(candidates)}"
    )

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        if candidate.ndim == 2:

            channels = 1

        else:

            channels = candidate.shape[2]

        print(
            f"  Candidate {index}: "
            f"{candidate.shape[1]} x "
            f"{candidate.shape[0]} "
            f"channels={channels}"
        )

    print("\n[PASS] Image quality preprocessing test completed.")