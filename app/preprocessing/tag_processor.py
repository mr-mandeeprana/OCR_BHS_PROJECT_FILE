"""
OCR_BHS - IATA Tag Image Processor

Responsibilities:
- Safely crop detected IATA tags from camera frames
- Clamp bounding boxes to image boundaries
- Reject invalid/outside bounding boxes
- Add configurable padding
- Resize crops safely
- Preserve aspect ratio
- Prevent malformed / extreme image dimensions
- Apply controlled image enhancement
- Generate controlled rotations
- Normalize image format for OCR / barcode engines

This module does NOT perform YOLO detection.
This module does NOT perform OCR.
This module does NOT perform barcode decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.preprocessing.image_quality import (
    ImageQualityChecker,
)


# ============================================================================
# DATA MODEL
# ============================================================================


@dataclass
class TagCrop:
    """
    Represents a safely extracted IATA tag image.
    """

    image: np.ndarray
    bbox: Tuple[int, int, int, int]

    rotation: int = 0
    scale: float = 1.0

    source_width: int = 0
    source_height: int = 0

    @property
    def width(self) -> int:
        return int(
            self.image.shape[1]
        )

    @property
    def height(self) -> int:
        return int(
            self.image.shape[0]
        )

    @property
    def area(self) -> int:
        return (
            self.width
            * self.height
        )

    @property
    def aspect_ratio(self) -> float:

        if self.height <= 0:
            return 0.0

        return (
            self.width
            / float(self.height)
        )


# ============================================================================
# TAG PROCESSOR
# ============================================================================


class TagProcessor:
    """
    Safe image processor for detected IATA tags.

    Pipeline:

        camera frame
             ↓
        YOLO bbox
             ↓
        safe crop
             ↓
        padding
             ↓
        aspect-ratio validation
             ↓
        safe resize
             ↓
        image enhancement
             ↓
        OCR / barcode
    """

    def __init__(
        self,
        config=None,
    ):

        self.config = config

        # --------------------------------------------------------------
        # Crop configuration
        # --------------------------------------------------------------

        self.padding_ratio = 0.08

        self.min_width = 32
        self.min_height = 32

        self.max_width = 2000
        self.max_height = 1200
        self.max_side = 2500

        self.min_aspect_ratio = 0.20
        self.max_aspect_ratio = 8.0

        self.upscale_factor = 1.0

        # --------------------------------------------------------------
        # Enhancement configuration
        # --------------------------------------------------------------

        self.enable_clahe = False
        self.enable_sharpening = False

        self.enable_gamma = False
        self.enable_denoise = False

        self.clahe_clip_limit = 2.0
        self.clahe_grid_size = (
            8,
            8,
        )

        self.gamma_dark = 1.20
        self.gamma_bright = 0.85

        self.bilateral_d = 5
        self.bilateral_sigma_color = 35.0
        self.bilateral_sigma_space = 35.0

        self.sharpen_amount = 0.8

        # --------------------------------------------------------------
        # Image quality checker
        # --------------------------------------------------------------

        self.quality_checker = ImageQualityChecker(
            ocr_scale=2.0,
            enable_clahe=True,
            enable_gamma=True,
            enable_denoise=True,
            enable_sharpening=True,
        )

        self._load_config()

    # ==================================================================
    # CONFIG
    # ==================================================================

    def _load_config(self) -> None:
        """
        Load preprocessing configuration.

        Missing values fall back to safe defaults.
        """

        try:

            if self.config is None:
                return

            data = getattr(
                self.config,
                "data",
                self.config,
            )

            preprocessing = data.get(
                "preprocessing",
                {},
            )

            if not isinstance(
                preprocessing,
                dict,
            ):
                return

            # ----------------------------------------------------------
            # Crop settings
            # ----------------------------------------------------------

            self.padding_ratio = float(
                preprocessing.get(
                    "padding_ratio",
                    preprocessing.get(
                        "padding",
                        self.padding_ratio,
                    ),
                )
            )

            self.upscale_factor = float(
                preprocessing.get(
                    "upscale_factor",
                    preprocessing.get(
                        "scale",
                        self.upscale_factor,
                    ),
                )
            )

            self.min_width = int(
                preprocessing.get(
                    "min_width",
                    self.min_width,
                )
            )

            self.min_height = int(
                preprocessing.get(
                    "min_height",
                    self.min_height,
                )
            )

            self.max_width = int(
                preprocessing.get(
                    "max_width",
                    self.max_width,
                )
            )

            self.max_height = int(
                preprocessing.get(
                    "max_height",
                    self.max_height,
                )
            )

            self.max_side = int(
                preprocessing.get(
                    "max_side",
                    self.max_side,
                )
            )

            self.min_aspect_ratio = float(
                preprocessing.get(
                    "min_aspect_ratio",
                    self.min_aspect_ratio,
                )
            )

            self.max_aspect_ratio = float(
                preprocessing.get(
                    "max_aspect_ratio",
                    self.max_aspect_ratio,
                )
            )

            # ----------------------------------------------------------
            # Enhancement settings
            # ----------------------------------------------------------

            self.enable_clahe = bool(
                preprocessing.get(
                    "clahe",
                    self.enable_clahe,
                )
            )

            self.enable_sharpening = bool(
                preprocessing.get(
                    "sharpening",
                    self.enable_sharpening,
                )
            )

            self.enable_gamma = bool(
                preprocessing.get(
                    "gamma",
                    self.enable_gamma,
                )
            )

            self.enable_denoise = bool(
                preprocessing.get(
                    "denoise",
                    self.enable_denoise,
                )
            )

            # ----------------------------------------------------------
            # Nested image_quality config
            # ----------------------------------------------------------

            quality_config = preprocessing.get(
                "image_quality",
                {},
            )

            if isinstance(
                quality_config,
                dict,
            ):

                # Explicit nested config takes priority.
                self.enable_clahe = bool(
                    quality_config.get(
                        "clahe",
                        self.enable_clahe,
                    )
                )

                self.enable_gamma = bool(
                    quality_config.get(
                        "gamma",
                        self.enable_gamma,
                    )
                )

                self.enable_denoise = bool(
                    quality_config.get(
                        "denoise",
                        self.enable_denoise,
                    )
                )

                self.enable_sharpening = bool(
                    quality_config.get(
                        "sharpening",
                        self.enable_sharpening,
                    )
                )

                # Quality thresholds
                min_brightness = float(
                    quality_config.get(
                        "min_brightness",
                        self.quality_checker.min_brightness,
                    )
                )

                max_brightness = float(
                    quality_config.get(
                        "max_brightness",
                        self.quality_checker.max_brightness,
                    )
                )

                min_contrast = float(
                    quality_config.get(
                        "min_contrast",
                        self.quality_checker.min_contrast,
                    )
                )

                min_sharpness = float(
                    quality_config.get(
                        "min_sharpness",
                        self.quality_checker.min_sharpness,
                    )
                )

                self.quality_checker.min_brightness = (
                    min_brightness
                )

                self.quality_checker.max_brightness = (
                    max_brightness
                )

                self.quality_checker.min_contrast = (
                    min_contrast
                )

                self.quality_checker.min_sharpness = (
                    min_sharpness
                )

        except Exception:
            # Configuration must never stop the live pipeline.
            pass

        # --------------------------------------------------------------
        # Safety normalization
        # --------------------------------------------------------------

        self.padding_ratio = max(
            0.0,
            min(
                self.padding_ratio,
                0.50,
            ),
        )

        self.upscale_factor = max(
            0.25,
            min(
                self.upscale_factor,
                4.0,
            ),
        )

        self.min_width = max(
            8,
            self.min_width,
        )

        self.min_height = max(
            8,
            self.min_height,
        )

        self.max_width = max(
            self.min_width,
            self.max_width,
        )

        self.max_height = max(
            self.min_height,
            self.max_height,
        )

        self.max_side = max(
            self.max_width,
            self.max_height,
            self.max_side,
        )

        self.min_aspect_ratio = max(
            0.01,
            self.min_aspect_ratio,
        )

        self.max_aspect_ratio = max(
            self.min_aspect_ratio,
            self.max_aspect_ratio,
        )

        # --------------------------------------------------------------
        # Synchronize enhancement settings with quality checker
        # --------------------------------------------------------------

        self.quality_checker.enable_clahe = (
            self.enable_clahe
        )

        self.quality_checker.enable_gamma = (
            self.enable_gamma
        )

        self.quality_checker.enable_denoise = (
            self.enable_denoise
        )

        self.quality_checker.enable_sharpening = (
            self.enable_sharpening
        )

    # ==================================================================
    # VALIDATION
    # ==================================================================

    @staticmethod
    def _valid_image(
        image: Optional[np.ndarray],
    ) -> bool:

        return (
            image is not None
            and isinstance(
                image,
                np.ndarray,
            )
            and image.size > 0
            and image.ndim in (
                2,
                3,
            )
        )

    # ==================================================================
    # BBOX SANITIZATION
    # ==================================================================

    @staticmethod
    def _sanitize_bbox(
        bbox,
        frame_width: int,
        frame_height: int,
    ) -> Optional[
        Tuple[int, int, int, int]
    ]:

        if bbox is None:
            return None

        try:

            values = list(bbox)

            if len(values) != 4:
                return None

            x1, y1, x2, y2 = (
                float(values[0]),
                float(values[1]),
                float(values[2]),
                float(values[3]),
            )

        except Exception:

            return None

        if not all(
            np.isfinite(value)
            for value in (
                x1,
                y1,
                x2,
                y2,
            )
        ):
            return None

        # --------------------------------------------------------------
        # Normalize inverted boxes
        # --------------------------------------------------------------

        if x2 < x1:
            x1, x2 = x2, x1

        if y2 < y1:
            y1, y2 = y2, y1

        # --------------------------------------------------------------
        # Clamp
        # --------------------------------------------------------------

        x1 = int(
            max(
                0,
                min(
                    frame_width,
                    np.floor(x1),
                ),
            )
        )

        y1 = int(
            max(
                0,
                min(
                    frame_height,
                    np.floor(y1),
                ),
            )
        )

        x2 = int(
            max(
                0,
                min(
                    frame_width,
                    np.ceil(x2),
                ),
            )
        )

        y2 = int(
            max(
                0,
                min(
                    frame_height,
                    np.ceil(y2),
                ),
            )
        )

        if x2 <= x1:
            return None

        if y2 <= y1:
            return None

        return (
            x1,
            y1,
            x2,
            y2,
        )

    # ==================================================================
    # CROP
    # ==================================================================

    def crop(
        self,
        frame: np.ndarray,
        bbox,
        padding: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Safely crop an IATA tag from a camera frame.
        """

        if not self._valid_image(
            frame
        ):
            return None

        frame_height, frame_width = (
            frame.shape[:2]
        )

        safe_bbox = self._sanitize_bbox(
            bbox,
            frame_width,
            frame_height,
        )

        if safe_bbox is None:
            return None

        if padding is None:

            padding_ratio = (
                self.padding_ratio
            )

        else:

            try:
                padding_ratio = float(
                    padding
                )

            except Exception:
                padding_ratio = (
                    self.padding_ratio
                )

            padding_ratio = max(
                0.0,
                min(
                    padding_ratio,
                    0.50,
                ),
            )

        x1, y1, x2, y2 = (
            safe_bbox
        )

        width = x2 - x1
        height = y2 - y1

        pad_x = int(
            round(
                width
                * padding_ratio
            )
        )

        pad_y = int(
            round(
                height
                * padding_ratio
            )
        )

        x1 = max(
            0,
            x1 - pad_x,
        )

        y1 = max(
            0,
            y1 - pad_y,
        )

        x2 = min(
            frame_width,
            x2 + pad_x,
        )

        y2 = min(
            frame_height,
            y2 + pad_y,
        )

        if x2 <= x1:
            return None

        if y2 <= y1:
            return None

        crop = frame[
            y1:y2,
            x1:x2,
        ]

        if not self._valid_image(
            crop
        ):
            return None

        return crop.copy()

    # ==================================================================
    # SAFE RESIZE
    # ==================================================================

    def resize_safe(
        self,
        image: np.ndarray,
        upscale: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Resize while preserving aspect ratio.

        No output dimension can exceed configured limits.
        """

        if not self._valid_image(
            image
        ):
            return None

        height, width = (
            image.shape[:2]
        )

        if width <= 0 or height <= 0:
            return None

        if upscale is None:

            scale = (
                self.upscale_factor
            )

        else:

            try:
                scale = float(
                    upscale
                )

            except Exception:
                scale = (
                    self.upscale_factor
                )

        scale = max(
            0.25,
            min(
                scale,
                4.0,
            ),
        )

        target_width = max(
            1,
            int(
                round(
                    width * scale
                )
            ),
        )

        target_height = max(
            1,
            int(
                round(
                    height * scale
                )
            ),
        )

        # --------------------------------------------------------------
        # Maximum width
        # --------------------------------------------------------------

        if target_width > self.max_width:

            ratio = (
                self.max_width
                / float(target_width)
            )

            target_width = (
                self.max_width
            )

            target_height = max(
                1,
                int(
                    round(
                        target_height
                        * ratio
                    )
                ),
            )

        # --------------------------------------------------------------
        # Maximum height
        # --------------------------------------------------------------

        if target_height > self.max_height:

            ratio = (
                self.max_height
                / float(target_height)
            )

            target_height = (
                self.max_height
            )

            target_width = max(
                1,
                int(
                    round(
                        target_width
                        * ratio
                    )
                ),
            )

        # --------------------------------------------------------------
        # Maximum side
        # --------------------------------------------------------------

        largest_side = max(
            target_width,
            target_height,
        )

        if largest_side > self.max_side:

            ratio = (
                self.max_side
                / float(largest_side)
            )

            target_width = max(
                1,
                int(
                    round(
                        target_width
                        * ratio
                    )
                ),
            )

            target_height = max(
                1,
                int(
                    round(
                        target_height
                        * ratio
                    )
                ),
            )

        # --------------------------------------------------------------
        # Final limits
        # --------------------------------------------------------------

        target_width = min(
            target_width,
            self.max_width,
            self.max_side,
        )

        target_height = min(
            target_height,
            self.max_height,
            self.max_side,
        )

        # --------------------------------------------------------------
        # No resize required
        # --------------------------------------------------------------

        if (
            target_width == width
            and target_height == height
        ):
            return image.copy()

        # --------------------------------------------------------------
        # Interpolation
        # --------------------------------------------------------------

        if (
            target_width < width
            or target_height < height
        ):

            interpolation = (
                cv2.INTER_AREA
            )

        else:

            interpolation = (
                cv2.INTER_CUBIC
            )

        try:

            resized = cv2.resize(
                image,
                (
                    target_width,
                    target_height,
                ),
                interpolation=interpolation,
            )

        except Exception:

            return None

        if not self._valid_image(
            resized
        ):
            return None

        return resized

    # ==================================================================
    # ASPECT RATIO
    # ==================================================================

    def _reasonable_aspect_ratio(
        self,
        image: np.ndarray,
    ) -> bool:

        if not self._valid_image(
            image
        ):
            return False

        height, width = (
            image.shape[:2]
        )

        if height <= 0 or width <= 0:
            return False

        aspect = (
            width
            / float(height)
        )

        return (
            self.min_aspect_ratio
            <= aspect
            <= self.max_aspect_ratio
        )

    # ==================================================================
    # ROTATION
    # ==================================================================

    @staticmethod
    def rotate(
        image: np.ndarray,
        angle: int,
    ) -> Optional[np.ndarray]:

        if image is None:
            return None

        if image.size == 0:
            return None

        angle = (
            int(angle)
            % 360
        )

        if angle == 0:
            return image.copy()

        if angle == 90:

            return cv2.rotate(
                image,
                cv2.ROTATE_90_CLOCKWISE,
            )

        if angle == 180:

            return cv2.rotate(
                image,
                cv2.ROTATE_180,
            )

        if angle == 270:

            return cv2.rotate(
                image,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            )

        return None

    # ==================================================================
    # GRAYSCALE
    # ==================================================================

    @staticmethod
    def to_gray(
        image: np.ndarray,
    ) -> Optional[np.ndarray]:

        if not TagProcessor._valid_image(
            image
        ):
            return None

        if image.ndim == 2:
            return image.copy()

        try:

            return cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

        except Exception:

            return None

    # ==================================================================
    # GAMMA
    # ==================================================================

    @staticmethod
    def apply_gamma(
        image: np.ndarray,
        gamma: float,
    ) -> Optional[np.ndarray]:
        """
        Apply gamma correction while preserving channels.
        """

        if not TagProcessor._valid_image(
            image
        ):
            return None

        gamma = max(
            0.1,
            float(gamma),
        )

        try:

            inverse_gamma = (
                1.0 / gamma
            )

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

        except Exception:

            return image.copy()

    # ==================================================================
    # CLAHE
    # ==================================================================

    def apply_clahe(
        self,
        image: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Apply CLAHE to the luminance channel.

        BGR is preserved.
        """

        if not self._valid_image(
            image
        ):
            return None

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

            l_channel, a_channel, b_channel = (
                cv2.split(lab)
            )

            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_grid_size,
            )

            l_channel = clahe.apply(
                l_channel
            )

            merged = cv2.merge(
                (
                    l_channel,
                    a_channel,
                    b_channel,
                )
            )

            return cv2.cvtColor(
                merged,
                cv2.COLOR_LAB2BGR,
            )

        except Exception:

            return image.copy()

    # ==================================================================
    # DENOISE
    # ==================================================================

    def apply_denoise(
        self,
        image: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Mild bilateral filtering.

        Preserves edges better than normal Gaussian blur.
        """

        if not self._valid_image(
            image
        ):
            return None

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
    # SHARPEN
    # ==================================================================

    def sharpen(
        self,
        image: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Mild unsharp-mask sharpening.
        """

        if not self._valid_image(
            image
        ):
            return None

        try:

            blurred = cv2.GaussianBlur(
                image,
                (0, 0),
                1.0,
            )

            amount = max(
                0.0,
                self.sharpen_amount,
            )

            return cv2.addWeighted(
                image,
                1.0 + amount,
                blurred,
                -amount,
                0,
            )

        except Exception:

            return image.copy()

    # ==================================================================
    # NORMALIZE
    # ==================================================================

    def normalize(
        self,
        image: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Apply controlled enhancement while preserving BGR.

        Pipeline:

            input
              ↓
            gamma when required
              ↓
            CLAHE
              ↓
            bilateral denoise
              ↓
            mild sharpening
        """

        if not self._valid_image(
            image
        ):
            return None

        output = image.copy()

        # --------------------------------------------------------------
        # Quality measurement
        # --------------------------------------------------------------

        try:

            quality = (
                self.quality_checker.check(
                    output
                )
            )

        except Exception:

            quality = None

        # --------------------------------------------------------------
        # Gamma
        # --------------------------------------------------------------

        if self.enable_gamma:

            if quality is not None:

                if (
                    quality.brightness
                    < self.quality_checker.min_brightness
                ):

                    gamma_image = (
                        self.apply_gamma(
                            output,
                            self.gamma_dark,
                        )
                    )

                    if gamma_image is not None:
                        output = gamma_image

                elif (
                    quality.brightness
                    > self.quality_checker.max_brightness
                ):

                    gamma_image = (
                        self.apply_gamma(
                            output,
                            self.gamma_bright,
                        )
                    )

                    if gamma_image is not None:
                        output = gamma_image

        # --------------------------------------------------------------
        # CLAHE
        # --------------------------------------------------------------

        if self.enable_clahe:

            clahe_image = (
                self.apply_clahe(
                    output
                )
            )

            if clahe_image is not None:
                output = clahe_image

        # --------------------------------------------------------------
        # Denoising
        # --------------------------------------------------------------

        if self.enable_denoise:

            denoised = (
                self.apply_denoise(
                    output
                )
            )

            if denoised is not None:
                output = denoised

        # --------------------------------------------------------------
        # Sharpening
        # --------------------------------------------------------------

        if self.enable_sharpening:

            sharpened = (
                self.sharpen(
                    output
                )
            )

            if sharpened is not None:
                output = sharpened

        return output

    # ==================================================================
    # PROCESS SINGLE TAG
    # ==================================================================

    def process(
        self,
        frame: np.ndarray,
        bbox,
        padding: Optional[float] = None,
        upscale: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Complete tag preprocessing pipeline.

        frame
          ↓
        bbox validation
          ↓
        crop
          ↓
        padding
          ↓
        aspect validation
          ↓
        safe resize
          ↓
        image enhancement
        """

        crop = self.crop(
            frame,
            bbox,
            padding=padding,
        )

        if crop is None:
            return None

        # --------------------------------------------------------------
        # Reject pathological crops before enhancement.
        # --------------------------------------------------------------

        if not self._reasonable_aspect_ratio(
            crop
        ):
            return None

        # --------------------------------------------------------------
        # Resize
        # --------------------------------------------------------------

        resized = self.resize_safe(
            crop,
            upscale=upscale,
        )

        if resized is None:
            return None

        # --------------------------------------------------------------
        # Validate again
        # --------------------------------------------------------------

        if not self._reasonable_aspect_ratio(
            resized
        ):
            return None

        # --------------------------------------------------------------
        # Enhancement
        # --------------------------------------------------------------

        normalized = self.normalize(
            resized
        )

        if normalized is None:
            return None

        # --------------------------------------------------------------
        # Final protection
        # --------------------------------------------------------------

        if not self._valid_image(
            normalized
        ):
            return None

        height, width = (
            normalized.shape[:2]
        )

        if width > self.max_width:
            return None

        if height > self.max_height:
            return None

        if width > self.max_side:
            return None

        if height > self.max_side:
            return None

        return normalized

    # ==================================================================
    # PROCESS DETECTION OBJECT
    # ==================================================================

    def process_detection(
        self,
        frame: np.ndarray,
        detection,
    ) -> Optional[np.ndarray]:
        """
        Process project's Detection object.
        """

        if detection is None:
            return None

        bbox = getattr(
            detection,
            "bbox",
            None,
        )

        if bbox is None:
            return None

        return self.process(
            frame,
            bbox,
        )

    # ==================================================================
    # GENERATE OCR VARIANTS
    # ==================================================================

    def generate_variants(
        self,
        image: np.ndarray,
        rotations: Optional[
            List[int]
        ] = None,
    ) -> List[TagCrop]:
        """
        Generate safe OCR variants.

        Every variant is validated before being returned.
        """

        if not self._valid_image(
            image
        ):
            return []

        if rotations is None:

            rotations = [
                0,
                90,
                180,
                270,
            ]

        variants: List[
            TagCrop
        ] = []

        source_height, source_width = (
            image.shape[:2]
        )

        for angle in rotations:

            angle = (
                int(angle)
                % 360
            )

            rotated = self.rotate(
                image,
                angle,
            )

            if rotated is None:
                continue

            if not self._valid_image(
                rotated
            ):
                continue

            if not self._reasonable_aspect_ratio(
                rotated
            ):
                continue

            resized = self.resize_safe(
                rotated,
                upscale=self.upscale_factor,
            )

            if resized is None:
                continue

            if not self._reasonable_aspect_ratio(
                resized
            ):
                continue

            normalized = self.normalize(
                resized
            )

            if normalized is None:
                continue

            if not self._valid_image(
                normalized
            ):
                continue

            height, width = (
                normalized.shape[:2]
            )

            # ----------------------------------------------------------
            # Hard protection
            # ----------------------------------------------------------

            if width > self.max_width:
                continue

            if height > self.max_height:
                continue

            if width > self.max_side:
                continue

            if height > self.max_side:
                continue

            variants.append(
                TagCrop(
                    image=normalized,
                    bbox=(
                        0,
                        0,
                        source_width,
                        source_height,
                    ),
                    rotation=angle,
                    scale=(
                        width
                        / float(
                            source_width
                        )
                        if source_width > 0
                        else 1.0
                    ),
                    source_width=source_width,
                    source_height=source_height,
                )
            )

        return variants

    # ==================================================================
    # ALIASES
    # ==================================================================

    def extract(
        self,
        frame: np.ndarray,
        bbox,
    ) -> Optional[np.ndarray]:

        return self.crop(
            frame,
            bbox,
        )

    def prepare(
        self,
        frame: np.ndarray,
        bbox,
    ) -> Optional[np.ndarray]:

        return self.process(
            frame,
            bbox,
        )

    # ==================================================================
    # DIAGNOSTICS
    # ==================================================================

    def diagnostics(self) -> dict:

        return {
            "padding_ratio": self.padding_ratio,
            "upscale_factor": self.upscale_factor,
            "min_width": self.min_width,
            "min_height": self.min_height,
            "max_width": self.max_width,
            "max_height": self.max_height,
            "max_side": self.max_side,
            "min_aspect_ratio": self.min_aspect_ratio,
            "max_aspect_ratio": self.max_aspect_ratio,
            "clahe": self.enable_clahe,
            "gamma": self.enable_gamma,
            "denoise": self.enable_denoise,
            "sharpening": self.enable_sharpening,
        }


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

TagImageProcessor = TagProcessor


# ============================================================================
# SELF TEST
# ============================================================================


def _self_test() -> None:

    print("=" * 70)
    print("OCR_BHS TAG PROCESSOR TEST")
    print("=" * 70)

    processor = TagProcessor()

    print("\nConfiguration:")

    for key, value in (
        processor.diagnostics()
    ).items():

        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------------
    # Synthetic camera frame
    # --------------------------------------------------------------

    frame = np.full(
        (
            720,
            1280,
            3,
        ),
        180,
        dtype=np.uint8,
    )

    # Simulated tag
    cv2.rectangle(
        frame,
        (300, 200),
        (900, 500),
        (245, 245, 245),
        -1,
    )

    cv2.putText(
        frame,
        "IATA TEST TAG",
        (350, 360),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (20, 20, 20),
        4,
        cv2.LINE_AA,
    )

    bbox = (
        300,
        200,
        900,
        500,
    )

    # --------------------------------------------------------------
    # Crop
    # --------------------------------------------------------------

    crop = processor.crop(
        frame,
        bbox,
    )

    if crop is None:

        print(
            "[FAIL] Crop failed."
        )

        return

    print(
        f"\nCrop:"
        f" {crop.shape[1]} x "
        f"{crop.shape[0]}"
    )

    # --------------------------------------------------------------
    # Process
    # --------------------------------------------------------------

    processed = processor.process(
        frame,
        bbox,
    )

    if processed is None:

        print(
            "[FAIL] Processing failed."
        )

        return

    print(
        f"Processed:"
        f" {processed.shape[1]} x "
        f"{processed.shape[0]}"
    )

    print(
        f"Channels:"
        f" {processed.shape[2] if processed.ndim == 3 else 1}"
    )

    # --------------------------------------------------------------
    # Variants
    # --------------------------------------------------------------

    variants = (
        processor.generate_variants(
            crop
        )
    )

    print(
        f"OCR variants:"
        f" {len(variants)}"
    )

    # --------------------------------------------------------------
    # Quality
    # --------------------------------------------------------------

    quality = (
        processor.quality_checker.check(
            processed
        )
    )

    print(
        f"\nQuality:"
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

    print(
        "\n[PASS] "
        "Tag processor test completed."
    )


if __name__ == "__main__":
    _self_test()