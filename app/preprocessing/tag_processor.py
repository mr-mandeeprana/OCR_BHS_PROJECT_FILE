"""
OCR_BHS - IATA Tag Image Processor

Responsibilities:
- Safely crop detected IATA tags from camera frames
- Clamp bounding boxes to image boundaries
- Reject completely invalid/outside bounding boxes
- Add configurable padding
- Resize crops safely
- Preserve aspect ratio
- Prevent malformed / extreme image dimensions
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
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        if self.height <= 0:
            return 0.0

        return self.width / float(self.height)


# ============================================================================
# TAG PROCESSOR
# ============================================================================

class TagProcessor:
    """
    Safe image processor for detected IATA tags.

    The processor protects OCR from pathological images such as:

        64 x 14720
        64 x 7552
        14720 x 64

    These images can cause PaddleOCR to perform excessive resizing.

    The processor therefore enforces:

        maximum width
        maximum height
        maximum side
        reasonable aspect ratio
        valid bounding box
    """

    def __init__(self, config=None):

        self.config = config

        # --------------------------------------------------------------
        # Defaults
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

        self.enable_clahe = False
        self.enable_sharpening = False

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

            if not isinstance(preprocessing, dict):
                return

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

        except Exception:
            # Never allow config errors to kill the live pipeline.
            pass

        # --------------------------------------------------------------
        # Safety normalization
        # --------------------------------------------------------------

        self.padding_ratio = max(
            0.0,
            min(self.padding_ratio, 0.50),
        )

        self.upscale_factor = max(
            0.25,
            min(self.upscale_factor, 4.0),
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

    # ==================================================================
    # IMAGE VALIDATION
    # ==================================================================

    @staticmethod
    def _valid_image(
        image: Optional[np.ndarray],
    ) -> bool:

        if image is None:
            return False

        if not isinstance(image, np.ndarray):
            return False

        if image.size == 0:
            return False

        if image.ndim not in (2, 3):
            return False

        if image.shape[0] <= 0:
            return False

        if image.shape[1] <= 0:
            return False

        return True

    # ==================================================================
    # BBOX SANITIZATION
    # ==================================================================

    @staticmethod
    def _sanitize_bbox(
        bbox,
        frame_width: int,
        frame_height: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Convert an xyxy bbox into a safe integer bbox.

        IMPORTANT:
        A bbox that is completely outside the image is rejected.

        Example:

            frame = 848 x 478

            (-500, -500, -100, -100)

        is rejected instead of being clamped into the image.

        Partially outside boxes are allowed and clipped.
        """

        if bbox is None:
            return None

        try:

            if len(bbox) != 4:
                return None

            x1, y1, x2, y2 = [
                float(value)
                for value in bbox
            ]

        except Exception:
            return None

        # --------------------------------------------------------------
        # Reject NaN / infinity
        # --------------------------------------------------------------

        if not all(
            np.isfinite(value)
            for value in (x1, y1, x2, y2)
        ):
            return None

        # --------------------------------------------------------------
        # Fix reversed coordinates
        # --------------------------------------------------------------

        if x2 < x1:
            x1, x2 = x2, x1

        if y2 < y1:
            y1, y2 = y2, y1

        # --------------------------------------------------------------
        # IMPORTANT:
        # Reject boxes completely outside the frame.
        # --------------------------------------------------------------

        if x2 <= 0:
            return None

        if y2 <= 0:
            return None

        if x1 >= frame_width:
            return None

        if y1 >= frame_height:
            return None

        # --------------------------------------------------------------
        # Convert to integer
        # --------------------------------------------------------------

        x1 = int(np.floor(x1))
        y1 = int(np.floor(y1))
        x2 = int(np.ceil(x2))
        y2 = int(np.ceil(y2))

        # --------------------------------------------------------------
        # Clamp partially outside boxes
        # --------------------------------------------------------------

        x1 = max(
            0,
            min(
                x1,
                frame_width - 1,
            ),
        )

        y1 = max(
            0,
            min(
                y1,
                frame_height - 1,
            ),
        )

        x2 = max(
            1,
            min(
                x2,
                frame_width,
            ),
        )

        y2 = max(
            1,
            min(
                y2,
                frame_height,
            ),
        )

        # --------------------------------------------------------------
        # Final validation
        # --------------------------------------------------------------

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
    # PADDING
    # ==================================================================

    def _add_padding(
        self,
        bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[int, int, int, int]:

        x1, y1, x2, y2 = bbox

        width = x2 - x1
        height = y2 - y1

        pad_x = int(
            round(
                width * self.padding_ratio
            )
        )

        pad_y = int(
            round(
                height * self.padding_ratio
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
        Safely crop a tag from a camera frame.
        """

        if not self._valid_image(frame):
            return None

        frame_height, frame_width = frame.shape[:2]

        safe_bbox = self._sanitize_bbox(
            bbox,
            frame_width,
            frame_height,
        )

        if safe_bbox is None:
            return None

        # --------------------------------------------------------------
        # Padding
        # --------------------------------------------------------------

        if padding is None:

            padding_ratio = self.padding_ratio

        else:

            try:
                padding_ratio = float(padding)

            except Exception:
                padding_ratio = self.padding_ratio

            padding_ratio = max(
                0.0,
                min(
                    padding_ratio,
                    0.50,
                ),
            )

        x1, y1, x2, y2 = safe_bbox

        width = x2 - x1
        height = y2 - y1

        pad_x = int(
            round(
                width * padding_ratio
            )
        )

        pad_y = int(
            round(
                height * padding_ratio
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

        if not self._valid_image(crop):
            return None

        # Important:
        # copy prevents the crop from referencing the camera frame.
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

        No output dimension can exceed configured safety limits.
        """

        if not self._valid_image(image):
            return None

        height, width = image.shape[:2]

        if height <= 0 or width <= 0:
            return None

        # --------------------------------------------------------------
        # Scale
        # --------------------------------------------------------------

        if upscale is None:

            scale = self.upscale_factor

        else:

            try:
                scale = float(upscale)

            except Exception:
                scale = self.upscale_factor

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

            target_width = self.max_width

            target_height = max(
                1,
                int(
                    round(
                        target_height * ratio
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

            target_height = self.max_height

            target_width = max(
                1,
                int(
                    round(
                        target_width * ratio
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
                        target_width * ratio
                    )
                ),
            )

            target_height = max(
                1,
                int(
                    round(
                        target_height * ratio
                    )
                ),
            )

        # --------------------------------------------------------------
        # Absolute final limits
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

            interpolation = cv2.INTER_AREA

        else:

            interpolation = cv2.INTER_CUBIC

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

        if not self._valid_image(resized):
            return None

        return resized

    # ==================================================================
    # ASPECT RATIO
    # ==================================================================

    def _reasonable_aspect_ratio(
        self,
        image: np.ndarray,
    ) -> bool:

        if not self._valid_image(image):
            return False

        height, width = image.shape[:2]

        if height <= 0 or width <= 0:
            return False

        aspect = width / float(height)

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
        """
        Rotate only by 0 / 90 / 180 / 270 degrees.
        """

        if image is None:
            return None

        if image.size == 0:
            return None

        angle = int(angle) % 360

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

        if not TagProcessor._valid_image(image):
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
    # CLAHE
    # ==================================================================

    @staticmethod
    def apply_clahe(
        image: np.ndarray,
    ) -> Optional[np.ndarray]:

        if not TagProcessor._valid_image(image):
            return None

        gray = TagProcessor.to_gray(
            image
        )

        if gray is None:
            return None

        try:

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8),
            )

            return clahe.apply(gray)

        except Exception:

            return gray

    # ==================================================================
    # SHARPEN
    # ==================================================================

    @staticmethod
    def sharpen(
        image: np.ndarray,
    ) -> Optional[np.ndarray]:

        if not TagProcessor._valid_image(image):
            return None

        try:

            kernel = np.array(
                [
                    [0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0],
                ],
                dtype=np.float32,
            )

            return cv2.filter2D(
                image,
                -1,
                kernel,
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

        if not self._valid_image(image):
            return None

        output = image.copy()

        if self.enable_clahe:

            clahe_image = self.apply_clahe(
                output
            )

            if clahe_image is not None:
                output = clahe_image

        if self.enable_sharpening:

            sharpened = self.sharpen(
                output
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
        aspect validation
          ↓
        safe resize
          ↓
        normalization
        """

        crop = self.crop(
            frame,
            bbox,
            padding=padding,
        )

        if crop is None:
            return None

        # Reject pathological crops before OCR.
        if not self._reasonable_aspect_ratio(
            crop
        ):
            return None

        resized = self.resize_safe(
            crop,
            upscale=upscale,
        )

        if resized is None:
            return None

        # Validate again.
        if not self._reasonable_aspect_ratio(
            resized
        ):
            return None

        return self.normalize(
            resized
        )

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
        rotations: Optional[List[int]] = None,
    ) -> List[TagCrop]:
        """
        Generate safe OCR variants.

        Every variant is validated before being returned.
        """

        if not self._valid_image(image):
            return []

        if rotations is None:
            rotations = [
                0,
                90,
                180,
                270,
            ]

        variants: List[TagCrop] = []

        source_height, source_width = (
            image.shape[:2]
        )

        for angle in rotations:

            angle = int(angle) % 360

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

            # ----------------------------------------------------------
            # Validate before resize
            # ----------------------------------------------------------

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

            # ----------------------------------------------------------
            # Validate after resize
            # ----------------------------------------------------------

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
            # Final hard protection
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
                        / float(source_width)
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

    print("Configuration:")

    for key, value in processor.diagnostics().items():
        print(
            f"  {key}: {value}"
        )

    # ------------------------------------------------------------------
    # Fake camera frame
    # ------------------------------------------------------------------

    frame = np.zeros(
        (478, 848, 3),
        dtype=np.uint8,
    )

    bbox = (
        250,
        150,
        600,
        300,
    )

    crop = processor.crop(
        frame,
        bbox,
    )

    assert crop is not None

    print(
        f"[PASS] Safe crop: "
        f"{crop.shape[1]}x{crop.shape[0]}"
    )

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    processed = processor.process(
        frame,
        bbox,
    )

    assert processed is not None

    print(
        f"[PASS] Processed crop: "
        f"{processed.shape[1]}x{processed.shape[0]}"
    )

    # ------------------------------------------------------------------
    # Pathological image test
    # ------------------------------------------------------------------

    pathological = np.zeros(
        (64, 14720),
        dtype=np.uint8,
    )

    safe = processor.resize_safe(
        pathological,
        upscale=1.0,
    )

    assert safe is not None

    safe_height, safe_width = (
        safe.shape[:2]
    )

    print(
        f"[PASS] Pathological input protected: "
        f"14720x64 -> "
        f"{safe_width}x{safe_height}"
    )

    assert safe_width <= processor.max_width
    assert safe_height <= processor.max_height
    assert safe_width <= processor.max_side
    assert safe_height <= processor.max_side

    # ------------------------------------------------------------------
    # Rotation test
    # ------------------------------------------------------------------

    variants = processor.generate_variants(
        crop,
        rotations=[
            0,
            90,
            180,
            270,
        ],
    )

    print(
        f"[PASS] Safe OCR variants generated: "
        f"{len(variants)}"
    )

    for variant in variants:

        h, w = variant.image.shape[:2]

        assert w <= processor.max_width
        assert h <= processor.max_height
        assert w <= processor.max_side
        assert h <= processor.max_side

        print(
            f"       rotation={variant.rotation:3d}° "
            f"size={w}x{h} "
            f"aspect={variant.aspect_ratio:.3f}"
        )

    # ------------------------------------------------------------------
    # Completely outside bbox
    # ------------------------------------------------------------------

    invalid = processor.crop(
        frame,
        (
            -500,
            -500,
            -100,
            -100,
        ),
    )

    assert invalid is None

    print(
        "[PASS] Completely outside bbox rejected."
    )

    # ------------------------------------------------------------------
    # Partially outside bbox
    # ------------------------------------------------------------------

    partial = processor.crop(
        frame,
        (
            -100,
            -50,
            300,
            250,
        ),
    )

    assert partial is not None

    print(
        f"[PASS] Partially outside bbox clipped safely: "
        f"{partial.shape[1]}x{partial.shape[0]}"
    )

    # ------------------------------------------------------------------
    # Invalid bbox
    # ------------------------------------------------------------------

    invalid_nan = processor.crop(
        frame,
        (
            np.nan,
            100,
            300,
            300,
        ),
    )

    assert invalid_nan is None

    print(
        "[PASS] NaN bbox rejected."
    )

    # ------------------------------------------------------------------
    # Completely right of image
    # ------------------------------------------------------------------

    invalid_right = processor.crop(
        frame,
        (
            1000,
            100,
            1200,
            300,
        ),
    )

    assert invalid_right is None

    print(
        "[PASS] Right-side outside bbox rejected."
    )

    # ------------------------------------------------------------------
    # Completely below image
    # ------------------------------------------------------------------

    invalid_bottom = processor.crop(
        frame,
        (
            100,
            600,
            300,
            800,
        ),
    )

    assert invalid_bottom is None

    print(
        "[PASS] Bottom-side outside bbox rejected."
    )

    print()
    print("=" * 70)
    print("TAG PROCESSOR TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    _self_test()