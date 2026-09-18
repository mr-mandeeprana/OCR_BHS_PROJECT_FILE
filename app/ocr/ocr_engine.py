"""
OCR Engine
==========

Centralized, config-driven OCR engine for OCR_BHS.

Features
--------
- PaddleOCR 3.x primary OCR engine
- Optional Tesseract fallback
- Configuration-driven settings
- 0/90/180/270 degree rotation support
- CPU-safe PaddlePaddle configuration for Windows
- Structured OCRResult
- OCR confidence extraction
- Normalized OCR text
- Safe OCR image resizing
- Extreme aspect-ratio protection
- PaddleOCR input validation
- No RTSP/camera logic
- Worker configuration read from config.yaml
"""

from __future__ import annotations

# ============================================================
# ENVIRONMENT
# ============================================================

import os

# Disable oneDNN / MKL-DNN before PaddleOCR is imported.
os.environ.setdefault(
    "FLAGS_use_mkldnn",
    "0",
)

# Force CPU.
os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES",
    "",
)

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

from app.config.config_loader import Config


# ============================================================
# OCR RESULT
# ============================================================


@dataclass
class OCRResult:
    """
    Standardized OCR result returned by the OCR engine.
    """

    success: bool = False
    text: str = ""
    confidence: float = 0.0
    engine: str = ""
    error: Optional[str] = None
    raw: Any = None
    details: list[dict[str, Any]] = field(
        default_factory=list
    )

    rotation: int = 0
    elapsed_ms: float = 0.0

    @property
    def normalized_text(self) -> str:
        """
        Normalize OCR text for downstream validation.
        """

        if not self.text:
            return ""

        text = str(self.text)

        text = text.replace(
            "\r",
            " ",
        )

        text = text.replace(
            "\n",
            " ",
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert result to dictionary.
        """

        return {
            "success": self.success,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "confidence": self.confidence,
            "engine": self.engine,
            "error": self.error,
            "rotation": self.rotation,
            "elapsed_ms": self.elapsed_ms,
            "details": self.details,
        }


# ============================================================
# IMAGE PREPROCESSOR
# ============================================================


class OCRPreprocessor:
    """
    Safe image preprocessing utilities.

    IMPORTANT
    ---------
    PaddleOCR can internally resize an image based on its
    aspect ratio.

    A malformed input such as:

        64 x 14720

    can therefore cause PaddleOCR to produce warnings such as:

        Resized image size (64x14720)
        exceeds max_side_limit of 4000

    This class prevents such inputs from reaching PaddleOCR.
    """

    # --------------------------------------------------------
    # Hard safety limits
    # --------------------------------------------------------

    MAX_WIDTH = 2000
    MAX_HEIGHT = 1200
    MAX_SIDE = 2500

    MIN_WIDTH = 16
    MIN_HEIGHT = 16

    MIN_ASPECT_RATIO = 0.20
    MAX_ASPECT_RATIO = 8.0

    # PaddleOCR itself uses 4000 as an internal maximum.
    PADDLE_MAX_SIDE = 4000

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    @staticmethod
    def validate_image(
        image: Any,
    ) -> bool:
        """
        Validate an OpenCV image.
        """

        if image is None:
            return False

        if not isinstance(
            image,
            np.ndarray,
        ):
            return False

        if image.size == 0:
            return False

        if len(image.shape) not in (
            2,
            3,
        ):
            return False

        height, width = image.shape[:2]

        if height <= 0 or width <= 0:
            return False

        return True

    # --------------------------------------------------------
    # ASPECT RATIO
    # --------------------------------------------------------

    @classmethod
    def aspect_ratio(
        cls,
        image: np.ndarray,
    ) -> float:
        """
        Return width / height.
        """

        if not cls.validate_image(image):
            return 0.0

        height, width = image.shape[:2]

        if height <= 0:
            return 0.0

        return float(width) / float(height)

    @classmethod
    def validate_aspect_ratio(
        cls,
        image: np.ndarray,
    ) -> bool:
        """
        Validate image aspect ratio.
        """

        ratio = cls.aspect_ratio(image)

        if ratio <= 0:
            return False

        return (
            cls.MIN_ASPECT_RATIO
            <= ratio
            <= cls.MAX_ASPECT_RATIO
        )

    # --------------------------------------------------------
    # SAFE RESIZE
    # --------------------------------------------------------

    @classmethod
    def resize_safe(
        cls,
        image: np.ndarray,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        max_side: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Resize an image while preserving its aspect ratio.

        No output dimension is allowed to exceed the configured
        safety limits.

        IMPORTANT:
        This method never stretches an image.
        """

        if not cls.validate_image(image):
            return None

        height, width = image.shape[:2]

        max_width = (
            cls.MAX_WIDTH
            if max_width is None
            else max(
                cls.MIN_WIDTH,
                int(max_width),
            )
        )

        max_height = (
            cls.MAX_HEIGHT
            if max_height is None
            else max(
                cls.MIN_HEIGHT,
                int(max_height),
            )
        )

        max_side = (
            cls.MAX_SIDE
            if max_side is None
            else max(
                max_width,
                max_height,
                int(max_side),
            )
        )

        # ----------------------------------------------------
        # Calculate scale without changing aspect ratio.
        # ----------------------------------------------------

        scale = 1.0

        if width > max_width:
            scale = min(
                scale,
                max_width / float(width),
            )

        if height > max_height:
            scale = min(
                scale,
                max_height / float(height),
            )

        largest_side = max(
            width,
            height,
        )

        if largest_side > max_side:
            scale = min(
                scale,
                max_side / float(largest_side),
            )

        # ----------------------------------------------------
        # Nothing to resize.
        # ----------------------------------------------------

        if scale >= 1.0:
            return image.copy()

        new_width = max(
            1,
            int(round(width * scale)),
        )

        new_height = max(
            1,
            int(round(height * scale)),
        )

        # Absolute final protection.
        new_width = min(
            new_width,
            max_width,
            max_side,
        )

        new_height = min(
            new_height,
            max_height,
            max_side,
        )

        try:

            resized = cv2.resize(
                image,
                (
                    new_width,
                    new_height,
                ),
                interpolation=cv2.INTER_AREA,
            )

        except Exception:
            return None

        if not cls.validate_image(
            resized
        ):
            return None

        return resized

    # --------------------------------------------------------
    # SAFE OCR INPUT
    # --------------------------------------------------------

    @classmethod
    def prepare_for_paddle(
        cls,
        image: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Prepare an image immediately before PaddleOCR.

        This is the most important safety barrier in this file.

        Pipeline:

            input
              ↓
            validation
              ↓
            BGR conversion
              ↓
            aspect validation
              ↓
            safe resize
              ↓
            final validation
              ↓
            PaddleOCR
        """

        if not cls.validate_image(image):
            return None

        # ----------------------------------------------------
        # Convert grayscale / BGRA to BGR.
        # ----------------------------------------------------

        image = cls.ensure_bgr(
            image
        )

        if not cls.validate_image(image):
            return None

        # ----------------------------------------------------
        # Reject pathological aspect ratios.
        #
        # Example:
        #
        # 64 x 14720
        #
        # ratio = 230.0
        #
        # This is not a valid IATA tag image.
        # ----------------------------------------------------

        if not cls.validate_aspect_ratio(
            image
        ):
            return None

        # ----------------------------------------------------
        # Resize safely.
        # ----------------------------------------------------

        image = cls.resize_safe(
            image,
            max_width=cls.MAX_WIDTH,
            max_height=cls.MAX_HEIGHT,
            max_side=cls.MAX_SIDE,
        )

        if image is None:
            return None

        # ----------------------------------------------------
        # Final dimension validation.
        # ----------------------------------------------------

        height, width = image.shape[:2]

        if width <= 0 or height <= 0:
            return None

        if width > cls.MAX_WIDTH:
            return None

        if height > cls.MAX_HEIGHT:
            return None

        if width > cls.PADDLE_MAX_SIDE:
            return None

        if height > cls.PADDLE_MAX_SIDE:
            return None

        # ----------------------------------------------------
        # Final aspect validation.
        # ----------------------------------------------------

        if not cls.validate_aspect_ratio(
            image
        ):
            return None

        return np.ascontiguousarray(
            image
        )

    # --------------------------------------------------------
    # ROTATION
    # --------------------------------------------------------

    @staticmethod
    def rotate(
        image: np.ndarray,
        angle: int,
    ) -> np.ndarray:
        """
        Rotate image by 0, 90, 180 or 270 degrees.
        """

        if image is None:
            raise ValueError(
                "Cannot rotate None image."
            )

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

        raise ValueError(
            f"Unsupported OCR rotation: {angle}. "
            "Supported rotations are 0, 90, 180 and 270."
        )

    # --------------------------------------------------------
    # BGR
    # --------------------------------------------------------

    @staticmethod
    def ensure_bgr(
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Ensure image is BGR for PaddleOCR/Tesseract.
        """

        if not isinstance(
            image,
            np.ndarray,
        ):
            raise ValueError(
                "Image must be numpy.ndarray."
            )

        if len(image.shape) == 2:

            return cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR,
            )

        if len(image.shape) == 3:

            channels = image.shape[2]

            if channels == 4:

                return cv2.cvtColor(
                    image,
                    cv2.COLOR_BGRA2BGR,
                )

            if channels == 3:

                return image.copy()

        raise ValueError(
            f"Unsupported image shape: "
            f"{image.shape}"
        )


# ============================================================
# PADDLE OCR ENGINE
# ============================================================


class PaddleOCREngine:
    """
    PaddleOCR 3.x wrapper.

    PaddleOCR is initialized once per OCR worker.

    The WorkerManager creates one OCREngine per worker so
    PaddleOCR instances are not shared between worker threads.
    """

    def __init__(
        self,
        language: str = "en",
    ) -> None:

        if PaddleOCR is None:

            raise RuntimeError(
                "PaddleOCR is not installed. "
                "Install paddleocr and paddlepaddle."
            )

        self.language = language

        self.ocr: Optional[Any] = None

        self._initialize()

    # --------------------------------------------------------
    # INITIALIZE
    # --------------------------------------------------------

    def _initialize(self) -> None:
        """
        Initialize PaddleOCR.

        CPU mode is intentionally used for this project.
        """

        try:

            self.ocr = PaddleOCR(
                lang=self.language,

                # We handle document orientation ourselves.
                use_doc_orientation_classify=False,

                # Not required for IATA tags.
                use_doc_unwarping=False,

                # We handle rotations ourselves.
                use_textline_orientation=False,

                # Windows CPU stability.
                enable_mkldnn=False,
            )

        except TypeError:

            # Compatibility fallback.
            self.ocr = PaddleOCR(
                lang=self.language,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

    # --------------------------------------------------------
    # SAFE FLOAT
    # --------------------------------------------------------

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return default

    # --------------------------------------------------------
    # EXTRACT DICT
    # --------------------------------------------------------

    @staticmethod
    def _extract_from_dict(
        result: Any,
    ) -> tuple[
        str,
        float,
        list[dict[str, Any]],
    ]:
        """
        Extract OCR text/confidence from PaddleOCR
        dictionary-like output.
        """

        if not isinstance(
            result,
            dict,
        ):
            return "", 0.0, []

        texts: list[str] = []
        confidences: list[float] = []
        details: list[dict[str, Any]] = []

        # ----------------------------------------------------
        # PaddleOCR 3.x
        # ----------------------------------------------------

        rec_texts = result.get(
            "rec_texts"
        )

        rec_scores = result.get(
            "rec_scores"
        )

        if rec_texts is not None:

            if not isinstance(
                rec_texts,
                (
                    list,
                    tuple,
                ),
            ):
                rec_texts = [
                    rec_texts
                ]

            if rec_scores is None:

                rec_scores = []

            if not isinstance(
                rec_scores,
                (
                    list,
                    tuple,
                ),
            ):
                rec_scores = [
                    rec_scores
                ]

            for index, raw_text in enumerate(
                rec_texts
            ):

                if raw_text is None:
                    continue

                text = str(
                    raw_text
                ).strip()

                if not text:
                    continue

                score = 0.0

                if index < len(
                    rec_scores
                ):

                    score = (
                        PaddleOCREngine
                        ._safe_float(
                            rec_scores[index]
                        )
                    )

                texts.append(
                    text
                )

                confidences.append(
                    score
                )

                details.append(
                    {
                        "text": text,
                        "confidence": score,
                    }
                )

        # ----------------------------------------------------
        # Alternative PaddleOCR output.
        # ----------------------------------------------------

        if not texts:

            text_list = result.get(
                "text"
            )

            score_list = result.get(
                "text_score"
            )

            if text_list is not None:

                if not isinstance(
                    text_list,
                    (
                        list,
                        tuple,
                    ),
                ):
                    text_list = [
                        text_list
                    ]

                if score_list is None:
                    score_list = []

                if not isinstance(
                    score_list,
                    (
                        list,
                        tuple,
                    ),
                ):
                    score_list = [
                        score_list
                    ]

                for index, raw_text in enumerate(
                    text_list
                ):

                    if raw_text is None:
                        continue

                    text = str(
                        raw_text
                    ).strip()

                    if not text:
                        continue

                    score = 0.0

                    if index < len(
                        score_list
                    ):

                        score = (
                            PaddleOCREngine
                            ._safe_float(
                                score_list[index]
                            )
                        )

                    texts.append(
                        text
                    )

                    confidences.append(
                        score
                    )

                    details.append(
                        {
                            "text": text,
                            "confidence": score,
                        }
                    )

        combined_text = " ".join(
            texts
        ).strip()

        confidence = (
            sum(confidences)
            / len(confidences)
            if confidences
            else 0.0
        )

        return (
            combined_text,
            float(confidence),
            details,
        )

    # --------------------------------------------------------
    # EXTRACT OBJECT
    # --------------------------------------------------------

    def _extract_result(
        self,
        result: Any,
    ) -> tuple[
        str,
        float,
        list[dict[str, Any]],
    ]:
        """
        Extract OCR data from PaddleOCR 3.x / PaddleX result
        objects.

        PaddleOCR 3.x may return a PaddleX result object whose
        printed representation looks like a dictionary but is
        not necessarily an actual Python dict.

        Supported sources:
            - dict
            - Mapping
            - json property
            - json() method
            - rec_texts / rec_scores attributes
            - object attributes containing the OCR result
        """

        # ----------------------------------------------------
        # Helper
        # ----------------------------------------------------

        def extract_from_data(data: Any):
            if data is None:
                return "", 0.0, []

            # Normal dictionary
            if isinstance(data, dict):
                return self._extract_from_dict(data)

            # Mapping-like objects
            try:
                from collections.abc import Mapping

                if isinstance(data, Mapping):
                    return self._extract_from_dict(dict(data))
            except Exception:
                pass

            # Object that supports dictionary-style access
            try:
                rec_texts = data["rec_texts"]
                rec_scores = data.get("rec_scores")

                return self._extract_from_dict(
                    {
                        "rec_texts": rec_texts,
                        "rec_scores": rec_scores,
                    }
                )
            except Exception:
                pass

            # Direct attributes
            try:
                rec_texts = getattr(
                    data,
                    "rec_texts",
                    None,
                )

                rec_scores = getattr(
                    data,
                    "rec_scores",
                    None,
                )

                if rec_texts is not None:
                    return self._extract_from_dict(
                        {
                            "rec_texts": rec_texts,
                            "rec_scores": rec_scores,
                        }
                    )
            except Exception:
                pass

            return "", 0.0, []

        # ----------------------------------------------------
        # 1. Direct object
        # ----------------------------------------------------

        text, confidence, details = extract_from_data(result)

        if text:
            return text, confidence, details

        # ----------------------------------------------------
        # 2. PaddleX json property / method
        # ----------------------------------------------------

        try:
            data = getattr(
                result,
                "json",
                None,
            )

            if callable(data):
                data = data()

            text, confidence, details = extract_from_data(
                data
            )

            if text:
                return text, confidence, details

        except Exception:
            pass

        # ----------------------------------------------------
        # 3. Try to convert object to dict
        # ----------------------------------------------------

        try:
            data = dict(result)

            text, confidence, details = extract_from_data(
                data
            )

            if text:
                return text, confidence, details

        except Exception:
            pass

        # ----------------------------------------------------
        # 4. Try result attributes through __dict__
        # ----------------------------------------------------

        try:
            object_data = getattr(
                result,
                "__dict__",
                None,
            )

            if isinstance(object_data, dict):

                text, confidence, details = extract_from_data(
                    object_data
                )

                if text:
                    return text, confidence, details

        except Exception:
            pass

        # ----------------------------------------------------
        # Nothing found
        # ----------------------------------------------------

        return "", 0.0, []
    # --------------------------------------------------------
    # READ
    # --------------------------------------------------------

    def read(
        self,
        image: np.ndarray,
    ) -> OCRResult:
        """
        Run PaddleOCR on one image.

        The image is protected immediately before the actual
        PaddleOCR predict() call.
        """

        start = time.perf_counter()

        if not OCRPreprocessor.validate_image(
            image
        ):

            return OCRResult(
                success=False,
                engine="paddle",
                error="EMPTY_OR_INVALID_IMAGE",
                elapsed_ms=(
                    time.perf_counter()
                    - start
                )
                * 1000.0,
            )

        # ----------------------------------------------------
        # HARD SAFETY BARRIER
        # ----------------------------------------------------

        safe_image = (
            OCRPreprocessor
            .prepare_for_paddle(
                image
            )
        )

        if safe_image is None:

            height, width = (
                image.shape[:2]
            )

            return OCRResult(
                success=False,
                engine="paddle",
                error=(
                    "UNSAFE_OCR_IMAGE:"
                    f"{width}x{height}"
                ),
                elapsed_ms=(
                    time.perf_counter()
                    - start
                )
                * 1000.0,
            )

        # ----------------------------------------------------
        # Final dimensions.
        # ----------------------------------------------------

        height, width = (
            safe_image.shape[:2]
        )

        if (
            width > OCRPreprocessor.PADDLE_MAX_SIDE
            or height > OCRPreprocessor.PADDLE_MAX_SIDE
        ):

            return OCRResult(
                success=False,
                engine="paddle",
                error=(
                    "PADDLE_IMAGE_TOO_LARGE:"
                    f"{width}x{height}"
                ),
                elapsed_ms=(
                    time.perf_counter()
                    - start
                )
                * 1000.0,
            )

        # ----------------------------------------------------
        # PaddleOCR
        # ----------------------------------------------------

        try:

            if self.ocr is None:

                self._initialize()

            predictions = self.ocr.predict(
                safe_image
            )

            all_texts: list[str] = []
            all_scores: list[float] = []
            all_details: list[
                dict[str, Any]
            ] = []

            for prediction in predictions:

                (
                    texts,
                    scores,
                    details,
                ) = self._extract_result(
                    prediction
                )

                if texts:
                    all_texts.append(
                        texts
                    )

                if scores > 0:
                    all_scores.append(
                        float(scores)
                    )

                if details:
                    all_details.extend(
                        details
                    )

            text = " ".join(
                value.strip()
                for value in all_texts
                if value.strip()
            ).strip()

            confidence = (
                sum(all_scores)
                / len(all_scores)
                if all_scores
                else 0.0
            )

            return OCRResult(
                success=bool(text),
                text=text,
                confidence=float(
                    confidence
                ),
                engine="paddle",
                raw=predictions,
                details=all_details,
                elapsed_ms=(
                    time.perf_counter()
                    - start
                )
                * 1000.0,
            )

        except Exception as exc:

            return OCRResult(
                success=False,
                engine="paddle",
                error=str(exc),
                elapsed_ms=(
                    time.perf_counter()
                    - start
                )
                * 1000.0,
            )


# ============================================================
# TESSERACT OCR ENGINE
# ============================================================


class TesseractEngine:
    """
    Optional Tesseract OCR fallback.
    """

    def __init__(
        self,
        language: str = "eng",
    ) -> None:

        self.language = language

        self.initialized = False

        if pytesseract is None:
            return

        # ----------------------------------------------------
        # Windows installation locations.
        # ----------------------------------------------------

        candidates = [
            os.path.join(
                os.environ.get(
                    "LOCALAPPDATA",
                    "",
                ),
                "Programs",
                "Tesseract-OCR",
                "tesseract.exe",
            ),
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]

        for path in candidates:

            if (
                path
                and os.path.exists(path)
            ):

                pytesseract.pytesseract.tesseract_cmd = (
                    path
                )

                self.initialized = True

                break

        # ----------------------------------------------------
        # PATH fallback.
        # ----------------------------------------------------

        if not self.initialized:

            try:

                pytesseract.get_tesseract_version()

                self.initialized = True

            except Exception:

                self.initialized = False

    # --------------------------------------------------------
    # READ
    # --------------------------------------------------------

    def read(
        self,
        image: np.ndarray,
    ) -> OCRResult:

        start = time.perf_counter()

        if not self.initialized:

            return OCRResult(
                success=False,
                engine="tesseract",
                error="Tesseract is not initialized",
                elapsed_ms=(
                    time.perf_counter()
                    - start
                )
                * 1000.0,
            )

        # ----------------------------------------------------
        # Safe image.
        # ----------------------------------------------------

        safe_image = (
            OCRPreprocessor
            .prepare_for_paddle(
                image
            )
        )

        if safe_image is None:

            return OCRResult(
                success=False,
                engine="tesseract",
                error="UNSAFE_OCR_IMAGE",
                elapsed_ms=(
                    time.perf_counter()
                    - start
                )
                * 1000.0,
            )

        try:

            rgb = cv2.cvtColor(
                safe_image,
                cv2.COLOR_BGR2RGB,
            )

            data = pytesseract.image_to_data(
                rgb,
                lang=self.language,
                config="--oem 3 --psm 6",
                output_type=pytesseract.Output.DICT,
            )

            texts: list[str] = []
            confidences: list[float] = []
            details: list[
                dict[str, Any]
            ] = []

            count = len(
                data.get(
                    "text",
                    [],
                )
            )

            for index in range(count):

                text = str(
                    data["text"][index]
                ).strip()

                if not text:
                    continue

                try:

                    confidence = float(
                        data["conf"][index]
                    )

                except (
                    TypeError,
                    ValueError,
                    IndexError,
                ):

                    confidence = 0.0

                confidence_normalized = (
                    confidence / 100.0
                )

                texts.append(
                    text
                )

                confidences.append(
                    confidence_normalized
                )

                try:

                    details.append(
                        {
                            "text": text,
                            "confidence": confidence_normalized,
                            "left": int(
                                data["left"][index]
                            ),
                            "top": int(
                                data["top"][index]
                            ),
                            "width": int(
                                data["width"][index]
                            ),
                            "height": int(
                                data["height"][index]
                            ),
                        }
                    )

                except (
                    TypeError,
                    ValueError,
                    IndexError,
                ):
                    pass

            text = " ".join(
                texts
            ).strip()

            average_confidence = (
                sum(confidences)
                / len(confidences)
                if confidences
                else 0.0
            )

            return OCRResult(
                success=bool(text),
                text=text,
                confidence=float(
                    average_confidence
                ),
                engine="tesseract",
                raw=data,
                details=details,
                elapsed_ms=(
                    time.perf_counter()
                    - start
                )
                * 1000.0,
            )

        except Exception as exc:

            return OCRResult(
                success=False,
                engine="tesseract",
                error=str(exc),
                elapsed_ms=(
                    time.perf_counter()
                    - start
                )
                * 1000.0,
            )


# ============================================================
# UNIFIED OCR ENGINE
# ============================================================


class OCREngine:
    """
    Unified, configuration-driven OCR interface.

    Configuration example:

        ocr:
          enabled: true
          engine: paddle
          languages:
            - en

          processing:
            rotations:
              - 0
              - 90
              - 180
              - 270

          worker:
            enabled: true
            count: 2
            queue_size: 50

    Supported engine values:

        paddle
        tesseract
        paddle+tesseract
        both
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        use_paddle: Optional[bool] = None,
        use_tesseract: Optional[bool] = None,
        tesseract_language: Optional[str] = None,
    ) -> None:

        self.config = config

        # ----------------------------------------------------
        # Defaults
        # ----------------------------------------------------

        self.enabled = True

        self.engine_name = "paddle"

        self.languages = ["en"]

        self.paddle_language = "en"

        self.tesseract_language = "eng"

        self.rotations = [
            0,
            90,
            180,
            270,
        ]

        self.worker_enabled = True

        self.worker_count = 1

        self.queue_size = 50

        # ----------------------------------------------------
        # Load config.
        # ----------------------------------------------------

        self._load_config()

        # ----------------------------------------------------
        # Explicit constructor overrides.
        # ----------------------------------------------------

        if use_paddle is not None:

            if use_paddle:

                if self.engine_name == "tesseract":
                    self.engine_name = "paddle+tesseract"

            else:

                if self.engine_name == "paddle":
                    self.engine_name = "tesseract"

        if use_tesseract is not None:

            if use_tesseract:

                if self.engine_name == "paddle":
                    self.engine_name = "paddle+tesseract"

            else:

                if self.engine_name == "paddle+tesseract":
                    self.engine_name = "paddle"

        if tesseract_language:

            self.tesseract_language = (
                tesseract_language
            )

        # ----------------------------------------------------
        # Engine instances.
        # ----------------------------------------------------

        self.paddle: Optional[
            PaddleOCREngine
        ] = None

        self.tesseract: Optional[
            TesseractEngine
        ] = None

        self._initialize_engines()

    # ========================================================
    # CONFIG
    # ========================================================

    def _load_config(self) -> None:
        """
        Read OCR configuration from Config.
        """

        if self.config is None:
            return

        try:

            ocr_config = (
                self.config.get(
                    "ocr",
                    {},
                )
                or {}
            )

        except Exception:

            try:

                ocr_config = (
                    self.config.data.get(
                        "ocr",
                        {},
                    )
                    or {}
                )

            except Exception:

                ocr_config = {}

        if not isinstance(
            ocr_config,
            dict,
        ):
            return

        # ----------------------------------------------------
        # Enabled
        # ----------------------------------------------------

        self.enabled = bool(
            ocr_config.get(
                "enabled",
                True,
            )
        )

        # ----------------------------------------------------
        # Engine
        # ----------------------------------------------------

        self.engine_name = str(
            ocr_config.get(
                "engine",
                "paddle",
            )
        ).lower().strip()

        if self.engine_name not in (
            "paddle",
            "tesseract",
            "paddle+tesseract",
            "both",
        ):

            self.engine_name = "paddle"

        # ----------------------------------------------------
        # Languages
        # ----------------------------------------------------

        languages = ocr_config.get(
            "languages",
            ["en"],
        )

        if not languages:
            languages = ["en"]

        if isinstance(
            languages,
            str,
        ):
            languages = [
                languages
            ]

        self.languages = [
            str(language).strip()
            for language in languages
            if str(language).strip()
        ]

        if not self.languages:
            self.languages = ["en"]

        self.paddle_language = (
            self.languages[0]
        )

        self.tesseract_language = (
            self._map_tesseract_language(
                self.paddle_language
            )
        )

        # ----------------------------------------------------
        # Processing
        # ----------------------------------------------------

        processing = (
            ocr_config.get(
                "processing",
                {},
            )
            or {}
        )

        rotations = processing.get(
            "rotations",
            [
                0,
                90,
                180,
                270,
            ],
        )

        if not rotations:
            rotations = [0]

        self.rotations = []

        for angle in rotations:

            try:

                angle = int(angle) % 360

                if angle in (
                    0,
                    90,
                    180,
                    270,
                ):

                    self.rotations.append(
                        angle
                    )

            except (
                TypeError,
                ValueError,
            ):
                continue

        if not self.rotations:
            self.rotations = [0]

        # Remove duplicates.
        self.rotations = list(
            dict.fromkeys(
                self.rotations
            )
        )

        # ----------------------------------------------------
        # Worker configuration
        # ----------------------------------------------------

        worker = (
            ocr_config.get(
                "worker",
                {},
            )
            or {}
        )

        self.worker_enabled = bool(
            worker.get(
                "enabled",
                True,
            )
        )

        try:

            self.worker_count = max(
                1,
                int(
                    worker.get(
                        "count",
                        1,
                    )
                ),
            )

        except (
            TypeError,
            ValueError,
        ):

            self.worker_count = 1

        try:

            self.queue_size = max(
                1,
                int(
                    worker.get(
                        "queue_size",
                        50,
                    )
                ),
            )

        except (
            TypeError,
            ValueError,
        ):

            self.queue_size = 50

    # ========================================================
    # LANGUAGE MAPPING
    # ========================================================

    @staticmethod
    def _map_tesseract_language(
        language: str,
    ) -> str:
        """
        Map PaddleOCR language identifiers to
        Tesseract language identifiers.
        """

        mapping = {
            "en": "eng",
            "english": "eng",
            "ch": "chi_sim",
            "chinese": "chi_sim",
            "de": "deu",
            "german": "deu",
            "fr": "fra",
            "french": "fra",
            "es": "spa",
            "spanish": "spa",
            "it": "ita",
            "italian": "ita",
        }

        return mapping.get(
            language.lower().strip(),
            "eng",
        )

    # ========================================================
    # INITIALIZE ENGINES
    # ========================================================

    def _initialize_engines(
        self,
    ) -> None:
        """
        Initialize configured OCR engines.
        """

        if not self.enabled:
            return

        # ----------------------------------------------------
        # Paddle
        # ----------------------------------------------------

        if self.engine_name in (
            "paddle",
            "paddle+tesseract",
            "both",
        ):

            try:

                self.paddle = (
                    PaddleOCREngine(
                        language=self.paddle_language
                    )
                )

            except Exception as exc:

                print(
                    "[OCR] PaddleOCR "
                    "initialization failed: "
                    f"{exc}"
                )

                self.paddle = None

        # ----------------------------------------------------
        # Tesseract
        # ----------------------------------------------------

        if self.engine_name in (
            "tesseract",
            "paddle+tesseract",
            "both",
        ):

            try:

                self.tesseract = (
                    TesseractEngine(
                        language=self.tesseract_language
                    )
                )

            except Exception as exc:

                print(
                    "[OCR] Tesseract "
                    "initialization failed: "
                    f"{exc}"
                )

                self.tesseract = None

    # ========================================================
    # IMAGE PREPARATION
    # ========================================================

    @staticmethod
    def _safe_image(
        image: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Final shared OCR image safety function.
        """

        try:

            return (
                OCRPreprocessor
                .prepare_for_paddle(
                    image
                )
            )

        except Exception:

            return None

    # ========================================================
    # PADDLE
    # ========================================================

    def read_paddle(
        self,
        image: np.ndarray,
        rotation: int = 0,
    ) -> OCRResult:
        """
        Run PaddleOCR on one rotation.

        IMPORTANT:
        Rotation happens first.
        The rotated image is then passed through the safety
        processor before PaddleOCR receives it.
        """

        if self.paddle is None:

            return OCRResult(
                success=False,
                engine="paddle",
                rotation=rotation,
                error="PaddleOCR is not initialized",
            )

        start = time.perf_counter()

        try:

            if not OCRPreprocessor.validate_image(
                image
            ):

                return OCRResult(
                    success=False,
                    engine="paddle",
                    rotation=rotation,
                    error="INVALID_IMAGE",
                )

            # ------------------------------------------------
            # Rotate.
            # ------------------------------------------------

            rotated = (
                OCRPreprocessor.rotate(
                    image,
                    rotation,
                )
            )

            # ------------------------------------------------
            # CRITICAL:
            # Validate AFTER rotation.
            # ------------------------------------------------

            safe_image = self._safe_image(
                rotated
            )

            if safe_image is None:

                height, width = (
                    rotated.shape[:2]
                )

                return OCRResult(
                    success=False,
                    engine="paddle",
                    rotation=rotation,
                    error=(
                        "UNSAFE_ROTATED_IMAGE:"
                        f"{width}x{height}"
                    ),
                    elapsed_ms=(
                        time.perf_counter()
                        - start
                    )
                    * 1000.0,
                )

            # ------------------------------------------------
            # Actual PaddleOCR call.
            # ------------------------------------------------

            result = self.paddle.read(
                safe_image
            )

            result.rotation = rotation

            result.elapsed_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            return result

        except Exception as exc:

            return OCRResult(
                success=False,
                engine="paddle",
                rotation=rotation,
                error=str(exc),
                elapsed_ms=(
                    time.perf_counter()
                    - start
                )
                * 1000.0,
            )

    # ========================================================
    # TESSERACT
    # ========================================================

    def read_tesseract(
        self,
        image: np.ndarray,
        rotation: int = 0,
    ) -> OCRResult:
        """
        Run Tesseract on one rotation.
        """

        if self.tesseract is None:

            return OCRResult(
                success=False,
                engine="tesseract",
                rotation=rotation,
                error="Tesseract is not initialized",
            )

        start = time.perf_counter()

        try:

            rotated = (
                OCRPreprocessor.rotate(
                    image,
                    rotation,
                )
            )

            safe_image = self._safe_image(
                rotated
            )

            if safe_image is None:

                return OCRResult(
                    success=False,
                    engine="tesseract",
                    rotation=rotation,
                    error="UNSAFE_ROTATED_IMAGE",
                )

            result = self.tesseract.read(
                safe_image
            )

            result.rotation = rotation

            result.elapsed_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            return result

        except Exception as exc:

            return OCRResult(
                success=False,
                engine="tesseract",
                rotation=rotation,
                error=str(exc),
                elapsed_ms=(
                    time.perf_counter()
                    - start
                )
                * 1000.0,
            )

    # ========================================================
    # BEST RESULT
    # ========================================================

    @staticmethod
    def _select_best_result(
        candidates: list[OCRResult],
    ) -> OCRResult:
        """
        Select highest-confidence successful result.

        If no candidate succeeds, return the candidate with
        the highest confidence/error information.
        """

        if not candidates:

            return OCRResult(
                success=False,
                engine="none",
                error="NO_OCR_CANDIDATES",
            )

        successful = [
            result
            for result in candidates
            if result.success
            and result.text.strip()
        ]

        if successful:

            return max(
                successful,
                key=lambda result: (
                    result.confidence,
                    len(
                        result.text.strip()
                    ),
                ),
            )

        return max(
            candidates,
            key=lambda result: (
                result.confidence,
                bool(result.text),
            ),
        )

    # ========================================================
    # READ
    # ========================================================

    def read(
        self,
        image: np.ndarray,
    ) -> OCRResult:
        """
        Run OCR using configured engine and rotations.

        Paddle:
            configured rotations are tested.

        Tesseract:
            configured rotations are tested.

        The highest-confidence successful result is returned.
        """

        overall_start = time.perf_counter()

        if not self.enabled:

            return OCRResult(
                success=False,
                engine=self.engine_name,
                error="OCR is disabled in configuration",
            )

        if not OCRPreprocessor.validate_image(
            image
        ):

            return OCRResult(
                success=False,
                engine=self.engine_name,
                error="Invalid image",
            )

        # ----------------------------------------------------
        # Initial safety validation.
        #
        # We intentionally do NOT reject here solely because
        # the image has an unusual aspect ratio. The final
        # rotation-aware safety barrier handles that.
        # ----------------------------------------------------

        candidates: list[OCRResult] = []

        # ====================================================
        # PADDLE
        # ====================================================

        if self.engine_name == "paddle":

            for rotation in self.rotations:

                result = self.read_paddle(
                    image,
                    rotation,
                )

                candidates.append(
                    result
                )

                # Strong result -> stop.
                if (
                    result.success
                    and result.confidence >= 0.90
                ):

                    break

        # ====================================================
        # TESSERACT
        # ====================================================

        elif self.engine_name == "tesseract":

            for rotation in self.rotations:

                result = self.read_tesseract(
                    image,
                    rotation,
                )

                candidates.append(
                    result
                )

                if (
                    result.success
                    and result.confidence >= 0.90
                ):

                    break

        # ====================================================
        # PADDLE + TESSERACT
        # ====================================================

        elif self.engine_name in (
            "paddle+tesseract",
            "both",
        ):

            # ------------------------------------------------
            # Paddle first.
            # ------------------------------------------------

            if self.paddle is not None:

                for rotation in self.rotations:

                    result = self.read_paddle(
                        image,
                        rotation,
                    )

                    candidates.append(
                        result
                    )

                    if (
                        result.success
                        and result.confidence >= 0.90
                    ):

                        break

            # ------------------------------------------------
            # Tesseract only if Paddle did not give a strong
            # result.
            # ------------------------------------------------

            paddle_success = any(
                result.success
                and result.text.strip()
                for result in candidates
                if result.engine == "paddle"
            )

            if (
                not paddle_success
                and self.tesseract is not None
            ):

                for rotation in self.rotations:

                    result = self.read_tesseract(
                        image,
                        rotation,
                    )

                    candidates.append(
                        result
                    )

                    if (
                        result.success
                        and result.confidence >= 0.90
                    ):

                        break

        else:

            return OCRResult(
                success=False,
                engine="none",
                error=(
                    f"Unsupported OCR engine: "
                    f"{self.engine_name}"
                ),
            )

        # ----------------------------------------------------
        # Select best result.
        # ----------------------------------------------------

        best = self._select_best_result(
            candidates
        )

        best.elapsed_ms = (
            time.perf_counter()
            - overall_start
        ) * 1000.0

        return best

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    def diagnostics(self) -> dict[str, Any]:
        """
        Return OCR engine configuration.
        """

        return {
            "enabled": self.enabled,
            "engine": self.engine_name,
            "languages": self.languages,
            "paddle_language": self.paddle_language,
            "tesseract_language": self.tesseract_language,
            "rotations": self.rotations,
            "worker_enabled": self.worker_enabled,
            "worker_count": self.worker_count,
            "queue_size": self.queue_size,
            "paddle_initialized": (
                self.paddle is not None
            ),
            "tesseract_initialized": (
                self.tesseract is not None
            ),
            "max_width": OCRPreprocessor.MAX_WIDTH,
            "max_height": OCRPreprocessor.MAX_HEIGHT,
            "max_side": OCRPreprocessor.MAX_SIDE,
            "min_aspect_ratio": (
                OCRPreprocessor.MIN_ASPECT_RATIO
            ),
            "max_aspect_ratio": (
                OCRPreprocessor.MAX_ASPECT_RATIO
            ),
        }


# ============================================================
# SELF TEST
# ============================================================


def _self_test() -> None:
    """
    OCR engine safety test.

    This test does NOT require PaddleOCR inference.

    It validates the image protection layer responsible for
    preventing malformed dimensions from reaching PaddleOCR.
    """

    print("=" * 70)
    print("OCR_BHS OCR ENGINE SAFETY TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Normal IATA-like image.
    # --------------------------------------------------------

    normal = np.zeros(
        (
            174,
            406,
            3,
        ),
        dtype=np.uint8,
    )

    safe = (
        OCRPreprocessor
        .prepare_for_paddle(
            normal
        )
    )

    assert safe is not None

    print(
        "[PASS] Normal image accepted: "
        f"{normal.shape[1]}x{normal.shape[0]}"
        " -> "
        f"{safe.shape[1]}x{safe.shape[0]}"
    )

    # --------------------------------------------------------
    # 90 degree rotation.
    # --------------------------------------------------------

    rotated = (
        OCRPreprocessor.rotate(
            normal,
            90,
        )
    )

    safe_rotated = (
        OCRPreprocessor
        .prepare_for_paddle(
            rotated
        )
    )

    assert safe_rotated is not None

    print(
        "[PASS] Rotated image accepted: "
        f"{rotated.shape[1]}x{rotated.shape[0]}"
        " -> "
        f"{safe_rotated.shape[1]}x"
        f"{safe_rotated.shape[0]}"
    )

    # --------------------------------------------------------
    # Pathological wide image.
    # --------------------------------------------------------

    pathological_wide = np.zeros(
        (
            64,
            14720,
        ),
        dtype=np.uint8,
    )

    protected_wide = (
        OCRPreprocessor
        .prepare_for_paddle(
            pathological_wide
        )
    )

    assert protected_wide is None

    print(
        "[PASS] Pathological wide image rejected: "
        "64x14720"
    )

    # --------------------------------------------------------
    # Pathological tall image.
    # --------------------------------------------------------

    pathological_tall = np.zeros(
        (
            14720,
            64,
        ),
        dtype=np.uint8,
    )

    protected_tall = (
        OCRPreprocessor
        .prepare_for_paddle(
            pathological_tall
        )
    )

    assert protected_tall is None

    print(
        "[PASS] Pathological tall image rejected: "
        "14720x64"
    )

    # --------------------------------------------------------
    # Large but reasonable image.
    # --------------------------------------------------------

    large = np.zeros(
        (
            3000,
            5000,
            3,
        ),
        dtype=np.uint8,
    )

    safe_large = (
        OCRPreprocessor
        .prepare_for_paddle(
            large
        )
    )

    assert safe_large is not None

    h, w = safe_large.shape[:2]

    assert w <= OCRPreprocessor.MAX_WIDTH
    assert h <= OCRPreprocessor.MAX_HEIGHT
    assert w <= OCRPreprocessor.MAX_SIDE
    assert h <= OCRPreprocessor.MAX_SIDE

    print(
        "[PASS] Large image safely resized: "
        f"5000x3000 -> {w}x{h}"
    )

    # --------------------------------------------------------
    # Grayscale.
    # --------------------------------------------------------

    grayscale = np.zeros(
        (
            200,
            500,
        ),
        dtype=np.uint8,
    )

    safe_gray = (
        OCRPreprocessor
        .prepare_for_paddle(
            grayscale
        )
    )

    assert safe_gray is not None
    assert len(safe_gray.shape) == 3
    assert safe_gray.shape[2] == 3

    print(
        "[PASS] Grayscale converted to BGR: "
        f"{safe_gray.shape}"
    )

    # --------------------------------------------------------
    # BGRA.
    # --------------------------------------------------------

    bgra = np.zeros(
        (
            200,
            500,
            4,
        ),
        dtype=np.uint8,
    )

    safe_bgra = (
        OCRPreprocessor
        .prepare_for_paddle(
            bgra
        )
    )

    assert safe_bgra is not None
    assert safe_bgra.shape[2] == 3

    print(
        "[PASS] BGRA converted to BGR: "
        f"{safe_bgra.shape}"
    )

    # --------------------------------------------------------
    # Invalid image.
    # --------------------------------------------------------

    invalid = (
        OCRPreprocessor
        .prepare_for_paddle(
            None
        )
    )

    assert invalid is None

    print(
        "[PASS] None image rejected."
    )

    # --------------------------------------------------------
    # Diagnostics.
    # --------------------------------------------------------

    print()
    print(
        "Safety limits:"
    )

    print(
        f"  MAX_WIDTH          : "
        f"{OCRPreprocessor.MAX_WIDTH}"
    )

    print(
        f"  MAX_HEIGHT         : "
        f"{OCRPreprocessor.MAX_HEIGHT}"
    )

    print(
        f"  MAX_SIDE           : "
        f"{OCRPreprocessor.MAX_SIDE}"
    )

    print(
        f"  MIN_ASPECT_RATIO   : "
        f"{OCRPreprocessor.MIN_ASPECT_RATIO}"
    )

    print(
        f"  MAX_ASPECT_RATIO   : "
        f"{OCRPreprocessor.MAX_ASPECT_RATIO}"
    )

    print()
    print("=" * 70)
    print("OCR_BHS OCR ENGINE SAFETY TEST PASSED")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    _self_test()