"""
OCR_BHS - Barcode Region Processor

Optimized barcode-region processor for IATA tags.

Strategy:

    IATA Tag Image
          |
          v
    Direct BarcodeEngine
          |
          +---- CODE128 found ----> Return
          |
          v
    Horizontal Region Detection
          |
          v
    Rank strongest regions
          |
          v
    Decode strongest region first
          |
          +---- Reliable result ----> Return
          |
          v
    Try next candidate
          |
          v
    Full-image fallback

The region processor is a FALLBACK.
The direct BarcodeEngine remains the primary decoder.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config.config_loader import load_config
from app.barcode.barcode_engine import BarcodeEngine


# ============================================================================
# Barcode Region
# ============================================================================


@dataclass
class BarcodeRegion:
    """
    Candidate barcode region in original-image coordinates.
    """

    x: int
    y: int
    width: int
    height: int

    score: float = 0.0

    @property
    def x1(self) -> int:
        return self.x

    @property
    def y1(self) -> int:
        return self.y

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        if self.height <= 0:
            return 0.0

        return self.width / float(self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (
            self.x + self.width / 2.0,
            self.y + self.height / 2.0,
        )

    @property
    def is_horizontal(self) -> bool:
        return self.width >= self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "score": round(float(self.score), 4),
            "area": self.area,
            "aspect_ratio": round(
                self.aspect_ratio,
                4,
            ),
            "is_horizontal": self.is_horizontal,
        }


# ============================================================================
# Processor
# ============================================================================


class BarcodeRegionProcessor:
    """
    Optimized barcode-region processor.

    The complete image is first sent directly to BarcodeEngine.
    Region detection is only used when direct decoding fails.

    All settings are read from config.yaml.
    """

    def __init__(
        self,
        config=None,
        barcode_engine: BarcodeEngine | None = None,
    ):
        self.config = config or load_config()

        barcode_cfg = (
            self.config.get("barcode", {}) or {}
        )

        region_cfg = (
            barcode_cfg.get("region", {}) or {}
        )

        consensus_cfg = (
            barcode_cfg.get("consensus", {}) or {}
        )

        self.enabled = bool(
            region_cfg.get(
                "enabled",
                True,
            )
        )

        self.max_regions = int(
            region_cfg.get(
                "max_regions",
                8,
            )
        )

        # For production, processing 8 expensive crops is excessive.
        # Default is reduced to 3.
        self.decode_max_regions = int(
            region_cfg.get(
                "decode_max_regions",
                min(self.max_regions, 3),
            )
        )

        self.min_area = int(
            region_cfg.get(
                "min_area",
                250,
            )
        )

        self.min_area_ratio = float(
            region_cfg.get(
                "min_area_ratio",
                0.008,
            )
        )

        self.max_area_ratio = float(
            region_cfg.get(
                "max_area_ratio",
                0.65,
            )
        )

        self.min_aspect_ratio = float(
            region_cfg.get(
                "min_aspect_ratio",
                1.4,
            )
        )

        self.max_aspect_ratio = float(
            region_cfg.get(
                "max_aspect_ratio",
                20.0,
            )
        )

        self.min_fill_ratio = float(
            region_cfg.get(
                "min_fill_ratio",
                0.04,
            )
        )

        self.iou_threshold = float(
            region_cfg.get(
                "iou_deduplication",
                0.60,
            )
        )

        self.padding_ratio = float(
            region_cfg.get(
                "padding_ratio",
                0.08,
            )
        )

        self.minimum_confidence = float(
            consensus_cfg.get(
                "minimum_confidence",
                0.70,
            )
        )

        # Primary direct decode.
        self.direct_first = bool(
            region_cfg.get(
                "direct_first",
                True,
            )
        )

        # Stop after reliable CODE128.
        self.stop_on_code128 = bool(
            region_cfg.get(
                "stop_on_code128",
                True,
            )
        )

        self.barcode_engine = (
            barcode_engine
            or BarcodeEngine(self.config)
        )

        self.last_regions: list[
            BarcodeRegion
        ] = []

        self.last_results: list[Any] = []

        self.last_elapsed_ms = 0.0

        self.last_direct_elapsed_ms = 0.0

        self.last_region_elapsed_ms = 0.0

        self.last_region_decode_elapsed_ms = 0.0

        self.last_detection_path = "none"

    # ========================================================================
    # Public API
    # ========================================================================

    def process(
        self,
        image: np.ndarray,
    ) -> list[Any]:
        """
        Decode barcode from an image.

        Processing order:

        1. Direct full-image decode.
        2. Horizontal candidate regions.
        3. Strongest candidate first.
        4. Stop on reliable CODE128.
        5. Full-image fallback if required.
        """

        start = time.perf_counter()

        self.last_regions = []
        self.last_results = []
        self.last_detection_path = "none"

        self.last_direct_elapsed_ms = 0.0
        self.last_region_elapsed_ms = 0.0
        self.last_region_decode_elapsed_ms = 0.0

        if not self.enabled:
            self.last_elapsed_ms = (
                time.perf_counter() - start
            ) * 1000.0

            return []

        if image is None:
            self.last_elapsed_ms = (
                time.perf_counter() - start
            ) * 1000.0

            return []

        if not isinstance(
            image,
            np.ndarray,
        ):
            raise TypeError(
                "image must be numpy.ndarray"
            )

        if image.size == 0:
            self.last_elapsed_ms = (
                time.perf_counter() - start
            ) * 1000.0

            return []

        # ====================================================================
        # STEP 1 - DIRECT FULL IMAGE
        # ====================================================================

        if self.direct_first:

            direct_start = time.perf_counter()

            try:
                direct_results = (
                    self.barcode_engine.read(
                        image
                    )
                )
            except Exception:
                direct_results = []

            self.last_direct_elapsed_ms = (
                time.perf_counter()
                - direct_start
            ) * 1000.0

            direct_results = (
                self._deduplicate_results(
                    direct_results
                )
            )

            strong_direct = (
                self._get_strong_code128(
                    direct_results
                )
            )

            if strong_direct is not None:

                self.last_detection_path = (
                    "direct_full_image"
                )

                self.last_results = (
                    direct_results
                )

                self.last_elapsed_ms = (
                    time.perf_counter()
                    - start
                ) * 1000.0

                return self.last_results

            # If direct decoding found another barcode,
            # keep it as a possible fallback.
            direct_fallback_results = (
                direct_results
            )

        else:
            direct_fallback_results = []

        # ====================================================================
        # STEP 2 - FIND HORIZONTAL REGIONS
        # ====================================================================

        region_start = time.perf_counter()

        regions = self.find_regions(
            image
        )

        self.last_regions = regions

        self.last_region_elapsed_ms = (
            time.perf_counter()
            - region_start
        ) * 1000.0

        # ====================================================================
        # STEP 3 - DECODE STRONGEST REGIONS FIRST
        # ====================================================================

        region_decode_start = (
            time.perf_counter()
        )

        region_results: list[Any] = []

        for region in regions[
            : self.decode_max_regions
        ]:

            crop = self._crop_region(
                image,
                region,
            )

            if crop is None:
                continue

            try:

                decoded = (
                    self.barcode_engine.read(
                        crop
                    )
                )

            except Exception:

                decoded = []

            for result in decoded:

                try:
                    result.region = region
                except Exception:
                    pass

                # ------------------------------------------------------------
                # Map coordinates back to original image when possible.
                # ------------------------------------------------------------

                self._map_result_coordinates(
                    result,
                    region,
                    crop,
                )

                region_results.append(
                    result
                )

            # ------------------------------------------------------------
            # Stop immediately when CODE128 is reliable.
            # ------------------------------------------------------------

            if self.stop_on_code128:

                strong = (
                    self._get_strong_code128(
                        region_results
                    )
                )

                if strong is not None:
                    break

        self.last_region_decode_elapsed_ms = (
            time.perf_counter()
            - region_decode_start
        ) * 1000.0

        region_results = (
            self._deduplicate_results(
                region_results
            )
        )

        # ====================================================================
        # STEP 4 - REGION RESULT
        # ====================================================================

        if region_results:

            strong_region = (
                self._get_strong_code128(
                    region_results
                )
            )

            if strong_region is not None:

                self.last_detection_path = (
                    "barcode_region"
                )

                self.last_results = (
                    region_results
                )

                self.last_elapsed_ms = (
                    time.perf_counter()
                    - start
                ) * 1000.0

                return self.last_results

        # ====================================================================
        # STEP 5 - DIRECT RESULT FALLBACK
        # ====================================================================

        if direct_fallback_results:

            self.last_detection_path = (
                "direct_fallback"
            )

            self.last_results = (
                direct_fallback_results
            )

            self.last_elapsed_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            return self.last_results

        # ====================================================================
        # STEP 6 - FINAL FULL IMAGE FALLBACK
        # ====================================================================

        if not self.direct_first:

            fallback_start = (
                time.perf_counter()
            )

            try:

                fallback_results = (
                    self.barcode_engine.read(
                        image
                    )
                )

            except Exception:

                fallback_results = []

            self.last_direct_elapsed_ms += (
                time.perf_counter()
                - fallback_start
            ) * 1000.0

            fallback_results = (
                self._deduplicate_results(
                    fallback_results
                )
            )

            if fallback_results:

                self.last_detection_path = (
                    "full_image_fallback"
                )

                self.last_results = (
                    fallback_results
                )

                self.last_elapsed_ms = (
                    time.perf_counter()
                    - start
                ) * 1000.0

                return self.last_results

        # ====================================================================
        # NOTHING FOUND
        # ====================================================================

        self.last_detection_path = "not_found"

        self.last_results = []

        self.last_elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        return []

    # ========================================================================
    # Region Detection
    # ========================================================================

    def find_regions(
        self,
        image: np.ndarray,
    ) -> list[BarcodeRegion]:
        """
        Find horizontal barcode-like regions.

        IMPORTANT:
        Unlike the old implementation, this method does NOT
        accept vertical regions by using inverse aspect ratio.

        A barcode candidate must satisfy:

            width / height >= min_aspect_ratio

        This prevents vertical text blocks from becoming candidates.
        """

        start = time.perf_counter()

        height, width = image.shape[:2]

        image_area = float(
            width * height
        )

        if image_area <= 0:
            return []

        # ====================================================================
        # GRAYSCALE
        # ====================================================================

        if len(image.shape) == 3:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

        else:

            gray = image.copy()

        # ====================================================================
        # LIGHT BLUR
        # ====================================================================

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0,
        )

        # ====================================================================
        # HORIZONTAL BLACKHAT ONLY
        # ====================================================================

        horizontal_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (31, 9),
            )
        )

        horizontal_blackhat = (
            cv2.morphologyEx(
                gray,
                cv2.MORPH_BLACKHAT,
                horizontal_kernel,
            )
        )

        # ====================================================================
        # NORMALIZE
        # ====================================================================

        horizontal_blackhat = (
            cv2.normalize(
                horizontal_blackhat,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
            )
        )

        # ====================================================================
        # THRESHOLD
        # ====================================================================

        _, threshold = cv2.threshold(
            horizontal_blackhat,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU,
        )

        # ====================================================================
        # HORIZONTAL MORPHOLOGICAL CLOSE
        # ====================================================================

        close_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (31, 5),
            )
        )

        threshold = cv2.morphologyEx(
            threshold,
            cv2.MORPH_CLOSE,
            close_kernel,
        )

        # ====================================================================
        # REMOVE SMALL NOISE
        # ====================================================================

        open_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (3, 3),
            )
        )

        threshold = cv2.morphologyEx(
            threshold,
            cv2.MORPH_OPEN,
            open_kernel,
        )

        # ====================================================================
        # CONTOURS
        # ====================================================================

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates: list[
            BarcodeRegion
        ] = []

        for contour in contours:

            x, y, w, h = (
                cv2.boundingRect(
                    contour
                )
            )

            if w <= 0 or h <= 0:
                continue

            box_area = float(
                w * h
            )

            area_ratio = (
                box_area
                / image_area
            )

            # =================================================================
            # AREA FILTER
            # =================================================================

            if box_area < self.min_area:
                continue

            if (
                area_ratio
                < self.min_area_ratio
            ):
                continue

            if (
                area_ratio
                > self.max_area_ratio
            ):
                continue

            # =================================================================
            # STRICT HORIZONTAL ASPECT RATIO
            # =================================================================

            aspect_ratio = (
                w / float(h)
            )

            # IMPORTANT:
            #
            # Do NOT use:
            #
            # max(aspect_ratio, 1/aspect_ratio)
            #
            # because that accepts vertical regions.
            #
            if (
                aspect_ratio
                < self.min_aspect_ratio
            ):
                continue

            if (
                aspect_ratio
                > self.max_aspect_ratio
            ):
                continue

            # =================================================================
            # FILL RATIO
            # =================================================================

            contour_area = cv2.contourArea(
                contour
            )

            if contour_area <= 0:
                continue

            fill_ratio = (
                contour_area
                / box_area
            )

            if (
                fill_ratio
                < self.min_fill_ratio
            ):
                continue

            # =================================================================
            # SCORE
            # =================================================================

            score = self._region_score(
                width=w,
                height=h,
                contour_area=contour_area,
                box_area=box_area,
                image_area=image_area,
            )

            candidates.append(
                BarcodeRegion(
                    x=int(x),
                    y=int(y),
                    width=int(w),
                    height=int(h),
                    score=score,
                )
            )

        # ====================================================================
        # SORT
        # ====================================================================

        candidates.sort(
            key=lambda region: region.score,
            reverse=True,
        )

        # ====================================================================
        # DEDUPLICATE
        # ====================================================================

        candidates = (
            self._deduplicate_regions(
                candidates
            )
        )

        return candidates[
            : self.max_regions
        ]

    # ========================================================================
    # Region Score
    # ========================================================================

    @staticmethod
    def _region_score(
        width: int,
        height: int,
        contour_area: float,
        box_area: float,
        image_area: float,
    ) -> float:
        """
        Score horizontal barcode-like regions.
        """

        if (
            box_area <= 0
            or image_area <= 0
        ):
            return 0.0

        fill_ratio = (
            contour_area
            / box_area
        )

        area_ratio = (
            box_area
            / image_area
        )

        aspect_ratio = (
            width
            / float(height)
            if height > 0
            else 0.0
        )

        # Prefer wide horizontal regions.
        aspect_score = min(
            aspect_ratio / 10.0,
            1.0,
        )

        # Prefer reasonable fill.
        fill_score = min(
            fill_ratio / 0.40,
            1.0,
        )

        # Prefer sufficiently large regions.
        area_score = min(
            area_ratio / 0.10,
            1.0,
        )

        score = (
            0.55 * aspect_score
            + 0.30 * fill_score
            + 0.15 * area_score
        )

        return float(
            min(
                max(score, 0.0),
                1.0,
            )
        )

    # ========================================================================
    # Crop
    # ========================================================================

    def _crop_region(
        self,
        image: np.ndarray,
        region: BarcodeRegion,
    ) -> np.ndarray | None:
        """
        Crop region with padding.
        """

        height, width = image.shape[:2]

        pad_x = int(
            region.width
            * self.padding_ratio
        )

        pad_y = int(
            region.height
            * self.padding_ratio
        )

        x1 = max(
            0,
            region.x - pad_x,
        )

        y1 = max(
            0,
            region.y - pad_y,
        )

        x2 = min(
            width,
            region.x2 + pad_x,
        )

        y2 = min(
            height,
            region.y2 + pad_y,
        )

        if (
            x2 <= x1
            or y2 <= y1
        ):
            return None

        crop = image[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:
            return None

        return crop

    # ========================================================================
    # Coordinate Mapping
    # ========================================================================

    def _map_result_coordinates(
        self,
        result: Any,
        region: BarcodeRegion,
        crop: np.ndarray,
    ) -> None:
        """
        Map BarcodeEngine coordinates back toward original image coordinates.

        BarcodeEngine may upscale the crop internally.
        """

        if not all(
            hasattr(result, attr)
            for attr in (
                "x",
                "y",
                "w",
                "h",
            )
        ):
            return

        try:

            result_x = float(
                result.x
            )

            result_y = float(
                result.y
            )

            result_w = float(
                result.w
            )

            result_h = float(
                result.h
            )

        except (
            TypeError,
            ValueError,
        ):

            return

        crop_h, crop_w = (
            crop.shape[:2]
        )

        # Detect internal upscaling.
        scale_x = 1.0
        scale_y = 1.0

        if (
            result_x + result_w
            > crop_w * 1.05
        ):

            scale_x = (
                crop_w
                / max(
                    result_x + result_w,
                    1.0,
                )
            )

        if (
            result_y + result_h
            > crop_h * 1.05
        ):

            scale_y = (
                crop_h
                / max(
                    result_y + result_h,
                    1.0,
                )
            )

        mapped_x = (
            int(
                result_x
                * scale_x
            )
            + region.x
        )

        mapped_y = (
            int(
                result_y
                * scale_y
            )
            + region.y
        )

        mapped_w = int(
            result_w
            * scale_x
        )

        mapped_h = int(
            result_h
            * scale_y
        )

        try:

            result.x = mapped_x
            result.y = mapped_y
            result.w = mapped_w
            result.h = mapped_h

        except Exception:
            pass

    # ========================================================================
    # Strong CODE128
    # ========================================================================

    def _get_strong_code128(
        self,
        results: list[Any],
    ) -> Any | None:
        """
        Return the strongest reliable CODE128 result.
        """

        candidates = []

        for result in results:

            value = str(
                getattr(
                    result,
                    "value",
                    "",
                )
            ).strip()

            fmt = str(
                getattr(
                    result,
                    "format",
                    "",
                )
            ).upper()

            confidence = float(
                getattr(
                    result,
                    "confidence",
                    0.0,
                )
            )

            if not value:
                continue

            if (
                fmt == "CODE128"
                and confidence
                >= self.minimum_confidence
            ):

                candidates.append(
                    result
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: float(
                getattr(
                    item,
                    "confidence",
                    0.0,
                )
            ),
            reverse=True,
        )

        return candidates[0]

    # ========================================================================
    # Region Deduplication
    # ========================================================================

    def _deduplicate_regions(
        self,
        regions: list[BarcodeRegion],
    ) -> list[BarcodeRegion]:
        """
        Remove heavily overlapping regions.
        """

        selected: list[
            BarcodeRegion
        ] = []

        for candidate in regions:

            duplicate = False

            for existing in selected:

                if (
                    self._iou(
                        candidate,
                        existing,
                    )
                    >= self.iou_threshold
                ):

                    duplicate = True
                    break

            if not duplicate:
                selected.append(
                    candidate
                )

        return selected

    # ========================================================================
    # IoU
    # ========================================================================

    @staticmethod
    def _iou(
        a: BarcodeRegion,
        b: BarcodeRegion,
    ) -> float:

        x1 = max(
            a.x1,
            b.x1,
        )

        y1 = max(
            a.y1,
            b.y1,
        )

        x2 = min(
            a.x2,
            b.x2,
        )

        y2 = min(
            a.y2,
            b.y2,
        )

        intersection_width = max(
            0,
            x2 - x1,
        )

        intersection_height = max(
            0,
            y2 - y1,
        )

        intersection = (
            intersection_width
            * intersection_height
        )

        if intersection <= 0:
            return 0.0

        union = (
            a.area
            + b.area
            - intersection
        )

        if union <= 0:
            return 0.0

        return (
            intersection
            / float(union)
        )

    # ========================================================================
    # Result Deduplication
    # ========================================================================

    @staticmethod
    def _deduplicate_results(
        results: list[Any],
    ) -> list[Any]:
        """
        Keep the strongest result for each value/format.
        """

        best: dict[
            tuple[str, str],
            Any,
        ] = {}

        for result in results:

            value = str(
                getattr(
                    result,
                    "value",
                    "",
                )
            ).strip()

            fmt = str(
                getattr(
                    result,
                    "format",
                    "",
                )
            ).upper()

            if not value:
                continue

            key = (
                value.upper(),
                fmt,
            )

            current = best.get(
                key
            )

            if current is None:

                best[key] = result

                continue

            current_confidence = float(
                getattr(
                    current,
                    "confidence",
                    0.0,
                )
            )

            new_confidence = float(
                getattr(
                    result,
                    "confidence",
                    0.0,
                )
            )

            if (
                new_confidence
                > current_confidence
            ):

                best[key] = result

        return list(
            best.values()
        )

    # ========================================================================
    # Status
    # ========================================================================

    def status(
        self,
    ) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "direct_first": self.direct_first,
            "stop_on_code128": self.stop_on_code128,
            "max_regions": self.max_regions,
            "decode_max_regions": (
                self.decode_max_regions
            ),
            "min_area": self.min_area,
            "min_area_ratio": (
                self.min_area_ratio
            ),
            "max_area_ratio": (
                self.max_area_ratio
            ),
            "min_aspect_ratio": (
                self.min_aspect_ratio
            ),
            "max_aspect_ratio": (
                self.max_aspect_ratio
            ),
            "min_fill_ratio": (
                self.min_fill_ratio
            ),
            "padding_ratio": (
                self.padding_ratio
            ),
            "regions_found": len(
                self.last_regions
            ),
            "results_found": len(
                self.last_results
            ),
            "detection_path": (
                self.last_detection_path
            ),
            "direct_elapsed_ms": round(
                self.last_direct_elapsed_ms,
                2,
            ),
            "region_elapsed_ms": round(
                self.last_region_elapsed_ms,
                2,
            ),
            "region_decode_elapsed_ms": round(
                self.last_region_decode_elapsed_ms,
                2,
            ),
            "last_elapsed_ms": round(
                self.last_elapsed_ms,
                2,
            ),
        }


# ============================================================================
# Standalone Test
# ============================================================================


def main() -> int:

    print()
    print("=" * 70)
    print(
        "OCR_BHS - OPTIMIZED BARCODE REGION PROCESSOR TEST"
    )
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------------

    try:

        config = load_config()

    except Exception as exc:

        print()
        print("CONFIGURATION ERROR")
        print("-" * 70)
        print(
            f"{type(exc).__name__}: {exc}"
        )

        return 1

    # ------------------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------------------

    try:

        barcode_engine = BarcodeEngine(
            config
        )

        processor = BarcodeRegionProcessor(
            config=config,
            barcode_engine=barcode_engine,
        )

    except Exception as exc:

        print()
        print("INITIALIZATION FAILED")
        print("-" * 70)
        print(
            f"{type(exc).__name__}: {exc}"
        )

        return 1

    # ------------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------------

    print()
    print("CONFIGURATION")
    print("-" * 70)

    print(
        f"Enabled              : "
        f"{processor.enabled}"
    )

    print(
        f"Direct first         : "
        f"{processor.direct_first}"
    )

    print(
        f"Stop on CODE128      : "
        f"{processor.stop_on_code128}"
    )

    print(
        f"Max regions          : "
        f"{processor.max_regions}"
    )

    print(
        f"Decode max regions   : "
        f"{processor.decode_max_regions}"
    )

    print(
        f"Min area             : "
        f"{processor.min_area}"
    )

    print(
        f"Min area ratio       : "
        f"{processor.min_area_ratio}"
    )

    print(
        f"Max area ratio       : "
        f"{processor.max_area_ratio}"
    )

    print(
        f"Aspect ratio         : "
        f"{processor.min_aspect_ratio} - "
        f"{processor.max_aspect_ratio}"
    )

    print(
        f"Padding              : "
        f"{processor.padding_ratio}"
    )

    print(
        f"Minimum confidence   : "
        f"{processor.minimum_confidence}"
    )

    print(
        f"PyZBar available     : "
        f"{barcode_engine.pyzbar_available}"
    )

    # ------------------------------------------------------------------------
    # Image path
    # ------------------------------------------------------------------------

    if len(sys.argv) >= 2:

        image_path = Path(
            sys.argv[1]
        ).expanduser()

    else:

        image_path = (
            Path(config.project_root)
            / "captures"
            / "test_tag_01.jpg"
        )

    image_path = image_path.resolve()

    print()
    print("INPUT IMAGE")
    print("-" * 70)

    print(
        f"Path                 : "
        f"{image_path}"
    )

    if not image_path.exists():

        print()
        print(
            "ERROR: Image does not exist."
        )

        return 1

    # ------------------------------------------------------------------------
    # Read image
    # ------------------------------------------------------------------------

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:

        print()
        print(
            "ERROR: OpenCV could not read image."
        )

        return 1

    height, width = image.shape[:2]

    print(
        f"Image size           : "
        f"{width} x {height}"
    )

    # ------------------------------------------------------------------------
    # Find regions independently
    # ------------------------------------------------------------------------

    print()
    print("FINDING HORIZONTAL BARCODE REGIONS")
    print("-" * 70)

    region_start = time.perf_counter()

    try:

        regions = processor.find_regions(
            image
        )

    except Exception as exc:

        print()
        print("REGION DETECTION FAILED")
        print("-" * 70)
        print(
            f"{type(exc).__name__}: {exc}"
        )

        import traceback

        traceback.print_exc()

        return 1

    region_elapsed = (
        time.perf_counter()
        - region_start
    ) * 1000.0

    print(
        f"Regions found        : "
        f"{len(regions)}"
    )

    print(
        f"Region time          : "
        f"{region_elapsed:.2f} ms"
    )

    if not regions:

        print(
            "No horizontal barcode "
            "candidate regions found."
        )

    for index, region in enumerate(
        regions,
        start=1,
    ):

        print()
        print(
            f"Region #{index}"
        )

        print(
            f"  BBox       : "
            f"({region.x}, {region.y}, "
            f"{region.width}, "
            f"{region.height})"
        )

        print(
            f"  Area       : "
            f"{region.area}"
        )

        print(
            f"  Aspect     : "
            f"{region.aspect_ratio:.3f}"
        )

        print(
            f"  Horizontal : "
            f"{region.is_horizontal}"
        )

        print(
            f"  Score      : "
            f"{region.score:.3f}"
        )

    # ------------------------------------------------------------------------
    # Full processing
    # ------------------------------------------------------------------------

    print()
    print("RUNNING OPTIMIZED BARCODE PROCESSOR")
    print("-" * 70)

    start = time.perf_counter()

    try:

        results = processor.process(
            image
        )

    except Exception as exc:

        print()
        print("BARCODE PROCESSING FAILED")
        print("-" * 70)
        print(
            f"{type(exc).__name__}: {exc}"
        )

        import traceback

        traceback.print_exc()

        return 1

    elapsed = (
        time.perf_counter()
        - start
    ) * 1000.0

    print(
        f"Processing time      : "
        f"{elapsed:.2f} ms"
    )

    # ------------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------------

    print()

    if not results:

        print(
            "NO BARCODE DETECTED"
        )

    else:

        print(
            f"BARCODE DETECTED "
            f"({len(results)} result(s))"
        )

        for index, result in enumerate(
            results,
            start=1,
        ):

            print()
            print(
                f"Barcode #{index}"
            )

            print(
                f"  Value      : "
                f"{getattr(result, 'value', '')}"
            )

            print(
                f"  Format     : "
                f"{getattr(result, 'format', '')}"
            )

            print(
                f"  Confidence : "
                f"{float(getattr(result, 'confidence', 0.0)):.3f}"
            )

            print(
                f"  Rotation   : "
                f"{getattr(result, 'rotation', 0)}°"
            )

            print(
                f"  Variant    : "
                f"{getattr(result, 'variant', '')}"
            )

            if all(
                hasattr(result, attr)
                for attr in (
                    "x",
                    "y",
                    "w",
                    "h",
                )
            ):

                print(
                    f"  BBox       : "
                    f"({result.x}, "
                    f"{result.y}, "
                    f"{result.w}, "
                    f"{result.h})"
                )

            region = getattr(
                result,
                "region",
                None,
            )

            if region is not None:

                print(
                    f"  Region     : "
                    f"({region.x}, "
                    f"{region.y}, "
                    f"{region.width}, "
                    f"{region.height})"
                )

    # ------------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------------

    print()
    print("PROCESSOR STATUS")
    print("-" * 70)

    status = processor.status()

    for key, value in status.items():

        print(
            f"{key:<22}: {value}"
        )

    # ------------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)

    if results:

        print(
            "BARCODE REGION TEST: SUCCESS"
        )

    else:

        print(
            "BARCODE REGION TEST: "
            "NO BARCODE"
        )

    print("=" * 70)

    return 0


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    raise SystemExit(main())
