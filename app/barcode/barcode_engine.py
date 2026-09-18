"""
OCR_BHS - Barcode Engine

Config-driven barcode reader for IATA tag processing.

Features:
    - PyZBar barcode decoding
    - Primary CODE128 detection
    - Configurable fallback formats
    - Image upscaling
    - Rotation handling
    - CLAHE enhancement
    - Sharpening
    - Otsu thresholding
    - Adaptive thresholding
    - Fixed threshold variants
    - Result deduplication
    - Consensus scoring
    - Standalone image testing

Usage:
    python -m app.barcode.barcode_engine

    python -m app.barcode.barcode_engine "path/to/image.jpg"
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config.config_loader import load_config


# ---------------------------------------------------------------------------
# Optional PyZBar import
# ---------------------------------------------------------------------------

try:
    from pyzbar.pyzbar import ZBarSymbol, decode

    PYZBAR_AVAILABLE = True
except Exception:
    ZBarSymbol = None
    decode = None
    PYZBAR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Barcode result
# ---------------------------------------------------------------------------

@dataclass
class BarcodeResult:
    """
    Represents one detected barcode.
    """

    value: str
    format: str

    confidence: float = 0.0

    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    rotation: int = 0
    variant: str = "original"

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "format": self.format,
            "confidence": round(float(self.confidence), 4),
            "bbox": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
            "rotation": self.rotation,
            "variant": self.variant,
        }


# ---------------------------------------------------------------------------
# Barcode Engine
# ---------------------------------------------------------------------------

class BarcodeEngine:
    """
    Central barcode decoding engine.

    The engine reads all barcode-related settings from config.yaml.
    """

    def __init__(self, config=None):
        self.config = config or load_config()

        barcode_cfg = self.config.get("barcode", {}) or {}

        self.enabled = bool(
            barcode_cfg.get("enabled", True)
        )

        self.engine_name = str(
            barcode_cfg.get("engine", "pyzbar")
        ).lower()

        self.primary_format = str(
            barcode_cfg.get("primary_format", "CODE128")
        ).upper()

        # ---------------------------------------------------------------
        # Fallback formats
        # ---------------------------------------------------------------

        fallback_cfg = barcode_cfg.get(
            "fallback_formats",
            {}
        ) or {}

        self.fallback_enabled = bool(
            fallback_cfg.get("enabled", True)
        )

        self.fallback_formats = [
            str(fmt).upper()
            for fmt in fallback_cfg.get(
                "formats",
                [
                    "CODE128",
                    "CODE39",
                    "CODE93",
                    "I25",
                    "EAN13",
                    "EAN8",
                    "UPCA",
                    "UPCE",
                ],
            )
        ]

        # ---------------------------------------------------------------
        # Processing
        # ---------------------------------------------------------------

        processing_cfg = barcode_cfg.get(
            "processing",
            {}
        ) or {}

        self.upscale = float(
            processing_cfg.get("upscale", 3.0)
        )

        self.max_variants = int(
            processing_cfg.get("max_variants", 30)
        )

        # ---------------------------------------------------------------
        # Rotations
        # ---------------------------------------------------------------

        rotation_cfg = processing_cfg.get(
            "rotations",
            {}
        ) or {}

        self.rotation_enabled = bool(
            rotation_cfg.get("enabled", True)
        )

        self.rotation_angles = [
            int(angle)
            for angle in rotation_cfg.get(
                "angles",
                [0, 90, 180, 270],
            )
        ]

        # ---------------------------------------------------------------
        # CLAHE
        # ---------------------------------------------------------------

        clahe_cfg = processing_cfg.get(
            "clahe",
            {}
        ) or {}

        self.clahe_enabled = bool(
            clahe_cfg.get("enabled", True)
        )

        self.clahe_clip_limit = float(
            clahe_cfg.get("clip_limit", 2.5)
        )

        self.clahe_tile_grid_size = int(
            clahe_cfg.get("tile_grid_size", 8)
        )

        # ---------------------------------------------------------------
        # Sharpening
        # ---------------------------------------------------------------

        sharpening_cfg = processing_cfg.get(
            "sharpening",
            {}
        ) or {}

        self.sharpening_enabled = bool(
            sharpening_cfg.get("enabled", True)
        )

        self.sharpening_sigma = float(
            sharpening_cfg.get("sigma", 1.2)
        )

        self.sharpening_amount = float(
            sharpening_cfg.get("amount", 1.8)
        )

        # ---------------------------------------------------------------
        # Thresholding
        # ---------------------------------------------------------------

        threshold_cfg = processing_cfg.get(
            "thresholding",
            {}
        ) or {}

        self.otsu_enabled = bool(
            threshold_cfg.get("otsu", True)
        )

        self.adaptive_enabled = bool(
            threshold_cfg.get("adaptive", True)
        )

        fixed_cfg = threshold_cfg.get(
            "fixed",
            {}
        ) or {}

        self.fixed_threshold_enabled = bool(
            fixed_cfg.get("enabled", True)
        )

        self.fixed_threshold_values = [
            int(value)
            for value in fixed_cfg.get(
                "values",
                [80, 100, 120, 140, 160, 180, 200],
            )
        ]

        # ---------------------------------------------------------------
        # Consensus
        # ---------------------------------------------------------------

        consensus_cfg = barcode_cfg.get(
            "consensus",
            {}
        ) or {}

        self.consensus_enabled = bool(
            consensus_cfg.get("enabled", True)
        )

        self.minimum_confidence = float(
            consensus_cfg.get(
                "minimum_confidence",
                0.70,
            )
        )

        # ---------------------------------------------------------------
        # Runtime state
        # ---------------------------------------------------------------

        self.pyzbar_available = PYZBAR_AVAILABLE

        self.last_results: list[BarcodeResult] = []

        self.last_elapsed_ms: float = 0.0

        self.total_reads: int = 0

        self.successful_reads: int = 0

    # ===================================================================
    # Public API
    # ===================================================================

    def read(
        self,
        image: np.ndarray,
    ) -> list[BarcodeResult]:
        """
        Read barcodes from an image.

        Returns:
            List of BarcodeResult objects.
        """

        start_time = time.perf_counter()

        self.total_reads += 1

        if not self.enabled:
            self.last_results = []
            self.last_elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000.0
            return []

        if image is None:
            self.last_results = []
            self.last_elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000.0
            return []

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "BarcodeEngine.read() expects a numpy.ndarray"
            )

        if image.size == 0:
            self.last_results = []
            self.last_elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000.0
            return []

        if not self.pyzbar_available:
            self.last_results = []
            self.last_elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000.0
            return []

        # ---------------------------------------------------------------
        # Generate variants
        # ---------------------------------------------------------------

        variants = self._generate_variants(image)

        # ---------------------------------------------------------------
        # Decode
        # ---------------------------------------------------------------

        raw_results: list[BarcodeResult] = []

        for variant_name, variant_image, rotation in variants:

            decoded = self._decode_image(
                variant_image,
                rotation=rotation,
                variant_name=variant_name,
            )

            raw_results.extend(decoded)

            # If a strong CODE128 result is found, we can stop early.
            if self._has_strong_primary_result(raw_results):
                break

            if len(raw_results) >= 100:
                break

        # ---------------------------------------------------------------
        # Deduplicate
        # ---------------------------------------------------------------

        results = self._deduplicate_results(raw_results)

        # ---------------------------------------------------------------
        # Consensus
        # ---------------------------------------------------------------

        if self.consensus_enabled:
            results = self._apply_consensus(results)

        # ---------------------------------------------------------------
        # Final confidence filtering
        # ---------------------------------------------------------------

        results = [
            result
            for result in results
            if result.confidence >= self.minimum_confidence
        ]

        # ---------------------------------------------------------------
        # Sort
        # ---------------------------------------------------------------

        results.sort(
            key=lambda item: (
                item.format == self.primary_format,
                item.confidence,
            ),
            reverse=True,
        )

        self.last_results = results

        if results:
            self.successful_reads += 1

        self.last_elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        return results

    # ===================================================================
    # Variant generation
    # ===================================================================

    def _generate_variants(
        self,
        image: np.ndarray,
    ) -> list[tuple[str, np.ndarray, int]]:
        """
        Generate preprocessing variants.
        """

        variants: list[
            tuple[str, np.ndarray, int]
        ] = []

        # ---------------------------------------------------------------
        # Base grayscale
        # ---------------------------------------------------------------

        if len(image.shape) == 3:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )
        else:
            gray = image.copy()

        # ---------------------------------------------------------------
        # Upscale
        # ---------------------------------------------------------------

        if self.upscale > 1.0:

            upscaled = cv2.resize(
                gray,
                None,
                fx=self.upscale,
                fy=self.upscale,
                interpolation=cv2.INTER_CUBIC,
            )

        else:
            upscaled = gray.copy()

        # ---------------------------------------------------------------
        # Processing variants
        # ---------------------------------------------------------------

        processed_variants: list[
            tuple[str, np.ndarray]
        ] = []

        processed_variants.append(
            ("gray", upscaled)
        )

        # ---------------------------------------------------------------
        # CLAHE
        # ---------------------------------------------------------------

        if self.clahe_enabled:

            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=(
                    self.clahe_tile_grid_size,
                    self.clahe_tile_grid_size,
                ),
            )

            clahe_image = clahe.apply(upscaled)

            processed_variants.append(
                ("clahe", clahe_image)
            )

        # ---------------------------------------------------------------
        # Sharpening
        # ---------------------------------------------------------------

        if self.sharpening_enabled:

            blurred = cv2.GaussianBlur(
                upscaled,
                (0, 0),
                self.sharpening_sigma,
            )

            sharpened = cv2.addWeighted(
                upscaled,
                self.sharpening_amount,
                blurred,
                -(self.sharpening_amount - 1.0),
                0,
            )

            processed_variants.append(
                ("sharpened", sharpened)
            )

        # ---------------------------------------------------------------
        # Otsu
        # ---------------------------------------------------------------

        if self.otsu_enabled:

            _, otsu = cv2.threshold(
                upscaled,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

            processed_variants.append(
                ("otsu", otsu)
            )

            _, otsu_inv = cv2.threshold(
                upscaled,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )

            processed_variants.append(
                ("otsu_inv", otsu_inv)
            )

        # ---------------------------------------------------------------
        # Adaptive threshold
        # ---------------------------------------------------------------

        if self.adaptive_enabled:

            adaptive = cv2.adaptiveThreshold(
                upscaled,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                7,
            )

            processed_variants.append(
                ("adaptive", adaptive)
            )

            adaptive_inv = cv2.adaptiveThreshold(
                upscaled,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                31,
                7,
            )

            processed_variants.append(
                ("adaptive_inv", adaptive_inv)
            )

        # ---------------------------------------------------------------
        # Fixed thresholds
        # ---------------------------------------------------------------

        if self.fixed_threshold_enabled:

            for threshold in self.fixed_threshold_values:

                _, fixed = cv2.threshold(
                    upscaled,
                    threshold,
                    255,
                    cv2.THRESH_BINARY,
                )

                processed_variants.append(
                    (
                        f"threshold_{threshold}",
                        fixed,
                    )
                )

        # ---------------------------------------------------------------
        # Add rotations
        # ---------------------------------------------------------------

        for variant_name, variant_image in processed_variants:

            if not self.rotation_enabled:

                variants.append(
                    (
                        variant_name,
                        variant_image,
                        0,
                    )
                )

                if len(variants) >= self.max_variants:
                    break

                continue

            for angle in self.rotation_angles:

                rotated = self._rotate(
                    variant_image,
                    angle,
                )

                variants.append(
                    (
                        f"{variant_name}_rot{angle}",
                        rotated,
                        angle,
                    )
                )

                if len(variants) >= self.max_variants:
                    break

            if len(variants) >= self.max_variants:
                break

        return variants

    # ===================================================================
    # Rotation
    # ===================================================================

    @staticmethod
    def _rotate(
        image: np.ndarray,
        angle: int,
    ) -> np.ndarray:

        angle = angle % 360

        if angle == 0:
            return image

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

        # Generic angle support
        height, width = image.shape[:2]

        center = (
            width / 2.0,
            height / 2.0,
        )

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0,
        )

        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    # ===================================================================
    # Decode
    # ===================================================================

    def _decode_image(
        self,
        image: np.ndarray,
        rotation: int,
        variant_name: str,
    ) -> list[BarcodeResult]:
        """
        Decode one preprocessed image.
        """

        if decode is None:
            return []

        symbols = self._get_symbols()

        try:
            decoded_objects = decode(
                image,
                symbols=symbols,
            )
        except TypeError:
            # Some pyzbar versions do not accept symbols=None.
            try:
                decoded_objects = decode(image)
            except Exception:
                return []
        except Exception:
            return []

        results: list[BarcodeResult] = []

        for obj in decoded_objects:

            try:
                value = obj.data.decode(
                    "utf-8",
                    errors="ignore",
                ).strip()
            except Exception:
                try:
                    value = obj.data.decode(
                        "latin-1",
                        errors="ignore",
                    ).strip()
                except Exception:
                    value = str(obj.data)

            if not value:
                continue

            barcode_format = str(
                obj.type or ""
            ).upper()

            rect = obj.rect

            confidence = self._estimate_confidence(
                obj,
                barcode_format,
                value,
                image.shape,
            )

            results.append(
                BarcodeResult(
                    value=value,
                    format=barcode_format,
                    confidence=confidence,
                    x=int(rect.left),
                    y=int(rect.top),
                    width=int(rect.width),
                    height=int(rect.height),
                    rotation=rotation,
                    variant=variant_name,
                )
            )

        return results

    # ===================================================================
    # PyZBar symbols
    # ===================================================================

    def _get_symbols(self):
        """
        Return PyZBar symbols for configured formats.

        If symbol mapping fails, return None and allow
        PyZBar to decode all available formats.
        """

        if ZBarSymbol is None:
            return None

        formats = []

        if self.primary_format:
            formats.append(
                self.primary_format
            )

        if self.fallback_enabled:
            formats.extend(
                self.fallback_formats
            )

        mapping = {
            "CODE128": "CODE128",
            "CODE39": "CODE39",
            "CODE93": "CODE93",
            "I25": "I25",
            "INTERLEAVED2OF5": "I25",
            "EAN13": "EAN13",
            "EAN8": "EAN8",
            "UPCA": "UPCA",
            "UPCE": "UPCE",
        }

        symbols = []

        for fmt in formats:

            enum_name = mapping.get(fmt)

            if enum_name is None:
                continue

            try:
                symbol = getattr(
                    ZBarSymbol,
                    enum_name,
                )

                if symbol not in symbols:
                    symbols.append(symbol)

            except AttributeError:
                continue

        return symbols or None

    # ===================================================================
    # Confidence
    # ===================================================================

    def _estimate_confidence(
        self,
        obj,
        barcode_format: str,
        value: str,
        image_shape,
    ) -> float:
        """
        Estimate a practical confidence score.

        PyZBar itself does not provide a native confidence value.
        This score combines format priority, decoded value quality,
        and barcode region size.
        """

        confidence = 0.60

        # ---------------------------------------------------------------
        # Primary format bonus
        # ---------------------------------------------------------------

        if barcode_format == self.primary_format:
            confidence += 0.20

        elif barcode_format in self.fallback_formats:
            confidence += 0.05

        # ---------------------------------------------------------------
        # Value quality
        # ---------------------------------------------------------------

        if value:

            if len(value) >= 4:
                confidence += 0.05

            if len(value) >= 8:
                confidence += 0.05

            if all(
                char.isprintable()
                for char in value
            ):
                confidence += 0.03

        # ---------------------------------------------------------------
        # Barcode size
        # ---------------------------------------------------------------

        try:
            image_height, image_width = image_shape[:2]

            area = (
                float(obj.rect.width)
                * float(obj.rect.height)
            )

            image_area = (
                float(image_width)
                * float(image_height)
            )

            if image_area > 0:

                area_ratio = area / image_area

                if area_ratio >= 0.01:
                    confidence += 0.05

                if area_ratio >= 0.03:
                    confidence += 0.05

        except Exception:
            pass

        return min(
            max(confidence, 0.0),
            1.0,
        )

    # ===================================================================
    # Strong result
    # ===================================================================

    def _has_strong_primary_result(
        self,
        results: list[BarcodeResult],
    ) -> bool:

        for result in results:

            if (
                result.format == self.primary_format
                and result.confidence
                >= self.minimum_confidence
            ):
                return True

        return False

    # ===================================================================
    # Deduplication
    # ===================================================================

    def _deduplicate_results(
        self,
        results: list[BarcodeResult],
    ) -> list[BarcodeResult]:
        """
        Deduplicate same barcode values.
        Keep the strongest observation.
        """

        best: dict[
            tuple[str, str],
            BarcodeResult
        ] = {}

        for result in results:

            key = (
                result.value.strip().upper(),
                result.format.upper(),
            )

            existing = best.get(key)

            if existing is None:

                best[key] = result

            elif result.confidence > existing.confidence:

                best[key] = result

        return list(best.values())

    # ===================================================================
    # Consensus
    # ===================================================================

    def _apply_consensus(
        self,
        results: list[BarcodeResult],
    ) -> list[BarcodeResult]:
        """
        Strengthen confidence when the same barcode is detected
        across multiple preprocessing variants.
        """

        if not results:
            return []

        grouped: dict[
            tuple[str, str],
            list[BarcodeResult]
        ] = {}

        for result in results:

            key = (
                result.value.strip().upper(),
                result.format.upper(),
            )

            grouped.setdefault(
                key,
                [],
            ).append(result)

        final_results: list[BarcodeResult] = []

        for _, observations in grouped.items():

            observations.sort(
                key=lambda item: item.confidence,
                reverse=True,
            )

            strongest = observations[0]

            observation_count = len(
                observations
            )

            # Multiple independent preprocessing
            # variants agreeing increases confidence.
            consensus_bonus = min(
                0.15,
                (observation_count - 1) * 0.03,
            )

            confidence = min(
                1.0,
                strongest.confidence
                + consensus_bonus,
            )

            final_results.append(
                BarcodeResult(
                    value=strongest.value,
                    format=strongest.format,
                    confidence=confidence,
                    x=strongest.x,
                    y=strongest.y,
                    width=strongest.width,
                    height=strongest.height,
                    rotation=strongest.rotation,
                    variant=strongest.variant,
                )
            )

        return final_results

    # ===================================================================
    # Statistics
    # ===================================================================

    def status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "engine": self.engine_name,
            "primary_format": self.primary_format,
            "fallback_enabled": self.fallback_enabled,
            "fallback_formats": self.fallback_formats,
            "upscale": self.upscale,
            "max_variants": self.max_variants,
            "rotation_enabled": self.rotation_enabled,
            "rotation_angles": self.rotation_angles,
            "clahe_enabled": self.clahe_enabled,
            "sharpening_enabled": self.sharpening_enabled,
            "otsu_enabled": self.otsu_enabled,
            "adaptive_enabled": self.adaptive_enabled,
            "fixed_threshold_enabled": (
                self.fixed_threshold_enabled
            ),
            "consensus_enabled": (
                self.consensus_enabled
            ),
            "minimum_confidence": (
                self.minimum_confidence
            ),
            "pyzbar_available": (
                self.pyzbar_available
            ),
            "total_reads": self.total_reads,
            "successful_reads": (
                self.successful_reads
            ),
            "last_elapsed_ms": (
                round(self.last_elapsed_ms, 2)
            ),
        }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Standalone barcode engine test.

    Usage:

        python -m app.barcode.barcode_engine

    or:

        python -m app.barcode.barcode_engine "image.jpg"
    """

    print("\n" + "=" * 60)
    print("OCR_BHS - BARCODE ENGINE TEST")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------

    try:
        config = load_config()

    except Exception as exc:

        print("\nCONFIGURATION ERROR")
        print("-" * 60)
        print(f"{type(exc).__name__}: {exc}")

        return 1

    # ------------------------------------------------------------------
    # Create engine
    # ------------------------------------------------------------------

    try:
        engine = BarcodeEngine(config)

    except Exception as exc:

        print("\nENGINE INITIALIZATION FAILED")
        print("-" * 60)
        print(f"{type(exc).__name__}: {exc}")

        return 1

    # ------------------------------------------------------------------
    # Configuration information
    # ------------------------------------------------------------------

    print(f"Enabled       : {engine.enabled}")
    print(f"Engine        : {engine.engine_name}")
    print(f"Primary       : {engine.primary_format}")
    print(f"Fallback      : {engine.fallback_enabled}")
    print(f"Upscale       : {engine.upscale}")
    print(f"Max variants  : {engine.max_variants}")
    print(f"Rotations     : {engine.rotation_angles}")
    print(f"Consensus     : {engine.consensus_enabled}")
    print(
        f"Min confidence: "
        f"{engine.minimum_confidence}"
    )
    print(f"PyZBar        : {engine.pyzbar_available}")

    # ------------------------------------------------------------------
    # Resolve image path
    # ------------------------------------------------------------------

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

    print("\n" + "-" * 60)
    print("INPUT IMAGE")
    print("-" * 60)

    print(f"Path          : {image_path}")

    # ------------------------------------------------------------------
    # Check image
    # ------------------------------------------------------------------

    if not image_path.exists():

        print("\nRESULT        : FAILED")
        print("ERROR         : Image file does not exist.")

        return 1

    # ------------------------------------------------------------------
    # Read image
    # ------------------------------------------------------------------

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:

        print("\nRESULT        : FAILED")
        print(
            "ERROR         : OpenCV could not read "
            "the image."
        )

        return 1

    height, width = image.shape[:2]

    print(f"Image size    : {width} x {height}")

    # ------------------------------------------------------------------
    # Run barcode reader
    # ------------------------------------------------------------------

    print("\n" + "-" * 60)
    print("RUNNING BARCODE DETECTION")
    print("-" * 60)

    start = time.perf_counter()

    try:

        results = engine.read(image)

    except Exception as exc:

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        print(
            f"\nElapsed       : "
            f"{elapsed_ms:.2f} ms"
        )

        print("\nRESULT        : FAILED")
        print(
            f"ERROR         : "
            f"{type(exc).__name__}: {exc}"
        )

        import traceback

        traceback.print_exc()

        return 1

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    # ------------------------------------------------------------------
    # Print timing
    # ------------------------------------------------------------------

    print(
        f"\nElapsed       : "
        f"{elapsed_ms:.2f} ms"
    )

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------

    print("\n" + "-" * 60)

    if not results:

        print("NO BARCODE DETECTED")

        print("-" * 60)

        print("\nPossible reasons:")
        print("  1. Barcode is too small.")
        print("  2. Barcode is blurred.")
        print("  3. Barcode is distorted.")
        print("  4. Barcode region needs cropping.")
        print("  5. Camera image quality is insufficient.")
        print("  6. CODE128 bars are not clear enough.")
        print("  7. More preprocessing may be required.")

    else:

        print(
            f"BARCODE DETECTED "
            f"({len(results)} result(s))"
        )

        print("-" * 60)

        for index, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\nBarcode #{index}"
            )

            print(
                f"  Value      : "
                f"{result.value}"
            )

            print(
                f"  Format     : "
                f"{result.format}"
            )

            print(
                f"  Confidence : "
                f"{result.confidence:.3f}"
            )

            print(
                f"  BoundingBox: "
                f"({result.x}, {result.y}, "
                f"{result.width}, {result.height})"
            )

            print(
                f"  Rotation   : "
                f"{result.rotation}°"
            )

            print(
                f"  Variant    : "
                f"{result.variant}"
            )

    # ------------------------------------------------------------------
    # Engine status
    # ------------------------------------------------------------------

    print("\n" + "-" * 60)
    print("ENGINE STATUS")
    print("-" * 60)

    status = engine.status()

    print(
        f"Total reads   : "
        f"{status['total_reads']}"
    )

    print(
        f"Successful     : "
        f"{status['successful_reads']}"
    )

    print(
        f"Last elapsed   : "
        f"{status['last_elapsed_ms']} ms"
    )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    print("\n" + "=" * 60)

    if results:

        print("BARCODE TEST RESULT: SUCCESS")

    else:

        print("BARCODE TEST RESULT: NO BARCODE")

    print("=" * 60)

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())