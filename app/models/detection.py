"""
OCR_BHS - Detection Model
=========================

Common detection object shared by:

    YOLODetector
        |
        v
    TagTracker
        |
        v
    OCR / Barcode processing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


BBox = Tuple[float, float, float, float]


@dataclass
class Detection:
    """
    Single YOLO detection.

    bbox format:
        (x1, y1, x2, y2)
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: BBox

    # ================================================================
    # GEOMETRY
    # ================================================================

    @property
    def x1(self) -> float:
        return float(self.bbox[0])

    @property
    def y1(self) -> float:
        return float(self.bbox[1])

    @property
    def x2(self) -> float:
        return float(self.bbox[2])

    @property
    def y2(self) -> float:
        return float(self.bbox[3])

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.x1 + self.x2) / 2.0,
            (self.y1 + self.y2) / 2.0,
        )

    # ================================================================
    # INTEGER BBOX
    # ================================================================

    @property
    def as_xyxy_int(self) -> Tuple[int, int, int, int]:
        return (
            int(round(self.x1)),
            int(round(self.y1)),
            int(round(self.x2)),
            int(round(self.y2)),
        )

    # ================================================================
    # DICTIONARY
    # ================================================================

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "center": self.center,
            "width": self.width,
            "height": self.height,
            "area": self.area,
        }

    # ================================================================
    # STRING
    # ================================================================

    def __repr__(self) -> str:
        return (
            "Detection("
            f"class_id={self.class_id}, "
            f"class_name='{self.class_name}', "
            f"confidence={self.confidence:.3f}, "
            f"bbox={self.bbox}"
            ")"
        )