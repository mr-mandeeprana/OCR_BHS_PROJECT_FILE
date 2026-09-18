"""
OCR_BHS - Result Models
=======================

Compatibility and application-level result models.

The canonical Detection model lives in:

    app.models.detection

Detection is re-exported here because older OCR_BHS modules
historically imported it from app.models.result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from app.models.detection import Detection


@dataclass
class OCRProcessingResult:
    """
    Standardized OCR processing result.
    """

    success: bool = False
    text: str = ""
    confidence: float = 0.0
    engine: str = ""
    error: Optional[str] = None
    rotation: str = "0deg"
    details: list[dict[str, Any]] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "text": self.text,
            "confidence": self.confidence,
            "engine": self.engine,
            "error": self.error,
            "rotation": self.rotation,
            "details": self.details,
        }


@dataclass
class BarcodeProcessingResult:
    """
    Standardized barcode processing result.
    """

    success: bool = False
    data: str = ""
    format: str = ""
    confidence: float = 0.0
    rotation: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "format": self.format,
            "confidence": self.confidence,
            "rotation": self.rotation,
            "error": self.error,
        }


@dataclass
class ProcessingResult:
    """
    Combined result associated with one camera/track.
    """

    camera_id: str
    track_id: int

    timestamp: str = field(
        default_factory=lambda:
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    frame_id: Optional[int] = None

    ocr_text: str = ""
    ocr_confidence: float = 0.0

    barcode_data: str = ""
    barcode_type: str = ""
    barcode_confidence: float = 0.0

    quality_score: float = 0.0

    readable: bool = False
    confidence: float = 0.0

    reason: str = "NOT_PROCESSED"

    ocr_success: bool = False
    barcode_success: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "ocr_text": self.ocr_text,
            "ocr_confidence": self.ocr_confidence,
            "barcode_data": self.barcode_data,
            "barcode_type": self.barcode_type,
            "barcode_confidence": self.barcode_confidence,
            "quality_score": self.quality_score,
            "readable": self.readable,
            "confidence": self.confidence,
            "reason": self.reason,
            "ocr_success": self.ocr_success,
            "barcode_success": self.barcode_success,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            "ProcessingResult("
            f"camera={self.camera_id!r}, "
            f"track={self.track_id}, "
            f"readable={self.readable}, "
            f"confidence={self.confidence:.3f}, "
            f"ocr={self.ocr_text!r}, "
            f"barcode={self.barcode_data!r}"
            ")"
        )


__all__ = [
    "Detection",
    "OCRProcessingResult",
    "BarcodeProcessingResult",
    "ProcessingResult",
]