"""
OCR_BHS - Tracking Models
=========================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.models.detection import Detection


@dataclass
class Track:
    """
    Project-level representation of one tracked IATA tag.
    """

    track_id: int

    detection: Detection

    last_frame_id: int

    missed_frames: int = 0

    age: int = 1

    hits: int = 1

    previous_center: Optional[
        tuple[float, float]
    ] = None

    current_center: Optional[
        tuple[float, float]
    ] = None

    velocity_x: float = 0.0

    velocity_y: float = 0.0

    track_confidence: float = 0.0

    is_confirmed: bool = False

    is_predicted: bool = False

    unstable: bool = False

    instability_count: int = 0

    iou: float = 1.0

    speed: float = 0.0

    distance: float = 0.0

    history: list[
        tuple[int, tuple[float, float]]
    ] = field(
        default_factory=list
    )

    @property
    def center(
        self,
    ) -> tuple[float, float]:

        if self.current_center is not None:
            return self.current_center

        return (
            (
                self.detection.x1
                + self.detection.x2
            ) / 2.0,

            (
                self.detection.y1
                + self.detection.y2
            ) / 2.0,
        )

    @property
    def bbox(
        self,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:

        return self.detection.bbox

    @property
    def x1(self) -> float:
        return self.detection.x1

    @property
    def y1(self) -> float:
        return self.detection.y1

    @property
    def x2(self) -> float:
        return self.detection.x2

    @property
    def y2(self) -> float:
        return self.detection.y2

    @property
    def direction(self) -> str:

        dx = self.velocity_x
        dy = self.velocity_y

        if (
            abs(dx) < 0.5
            and abs(dy) < 0.5
        ):
            return "stationary"

        if abs(dx) >= abs(dy):

            return (
                "right"
                if dx > 0
                else "left"
            )

        return (
            "down"
            if dy > 0
            else "up"
        )

    @property
    def status(self) -> str:

        if self.is_predicted:
            return "PREDICTED"

        if self.is_confirmed:
            return "DETECTED"

        return "TENTATIVE"

    def to_dict(self) -> dict:

        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "center": self.center,
            "last_frame_id": self.last_frame_id,
            "missed_frames": self.missed_frames,
            "age": self.age,
            "hits": self.hits,
            "velocity_x": self.velocity_x,
            "velocity_y": self.velocity_y,
            "speed": self.speed,
            "distance": self.distance,
            "track_confidence": self.track_confidence,
            "is_confirmed": self.is_confirmed,
            "is_predicted": self.is_predicted,
            "unstable": self.unstable,
            "instability_count": self.instability_count,
            "iou": self.iou,
            "direction": self.direction,
            "status": self.status,
        }


__all__ = [
    "Track",
]