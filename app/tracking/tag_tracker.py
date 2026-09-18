"""
OCR_BHS Continuous IATA Tag Tracker
====================================

YOLO26
   |
   v
IATA_TAG detections
   |
   v
supervision.ByteTrack
   |
   v
Stable Track ID
   |
   v
Motion / stability tracking
   |
   v
Continuous bounding box
   |
   v
ROI capture

Important:
- Only class 0 (IATA_TAG) is tracked.
- ByteTrack state belongs to this tracker instance.
- Temporary YOLO detection gaps do NOT immediately remove the visual
  track (see CONTINUOUS PREDICTION below).
- The last known motion is used to predict the box during short gaps.

--------------------------------------------------------------------
WHY THIS VERSION IS DIFFERENT
--------------------------------------------------------------------
The previous implementation drove the raw ``ultralytics.trackers
.byte_tracker.BYTETracker`` class directly. That class is an internal
implementation detail of Ultralytics (not a public/stable API): it
expects a hand-built ``Results``-like object, returns a bare numpy
array whose column layout is undocumented, and takes a
``SimpleNamespace`` of args whose accepted keys silently vary between
ultralytics releases. Every one of those assumptions was a place a
silent, hard-to-diagnose bug could hide -- and in this project's
debugging history, that's exactly what happened (guessed argument
names, guessed column order, etc).

This version instead uses ``supervision.ByteTrack``, a small,
stable, public wrapper around the same ByteTrack algorithm. It takes
a documented ``Detections`` object in and returns a documented
``Detections`` object out (``.xyxy``, ``.confidence``, ``.class_id``,
``.tracker_id`` -- all named attributes, never positional columns).
This is the same pattern already proven in production on this
person's other project (FillPac AI bag tracker).

All of the IATA-specific logic on top (motion smoothing, prediction
through missed YOLO frames, quality filtering, instability
detection) is unchanged in behavior from the previous version.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np

# supervision emits a FutureWarning that ByteTrack will be renamed in
# a future release. The class is fully functional today; we only
# silence the warning so it doesn't spam production logs every frame.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from supervision import ByteTrack, Detections

from app.models.detection import Detection

# ============================================================================
# TRACK MODEL
#
# Track is defined ONCE in app.models.tracking and shared project-wide
# (tag_tracker, pipeline, result_manager, etc).
# ============================================================================

from app.models.tracking import Track


# ============================================================================
# TAG TRACKER
# ============================================================================


class TagTracker:
    """
    Continuous ByteTrack tracker specialized for IATA tags.

    Pipeline:
        - ByteTrack (via supervision)
        - Track confirmation (min_hits)
        - Motion smoothing / instability detection
        - Continuous prediction through missed YOLO frames
        - Quality filtering (confidence + bbox size) on the OUTPUT
          side only -- ByteTrack's own internal state is always fed
          every observation, filtered or not, so a temporarily
          low-quality box never corrupts the track's identity.
    """

    IATA_TAG_CLASS_ID = 0

    def __init__(
        self,
        track_high_thresh: float = 0.30,
        track_low_thresh: float = 0.05,
        new_track_thresh: float = 0.30,
        track_buffer: int = 45,
        match_thresh: float = 0.85,
        fuse_score: bool = True,
        min_hits: int = 3,
        # Number of missed inference frames for visual prediction.
        prediction_buffer: int = 18,
        # Motion quality.
        max_jump_distance: float = 250.0,
        min_iou_warning: float = 0.10,
        unstable_frame_threshold: int = 3,
        # Frame dimensions are optional.
        frame_width: int = 0,
        frame_height: int = 0,
        # Minimum tag size.
        min_bbox_area: float = 0.0,
        min_bbox_area_ratio: float = 0.0,
        min_track_confidence: float = 0.25,
        # ---------------------------------------------------------------
        # Backward compatibility with old OCR_BHS live_pipeline.py
        # ---------------------------------------------------------------
        iou_threshold: float = 0.15,
        max_missed_frames: int = 12,
        max_center_distance: float = 180.0,
        min_size_similarity: float = 0.20,
        frame_rate: float = 30.0,
        # ---------------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------------
        # Every N frames, print a one-line summary of how many
        # observations passed/failed the quality filter and why. This
        # is what previously required manually patching in print()
        # statements to diagnose "zero detections" issues -- it is
        # now always available. Set to 0 to disable.
        quality_report_every: int = 30,
    ) -> None:

        # ==================================================================
        # BYTETRACK CONFIG
        # ==================================================================

        self.track_high_thresh = float(track_high_thresh)
        self.track_low_thresh = float(track_low_thresh)
        self.new_track_thresh = float(new_track_thresh)
        self.track_buffer = max(1, int(track_buffer))
        self.match_thresh = float(match_thresh)
        self.fuse_score = bool(fuse_score)

        # ==================================================================
        # TRACK CONFIRMATION
        # ==================================================================

        self.min_hits = max(1, int(min_hits))

        # ==================================================================
        # CONTINUOUS TRACKING
        # ==================================================================

        self.prediction_buffer = max(1, int(prediction_buffer))

        # ==================================================================
        # MOTION QUALITY
        # ==================================================================

        self.max_jump_distance = max(0.0, float(max_jump_distance))
        self.min_iou_warning = min(max(0.0, float(min_iou_warning)), 1.0)
        self.unstable_frame_threshold = max(1, int(unstable_frame_threshold))

        # ==================================================================
        # FRAME SIZE
        # ==================================================================

        self.frame_width = max(0, int(frame_width or 0))
        self.frame_height = max(0, int(frame_height or 0))

        # ==================================================================
        # QUALITY FILTERS
        # ==================================================================

        self.min_bbox_area = max(0.0, float(min_bbox_area))
        self.min_bbox_area_ratio = max(0.0, float(min_bbox_area_ratio))
        self.min_track_confidence = min(
            max(0.0, float(min_track_confidence)), 1.0
        )

        # ==================================================================
        # BACKWARD COMPATIBILITY
        # ==================================================================

        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = max(1, int(max_missed_frames))
        self.max_center_distance = max(0.0, float(max_center_distance))
        self.min_size_similarity = max(0.0, float(min_size_similarity))
        self.frame_rate = max(1.0, float(frame_rate))

        # ==================================================================
        # FRAME COUNTER
        # ==================================================================

        self.frame_index = 0

        # ==================================================================
        # PROJECT TRACK MEMORY
        # ==================================================================

        self._tracks: dict[int, Track] = {}

        self.track_age: dict[int, int] = {}
        self.track_last_seen: dict[int, int] = {}
        self.track_last_center: dict[int, tuple[float, float]] = {}
        self.track_last_bbox: dict[
            int, tuple[float, float, float, float]
        ] = {}
        self.track_velocity: dict[int, tuple[float, float]] = {}
        self.track_speed: dict[int, float] = {}
        self.track_distance: dict[int, float] = {}
        self.track_iou: dict[int, float] = {}
        self.track_unstable: dict[int, bool] = {}
        self.track_unstable_count: dict[int, int] = {}
        self.track_motion_jump: dict[int, bool] = {}

        # ==================================================================
        # QUALITY DIAGNOSTICS (reporting window counters)
        # ==================================================================

        self.quality_report_every = max(0, int(quality_report_every))
        self._reset_quality_window()

        # ==================================================================
        # SUPERVISION BYTETRACK
        # ==================================================================

        self._tracker = self._create_bytetrack()

    # ======================================================================
    # FRAME SIZE
    # ======================================================================

    def set_frame_size(self, width: int, height: int) -> None:
        width = int(width or 0)
        height = int(height or 0)
        if width > 0:
            self.frame_width = width
        if height > 0:
            self.frame_height = height

    # ======================================================================
    # CREATE BYTE TRACK
    # ======================================================================

    def _create_bytetrack(self) -> ByteTrack:
        """
        Build the underlying supervision.ByteTrack instance.

        ``track_activation_threshold`` is ByteTrack's high-confidence
        threshold (equivalent to the old ``track_high_thresh``) --
        detections at/above it can start a brand-new track; detections
        below it are still used to keep an existing track alive
        (ByteTrack's built-in two-stage association), but never spawn
        one. We separately pre-filter raw input at ``track_low_thresh``
        before it ever reaches the tracker, purely to keep pure noise
        out.

        ``minimum_consecutive_frames=1`` keeps ByteTrack's own
        confirmation gate a no-op -- track confirmation is handled by
        this class's own ``min_hits`` counter instead, exactly as
        before, so behavior doesn't shift underneath existing config.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return ByteTrack(
                track_activation_threshold=self.track_high_thresh,
                lost_track_buffer=self.track_buffer,
                minimum_matching_threshold=self.match_thresh,
                frame_rate=self.frame_rate,
                minimum_consecutive_frames=1,
            )

    # ======================================================================
    # UPDATE
    # ======================================================================

    def update(
        self,
        detections: list[Detection],
        frame_id: int,
        timestamp=None,
    ) -> list[Track]:

        del timestamp

        self.frame_index = int(frame_id)

        # ==================================================================
        # ONLY IATA TAGS, ABOVE THE NOISE FLOOR
        # ==================================================================

        iata_detections: list[Detection] = []

        for detection in detections or []:

            if int(detection.class_id) != self.IATA_TAG_CLASS_ID:
                continue

            confidence = float(detection.confidence)

            if confidence < self.track_low_thresh:
                continue

            if detection.x2 <= detection.x1 or detection.y2 <= detection.y1:
                continue

            iata_detections.append(detection)

        # ==================================================================
        # BUILD SUPERVISION DETECTIONS
        # ==================================================================

        if iata_detections:

            xyxy = np.asarray(
                [
                    [d.x1, d.y1, d.x2, d.y2]
                    for d in iata_detections
                ],
                dtype=np.float32,
            )

            confidence_arr = np.asarray(
                [float(d.confidence) for d in iata_detections],
                dtype=np.float32,
            )

            class_id_arr = np.zeros(len(iata_detections), dtype=int)

            detections_sv = Detections(
                xyxy=xyxy,
                confidence=confidence_arr,
                class_id=class_id_arr,
            )

        else:

            detections_sv = Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=int),
            )

        # ==================================================================
        # RUN BYTETRACK
        #
        # IMPORTANT: this call happens EVERY frame, including when
        # there are zero detections. ByteTrack's own lost-track
        # lifecycle (how long a track survives without a match) only
        # advances correctly if it is called every frame -- skipping
        # the call on empty frames would make tracks appear to live
        # far longer (or behave inconsistently) than the configured
        # track_buffer.
        # ==================================================================

        try:
            tracked = self._tracker.update_with_detections(detections_sv)
        except Exception as exc:
            # Never let a tracker-internal error silently swallow the
            # rest of the pipeline. Surface it once per occurrence.
            print(f"[TagTracker] ByteTrack update failed: {exc!r}")
            tracked = None

        # ==================================================================
        # PROCESS BYTETRACK OUTPUT
        # ==================================================================

        updated_ids: set[int] = set()
        current_tracks: list[Track] = []

        num_tracked = 0 if tracked is None else len(tracked)

        for index in range(num_tracked):

            if tracked.tracker_id is None:
                break

            raw_track_id = tracked.tracker_id[index]

            if raw_track_id is None or raw_track_id < 0:
                continue

            track_id = int(raw_track_id)

            x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[index])

            if tracked.confidence is not None:
                score = float(tracked.confidence[index])
            else:
                score = 0.0

            if tracked.class_id is not None:
                class_id = int(tracked.class_id[index])
                if class_id != self.IATA_TAG_CLASS_ID:
                    continue

            updated_ids.add(track_id)

            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            bbox = (x1, y1, x2, y2)

            detection_confidence = score

            detection = Detection(
                class_id=0,
                class_name="IATA_TAG",
                confidence=detection_confidence,
                bbox=bbox,
            )

            # --------------------------------------------------------------
            # Existing memory
            # --------------------------------------------------------------

            previous_center = self.track_last_center.get(track_id)
            previous_bbox = self.track_last_bbox.get(track_id)

            # --------------------------------------------------------------
            # Track age -- counts every ByteTrack observation,
            # regardless of whether it later passes the quality filter
            # below. This MUST happen before the quality filter, or a
            # track that briefly dips below the quality bar would
            # appear to "reset" its age/motion history once it comes
            # back, and a later frame's box would get compared against
            # a stale reference point and look like a false motion
            # jump.
            # --------------------------------------------------------------

            self.track_age[track_id] = self.track_age.get(track_id, 0) + 1
            self.track_last_seen[track_id] = self.frame_index

            # --------------------------------------------------------------
            # Motion memory (always updated, before quality filtering)
            # --------------------------------------------------------------

            self._update_motion_memory(
                track_id=track_id,
                center=center,
                bbox=bbox,
                previous_center=previous_center,
                previous_bbox=previous_bbox,
            )

            # --------------------------------------------------------------
            # Project Track object
            # --------------------------------------------------------------

            old_track = self._tracks.get(track_id)

            if old_track is None:

                track = Track(
                    track_id=track_id,
                    detection=detection,
                    last_frame_id=self.frame_index,
                    missed_frames=0,
                    age=1,
                    hits=1,
                    previous_center=None,
                    current_center=center,
                    velocity_x=0.0,
                    velocity_y=0.0,
                    track_confidence=score,
                    is_confirmed=(self.min_hits <= 1),
                    is_predicted=False,
                    unstable=self.track_unstable.get(track_id, False),
                    instability_count=self.track_unstable_count.get(
                        track_id, 0
                    ),
                    iou=self.track_iou.get(track_id, 1.0),
                    speed=self.track_speed.get(track_id, 0.0),
                    distance=self.track_distance.get(track_id, 0.0),
                )

            else:

                old_track.previous_center = old_track.current_center
                old_track.current_center = center
                old_track.detection = detection
                old_track.last_frame_id = self.frame_index
                old_track.missed_frames = 0
                old_track.age += 1
                old_track.hits += 1
                old_track.track_confidence = score

                velocity = self.track_velocity.get(track_id, (0.0, 0.0))
                old_track.velocity_x = velocity[0]
                old_track.velocity_y = velocity[1]

                old_track.is_predicted = False
                old_track.unstable = self.track_unstable.get(
                    track_id, False
                )
                old_track.instability_count = self.track_unstable_count.get(
                    track_id, 0
                )
                old_track.iou = self.track_iou.get(track_id, 1.0)
                old_track.speed = self.track_speed.get(track_id, 0.0)
                old_track.distance = self.track_distance.get(track_id, 0.0)

                if old_track.hits >= self.min_hits:
                    old_track.is_confirmed = True

                track = old_track

            self._tracks[track_id] = track

            # --------------------------------------------------------------
            # Quality filtering (output-only -- see docstring above)
            # --------------------------------------------------------------

            if not self._passes_quality(
                bbox=bbox,
                confidence=detection_confidence,
            ):
                continue

            current_tracks.append(track)

        # ==================================================================
        # CONTINUOUS PREDICTION FOR MISSING YOLO DETECTIONS
        # ==================================================================

        for track_id, track in list(self._tracks.items()):

            if track_id in updated_ids:
                continue

            track.missed_frames += 1

            # Don't predict forever.
            if track.missed_frames > self.prediction_buffer:
                del self._tracks[track_id]
                continue

            # Don't predict an unconfirmed track.
            if not track.is_confirmed:
                continue

            center = track.current_center

            if center is None:
                continue

            # Motion prediction. Velocity is damped as missed frames
            # increase, preventing the box from flying away.
            missed = track.missed_frames
            damping = max(0.25, 1.0 - (0.08 * missed))

            predicted_x = center[0] + (track.velocity_x * damping)
            predicted_y = center[1] + (track.velocity_y * damping)

            width = max(1.0, track.detection.x2 - track.detection.x1)
            height = max(1.0, track.detection.y2 - track.detection.y1)

            predicted_bbox = (
                predicted_x - width / 2.0,
                predicted_y - height / 2.0,
                predicted_x + width / 2.0,
                predicted_y + height / 2.0,
            )

            predicted_confidence = max(
                0.05, track.track_confidence * (0.96 ** missed)
            )

            track.detection = Detection(
                class_id=0,
                class_name="IATA_TAG",
                confidence=predicted_confidence,
                bbox=predicted_bbox,
            )

            track.previous_center = center
            track.current_center = (predicted_x, predicted_y)
            track.last_frame_id = self.frame_index
            track.track_confidence = predicted_confidence
            track.is_predicted = True

            current_tracks.append(track)

        # ==================================================================
        # SORT
        # ==================================================================

        current_tracks.sort(key=lambda item: item.track_id)

        # ==================================================================
        # HISTORY CLEANUP
        # ==================================================================

        self._trim_history()

        # ==================================================================
        # PERIODIC QUALITY DIAGNOSTICS
        # ==================================================================

        self._report_quality_window()

        return current_tracks

    # ======================================================================
    # QUALITY
    # ======================================================================

    def _passes_quality(self, bbox, confidence) -> bool:
        """
        Decide whether an observation is good enough to be reported
        this frame (it always still updates ByteTrack/motion state
        above, regardless of this result).

        Every decision is tallied into the reporting window so
        ``_report_quality_window`` can print a summary -- no more
        silent, invisible rejects.
        """

        self._quality_window["total"] += 1

        if confidence < self.min_track_confidence:
            self._quality_window["reject_confidence"] += 1
            self._quality_window["min_conf_seen"] = min(
                self._quality_window["min_conf_seen"], confidence
            )
            self._quality_window["max_conf_seen"] = max(
                self._quality_window["max_conf_seen"], confidence
            )
            return False

        x1, y1, x2, y2 = bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height

        minimum_area = self.min_bbox_area

        if (
            self.frame_width > 0
            and self.frame_height > 0
            and self.min_bbox_area_ratio > 0
        ):
            frame_area = self.frame_width * self.frame_height
            minimum_area = max(
                minimum_area, frame_area * self.min_bbox_area_ratio
            )

        if area < minimum_area:
            self._quality_window["reject_area"] += 1
            self._quality_window["min_area_seen"] = min(
                self._quality_window["min_area_seen"], area
            )
            return False

        self._quality_window["accept"] += 1
        self._quality_window["min_conf_seen"] = min(
            self._quality_window["min_conf_seen"], confidence
        )
        self._quality_window["max_conf_seen"] = max(
            self._quality_window["max_conf_seen"], confidence
        )
        self._quality_window["min_area_seen"] = min(
            self._quality_window["min_area_seen"], area
        )

        return True

    def _reset_quality_window(self) -> None:
        self._quality_window = {
            "total": 0,
            "accept": 0,
            "reject_confidence": 0,
            "reject_area": 0,
            "min_conf_seen": float("inf"),
            "max_conf_seen": float("-inf"),
            "min_area_seen": float("inf"),
        }

    def _report_quality_window(self) -> None:

        if self.quality_report_every <= 0:
            return

        if self.frame_index % self.quality_report_every != 0:
            return

        window = self._quality_window

        if window["total"] == 0:
            # No IATA_TAG detections reached the quality filter at all
            # in this window. Either YOLO isn't detecting anything, or
            # nothing survived the track_low_thresh pre-filter. This
            # is the "zero across the whole run" signature -- if you
            # see this repeated forever, look upstream of TagTracker
            # (YOLO confidence, camera frames, pipeline wiring), not
            # inside it.
            print(
                f"[TagTracker] frame {self.frame_index}: "
                "0 observations reached the quality filter in the last "
                f"{self.quality_report_every} frames "
                "(check YOLO output / pipeline wiring upstream)."
            )
        else:
            conf_range = (
                f"{window['min_conf_seen']:.3f}-{window['max_conf_seen']:.3f}"
            )
            area_note = (
                f"{window['min_area_seen']:.0f}px"
                if window["min_area_seen"] != float("inf")
                else "n/a"
            )
            print(
                f"[TagTracker] frame {self.frame_index}: "
                f"{window['accept']}/{window['total']} accepted, "
                f"{window['reject_confidence']} rejected (confidence), "
                f"{window['reject_area']} rejected (area) | "
                f"conf seen: {conf_range} | "
                f"min area seen: {area_note} | "
                f"thresholds: conf>={self.min_track_confidence}, "
                f"area>={self.min_bbox_area}"
            )

        self._reset_quality_window()

    # ======================================================================
    # MOTION MEMORY
    # ======================================================================

    def _update_motion_memory(
        self,
        track_id,
        center,
        bbox,
        previous_center,
        previous_bbox,
    ) -> None:

        if previous_center is None:
            velocity = (0.0, 0.0)
            frame_distance = 0.0
        else:
            velocity = (
                center[0] - previous_center[0],
                center[1] - previous_center[1],
            )
            frame_distance = math.hypot(velocity[0], velocity[1])

        iou = self._bbox_iou(previous_bbox, bbox)

        motion_jump = self._is_motion_jump(previous_center, center)

        low_iou = (
            previous_bbox is not None and iou < self.min_iou_warning
        )

        # Smooth velocity.
        old_velocity = self.track_velocity.get(track_id, (0.0, 0.0))

        smooth_velocity = (
            (old_velocity[0] * 0.65) + (velocity[0] * 0.35),
            (old_velocity[1] * 0.65) + (velocity[1] * 0.35),
        )

        self.track_velocity[track_id] = smooth_velocity
        self.track_speed[track_id] = frame_distance
        self.track_distance[track_id] = (
            self.track_distance.get(track_id, 0.0) + frame_distance
        )
        self.track_iou[track_id] = iou
        self.track_motion_jump[track_id] = motion_jump
        self.track_last_center[track_id] = center
        self.track_last_bbox[track_id] = bbox

        # Instability.
        if motion_jump or low_iou:
            self.track_unstable_count[track_id] = (
                self.track_unstable_count.get(track_id, 0) + 1
            )
        else:
            self.track_unstable_count[track_id] = 0

        self.track_unstable[track_id] = (
            self.track_unstable_count.get(track_id, 0)
            >= self.unstable_frame_threshold
        )

    # ======================================================================
    # MOTION JUMP
    # ======================================================================

    def _is_motion_jump(self, previous_center, center) -> bool:

        if previous_center is None:
            return False

        if self.max_jump_distance <= 0:
            return False

        distance = math.hypot(
            center[0] - previous_center[0],
            center[1] - previous_center[1],
        )

        return distance > self.max_jump_distance

    # ======================================================================
    # IOU
    # ======================================================================

    @staticmethod
    def _bbox_iou(previous_bbox, bbox) -> float:

        if previous_bbox is None:
            return 1.0

        ax1, ay1, ax2, ay2 = previous_bbox
        bx1, by1, bx2, by2 = bbox

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    # ======================================================================
    # ACCESS
    # ======================================================================

    def get_active_tracks(self) -> list[Track]:
        return sorted(
            self._tracks.values(), key=lambda track: track.track_id
        )

    def get_confirmed_tracks(self) -> list[Track]:
        return sorted(
            [t for t in self._tracks.values() if t.is_confirmed],
            key=lambda track: track.track_id,
        )

    def get_track(self, track_id: int) -> Optional[Track]:
        return self._tracks.get(int(track_id))

    def get_active_track_ids(self) -> list[int]:
        return sorted(self._tracks.keys())

    # ======================================================================
    # RESET
    # ======================================================================

    def reset(self) -> None:

        self._tracker.reset()

        self._tracks.clear()
        self.track_age.clear()
        self.track_last_seen.clear()
        self.track_last_center.clear()
        self.track_last_bbox.clear()
        self.track_velocity.clear()
        self.track_speed.clear()
        self.track_distance.clear()
        self.track_iou.clear()
        self.track_unstable.clear()
        self.track_unstable_count.clear()
        self.track_motion_jump.clear()

        self.frame_index = 0

        self._reset_quality_window()

    # ======================================================================
    # HISTORY CLEANUP
    # ======================================================================

    def _trim_history(self) -> None:

        ttl = max(self.track_buffer * 4, 120)

        retained_ids = {
            track_id
            for track_id, last_seen in self.track_last_seen.items()
            if (self.frame_index - last_seen) <= ttl
        }

        # Always retain currently active project tracks.
        retained_ids.update(self._tracks.keys())

        dictionaries = [
            self.track_age,
            self.track_last_seen,
            self.track_last_center,
            self.track_last_bbox,
            self.track_velocity,
            self.track_speed,
            self.track_distance,
            self.track_iou,
            self.track_unstable,
            self.track_unstable_count,
            self.track_motion_jump,
        ]

        for dictionary in dictionaries:
            for track_id in list(dictionary.keys()):
                if track_id not in retained_ids:
                    del dictionary[track_id]


# ============================================================================
# STANDALONE DIAGNOSTIC
# ============================================================================


def _make_detection(x1, y1, x2, y2, confidence=0.90):
    return Detection(
        class_id=0,
        class_name="IATA_TAG",
        confidence=confidence,
        bbox=(x1, y1, x2, y2),
    )


def _diagnostic() -> None:

    print("=" * 75)
    print("OCR_BHS CONTINUOUS IATA BYTE TRACK TEST (supervision backend)")
    print("=" * 75)

    tracker = TagTracker(
        track_high_thresh=0.40,
        track_low_thresh=0.10,
        new_track_thresh=0.40,
        track_buffer=45,
        match_thresh=0.85,
        min_hits=3,
        prediction_buffer=18,
        min_track_confidence=0.25,
        quality_report_every=0,  # quiet during the self-test
    )

    print()
    print("BYTE TRACKER:", type(tracker._tracker).__name__)

    if type(tracker._tracker).__name__ != "ByteTrack":
        print("[FAIL] supervision.ByteTrack was not created.")
        return

    print("[PASS] Real supervision.ByteTrack loaded.")

    print()
    print("-" * 75)
    print("TEST 1 - CONTINUOUS TRACK THROUGH YOLO GAPS")
    print("-" * 75)

    frames = {
        1: [_make_detection(100, 200, 300, 300)],
        2: [_make_detection(125, 200, 325, 300)],
        3: [_make_detection(150, 200, 350, 300)],
        # YOLO misses two inference frames.
        4: [],
        5: [],
        # YOLO detects again.
        6: [_make_detection(225, 200, 425, 300)],
        7: [_make_detection(250, 200, 450, 300)],
    }

    first_id = None
    all_ids = set()
    predicted_count = 0

    for frame_id, detections in frames.items():

        tracks = tracker.update(detections=detections, frame_id=frame_id)

        print(
            f"Frame {frame_id:02d} | "
            f"YOLO={len(detections)} | "
            f"TRACKS={len(tracks)}"
        )

        for track in tracks:

            all_ids.add(track.track_id)

            if first_id is None:
                first_id = track.track_id

            status = "PREDICTED" if track.is_predicted else "DETECTED"

            if track.is_predicted:
                predicted_count += 1

            print(
                f"    ID={track.track_id} | {status} | "
                f"hits={track.hits} | miss={track.missed_frames} | "
                f"conf={track.track_confidence:.3f} | "
                f"center=({track.center[0]:.1f}, {track.center[1]:.1f}) | "
                f"speed={track.speed:.1f} | direction={track.direction}"
            )

    print()

    if len(all_ids) == 1:
        print("[PASS] Track ID remained stable:", first_id)
    else:
        print("[FAIL] Track ID changed:", sorted(all_ids))
        return

    if predicted_count >= 2:
        print("[PASS] Track remained visible during YOLO detection gaps.")
    else:
        print("[FAIL] Continuous prediction did not occur.")
        return

    # ========================================================================
    # TEST 2
    # ========================================================================

    print()
    print("-" * 75)
    print("TEST 2 - TWO INDEPENDENT IATA TAGS")
    print("-" * 75)

    tracker.reset()

    two_tag_frames = {
        1: [
            _make_detection(100, 200, 300, 300),
            _make_detection(700, 200, 900, 300),
        ],
        2: [
            _make_detection(125, 200, 325, 300),
            _make_detection(675, 200, 875, 300),
        ],
        3: [
            _make_detection(150, 200, 350, 300),
            _make_detection(650, 200, 850, 300),
        ],
        4: [
            _make_detection(175, 200, 375, 300),
            _make_detection(625, 200, 825, 300),
        ],
    }

    final_ids = []

    for frame_id, detections in two_tag_frames.items():

        tracks = tracker.update(detections=detections, frame_id=frame_id)

        ids = [track.track_id for track in tracks]
        final_ids = ids

        print(f"Frame {frame_id}: IDs={ids}")

    if len(final_ids) != 2:
        print("[FAIL] Expected two simultaneous tracks.")
        return

    if final_ids[0] == final_ids[1]:
        print("[FAIL] Two tags received the same ID.")
        return

    print("[PASS] Two tags have independent IDs:", final_ids)

    # ========================================================================
    # TEST 3
    # ========================================================================

    print()
    print("-" * 75)
    print("TEST 3 - TRACK CONFIRMATION")
    print("-" * 75)

    confirmed = tracker.get_confirmed_tracks()

    print("Confirmed IDs:", [track.track_id for track in confirmed])

    if len(confirmed) == 2:
        print("[PASS] Both tracks confirmed.")
    else:
        print("[FAIL] Expected two confirmed tracks.")
        return

    print()
    print("=" * 75)
    print("ALL CONTINUOUS BYTE TRACK TESTS PASSED")
    print("=" * 75)


if __name__ == "__main__":
    _diagnostic()