"""
OCR_BHS Continuous IATA Tag Tracker  (smooth-box version)
=========================================================

YOLO26 (own thread, as fast as the CPU allows)
   |
   v  detections + the timestamp of the frame they were computed on
supervision.ByteTrack           -> stable Track ID (identity only)
   |
   v
Per-track Kalman filter          -> smooth box, velocity in px/second
   |
   +--> update(detections, frame_id, timestamp)   once per YOLO result
   +--> predict(frame_id, timestamp)              once per CAMERA frame
                                                  (box glides with the tag)

WHAT CHANGED vs. the previous version
-------------------------------------
1. ByteTrack is now used ONLY for identity (which box is which tag).
   supervision's ByteTrack returns the raw YOLO box for matched
   detections, so the previous version displayed raw, jittery YOLO
   boxes. The displayed box now comes from a per-track Kalman filter.

2. Motion is time based (pixels / second, from real frame timestamps),
   not "pixels per tracker call". The old velocity was garbage whenever
   the tracker was called irregularly (or with stale detections), so
   prediction through YOLO gaps did not move the box.

3. New ``predict()`` method: call it on EVERY camera frame. It moves
   each box to where the tag should be *now* using the filter velocity,
   without touching ByteTrack. ``update()`` is called only when a new
   YOLO result exists (never with stale detections).

4. Latency compensation: a YOLO result describes the frame it was run
   on (a few frames ago). The filter is updated at that frame's time
   and the box is then projected forward to the current frame.

5. ByteTrack's hidden new-track offset is compensated. supervision
   creates a NEW track only if score >= track_activation_threshold+0.1.
   The activation threshold is now ``new_track_thresh - 0.10`` so
   ``new_track_thresh`` means what it says.

Everything the pipeline/viewer already uses (Track fields, quality
filter, min_hits, get_*_tracks, reset, diagnostics) is unchanged.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from supervision import ByteTrack, Detections

from app.models.detection import Detection
from app.models.tracking import Track


# ============================================================================
# 1-D CONSTANT-VELOCITY KALMAN FILTER  (state = position, velocity)
# ============================================================================


class _Axis1D:
    __slots__ = ("x", "v", "p00", "p01", "p11")

    def __init__(self, x: float, pos_var: float, vel_var: float) -> None:
        self.x = x
        self.v = 0.0
        self.p00 = pos_var
        self.p01 = 0.0
        self.p11 = vel_var

    def predict(self, dt: float, q: float) -> None:
        if dt <= 0.0:
            return
        dt2 = dt * dt
        self.x += self.v * dt
        self.p00 += 2.0 * dt * self.p01 + dt2 * self.p11 + q * dt2 * dt2 / 4.0
        self.p01 += dt * self.p11 + q * dt2 * dt / 2.0
        self.p11 += q * dt2

    def update(self, z: float, r: float) -> None:
        s = self.p00 + r
        k0 = self.p00 / s
        k1 = self.p01 / s
        y = z - self.x
        self.x += k0 * y
        self.v += k1 * y
        p00 = (1.0 - k0) * self.p00
        p01 = (1.0 - k0) * self.p01
        p11 = self.p11 - k1 * self.p01
        self.p00 = p00
        self.p01 = p01
        self.p11 = max(p11, 1e-3)


class _BoxFilter:
    """Smoothed box: Kalman on centre x/y, EMA on width/height."""

    __slots__ = ("ax", "ay", "w", "h", "t", "t_meas", "conf")

    def __init__(self, box, t: float, conf: float, pos_var: float, vel_var: float):
        x1, y1, x2, y2 = box
        self.ax = _Axis1D((x1 + x2) / 2.0, pos_var, vel_var)
        self.ay = _Axis1D((y1 + y2) / 2.0, pos_var, vel_var)
        self.w = max(1.0, x2 - x1)
        self.h = max(1.0, y2 - y1)
        self.t = t
        self.t_meas = t
        self.conf = conf

    def predict_to(self, t: float, q: float, coasting: bool, half_life: float) -> None:
        dt = t - self.t
        if dt <= 0.0:
            return
        if coasting and half_life > 0.0:
            decay = 0.5 ** (dt / half_life)
            self.ax.v *= decay
            self.ay.v *= decay
        self.ax.predict(dt, q)
        self.ay.predict(dt, q)
        self.t = t

    def update(self, box, conf: float, r_std: float, size_alpha: float) -> None:
        x1, y1, x2, y2 = box
        r = r_std * r_std
        self.ax.update((x1 + x2) / 2.0, r)
        self.ay.update((y1 + y2) / 2.0, r)
        self.w += size_alpha * (max(1.0, x2 - x1) - self.w)
        self.h += size_alpha * (max(1.0, y2 - y1) - self.h)
        self.t_meas = self.t
        self.conf = conf

    def box_at(self, t: float, max_horizon: float, min_speed: float):
        """Box extrapolated to time ``t`` (does NOT mutate the filter)."""
        horizon = min(max(t - self.t, 0.0), max_horizon)
        vx = self.ax.v if abs(self.ax.v) >= min_speed else 0.0
        vy = self.ay.v if abs(self.ay.v) >= min_speed else 0.0
        cx = self.ax.x + vx * horizon
        cy = self.ay.x + vy * horizon
        return (
            cx - self.w / 2.0,
            cy - self.h / 2.0,
            cx + self.w / 2.0,
            cy + self.h / 2.0,
        )


# ============================================================================
# TAG TRACKER
# ============================================================================


class TagTracker:
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
        # Max consecutive missed YOLO RESULTS a box is still drawn for.
        prediction_buffer: int = 18,
        # Motion quality.
        max_jump_distance: float = 250.0,
        min_iou_warning: float = 0.10,
        unstable_frame_threshold: int = 3,
        frame_width: int = 0,
        frame_height: int = 0,
        min_bbox_area: float = 0.0,
        min_bbox_area_ratio: float = 0.0,
        min_track_confidence: float = 0.25,
        # Backward compatibility with old OCR_BHS live_pipeline.py
        iou_threshold: float = 0.15,
        max_missed_frames: int = 12,
        max_center_distance: float = 180.0,
        min_size_similarity: float = 0.20,
        frame_rate: float = 30.0,
        quality_report_every: int = 30,
        # ---------------------------------------------------------------
        # Smoothing / prediction (new)
        # ---------------------------------------------------------------
        # Std-dev of YOLO centre jitter as a fraction of sqrt(w*h).
        # Higher = smoother but slower to react.
        measurement_noise_ratio: float = 0.03,
        # Std-dev of tag acceleration (px/s^2). Lower = smoother /
        # straighter motion, higher = follows sudden speed changes.
        process_noise_accel: float = 300.0,
        # EMA factor for box width/height (0..1, lower = steadier size).
        size_smoothing: float = 0.35,
        # Never draw a box longer than this without a real detection.
        max_coast_seconds: float = 1.0,
        # While coasting, velocity halves every N seconds (no fly-away).
        coast_velocity_half_life: float = 1.0,
        # Velocities below this (px/s) are treated as "standing still".
        min_render_speed: float = 5.0,
    ) -> None:

        # ByteTrack config
        self.track_high_thresh = float(track_high_thresh)
        self.track_low_thresh = float(track_low_thresh)
        self.new_track_thresh = float(new_track_thresh)
        self.track_buffer = max(1, int(track_buffer))
        self.match_thresh = float(match_thresh)
        self.fuse_score = bool(fuse_score)

        self.min_hits = max(1, int(min_hits))
        self.prediction_buffer = max(1, int(prediction_buffer))

        self.max_jump_distance = max(0.0, float(max_jump_distance))
        self.min_iou_warning = min(max(0.0, float(min_iou_warning)), 1.0)
        self.unstable_frame_threshold = max(1, int(unstable_frame_threshold))

        self.frame_width = max(0, int(frame_width or 0))
        self.frame_height = max(0, int(frame_height or 0))

        self.min_bbox_area = max(0.0, float(min_bbox_area))
        self.min_bbox_area_ratio = max(0.0, float(min_bbox_area_ratio))
        self.min_track_confidence = min(max(0.0, float(min_track_confidence)), 1.0)

        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = max(1, int(max_missed_frames))
        self.max_center_distance = max(0.0, float(max_center_distance))
        self.min_size_similarity = max(0.0, float(min_size_similarity))
        self.frame_rate = max(1.0, float(frame_rate))

        # Smoothing
        self.measurement_noise_ratio = max(0.001, float(measurement_noise_ratio))
        self.process_noise_q = max(1.0, float(process_noise_accel)) ** 2
        self.size_smoothing = min(max(0.01, float(size_smoothing)), 1.0)
        self.max_coast_seconds = max(0.05, float(max_coast_seconds))
        self.coast_velocity_half_life = max(0.0, float(coast_velocity_half_life))
        self.min_render_speed = max(0.0, float(min_render_speed))

        self.frame_index = 0
        self._cycle = 0  # number of update() calls (== YOLO results consumed)

        # Project track memory
        self._tracks: dict[int, Track] = {}
        self._filters: dict[int, _BoxFilter] = {}
        self._render_ids: set[int] = set()

        self.track_age: dict[int, int] = {}
        self.track_last_seen: dict[int, int] = {}
        self.track_last_center: dict[int, tuple[float, float]] = {}
        self.track_last_bbox: dict[int, tuple[float, float, float, float]] = {}
        self.track_velocity: dict[int, tuple[float, float]] = {}
        self.track_speed: dict[int, float] = {}
        self.track_distance: dict[int, float] = {}
        self.track_iou: dict[int, float] = {}
        self.track_unstable: dict[int, bool] = {}
        self.track_unstable_count: dict[int, int] = {}
        self.track_motion_jump: dict[int, bool] = {}

        self.quality_report_every = max(0, int(quality_report_every))
        self._reset_quality_window()

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
    # BYTETRACK
    # ======================================================================

    def _create_bytetrack(self) -> ByteTrack:
        """
        supervision.ByteTrack creates a new track only when
        ``score >= track_activation_threshold + 0.1``. We subtract that
        0.1 here so ``new_track_thresh`` is the real "start a track"
        confidence. Detections between 0.1 and the activation threshold
        are used by ByteTrack's second (low-score) association stage to
        keep existing tracks alive through motion blur -- that only
        works if YOLO's own ``confidence`` is <= ~0.15 (see config).
        """
        activation = max(0.05, self.new_track_thresh - 0.10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return ByteTrack(
                track_activation_threshold=activation,
                lost_track_buffer=self.track_buffer,
                minimum_matching_threshold=self.match_thresh,
                frame_rate=30,  # track_buffer is counted in YOLO results
                minimum_consecutive_frames=1,
            )

    # ======================================================================
    # TIME
    # ======================================================================

    def _resolve_time(self, frame_id, timestamp) -> float:
        if timestamp is not None:
            try:
                value = float(timestamp)
                if math.isfinite(value):
                    return value
            except (TypeError, ValueError):
                pass
        return int(frame_id or 0) / self.frame_rate

    # ======================================================================
    # UPDATE  (call once per NEW YOLO result, even if it is empty)
    # ======================================================================

    def update(
        self,
        detections: list[Detection],
        frame_id: int,
        timestamp=None,
    ) -> list[Track]:
        """
        ``frame_id`` / ``timestamp`` must describe the frame the
        detections were computed on (not the newest camera frame).
        """

        self.frame_index = int(frame_id)
        self._cycle += 1
        ts = self._resolve_time(frame_id, timestamp)

        # ---- only IATA tags above the noise floor ------------------------
        iata_detections: list[Detection] = []
        for detection in detections or []:
            if int(detection.class_id) != self.IATA_TAG_CLASS_ID:
                continue
            if float(detection.confidence) < self.track_low_thresh:
                continue
            if detection.x2 <= detection.x1 or detection.y2 <= detection.y1:
                continue
            iata_detections.append(detection)

        if iata_detections:
            detections_sv = Detections(
                xyxy=np.asarray(
                    [[d.x1, d.y1, d.x2, d.y2] for d in iata_detections],
                    dtype=np.float32,
                ),
                confidence=np.asarray(
                    [float(d.confidence) for d in iata_detections],
                    dtype=np.float32,
                ),
                class_id=np.zeros(len(iata_detections), dtype=int),
            )
        else:
            detections_sv = Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=int),
            )

        # ---- ByteTrack: identity only (called once per YOLO result) ------
        try:
            tracked = self._tracker.update_with_detections(detections_sv)
        except Exception as exc:
            print(f"[TagTracker] ByteTrack update failed: {exc!r}")
            tracked = None

        matched: list[tuple[int, tuple[float, float, float, float], float]] = []
        num_tracked = 0 if tracked is None else len(tracked)

        if tracked is not None and tracked.tracker_id is not None:
            for index in range(num_tracked):
                raw_id = tracked.tracker_id[index]
                if raw_id is None or raw_id < 0:
                    continue
                if tracked.class_id is not None:
                    if int(tracked.class_id[index]) != self.IATA_TAG_CLASS_ID:
                        continue
                x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[index])
                score = (
                    float(tracked.confidence[index])
                    if tracked.confidence is not None
                    else 0.0
                )
                matched.append((int(raw_id), (x1, y1, x2, y2), score))

        updated_ids = {m[0] for m in matched}

        # ---- advance every filter to the detection time ------------------
        for track_id, flt in self._filters.items():
            flt.predict_to(
                ts,
                self.process_noise_q,
                coasting=(track_id not in updated_ids),
                half_life=self.coast_velocity_half_life,
            )

        current_tracks: list[Track] = []

        # ---- matched tracks ----------------------------------------------
        for track_id, bbox, score in matched:

            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            previous_center = self.track_last_center.get(track_id)
            previous_bbox = self.track_last_bbox.get(track_id)

            self.track_age[track_id] = self.track_age.get(track_id, 0) + 1
            self.track_last_seen[track_id] = self.frame_index

            # Kalman filter
            size = math.sqrt(max(1.0, (x2 - x1) * (y2 - y1)))
            r_std = max(1.5, self.measurement_noise_ratio * size) * (
                2.0 - min(max(score, 0.0), 1.0)
            )
            flt = self._filters.get(track_id)
            if flt is None:
                flt = _BoxFilter(
                    bbox,
                    ts,
                    score,
                    pos_var=r_std * r_std,
                    vel_var=(0.8 * size * self.frame_rate) ** 2,
                )
                self._filters[track_id] = flt
            else:
                flt.update(bbox, score, r_std, self.size_smoothing)

            self._update_motion_memory(
                track_id=track_id,
                center=center,
                bbox=bbox,
                previous_center=previous_center,
                previous_bbox=previous_bbox,
                flt=flt,
            )

            # Project Track object
            track = self._tracks.get(track_id)
            hits = self.track_age[track_id]

            if track is None:
                track = Track(
                    track_id=track_id,
                    detection=Detection(0, "IATA_TAG", score, bbox),
                    last_frame_id=self.frame_index,
                    missed_frames=0,
                    age=hits,
                    hits=hits,
                    previous_center=None,
                    current_center=center,
                    track_confidence=score,
                    is_confirmed=(hits >= self.min_hits),
                )
                self._tracks[track_id] = track
            else:
                track.previous_center = track.current_center
                track.age += 1
                track.hits += 1
                track.missed_frames = 0
                track.track_confidence = score
                if track.hits >= self.min_hits:
                    track.is_confirmed = True

            self._render_track(track, flt, ts, predicted=False, confidence=score)

            track.unstable = self.track_unstable.get(track_id, False)
            track.instability_count = self.track_unstable_count.get(track_id, 0)
            track.iou = self.track_iou.get(track_id, 1.0)
            track.speed = self.track_speed.get(track_id, 0.0)
            track.distance = self.track_distance.get(track_id, 0.0)

            # Quality filter is OUTPUT-only
            if self._passes_quality(bbox=track.bbox, confidence=score):
                current_tracks.append(track)

        # ---- tracks YOLO missed this time: coast on the filter -----------
        for track_id, track in list(self._tracks.items()):

            if track_id in updated_ids:
                continue

            flt = self._filters.get(track_id)
            track.missed_frames += 1

            if (
                flt is None
                or track.missed_frames > self.prediction_buffer
                or (ts - flt.t_meas) > self.max_coast_seconds
            ):
                self._tracks.pop(track_id, None)
                self._filters.pop(track_id, None)
                continue

            if not track.is_confirmed:
                continue

            confidence = max(0.05, flt.conf * (0.96 ** track.missed_frames))
            self._render_track(track, flt, ts, predicted=True, confidence=confidence)
            current_tracks.append(track)

        current_tracks.sort(key=lambda item: item.track_id)
        self._render_ids = {t.track_id for t in current_tracks}

        self._trim_history()
        self._report_quality_window()

        return current_tracks

    # ======================================================================
    # PREDICT  (call on EVERY camera frame)
    # ======================================================================

    def predict(self, frame_id: Optional[int] = None, timestamp=None) -> list[Track]:
        """
        Move every visible box to where its tag should be at ``timestamp``
        using the filter's velocity. Does not touch ByteTrack or the
        filters, so it is cheap (microseconds per track) and safe to call
        at the full camera frame rate.
        """

        if frame_id is not None:
            self.frame_index = int(frame_id)

        ts = self._resolve_time(self.frame_index, timestamp)

        output: list[Track] = []

        for track_id in sorted(self._render_ids):
            track = self._tracks.get(track_id)
            flt = self._filters.get(track_id)
            if track is None or flt is None:
                continue
            if (ts - flt.t_meas) > self.max_coast_seconds:
                continue

            predicted = track.missed_frames > 0
            confidence = (
                max(0.05, flt.conf * (0.96 ** track.missed_frames))
                if predicted
                else flt.conf
            )
            self._render_track(track, flt, ts, predicted=predicted, confidence=confidence)
            output.append(track)

        return output

    # ======================================================================
    # RENDER
    # ======================================================================

    def _render_track(
        self,
        track: Track,
        flt: _BoxFilter,
        ts: float,
        predicted: bool,
        confidence: float,
    ) -> None:

        box = flt.box_at(ts, self.max_coast_seconds, self.min_render_speed)
        center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

        if track.current_center is not None and track.current_center != center:
            track.previous_center = track.current_center

        track.detection = Detection(
            class_id=0,
            class_name="IATA_TAG",
            confidence=float(confidence),
            bbox=box,
        )
        track.current_center = center
        track.last_frame_id = self.frame_index
        track.is_predicted = bool(predicted)

        # px per 1/frame_rate seconds (same unit the viewer used before)
        track.velocity_x = flt.ax.v / self.frame_rate
        track.velocity_y = flt.ay.v / self.frame_rate

    # ======================================================================
    # QUALITY
    # ======================================================================

    def _passes_quality(self, bbox, confidence) -> bool:

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
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        minimum_area = self.min_bbox_area
        if (
            self.frame_width > 0
            and self.frame_height > 0
            and self.min_bbox_area_ratio > 0
        ):
            minimum_area = max(
                minimum_area,
                self.frame_width * self.frame_height * self.min_bbox_area_ratio,
            )

        if area < minimum_area:
            self._quality_window["reject_area"] += 1
            self._quality_window["min_area_seen"] = min(
                self._quality_window["min_area_seen"], area
            )
            return False

        window = self._quality_window
        window["accept"] += 1
        window["min_conf_seen"] = min(window["min_conf_seen"], confidence)
        window["max_conf_seen"] = max(window["max_conf_seen"], confidence)
        window["min_area_seen"] = min(window["min_area_seen"], area)
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
        if self._cycle % self.quality_report_every != 0:
            return

        window = self._quality_window

        if window["total"] == 0:
            print(
                f"[TagTracker] cycle {self._cycle}: 0 observations reached the "
                f"quality filter in the last {self.quality_report_every} YOLO "
                "results (check YOLO output / pipeline wiring upstream)."
            )
        else:
            conf_range = f"{window['min_conf_seen']:.3f}-{window['max_conf_seen']:.3f}"
            area_note = (
                f"{window['min_area_seen']:.0f}px"
                if window["min_area_seen"] != float("inf")
                else "n/a"
            )
            print(
                f"[TagTracker] cycle {self._cycle}: "
                f"{window['accept']}/{window['total']} accepted, "
                f"{window['reject_confidence']} rejected (confidence), "
                f"{window['reject_area']} rejected (area) | "
                f"conf seen: {conf_range} | min area seen: {area_note} | "
                f"thresholds: conf>={self.min_track_confidence}, "
                f"area>={self.min_bbox_area}"
            )

        self._reset_quality_window()

    # ======================================================================
    # MOTION MEMORY  (per YOLO result, raw measurements)
    # ======================================================================

    def _update_motion_memory(
        self,
        track_id,
        center,
        bbox,
        previous_center,
        previous_bbox,
        flt: _BoxFilter,
    ) -> None:

        if previous_center is None:
            cycle_distance = 0.0
        else:
            cycle_distance = math.hypot(
                center[0] - previous_center[0],
                center[1] - previous_center[1],
            )

        iou = self._bbox_iou(previous_bbox, bbox)
        motion_jump = self._is_motion_jump(previous_center, center)
        low_iou = previous_bbox is not None and iou < self.min_iou_warning

        vx = flt.ax.v / self.frame_rate
        vy = flt.ay.v / self.frame_rate

        self.track_velocity[track_id] = (vx, vy)
        self.track_speed[track_id] = math.hypot(vx, vy)
        self.track_distance[track_id] = (
            self.track_distance.get(track_id, 0.0) + cycle_distance
        )
        self.track_iou[track_id] = iou
        self.track_motion_jump[track_id] = motion_jump
        self.track_last_center[track_id] = center
        self.track_last_bbox[track_id] = bbox

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

    def _is_motion_jump(self, previous_center, center) -> bool:
        if previous_center is None or self.max_jump_distance <= 0:
            return False
        return (
            math.hypot(
                center[0] - previous_center[0],
                center[1] - previous_center[1],
            )
            > self.max_jump_distance
        )

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

        return 0.0 if union <= 0 else intersection / union

    # ======================================================================
    # ACCESS
    # ======================================================================

    def get_active_tracks(self) -> list[Track]:
        return sorted(self._tracks.values(), key=lambda t: t.track_id)

    def get_confirmed_tracks(self) -> list[Track]:
        return sorted(
            [t for t in self._tracks.values() if t.is_confirmed],
            key=lambda t: t.track_id,
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
        self._filters.clear()
        self._render_ids.clear()

        for dictionary in (
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
        ):
            dictionary.clear()

        self.frame_index = 0
        self._cycle = 0
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
        retained_ids.update(self._tracks.keys())

        for dictionary in (
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
        ):
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
    print("OCR_BHS CONTINUOUS IATA BYTE TRACK TEST (Kalman-smoothed)")
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
        quality_report_every=0,
    )

    if type(tracker._tracker).__name__ != "ByteTrack":
        print("[FAIL] supervision.ByteTrack was not created.")
        return
    print("[PASS] Real supervision.ByteTrack loaded.")

    # ---- TEST 1: continuous track through YOLO gaps ----------------------
    print("\n" + "-" * 75 + "\nTEST 1 - CONTINUOUS TRACK THROUGH YOLO GAPS\n" + "-" * 75)

    frames = {
        1: [_make_detection(100, 200, 300, 300)],
        2: [_make_detection(125, 200, 325, 300)],
        3: [_make_detection(150, 200, 350, 300)],
        4: [],
        5: [],
        6: [_make_detection(225, 200, 425, 300)],
        7: [_make_detection(250, 200, 450, 300)],
    }

    first_id = None
    all_ids = set()
    predicted_count = 0
    expected_cx = {4: 275.0, 5: 300.0}  # true centres during the gap

    for frame_id, detections in frames.items():
        tracks = tracker.update(detections=detections, frame_id=frame_id)
        print(f"Frame {frame_id:02d} | YOLO={len(detections)} | TRACKS={len(tracks)}")
        for track in tracks:
            all_ids.add(track.track_id)
            first_id = first_id if first_id is not None else track.track_id
            if track.is_predicted:
                predicted_count += 1
            print(
                f"    ID={track.track_id} | {track.status} | hits={track.hits} | "
                f"miss={track.missed_frames} | center=({track.center[0]:.1f}, "
                f"{track.center[1]:.1f}) | speed={track.speed:.1f} | "
                f"direction={track.direction}"
            )
            if frame_id in expected_cx:
                error = abs(track.center[0] - expected_cx[frame_id])
                print(f"    predicted-centre error vs truth: {error:.1f}px")

    if len(all_ids) != 1:
        print("[FAIL] Track ID changed:", sorted(all_ids))
        return
    print("[PASS] Track ID remained stable:", first_id)

    if predicted_count < 2:
        print("[FAIL] Continuous prediction did not occur.")
        return
    print("[PASS] Track remained visible during YOLO detection gaps.")

    # ---- TEST 2: two independent tags ------------------------------------
    print("\n" + "-" * 75 + "\nTEST 2 - TWO INDEPENDENT IATA TAGS\n" + "-" * 75)

    tracker.reset()
    final_ids = []
    for frame_id in range(1, 5):
        d = 25 * (frame_id - 1)
        tracks = tracker.update(
            detections=[
                _make_detection(100 + d, 200, 300 + d, 300),
                _make_detection(700 - d, 200, 900 - d, 300),
            ],
            frame_id=frame_id,
        )
        final_ids = [t.track_id for t in tracks]
        print(f"Frame {frame_id}: IDs={final_ids}")

    if len(final_ids) != 2 or final_ids[0] == final_ids[1]:
        print("[FAIL] Expected two independent tracks.")
        return
    print("[PASS] Two tags have independent IDs:", final_ids)

    # ---- TEST 3: confirmation --------------------------------------------
    print("\n" + "-" * 75 + "\nTEST 3 - TRACK CONFIRMATION\n" + "-" * 75)
    confirmed = tracker.get_confirmed_tracks()
    print("Confirmed IDs:", [t.track_id for t in confirmed])
    if len(confirmed) != 2:
        print("[FAIL] Expected two confirmed tracks.")
        return
    print("[PASS] Both tracks confirmed.")

    # ---- TEST 4: per-frame predict() between YOLO results ----------------
    print("\n" + "-" * 75 + "\nTEST 4 - predict() BETWEEN YOLO RESULTS\n" + "-" * 75)

    tracker.reset()
    # Tag moves 600 px/s. YOLO result every 0.1 s (frames 0, 3, 6, ...).
    for i in range(6):
        t = i * 0.1
        x = 100 + 600 * t
        tracker.update([_make_detection(x, 200, x + 200, 300)], frame_id=i * 3, timestamp=t)

    t_last = 5 * 0.1
    worst = 0.0
    for k in range(1, 4):  # the three camera frames after the last YOLO result
        t = t_last + k / 30.0
        tracks = tracker.predict(frame_id=15 + k, timestamp=t)
        if not tracks:
            print("[FAIL] predict() returned no tracks.")
            return
        true_cx = 100 + 600 * t + 100
        error = abs(tracks[0].center[0] - true_cx)
        worst = max(worst, error)
        print(f"  t=+{k / 30.0 * 1000:.0f}ms  box centre={tracks[0].center[0]:.1f}  truth={true_cx:.1f}  err={error:.1f}px")

    if worst > 8.0:
        print(f"[FAIL] predict() error too large: {worst:.1f}px")
        return
    print(f"[PASS] predict() follows the tag (worst error {worst:.1f}px).")

    print("\n" + "=" * 75)
    print("ALL CONTINUOUS BYTE TRACK TESTS PASSED")
    print("=" * 75)


if __name__ == "__main__":
    _diagnostic()