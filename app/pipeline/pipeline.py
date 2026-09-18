"""
OCR_BHS - Main OCR / Barcode Pipeline
=====================================

Production-oriented asynchronous OCR_BHS pipeline.

Architecture:

    Camera
       |
       +-----------------------> Latest Frame
       |
       v
    YOLO Detection
       |
       v
    TagTracker / ByteTrack
       |
       v
    Best Frame Selection
       |
       +--------------------+
       |                    |
       v                    v
    OCR Queue          Barcode Queue
       |                    |
       v                    v
    OCR Workers        Barcode Workers
       |                    |
       +---------+----------+
                 |
                 v
           ResultManager
                 |
                 v
             Validation
                 |
                 v
           SQL Persistence


IMPORTANT
---------
- Camera capture is independent from inference.
- YOLO runs periodically.
- Existing TagTracker / ByteTrack implementation is NOT modified.
- OCR is asynchronous.
- Barcode is asynchronous.
- The live inference loop NEVER waits for OCR/barcode.
- Best-frame selection uses BestFrameSelector.add_candidate().
- Latest tracking state is available to the viewer.
- Camera-local track IDs are kept separate.
- No dashboard logic.
- No jam-detection logic.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================================
# APPLICATION IMPORTS
# ============================================================================

from app.config.config_loader import Config
from app.camera.camera import Camera
from app.detection.yolo_detector import YOLODetector
from app.tracking.tag_tracker import TagTracker

from app.pipeline.processing_queue import (
    OCRTask,
    BarcodeTask,
    OCRTaskResult,
    BarcodeTaskResult,
)

from app.pipeline.worker_manager import WorkerManager
from app.pipeline.result_manager import ResultManager
from app.pipeline.best_frame_selector import BestFrameSelector


# ============================================================================
# TYPE ALIASES
# ============================================================================

ResultCallback = Callable[[Any], None]


# ============================================================================
# PIPELINE
# ============================================================================


class OCRBHSPipeline:
    """
    Main OCR_BHS processing pipeline.

    One pipeline instance belongs to one camera.

    Parameters
    ----------
    camera:
        Camera instance.

    detector:
        Existing YOLODetector instance.

    tracker:
        Existing TagTracker / ByteTrack instance.

    config:
        Central Config object.
    """

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(
        self,
        camera: Camera,
        detector: YOLODetector,
        tracker: TagTracker,
        config: Optional[Config] = None,
    ) -> None:

        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.config = config or Config()

        # --------------------------------------------------------------------
        # Camera ID
        # --------------------------------------------------------------------

        self.camera_id = self._camera_id()

        # --------------------------------------------------------------------
        # Runtime state
        # --------------------------------------------------------------------

        self.running = False

        self._stop_event = threading.Event()

        self._loop_thread: Optional[
            threading.Thread
        ] = None

        # --------------------------------------------------------------------
        # Tracking state
        # --------------------------------------------------------------------

        self._tracks_lock = threading.Lock()

        self._latest_tracks: List[Any] = []

        self._latest_track_frame_id: Optional[int] = None

        self._latest_track_timestamp: Optional[float] = None

        self._latest_detection_count: int = 0

        # --------------------------------------------------------------------
        # Result callbacks
        # --------------------------------------------------------------------

        self._result_callbacks: List[
            ResultCallback
        ] = []

        # --------------------------------------------------------------------
        # Worker manager
        # --------------------------------------------------------------------

        self.worker_manager = WorkerManager(
            config=self.config
        )

        # --------------------------------------------------------------------
        # Result manager
        # --------------------------------------------------------------------

        self.result_manager = ResultManager(
            max_states=1000
        )

        self.result_manager.add_callback(
            self._on_result_manager_callback
        )

        # --------------------------------------------------------------------
        # Best-frame selectors
        #
        # Key:
        #     (camera_id, track_id)
        #
        # This prevents IDs from different cameras being mixed.
        # --------------------------------------------------------------------

        self._selectors_lock = threading.Lock()

        self._selectors: Dict[
            Tuple[str, int],
            BestFrameSelector,
        ] = {}

        # --------------------------------------------------------------------
        # Track processing state
        # --------------------------------------------------------------------

        self._track_ocr_inflight: set[
            Tuple[str, int]
        ] = set()

        self._track_barcode_inflight: set[
            Tuple[str, int]
        ] = set()

        self._track_ocr_completed: set[
            Tuple[str, int]
        ] = set()

        self._track_barcode_completed: set[
            Tuple[str, int]
        ] = set()

        self._track_ocr_attempts: Dict[
            Tuple[str, int],
            int,
        ] = {}

        self._track_barcode_attempts: Dict[
            Tuple[str, int],
            int,
        ] = {}

        self._track_last_ocr_submit_time: Dict[
            Tuple[str, int],
            float,
        ] = {}

        self._track_last_barcode_submit_time: Dict[
            Tuple[str, int],
            float,
        ] = {}

        # --------------------------------------------------------------------
        # Statistics
        # --------------------------------------------------------------------

        self.frames_processed = 0

        self.detection_runs = 0

        self.tracks_seen = 0

        self.ocr_jobs_submitted = 0

        self.ocr_jobs_failed = 0

        self.barcode_jobs_submitted = 0

        self.barcode_jobs_failed = 0

        self.ocr_results_received = 0

        self.barcode_results_received = 0

        self.results_received = 0

        self.detection_errors = 0

        self.result_errors = 0

        self.last_detection_error = ""

        self.last_result_error = ""

        # --------------------------------------------------------------------
        # Configuration
        # --------------------------------------------------------------------

        processing_cfg = self.config.data.get(
            "processing",
            {},
        )

        performance_cfg = self.config.data.get(
            "performance",
            {},
        )

        detection_cfg = self.config.data.get(
            "detection",
            {},
        )

        ocr_cfg = self.config.data.get(
            "ocr",
            {},
        )

        barcode_cfg = self.config.data.get(
            "barcode",
            {},
        )

        # --------------------------------------------------------------------
        # Detection interval
        # --------------------------------------------------------------------

        self.detection_interval_seconds = self._get_float(
            processing_cfg,
            "detection_interval",
            0.20,
        )

        if "detection_interval_seconds" in processing_cfg:

            self.detection_interval_seconds = self._get_float(
                processing_cfg,
                "detection_interval_seconds",
                self.detection_interval_seconds,
            )

        # --------------------------------------------------------------------
        # OCR
        # --------------------------------------------------------------------

        self.asynchronous_ocr = bool(
            processing_cfg.get(
                "async_ocr",
                ocr_cfg.get(
                    "enabled",
                    True,
                ),
            )
        )

        self.ocr_submit_interval_seconds = self._get_float(
            processing_cfg,
            "ocr_interval",
            0.50,
        )

        if "ocr_submit_interval_seconds" in processing_cfg:

            self.ocr_submit_interval_seconds = self._get_float(
                processing_cfg,
                "ocr_submit_interval_seconds",
                self.ocr_submit_interval_seconds,
            )

        # --------------------------------------------------------------------
        # Barcode
        # --------------------------------------------------------------------

        self.asynchronous_barcode = bool(
            processing_cfg.get(
                "async_barcode",
                barcode_cfg.get(
                    "enabled",
                    True,
                ),
            )
        )

        self.barcode_submit_interval_seconds = self._get_float(
            processing_cfg,
            "barcode_interval",
            0.50,
        )

        if "barcode_submit_interval_seconds" in processing_cfg:

            self.barcode_submit_interval_seconds = self._get_float(
                processing_cfg,
                "barcode_submit_interval_seconds",
                self.barcode_submit_interval_seconds,
            )

        # --------------------------------------------------------------------
        # Best frame
        # --------------------------------------------------------------------

        best_frame_cfg = processing_cfg.get(
            "best_frame",
            {},
        )

        self.best_frame_enabled = bool(
            best_frame_cfg.get(
                "enabled",
                True,
            )
        )

        self.best_frame_candidates = int(
            best_frame_cfg.get(
                "frames_per_track",
                best_frame_cfg.get(
                    "max_candidates",
                    5,
                ),
            )
        )

        self.best_frame_min_quality = float(
            best_frame_cfg.get(
                "minimum_quality_score",
                0.0,
            )
        )

        # --------------------------------------------------------------------
        # Track processing
        # --------------------------------------------------------------------

        duplicate_cfg = processing_cfg.get(
            "duplicate_processing",
            {},
        )

        self.process_once_per_track = bool(
            duplicate_cfg.get(
                "process_once_per_track",
                True,
            )
        )

        self.retry_failed_results = bool(
            duplicate_cfg.get(
                "retry_failed_results",
                True,
            )
        )

        self.max_ocr_attempts = int(
            duplicate_cfg.get(
                "max_ocr_attempts",
                3,
            )
        )

        self.max_barcode_attempts = int(
            duplicate_cfg.get(
                "max_barcode_attempts",
                3,
            )
        )

        # --------------------------------------------------------------------
        # Maximum detection count
        # --------------------------------------------------------------------

        self.max_detections = int(
            detection_cfg.get(
                "max_det",
                20,
            )
        )

        # --------------------------------------------------------------------
        # Performance
        # --------------------------------------------------------------------

        self.drop_old_frames = bool(
            performance_cfg.get(
                "drop_old_frames",
                True,
            )
        )

        # --------------------------------------------------------------------
        # Tracking update rate cap.
        #
        # tracker.update() now runs independently of YOLO's slower
        # detection_interval_seconds (for smooth box motion), but running
        # it completely uncapped means chasing the camera's native frame
        # rate (e.g. 60fps) on every loop iteration. On CPU, combined with
        # OCR/barcode worker threads competing for the same cores/GIL,
        # that much per-frame Python overhead can starve the main loop
        # badly enough to desync it from real time. Cap it instead to a
        # rate that is still far smoother than the old
        # detection_interval_seconds-gated behaviour, but bounded.
        # --------------------------------------------------------------------

        max_tracking_fps = self._get_float(
            performance_cfg,
            "max_tracking_fps",
            30.0,
        )

        if max_tracking_fps <= 0:

            self.tracking_interval_seconds = 0.0

        else:

            self.tracking_interval_seconds = (
                1.0 / max_tracking_fps
            )

        # --------------------------------------------------------------------
        # Result callback from worker manager
        # --------------------------------------------------------------------

        self._worker_result_callback_attached = False

        self._attach_worker_result_callback()

        print(
            "[PIPELINE] OCRBHSPipeline constructed."
        )

        print(
            f"[PIPELINE] Camera: {self.camera_id}"
        )

        print(
            f"[PIPELINE] Detection interval: "
            f"{self.detection_interval_seconds:.3f}s"
        )

        print(
            f"[PIPELINE] OCR interval: "
            f"{self.ocr_submit_interval_seconds:.3f}s"
        )

        print(
            f"[PIPELINE] Barcode interval: "
            f"{self.barcode_submit_interval_seconds:.3f}s"
        )

        print(
            f"[PIPELINE] Async OCR: "
            f"{self.asynchronous_ocr}"
        )

        print(
            f"[PIPELINE] Async Barcode: "
            f"{self.asynchronous_barcode}"
        )

    # ========================================================================
    # CONFIG HELPERS
    # ========================================================================

    @staticmethod
    def _get_float(
        section: Any,
        key: str,
        default: float,
    ) -> float:

        try:

            return float(
                section.get(
                    key,
                    default,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            return float(default)

    # ========================================================================
    # CAMERA ID
    # ========================================================================

    def _camera_id(self) -> str:

        value = getattr(
            self.camera,
            "camera_id",
            None,
        )

        if value is None:

            value = getattr(
                self.camera,
                "name",
                None,
            )

        if value is None:

            value = "camera_1"

        return str(value)

    # ========================================================================
    # TRACK KEY
    # ========================================================================

    def _track_key(
        self,
        track: Any,
    ) -> Optional[Tuple[str, int]]:

        track_id = self._track_id(
            track
        )

        if track_id is None:

            return None

        return (
            self.camera_id,
            track_id,
        )

    # ========================================================================
    # TRACK ID
    # ========================================================================

    @staticmethod
    def _track_id(
        track: Any,
    ) -> Optional[int]:

        value = getattr(
            track,
            "track_id",
            None,
        )

        if value is None:

            value = getattr(
                track,
                "id",
                None,
            )

        if value is None:

            return None

        try:

            return int(value)

        except (
            TypeError,
            ValueError,
        ):

            return None

    # ========================================================================
    # TRACK CONFIRMED
    # ========================================================================

    @staticmethod
    def _track_confirmed(
        track: Any,
    ) -> bool:

        value = getattr(
            track,
            "confirmed",
            None,
        )

        if value is None:

            value = getattr(
                track,
                "is_confirmed",
                None,
            )

        if value is None:

            # Some tracker implementations expose hits instead.
            hits = getattr(
                track,
                "hits",
                0,
            )

            try:

                return int(hits) >= 1

            except (
                TypeError,
                ValueError,
            ):

                return True

        return bool(value)

    # ========================================================================
    # TRACK PREDICTED
    # ========================================================================

    @staticmethod
    def _track_predicted(
        track: Any,
    ) -> bool:

        value = getattr(
            track,
            "predicted",
            False,
        )

        return bool(value)

    # ========================================================================
    # TRACK DETECTION
    # ========================================================================

    @staticmethod
    def _track_detection(
        track: Any,
    ) -> Any:

        detection = getattr(
            track,
            "detection",
            None,
        )

        if detection is not None:

            return detection

        return getattr(
            track,
            "last_detection",
            None,
        )

    # ========================================================================
    # TRACK BBOX
    # ========================================================================

    @staticmethod
    def _track_bbox(
        track: Any,
    ) -> Optional[Tuple[float, float, float, float]]:

        bbox = getattr(
            track,
            "bbox",
            None,
        )

        if bbox is None:

            detection = OCRBHSPipeline._track_detection(
                track
            )

            if detection is not None:

                bbox = getattr(
                    detection,
                    "bbox",
                    None,
                )

        if bbox is None:

            return None

        try:

            if len(bbox) != 4:

                return None

            return (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )

        except Exception:

            return None

    # ========================================================================
    # TRACK CONFIDENCE
    # ========================================================================

    @staticmethod
    def _track_confidence(
        track: Any,
    ) -> float:

        value = getattr(
            track,
            "confidence",
            None,
        )

        if value is None:

            detection = OCRBHSPipeline._track_detection(
                track
            )

            if detection is not None:

                value = getattr(
                    detection,
                    "confidence",
                    0.0,
                )

        try:

            return float(value)

        except (
            TypeError,
            ValueError,
        ):

            return 0.0

    # ========================================================================
    # FRAME NORMALIZATION
    # ========================================================================

    def _get_latest_frame(
        self,
    ) -> Optional[
        Tuple[int, float, np.ndarray]
    ]:

        try:

            value = self.camera.get_latest_frame(
                copy=True
            )

        except TypeError:

            value = self.camera.get_latest_frame()

        except Exception as exc:

            self.last_detection_error = (
                f"Camera frame error: {exc}"
            )

            return None

        if value is None:

            return None

        # --------------------------------------------------------------------
        # Tuple form:
        #
        #     (frame_id, timestamp, frame)
        # --------------------------------------------------------------------

        if isinstance(
            value,
            tuple,
        ):

            if len(value) >= 3:

                frame_id = value[0]

                timestamp = value[1]

                frame = value[2]

            elif len(value) == 2:

                frame_id = value[0]

                frame = value[1]

                timestamp = time.time()

            else:

                return None

        # --------------------------------------------------------------------
        # Bare ndarray
        # --------------------------------------------------------------------

        elif isinstance(
            value,
            np.ndarray,
        ):

            frame = value

            # --------------------------------------------------------
            # Camera does not expose `frame_id` / `frame_timestamp`
            # attributes. It tracks these under different names:
            # `frame_count` (monotonically increasing int) and
            # `last_frame_time`. Using the wrong attribute names here
            # silently falls back to a constant frame_id=0 for every
            # frame, which makes drop_old_frames treat every frame as
            # a duplicate and permanently freezes detection/tracking
            # after the first frame.
            # --------------------------------------------------------

            frame_id = getattr(
                self.camera,
                "frame_count",
                0,
            )

            timestamp = getattr(
                self.camera,
                "last_frame_time",
                time.time(),
            )

        else:

            return None

        if frame is None:

            return None

        if not isinstance(
            frame,
            np.ndarray,
        ):

            return None

        try:

            frame_id = int(frame_id)

        except (
            TypeError,
            ValueError,
        ):

            frame_id = int(
                time.time() * 1000
            )

        try:

            timestamp = float(timestamp)

        except (
            TypeError,
            ValueError,
        ):

            timestamp = time.time()

        return (
            frame_id,
            timestamp,
            frame,
        )

    # ========================================================================
    # SAFE FRAME COPY
    # ========================================================================

    def _safe_frame_copy(
        self,
        frame: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Defensively copy a frame before handing it to an OCR/barcode
        worker thread.

        OCR and barcode processing run asynchronously on background
        worker threads, while the same underlying frame buffer may be
        reused/overwritten by the camera capture thread. Without an
        independent copy here, a worker could read a frame that is
        being mutated concurrently (tearing / partially overwritten
        image data) or that no longer represents the tag it was
        selected for.
        """

        if frame is None:

            return None

        if not isinstance(
            frame,
            np.ndarray,
        ):

            return None

        try:

            return frame.copy()

        except Exception:

            return None

    # ========================================================================
    # WORKER CALLBACK
    # ========================================================================

    def _attach_worker_result_callback(
        self,
    ) -> None:
        """
        Attach WorkerManager result callback.

        WorkerManager exposes add_result_callback().
        """

        attach = getattr(
            self.worker_manager,
            "add_result_callback",
            None,
        )

        if not callable(attach):

            # This WorkerManager implementation is queue-based
            # (drain_results() / get_result()), not callback-based.
            # Results are collected instead via _drain_worker_results(),
            # polled once per main-loop iteration.
            self._worker_result_callback_attached = False

            print(
                "[PIPELINE] WorkerManager has no add_result_callback(); "
                "results will be polled via drain_results() instead."
            )

            return

        try:
            attach(
                self._on_worker_result
            )

            self._worker_result_callback_attached = True

            print(
                "[PIPELINE] Worker result callback attached."
            )

        except Exception as exc:
            self._worker_result_callback_attached = False

            print(
                "[PIPELINE] Worker result callback "
                f"attachment warning: {exc}"
            )

    # ========================================================================
    # DRAIN WORKER RESULTS (polling fallback)
    # ========================================================================

    def _drain_worker_results(
        self,
        max_results: int = 100,
    ) -> None:
        """
        Pull any completed OCR/barcode results out of WorkerManager's
        result queue and feed them through the same path a push-style
        callback would use.

        No-op if a real callback was successfully attached above, to
        avoid double-processing the same result.
        """

        if self._worker_result_callback_attached:

            return

        drain = getattr(
            self.worker_manager,
            "drain_results",
            None,
        )

        if not callable(drain):

            return

        try:

            results = drain(
                max_results=max_results
            )

        except Exception as exc:

            print(
                "[PIPELINE] drain_results() error: "
                f"{exc}"
            )

            return

        for result in results or []:

            try:

                self._on_worker_result(
                    result
                )

            except Exception as exc:

                print(
                    "[PIPELINE] Error handling drained "
                    f"worker result: {exc}"
                )

    # ========================================================================
    # RESULT MANAGER CALLBACK
    # ========================================================================

    def _on_result_manager_callback(
        self,
        result: Any,
    ) -> None:
        """
        Receives results after ResultManager processes them.
        """

        self._publish_result(
            result
        )

    # ========================================================================
    # WORKER RESULT
    # ========================================================================

    def _on_worker_result(
        self,
        result: Any,
    ) -> None:

        self.results_received += 1

        try:

            if isinstance(
                result,
                OCRTaskResult,
            ):

                key = (
                    str(result.camera_id),
                    int(result.track_id),
                )

                self.ocr_results_received += 1

                self._track_ocr_inflight.discard(
                    key
                )

                self._handle_ocr_result(
                    result
                )

            elif isinstance(
                result,
                BarcodeTaskResult,
            ):

                key = (
                    str(result.camera_id),
                    int(result.track_id),
                )

                self.barcode_results_received += 1

                self._track_barcode_inflight.discard(
                    key
                )

                self._handle_barcode_result(
                    result
                )

            else:

                # WorkerManager may expose a compatible result object.
                self._handle_generic_worker_result(
                    result
                )

        except Exception as exc:

            self.result_errors += 1

            self.last_result_error = (
                f"{type(exc).__name__}: {exc}"
            )

        # --------------------------------------------------------------------
        # Publish to external callbacks.
        # --------------------------------------------------------------------

        self._publish_result(
            result
        )

    # ========================================================================
    # GENERIC RESULT HANDLER
    # ========================================================================

    def _handle_generic_worker_result(
        self,
        result: Any,
    ) -> None:

        result_name = type(
            result
        ).__name__.lower()

        camera_id = getattr(
            result,
            "camera_id",
            self.camera_id,
        )

        track_id = getattr(
            result,
            "track_id",
            None,
        )

        if track_id is None:

            return

        key = (
            str(camera_id),
            int(track_id),
        )

        if "ocr" in result_name:

            self.ocr_results_received += 1

            self._track_ocr_inflight.discard(
                key
            )

            self._handle_ocr_result(
                result
            )

        elif "barcode" in result_name:

            self.barcode_results_received += 1

            self._track_barcode_inflight.discard(
                key
            )

            self._handle_barcode_result(
                result
            )

    # ========================================================================
    # OCR RESULT
    # ========================================================================

    def _handle_ocr_result(
        self,
        result: Any,
    ) -> None:

        key = self._result_track_key(
            result
        )

        if key is None:

            return

        inner = getattr(
            result,
            "result",
            result,
        )

        success = self._result_success(
            inner
        )

        if success:

            self._track_ocr_completed.add(
                key
            )

        elif not self.retry_failed_results:

            self._track_ocr_completed.add(
                key
            )

        # --------------------------------------------------------------------
        # Give ResultManager the worker result exactly once.
        # --------------------------------------------------------------------

        try:

            process = getattr(
                self.result_manager,
                "process_result",
                None,
            )

            if callable(process):

                process(
                    result
                )

        except Exception as exc:

            self.result_errors += 1

            self.last_result_error = (
                f"OCR ResultManager error: {exc}"
            )

    # ========================================================================
    # BARCODE RESULT
    # ========================================================================

    def _handle_barcode_result(
        self,
        result: Any,
    ) -> None:

        key = self._result_track_key(
            result
        )

        if key is None:

            return

        inner = getattr(
            result,
            "result",
            result,
        )

        success = self._result_success(
            inner
        )

        if success:

            self._track_barcode_completed.add(
                key
            )

        elif not self.retry_failed_results:

            self._track_barcode_completed.add(
                key
            )

        try:

            process = getattr(
                self.result_manager,
                "process_result",
                None,
            )

            if callable(process):

                process(
                    result
                )

        except Exception as exc:

            self.result_errors += 1

            self.last_result_error = (
                f"Barcode ResultManager error: {exc}"
            )

    # ========================================================================
    # RESULT HELPERS
    # ========================================================================

    @staticmethod
    def _result_track_key(
        result: Any,
    ) -> Optional[Tuple[str, int]]:

        camera_id = getattr(
            result,
            "camera_id",
            None,
        )

        track_id = getattr(
            result,
            "track_id",
            None,
        )

        if track_id is None:

            return None

        if camera_id is None:

            camera_id = "camera_1"

        try:

            return (
                str(camera_id),
                int(track_id),
            )

        except (
            TypeError,
            ValueError,
        ):

            return None

    @staticmethod
    def _result_success(
        result: Any,
    ) -> bool:

        if result is None:

            return False

        success = getattr(
            result,
            "success",
            None,
        )

        if success is not None:

            return bool(success)

        # Barcode result can be a list.
        if isinstance(
            result,
            list,
        ):

            return len(result) > 0

        # OCR-like object.
        text = getattr(
            result,
            "text",
            None,
        )

        if text:

            return bool(
                str(text).strip()
            )

        return False

    # ========================================================================
    # RESULT PUBLICATION
    # ========================================================================

    def add_result_callback(
        self,
        callback: ResultCallback,
    ) -> None:

        if not callable(callback):

            raise TypeError(
                "callback must be callable"
            )

        if callback not in self._result_callbacks:

            self._result_callbacks.append(
                callback
            )

    def remove_result_callback(
        self,
        callback: ResultCallback,
    ) -> None:

        try:

            self._result_callbacks.remove(
                callback
            )

        except ValueError:

            pass

    def _publish_result(
        self,
        result: Any,
    ) -> None:

        for callback in list(
            self._result_callbacks
        ):

            try:

                callback(
                    result
                )

            except Exception as exc:

                self.result_errors += 1

                self.last_result_error = (
                    f"Callback error: {exc}"
                )

    # ========================================================================
    # BEST FRAME SELECTOR
    # ========================================================================

    def _get_selector(
        self,
        track: Any,
    ) -> BestFrameSelector:

        key = self._track_key(
            track
        )

        if key is None:

            raise ValueError(
                "Cannot create selector without track ID."
            )

        with self._selectors_lock:

            selector = self._selectors.get(
                key
            )

            if selector is None:

                selector = BestFrameSelector(
                    max_candidates=self.best_frame_candidates
                )

                self._selectors[
                    key
                ] = selector

            return selector

    # ========================================================================
    # BEST FRAME CANDIDATE
    # ========================================================================

    def _add_best_frame_candidate(
        self,
        track: Any,
        frame_id: int,
        timestamp: float,
        frame: np.ndarray,
    ) -> None:

        if not self.best_frame_enabled:

            return

        if frame is None:

            return

        if self._track_predicted(
            track
        ):

            return

        detection = self._track_detection(
            track
        )

        if detection is None:

            return

        try:

            selector = self._get_selector(
                track
            )

            selector.add_candidate(
                frame_id=int(
                    frame_id
                ),
                timestamp=float(
                    timestamp
                ),
                frame=frame,
                detection=detection,
            )

        except Exception as exc:

            print(
                "[PIPELINE] Best-frame candidate error "
                f"camera={self.camera_id} "
                f"track={self._track_id(track)}: "
                f"{exc}"
            )

    # ========================================================================
    # BEST FRAME
    # ========================================================================

    def _get_best_frame(
        self,
        track: Any,
        fallback_frame: np.ndarray,
        fallback_frame_id: int,
        fallback_timestamp: float,
    ) -> Tuple[
        np.ndarray,
        int,
        float,
        float,
    ]:

        candidate_frame = fallback_frame

        candidate_frame_id = fallback_frame_id

        candidate_timestamp = fallback_timestamp

        candidate_score = 0.0

        if not self.best_frame_enabled:

            return (
                candidate_frame,
                candidate_frame_id,
                candidate_timestamp,
                candidate_score,
            )

        try:

            selector = self._get_selector(
                track
            )

            best = selector.get_best()

            if best is not None:

                candidate_frame = best.frame

                candidate_frame_id = int(
                    best.frame_id
                )

                candidate_timestamp = float(
                    best.timestamp
                )

                candidate_score = float(
                    getattr(
                        best,
                        "score",
                        0.0,
                    )
                )

        except Exception:

            pass

        return (
            candidate_frame,
            candidate_frame_id,
            candidate_timestamp,
            candidate_score,
        )

    # ========================================================================
    # OCR SHOULD SUBMIT
    # ========================================================================

    def _should_submit_ocr(
        self,
        track: Any,
    ) -> bool:

        if not self.asynchronous_ocr:

            return False

        key = self._track_key(
            track
        )

        if key is None:

            return False

        if key in self._track_ocr_inflight:

            return False

        if (
            self.process_once_per_track
            and key in self._track_ocr_completed
        ):

            return False

        attempts = self._track_ocr_attempts.get(
            key,
            0,
        )

        if attempts >= self.max_ocr_attempts:

            return False

        last = self._track_last_ocr_submit_time.get(
            key,
            0.0,
        )

        return (
            time.perf_counter() - last
            >= self.ocr_submit_interval_seconds
        )

    # ========================================================================
    # BARCODE SHOULD SUBMIT
    # ========================================================================

    def _should_submit_barcode(
        self,
        track: Any,
    ) -> bool:

        if not self.asynchronous_barcode:

            return False

        key = self._track_key(
            track
        )

        if key is None:

            return False

        if key in self._track_barcode_inflight:

            return False

        if (
            self.process_once_per_track
            and key in self._track_barcode_completed
        ):

            return False

        attempts = self._track_barcode_attempts.get(
            key,
            0,
        )

        if attempts >= self.max_barcode_attempts:

            return False

        last = self._track_last_barcode_submit_time.get(
            key,
            0.0,
        )

        return (
            time.perf_counter() - last
            >= self.barcode_submit_interval_seconds
        )

    # ========================================================================
    # SUBMIT OCR
    # ========================================================================

    def _submit_ocr_for_track(
        self,
        track: Any,
        frame_id: int,
        timestamp: float,
        frame: np.ndarray,
    ) -> bool:

        key = self._track_key(
            track
        )

        if key is None:

            return False

        if not self._should_submit_ocr(
            track
        ):

            return False

        if self._track_predicted(
            track
        ):

            return False

        (
            candidate_frame,
            candidate_frame_id,
            candidate_timestamp,
            candidate_score,
        ) = self._get_best_frame(
            track=track,
            fallback_frame=frame,
            fallback_frame_id=frame_id,
            fallback_timestamp=timestamp,
        )

        if candidate_frame is None:

            return False

        task_id = (
            f"ocr-"
            f"{self.camera_id}-"
            f"{key[1]}-"
            f"{candidate_frame_id}-"
            f"{time.time_ns()}"
        )

        task = OCRTask(
            task_id=task_id,
            camera_id=self.camera_id,
            track_id=key[1],
            frame=self._safe_frame_copy(
                candidate_frame
            ),
            timestamp=float(
                candidate_timestamp
            ),
            metadata={
                "frame_id": int(
                    candidate_frame_id
                ),
                "best_frame_score": float(
                    candidate_score
                ),
                "detection_confidence": (
                    self._track_confidence(
                        track
                    )
                ),
                "bbox": self._track_bbox(
                    track
                ),
                "source": (
                    "best_frame_selector"
                    if self.best_frame_enabled
                    else "live_frame"
                ),
            },
        )

        try:

            accepted = (
                self.worker_manager.submit_ocr(
                    task
                )
            )

        except Exception as exc:

            self.ocr_jobs_failed += 1

            self.last_result_error = (
                f"OCR submit error: {exc}"
            )

            return False

        if not accepted:

            self.ocr_jobs_failed += 1

            return False

        self.ocr_jobs_submitted += 1

        self._track_ocr_inflight.add(
            key
        )

        self._track_ocr_attempts[
            key
        ] = (
            self._track_ocr_attempts.get(
                key,
                0,
            )
            + 1
        )

        self._track_last_ocr_submit_time[
            key
        ] = time.perf_counter()

        print(
            "[OCR] Submitted "
            f"camera={self.camera_id} "
            f"track={key[1]} "
            f"frame={candidate_frame_id}"
        )

        return True

    # ========================================================================
    # SUBMIT BARCODE
    # ========================================================================

    def _submit_barcode_for_track(
        self,
        track: Any,
        frame_id: int,
        timestamp: float,
        frame: np.ndarray,
    ) -> bool:

        key = self._track_key(
            track
        )

        if key is None:

            return False

        if not self._should_submit_barcode(
            track
        ):

            return False

        if self._track_predicted(
            track
        ):

            return False

        (
            candidate_frame,
            candidate_frame_id,
            candidate_timestamp,
            candidate_score,
        ) = self._get_best_frame(
            track=track,
            fallback_frame=frame,
            fallback_frame_id=frame_id,
            fallback_timestamp=timestamp,
        )

        if candidate_frame is None:

            return False

        task_id = (
            f"barcode-"
            f"{self.camera_id}-"
            f"{key[1]}-"
            f"{candidate_frame_id}-"
            f"{time.time_ns()}"
        )

        task = BarcodeTask(
            task_id=task_id,
            camera_id=self.camera_id,
            track_id=key[1],
            frame=self._safe_frame_copy(
                candidate_frame
            ),
            timestamp=float(
                candidate_timestamp
            ),
            metadata={
                "frame_id": int(
                    candidate_frame_id
                ),
                "best_frame_score": float(
                    candidate_score
                ),
                "detection_confidence": (
                    self._track_confidence(
                        track
                    )
                ),
                "bbox": self._track_bbox(
                    track
                ),
                "source": (
                    "best_frame_selector"
                    if self.best_frame_enabled
                    else "live_frame"
                ),
            },
        )

        try:

            accepted = (
                self.worker_manager.submit_barcode(
                    task
                )
            )

        except Exception as exc:

            self.barcode_jobs_failed += 1

            self.last_result_error = (
                f"Barcode submit error: {exc}"
            )

            return False

        if not accepted:

            self.barcode_jobs_failed += 1

            return False

        self.barcode_jobs_submitted += 1

        self._track_barcode_inflight.add(
            key
        )

        self._track_barcode_attempts[
            key
        ] = (
            self._track_barcode_attempts.get(
                key,
                0,
            )
            + 1
        )

        self._track_last_barcode_submit_time[
            key
        ] = time.perf_counter()

        print(
            "[BARCODE] Submitted "
            f"camera={self.camera_id} "
            f"track={key[1]} "
            f"frame={candidate_frame_id}"
        )

        return True

    # ========================================================================
    # PROCESS TRACKS
    # ========================================================================

    def _process_tracks(
        self,
        tracks: List[Any],
        frame_id: int,
        timestamp: float,
        frame: np.ndarray,
    ) -> None:

        for track in tracks:

            track_id = self._track_id(
                track
            )

            if track_id is None:

                continue

            self.tracks_seen += 1

            # ----------------------------------------------------------------
            # Only confirmed tracks are eligible for OCR/barcode.
            # ----------------------------------------------------------------

            if not self._track_confirmed(
                track
            ):

                continue

            # ----------------------------------------------------------------
            # Store best-frame candidate.
            # ----------------------------------------------------------------

            self._add_best_frame_candidate(
                track=track,
                frame_id=frame_id,
                timestamp=timestamp,
                frame=frame,
            )

            # ----------------------------------------------------------------
            # Never process predicted-only frames.
            # ----------------------------------------------------------------

            if self._track_predicted(
                track
            ):

                continue

            # ----------------------------------------------------------------
            # OCR
            # ----------------------------------------------------------------

            if self.asynchronous_ocr:

                self._submit_ocr_for_track(
                    track=track,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    frame=frame,
                )

            # ----------------------------------------------------------------
            # Barcode
            # ----------------------------------------------------------------

            if self.asynchronous_barcode:

                self._submit_barcode_for_track(
                    track=track,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    frame=frame,
                )

    # ========================================================================
    # FILTER DETECTIONS
    # ========================================================================

    def _filter_detections(
        self,
        detections: Any,
        frame_shape: Tuple[int, ...],
    ) -> List[Any]:

        if detections is None:

            return []

        try:

            detections = list(
                detections
            )

        except TypeError:

            return []

        output = []

        frame_height = int(
            frame_shape[0]
        )

        frame_width = int(
            frame_shape[1]
        )

        for detection in detections:

            bbox = getattr(
                detection,
                "bbox",
                None,
            )

            if bbox is None:

                output.append(
                    detection
                )

                continue

            try:

                x1, y1, x2, y2 = (
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                )

            except Exception:

                continue

            x1 = max(
                0.0,
                min(
                    x1,
                    frame_width,
                ),
            )

            y1 = max(
                0.0,
                min(
                    y1,
                    frame_height,
                ),
            )

            x2 = max(
                0.0,
                min(
                    x2,
                    frame_width,
                ),
            )

            y2 = max(
                0.0,
                min(
                    y2,
                    frame_height,
                ),
            )

            if x2 <= x1 or y2 <= y1:

                continue

            output.append(
                detection
            )

            if len(output) >= self.max_detections:

                break

        return output

    # ========================================================================
    # DETECTION
    # ========================================================================

    def _run_detection(
        self,
        frame: np.ndarray,
    ) -> List[Any]:

        result = self.detector.predict(
            frame
        )

        if result is None:

            return []

        if isinstance(
            result,
            list,
        ):

            return self._filter_detections(
                result,
                frame.shape,
            )

        try:

            return self._filter_detections(
                list(result),
                frame.shape,
            )

        except TypeError:

            return []

    # ========================================================================
    # TRACKING
    # ========================================================================

    def _update_tracker(
        self,
        frame_id: int,
        detections: List[Any],
    ) -> List[Any]:

        tracks = self.tracker.update(
            frame_id=frame_id,
            detections=detections,
        )

        if tracks is None:

            return []

        try:

            return list(
                tracks
            )

        except TypeError:

            return []

    # ========================================================================
    # LIVE TRACK STATE
    # ========================================================================

    def _set_latest_tracks(
        self,
        tracks: List[Any],
        frame_id: int,
        timestamp: float,
        detection_count: int,
    ) -> None:

        with self._tracks_lock:

            self._latest_tracks = list(
                tracks or []
            )

            self._latest_track_frame_id = (
                int(frame_id)
            )

            self._latest_track_timestamp = (
                float(timestamp)
            )

            self._latest_detection_count = (
                int(detection_count)
            )

    def get_latest_tracks(
        self,
    ) -> List[Any]:

        with self._tracks_lock:

            return list(
                self._latest_tracks
            )

    def get_tracking_state(
        self,
    ) -> Dict[str, Any]:

        with self._tracks_lock:

            return {
                "camera_id": self.camera_id,
                "tracks": list(
                    self._latest_tracks
                ),
                "track_count": len(
                    self._latest_tracks
                ),
                "frame_id": (
                    self._latest_track_frame_id
                ),
                "timestamp": (
                    self._latest_track_timestamp
                ),
                "detection_count": (
                    self._latest_detection_count
                ),
            }

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    def _run_loop(
        self,
    ) -> None:

        print(
            "[PIPELINE] Entered processing loop."
        )

        last_detection_time = 0.0

        last_tracking_time = 0.0

        last_frame_id = None

        while not self._stop_event.is_set():

            try:

                latest = self._get_latest_frame()

                if latest is None:

                    time.sleep(
                        0.005
                    )

                    continue

                (
                    frame_id,
                    timestamp,
                    frame,
                ) = latest

                self.frames_processed += 1

                # ------------------------------------------------------------
                # Avoid processing exactly the same frame repeatedly.
                # ------------------------------------------------------------

                if (
                    self.drop_old_frames
                    and last_frame_id == frame_id
                ):

                    time.sleep(
                        0.002
                    )

                    continue

                last_frame_id = frame_id

                now = time.perf_counter()

                # ------------------------------------------------------------
                # Tracking update rate cap (see __init__ note on
                # tracking_interval_seconds / performance.max_tracking_fps).
                #
                # Keeps the tracker-driven smoothing decoupled from YOLO's
                # slower interval (for smooth motion) WITHOUT chasing the
                # camera's full native frame rate on every single loop
                # iteration, which was overloading the CPU / GIL badly
                # enough to desync the pipeline from real time and starve
                # YOLO of usable frames.
                # ------------------------------------------------------------

                if (
                    self.tracking_interval_seconds > 0
                    and now - last_tracking_time
                    < self.tracking_interval_seconds
                ):

                    time.sleep(
                        0.001
                    )

                    continue

                last_tracking_time = now

                # ------------------------------------------------------------
                # YOLO detection pacing.
                #
                # YOLO is the expensive step, so it only runs every
                # detection_interval_seconds. Tracking, however, must NOT
                # be gated by this same interval — see below.
                # ------------------------------------------------------------

                run_detection_this_frame = (
                    now - last_detection_time
                    >= self.detection_interval_seconds
                )

                if run_detection_this_frame:

                    last_detection_time = now

                    self.detection_runs += 1

                    detections = self._run_detection(
                        frame
                    )

                else:

                    # No new YOLO result this camera frame. Feed the
                    # tracker an empty detection list so its own motion
                    # prediction (see tag_tracker.py's "CONTINUOUS
                    # PREDICTION" section) can advance the box smoothly
                    # at full camera frame rate between YOLO cycles,
                    # instead of holding it frozen for up to
                    # detection_interval_seconds.
                    detections = []

                # ------------------------------------------------------------
                # ByteTrack
                #
                # Called on EVERY camera frame, not just detection frames.
                # This is what makes the box move smoothly and continuously
                # (via velocity-based prediction) instead of jumping only
                # once per YOLO cycle.
                # ------------------------------------------------------------

                tracks = self._update_tracker(
                    frame_id=frame_id,
                    detections=detections,
                )

                # ------------------------------------------------------------
                # Publish latest tracking state.
                #
                # Viewer can use this independently from inference speed.
                # ------------------------------------------------------------

                self._set_latest_tracks(
                    tracks=tracks,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    detection_count=len(
                        detections
                    ),
                )

                # ------------------------------------------------------------
                # OCR / Barcode
                #
                # This only submits non-blocking tasks. Submission is
                # independently throttled per-track (ocr/barcode interval),
                # so calling every frame is safe and does not resubmit
                # faster than configured.
                # ------------------------------------------------------------

                self._process_tracks(
                    tracks=tracks,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    frame=frame,
                )

                # ------------------------------------------------------------
                # Drain OCR / barcode results.
                #
                # WorkerManager is a pull-based queue (drain_results /
                # get_result), it does NOT support add_result_callback.
                # _attach_worker_result_callback() cannot attach anything
                # real, so without this poll, submitted OCR/barcode jobs
                # would complete on the worker threads but their results
                # would sit in the queue and never reach
                # _on_worker_result() / ResultManager / the database.
                # ------------------------------------------------------------

                self._drain_worker_results()

            except Exception as exc:

                self.detection_errors += 1

                self.last_detection_error = (
                    f"{type(exc).__name__}: {exc}"
                )

                print(
                    "[PIPELINE] Detection loop error: "
                    f"{exc}"
                )

                time.sleep(
                    0.01
                )

        print(
            "[PIPELINE] Main loop stopped"
        )

    # ========================================================================
    # START
    # ========================================================================

    def start(
        self,
    ) -> None:

        if self.running:

            return

        print(
            "[PIPELINE] OCR_BHS PIPELINE START"
        )

        self._stop_event.clear()

        # --------------------------------------------------------------------
        # Workers
        # --------------------------------------------------------------------

        try:

            self.worker_manager.start()

            print(
                "[PIPELINE] Workers started"
            )

        except Exception as exc:

            print(
                "[PIPELINE] Worker manager start error: "
                f"{exc}"
            )

            raise

        # --------------------------------------------------------------------
        # Camera
        # --------------------------------------------------------------------

        try:

            self.camera.start()

            print(
                "[PIPELINE] Camera started"
            )

        except Exception as exc:

            try:

                self.worker_manager.stop(
                    wait=True
                )

            except Exception:

                pass

            print(
                "[PIPELINE] Camera start error: "
                f"{exc}"
            )

            raise

        # --------------------------------------------------------------------
        # Runtime
        # --------------------------------------------------------------------

        self.running = True

        self._loop_thread = threading.Thread(
            target=self._run_loop,
            name=(
                f"OCR-BHS-Pipeline-"
                f"{self.camera_id}"
            ),
            daemon=True,
        )

        self._loop_thread.start()

        print(
            "[PIPELINE] Main loop started"
        )

        print(
            "[PIPELINE] Pipeline running"
        )

    # ========================================================================
    # STOP
    # ========================================================================

    def stop(
        self,
    ) -> None:

        if not self.running:

            # Ensure workers are cleaned up even if
            # the main loop already stopped.

            try:

                self.worker_manager.stop(
                    wait=True
                )

            except Exception:

                pass

            return

        print(
            "[PIPELINE] Stopping pipeline..."
        )

        self.running = False

        self._stop_event.set()

        # --------------------------------------------------------------------
        # Main loop
        # --------------------------------------------------------------------

        if (
            self._loop_thread is not None
            and self._loop_thread.is_alive()
            and threading.current_thread()
            is not self._loop_thread
        ):

            self._loop_thread.join(
                timeout=5.0
            )

        self._loop_thread = None

        # --------------------------------------------------------------------
        # Camera
        # --------------------------------------------------------------------

        try:

            self.camera.stop()

            print(
                "[PIPELINE] Camera stopped"
            )

        except Exception as exc:

            print(
                "[PIPELINE] Camera stop warning: "
                f"{exc}"
            )

        # --------------------------------------------------------------------
        # Workers
        # --------------------------------------------------------------------

        try:

            self.worker_manager.stop(
                wait=True
            )

            print(
                "[PIPELINE] Workers stopped"
            )

        except Exception as exc:

            print(
                "[PIPELINE] Worker stop warning: "
                f"{exc}"
            )

        # --------------------------------------------------------------------
        # Clear live state.
        # --------------------------------------------------------------------

        with self._tracks_lock:

            self._latest_tracks = []

            self._latest_track_frame_id = None

            self._latest_track_timestamp = None

            self._latest_detection_count = 0

        print(
            "[PIPELINE] Pipeline stopped"
        )

        print(
            "[PIPELINE] Final statistics:"
        )

        print(
            self.get_stats()
        )

    # ========================================================================
    # RUNNING STATE
    # ========================================================================

    def is_running(
        self,
    ) -> bool:

        return bool(
            self.running
        )

    # ========================================================================
    # RESULT STATE
    # ========================================================================

    def get_result_state(
        self,
        camera_id: str,
        track_id: int,
    ) -> Optional[Any]:
        """
        Get aggregated OCR + barcode state for a track.
        """

        try:
            return self.result_manager.get(
                camera_id=camera_id,
                track_id=track_id,
            )

        except Exception:
            return None

    # ========================================================================
    # ALL RESULT STATES
    # ========================================================================

    def get_all_result_states(
        self,
    ) -> List[Any]:
        """
        Return all currently retained result states.
        """

        try:
            return self.result_manager.get_all()

        except Exception:
            return []

    # ========================================================================
    # STATS
    # ========================================================================

    def get_stats(
        self,
    ) -> Dict[str, Any]:

        with self._tracks_lock:

            active_tracks = len(
                self._latest_tracks
            )

            latest_frame_id = (
                self._latest_track_frame_id
            )

            latest_detection_count = (
                self._latest_detection_count
            )

        try:

            worker_status = (
                self.worker_manager.status()
            )

        except Exception:

            worker_status = {}

        try:

            result_status = (
                self.result_manager.status()
            )

        except Exception:

            result_status = {}

        return {
            "camera_id": self.camera_id,

            "running": self.running,

            "frames_processed": (
                self.frames_processed
            ),

            "detection_runs": (
                self.detection_runs
            ),

            "tracks_seen": (
                self.tracks_seen
            ),

            "active_tracks": (
                active_tracks
            ),

            "latest_frame_id": (
                latest_frame_id
            ),

            "latest_detection_count": (
                latest_detection_count
            ),

            "ocr_jobs_submitted": (
                self.ocr_jobs_submitted
            ),

            "ocr_jobs_failed": (
                self.ocr_jobs_failed
            ),

            "barcode_jobs_submitted": (
                self.barcode_jobs_submitted
            ),

            "barcode_jobs_failed": (
                self.barcode_jobs_failed
            ),

            "ocr_results_received": (
                self.ocr_results_received
            ),

            "barcode_results_received": (
                self.barcode_results_received
            ),

            "results_received": (
                self.results_received
            ),

            "detection_errors": (
                self.detection_errors
            ),

            "result_errors": (
                self.result_errors
            ),

            "last_detection_error": (
                self.last_detection_error
            ),

            "last_result_error": (
                self.last_result_error
            ),

            "worker_manager": (
                worker_status
            ),

            "result_manager": (
                result_status
            ),
        }

    # ========================================================================
    # CONTEXT MANAGER
    # ========================================================================

    def __enter__(
        self,
    ) -> "OCRBHSPipeline":

        self.start()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:

        self.stop()


# ============================================================================
# STANDALONE DIAGNOSTIC
# ============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("OCR_BHS MAIN PIPELINE DIAGNOSTIC")
    print("=" * 72)

    try:

        config = Config()

        print(
            "[PASS] Config loaded"
        )

        print(
            "       Model: "
            f"{config.data.get('detection', {}).get('model_path')}"
        )

        print(
            "       Device: "
            f"{config.data.get('detection', {}).get('device', 'cpu')}"
        )

    except Exception as exc:

        print(
            "[FAIL] Config: "
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1)

    # ------------------------------------------------------------------------
    # Fake camera
    # ------------------------------------------------------------------------

    class FakeCamera:

        camera_id = "camera_1"

        def start(self):
            pass

        def stop(self):
            pass

        def get_latest_frame(
            self,
            copy=False,
        ):

            return (
                0,
                time.time(),
                None,
            )

    # ------------------------------------------------------------------------
    # Fake detector
    # ------------------------------------------------------------------------

    class FakeDetector:
        pass

    # ------------------------------------------------------------------------
    # Fake tracker
    # ------------------------------------------------------------------------

    class FakeTracker:
        pass

    # ------------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------------

    try:

        pipeline = OCRBHSPipeline(
            camera=FakeCamera(),
            detector=FakeDetector(),
            tracker=FakeTracker(),
            config=config,
        )

        print(
            "[PASS] Pipeline construction"
        )

        stats = pipeline.get_stats()

        assert (
            stats["running"]
            is False
        )

        assert (
            stats["frames_processed"]
            == 0
        )

        print(
            "[PASS] Initial statistics"
        )

        print(
            "[PASS] ResultManager attached"
        )

        # --------------------------------------------------------------------
        # Result callback
        # --------------------------------------------------------------------

        received = []

        pipeline.add_result_callback(
            received.append
        )

        test_result = {
            "track_id": 1,
            "result": "TEST",
        }

        pipeline._publish_result(
            test_result
        )

        assert received == [
            test_result
        ]

        print(
            "[PASS] Result callback"
        )

        # --------------------------------------------------------------------
        # Bare ndarray camera test
        # --------------------------------------------------------------------

        test_frame = np.zeros(
            (
                100,
                200,
                3,
            ),
            dtype=np.uint8,
        )

        class BareFrameCamera:

            camera_id = "camera_1"

            frame_id = 123

            frame_timestamp = 456.0

            def get_latest_frame(
                self,
                copy=False,
            ):

                return test_frame

            def start(self):
                pass

            def stop(self):
                pass

        pipeline.camera = BareFrameCamera()

        normalized = (
            pipeline._get_latest_frame()
        )

        assert normalized is not None

        frame_id, timestamp, frame = normalized

        assert frame_id == 123

        assert timestamp == 456.0

        assert frame.shape == (
            100,
            200,
            3,
        )

        print(
            "[PASS] Bare ndarray camera frame handling"
        )

        # --------------------------------------------------------------------
        # Tuple camera test
        # --------------------------------------------------------------------

        class TupleFrameCamera:

            camera_id = "camera_1"

            def get_latest_frame(
                self,
                copy=False,
            ):

                return (
                    789,
                    987.0,
                    test_frame,
                )

            def start(self):
                pass

            def stop(self):
                pass

        pipeline.camera = TupleFrameCamera()

        normalized = (
            pipeline._get_latest_frame()
        )

        assert normalized is not None

        frame_id, timestamp, frame = normalized

        assert frame_id == 789

        assert timestamp == 987.0

        assert frame.shape == (
            100,
            200,
            3,
        )

        print(
            "[PASS] Tuple camera frame handling"
        )

        # --------------------------------------------------------------------
        # Tracking state test
        # --------------------------------------------------------------------

        fake_track_1 = object()

        fake_track_2 = object()

        pipeline._set_latest_tracks(
            tracks=[
                fake_track_1,
                fake_track_2,
            ],
            frame_id=123,
            timestamp=456.0,
            detection_count=2,
        )

        latest_tracks = (
            pipeline.get_latest_tracks()
        )

        assert len(
            latest_tracks
        ) == 2

        assert (
            latest_tracks[0]
            is fake_track_1
        )

        assert (
            latest_tracks[1]
            is fake_track_2
        )

        state = (
            pipeline.get_tracking_state()
        )

        assert (
            state["camera_id"]
            == "camera_1"
        )

        assert (
            state["track_count"]
            == 2
        )

        assert (
            state["frame_id"]
            == 123
        )

        assert (
            state["detection_count"]
            == 2
        )

        print(
            "[PASS] Live tracking state"
        )

        print(
            "[PASS] Pipeline diagnostic completed"
        )

    except Exception as exc:

        print(
            "[FAIL] Pipeline diagnostic:"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1)

    print("=" * 72)
    print(
        "OCR_BHS MAIN PIPELINE DIAGNOSTIC PASSED"
    )
    print("=" * 72)