"""
OCR_BHS - Result Manager
========================

Aggregates asynchronous OCR and barcode worker results.

Flow:

    OCR Worker
        \
         -> ResultManager -> Validator
        /
    Barcode Worker

Results are grouped by:

    (camera_id, track_id)

When both OCR and barcode results have arrived,
the state is validated and published.

The ResultManager does NOT write directly to SQL Server.
Use callbacks to connect DatabaseWorker / persistence.
"""

from __future__ import annotations

import threading
import time

from dataclasses import dataclass, field

from typing import Any, Callable, Optional

from app.pipeline.processing_queue import (
    BarcodeTaskResult,
    OCRTaskResult,
)

from app.validation.validator import (
    ValidationResult,
    Validator,
)


# ======================================================================
# TRACK RESULT STATE
# ======================================================================


@dataclass
class TrackResultState:
    """
    Final asynchronous state for one camera-local
    tracked IATA tag.
    """

    camera_id: str
    track_id: int

    # --------------------------------------------------------------
    # Worker results
    # --------------------------------------------------------------

    ocr_result: Optional[Any] = None

    barcode_result: Optional[Any] = None

    # --------------------------------------------------------------
    # Result arrival state
    # --------------------------------------------------------------

    ocr_received: bool = False

    barcode_received: bool = False

    # --------------------------------------------------------------
    # Result success state
    # --------------------------------------------------------------

    ocr_success: bool = False

    barcode_success: bool = False

    # --------------------------------------------------------------
    # Final validation
    # --------------------------------------------------------------

    validation_result: Optional[
        ValidationResult
    ] = None

    # --------------------------------------------------------------
    # Timing
    # --------------------------------------------------------------

    first_timestamp: float = field(
        default_factory=time.time
    )

    last_timestamp: float = field(
        default_factory=time.time
    )

    # --------------------------------------------------------------
    # Completion
    # --------------------------------------------------------------

    completed: bool = False

    # --------------------------------------------------------------
    # Extra information
    # --------------------------------------------------------------

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # ==============================================================
    # DICTIONARY
    # ==============================================================

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert state to a database/callback-friendly
        dictionary.

        Raw OCR/barcode results are intentionally included.
        """

        return {
            "camera_id": self.camera_id,
            "track_id": self.track_id,

            "ocr_result": self.ocr_result,
            "barcode_result": self.barcode_result,

            "ocr_received": self.ocr_received,
            "barcode_received": self.barcode_received,

            "ocr_success": self.ocr_success,
            "barcode_success": self.barcode_success,

            "completed": self.completed,

            "validation": (
                self.validation_result.to_dict()
                if self.validation_result is not None
                else None
            ),

            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,

            "metadata": dict(self.metadata),
        }


# ======================================================================
# RESULT MANAGER
# ======================================================================


class ResultManager:
    """
    Aggregates asynchronous OCR and barcode results.

    Results are grouped by:

        camera_id + track_id

    OCR and barcode can finish in either order.

    Once BOTH worker results arrive,
    Validator generates the final result.
    """

    def __init__(
        self,
        max_states: int = 1000,
        validator: Optional[Validator] = None,
    ) -> None:

        self.max_states = max(
            1,
            int(max_states),
        )

        self.validator = (
            validator
            if validator is not None
            else Validator()
        )

        self._lock = threading.RLock()

        self._states: dict[
            tuple[str, int],
            TrackResultState,
        ] = {}

        self._callbacks: list[
            Callable[[Any], None]
        ] = []

        # ----------------------------------------------------------
        # Counters
        # ----------------------------------------------------------

        self.ocr_results = 0
        self.barcode_results = 0
        self.total_results = 0

        self.completed_tracks = 0
        self.validated_tracks = 0
        self.valid_tracks = 0
        self.invalid_tracks = 0

        self.callback_errors = 0

    # ==================================================================
    # CALLBACKS
    # ==================================================================

    def add_callback(
        self,
        callback: Callable[[Any], None],
    ) -> None:
        """
        Add callback for published results.

        Callback receives:

        - OCRTaskResult
        - BarcodeTaskResult
        - TrackResultState

        A TrackResultState is published only when
        OCR + barcode are both available.
        """

        if not callable(callback):
            raise TypeError(
                "callback must be callable"
            )

        with self._lock:

            if callback not in self._callbacks:

                self._callbacks.append(
                    callback
                )

    def remove_callback(
        self,
        callback: Callable[[Any], None],
    ) -> None:

        with self._lock:

            if callback in self._callbacks:

                self._callbacks.remove(
                    callback
                )

    # ==================================================================
    # GET / CREATE STATE
    # ==================================================================

    def _get_state(
        self,
        camera_id: str,
        track_id: int,
    ) -> TrackResultState:

        key = (
            str(camera_id),
            int(track_id),
        )

        state = self._states.get(key)

        if state is None:

            now = time.time()

            state = TrackResultState(
                camera_id=str(camera_id),
                track_id=int(track_id),
                first_timestamp=now,
                last_timestamp=now,
            )

            self._states[key] = state

            self._trim()

        return state

    # ==================================================================
    # TRIM
    # ==================================================================

    def _trim(self) -> None:

        if len(self._states) <= self.max_states:
            return

        ordered = sorted(
            self._states.items(),
            key=lambda item: item[1].last_timestamp,
        )

        remove_count = (
            len(self._states)
            - self.max_states
        )

        for key, _ in ordered[:remove_count]:

            self._states.pop(
                key,
                None,
            )

    # ==================================================================
    # OCR EXTRACTION
    # ==================================================================

    @staticmethod
    def _extract_ocr(
        result: Any,
    ) -> tuple[str, float, bool]:

        if result is None:

            return "", 0.0, False

        if isinstance(result, dict):

            text = (
                result.get("normalized_text")
                or result.get("text")
                or ""
            )

            confidence = result.get(
                "confidence",
                0.0,
            )

            success = bool(
                result.get(
                    "success",
                    bool(text),
                )
            )

        else:

            text = getattr(
                result,
                "normalized_text",
                None,
            )

            if not text:

                text = getattr(
                    result,
                    "text",
                    "",
                )

            confidence = getattr(
                result,
                "confidence",
                0.0,
            )

            success = bool(
                getattr(
                    result,
                    "success",
                    bool(text),
                )
            )

        try:

            confidence = float(
                confidence
            )

        except (
            TypeError,
            ValueError,
        ):

            confidence = 0.0

        return (
            str(text or ""),
            confidence,
            success,
        )

    # ==================================================================
    # BARCODE EXTRACTION
    # ==================================================================

    @classmethod
    def _extract_barcode(
        cls,
        result: Any,
    ) -> tuple[str, float, bool]:

        if result is None:

            return "", 0.0, False

        if isinstance(result, dict):

            results = result.get(
                "results"
            )

            if isinstance(
                results,
                (list, tuple),
            ):

                return cls._best_barcode(
                    results
                )

            value = (
                result.get("value")
                or result.get("data")
                or result.get("barcode")
                or ""
            )

            confidence = result.get(
                "confidence",
                0.0,
            )

            success = bool(
                result.get(
                    "success",
                    bool(value),
                )
            )

        else:

            results = getattr(
                result,
                "results",
                None,
            )

            if isinstance(
                results,
                (list, tuple),
            ):

                return cls._best_barcode(
                    results
                )

            value = (
                getattr(
                    result,
                    "value",
                    None,
                )
                or getattr(
                    result,
                    "data",
                    None,
                )
                or getattr(
                    result,
                    "barcode",
                    None,
                )
                or ""
            )

            confidence = getattr(
                result,
                "confidence",
                0.0,
            )

            success = bool(
                getattr(
                    result,
                    "success",
                    bool(value),
                )
            )

        try:

            confidence = float(
                confidence
            )

        except (
            TypeError,
            ValueError,
        ):

            confidence = 0.0

        return (
            str(value or ""),
            confidence,
            success,
        )

    # ==================================================================
    # BEST BARCODE
    # ==================================================================

    @staticmethod
    def _best_barcode(
        items: list[Any] | tuple[Any, ...],
    ) -> tuple[str, float, bool]:

        candidates: list[
            tuple[str, float]
        ] = []

        for item in items:

            if isinstance(item, dict):

                value = (
                    item.get("value")
                    or item.get("data")
                    or item.get("barcode")
                    or ""
                )

                confidence = item.get(
                    "confidence",
                    0.0,
                )

            else:

                value = (
                    getattr(
                        item,
                        "value",
                        None,
                    )
                    or getattr(
                        item,
                        "data",
                        None,
                    )
                    or getattr(
                        item,
                        "barcode",
                        None,
                    )
                    or ""
                )

                confidence = getattr(
                    item,
                    "confidence",
                    0.0,
                )

            try:

                confidence = float(
                    confidence
                )

            except (
                TypeError,
                ValueError,
            ):

                confidence = 0.0

            if value:

                candidates.append(
                    (
                        str(value),
                        confidence,
                    )
                )

        if not candidates:

            return "", 0.0, False

        value, confidence = max(
            candidates,
            key=lambda item: (
                item[1],
                len(item[0]),
            ),
        )

        return (
            value,
            confidence,
            True,
        )

    # ==================================================================
    # PROCESS RESULT
    # ==================================================================

    def process_result(
        self,
        result: Any,
    ) -> Optional[TrackResultState]:

        if isinstance(
            result,
            OCRTaskResult,
        ):

            return self._process_ocr(
                result
            )

        if isinstance(
            result,
            BarcodeTaskResult,
        ):

            return self._process_barcode(
                result
            )

        return None

    # ==================================================================
    # OCR
    # ==================================================================

    def _process_ocr(
        self,
        result: OCRTaskResult,
    ) -> TrackResultState:

        with self._lock:

            state = self._get_state(
                result.camera_id,
                result.track_id,
            )

            state.ocr_result = (
                result.result
            )

            state.ocr_received = True

            (
                _,
                _,
                state.ocr_success,
            ) = self._extract_ocr(
                result.result
            )

            timestamp = self._safe_timestamp(
                result.timestamp
            )

            state.last_timestamp = timestamp

            state.metadata[
                "ocr_task_id"
            ] = result.task_id

            state.metadata[
                "ocr_metadata"
            ] = (
                result.metadata
                if isinstance(
                    result.metadata,
                    dict,
                )
                else {}
            )

            if isinstance(
                result.metadata,
                dict,
            ):

                if "frame_id" in result.metadata:

                    state.metadata[
                        "ocr_frame_id"
                    ] = result.metadata[
                        "frame_id"
                    ]

                if "source" in result.metadata:

                    state.metadata[
                        "ocr_source"
                    ] = result.metadata[
                        "source"
                    ]

            self.ocr_results += 1
            self.total_results += 1

            completed_now = (
                self._check_complete(
                    state
                )
            )

        # Publish outside the lock.
        self._publish(result)

        if completed_now:

            self._publish(state)

        return state

    # ==================================================================
    # BARCODE
    # ==================================================================

    def _process_barcode(
        self,
        result: BarcodeTaskResult,
    ) -> TrackResultState:

        with self._lock:

            state = self._get_state(
                result.camera_id,
                result.track_id,
            )

            state.barcode_result = (
                result.result
            )

            state.barcode_received = True

            (
                _,
                _,
                state.barcode_success,
            ) = self._extract_barcode(
                result.result
            )

            timestamp = self._safe_timestamp(
                result.timestamp
            )

            state.last_timestamp = timestamp

            state.metadata[
                "barcode_task_id"
            ] = result.task_id

            state.metadata[
                "barcode_metadata"
            ] = (
                result.metadata
                if isinstance(
                    result.metadata,
                    dict,
                )
                else {}
            )

            if isinstance(
                result.metadata,
                dict,
            ):

                if "frame_id" in result.metadata:

                    state.metadata[
                        "barcode_frame_id"
                    ] = result.metadata[
                        "frame_id"
                    ]

                if "source" in result.metadata:

                    state.metadata[
                        "barcode_source"
                    ] = result.metadata[
                        "source"
                    ]

            self.barcode_results += 1
            self.total_results += 1

            completed_now = (
                self._check_complete(
                    state
                )
            )

        self._publish(result)

        if completed_now:

            self._publish(state)

        return state

    # ==================================================================
    # COMPLETE + VALIDATE
    # ==================================================================

    def _check_complete(
        self,
        state: TrackResultState,
    ) -> bool:

        if state.completed:
            return False

        # A failed worker result still counts as received.
        if not (
            state.ocr_received
            and state.barcode_received
        ):
            return False

        (
            ocr_text,
            ocr_confidence,
            _,
        ) = self._extract_ocr(
            state.ocr_result
        )

        (
            barcode_value,
            barcode_confidence,
            _,
        ) = self._extract_barcode(
            state.barcode_result
        )

        state.validation_result = (
            self.validator.validate(
                ocr_text=ocr_text,
                ocr_confidence=ocr_confidence,
                barcode_data=barcode_value,
                barcode_confidence=barcode_confidence,
            )
        )

        state.completed = True

        self.completed_tracks += 1
        self.validated_tracks += 1

        if (
            state.validation_result
            and state.validation_result.valid
        ):

            self.valid_tracks += 1

        else:

            self.invalid_tracks += 1

        if state.validation_result:

            state.metadata[
                "validation"
            ] = (
                state.validation_result.to_dict()
            )

        return True

    # ==================================================================
    # CALLBACK PUBLISH
    # ==================================================================

    def _publish(
        self,
        result: Any,
    ) -> None:

        with self._lock:

            callbacks = list(
                self._callbacks
            )

        for callback in callbacks:

            try:

                callback(result)

            except Exception as exc:

                with self._lock:

                    self.callback_errors += 1

                print(
                    "[RESULT MANAGER] "
                    f"Callback error: {exc}"
                )

    # ==================================================================
    # GET
    # ==================================================================

    def get(
        self,
        camera_id: str,
        track_id: int,
    ) -> Optional[TrackResultState]:

        key = (
            str(camera_id),
            int(track_id),
        )

        with self._lock:

            return self._states.get(
                key
            )

    # ==================================================================
    # GET ALL
    # ==================================================================

    def get_all(
        self,
    ) -> list[TrackResultState]:

        with self._lock:

            return list(
                self._states.values()
            )

    # ==================================================================
    # REMOVE
    # ==================================================================

    def remove(
        self,
        camera_id: str,
        track_id: int,
    ) -> None:

        key = (
            str(camera_id),
            int(track_id),
        )

        with self._lock:

            self._states.pop(
                key,
                None,
            )

    # ==================================================================
    # CLEAR
    # ==================================================================

    def clear(self) -> None:

        with self._lock:

            self._states.clear()

    # ==================================================================
    # STATUS
    # ==================================================================

    def status(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                "states": len(
                    self._states
                ),
                "ocr_results": (
                    self.ocr_results
                ),
                "barcode_results": (
                    self.barcode_results
                ),
                "total_results": (
                    self.total_results
                ),
                "completed_tracks": (
                    self.completed_tracks
                ),
                "validated_tracks": (
                    self.validated_tracks
                ),
                "valid_tracks": (
                    self.valid_tracks
                ),
                "invalid_tracks": (
                    self.invalid_tracks
                ),
                "callback_errors": (
                    self.callback_errors
                ),
            }

    # ==================================================================
    # SAFE TIMESTAMP
    # ==================================================================

    @staticmethod
    def _safe_timestamp(
        timestamp: Any,
    ) -> float:

        try:

            return float(timestamp)

        except (
            TypeError,
            ValueError,
        ):

            return time.time()


# ======================================================================
# TEST
# ======================================================================


if __name__ == "__main__":

    print("=" * 72)
    print("OCR_BHS RESULT MANAGER TEST")
    print("=" * 72)

    manager = ResultManager()

    published = []

    manager.add_callback(
        published.append
    )

    now = time.time()

    ocr = OCRTaskResult(
        task_id="ocr-test",
        camera_id="camera_1",
        track_id=17,
        result={
            "success": True,
            "text": (
                "BAG DEL BLR FLIGHT "
                "5756567890AER"
            ),
            "confidence": 0.92,
        },
        timestamp=now,
        metadata={
            "frame_id": 100,
            "source": "test",
        },
    )

    barcode = BarcodeTaskResult(
        task_id="barcode-test",
        camera_id="camera_1",
        track_id=17,
        result={
            "success": True,
            "value": "5756567890AER",
            "confidence": 1.0,
        },
        timestamp=now + 0.1,
        metadata={
            "frame_id": 100,
            "source": "test",
        },
    )

    # --------------------------------------------------------------
    # OCR first
    # --------------------------------------------------------------

    state = manager.process_result(
        ocr
    )

    assert state is not None
    assert state.ocr_received
    assert not state.completed

    # --------------------------------------------------------------
    # Barcode second
    # --------------------------------------------------------------

    state = manager.process_result(
        barcode
    )

    assert state is not None
    assert state.barcode_received
    assert state.completed

    assert (
        state.validation_result
        is not None
    )

    assert (
        state.validation_result.valid
    )

    assert (
        state.validation_result
        .identifier_match
    )

    assert len(published) == 3

    print(
        "Camera     :",
        state.camera_id,
    )

    print(
        "Track      :",
        state.track_id,
    )

    print(
        "OCR        :",
        state.ocr_result,
    )

    print(
        "Barcode    :",
        state.barcode_result,
    )

    print(
        "Completed  :",
        state.completed,
    )

    print(
        "Validation :",
        state.validation_result.to_dict(),
    )

    print()

    print("[PASS] OCR result")
    print("[PASS] Barcode result")
    print("[PASS] Track aggregation")
    print("[PASS] Validator integration")
    print("[PASS] Callback publication")

    print()
    print("=" * 72)
    print("RESULT MANAGER TEST PASSED")
    print("=" * 72)