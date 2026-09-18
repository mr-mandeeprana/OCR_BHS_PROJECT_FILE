"""
OCR_BHS - Result Persistence Bridge

Connects:

    OCR/Barcode Worker Results
                ↓
          ResultManager
                ↓
       ResultPersistence
                ↓
         DatabaseWorker
                ↓
            SQL Server

IMPORTANT
---------
This module NEVER performs a synchronous database insert.

It also NEVER calls ResultManager.process_result().

ResultManager is the single source of truth for processing results.
This class only persists results after ResultManager has processed them.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from app.pipeline.processing_queue import (
    OCRTaskResult,
    BarcodeTaskResult,
)

from app.pipeline.result_manager import (
    ResultManager,
    TrackResultState,
)

from app.database.database_manager import (
    DatabaseWorker,
)


class ResultPersistence:
    """
    Connect ResultManager with asynchronous SQL Server persistence.

    One instance should be created for one application runtime.
    """

    def __init__(
        self,
        result_manager: ResultManager,
        database_worker: Optional[DatabaseWorker] = None,
    ) -> None:

        self.result_manager = result_manager
        self.database_worker = database_worker

        self.enabled = database_worker is not None

        # Prevent duplicate final-record submissions.
        self._final_submitted: set[tuple[str, int]] = set()

        self.ocr_submitted = 0
        self.barcode_submitted = 0
        self.final_submitted = 0

        self.ocr_dropped = 0
        self.barcode_dropped = 0
        self.final_dropped = 0

        self.errors = 0

    # ==============================================================
    # PUBLIC CALLBACK
    # ==============================================================

    def handle_result(
        self,
        result: Any,
    ) -> None:
        """
        Handle a result AFTER ResultManager has processed it.

        IMPORTANT:
        Do NOT call result_manager.process_result() here.

        ResultManager invokes this callback after its own processing.
        """

        if result is None:
            return

        try:

            if isinstance(result, OCRTaskResult):

                self._persist_ocr(result)

                # The ResultManager callback may provide the updated
                # state separately through another callback depending
                # on the current implementation.

            elif isinstance(result, BarcodeTaskResult):

                self._persist_barcode(result)

            elif isinstance(result, TrackResultState):

                if result.completed:
                    self._persist_final(result)

        except Exception as exc:

            self.errors += 1

            print(
                "[PERSISTENCE] "
                f"Result handling error: {exc}"
            )

    # ==============================================================
    # OCR
    # ==============================================================

    def _persist_ocr(
        self,
        result: OCRTaskResult,
    ) -> None:

        if not self.enabled:
            return

        payload = self._ocr_payload(result)

        accepted = self.database_worker.submit(
            "ocr",
            payload,
            block=False,
        )

        if accepted:
            self.ocr_submitted += 1
        else:
            self.ocr_dropped += 1

    # ==============================================================
    # BARCODE
    # ==============================================================

    def _persist_barcode(
        self,
        result: BarcodeTaskResult,
    ) -> None:

        if not self.enabled:
            return

        payload = self._barcode_payload(result)

        accepted = self.database_worker.submit(
            "barcode",
            payload,
            block=False,
        )

        if accepted:
            self.barcode_submitted += 1
        else:
            self.barcode_dropped += 1

    # ==============================================================
    # FINAL
    # ==============================================================

    def _persist_final(
        self,
        state: TrackResultState,
    ) -> None:

        if not self.enabled:
            return

        key = (
            str(state.camera_id),
            int(state.track_id),
        )

        if key in self._final_submitted:
            return

        payload = self._final_payload(state)

        accepted = self.database_worker.submit(
            "final",
            payload,
            block=False,
        )

        if accepted:

            self._final_submitted.add(key)
            self.final_submitted += 1

        else:

            self.final_dropped += 1

    # ==============================================================
    # OCR PAYLOAD
    # ==============================================================

    @staticmethod
    def _ocr_payload(
        result: OCRTaskResult,
    ) -> dict[str, Any]:

        data = ResultPersistence._to_dict(result.result)

        metadata = (
            result.metadata
            if isinstance(result.metadata, dict)
            else {}
        )

        return {
            # Exact field names consumed by DatabaseManager.insert_ocr_reading().
            "event_id": f"ocr_{result.task_id}",
            "task_id": result.task_id,
            "camera_id": result.camera_id,
            "track_id": result.track_id,
            "frame_id": metadata.get("frame_id"),
            "captured_at": result.timestamp,
            "text": data.get("text") or data.get("normalized_text") or "",
            "normalized_text": data.get("normalized_text") or data.get("text") or "",
            "confidence": ResultPersistence._float(data.get("confidence", 0.0)),
            "engine": data.get("engine", ""),
            "rotation": data.get("rotation", 0),
            "success": bool(data.get("success", bool(data.get("text")))),
            "elapsed_ms": data.get("elapsed_ms", metadata.get("elapsed_ms")),
            "error": data.get("error"),
            "details": data.get("details", metadata),
        }

    # ==============================================================
    # BARCODE PAYLOAD
    # ==============================================================

    @staticmethod
    def _barcode_payload(
        result: BarcodeTaskResult,
    ) -> dict[str, Any]:

        data = ResultPersistence._to_dict(result.result)

        metadata = (
            result.metadata
            if isinstance(result.metadata, dict)
            else {}
        )

        barcode_value = (
            data.get("data")
            or data.get("value")
            or data.get("barcode_data")
            or ""
        )

        barcode_type = (
            data.get("barcode_type")
            or data.get("format")
            or data.get("type")
            or ""
        )

        return {
            # Exact field names consumed by DatabaseManager.insert_barcode_reading().
            "event_id": f"barcode_{result.task_id}",
            "task_id": result.task_id,
            "camera_id": result.camera_id,
            "track_id": result.track_id,
            "frame_id": metadata.get("frame_id"),
            "captured_at": result.timestamp,
            "barcode_value": str(barcode_value),
            "barcode_type": str(barcode_type),
            "confidence": ResultPersistence._float(data.get("confidence", 0.0)),
            "rotation": data.get("rotation", 0),
            "variant": data.get("variant"),
            "success": bool(data.get("success", bool(barcode_value))),
            "elapsed_ms": data.get("elapsed_ms", metadata.get("elapsed_ms")),
            "error": data.get("error"),
            "details": data,
        }

    # ==============================================================
    # FINAL PAYLOAD
    # ==============================================================

    @staticmethod
    def _final_payload(
        state: TrackResultState,
    ) -> dict[str, Any]:

        ocr = ResultPersistence._to_dict(
            state.ocr_result
        )

        barcode = ResultPersistence._best_barcode(
            state.barcode_result
        )

        validation = getattr(
            state,
            "validation_result",
            None,
        )

        validation_dict = ResultPersistence._to_dict(
            validation
        )

        metadata = (
            state.metadata
            if isinstance(state.metadata, dict)
            else {}
        )

        ocr_text = (
            ocr.get("text")
            or ocr.get("normalized_text")
            or ""
        )

        barcode_value = (
            barcode.get("data")
            or barcode.get("value")
            or barcode.get("barcode_data")
            or ""
        )

        barcode_type = (
            barcode.get("barcode_type")
            or barcode.get("format")
            or barcode.get("type")
            or ""
        )

        valid = validation_dict.get(
            "valid",
            False,
        )

        confidence = validation_dict.get(
            "confidence",
            0.0,
        )

        reason = validation_dict.get(
            "reason",
            "",
        )

        identifier_match = validation_dict.get(
            "identifier_match",
            False,
        )

        frame_id = metadata.get(
            "frame_id"
        )

        task_id = (
            metadata.get("ocr_task_id")
            or metadata.get("barcode_task_id")
        )

        return {
            "event_id": (
                f"{state.camera_id}_"
                f"{state.track_id}_"
                f"{int(time.time() * 1000)}"
            ),

            "camera_id": state.camera_id,

            "track_id": state.track_id,

            "task_id": task_id,

            "frame_id": frame_id,

            "captured_at": state.last_timestamp,

            "ocr_text": str(ocr_text),

            "ocr_confidence": (
                ResultPersistence._float(
                    ocr.get("confidence", 0.0)
                )
            ),

            "ocr_normalized_text": str(
                ocr.get("normalized_text") or ocr_text
            ),

            "ocr_engine": str(
                ocr.get("engine") or ""
            ),

            "ocr_success": bool(
                getattr(state, "ocr_success", False)
            ),

            "barcode_value": str(
                barcode_value
            ),

            "barcode_type": str(
                barcode_type
            ),

            "barcode_confidence": (
                ResultPersistence._float(
                    barcode.get("confidence", 0.0)
                )
            ),

            "barcode_success": bool(
                getattr(state, "barcode_success", False)
            ),

            "validation_valid": bool(valid),

            "validation_confidence": (
                ResultPersistence._float(
                    confidence
                )
            ),

            "validation_reason": str(
                reason
            ),

            "identifier_match": bool(
                identifier_match
            ),

            "status": (
                "VALID"
                if valid
                else "INVALID"
            ),

            "processing_ms": (
                metadata.get(
                    "processing_ms",
                    0.0,
                )
            ),

            "raw_json": ResultPersistence._json(
                {
                    "ocr": ocr,
                    "barcode": barcode,
                    "validation": validation_dict,
                    "metadata": metadata,
                }
            ),
        }

    # ==============================================================
    # HELPERS
    # ==============================================================

    @staticmethod
    def _to_dict(
        value: Any,
    ) -> dict[str, Any]:

        if value is None:
            return {}

        if isinstance(value, dict):
            return dict(value)

        try:

            if hasattr(value, "to_dict"):

                output = value.to_dict()

                if isinstance(output, dict):
                    return output

        except Exception:
            pass

        output = {}

        for name in (
            "success",
            "text",
            "normalized_text",
            "confidence",
            "data",
            "value",
            "barcode_data",
            "barcode_type",
            "format",
            "type",
            "valid",
            "reason",
            "identifier_match",
        ):

            if hasattr(value, name):

                output[name] = getattr(
                    value,
                    name,
                )

        return output

    @staticmethod
    def _best_barcode(
        value: Any,
    ) -> dict[str, Any]:

        if isinstance(value, list):

            if not value:
                return {}

            converted = [
                ResultPersistence._to_dict(item)
                for item in value
            ]

            return max(
                converted,
                key=lambda item:
                    ResultPersistence._float(
                        item.get(
                            "confidence",
                            0.0,
                        )
                    ),
            )

        return ResultPersistence._to_dict(
            value
        )

    @staticmethod
    def _float(
        value: Any,
    ) -> float:

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    @staticmethod
    def _json(
        value: Any,
    ) -> str:

        try:

            return json.dumps(
                value,
                default=str,
                ensure_ascii=False,
            )

        except Exception:

            return json.dumps(
                str(value)
            )

    # ==============================================================
    # STATUS
    # ==============================================================

    def status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,

            "ocr_submitted": self.ocr_submitted,

            "barcode_submitted": (
                self.barcode_submitted
            ),

            "final_submitted": (
                self.final_submitted
            ),

            "ocr_dropped": (
                self.ocr_dropped
            ),

            "barcode_dropped": (
                self.barcode_dropped
            ),

            "final_dropped": (
                self.final_dropped
            ),

            "errors": self.errors,
        }