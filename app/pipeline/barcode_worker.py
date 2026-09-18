from __future__ import annotations

import threading
import time
from typing import Any

from app.barcode.barcode_region_processor import BarcodeRegionProcessor
from app.pipeline.processing_queue import (
    BarcodeTaskResult,
    ProcessingQueue,
)


class BarcodeWorker:
    """
    Background barcode worker.

    The live camera / YOLO / ByteTrack pipeline is never blocked
    by barcode processing.
    """

    def __init__(
        self,
        worker_id: int,
        processor: BarcodeRegionProcessor,
        processing_queue: ProcessingQueue,
    ) -> None:

        self.worker_id = int(worker_id)

        self.processor = processor

        self.processing_queue = (
            processing_queue
        )

        self._stop_event = (
            threading.Event()
        )

        self._thread: (
            threading.Thread | None
        ) = None

        self.processed = 0
        self.successful = 0
        self.failed = 0

        self.total_processing_ms = 0.0

    # ============================================================
    # START
    # ============================================================

    def start(self) -> None:

        if (
            self._thread is not None
            and self._thread.is_alive()
        ):
            return

        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._run,
            name=(
                f"BarcodeWorker-"
                f"{self.worker_id}"
            ),
            daemon=True,
        )

        self._thread.start()

    # ============================================================
    # STOP
    # ============================================================

    def stop(self) -> None:

        self._stop_event.set()

    # ============================================================
    # JOIN
    # ============================================================

    def join(
        self,
        timeout: float = 2.0,
    ) -> None:

        if self._thread is not None:

            self._thread.join(
                timeout=timeout
            )

    # ============================================================
    # RUN
    # ============================================================

    def _run(self) -> None:

        while not self._stop_event.is_set():

            task = (
                self.processing_queue
                .get_barcode(
                    timeout=0.5
                )
            )

            if task is None:
                continue

            start = time.perf_counter()

            try:

                result = (
                    self.processor.read(
                        task.frame
                    )
                )

                elapsed_ms = (
                    time.perf_counter()
                    - start
                ) * 1000.0

                self.processed += 1

                self.total_processing_ms += (
                    elapsed_ms
                )

                success = bool(
                    getattr(
                        result,
                        "success",
                        False,
                    )
                )

                if success:

                    self.successful += 1

                else:

                    self.failed += 1

                task_result = (
                    BarcodeTaskResult(
                        task_id=task.task_id,
                        camera_id=(
                            task.camera_id
                        ),
                        track_id=(
                            task.track_id
                        ),
                        result=result,
                        timestamp=(
                            task.timestamp
                        ),
                        metadata={
                            **(
                                task.metadata
                                or {}
                            ),
                            "worker_id": (
                                self.worker_id
                            ),
                            "processing_ms": (
                                elapsed_ms
                            ),
                            "success": (
                                success
                            ),
                        },
                    )
                )

            except Exception as exc:

                elapsed_ms = (
                    time.perf_counter()
                    - start
                ) * 1000.0

                self.processed += 1
                self.failed += 1

                self.total_processing_ms += (
                    elapsed_ms
                )

                task_result = (
                    BarcodeTaskResult(
                        task_id=task.task_id,
                        camera_id=(
                            task.camera_id
                        ),
                        track_id=(
                            task.track_id
                        ),
                        result=None,
                        timestamp=(
                            task.timestamp
                        ),
                        metadata={
                            **(
                                task.metadata
                                or {}
                            ),
                            "worker_id": (
                                self.worker_id
                            ),
                            "processing_ms": (
                                elapsed_ms
                            ),
                            "success": False,
                            "error": str(
                                exc
                            ),
                        },
                    )
                )

            finally:

                # Submit the result BEFORE marking the
                # barcode task complete. This prevents a
                # completion waiter from seeing zero pending
                # work while the result is not yet available.
                try:

                    self.processing_queue.submit_result(
                        task_result
                    )

                finally:

                    self.processing_queue.barcode_done()

    # ============================================================
    # STATUS
    # ============================================================

    def status(
        self,
    ) -> dict[str, Any]:

        average_ms = (
            self.total_processing_ms
            / self.processed
            if self.processed
            else 0.0
        )

        return {

            "worker_id": (
                self.worker_id
            ),

            "running": (
                self._thread is not None
                and self._thread.is_alive()
            ),

            "processed": (
                self.processed
            ),

            "successful": (
                self.successful
            ),

            "failed": (
                self.failed
            ),

            "average_processing_ms": (
                average_ms
            ),
        }