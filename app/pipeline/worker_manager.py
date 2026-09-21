from __future__ import annotations

import threading
import time
from typing import Any, Optional

from app.barcode.barcode_engine import BarcodeEngine
from app.barcode.barcode_region_processor import BarcodeRegionProcessor
from app.config.config_loader import Config
from app.ocr.ocr_engine import OCREngine
from app.pipeline.processing_queue import (
    BarcodeTask,
    BarcodeTaskResult,
    OCRTask,
    OCRTaskResult,
    ProcessingQueue,
)


class WorkerManager:
    """
    Central asynchronous OCR + Barcode worker manager.

    Architecture:

        Camera
           |
           v
        Detection
           |
           v
        ByteTrack
           |
           +----------------------+
           |                      |
           v                      v
        OCR Queue             Barcode Queue
           |                      |
           v                      v
       OCR Workers           Barcode Workers
           |                      |
           +----------+-----------+
                      |
                      v
                Result Queue

    IMPORTANT:
        The live RTSP / YOLO / tracking thread NEVER waits
        for OCR or barcode processing.

    Queue submission is non-blocking.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        processing_queue: Optional[ProcessingQueue] = None,
    ) -> None:

        self.config = config or Config()

        pipeline_cfg = (
            self.config.get("pipeline", {})
            or {}
        )

        # ---------------------------------------------------------
        # Processing queue
        # ---------------------------------------------------------

        if processing_queue is not None:

            self.queue = processing_queue

        else:

            queue_size = self._get_queue_size()

            self.queue = ProcessingQueue(
                max_size=queue_size
            )

        # ---------------------------------------------------------
        # OCR configuration
        # ---------------------------------------------------------

        ocr_cfg = (
            self.config.get("ocr", {})
            or {}
        )

        self.ocr_enabled = bool(
            ocr_cfg.get(
                "enabled",
                True,
            )
        )

        ocr_worker_cfg = (
            ocr_cfg.get("worker", {})
            or {}
        )

        self.ocr_worker_enabled = bool(
            ocr_worker_cfg.get(
                "enabled",
                True,
            )
        )

        self.ocr_worker_count = max(
            1,
            int(
                ocr_worker_cfg.get(
                    "count",
                    1,
                )
            ),
        )

        # ---------------------------------------------------------
        # Barcode configuration
        # ---------------------------------------------------------

        barcode_cfg = (
            self.config.get("barcode", {})
            or {}
        )

        self.barcode_enabled = bool(
            barcode_cfg.get(
                "enabled",
                True,
            )
        )

        barcode_worker_cfg = (
            barcode_cfg.get("worker", {})
            or {}
        )

        self.barcode_worker_enabled = bool(
            barcode_worker_cfg.get(
                "enabled",
                True,
            )
        )

        self.barcode_worker_count = max(
            1,
            int(
                barcode_worker_cfg.get(
                    "count",
                    1,
                )
            ),
        )

        # ---------------------------------------------------------
        # Worker runtime
        # ---------------------------------------------------------

        self._running = False

        self._stop_event = threading.Event()

        self._threads: list[
            threading.Thread
        ] = []

        # ---------------------------------------------------------
        # Worker counters
        # ---------------------------------------------------------

        self._lock = threading.Lock()

        self._ocr_started = 0
        self._ocr_completed = 0
        self._ocr_failed = 0

        self._barcode_started = 0
        self._barcode_completed = 0
        self._barcode_failed = 0

        self._results_failed = 0

        # ---------------------------------------------------------
        # Engine initialization
        #
        # Barcode engine is lightweight.
        #
        # OCR PaddleOCR is heavy, so we intentionally create
        # one OCR engine per worker thread instead of sharing
        # a PaddleOCR object between threads.
        # ---------------------------------------------------------

        # NOTE: BarcodeRegionProcessor internally owns/wraps a
        # BarcodeEngine (direct full-image decode first, then a
        # region-based fallback when that fails), so it is a drop-in,
        # strictly more capable replacement for using BarcodeEngine
        # directly here.
        self._barcode_engine: Optional[
            BarcodeRegionProcessor
        ] = None

        if (
            self.barcode_enabled
            and self.barcode_worker_enabled
        ):

            self._barcode_engine = BarcodeRegionProcessor(
                self.config
            )

        # OCR engines are created inside OCR worker threads.
        self._ocr_engines: dict[
            str,
            OCREngine,
        ] = {}

        # ---------------------------------------------------------
        # Pipeline configuration
        # ---------------------------------------------------------

        self.pipeline_enabled = bool(
            pipeline_cfg.get(
                "enabled",
                True,
            )
        )

    # =================================================================
    # CONFIG
    # =================================================================

    def _get_queue_size(self) -> int:
        """
        Resolve queue size.

        Priority:

            pipeline.processing_queue_size
            OCR worker queue_size
            Barcode worker queue_size
            50
        """

        pipeline_cfg = (
            self.config.get(
                "pipeline",
                {}
            )
            or {}
        )

        value = pipeline_cfg.get(
            "processing_queue_size",
            None,
        )

        if value is not None:

            try:
                return max(
                    1,
                    int(value),
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

        ocr_cfg = (
            self.config.get(
                "ocr",
                {}
            )
            or {}
        )

        ocr_worker = (
            ocr_cfg.get(
                "worker",
                {}
            )
            or {}
        )

        value = ocr_worker.get(
            "queue_size",
            None,
        )

        if value is not None:

            try:
                return max(
                    1,
                    int(value),
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

        barcode_cfg = (
            self.config.get(
                "barcode",
                {}
            )
            or {}
        )

        barcode_worker = (
            barcode_cfg.get(
                "worker",
                {}
            )
            or {}
        )

        value = barcode_worker.get(
            "queue_size",
            None,
        )

        if value is not None:

            try:
                return max(
                    1,
                    int(value),
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

        return 50

    # =================================================================
    # START
    # =================================================================

    def start(self) -> None:
        """
        Start all configured OCR and barcode workers.
        """

        if self._running:

            print(
                "[WORKER] Worker manager already running."
            )

            return

        if not self.pipeline_enabled:

            print(
                "[WORKER] Pipeline disabled in configuration."
            )

            return

        self._stop_event.clear()

        self.queue.reopen()

        self._threads.clear()

        # ---------------------------------------------------------
        # OCR workers
        # ---------------------------------------------------------

        if (
            self.ocr_enabled
            and self.ocr_worker_enabled
        ):

            for index in range(
                self.ocr_worker_count
            ):

                thread = threading.Thread(
                    target=self._ocr_worker_loop,
                    args=(index,),
                    name=f"OCRWorker-{index + 1}",
                    daemon=True,
                )

                self._threads.append(
                    thread
                )

                thread.start()

        # ---------------------------------------------------------
        # Barcode workers
        # ---------------------------------------------------------

        if (
            self.barcode_enabled
            and self.barcode_worker_enabled
        ):

            for index in range(
                self.barcode_worker_count
            ):

                thread = threading.Thread(
                    target=self._barcode_worker_loop,
                    args=(index,),
                    name=f"BarcodeWorker-{index + 1}",
                    daemon=True,
                )

                self._threads.append(
                    thread
                )

                thread.start()

        self._running = True

        print()
        print("=" * 70)
        print("OCR_BHS - WORKER MANAGER STARTED")
        print("=" * 70)

        print(
            f"OCR enabled             : "
            f"{self.ocr_enabled}"
        )

        print(
            f"OCR workers             : "
            f"{self.ocr_worker_count}"
        )

        print(
            f"Barcode enabled         : "
            f"{self.barcode_enabled}"
        )

        print(
            f"Barcode workers         : "
            f"{self.barcode_worker_count}"
        )

        print(
            f"Queue size              : "
            f"{self.queue.max_size}"
        )

        print(
            f"Total worker threads    : "
            f"{len(self._threads)}"
        )

        print("=" * 70)
        print()

    # =================================================================
    # OCR WORKER
    # =================================================================

    def _ocr_worker_loop(
        self,
        worker_index: int,
    ) -> None:
        """
        OCR worker thread.

        Each OCR worker owns its own OCREngine instance.
        """

        worker_name = (
            f"OCRWorker-{worker_index + 1}"
        )

        print(
            f"[{worker_name}] Starting."
        )

        # ---------------------------------------------------------
        # Create OCR engine inside this worker.
        # ---------------------------------------------------------

        try:

            engine = OCREngine(
                config=self.config
            )

            self._ocr_engines[
                worker_name
            ] = engine

            print(
                f"[{worker_name}] OCR engine initialized."
            )

        except Exception as exc:

            print(
                f"[{worker_name}] "
                f"OCR engine initialization failed: "
                f"{type(exc).__name__}: {exc}"
            )

            with self._lock:
                self._ocr_failed += 1

            return

        # ---------------------------------------------------------
        # Main worker loop
        # ---------------------------------------------------------

        while not self._stop_event.is_set():

            task = self.queue.get_ocr(
                timeout=0.2
            )

            if task is None:
                continue

            with self._lock:
                self._ocr_started += 1

            try:

                self._process_ocr_task(
                    task=task,
                    engine=engine,
                    worker_name=worker_name,
                )

                with self._lock:
                    self._ocr_completed += 1

            except Exception as exc:

                with self._lock:
                    self._ocr_failed += 1

                print(
                    f"[{worker_name}] "
                    f"OCR task failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                self._submit_ocr_error_result(
                    task,
                    exc,
                )

            finally:

                self.queue.ocr_done()

        print(
            f"[{worker_name}] Stopped."
        )

    # =================================================================
    # OCR TASK
    # =================================================================

    def _process_ocr_task(
        self,
        task: OCRTask,
        engine: OCREngine,
        worker_name: str,
    ) -> None:
        """
        Process one OCR task and submit result.
        """

        start = time.perf_counter()

        result = engine.read(
            task.frame
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        metadata = dict(
            task.metadata
        )

        metadata.update(
            {
                "worker": worker_name,
                "processing_elapsed_ms": round(
                    elapsed_ms,
                    2,
                ),
            }
        )

        task_result = OCRTaskResult(
            task_id=task.task_id,
            camera_id=task.camera_id,
            track_id=task.track_id,
            result=result,
            timestamp=task.timestamp,
            metadata=metadata,
        )

        accepted = self.queue.submit_result(
            task_result
        )

        if not accepted:

            with self._lock:
                self._results_failed += 1

            print(
                f"[{worker_name}] "
                f"Result queue full/closed for "
                f"OCR task {task.task_id}"
            )

    # =================================================================
    # OCR ERROR
    # =================================================================

    def _submit_ocr_error_result(
        self,
        task: OCRTask,
        exc: Exception,
    ) -> None:
        """
        Submit a structured OCR error result.
        """

        metadata = dict(
            task.metadata
        )

        metadata.update(
            {
                "worker_error": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )

        result = OCRTaskResult(
            task_id=task.task_id,
            camera_id=task.camera_id,
            track_id=task.track_id,
            result=None,
            timestamp=task.timestamp,
            metadata=metadata,
        )

        self.queue.submit_result(
            result
        )

    # =================================================================
    # BARCODE WORKER
    # =================================================================

    def _barcode_worker_loop(
        self,
        worker_index: int,
    ) -> None:
        """
        Barcode worker thread.
        """

        worker_name = (
            f"BarcodeWorker-{worker_index + 1}"
        )

        print(
            f"[{worker_name}] Starting."
        )

        # ---------------------------------------------------------
        # Each barcode worker gets its own engine.
        #
        # Use BarcodeRegionProcessor rather than a bare BarcodeEngine.
        # BarcodeRegionProcessor already tries a direct full-image
        # decode first (identical to BarcodeEngine.read()) and only
        # falls back to locating/decoding barcode sub-regions when
        # that direct decode fails - exactly the region-based fallback
        # the barcode region processor was built for, and exactly what
        # the standalone diagnostic script exercises alongside
        # BarcodeEngine. Previously this worker only ever got the
        # direct-decode behaviour and the region fallback was unused
        # dead code (barcode_region_processor.py was never imported
        # from anywhere in the running pipeline).
        #
        # This is otherwise lightweight and avoids sharing mutable
        # runtime statistics between threads.
        # ---------------------------------------------------------

        try:

            engine = BarcodeRegionProcessor(
                self.config
            )

        except Exception as exc:

            print(
                f"[{worker_name}] "
                f"Barcode engine initialization failed: "
                f"{type(exc).__name__}: {exc}"
            )

            with self._lock:
                self._barcode_failed += 1

            return

        # ---------------------------------------------------------
        # Main loop
        # ---------------------------------------------------------

        while not self._stop_event.is_set():

            task = self.queue.get_barcode(
                timeout=0.2
            )

            if task is None:
                continue

            with self._lock:
                self._barcode_started += 1

            try:

                self._process_barcode_task(
                    task=task,
                    engine=engine,
                    worker_name=worker_name,
                )

                with self._lock:
                    self._barcode_completed += 1

            except Exception as exc:

                with self._lock:
                    self._barcode_failed += 1

                print(
                    f"[{worker_name}] "
                    f"Barcode task failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                self._submit_barcode_error_result(
                    task,
                    exc,
                )

            finally:

                self.queue.barcode_done()

        print(
            f"[{worker_name}] Stopped."
        )

    # =================================================================
    # BARCODE TASK
    # =================================================================

    def _process_barcode_task(
        self,
        task: BarcodeTask,
        engine: BarcodeRegionProcessor,
        worker_name: str,
    ) -> None:
        """
        Process one barcode task and submit result.
        """

        start = time.perf_counter()

        # BarcodeRegionProcessor.process() returns the same
        # list[BarcodeResult] shape as BarcodeEngine.read() (it tries
        # the direct decode first, internally), so this is a drop-in
        # call-site swap - no downstream change needed in
        # ResultManager/_best_barcode(), which already reads results
        # generically via getattr(item, "value"/"data"/"barcode").
        results = engine.process(
            task.frame
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        metadata = dict(
            task.metadata
        )

        metadata.update(
            {
                "worker": worker_name,
                "processing_elapsed_ms": round(
                    elapsed_ms,
                    2,
                ),
                "result_count": len(results),
            }
        )

        task_result = BarcodeTaskResult(
            task_id=task.task_id,
            camera_id=task.camera_id,
            track_id=task.track_id,
            result=results,
            timestamp=task.timestamp,
            metadata=metadata,
        )

        accepted = self.queue.submit_result(
            task_result
        )

        if not accepted:

            with self._lock:
                self._results_failed += 1

            print(
                f"[{worker_name}] "
                f"Result queue full/closed for "
                f"Barcode task {task.task_id}"
            )

    # =================================================================
    # BARCODE ERROR
    # =================================================================

    def _submit_barcode_error_result(
        self,
        task: BarcodeTask,
        exc: Exception,
    ) -> None:
        """
        Submit a structured barcode error result.
        """

        metadata = dict(
            task.metadata
        )

        metadata.update(
            {
                "worker_error": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )

        result = BarcodeTaskResult(
            task_id=task.task_id,
            camera_id=task.camera_id,
            track_id=task.track_id,
            result=[],
            timestamp=task.timestamp,
            metadata=metadata,
        )

        self.queue.submit_result(
            result
        )

    # =================================================================
    # SUBMIT OCR
    # =================================================================

    def submit_ocr(
        self,
        task_id: str,
        camera_id: str,
        track_id: int,
        frame: Any,
        timestamp: float,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> bool:
        """
        Submit OCR work without blocking.

        Returns:
            True  = accepted
            False = dropped
        """

        task = OCRTask(
            task_id=task_id,
            camera_id=camera_id,
            track_id=track_id,
            frame=frame,
            timestamp=timestamp,
            metadata=metadata or {},
        )

        return self.queue.submit_ocr(
            task
        )

    # =================================================================
    # SUBMIT BARCODE
    # =================================================================

    def submit_barcode(
        self,
        task_id: str,
        camera_id: str,
        track_id: int,
        frame: Any,
        timestamp: float,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> bool:
        """
        Submit barcode work without blocking.

        Returns:
            True  = accepted
            False = dropped
        """

        task = BarcodeTask(
            task_id=task_id,
            camera_id=camera_id,
            track_id=track_id,
            frame=frame,
            timestamp=timestamp,
            metadata=metadata or {},
        )

        return self.queue.submit_barcode(
            task
        )

    # =================================================================
    # GET RESULT
    # =================================================================

    def get_result(
        self,
        timeout: float = 0.01,
    ):
        """
        Retrieve an OCR or Barcode result.

        This is non-blocking for the live pipeline when
        timeout is kept very small.
        """

        result = self.queue.get_result(
            timeout=timeout
        )

        if result is not None:

            self.queue.result_done()

        return result

    # =================================================================
    # DRAIN RESULTS
    # =================================================================

    def drain_results(
        self,
        max_results: int = 100,
    ) -> list[
        OCRTaskResult | BarcodeTaskResult
    ]:
        """
        Retrieve currently available results.

        Never waits for processing.
        """

        results = []

        limit = max(
            1,
            int(max_results),
        )

        for _ in range(limit):

            result = self.queue.get_result(
                timeout=0.0
            )

            if result is None:
                break

            results.append(
                result
            )

            self.queue.result_done()

        return results

    # =================================================================
    # STATUS
    # =================================================================

    def status(self) -> dict[str, Any]:
        """
        Return worker + queue status.
        """

        with self._lock:

            return {
                "running": self._running,

                "pipeline_enabled": (
                    self.pipeline_enabled
                ),

                "ocr_enabled": (
                    self.ocr_enabled
                ),

                "ocr_worker_enabled": (
                    self.ocr_worker_enabled
                ),

                "ocr_worker_count": (
                    self.ocr_worker_count
                ),

                "ocr_threads_started": (
                    self._ocr_started
                ),

                "ocr_completed": (
                    self._ocr_completed
                ),

                "ocr_failed": (
                    self._ocr_failed
                ),

                "barcode_enabled": (
                    self.barcode_enabled
                ),

                "barcode_worker_enabled": (
                    self.barcode_worker_enabled
                ),

                "barcode_worker_count": (
                    self.barcode_worker_count
                ),

                "barcode_threads_started": (
                    self._barcode_started
                ),

                "barcode_completed": (
                    self._barcode_completed
                ),

                "barcode_failed": (
                    self._barcode_failed
                ),

                "results_failed": (
                    self._results_failed
                ),

                "thread_count": len(
                    self._threads
                ),

                "queue": self.queue.status(),
            }

    # =================================================================
    # WAIT FOR WORK
    # =================================================================

    def wait_for_completion(
        self,
        timeout: float = 10.0,
    ) -> bool:
        """
        Wait for pending OCR and barcode work.

        Intended for tests/shutdown only.

        NEVER use this in the live camera loop.
        """

        timeout = max(
            0.0,
            float(timeout),
        )

        start = time.monotonic()

        while True:

            status = self.queue.status()

            if (
                status["ocr_pending"] == 0
                and status["barcode_pending"] == 0
            ):

                return True

            if (
                time.monotonic() - start
                >= timeout
            ):

                return False

            time.sleep(
                0.01
            )

    # =================================================================
    # STOP
    # =================================================================

    def stop(
        self,
        wait: bool = True,
        timeout: float = 10.0,
        clear_pending: bool = False,
    ) -> None:
        """
        Stop all workers gracefully.

        Parameters
        ----------
        wait:
            Wait for worker threads.

        timeout:
            Maximum shutdown wait.

        clear_pending:
            Remove queued work before stopping.
        """

        if not self._running:

            return

        print()
        print(
            "[WORKER] Stopping worker manager..."
        )

        # ---------------------------------------------------------
        # Optionally remove pending tasks.
        # ---------------------------------------------------------

        if clear_pending:

            cleared = self.queue.clear()

            print(
                f"[WORKER] Cleared pending work: "
                f"{cleared}"
            )

        # ---------------------------------------------------------
        # Signal workers.
        # ---------------------------------------------------------

        self._stop_event.set()

        # ---------------------------------------------------------
        # Close queue AFTER workers have been signaled.
        #
        # Workers already processing a task can still submit their
        # result before queue closure.
        # ---------------------------------------------------------

        if wait:

            deadline = (
                time.monotonic()
                + max(
                    0.0,
                    float(timeout),
                )
            )

            for thread in self._threads:

                remaining = (
                    deadline
                    - time.monotonic()
                )

                if remaining <= 0:
                    break

                thread.join(
                    timeout=remaining
                )

        # ---------------------------------------------------------
        # Now close queue.
        # ---------------------------------------------------------

        self.queue.close(
            clear_pending=clear_pending
        )

        self._running = False

        self._threads.clear()

        self._ocr_engines.clear()

        print(
            "[WORKER] Worker manager stopped."
        )

    # =================================================================
    # RESTART
    # =================================================================

    def restart(self) -> None:
        """
        Restart the worker manager.
        """

        self.stop(
            wait=True
        )

        self.start()


# =====================================================================
# STANDALONE TEST
# =====================================================================

def main() -> None:

    print()
    print("=" * 70)
    print("OCR_BHS - WORKER MANAGER TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------

    try:

        config = Config()

    except Exception as exc:

        print()
        print(
            f"CONFIG ERROR: "
            f"{type(exc).__name__}: {exc}"
        )

        return

    # ---------------------------------------------------------
    # Create worker manager
    # ---------------------------------------------------------

    try:

        manager = WorkerManager(
            config=config
        )

    except Exception as exc:

        print()
        print(
            f"WORKER MANAGER INITIALIZATION ERROR: "
            f"{type(exc).__name__}: {exc}"
        )

        return

    # ---------------------------------------------------------
    # Display configuration
    # ---------------------------------------------------------

    print()
    print("WORKER CONFIGURATION")
    print("-" * 70)

    print(
        f"OCR enabled          : "
        f"{manager.ocr_enabled}"
    )

    print(
        f"OCR workers          : "
        f"{manager.ocr_worker_count}"
    )

    print(
        f"Barcode enabled      : "
        f"{manager.barcode_enabled}"
    )

    print(
        f"Barcode workers      : "
        f"{manager.barcode_worker_count}"
    )

    print(
        f"Queue size           : "
        f"{manager.queue.max_size}"
    )

    # ---------------------------------------------------------
    # Start
    # ---------------------------------------------------------

    manager.start()

    print()
    print("INITIAL STATUS")
    print("-" * 70)

    print(
        manager.status()
    )

    # ---------------------------------------------------------
    # Test barcode submission with no image.
    #
    # This only verifies asynchronous queue submission.
    # Actual image decoding is tested separately.
    # ---------------------------------------------------------

    print()
    print("SUBMITTING TEST TASKS")
    print("-" * 70)

    now = time.time()

    barcode_accepted = (
        manager.submit_barcode(
            task_id="worker-test-barcode-1",
            camera_id="camera_1",
            track_id=1,
            frame=None,
            timestamp=now,
            metadata={
                "test": True,
            },
        )
    )

    print(
        "Barcode task: ",
        "ACCEPTED"
        if barcode_accepted
        else "DROPPED",
    )

    # ---------------------------------------------------------
    # OCR task with no image.
    #
    # OCREngine will safely return an invalid-image result.
    # ---------------------------------------------------------

    ocr_accepted = (
        manager.submit_ocr(
            task_id="worker-test-ocr-1",
            camera_id="camera_1",
            track_id=1,
            frame=None,
            timestamp=now,
            metadata={
                "test": True,
            },
        )
    )

    print(
        "OCR task    : ",
        "ACCEPTED"
        if ocr_accepted
        else "DROPPED",
    )

    # ---------------------------------------------------------
    # Wait for workers
    # ---------------------------------------------------------

    print()
    print(
        "Waiting for worker processing..."
    )

    manager.wait_for_completion(
        timeout=10.0
    )

    # ---------------------------------------------------------
    # Drain results
    # ---------------------------------------------------------

    print()
    print("RESULTS")
    print("-" * 70)

    results = manager.drain_results(
        max_results=20
    )

    print(
        f"Results received: "
        f"{len(results)}"
    )

    for result in results:

        print()
        print(
            f"Type      : "
            f"{type(result).__name__}"
        )

        print(
            f"Task ID   : "
            f"{result.task_id}"
        )

        print(
            f"Camera    : "
            f"{result.camera_id}"
        )

        print(
            f"Track ID  : "
            f"{result.track_id}"
        )

        print(
            f"Metadata  : "
            f"{result.metadata}"
        )

    # ---------------------------------------------------------
    # Status
    # ---------------------------------------------------------

    print()
    print("FINAL STATUS")
    print("-" * 70)

    print(
        manager.status()
    )

    # ---------------------------------------------------------
    # Stop
    # ---------------------------------------------------------

    manager.stop(
        wait=True,
        timeout=10.0,
        clear_pending=True,
    )

    print()
    print("=" * 70)
    print("WORKER MANAGER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()