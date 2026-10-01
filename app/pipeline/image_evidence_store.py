"""
OCR_BHS - Configurable asynchronous image evidence saving.

Saves (only when image_saving.enabled is true):

    captures/tags/      raw IATA tag crop selected by BestFrameSelector
    captures/enhanced/  tag after TagProcessor.normalize()
    captures/ocr/       exact image handed to the OCR worker
    captures/barcode/   exact image handed to the barcode worker

All disk writes happen on one background thread so the detection /
tracking loop is never blocked by disk I/O.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


class ImageEvidenceStore:
    """Config-driven image saver for OCR_BHS processing evidence."""

    # Raw/enhanced tag images are saved once per (camera, track, frame).
    # Bounded so a long-running process does not leak memory.
    _MAX_REMEMBERED_KEYS = 5000

    def __init__(self, config: Any = None, camera_id: str = "CAM01") -> None:
        self.camera_id = str(camera_id)

        data = getattr(config, "data", config) or {}
        saving_cfg = data.get("image_saving", {}) or {}
        paths_cfg = data.get("paths", {}) or {}

        # Relative paths resolve against the folder that holds config.yaml
        # (the project root), not against the process working directory.
        config_file = getattr(config, "path", None)
        base_dir = (
            Path(config_file).resolve().parent
            if config_file is not None
            else Path.cwd()
        )

        self.enabled = bool(saving_cfg.get("enabled", False))

        tag_cfg = saving_cfg.get("tag", {}) or {}
        ocr_cfg = saving_cfg.get("ocr", {}) or {}
        barcode_cfg = saving_cfg.get("barcode", {}) or {}
        failed_cfg = saving_cfg.get("failed", {}) or {}

        self.tag_enabled = bool(tag_cfg.get("enabled", True))
        self.save_raw = bool(tag_cfg.get("raw", True))
        self.save_enhanced = bool(tag_cfg.get("enhanced", True))
        self.ocr_enabled = bool(ocr_cfg.get("enabled", True))
        self.barcode_enabled = bool(barcode_cfg.get("enabled", True))

        # Reserved: failed-evidence saving is intentionally kept separate
        # from successful-image saving and is not driven by this class yet.
        self.failed_enabled = bool(failed_cfg.get("enabled", True))

        self.jpeg_quality = int(saving_cfg.get("jpeg_quality", 95))

        # Backlog limit: if the disk cannot keep up, drop images instead of
        # letting queued copies grow without bound in RAM.
        self.max_pending = int(saving_cfg.get("max_pending", 64))

        def _resolve(key: str, default: str) -> Path:
            path = Path(paths_cfg.get(key, default))
            return path if path.is_absolute() else base_dir / path

        self.tag_root = _resolve("tag_captures", "captures/tags")
        self.enhanced_root = _resolve("enhanced_captures", "captures/enhanced")
        self.ocr_root = _resolve("ocr_captures", "captures/ocr")
        self.barcode_root = _resolve("barcode_captures", "captures/barcode")

        self._executor: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._pending = 0
        self.saved_count = 0
        self.dropped_count = 0
        self._saved_tag_keys: "OrderedDict[tuple[str, int, int], None]" = (
            OrderedDict()
        )

        if self.enabled:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="ImageEvidenceWriter",
            )

            for directory in (
                self.tag_root,
                self.enhanced_root,
                self.ocr_root,
                self.barcode_root,
            ):
                directory.mkdir(parents=True, exist_ok=True)

        print(
            "[IMAGE-SAVE] "
            f"enabled={self.enabled} "
            f"tag={self.tag_enabled} "
            f"raw={self.save_raw} "
            f"enhanced={self.save_enhanced} "
            f"ocr={self.ocr_enabled} "
            f"barcode={self.barcode_enabled}"
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _valid(image: Optional[np.ndarray]) -> bool:
        return (
            image is not None
            and isinstance(image, np.ndarray)
            and image.size > 0
        )

    def _submit_write(
        self,
        image: Optional[np.ndarray],
        directory: Path,
        prefix: str,
        frame_id: int,
        track_id: int,
    ) -> Optional[str]:

        executor = self._executor

        if not self.enabled or executor is None:
            return None

        if not self._valid(image):
            return None

        with self._lock:
            if self._pending >= self.max_pending:
                self.dropped_count += 1
                return None
            self._pending += 1

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")

        filename = (
            f"{self.camera_id}"
            f"_track{track_id}"
            f"_frame{frame_id}"
            f"_{timestamp}"
            f"_{prefix}.jpg"
        )

        path = directory / filename
        image_copy = image.copy()

        def _write() -> None:
            try:
                directory.mkdir(parents=True, exist_ok=True)

                ok = cv2.imwrite(
                    str(path),
                    image_copy,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )

                if ok:
                    with self._lock:
                        self.saved_count += 1
                else:
                    print(f"[IMAGE-SAVE] cv2.imwrite failed: {path}")

            except Exception as exc:
                print(
                    "[IMAGE-SAVE] write failed "
                    f"{path}: {type(exc).__name__}: {exc}"
                )

            finally:
                with self._lock:
                    self._pending -= 1

        try:
            executor.submit(_write)
        except RuntimeError:
            # Executor already shut down.
            with self._lock:
                self._pending -= 1
            return None

        return str(path)

    # ------------------------------------------------------------------

    def save_processing_images(
        self,
        raw_tag: Optional[np.ndarray],
        enhanced_tag: Optional[np.ndarray],
        processing_frame: Optional[np.ndarray],
        stage: str,
        frame_id: int,
        track_id: int,
    ) -> dict[str, Optional[str]]:
        """
        Save (never raises):
          1. raw selected IATA tag        -> captures/tags
          2. enhanced IATA tag            -> captures/enhanced
          3. exact OCR/barcode input      -> captures/ocr | captures/barcode
        """

        paths: dict[str, Optional[str]] = {
            "raw_tag": None,
            "enhanced_tag": None,
            "processing_input": None,
        }

        if not self.enabled:
            return paths

        try:
            stage = str(stage).strip().lower()

            if stage not in {"ocr", "barcode"}:
                return paths

            if stage == "ocr" and not self.ocr_enabled:
                stage_enabled = False
            elif stage == "barcode" and not self.barcode_enabled:
                stage_enabled = False
            else:
                stage_enabled = True

            key = (self.camera_id, int(track_id), int(frame_id))

            # Raw/enhanced are shared by OCR and barcode: save them once
            # per camera/track/frame.
            with self._lock:
                first_for_tag = key not in self._saved_tag_keys

                if first_for_tag:
                    self._saved_tag_keys[key] = None

                    while len(self._saved_tag_keys) > self._MAX_REMEMBERED_KEYS:
                        self._saved_tag_keys.popitem(last=False)

            if self.tag_enabled and first_for_tag:

                if self.save_raw:
                    paths["raw_tag"] = self._submit_write(
                        raw_tag, self.tag_root, "raw", frame_id, track_id
                    )

                if self.save_enhanced:
                    paths["enhanced_tag"] = self._submit_write(
                        enhanced_tag,
                        self.enhanced_root,
                        "enhanced",
                        frame_id,
                        track_id,
                    )

            if stage_enabled:
                root = self.ocr_root if stage == "ocr" else self.barcode_root

                paths["processing_input"] = self._submit_write(
                    processing_frame, root, stage, frame_id, track_id
                )

        except Exception as exc:
            print(
                "[IMAGE-SAVE] save_processing_images error: "
                f"{type(exc).__name__}: {exc}"
            )

        return paths

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush pending image writes. Safe to call more than once."""

        executor = self._executor

        if executor is None:
            return

        self._executor = None

        executor.shutdown(wait=True, cancel_futures=False)

        print(
            "[IMAGE-SAVE] Writer stopped. "
            f"saved={self.saved_count} dropped={self.dropped_count}"
        )