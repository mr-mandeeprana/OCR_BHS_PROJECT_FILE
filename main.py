"""
OCR_BHS - Main Application Entry Point

Architecture:
    Camera
       ↓
    YOLO Detection
       ↓
    ByteTrack Tracking
       ↓
    Best Frame Selection
       ↓
    Asynchronous Processing Queue
       ├── OCR Workers
       └── Barcode Workers
              ↓
       Result Manager
              ↓
       Validation

Supported:
    - Multiple cameras
    - RTSP cameras
    - Recorded video testing
    - YOLO IATA tag detection
    - Continuous ByteTrack tracking
    - Asynchronous OCR
    - Asynchronous barcode processing
    - Central YAML configuration
    - Graceful shutdown
    - Environment variables
    - CPU/GPU configuration

Important:
    OCRBHSPipeline.start() is NON-BLOCKING.
    Therefore this file explicitly keeps the application alive
    until Ctrl+C or the pipeline reports that it has stopped.
"""

from __future__ import annotations

import inspect
import logging
import os
import signal
import sys
import time
import cv2
from pathlib import Path
from typing import Any, Dict, Optional


# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# OPTIONAL DOTENV
# ============================================================================

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

except ImportError:
    pass


# ============================================================================
# APPLICATION IMPORTS
# ============================================================================

from app.config.config_loader import Config
from app.camera.camera import Camera
from app.detection.yolo_detector import YOLODetector
from app.tracking.tag_tracker import TagTracker
from app.pipeline.pipeline import OCRBHSPipeline
from app.database.database_manager import (
    DatabaseManager,
    DatabaseWorker,
    load_database_config_from_yaml,
)
from app.database.result_persistence import ResultPersistence


# ============================================================================
# LOGGING
# ============================================================================

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)

logger = logging.getLogger("OCR_BHS")


# ============================================================================
# GLOBAL APPLICATION STATE
# ============================================================================

_shutdown_requested = False

_active_pipelines: Dict[str, OCRBHSPipeline] = {}
_active_cameras: Dict[str, Camera] = {}

# One shared asynchronous SQL Server worker for all cameras.
_database_manager: Optional[DatabaseManager] = None
_database_worker: Optional[DatabaseWorker] = None
_database_persistence: Dict[str, ResultPersistence] = {}


# ============================================================================
# LOG HELPERS
# ============================================================================

def log_info(message: str) -> None:
    logger.info(message)


def log_warning(message: str) -> None:
    logger.warning(message)


def log_error(message: str) -> None:
    logger.error(message)


# ============================================================================
# PATH HELPERS
# ============================================================================

def resolve_path(
    path_value: Optional[str],
    base_dir: Path = PROJECT_ROOT,
) -> Optional[Path]:
    """
    Resolve a relative project path.

    Environment variables and ~ are expanded.
    """

    if path_value is None:
        return None

    value = str(path_value).strip()

    if not value:
        return None

    value = os.path.expandvars(value)
    value = os.path.expanduser(value)

    path = Path(value)

    if not path.is_absolute():
        path = base_dir / path

    return path.resolve()


# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def get_config_value(
    config: Config,
    section: str,
    key: str,
    default: Any = None,
) -> Any:
    """
    Safely retrieve:
        config.data[section][key]
    """

    try:
        data = getattr(config, "data", None)

        if not isinstance(data, dict):
            return default

        section_data = data.get(section, {})

        if not isinstance(section_data, dict):
            return default

        return section_data.get(key, default)

    except Exception:
        return default


def get_enabled_cameras(config: Config) -> list[str]:
    """
    Return enabled camera IDs from central configuration.
    """

    cameras = get_config_value(
        config,
        "cameras",
        "cameras",
        None,
    )

    # Normal configuration structure:
    #
    # cameras:
    #   camera_1:
    #       enabled: true
    #   camera_2:
    #       enabled: false
    #
    camera_config = get_config_value(
        config,
        "cameras",
        None,
        None,
    )

    # Because get_config_value requires a key, handle the
    # cameras section directly.
    try:
        data = config.data

        camera_section = data.get("cameras", {})

        if not isinstance(camera_section, dict):
            return []

        enabled = []

        for camera_id, camera_data in camera_section.items():

            if not isinstance(camera_data, dict):
                continue

            if bool(camera_data.get("enabled", False)):
                enabled.append(str(camera_id))

        return enabled

    except Exception as exc:
        raise RuntimeError(
            f"Unable to read camera configuration: {exc}"
        ) from exc


def get_camera_config(
    config: Config,
    camera_id: str,
) -> Dict[str, Any]:
    """
    Return configuration dictionary for one camera.
    """

    try:
        cameras = config.data.get("cameras", {})

        if camera_id not in cameras:
            raise KeyError(
                f"Camera '{camera_id}' not found in config.yaml"
            )

        camera_config = cameras[camera_id]

        if not isinstance(camera_config, dict):
            raise TypeError(
                f"Camera '{camera_id}' configuration must be a dictionary."
            )

        return camera_config

    except Exception as exc:
        raise RuntimeError(
            f"Unable to load configuration for {camera_id}: {exc}"
        ) from exc


# ============================================================================
# CAMERA SOURCE
# ============================================================================

def resolve_camera_source(
    config: Config,
    camera_id: str,
) -> Any:
    """
    Resolve camera source.

    Supported examples:

        source_type: video
        video_source: videos/input/camera_1_test.mp4

    or:

        source_type: rtsp
        source_env: CAMERA_1_RTSP

    Environment variables are supported.
    """

    camera_config = get_camera_config(config, camera_id)

    source_type = str(
        camera_config.get("source_type", "rtsp")
    ).lower().strip()

    # ------------------------------------------------------------------------
    # VIDEO
    # ------------------------------------------------------------------------

    if source_type == "video":

        video_source = camera_config.get(
            "video_source"
        )

        if video_source is None:
            raise RuntimeError(
                f"{camera_id}: source_type=video but "
                f"'video_source' is missing."
            )

        video_path = resolve_path(str(video_source))

        if video_path is None:
            raise RuntimeError(
                f"{camera_id}: Invalid video source."
            )

        if not video_path.exists():
            raise FileNotFoundError(
                f"{camera_id}: Video file does not exist:\n"
                f"{video_path}"
            )

        return str(video_path)

    # ------------------------------------------------------------------------
    # RTSP
    # ------------------------------------------------------------------------

    if source_type == "rtsp":

        source_env = camera_config.get(
            "source_env"
        )

        if source_env:
            source_env = str(source_env).strip()

            rtsp_url = os.getenv(source_env)

            if rtsp_url:
                return rtsp_url

            log_warning(
                f"{camera_id}: Environment variable "
                f"{source_env} is not set."
            )

        # Optional direct RTSP URL.
        direct_source = camera_config.get(
            "rtsp_url"
        )

        if direct_source:
            return str(direct_source)

        raise RuntimeError(
            f"{camera_id}: No RTSP source found. "
            f"Set the configured environment variable."
        )

    # ------------------------------------------------------------------------
    # WEBCAM / OTHER NUMERIC SOURCE
    # ------------------------------------------------------------------------

    if source_type in {
        "webcam",
        "camera",
        "usb",
    }:

        source = camera_config.get(
            "source",
            camera_config.get("camera_index", 0),
        )

        try:
            return int(source)
        except (TypeError, ValueError):
            return source

    raise RuntimeError(
        f"{camera_id}: Unsupported source_type='{source_type}'."
    )


def mask_camera_source(source: Any) -> str:
    """
    Hide RTSP credentials in logs.
    """

    source_text = str(source)

    if "@" in source_text and "://" in source_text:

        try:
            scheme, remainder = source_text.split(
                "://",
                1,
            )

            if "@" in remainder:

                credentials, host = remainder.split(
                    "@",
                    1,
                )

                if ":" in credentials:

                    username = credentials.split(
                        ":",
                        1,
                    )[0]

                    return (
                        f"{scheme}://"
                        f"{username}:******@{host}"
                    )

        except Exception:
            pass

    return source_text


# ============================================================================
# DETECTION CONFIGURATION
# ============================================================================

def get_detection_config(
    config: Config,
) -> Dict[str, Any]:
    """
    Return detection section from config.yaml.
    """

    try:
        detection = config.data.get(
            "detection",
            {},
        )

        if not isinstance(detection, dict):
            raise TypeError(
                "Detection configuration must be a dictionary."
            )

        return detection

    except Exception as exc:
        raise RuntimeError(
            f"Unable to read detection configuration: {exc}"
        ) from exc


def resolve_model_path(
    config: Config,
) -> Path:
    """
    Resolve YOLO model path.

    Correct configuration source:
        config.data["detection"]["model_path"]

    NOT:
        config.model
    """

    detection = get_detection_config(config)

    model_path_value = detection.get(
        "model_path"
    )

    if not model_path_value:
        raise RuntimeError(
            "detection.model_path is missing from config.yaml."
        )

    model_path = resolve_path(
        str(model_path_value)
    )

    if model_path is None:
        raise RuntimeError(
            "Invalid YOLO model path."
        )

    return model_path


# ============================================================================
# YOLO DETECTOR
# ============================================================================

def create_detector(
    config: Config,
) -> YOLODetector:
    """
    Create YOLO detector using the actual detector constructor.

    inspect.signature() is used so this entry point remains compatible
    with the current YOLODetector implementation without hard-coding
    unsupported constructor arguments.
    """

    detection = get_detection_config(config)

    model_path = resolve_model_path(config)

    confidence = float(
        detection.get(
            "confidence",
            0.30,
        )
    )

    image_size = int(
        detection.get(
            "image_size",
            640,
        )
    )

    device = detection.get(
        "device",
        "cpu",
    )

    half = bool(
        detection.get(
            "half",
            False,
        )
    )

    max_det = int(
        detection.get(
            "max_det",
            20,
        )
    )

    log_info("Creating YOLO detector...")

    if not model_path.exists():
        raise FileNotFoundError(
            "YOLO model file does not exist:\n"
            f"{model_path}"
        )

    log_info(
        f"YOLO model path: {model_path}"
    )

    log_info(
        f"YOLO device: {device}"
    )

    # ------------------------------------------------------------------------
    # Inspect current constructor
    # ------------------------------------------------------------------------

    try:
        signature = inspect.signature(
            YOLODetector.__init__
        )

        parameters = signature.parameters

    except Exception:
        parameters = {}

    kwargs: Dict[str, Any] = {}

    # Model path
    for name in (
        "model_path",
        "model",
        "weights",
    ):
        if name in parameters:
            kwargs[name] = str(model_path)
            break

    # Confidence
    for name in (
        "confidence",
        "conf",
        "confidence_threshold",
    ):
        if name in parameters:
            kwargs[name] = confidence
            break

    # Image size
    for name in (
        "image_size",
        "imgsz",
        "image_size_px",
    ):
        if name in parameters:
            kwargs[name] = image_size
            break

    # Device
    if "device" in parameters:
        kwargs["device"] = device

    # Half precision
    for name in (
        "half",
        "use_half",
    ):
        if name in parameters:
            kwargs[name] = half
            break

    # Maximum detections
    for name in (
        "max_det",
        "max_detections",
    ):
        if name in parameters:
            kwargs[name] = max_det
            break

    # ------------------------------------------------------------------------
    # Instantiate
    # ------------------------------------------------------------------------

    try:
        detector = YOLODetector(
            **kwargs
        )

    except TypeError as first_error:

        log_warning(
            "YOLODetector constructor did not accept "
            "the detected keyword arguments."
        )

        log_warning(
            f"Constructor error: {first_error}"
        )

        # Fallback to positional model path.
        try:
            detector = YOLODetector(
                str(model_path)
            )

        except Exception as second_error:

            raise RuntimeError(
                "Unable to create YOLO detector.\n"
                f"First error: {first_error}\n"
                f"Second error: {second_error}"
            ) from second_error

    except Exception as exc:

        raise RuntimeError(
            f"Failed to create YOLO detector: {exc}"
        ) from exc

    log_info(
        "YOLO detector created successfully."
    )

    return detector


# ============================================================================
# BYTE TRACK
# ============================================================================

def create_tracker(
    config: Config,
) -> TagTracker:
    """
    Create the real Ultralytics ByteTrack wrapper.

    Configuration values are passed directly from:

        config.data["tracking"]
    """

    tracking = config.data.get(
        "tracking",
        {},
    )

    if not isinstance(tracking, dict):
        tracking = {}

    log_info(
        "Creating ByteTrack tracker..."
    )

    tracker = TagTracker(
        track_high_thresh=float(
            tracking.get(
                "track_high_thresh",
                0.4,
            )
        ),

        track_low_thresh=float(
            tracking.get(
                "track_low_thresh",
                0.1,
            )
        ),

        new_track_thresh=float(
            tracking.get(
                "new_track_thresh",
                0.4,
            )
        ),

        track_buffer=int(
            tracking.get(
                "track_buffer",
                30,
            )
        ),

        match_thresh=float(
            tracking.get(
                "match_thresh",
                0.8,
            )
        ),

        fuse_score=bool(
            tracking.get(
                "fuse_score",
                True,
            )
        ),

        min_hits=int(
            tracking.get(
                "min_hits",
                3,
            )
        ),

        prediction_buffer=int(
            tracking.get(
                "prediction_buffer",
                12,
            )
        ),

        max_jump_distance=float(
            tracking.get(
                "max_jump_distance",
                250.0,
            )
        ),

        min_iou_warning=float(
            tracking.get(
                "min_iou_warning",
                0.1,
            )
        ),

        unstable_frame_threshold=int(
            tracking.get(
                "unstable_frame_threshold",
                3,
            )
        ),

        frame_width=int(
            tracking.get(
                "frame_width",
                0,
            )
        ),

        frame_height=int(
            tracking.get(
                "frame_height",
                0,
            )
        ),

        min_bbox_area=float(
            tracking.get(
                "min_bbox_area",
                0.0,
            )
        ),

        min_bbox_area_ratio=float(
            tracking.get(
                "min_bbox_area_ratio",
                0.0,
            )
        ),

        min_track_confidence=float(
            tracking.get(
                "min_track_confidence",
                0.25,
            )
        ),

        iou_threshold=float(
            tracking.get(
                "iou_threshold",
                0.15,
            )
        ),

        max_missed_frames=int(
            tracking.get(
                "max_missed_frames",
                12,
            )
        ),

        max_center_distance=float(
            tracking.get(
                "max_center_distance",
                180.0,
            )
        ),

        min_size_similarity=float(
            tracking.get(
                "min_size_similarity",
                0.2,
            )
        ),

        frame_rate=float(
            tracking.get(
                "frame_rate",
                30.0,
            )
        ),
    )

    log_info(
        "ByteTrack tracker created successfully."
    )

    return tracker


# ============================================================================
# CAMERA CREATION
# ============================================================================

def create_camera(
    config: Config,
    camera_id: str,
) -> Camera:
    """
    Create Camera using the project's actual constructor:

        Camera(camera_id, config)
    """

    log_info(
        f"Creating camera: {camera_id}"
    )

    camera_config = get_camera_config(
        config,
        camera_id,
    )

    source = resolve_camera_source(
        config,
        camera_id,
    )

    log_info(
        f"{camera_id}: Source = "
        f"{mask_camera_source(source)}"
    )

    # Camera itself resolves the configured source.
    #
    # We intentionally do not inject `source` into the Camera
    # constructor because the current Camera API is:
    #
    #     Camera(camera_id, config)

    camera = Camera(
        camera_id,
        config,
    )

    log_info(
        f"{camera_id}: camera object created."
    )

    return camera


# ============================================================================
# PIPELINE CREATION
# ============================================================================

def create_pipeline(
    config: Config,
    camera_id: str,
) -> OCRBHSPipeline:
    """
    Create complete pipeline for one camera.
    """

    camera = create_camera(
        config,
        camera_id,
    )

    detector = create_detector(
        config,
    )

    tracker = create_tracker(
        config,
    )

    pipeline = OCRBHSPipeline(
        camera=camera,
        detector=detector,
        tracker=tracker,
        config=config,
    )

    log_info(
        f"{camera_id}: complete pipeline constructed."
    )

    return pipeline


# ============================================================================
# RUNTIME VALIDATION
# ============================================================================

def validate_runtime(
    config: Config,
    camera_id: Optional[str] = None,
) -> None:
    """
    Validate critical runtime configuration before starting.
    """

    if camera_id is not None:

        camera_config = get_camera_config(
            config,
            camera_id,
        )

        if not bool(
            camera_config.get(
                "enabled",
                False,
            )
        ):
            raise RuntimeError(
                f"{camera_id}: camera is disabled."
            )

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    model_path = resolve_model_path(
        config
    )

    if not model_path.exists():

        raise FileNotFoundError(
            "YOLO model does not exist:\n"
            f"{model_path}"
        )

    # ------------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------------

    detection = get_detection_config(
        config
    )

    device = str(
        detection.get(
            "device",
            "cpu",
        )
    ).lower()

    if device == "cuda":

        try:
            import torch

            if not torch.cuda.is_available():

                log_warning(
                    "CUDA is configured but "
                    "torch.cuda.is_available() is False."
                )

                log_warning(
                    "The detector may fall back to CPU."
                )

        except ImportError:

            log_warning(
                "PyTorch could not be imported while "
                "CUDA is configured."
            )

    log_info(
        f"{camera_id or 'Runtime'} validation passed."
    )

    log_info(
        f"Model path exists: {model_path}"
    )


# ============================================================================
# CONFIGURATION DISPLAY
# ============================================================================

def print_configuration(
    config: Config,
) -> None:
    """
    Print important runtime configuration.
    """

    project = config.data.get(
        "project",
        {},
    )

    detection = config.data.get(
        "detection",
        {},
    )

    tracking = config.data.get(
        "tracking",
        {},
    )

    pipeline = config.data.get(
        "pipeline",
        {},
    )

    enabled_cameras = get_enabled_cameras(
        config
    )

    print()
    print("=" * 70)
    print("OCR_BHS PROJECT")
    print("=" * 70)

    print(
        f"Project     : "
        f"{project.get('name', 'OCR_BHS')}"
    )

    print(
        f"Version     : "
        f"{project.get('version', '1.0.0')}"
    )

    print(
        f"Environment : "
        f"{project.get('environment', 'development')}"
    )

    print(
        f"Model       : "
        f"{detection.get('model_path')}"
    )

    print(
        f"Device      : "
        f"{detection.get('device', 'cpu')}"
    )

    print(
        f"Confidence  : "
        f"{detection.get('confidence', 0.3)}"
    )

    print(
        f"Tracker     : "
        f"{tracking.get('tracker', 'bytetrack')}"
    )

    print(
        f"Track buffer: "
        f"{tracking.get('track_buffer', 30)}"
    )

    print(
        f"Enabled cams: "
        f"{enabled_cameras}"
    )

    print(
        f"Pipeline    : "
        f"{pipeline.get('mode', 'live')}"
    )

    print(
        f"Async OCR   : "
        f"{pipeline.get('asynchronous_ocr', True)}"
    )

    print(
        f"Async Barcode: "
        f"{pipeline.get('asynchronous_barcode', True)}"
    )

    print("=" * 70)
    print()


# ============================================================================
# DATABASE STARTUP
# ============================================================================

def setup_database(config: Config) -> None:
    """Initialize SQL Server and start the shared async DB worker."""

    global _database_manager
    global _database_worker

    database_cfg = config.data.get("database", {}) or {}

    if not bool(database_cfg.get("enabled", False)):
        log_info("Database persistence disabled in config.yaml.")
        return

    db_config = load_database_config_from_yaml(
        config.path
    )

    if not db_config.enabled:
        log_info("Database persistence disabled by resolved configuration.")
        return

    if not db_config.host:
        raise RuntimeError("DB_HOST is not configured in .env.")

    if not db_config.database:
        raise RuntimeError("DB_NAME is not configured in .env.")

    if not db_config.username:
        raise RuntimeError("DB_USERNAME is not configured in .env.")

    if not db_config.password:
        raise RuntimeError("DB_PASSWORD is not configured in .env.")

    log_info("Initializing SQL Server...")
    log_info(f"Database server: {db_config.host}")
    log_info(f"Database name: {db_config.database}")

    manager = DatabaseManager(db_config)
    manager.setup()

    worker = DatabaseWorker(
        manager=manager,
        max_queue_size=int(
            database_cfg.get("queue_size", 500)
        ),
    )
    worker.start()

    _database_manager = manager
    _database_worker = worker

    log_info("SQL Server initialized successfully.")
    log_info("DatabaseWorker started.")


def attach_database_persistence(
    pipeline: OCRBHSPipeline,
    camera_id: str,
) -> None:
    """Attach persistence to the pipeline's ResultManager."""

    if _database_worker is None:
        return

    persistence = ResultPersistence(
        result_manager=pipeline.result_manager,
        database_worker=_database_worker,
    )

    pipeline.result_manager.add_callback(
        persistence.handle_result
    )

    _database_persistence[camera_id] = persistence

    log_info(
        f"{camera_id}: SQL persistence callback attached."
    )


def stop_database() -> None:
    """Stop the DB worker after all pipeline results have been submitted."""

    global _database_manager
    global _database_worker

    if _database_worker is not None:
        try:
            log_info("Stopping DatabaseWorker and draining pending records...")
            _database_worker.stop(wait=True)
            log_info(
                "DatabaseWorker final status: "
                f"{_database_worker.status()}"
            )
        except Exception as exc:
            log_warning(f"DatabaseWorker shutdown error: {exc}")

    if _database_manager is not None:
        try:
            _database_manager.close()
        except Exception as exc:
            log_warning(f"Database manager shutdown error: {exc}")

    _database_worker = None
    _database_manager = None
    _database_persistence.clear()


# ============================================================================
# PIPELINE START
# ============================================================================

def start_pipeline(
    pipeline: OCRBHSPipeline,
) -> None:
    """
    Start the pipeline.

    IMPORTANT:
        OCRBHSPipeline.start() is non-blocking.

    Therefore this function starts it and returns to the
    application's wait loop instead of treating start()
    as a blocking function.
    """

    method = getattr(
        pipeline,
        "start",
        None,
    )

    if callable(method):

        log_info(
            "Starting pipeline with .start()..."
        )

        method()

        log_info(
            "Pipeline start() returned. "
            "Pipeline is running in its internal loop/thread."
        )

        return

    # ------------------------------------------------------------------------
    # Fallback for blocking run() implementations
    # ------------------------------------------------------------------------

    method = getattr(
        pipeline,
        "run",
        None,
    )

    if callable(method):

        log_info(
            "Starting pipeline with .run()..."
        )

        method()

        return

    raise RuntimeError(
        "Pipeline object has neither a callable "
        "start() nor run() method."
    )


# ============================================================================
# PIPELINE STATE
# ============================================================================

def get_pipeline_running_state(
    pipeline: OCRBHSPipeline,
) -> Optional[bool]:
    """
    Try to determine whether the pipeline is running.

    Returns:
        True     -> explicitly running
        False    -> explicitly stopped
        None     -> state cannot be determined
    """

    # Common public/private state names.
    attributes = (
        "running",
        "is_running",
        "_running",
    )

    for attribute_name in attributes:

        if not hasattr(
            pipeline,
            attribute_name,
        ):
            continue

        try:

            value = getattr(
                pipeline,
                attribute_name,
            )

            if callable(value):
                value = value()

            if isinstance(
                value,
                bool,
            ):
                return value

        except Exception:
            continue

    return None


# ============================================================================
# PIPELINE WAIT
# ============================================================================

def wait_for_pipeline(
    pipeline: OCRBHSPipeline,
) -> None:
    """
    Keep the main application alive.

    The previous implementation exited immediately because
    OCRBHSPipeline.start() is asynchronous.

    This loop prevents main.py from reaching finally/stop()
    immediately after start().
    """

    global _shutdown_requested

    log_info(
        "Pipeline is active. Press Ctrl+C to stop."
    )

    while not _shutdown_requested:

        running = get_pipeline_running_state(
            pipeline
        )

        # --------------------------------------------------------------
        # Explicitly stopped
        # --------------------------------------------------------------

        if running is False:

            log_warning(
                "Pipeline reported that it is no longer running."
            )

            break

        # --------------------------------------------------------------
        # Keep main process alive
        # --------------------------------------------------------------

        try:

            time.sleep(1.0)

        except KeyboardInterrupt:

            _shutdown_requested = True

            log_info(
                "Keyboard interrupt received."
            )

            break


# ============================================================================
# PIPELINE STOP
# ============================================================================

def stop_pipeline(
    pipeline: Optional[OCRBHSPipeline],
) -> None:
    """
    Gracefully stop one pipeline.
    """

    if pipeline is None:
        return

    try:

        method = getattr(
            pipeline,
            "stop",
            None,
        )

        if callable(method):

            log_info(
                "Stopping pipeline with .stop()..."
            )

            method()

        else:

            log_warning(
                "Pipeline does not expose a stop() method."
            )

    except Exception as exc:

        log_error(
            f"Error while stopping pipeline: {exc}"
        )


# ============================================================================
# CAMERA STOP
# ============================================================================

def stop_camera(
    camera: Optional[Camera],
    camera_id: str,
) -> None:
    """
    Attempt to stop camera independently if supported.
    """

    if camera is None:
        return

    try:

        method = getattr(
            camera,
            "stop",
            None,
        )

        if callable(method):

            method()

            log_info(
                f"{camera_id}: camera stopped."
            )

            return

        method = getattr(
            camera,
            "release",
            None,
        )

        if callable(method):

            method()

            log_info(
                f"{camera_id}: camera released."
            )

            return

    except Exception as exc:

        log_warning(
            f"{camera_id}: camera shutdown warning: {exc}"
        )


# ============================================================================
# SIGNAL HANDLING
# ============================================================================

def request_shutdown(
    signum: int,
    frame: Any,
) -> None:
    """
    Handle Ctrl+C / termination signals.
    """

    global _shutdown_requested

    if _shutdown_requested:

        log_warning(
            "Shutdown already requested."
        )

        return

    _shutdown_requested = True

    try:

        signal_name = signal.Signals(
            signum
        ).name

    except Exception:

        signal_name = str(
            signum
        )

    log_info(
        f"Shutdown signal received: {signal_name}"
    )


def install_signal_handlers() -> None:
    """
    Install graceful shutdown handlers.
    """

    try:

        signal.signal(
            signal.SIGINT,
            request_shutdown,
        )

    except Exception:
        pass

    try:

        signal.signal(
            signal.SIGTERM,
            request_shutdown,
        )

    except Exception:
        pass


# ============================================================================
# SINGLE CAMERA RUNNER
# ============================================================================

def run_camera(
    config: Config,
    camera_id: str,
) -> None:
    """
    Run one configured camera pipeline.

    Current architecture uses one complete pipeline per enabled camera.
    """

    global _active_pipelines
    global _active_cameras
    global _shutdown_requested

    print()
    print(
        f"STARTING CAMERA: {camera_id}"
    )

    try:

        source = resolve_camera_source(
            config,
            camera_id,
        )

        print(
            f"{camera_id}: Source = "
            f"{mask_camera_source(source)}"
        )

        # --------------------------------------------------------------
        # Runtime validation
        # --------------------------------------------------------------

        validate_runtime(
            config,
            camera_id,
        )

        # --------------------------------------------------------------
        # Construct pipeline
        # --------------------------------------------------------------

        camera = create_camera(
            config,
            camera_id,
        )

        # Keep reference for graceful shutdown.
        _active_cameras[
            camera_id
        ] = camera

        detector = create_detector(
            config
        )

        tracker = create_tracker(
            config
        )

        pipeline = OCRBHSPipeline(
            camera=camera,
            detector=detector,
            tracker=tracker,
            config=config,
        )

        _active_pipelines[
            camera_id
        ] = pipeline

        log_info(
            "OCR BHS pipeline created successfully."
        )

        log_info(
            f"{camera_id}: complete pipeline constructed."
        )

        # --------------------------------------------------------------
        # Start
        # --------------------------------------------------------------

        start_pipeline(
            pipeline
        )

        # --------------------------------------------------------------
        # IMPORTANT
        #
        # start() is asynchronous.
        # Do NOT return here.
        # --------------------------------------------------------------

        wait_for_pipeline(
            pipeline
        )

    except KeyboardInterrupt:

        _shutdown_requested = True

        log_info(
            f"{camera_id}: Keyboard interrupt."
        )

    except Exception as exc:

        log_error(
            f"{camera_id}: pipeline failed: {exc}"
        )

        logger.exception(
            "%s: detailed exception",
            camera_id,
        )

        # Do not silently continue with a broken camera.
        # Main shutdown will clean up all references.

    finally:

        pipeline = _active_pipelines.get(
            camera_id
        )

        camera = _active_cameras.get(
            camera_id
        )

        stop_pipeline(
            pipeline
        )

        stop_camera(
            camera,
            camera_id,
        )

        log_info(
            f"{camera_id} shutdown complete."
        )


# ============================================================================
# LIVE VIDEO DISPLAY
# ============================================================================

def draw_tracking_video(
    camera_id: str,
    camera: Camera,
    pipeline: OCRBHSPipeline,
) -> bool:
    """
    Display the latest camera frame with the latest YOLO + ByteTrack state.

    The camera continuously captures frames in its own thread.
    The pipeline performs YOLO/ByteTrack at its configured interval.

    Therefore:
        Camera FPS       -> smooth display
        YOLO interval    -> background detection
        ByteTrack state  -> latest available tracking overlay

    Returns:
        True  -> continue running
        False -> user requested shutdown
    """

    try:
        frame = camera.get_latest_frame(copy=True)

    except Exception as exc:
        logger.debug(
            "%s: unable to get latest frame: %s",
            camera_id,
            exc,
        )
        return True

    if frame is None:
        return True

    if not hasattr(frame, "shape"):
        return True

    display = frame.copy()

    # ------------------------------------------------------------------------
    # Get latest tracking state
    # ------------------------------------------------------------------------

    tracks = []

    try:
        get_tracks = getattr(
            pipeline,
            "get_latest_tracks",
            None,
        )

        if callable(get_tracks):
            tracks = get_tracks() or []

    except Exception as exc:
        logger.debug(
            "%s: tracking overlay error: %s",
            camera_id,
            exc,
        )

    # ------------------------------------------------------------------------
    # Draw tracks
    # ------------------------------------------------------------------------

    if isinstance(tracks, dict):
        track_items = list(tracks.values())
    else:
        track_items = tracks

    for track in track_items:

        try:
            # --------------------------------------------------------------
            # Support dictionary-style track objects
            # --------------------------------------------------------------

            if isinstance(track, dict):

                bbox = (
                    track.get("bbox")
                    or track.get("xyxy")
                    or track.get("box")
                )

                track_id = (
                    track.get("track_id")
                    or track.get("id")
                    or "?"
                )

                confidence = track.get(
                    "confidence",
                    track.get(
                        "conf",
                        track.get("track_confidence", 0.0),
                    ),
                )

                is_confirmed = track.get(
                    "is_confirmed",
                    True,
                )

                is_predicted = track.get(
                    "is_predicted",
                    False,
                )

            # --------------------------------------------------------------
            # Support Detection/Track dataclass-style objects
            # --------------------------------------------------------------

            else:

                bbox = getattr(
                    track,
                    "bbox",
                    None,
                )

                if bbox is None:
                    bbox = getattr(
                        track,
                        "xyxy",
                        None,
                    )

                track_id = getattr(
                    track,
                    "track_id",
                    getattr(
                        track,
                        "id",
                        "?",
                    ),
                )

                # ----------------------------------------------------------
                # The project's Track dataclass (app/tracking/tag_tracker.py)
                # exposes confidence as `track_confidence`, and the raw
                # per-detection confidence as `track.detection.confidence`.
                # It does NOT expose `.confidence` / `.conf` directly, so
                # those must be tried last as generic fallbacks.
                # ----------------------------------------------------------

                confidence = getattr(
                    track,
                    "track_confidence",
                    None,
                )

                if confidence is None:

                    detection_obj = getattr(
                        track,
                        "detection",
                        None,
                    )

                    confidence = getattr(
                        detection_obj,
                        "confidence",
                        None,
                    )

                if confidence is None:

                    confidence = getattr(
                        track,
                        "confidence",
                        getattr(
                            track,
                            "conf",
                            0.0,
                        ),
                    )

                is_confirmed = getattr(
                    track,
                    "is_confirmed",
                    True,
                )

                is_predicted = getattr(
                    track,
                    "is_predicted",
                    False,
                )

            if bbox is None:
                continue

            if len(bbox) < 4:
                continue

            x1 = int(float(bbox[0]))
            y1 = int(float(bbox[1]))
            x2 = int(float(bbox[2]))
            y2 = int(float(bbox[3]))

            # --------------------------------------------------------------
            # Keep coordinates inside image
            # --------------------------------------------------------------

            height, width = display.shape[:2]

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            # --------------------------------------------------------------
            # Bounding box
            #
            # Color communicates ByteTrack state so a smooth (predicted)
            # track through a brief occlusion is visually distinguishable
            # from a fresh, confirmed detection:
            #   green  = confirmed track backed by a real detection
            #   yellow = predicted / coasted (no detection this cycle)
            # --------------------------------------------------------------

            box_color = (0, 255, 0)

            if is_predicted:
                box_color = (0, 220, 255)
            elif not is_confirmed:
                box_color = (0, 165, 255)

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                box_color,
                2,
            )

            # --------------------------------------------------------------
            # Label
            # --------------------------------------------------------------

            try:
                conf_text = f"{float(confidence):.2f}"
            except Exception:
                conf_text = "?"

            label = f"IATA_TAG | ID {track_id} | {conf_text}"

            if is_predicted:
                label += " | PRED"
            elif not is_confirmed:
                label += " | TENT"

            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                1,
            )

            label_y1 = max(
                0,
                y1 - text_height - baseline - 5,
            )

            label_y2 = max(
                text_height + baseline + 5,
                y1,
            )

            cv2.rectangle(
                display,
                (x1, label_y1),
                (
                    min(width - 1, x1 + text_width + 8),
                    min(height - 1, label_y2),
                ),
                box_color,
                -1,
            )

            cv2.putText(
                display,
                label,
                (x1 + 4, label_y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        except Exception as exc:

            logger.debug(
                "%s: unable to draw track: %s",
                camera_id,
                exc,
            )

    # ------------------------------------------------------------------------
    # Information overlay
    # ------------------------------------------------------------------------

    try:
        pipeline_stats = pipeline.get_stats()

        detection_count = pipeline_stats.get(
            "latest_detection_count",
            0,
        )

        track_count = pipeline_stats.get(
            "latest_track_count",
            len(track_items),
        )

        frames_processed = pipeline_stats.get(
            "frames_processed",
            0,
        )

    except Exception:

        detection_count = 0
        track_count = len(track_items)
        frames_processed = 0

    cv2.rectangle(
        display,
        (10, 10),
        (390, 105),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        display,
        f"Camera: {camera_id}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        display,
        f"YOLO detections: {detection_count}",
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        display,
        f"ByteTrack tracks: {track_count}",
        (20, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        display,
        f"Frames processed: {frames_processed}",
        (20, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # ------------------------------------------------------------------------
    # Show window
    # ------------------------------------------------------------------------

    window_name = f"OCR_BHS - {camera_id}"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.imshow(
        window_name,
        display,
    )

    # ------------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key in (ord("q"), ord("Q"), 27):

        logger.info(
            "%s: Q/ESC pressed. Requesting shutdown.",
            camera_id,
        )

        return False

    return True


# ============================================================================
# MULTI-CAMERA RUNNER
# ============================================================================

def run_enabled_cameras(
    config: Config,
) -> None:
    """
    Run all enabled cameras.

    Current OCRBHSPipeline implementation is camera-specific.
    This function starts each camera pipeline.

    For the current architecture, camera_1 can be used for testing
    independently before enabling the remaining RTSP cameras.
    """

    enabled_cameras = get_enabled_cameras(
        config
    )

    if not enabled_cameras:

        raise RuntimeError(
            "No enabled cameras found in config.yaml."
        )

    # ------------------------------------------------------------------------
    # IMPORTANT
    #
    # The current pipeline object owns its own processing loop.
    #
    # To avoid creating unsafe shared state between camera pipelines,
    # start each pipeline and keep references alive.
    #
    # For now, if multiple cameras are enabled, we construct and start
    # each one and then wait globally.
    # ------------------------------------------------------------------------

    pipelines: Dict[
        str,
        OCRBHSPipeline,
    ] = {}

    cameras: Dict[
        str,
        Camera,
    ] = {}

    global _active_pipelines
    global _active_cameras
    global _shutdown_requested

    try:

        # --------------------------------------------------------------
        # Construct all enabled cameras
        # --------------------------------------------------------------

        for camera_id in enabled_cameras:

            print()
            print(
                f"STARTING CAMERA: {camera_id}"
            )

            source = resolve_camera_source(
                config,
                camera_id,
            )

            print(
                f"{camera_id}: Source = "
                f"{mask_camera_source(source)}"
            )

            validate_runtime(
                config,
                camera_id,
            )

            camera = create_camera(
                config,
                camera_id,
            )

            detector = create_detector(
                config
            )

            tracker = create_tracker(
                config
            )

            pipeline = OCRBHSPipeline(
                camera=camera,
                detector=detector,
                tracker=tracker,
                config=config,
            )

            cameras[
                camera_id
            ] = camera

            pipelines[
                camera_id
            ] = pipeline

            _active_cameras[
                camera_id
            ] = camera

            _active_pipelines[
                camera_id
            ] = pipeline

            attach_database_persistence(
                pipeline,
                camera_id,
            )

            log_info(
                f"{camera_id}: complete pipeline constructed."
            )

        # --------------------------------------------------------------
        # Start all pipelines
        # --------------------------------------------------------------

        for camera_id, pipeline in pipelines.items():

            log_info(
                f"{camera_id}: starting pipeline..."
            )

            start_pipeline(
                pipeline
            )

        # --------------------------------------------------------------
        # LIVE VIDEO LOOP
        # --------------------------------------------------------------
        #
        # Pipelines run in their own background threads.
        #
        # This loop only handles:
        #   1. Smooth camera-frame display
        #   2. Latest ByteTrack overlay
        #   3. OpenCV keyboard events
        #
        # It does NOT run YOLO again.
        # It does NOT run ByteTrack again.
        # --------------------------------------------------------------

        log_info(
            "All enabled camera pipelines are running."
        )

        log_info(
            "Live tracking video display started."
        )

        log_info(
            "Press Q or ESC in the video window to stop."
        )

        while not _shutdown_requested:

            any_explicitly_stopped = False

            for camera_id, pipeline in pipelines.items():

                camera = cameras.get(camera_id)

                if camera is None:
                    continue

                state = get_pipeline_running_state(
                    pipeline
                )

                if state is False:

                    log_warning(
                        f"{camera_id}: pipeline stopped."
                    )

                    any_explicitly_stopped = True

                    continue

                try:

                    should_continue = draw_tracking_video(
                        camera_id=camera_id,
                        camera=camera,
                        pipeline=pipeline,
                    )

                    if not should_continue:

                        _shutdown_requested = True
                        break

                except Exception as exc:

                    log_warning(
                        f"{camera_id}: video display error: {exc}"
                    )

            if _shutdown_requested:
                break

            if any_explicitly_stopped:

                # Do not terminate all cameras automatically unless
                # requested by configuration.
                pipeline_config = config.data.get(
                    "pipeline",
                    {},
                )

                stop_on_camera_failure = bool(
                    pipeline_config.get(
                        "stop_on_camera_failure",
                        False,
                    )
                )

                if stop_on_camera_failure:

                    log_warning(
                        "A camera pipeline stopped and "
                        "stop_on_camera_failure=True."
                    )

                    break

            # ----------------------------------------------------------
            # Small sleep only to prevent 100% CPU in the display loop.
            #
            # Camera itself continues running at camera FPS.
            # ----------------------------------------------------------

            try:

                time.sleep(0.001)

            except KeyboardInterrupt:

                _shutdown_requested = True

                break

        cv2.destroyAllWindows()

    except KeyboardInterrupt:

        _shutdown_requested = True

    finally:

        log_info(
            "Stopping all camera pipelines..."
        )

        # --------------------------------------------------------------
        # Stop pipelines
        # --------------------------------------------------------------

        for camera_id, pipeline in pipelines.items():

            try:

                log_info(
                    f"{camera_id}: stopping pipeline..."
                )

                stop_pipeline(
                    pipeline
                )

            except Exception as exc:

                log_warning(
                    f"{camera_id}: shutdown error: {exc}"
                )

        # --------------------------------------------------------------
        # Stop cameras
        # --------------------------------------------------------------

        for camera_id, camera in cameras.items():

            try:

                stop_camera(
                    camera,
                    camera_id,
                )

            except Exception as exc:

                log_warning(
                    f"{camera_id}: camera shutdown error: {exc}"
                )

        try:

            cv2.destroyAllWindows()

        except Exception:
            pass

        log_info(
            "All camera pipelines stopped."
        )


# ============================================================================
# APPLICATION MAIN
# ============================================================================

def main() -> int:
    """
    Main application entry point.
    """

    global _shutdown_requested

    print()
    print("=" * 70)
    print("OCR_BHS APPLICATION")
    print("=" * 70)
    print(
        f"Project root: {PROJECT_ROOT}"
    )
    print("=" * 70)
    print()

    install_signal_handlers()

    try:

        # --------------------------------------------------------------
        # Load configuration
        # --------------------------------------------------------------

        log_info(
            "Loading configuration..."
        )

        config_path = PROJECT_ROOT / "config.yaml"

        if not config_path.exists():

            raise FileNotFoundError(
                "config.yaml not found:\n"
                f"{config_path}"
            )

        config = Config(
            config_path
        )

        print(
            "Configuration loaded successfully."
        )

        # --------------------------------------------------------------
        # Print configuration
        # --------------------------------------------------------------

        print_configuration(
            config
        )

        # --------------------------------------------------------------
        # Initialize SQL Server persistence
        # --------------------------------------------------------------

        setup_database(config)

        # --------------------------------------------------------------
        # Validate enabled cameras
        # --------------------------------------------------------------

        enabled_cameras = get_enabled_cameras(
            config
        )

        if not enabled_cameras:

            raise RuntimeError(
                "No enabled cameras configured."
            )

        # --------------------------------------------------------------
        # Run cameras
        # --------------------------------------------------------------

        run_enabled_cameras(
            config
        )

        return 0

    except KeyboardInterrupt:

        _shutdown_requested = True

        log_info(
            "Application interrupted by user."
        )

        return 0

    except FileNotFoundError as exc:

        log_error(
            str(exc)
        )

        return 1

    except Exception as exc:

        log_error(
            f"Application error: {exc}"
        )

        logger.exception(
            "Detailed application exception"
        )

        return 1

    finally:

        # --------------------------------------------------------------
        # Final cleanup
        # --------------------------------------------------------------

        log_info(
            "Final application cleanup..."
        )

        for camera_id, pipeline in list(
            _active_pipelines.items()
        ):

            try:

                stop_pipeline(
                    pipeline
                )

            except Exception:
                pass

        for camera_id, camera in list(
            _active_cameras.items()
        ):

            try:

                stop_camera(
                    camera,
                    camera_id,
                )

            except Exception:
                pass

        _active_pipelines.clear()
        _active_cameras.clear()

        # Pipelines are now stopped and their final callbacks have been
        # submitted to DatabaseWorker. Stop the worker only after that so
        # its queue can drain safely.
        stop_database()

        print()
        print(
            "OCR_BHS application stopped."
        )
        print()


# ============================================================================
# SCRIPT ENTRY
# ============================================================================

if __name__ == "__main__":

    exit_code = main()

    sys.exit(
        exit_code
    )