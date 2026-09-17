"""
OCR_BHS - Camera Manager

Features:
    - RTSP live camera support
    - Recorded video support
    - Automatic reconnect
    - Video looping
    - Low-latency latest-frame buffer
    - Continuous background capture
    - Multi-camera CameraManager
    - Explicit .env loading from project root

Run from project root:

    python -m app.camera.camera
"""

from __future__ import annotations

import os
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Iterator, Any

import cv2
from dotenv import load_dotenv

from app.config.config_loader import Config, load_config


# ============================================================
# PROJECT / ENVIRONMENT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

ENV_FILE = os.path.join(
    PROJECT_ROOT,
    ".env",
)

# Explicitly load .env from project root.
load_dotenv(
    dotenv_path=ENV_FILE,
    override=False,
)


# ============================================================
# CAMERA STATUS
# ============================================================

@dataclass
class CameraStatus:
    camera_id: str
    connected: bool = False
    running: bool = False
    source_type: str = ""
    frame_count: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    last_frame_time: float = 0.0
    reconnect_count: int = 0
    error: Optional[str] = None


# ============================================================
# CAMERA
# ============================================================

class Camera:
    """
    Single camera capture manager.

    The camera capture runs in a background thread.

    buffer_size=1 means:
        Only the newest frame is retained.

    This is preferred for real-time RTSP processing because
    downstream processing should not consume old frames.
    """

    def __init__(
        self,
        camera_id: str,
        config: Config,
    ):
        self.camera_id = camera_id
        self.config = config

        # ----------------------------------------------------
        # Load camera configuration
        # ----------------------------------------------------

        camera_config = config.get_camera(
            camera_id
        )

        if camera_config is None:
            raise ValueError(
                f"Camera '{camera_id}' "
                f"not found in config.yaml."
            )

        self.camera_config = camera_config

        self.enabled = bool(
            camera_config.get(
                "enabled",
                False,
            )
        )

        self.source_type = str(
            camera_config.get(
                "source_type",
                "rtsp",
            )
        ).lower().strip()

        self.source_env = str(
            camera_config.get(
                "source_env",
                "",
            )
        ).strip()

        self.video_source = str(
            camera_config.get(
                "video_source",
                "",
            )
        ).strip()

        self.camera_name = str(
            camera_config.get(
                "name",
                camera_id,
            )
        )

        self.reconnect_enabled = bool(
            camera_config.get(
                "reconnect",
                True,
            )
        )

        self.reconnect_delay = float(
            camera_config.get(
                "reconnect_delay",
                2,
            )
        )

        self.buffer_size = int(
            camera_config.get(
                "buffer_size",
                1,
            )
        )

        if self.buffer_size < 1:
            self.buffer_size = 1

        self.target_fps = float(
            camera_config.get(
                "target_fps",
                0,
            )
        )

        self.video_loop = bool(
            camera_config.get(
                "video_loop",
                True,
            )
        )

        # ----------------------------------------------------
        # OpenCV
        # ----------------------------------------------------

        self.cap: Optional[
            cv2.VideoCapture
        ] = None

        # ----------------------------------------------------
        # Threading
        # ----------------------------------------------------

        self.running = False
        self.connected = False

        self.stop_event = threading.Event()

        self.capture_thread: Optional[
            threading.Thread
        ] = None

        self.frame_lock = threading.Lock()

        # ----------------------------------------------------
        # Frame buffer
        # ----------------------------------------------------

        self.frames = deque(
            maxlen=self.buffer_size
        )

        self.latest_frame = None

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        self.frame_count = 0

        self.start_time = 0.0

        self.last_frame_time = 0.0

        self.last_fps_time = time.time()

        self.fps_frame_count = 0

        self.current_fps = 0.0

        self.width = 0

        self.height = 0

        self.source_fps = 0.0

        self.reconnect_count = 0

        self.last_error: Optional[str] = None

        self.video_eof = False

    # ========================================================
    # SOURCE
    # ========================================================

    def _get_source(self) -> Any:
        """
        Resolve configured source.

        RTSP:
            Reads URL from .env.

        Video:
            Reads local video path from config.yaml.
        """

        # ----------------------------------------------------
        # RTSP
        # ----------------------------------------------------

        if self.source_type == "rtsp":

            if not self.source_env:

                raise ValueError(
                    f"[{self.camera_id}] "
                    "source_env is empty."
                )

            # Re-load .env in case it was modified.
            load_dotenv(
                dotenv_path=ENV_FILE,
                override=False,
            )

            source = os.getenv(
                self.source_env,
                "",
            ).strip()

            if not source:

                raise ValueError(
                    f"[{self.camera_id}] "
                    f"Environment variable "
                    f"'{self.source_env}' is empty."
                )

            return source

        # ----------------------------------------------------
        # VIDEO
        # ----------------------------------------------------

        if self.source_type == "video":

            if not self.video_source:

                raise ValueError(
                    f"[{self.camera_id}] "
                    "video_source is empty."
                )

            project_root = getattr(
                self.config,
                "project_root",
                PROJECT_ROOT,
            )

            source = os.path.join(
                str(project_root),
                self.video_source,
            )

            source = os.path.abspath(
                source
            )

            if not os.path.exists(source):

                raise FileNotFoundError(
                    f"[{self.camera_id}] "
                    f"Video file not found:\n"
                    f"{source}"
                )

            return source

        # ----------------------------------------------------
        # Unsupported
        # ----------------------------------------------------

        raise ValueError(
            f"[{self.camera_id}] "
            f"Unsupported source_type: "
            f"'{self.source_type}'. "
            f"Use 'rtsp' or 'video'."
        )

    # ========================================================
    # OPEN CAPTURE
    # ========================================================

    def _open(self) -> bool:
        """
        Open RTSP or video source.
        """

        self._release_capture()

        try:

            source = self._get_source()

            # ------------------------------------------------
            # Do NOT print source.
            #
            # RTSP source may contain credentials.
            # ------------------------------------------------

            print(
                f"[{self.camera_id}] "
                f"Opening {self.source_type} source..."
            )

            # ------------------------------------------------
            # RTSP
            # ------------------------------------------------

            if self.source_type == "rtsp":

                self.cap = cv2.VideoCapture(
                    source,
                    cv2.CAP_FFMPEG,
                )

                # Low-latency buffer.
                try:
                    self.cap.set(
                        cv2.CAP_PROP_BUFFERSIZE,
                        1,
                    )
                except Exception:
                    pass

            # ------------------------------------------------
            # VIDEO
            # ------------------------------------------------

            else:

                self.cap = cv2.VideoCapture(
                    source
                )

            # ------------------------------------------------
            # Check object
            # ------------------------------------------------

            if self.cap is None:

                self.last_error = (
                    "VideoCapture object "
                    "could not be created."
                )

                return False

            # ------------------------------------------------
            # Check opened
            # ------------------------------------------------

            if not self.cap.isOpened():

                self.last_error = (
                    "VideoCapture failed "
                    "to open source."
                )

                self._release_capture()

                return False

            # ------------------------------------------------
            # Properties
            # ------------------------------------------------

            self.width = int(
                self.cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            self.height = int(
                self.cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            self.source_fps = float(
                self.cap.get(
                    cv2.CAP_PROP_FPS
                )
            )

            # Some RTSP streams report 0 FPS.
            if self.source_fps <= 0:

                self.source_fps = 25.0

            self.connected = True

            self.video_eof = False

            self.last_error = None

            print(
                f"[{self.camera_id}] "
                f"Connected | "
                f"Resolution: "
                f"{self.width}x{self.height} | "
                f"FPS: "
                f"{self.source_fps:.2f}"
            )

            return True

        except Exception as exc:

            self.connected = False

            self.last_error = str(
                exc
            )

            print(
                f"[{self.camera_id}] "
                f"Open error: {exc}"
            )

            self._release_capture()

            return False

    # ========================================================
    # RELEASE
    # ========================================================

    def _release_capture(self) -> None:
        """
        Safely release VideoCapture.
        """

        if self.cap is not None:

            try:

                self.cap.release()

            except Exception:
                pass

            self.cap = None

        self.connected = False

    # ========================================================
    # START
    # ========================================================

    def start(self) -> bool:
        """
        Start the camera.

        Returns:
            True  -> camera connected and thread started
            False -> initial connection failed
        """

        if not self.enabled:

            print(
                f"[{self.camera_id}] "
                "Camera disabled."
            )

            return False

        if self.running:

            return True

        # ----------------------------------------------------
        # Initial connection
        # ----------------------------------------------------

        if not self._open():

            # Start background reconnect only if enabled.
            if not self.reconnect_enabled:

                print(
                    f"[{self.camera_id}] "
                    "Initial connection failed."
                )

                return False

            print(
                f"[{self.camera_id}] "
                "Initial connection failed. "
                "Background reconnect enabled."
            )

        # ----------------------------------------------------
        # Start capture thread
        # ----------------------------------------------------

        self.stop_event.clear()

        self.running = True

        self.start_time = time.time()

        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"OCR-BHS-Camera-{self.camera_id}",
            daemon=True,
        )

        self.capture_thread.start()

        return True

    # ========================================================
    # STOP
    # ========================================================

    def stop(self) -> None:
        """
        Stop camera capture.
        """

        self.running = False

        self.stop_event.set()

        if self.capture_thread is not None:

            if (
                self.capture_thread.is_alive()
                and
                threading.current_thread()
                is not self.capture_thread
            ):

                self.capture_thread.join(
                    timeout=3
                )

        self.capture_thread = None

        self._release_capture()

    # ========================================================
    # CAPTURE LOOP
    # ========================================================

    def _capture_loop(self) -> None:
        """
        Background capture loop.
        """

        while not self.stop_event.is_set():

            # ------------------------------------------------
            # Connection check
            # ------------------------------------------------

            if (
                self.cap is None
                or not self.connected
            ):

                if not self.reconnect_enabled:

                    time.sleep(
                        0.1
                    )

                    continue

                self.reconnect_count += 1

                print(
                    f"[{self.camera_id}] "
                    f"Reconnecting "
                    f"(attempt "
                    f"{self.reconnect_count})..."
                )

                if not self._open():

                    time.sleep(
                        self.reconnect_delay
                    )

                    continue

            # ------------------------------------------------
            # Read frame
            # ------------------------------------------------

            ret, frame = self.cap.read()

            # ------------------------------------------------
            # Failed frame
            # ------------------------------------------------

            if (
                not ret
                or frame is None
            ):

                self.connected = False

                self.last_error = (
                    "Frame read failed."
                )

                # --------------------------------------------
                # Recorded video
                # --------------------------------------------

                if self.source_type == "video":

                    self.video_eof = True

                    if self.video_loop:

                        print(
                            f"[{self.camera_id}] "
                            "Video ended. "
                            "Restarting..."
                        )

                        self._release_capture()

                        time.sleep(
                            0.2
                        )

                        continue

                    print(
                        f"[{self.camera_id}] "
                        "Video ended."
                    )

                    self.running = False

                    break

                # --------------------------------------------
                # RTSP
                # --------------------------------------------

                if self.reconnect_enabled:

                    print(
                        f"[{self.camera_id}] "
                        "Frame read failed. "
                        "Reconnecting..."
                    )

                    self._release_capture()

                    time.sleep(
                        self.reconnect_delay
                    )

                    continue

                time.sleep(
                    0.05
                )

                continue

            # ------------------------------------------------
            # Target FPS
            # ------------------------------------------------

            if self.target_fps > 0:

                frame_interval = (
                    1.0
                    / self.target_fps
                )

                now = time.time()

                if (
                    self.last_frame_time > 0
                    and
                    (
                        now
                        - self.last_frame_time
                    )
                    < frame_interval
                ):

                    sleep_time = (
                        frame_interval
                        -
                        (
                            now
                            - self.last_frame_time
                        )
                    )

                    if sleep_time > 0:

                        time.sleep(
                            sleep_time
                        )

            # ------------------------------------------------
            # Timestamp
            # ------------------------------------------------

            timestamp = time.time()

            # ------------------------------------------------
            # Store latest frame
            # ------------------------------------------------

            with self.frame_lock:

                self.frames.append(
                    (
                        timestamp,
                        frame,
                    )
                )

                self.latest_frame = frame

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            self.frame_count += 1

            self.fps_frame_count += 1

            self.last_frame_time = (
                timestamp
            )

            elapsed = (
                timestamp
                -
                self.last_fps_time
            )

            if elapsed >= 1.0:

                self.current_fps = (
                    self.fps_frame_count
                    /
                    elapsed
                )

                self.fps_frame_count = 0

                self.last_fps_time = (
                    timestamp
                )

    # ========================================================
    # GET LATEST FRAME
    # ========================================================

    def get_latest_frame(
        self,
        copy: bool = True,
    ):
        """
        Get newest frame.

        copy=True:
            Safest for downstream processing.

        copy=False:
            Faster, but caller must not modify frame.
        """

        with self.frame_lock:

            if self.latest_frame is None:

                return None

            if copy:

                return self.latest_frame.copy()

            return self.latest_frame

    # ========================================================
    # GET FRAME + TIMESTAMP
    # ========================================================

    def get_latest_frame_with_timestamp(
        self,
        copy: bool = True,
    ):
        """
        Returns:

            (timestamp, frame)

        or:

            None
        """

        with self.frame_lock:

            if not self.frames:

                return None

            timestamp, frame = (
                self.frames[-1]
            )

            if copy:

                frame = frame.copy()

            return (
                timestamp,
                frame,
            )

    # ========================================================
    # FRAME GENERATOR
    # ========================================================

    def frames_generator(
        self,
        copy: bool = True,
    ) -> Iterator:
        """
        Yield only new frames.
        """

        last_timestamp = 0.0

        while self.running:

            result = (
                self.get_latest_frame_with_timestamp(
                    copy=copy
                )
            )

            if result is None:

                time.sleep(
                    0.001
                )

                continue

            timestamp, frame = result

            if timestamp <= last_timestamp:

                time.sleep(
                    0.001
                )

                continue

            last_timestamp = timestamp

            yield frame

    # ========================================================
    # BUFFERED FRAMES
    # ========================================================

    def get_buffered_frames(
        self,
        copy: bool = True,
    ):
        """
        Return all currently buffered frames.
        """

        with self.frame_lock:

            result = []

            for timestamp, frame in self.frames:

                if copy:

                    frame = frame.copy()

                result.append(
                    (
                        timestamp,
                        frame,
                    )
                )

            return result

    # ========================================================
    # WAIT FOR FIRST FRAME
    # ========================================================

    def wait_for_frame(
        self,
        timeout: float = 10.0,
    ):
        """
        Wait until first frame arrives.
        """

        start = time.time()

        while (
            time.time() - start
            < timeout
        ):

            frame = (
                self.get_latest_frame()
            )

            if frame is not None:

                return frame

            time.sleep(
                0.01
            )

        return None

    # ========================================================
    # FPS
    # ========================================================

    def get_fps(self) -> float:

        return float(
            self.current_fps
        )

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(
        self
    ) -> CameraStatus:

        return CameraStatus(
            camera_id=self.camera_id,
            connected=self.connected,
            running=self.running,
            source_type=self.source_type,
            frame_count=self.frame_count,
            fps=self.current_fps,
            width=self.width,
            height=self.height,
            last_frame_time=self.last_frame_time,
            reconnect_count=self.reconnect_count,
            error=self.last_error,
        )

    # ========================================================
    # RUNNING
    # ========================================================

    def is_running(self) -> bool:

        return self.running

    # ========================================================
    # CONNECTED
    # ========================================================

    def is_connected(self) -> bool:

        return self.connected


# ============================================================
# CAMERA MANAGER
# ============================================================

class CameraManager:
    """
    Multi-camera manager.
    """

    def __init__(
        self,
        config: Config,
    ):

        self.config = config

        self.cameras: dict[
            str,
            Camera
        ] = {}

        enabled_cameras = (
            config.get_enabled_cameras()
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # get_enabled_cameras() returns a dictionary.
        #
        # Example:
        #
        # {
        #     "camera_1": {...},
        #     "camera_2": {...}
        # }
        # ----------------------------------------------------

        if isinstance(
            enabled_cameras,
            dict,
        ):

            camera_ids = (
                enabled_cameras.keys()
            )

        else:

            camera_ids = (
                enabled_cameras
            )

        for camera_id in camera_ids:

            self.cameras[
                camera_id
            ] = Camera(
                camera_id=camera_id,
                config=config,
            )

    # ========================================================
    # START ALL
    # ========================================================

    def start_all(
        self
    ) -> dict[str, bool]:

        results = {}

        for (
            camera_id,
            camera,
        ) in self.cameras.items():

            try:

                results[camera_id] = (
                    camera.start()
                )

            except Exception as exc:

                print(
                    f"[{camera_id}] "
                    f"Start error: {exc}"
                )

                results[camera_id] = False

        return results

    # ========================================================
    # STOP ALL
    # ========================================================

    def stop_all(self) -> None:

        for camera in (
            self.cameras.values()
        ):

            try:

                camera.stop()

            except Exception as exc:

                print(
                    f"Camera stop error: "
                    f"{exc}"
                )

    # ========================================================
    # GET CAMERA
    # ========================================================

    def get_camera(
        self,
        camera_id: str,
    ) -> Optional[Camera]:

        return self.cameras.get(
            camera_id
        )

    # ========================================================
    # GET ALL
    # ========================================================

    def get_all_cameras(
        self
    ) -> dict[str, Camera]:

        return self.cameras

    # ========================================================
    # GET FRAMES
    # ========================================================

    def get_latest_frames(self):

        frames = {}

        for (
            camera_id,
            camera,
        ) in self.cameras.items():

            frame = (
                camera.get_latest_frame()
            )

            if frame is not None:

                frames[
                    camera_id
                ] = frame

        return frames

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self):

        return {
            camera_id:
                camera.get_status()
            for (
                camera_id,
                camera
            )
            in self.cameras.items()
        }


# ============================================================
# ENVIRONMENT TEST
# ============================================================

def test_environment() -> bool:
    """
    Verify that .env exists and required RTSP variable
    is available.

    Never prints the actual RTSP URL.
    """

    print()
    print(
        "Environment:"
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f".env exists : "
        f"{os.path.exists(ENV_FILE)}"
    )

    # Load explicitly.
    load_dotenv(
        dotenv_path=ENV_FILE,
        override=False,
    )

    value = os.getenv(
        "CAMERA_1_RTSP",
        "",
    ).strip()

    if value:

        print(
            "CAMERA_1_RTSP: FOUND"
        )

        return True

    print(
        "CAMERA_1_RTSP: EMPTY"
    )

    return False


# ============================================================
# STANDALONE TEST
# ============================================================

def main():
    """
    Standalone camera test.

    Run from project root:

        python -m app.camera.camera
    """

    print(
        "=" * 60
    )

    print(
        " OCR_BHS CAMERA TEST"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env_ok = test_environment()

    print()

    if not env_ok:

        print(
            "ERROR: CAMERA_1_RTSP was not found."
        )

        print()
        print(
            "Check your .env file:"
        )

        print(
            f"{ENV_FILE}"
        )

        print()

        print(
            "Expected variable:"
        )

        print(
            "CAMERA_1_RTSP=<your RTSP URL>"
        )

        print()

        raise SystemExit(1)

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    try:

        config = load_config()

    except Exception as exc:

        print(
            f"Configuration error: {exc}"
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # Enabled cameras
    # --------------------------------------------------------

    cameras = (
        config.get_enabled_cameras()
    )

    print(
        f"Enabled cameras: {cameras}"
    )

    if not cameras:

        print(
            "No enabled cameras found."
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # Dictionary-safe camera ID
    # --------------------------------------------------------

    if isinstance(
        cameras,
        dict,
    ):

        camera_id = next(
            iter(cameras)
        )

    else:

        camera_id = cameras[0]

    print(
        f"Testing camera: "
        f"{camera_id}"
    )

    # --------------------------------------------------------
    # Create camera
    # --------------------------------------------------------

    try:

        camera = Camera(
            camera_id=camera_id,
            config=config,
        )

    except Exception as exc:

        print(
            f"Camera creation failed: "
            f"{exc}"
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    started = camera.start()

    if not started:

        print()
        print(
            "Initial camera connection failed."
        )

        print(
            "Reconnect is disabled or "
            "camera could not be started."
        )

        raise SystemExit(1)

    print()

    print(
        "Camera capture thread started."
    )

    print(
        "Waiting for first frame..."
    )

    print(
        "Press Ctrl+C to stop."
    )

    print()

    # --------------------------------------------------------
    # First frame
    # --------------------------------------------------------

    first_frame = (
        camera.wait_for_frame(
            timeout=10
        )
    )

    if first_frame is None:

        print(
            "WARNING: No frame received "
            "within 10 seconds."
        )

    else:

        height, width = (
            first_frame.shape[:2]
        )

        print(
            f"First frame received: "
            f"{width}x{height}"
        )

        print()

    # --------------------------------------------------------
    # Monitoring
    # --------------------------------------------------------

    last_status_time = time.time()

    try:

        while camera.is_running():

            time.sleep(
                0.1
            )

            now = time.time()

            if (
                now
                - last_status_time
                >= 2.0
            ):

                frame = (
                    camera.get_latest_frame(
                        copy=False
                    )
                )

                if frame is not None:

                    height, width = (
                        frame.shape[:2]
                    )

                    print(
                        f"Camera: "
                        f"{camera_id} | "
                        f"Frame: "
                        f"{camera.frame_count} | "
                        f"Size: "
                        f"{width}x{height} | "
                        f"FPS: "
                        f"{camera.get_fps():.2f} | "
                        f"Connected: "
                        f"{camera.is_connected()}"
                    )

                else:

                    print(
                        f"Camera: "
                        f"{camera_id} | "
                        "Waiting for frame..."
                    )

                last_status_time = now

    except KeyboardInterrupt:

        print()
        print(
            "Keyboard interrupt received."
        )

    finally:

        print()

        print(
            f"[{camera_id}] "
            "Stopping camera..."
        )

        camera.stop()

        print(
            "Camera stopped."
        )

        print(
            f"Total frames received: "
            f"{camera.frame_count}"
        )

        print(
            "=" * 60
        )

        print(
            " CAMERA TEST COMPLETE"
        )

        print(
            "=" * 60
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()