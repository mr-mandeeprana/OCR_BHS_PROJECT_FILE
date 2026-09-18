from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when the application configuration is invalid."""


class Config:
    """
    Central configuration loader for OCR_BHS.

    Loads config.yaml once and provides convenient access to
    configuration values.
    """

    def __init__(self, config_path: str | Path | None = None):
        if config_path is None:
            config_path = Path(__file__).resolve().parents[2] / "config.yaml"

        self.path = Path(config_path).resolve()

        if not self.path.exists():
            raise ConfigError(
                f"Configuration file not found: {self.path}"
            )

        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            with self.path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file)

        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Invalid YAML configuration: {exc}"
            ) from exc

        except OSError as exc:
            raise ConfigError(
                f"Unable to read configuration file: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ConfigError(
                "config.yaml must contain a YAML dictionary/object."
            )

        return data

    # --------------------------------------------------------
    # Generic access
    # --------------------------------------------------------

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Get a top-level configuration value.

        Example:
            config.get("project")
        """

        return self.data.get(key, default)

    def get_path(
        self,
        path: str,
        default: Any = None,
    ) -> Any:
        """
        Get a nested configuration value.

        Example:
            config.get_path("detection.model.confidence")
        """

        current: Any = self.data

        for key in path.split("."):
            if not isinstance(current, dict):
                return default

            if key not in current:
                return default

            current = current[key]

        return current

    # --------------------------------------------------------
    # Cameras
    # --------------------------------------------------------

    @property
    def cameras(self) -> dict[str, dict[str, Any]]:
        return self.data.get("cameras", {})

    def get_camera(self, camera_id: str) -> dict[str, Any]:
        camera = self.cameras.get(camera_id)

        if camera is None:
            raise ConfigError(
                f"Camera '{camera_id}' is not configured."
            )

        return camera

    def get_enabled_cameras(self) -> dict[str, dict[str, Any]]:
        return {
            camera_id: camera_config
            for camera_id, camera_config in self.cameras.items()
            if camera_config.get("enabled", False)
        }

    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    @property
    def detection(self) -> dict[str, Any]:
        return self.data.get("detection", {})

    @property
    def model_path(self) -> str:
        return self.get_path(
            "detection.model.path",
            "models/yolo/iata_tag_yolo26n_best.pt",
        )

    @property
    def detection_confidence(self) -> float:
        return float(
            self.get_path(
                "detection.model.confidence",
                0.30,
            )
        )

    @property
    def image_size(self) -> int:
        return int(
            self.get_path(
                "detection.model.image_size",
                640,
            )
        )

    @property
    def device(self) -> str:
        return str(
            self.get_path(
                "detection.model.device",
                "cpu",
            )
        )

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    @property
    def tracking(self) -> dict[str, Any]:
        return self.data.get("tracking", {})

    # --------------------------------------------------------
    # Barcode
    # --------------------------------------------------------

    @property
    def barcode(self) -> dict[str, Any]:
        return self.data.get("barcode", {})

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    @property
    def ocr(self) -> dict[str, Any]:
        return self.data.get("ocr", {})

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    @property
    def validation(self) -> dict[str, Any]:
        return self.data.get("validation", {})

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    @property
    def pipeline(self) -> dict[str, Any]:
        return self.data.get("pipeline", {})

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    @property
    def database(self) -> dict[str, Any]:
        return self.data.get("database", {})

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    @property
    def paths(self) -> dict[str, Any]:
        return self.data.get("paths", {})

    # --------------------------------------------------------
    # Environment variables
    # --------------------------------------------------------

    @staticmethod
    def env(
        name: str,
        default: str | None = None,
        required: bool = False,
    ) -> str | None:

        value = os.getenv(name, default)

        if required and not value:
            raise ConfigError(
                f"Required environment variable '{name}' is not set."
            )

        return value

    def camera_source(self, camera_id: str) -> str:
        """
        Resolve a camera source from the environment variable
        specified in config.yaml.
        """

        camera = self.get_camera(camera_id)

        source_type = camera.get("source_type", "rtsp")

        source_env = camera.get("source_env")

        if source_env:
            source = os.getenv(source_env)

            if source:
                return source

        # Allow direct source in config for video/webcam testing.
        source = camera.get("source")

        if source is not None:
            return str(source)

        if source_type == "webcam":
            return "0"

        raise ConfigError(
            f"No source configured for camera '{camera_id}'. "
            f"Expected environment variable '{source_env}'."
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def validate(self) -> None:
        """
        Validate essential configuration values.
        """

        required_sections = [
            "project",
            "cameras",
            "detection",
            "tracking",
            "barcode",
            "ocr",
            "validation",
            "pipeline",
            "paths",
        ]

        missing = [
            section
            for section in required_sections
            if section not in self.data
        ]

        if missing:
            raise ConfigError(
                "Missing configuration sections: "
                + ", ".join(missing)
            )

        if not self.cameras:
            raise ConfigError(
                "At least one camera must be configured."
            )

        if self.detection_confidence < 0 or self.detection_confidence > 1:
            raise ConfigError(
                "Detection confidence must be between 0 and 1."
            )

        if self.image_size <= 0:
            raise ConfigError(
                "Detection image_size must be greater than 0."
            )

        enabled_cameras = self.get_enabled_cameras()

        if not enabled_cameras:
            raise ConfigError(
                "At least one camera must be enabled."
            )

    # --------------------------------------------------------
    # Representation
    # --------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Config("
            f"path='{self.path}', "
            f"cameras={len(self.cameras)}, "
            f"enabled={len(self.get_enabled_cameras())}"
            f")"
        )


# ------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------

def load_config(
    config_path: str | Path | None = None,
) -> Config:
    """
    Load and validate OCR_BHS configuration.
    """

    config = Config(config_path)
    config.validate()

    return config


if __name__ == "__main__":
    config = load_config()

    print("======================================")
    print(" OCR_BHS CONFIGURATION")
    print("======================================")
    print(f"Config file : {config.path}")
    print(f"Project     : {config.get_path('project.name')}")
    print(f"Model       : {config.model_path}")
    print(f"Device      : {config.device}")
    print(f"Confidence  : {config.detection_confidence}")
    print(
        f"Enabled cameras: "
        f"{list(config.get_enabled_cameras().keys())}"
    )