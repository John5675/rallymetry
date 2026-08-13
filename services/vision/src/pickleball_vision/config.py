"""Environment-based application configuration."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from pickleball_vision.errors import ConfigurationError

ENV_PREFIX = "PICKLEBALL_VISION_"
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
VALID_INFERENCE_DEVICE = re.compile(r"^(?:auto|cpu|mps|cuda(?::\d+)?|\d+)$")


class Environment(StrEnum):
    """Supported runtime environments."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class LogFormat(StrEnum):
    """Supported log output formats."""

    JSON = "json"
    CONSOLE = "console"


def _float_setting(source: Mapping[str, str], suffix: str, default: float) -> float:
    setting = f"{ENV_PREFIX}{suffix}"
    raw = source.get(setting, str(default))
    try:
        return float(raw)
    except ValueError as error:
        raise ConfigurationError(
            f"{setting} must be a number; received {raw!r}",
            setting=setting,
        ) from error


def _int_setting(source: Mapping[str, str], suffix: str, default: int) -> int:
    setting = f"{ENV_PREFIX}{suffix}"
    raw = source.get(setting, str(default))
    try:
        return int(raw)
    except ValueError as error:
        raise ConfigurationError(
            f"{setting} must be an integer; received {raw!r}",
            setting=setting,
        ) from error


@dataclass(frozen=True, slots=True)
class PersonDetectionSettings:
    """Validated pretrained person-detector inference settings."""

    model: str = "yolo26n.pt"
    device: str = "auto"
    min_confidence: float = 0.20
    image_size: int = 1280
    iou_threshold: float = 0.70
    max_detections: int = 100

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> PersonDetectionSettings:
        """Load and validate person inference settings."""

        source = os.environ if environ is None else environ
        defaults = cls()
        model = source.get(f"{ENV_PREFIX}PERSON_MODEL", defaults.model).strip()
        device = source.get(f"{ENV_PREFIX}PERSON_DEVICE", defaults.device).strip().lower()
        min_confidence = _float_setting(
            source,
            "PERSON_MIN_CONFIDENCE",
            defaults.min_confidence,
        )
        image_size = _int_setting(source, "PERSON_IMAGE_SIZE", defaults.image_size)
        iou_threshold = _float_setting(
            source,
            "PERSON_IOU_THRESHOLD",
            defaults.iou_threshold,
        )
        max_detections = _int_setting(
            source,
            "PERSON_MAX_DETECTIONS",
            defaults.max_detections,
        )

        validations = (
            (bool(model), "PERSON_MODEL", "must not be empty"),
            (
                VALID_INFERENCE_DEVICE.fullmatch(device) is not None,
                "PERSON_DEVICE",
                "must be auto, cpu, mps, cuda, cuda:N, or a numeric CUDA index",
            ),
            (
                0.0 <= min_confidence <= 1.0,
                "PERSON_MIN_CONFIDENCE",
                "must be between 0 and 1 inclusive",
            ),
            (image_size >= 32, "PERSON_IMAGE_SIZE", "must be at least 32"),
            (
                0.0 <= iou_threshold <= 1.0,
                "PERSON_IOU_THRESHOLD",
                "must be between 0 and 1 inclusive",
            ),
            (max_detections >= 1, "PERSON_MAX_DETECTIONS", "must be at least 1"),
        )
        for valid, suffix, reason in validations:
            if not valid:
                setting = f"{ENV_PREFIX}{suffix}"
                raise ConfigurationError(f"{setting} {reason}", setting=setting)

        return cls(
            model=model,
            device=device,
            min_confidence=min_confidence,
            image_size=image_size,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )

    def as_dict(self) -> dict[str, object]:
        """Return the complete non-secret inference configuration."""

        return {
            "model": self.model,
            "device_requested": self.device,
            "minimum_confidence": self.min_confidence,
            "image_size": self.image_size,
            "iou_threshold": self.iou_threshold,
            "maximum_detections_per_frame": self.max_detections,
            "person_class_id": 0,
        }


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated settings loaded once at an executable boundary."""

    environment: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.JSON
    output_dir: Path = Path("output")
    person_detection: PersonDetectionSettings = field(default_factory=PersonDetectionSettings)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Load settings from a mapping, defaulting to the process environment."""

        source = os.environ if environ is None else environ
        environment_raw = source.get(f"{ENV_PREFIX}ENVIRONMENT", Environment.DEVELOPMENT.value)
        log_level = source.get(f"{ENV_PREFIX}LOG_LEVEL", "INFO").upper()
        log_format_raw = source.get(f"{ENV_PREFIX}LOG_FORMAT", LogFormat.JSON.value)
        output_dir_raw = source.get(f"{ENV_PREFIX}OUTPUT_DIR", "output")

        try:
            environment = Environment(environment_raw.lower())
        except ValueError as error:
            raise ConfigurationError(
                f"Unsupported environment: {environment_raw!r}",
                setting=f"{ENV_PREFIX}ENVIRONMENT",
            ) from error

        if log_level not in VALID_LOG_LEVELS:
            raise ConfigurationError(
                f"Unsupported log level: {log_level!r}",
                setting=f"{ENV_PREFIX}LOG_LEVEL",
            )

        try:
            log_format = LogFormat(log_format_raw.lower())
        except ValueError as error:
            raise ConfigurationError(
                f"Unsupported log format: {log_format_raw!r}",
                setting=f"{ENV_PREFIX}LOG_FORMAT",
            ) from error

        if not output_dir_raw.strip():
            raise ConfigurationError(
                "Output directory cannot be empty",
                setting=f"{ENV_PREFIX}OUTPUT_DIR",
            )

        return cls(
            environment=environment,
            log_level=log_level,
            log_format=log_format,
            output_dir=Path(output_dir_raw).expanduser(),
            person_detection=PersonDetectionSettings.from_env(source),
        )

    def public_values(self) -> dict[str, object]:
        """Return non-secret settings suitable for diagnostics and logs."""

        return {
            "environment": self.environment.value,
            "log_level": self.log_level,
            "log_format": self.log_format.value,
            "output_dir": str(self.output_dir),
            "person_detection": self.person_detection.as_dict(),
        }
