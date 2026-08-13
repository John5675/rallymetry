"""Environment-based application configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pickleball_vision.errors import ConfigurationError

ENV_PREFIX = "PICKLEBALL_VISION_"
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class Environment(StrEnum):
    """Supported runtime environments."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class LogFormat(StrEnum):
    """Supported log output formats."""

    JSON = "json"
    CONSOLE = "console"


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated settings loaded once at an executable boundary."""

    environment: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.JSON
    output_dir: Path = Path("output")

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
        )

    def public_values(self) -> dict[str, str]:
        """Return non-secret settings suitable for diagnostics and logs."""

        return {
            "environment": self.environment.value,
            "log_level": self.log_level,
            "log_format": self.log_format.value,
            "output_dir": str(self.output_dir),
        }
