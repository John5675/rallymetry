"""Typed errors exposed by the vision service."""

from collections.abc import Mapping
from enum import StrEnum


class ErrorCode(StrEnum):
    """Stable machine-readable application error codes."""

    CONFIGURATION = "configuration_error"
    INTERNAL = "internal_error"


class PickleballVisionError(Exception):
    """Base class for expected, user-actionable application failures."""

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


class ConfigurationError(PickleballVisionError):
    """Raised when application configuration is invalid."""

    def __init__(self, message: str, *, setting: str) -> None:
        super().__init__(
            message,
            code=ErrorCode.CONFIGURATION,
            details={"setting": setting},
        )
