"""Typed errors exposed by the vision service."""

from collections.abc import Mapping
from enum import StrEnum


class ErrorCode(StrEnum):
    """Stable machine-readable application error codes."""

    CALIBRATION_CANCELLED = "calibration_cancelled"
    CALIBRATION_INVALID = "calibration_invalid"
    CALIBRATION_IO = "calibration_io_error"
    CONFIGURATION = "configuration_error"
    FRAME_DECODE = "frame_decode_error"
    INTERNAL = "internal_error"
    INVALID_SAMPLE_COUNT = "invalid_sample_count"
    INVALID_TIMESTAMP = "invalid_timestamp"
    OUTPUT_WRITE = "output_write_error"
    VIDEO_NOT_FOUND = "video_not_found"
    VIDEO_UNREADABLE = "video_unreadable"


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


class VideoNotFoundError(PickleballVisionError):
    """Raised when a requested local video path does not exist."""

    def __init__(self, path: str) -> None:
        super().__init__(
            f"Video file does not exist: {path}",
            code=ErrorCode.VIDEO_NOT_FOUND,
            details={"path": path},
        )


class VideoUnreadableError(PickleballVisionError):
    """Raised when OpenCV cannot open or validate a local video."""

    def __init__(self, path: str, *, reason: str) -> None:
        super().__init__(
            f"Unable to read video {path}: {reason}",
            code=ErrorCode.VIDEO_UNREADABLE,
            details={"path": path, "reason": reason},
        )


class InvalidTimestampError(PickleballVisionError):
    """Raised when a timestamp is outside a video's valid time span."""

    def __init__(self, timestamp_seconds: float, *, duration_seconds: float) -> None:
        super().__init__(
            (
                f"Timestamp must be finite and in the range [0, {duration_seconds:.6f}) "
                f"seconds; received {timestamp_seconds!r}"
            ),
            code=ErrorCode.INVALID_TIMESTAMP,
            details={
                "timestamp_seconds": timestamp_seconds,
                "duration_seconds": duration_seconds,
            },
        )


class InvalidSampleCountError(PickleballVisionError):
    """Raised when a requested sample count cannot produce unique frames."""

    def __init__(self, count: int, *, frame_count: int) -> None:
        super().__init__(
            f"Sample count must be between 1 and {frame_count}; received {count}",
            code=ErrorCode.INVALID_SAMPLE_COUNT,
            details={"count": count, "frame_count": frame_count},
        )


class FrameDecodeError(PickleballVisionError):
    """Raised when OpenCV cannot decode a requested frame."""

    def __init__(self, path: str, *, frame_index: int) -> None:
        super().__init__(
            f"Unable to decode frame {frame_index} from video: {path}",
            code=ErrorCode.FRAME_DECODE,
            details={"path": path, "frame_index": frame_index},
        )


class OutputWriteError(PickleballVisionError):
    """Raised when an extracted frame cannot be written."""

    def __init__(self, path: str, *, reason: str) -> None:
        super().__init__(
            f"Unable to write frame image {path}: {reason}",
            code=ErrorCode.OUTPUT_WRITE,
            details={"path": path, "reason": reason},
        )


class InvalidCalibrationError(PickleballVisionError):
    """Raised when court correspondences cannot define a valid homography."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Invalid court calibration: {reason}",
            code=ErrorCode.CALIBRATION_INVALID,
            details={"reason": reason},
        )


class CalibrationCancelledError(PickleballVisionError):
    """Raised when a user cancels manual landmark selection."""

    def __init__(self) -> None:
        super().__init__(
            "Court calibration was cancelled; no calibration was written",
            code=ErrorCode.CALIBRATION_CANCELLED,
        )


class CalibrationIoError(PickleballVisionError):
    """Raised when calibration JSON cannot be read or written."""

    def __init__(self, path: str, *, reason: str) -> None:
        super().__init__(
            f"Unable to access calibration {path}: {reason}",
            code=ErrorCode.CALIBRATION_IO,
            details={"path": path, "reason": reason},
        )
