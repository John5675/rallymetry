"""Typed errors exposed by the vision service."""

from collections.abc import Mapping
from enum import StrEnum


class ErrorCode(StrEnum):
    """Stable machine-readable application error codes."""

    CALIBRATION_CANCELLED = "calibration_cancelled"
    CALIBRATION_INVALID = "calibration_invalid"
    CALIBRATION_IO = "calibration_io_error"
    CONFIGURATION = "configuration_error"
    DATASET_INPUT = "dataset_input_error"
    DATASET_IO = "dataset_io_error"
    DETECTION_INPUT = "detection_input_error"
    DETECTION_IO = "detection_io_error"
    DETECTION_MODEL = "detection_model_error"
    FRAME_DECODE = "frame_decode_error"
    CLIP_EXTRACTION = "clip_extraction_error"
    INTERNAL = "internal_error"
    INVALID_SAMPLE_COUNT = "invalid_sample_count"
    INVALID_FRAME_INDEX = "invalid_frame_index"
    INVALID_TIMESTAMP = "invalid_timestamp"
    MEDIA_INSPECTION = "media_inspection_error"
    MATCH_ANNOTATION_INPUT = "match_annotation_input_error"
    MATCH_ANNOTATION_IO = "match_annotation_io_error"
    AUDIO_STREAM_NOT_FOUND = "audio_stream_not_found"
    AUDIO_EXTRACTION = "audio_extraction_error"
    AUDIO_ANALYSIS = "audio_analysis_error"
    AUDIO_CONVERSION_INVALID = "audio_conversion_invalid"
    BALL_EVALUATION = "ball_evaluation_error"
    BALL_INFERENCE = "ball_inference_error"
    BALL_MODEL = "ball_model_error"
    BALL_TRACKING_INPUT = "ball_tracking_input_error"
    BALL_TRAINING = "ball_training_error"
    BOUNCE_DETECTION_INPUT = "bounce_detection_input_error"
    OUTPUT_WRITE = "output_write_error"
    PLAYER_ASSIGNMENT_IO = "player_assignment_io_error"
    PLAYER_ISOLATION_CANCELLED = "player_isolation_cancelled"
    PLAYER_ISOLATION_INPUT = "player_isolation_input_error"
    PLAYER_ANALYSIS_INPUT = "player_analysis_input_error"
    PLAYER_TRACKING_INPUT = "player_tracking_input_error"
    RALLY_SEGMENTATION_INPUT = "rally_segmentation_input_error"
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


class MediaInspectionError(PickleballVisionError):
    """Raised when the FFmpeg media boundary cannot inspect source streams."""

    def __init__(self, path: str, *, reason: str) -> None:
        super().__init__(
            f"Unable to inspect media streams in {path}: {reason}",
            code=ErrorCode.MEDIA_INSPECTION,
            details={"path": path, "reason": reason},
        )


class MatchAnnotationInputError(PickleballVisionError):
    """Raised when match ground-truth annotation data is invalid."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Invalid match annotation: {reason}",
            code=ErrorCode.MATCH_ANNOTATION_INPUT,
            details={"reason": reason},
        )


class MatchAnnotationIoError(PickleballVisionError):
    """Raised when match ground-truth annotation data cannot be persisted."""

    def __init__(self, path: str, *, reason: str) -> None:
        super().__init__(
            f"Unable to access match annotation {path}: {reason}",
            code=ErrorCode.MATCH_ANNOTATION_IO,
            details={"path": path, "reason": reason},
        )


class AudioStreamNotFoundError(PickleballVisionError):
    """Raised when audio is requested from valid video-only media."""

    def __init__(self, path: str) -> None:
        super().__init__(
            f"Media file has no audio stream: {path}",
            code=ErrorCode.AUDIO_STREAM_NOT_FOUND,
            details={"path": path},
        )


class InvalidAudioConversionError(PickleballVisionError):
    """Raised when an explicit audio conversion request is unsupported."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Invalid audio conversion: {reason}",
            code=ErrorCode.AUDIO_CONVERSION_INVALID,
            details={"reason": reason},
        )


class AudioExtractionError(PickleballVisionError):
    """Raised when synchronized PCM audio cannot be extracted or recorded."""

    def __init__(self, path: str, *, reason: str) -> None:
        super().__init__(
            f"Unable to extract audio to {path}: {reason}",
            code=ErrorCode.AUDIO_EXTRACTION,
            details={"path": path, "reason": reason},
        )


class AudioAnalysisError(PickleballVisionError):
    """Raised when synchronized audio cannot be analyzed or persisted safely."""

    def __init__(self, reason: str, *, operation: str) -> None:
        super().__init__(
            f"Audio analysis failed during {operation}: {reason}",
            code=ErrorCode.AUDIO_ANALYSIS,
            details={"operation": operation, "reason": reason},
        )


class ClipExtractionError(PickleballVisionError):
    """Raised when a synchronized review clip cannot be created safely."""

    def __init__(self, path: str, *, reason: str) -> None:
        super().__init__(
            f"Unable to extract media clip to {path}: {reason}",
            code=ErrorCode.CLIP_EXTRACTION,
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


class InvalidFrameIndexError(PickleballVisionError):
    """Raised when requested source-frame indices are invalid or unordered."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Invalid frame index selection: {reason}",
            code=ErrorCode.INVALID_FRAME_INDEX,
            details={"reason": reason},
        )


class FrameDecodeError(PickleballVisionError):
    """Raised when OpenCV cannot decode a requested frame."""

    def __init__(self, path: str, *, frame_index: int) -> None:
        super().__init__(
            f"Unable to decode frame {frame_index} from video: {path}",
            code=ErrorCode.FRAME_DECODE,
            details={"path": path, "frame_index": frame_index},
        )


class DatasetInputError(PickleballVisionError):
    """Raised when dataset selection, clip, or split inputs are invalid."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Invalid dataset input: {reason}",
            code=ErrorCode.DATASET_INPUT,
            details={"reason": reason},
        )


class DatasetIoError(PickleballVisionError):
    """Raised when a dataset artifact or manifest cannot be read or written."""

    def __init__(self, path: str, *, reason: str) -> None:
        super().__init__(
            f"Unable to access dataset artifact {path}: {reason}",
            code=ErrorCode.DATASET_IO,
            details={"path": path, "reason": reason},
        )


class OutputWriteError(PickleballVisionError):
    """Raised when a generated artifact cannot be written."""

    def __init__(self, path: str, *, reason: str) -> None:
        super().__init__(
            f"Unable to write output {path}: {reason}",
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


class DetectionInputError(PickleballVisionError):
    """Raised when detection inputs are individually valid but incompatible."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Invalid person-detection input: {reason}",
            code=ErrorCode.DETECTION_INPUT,
            details={"reason": reason},
        )


class DetectionModelError(PickleballVisionError):
    """Raised when a pretrained detector cannot load or infer safely."""

    def __init__(self, reason: str, *, operation: str) -> None:
        super().__init__(
            f"Person detector failed during {operation}: {reason}",
            code=ErrorCode.DETECTION_MODEL,
            details={"operation": operation, "reason": reason},
        )


class DetectionIoError(PickleballVisionError):
    """Raised when persisted raw person detections cannot be loaded."""

    def __init__(self, path: str, *, reason: str) -> None:
        super().__init__(
            f"Unable to load person detections {path}: {reason}",
            code=ErrorCode.DETECTION_IO,
            details={"path": path, "reason": reason},
        )


class BallModelError(PickleballVisionError):
    """Raised when custom pickleball model loading or inference fails."""

    def __init__(self, reason: str, *, operation: str) -> None:
        super().__init__(
            f"Pickleball detector failed during {operation}: {reason}",
            code=ErrorCode.BALL_MODEL,
            details={"operation": operation, "reason": reason},
        )


class BallInferenceError(PickleballVisionError):
    """Raised when spatial ball inference cannot preserve its contract."""

    def __init__(self, reason: str, *, operation: str) -> None:
        super().__init__(
            f"Pickleball inference failed during {operation}: {reason}",
            code=ErrorCode.BALL_INFERENCE,
            details={"operation": operation, "reason": reason},
        )


class BallTrainingError(PickleballVisionError):
    """Raised when a custom model experiment cannot complete safely."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Pickleball detector training failed: {reason}",
            code=ErrorCode.BALL_TRAINING,
            details={"reason": reason},
        )


class BallEvaluationError(PickleballVisionError):
    """Raised when fixed-split detector evaluation cannot complete safely."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Pickleball detector evaluation failed: {reason}",
            code=ErrorCode.BALL_EVALUATION,
            details={"reason": reason},
        )


class BallTrackingInputError(PickleballVisionError):
    """Raised when raw detections, calibration, or video cannot be tracked safely."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Invalid ball-trajectory reconstruction input: {reason}",
            code=ErrorCode.BALL_TRACKING_INPUT,
            details={"reason": reason},
        )


class PlayerIsolationInputError(PickleballVisionError):
    """Raised when isolation inputs or derived assignments are invalid."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Invalid primary-player isolation input: {reason}",
            code=ErrorCode.PLAYER_ISOLATION_INPUT,
            details={"reason": reason},
        )


class PlayerIsolationCancelledError(PickleballVisionError):
    """Raised when manual logical-player selection is cancelled."""

    def __init__(self) -> None:
        super().__init__(
            "Primary-player selection was cancelled; no assignments were written",
            code=ErrorCode.PLAYER_ISOLATION_CANCELLED,
        )


class PlayerAssignmentIoError(PickleballVisionError):
    """Raised when logical-player assignments cannot be read or written."""

    def __init__(self, path: str, *, reason: str) -> None:
        super().__init__(
            f"Unable to access logical-player assignments {path}: {reason}",
            code=ErrorCode.PLAYER_ASSIGNMENT_IO,
            details={"path": path, "reason": reason},
        )


class PlayerTrackingInputError(PickleballVisionError):
    """Raised when tracking artifacts are missing, incompatible, or unusable."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Invalid persistent-player tracking input: {reason}",
            code=ErrorCode.PLAYER_TRACKING_INPUT,
            details={"reason": reason},
        )


class PlayerAnalysisInputError(PickleballVisionError):
    """Raised when player-position analysis inputs are incompatible or unusable."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Invalid player-position analysis input: {reason}",
            code=ErrorCode.PLAYER_ANALYSIS_INPUT,
            details={"reason": reason},
        )


class RallySegmentationInputError(PickleballVisionError):
    """Raised when rally segmentation artifacts are incompatible or malformed."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Invalid rally-segmentation input: {reason}",
            code=ErrorCode.RALLY_SEGMENTATION_INPUT,
            details={"reason": reason},
        )


class BounceDetectionInputError(PickleballVisionError):
    """Raised when bounce-detection artifacts are incompatible or malformed."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Invalid bounce-detection input: {reason}",
            code=ErrorCode.BOUNCE_DETECTION_INPUT,
            details={"reason": reason},
        )
