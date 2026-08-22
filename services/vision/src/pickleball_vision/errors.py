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
    MATCH_ANALYTICS_INPUT = "match_analytics_input_error"
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
    CONTACT_DETECTION_INPUT = "contact_detection_input_error"
    HITTER_IDENTIFICATION_INPUT = "hitter_identification_input_error"
    SHOT_RECONSTRUCTION_INPUT = "shot_reconstruction_input_error"
    SHOT_MODEL_INPUT = "shot_model_input_error"
    SHOT_MODEL_TRAINING = "shot_model_training_error"
    OUTPUT_WRITE = "output_write_error"
    PLAYER_ASSIGNMENT_IO = "player_assignment_io_error"
    PLAYER_ISOLATION_CANCELLED = "player_isolation_cancelled"
    PLAYER_ISOLATION_INPUT = "player_isolation_input_error"
    PLAYER_ANALYSIS_INPUT = "player_analysis_input_error"
    PLAYER_TRACKING_INPUT = "player_tracking_input_error"
    PERSISTENCE = "persistence_error"
    PERSISTENCE_VALIDATION = "persistence_validation_error"
    ARTIFACT_STORAGE = "artifact_storage_error"
    RALLY_SEGMENTATION_INPUT = "rally_segmentation_input_error"
    VIDEO_NOT_FOUND = "video_not_found"
    VIDEO_UNREADABLE = "video_unreadable"
    ANALYSIS_CONFIGURATION = "analysis_configuration_error"
    ANALYSIS_PIPELINE = "analysis_pipeline_error"
    ANALYSIS_SOURCE = "analysis_source_error"


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


class PersistenceValidationError(PickleballVisionError):
    """Raised when a record is unsafe or invalid for hosted persistence."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Invalid persistence record: {reason}",
            code=ErrorCode.PERSISTENCE_VALIDATION,
            details={"reason": reason},
        )


class PersistenceOperationError(PickleballVisionError):
    """Raised when a MongoDB persistence operation fails."""

    def __init__(self, operation: str, *, reason: str) -> None:
        super().__init__(
            f"Persistence operation {operation} failed: {reason}",
            code=ErrorCode.PERSISTENCE,
            details={"operation": operation, "reason": reason},
        )


class ArtifactStorageError(PickleballVisionError):
    """Raised when an artifact-store operation fails."""

    def __init__(self, operation: str, *, reason: str, pathname: str | None = None) -> None:
        details: dict[str, object] = {"operation": operation, "reason": reason}
        if pathname is not None:
            details["pathname"] = pathname
        super().__init__(
            f"Artifact storage operation {operation} failed: {reason}",
            code=ErrorCode.ARTIFACT_STORAGE,
            details=details,
        )


class AnalysisExecutionError(PickleballVisionError):
    """Expected failure recorded on an on-demand analysis job."""

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode,
        job_error_code: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)
        self.job_error_code = job_error_code


class AnalysisConfigurationError(AnalysisExecutionError):
    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Invalid analysis configuration: {reason}",
            code=ErrorCode.ANALYSIS_CONFIGURATION,
            job_error_code="ANALYSIS_CONFIGURATION",
            details={"reason": reason},
        )


class AnalysisSourceError(AnalysisExecutionError):
    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Unable to stage source media: {reason}",
            code=ErrorCode.ANALYSIS_SOURCE,
            job_error_code="ANALYSIS_SOURCE",
            details={"reason": reason},
        )


class AnalysisPipelineError(AnalysisExecutionError):
    def __init__(self, reason: str, *, stage: str | None = None) -> None:
        details: dict[str, object] = {"reason": reason}
        if stage is not None:
            details["stage"] = stage
        super().__init__(
            f"Analysis pipeline failed: {reason}",
            code=ErrorCode.ANALYSIS_PIPELINE,
            job_error_code="ANALYSIS_PIPELINE",
            details=details,
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


class ShotModelInputError(PickleballVisionError):
    """Raised when shot-model data cannot preserve split or provenance contracts."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Invalid shot-model input: {reason}",
            code=ErrorCode.SHOT_MODEL_INPUT,
            details={"reason": reason},
        )


class ShotModelTrainingError(PickleballVisionError):
    """Raised when a shot-model experiment cannot train or pretrain safely."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Shot-model training failed: {reason}",
            code=ErrorCode.SHOT_MODEL_TRAINING,
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


class ContactDetectionInputError(PickleballVisionError):
    """Raised when contact-detection artifacts are incompatible or malformed."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Invalid paddle-contact detection input: {reason}",
            code=ErrorCode.CONTACT_DETECTION_INPUT,
            details={"reason": reason},
        )


class HitterIdentificationInputError(PickleballVisionError):
    """Raised when hitter-identification artifacts are incompatible or malformed."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Invalid hitter-identification input: {reason}",
            code=ErrorCode.HITTER_IDENTIFICATION_INPUT,
            details={"reason": reason},
        )


class ShotReconstructionInputError(PickleballVisionError):
    """Raised when shot reconstruction artifacts are incompatible or malformed."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Invalid shot-reconstruction input: {reason}",
            code=ErrorCode.SHOT_RECONSTRUCTION_INPUT,
            details={"reason": reason},
        )


class MatchAnalyticsInputError(PickleballVisionError):
    """Raised when deterministic analytics artifacts are incompatible or malformed."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Invalid match-analytics input: {reason}",
            code=ErrorCode.MATCH_ANALYTICS_INPUT,
            details={"reason": reason},
        )
