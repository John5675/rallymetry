"""Environment-based application configuration."""

from __future__ import annotations

import math
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


class AudioAnalysisChannelMode(StrEnum):
    """How onset evidence is combined while retaining every source channel."""

    COMBINED = "combined"
    PER_CHANNEL = "per_channel"


@dataclass(frozen=True, slots=True)
class MediaSettings:
    """Canonical media-timeline settings shared by future A/V fusion stages."""

    audio_video_offset_ms: float = 0.0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> MediaSettings:
        """Load the configured correction applied to audio timestamps."""

        source = os.environ if environ is None else environ
        offset_ms = _float_setting(source, "AUDIO_VIDEO_OFFSET_MS", cls().audio_video_offset_ms)
        if not math.isfinite(offset_ms):
            setting = f"{ENV_PREFIX}AUDIO_VIDEO_OFFSET_MS"
            raise ConfigurationError(f"{setting} must be finite", setting=setting)
        return cls(audio_video_offset_ms=offset_ms)

    def as_dict(self) -> dict[str, object]:
        """Return non-secret timing configuration for provenance."""

        return {"audioVideoOffsetMs": self.audio_video_offset_ms}


@dataclass(frozen=True, slots=True)
class AudioAnalysisSettings:
    """Validated signal-window and generic transient-detection settings."""

    analysis_sample_rate_hz: int = 16000
    onset_sensitivity: float = 4.0
    minimum_event_separation_ms: float = 80.0
    channel_mode: AudioAnalysisChannelMode = AudioAnalysisChannelMode.COMBINED
    frame_duration_ms: float = 32.0
    hop_duration_ms: float = 10.0

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> AudioAnalysisSettings:
        """Load audio analysis configuration from prefixed environment variables."""

        source = os.environ if environ is None else environ
        defaults = cls()
        sample_rate_hz = _int_setting(
            source,
            "AUDIO_ANALYSIS_SAMPLE_RATE_HZ",
            defaults.analysis_sample_rate_hz,
        )
        sensitivity = _float_setting(
            source,
            "AUDIO_ANALYSIS_ONSET_SENSITIVITY",
            defaults.onset_sensitivity,
        )
        separation_ms = _float_setting(
            source,
            "AUDIO_ANALYSIS_MINIMUM_EVENT_SEPARATION_MS",
            defaults.minimum_event_separation_ms,
        )
        frame_duration_ms = _float_setting(
            source,
            "AUDIO_ANALYSIS_FRAME_DURATION_MS",
            defaults.frame_duration_ms,
        )
        hop_duration_ms = _float_setting(
            source,
            "AUDIO_ANALYSIS_HOP_DURATION_MS",
            defaults.hop_duration_ms,
        )
        channel_mode_raw = source.get(
            f"{ENV_PREFIX}AUDIO_ANALYSIS_CHANNEL_MODE",
            defaults.channel_mode.value,
        )
        try:
            channel_mode = AudioAnalysisChannelMode(channel_mode_raw.strip().lower())
        except ValueError as error:
            setting = f"{ENV_PREFIX}AUDIO_ANALYSIS_CHANNEL_MODE"
            raise ConfigurationError(
                f"{setting} must be combined or per_channel",
                setting=setting,
            ) from error
        validations = (
            (
                sample_rate_hz >= 8000,
                "AUDIO_ANALYSIS_SAMPLE_RATE_HZ",
                "must be at least 8000",
            ),
            (
                sensitivity > 0 and math.isfinite(sensitivity),
                "AUDIO_ANALYSIS_ONSET_SENSITIVITY",
                "must be finite and positive",
            ),
            (
                separation_ms > 0 and math.isfinite(separation_ms),
                "AUDIO_ANALYSIS_MINIMUM_EVENT_SEPARATION_MS",
                "must be finite and positive",
            ),
            (
                frame_duration_ms > 0 and math.isfinite(frame_duration_ms),
                "AUDIO_ANALYSIS_FRAME_DURATION_MS",
                "must be finite and positive",
            ),
            (
                hop_duration_ms > 0 and math.isfinite(hop_duration_ms),
                "AUDIO_ANALYSIS_HOP_DURATION_MS",
                "must be finite and positive",
            ),
            (
                hop_duration_ms <= frame_duration_ms,
                "AUDIO_ANALYSIS_HOP_DURATION_MS",
                "must not exceed AUDIO_ANALYSIS_FRAME_DURATION_MS",
            ),
        )
        for valid, suffix, reason in validations:
            if not valid:
                setting = f"{ENV_PREFIX}{suffix}"
                raise ConfigurationError(f"{setting} {reason}", setting=setting)
        return cls(
            analysis_sample_rate_hz=sample_rate_hz,
            onset_sensitivity=sensitivity,
            minimum_event_separation_ms=separation_ms,
            channel_mode=channel_mode,
            frame_duration_ms=frame_duration_ms,
            hop_duration_ms=hop_duration_ms,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "analysisSampleRateHz": self.analysis_sample_rate_hz,
            "onsetSensitivity": self.onset_sensitivity,
            "minimumEventSeparationMs": self.minimum_event_separation_ms,
            "channelMode": self.channel_mode.value,
            "frameDurationMs": self.frame_duration_ms,
            "hopDurationMs": self.hop_duration_ms,
        }


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
class PlayerIsolationSettings:
    """Validated geometry and short-gap candidate-selection settings."""

    near_court_margin_m: float = 1.5
    boundary_uncertainty_m: float = 0.25
    side_uncertainty_m: float = 0.25
    max_candidate_gap_s: float = 1.0
    max_candidate_speed_mps: float = 8.0
    min_candidate_observations: int = 15
    min_court_support_ratio: float = 0.60

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> PlayerIsolationSettings:
        """Load candidate selection settings from the shared environment boundary."""

        source = os.environ if environ is None else environ
        defaults = cls()
        near_court_margin_m = _float_setting(
            source,
            "ISOLATION_NEAR_MARGIN_METERS",
            defaults.near_court_margin_m,
        )
        boundary_uncertainty_m = _float_setting(
            source,
            "ISOLATION_BOUNDARY_UNCERTAINTY_METERS",
            defaults.boundary_uncertainty_m,
        )
        side_uncertainty_m = _float_setting(
            source,
            "ISOLATION_SIDE_UNCERTAINTY_METERS",
            defaults.side_uncertainty_m,
        )
        max_candidate_gap_s = _float_setting(
            source,
            "ISOLATION_MAX_CANDIDATE_GAP_SECONDS",
            defaults.max_candidate_gap_s,
        )
        max_candidate_speed_mps = _float_setting(
            source,
            "ISOLATION_MAX_CANDIDATE_SPEED_MPS",
            defaults.max_candidate_speed_mps,
        )
        min_candidate_observations = _int_setting(
            source,
            "ISOLATION_MIN_CANDIDATE_OBSERVATIONS",
            defaults.min_candidate_observations,
        )
        min_court_support_ratio = _float_setting(
            source,
            "ISOLATION_MIN_COURT_SUPPORT_RATIO",
            defaults.min_court_support_ratio,
        )
        validations = (
            (near_court_margin_m > 0, "ISOLATION_NEAR_MARGIN_METERS", "must be positive"),
            (
                boundary_uncertainty_m > 0,
                "ISOLATION_BOUNDARY_UNCERTAINTY_METERS",
                "must be positive",
            ),
            (
                side_uncertainty_m > 0,
                "ISOLATION_SIDE_UNCERTAINTY_METERS",
                "must be positive",
            ),
            (
                max_candidate_gap_s > 0,
                "ISOLATION_MAX_CANDIDATE_GAP_SECONDS",
                "must be positive",
            ),
            (
                max_candidate_speed_mps > 0,
                "ISOLATION_MAX_CANDIDATE_SPEED_MPS",
                "must be positive",
            ),
            (
                min_candidate_observations >= 2,
                "ISOLATION_MIN_CANDIDATE_OBSERVATIONS",
                "must be at least 2",
            ),
            (
                0 <= min_court_support_ratio <= 1,
                "ISOLATION_MIN_COURT_SUPPORT_RATIO",
                "must be between 0 and 1 inclusive",
            ),
        )
        for valid, suffix, reason in validations:
            if not valid:
                setting = f"{ENV_PREFIX}{suffix}"
                raise ConfigurationError(f"{setting} {reason}", setting=setting)
        return cls(
            near_court_margin_m=near_court_margin_m,
            boundary_uncertainty_m=boundary_uncertainty_m,
            side_uncertainty_m=side_uncertainty_m,
            max_candidate_gap_s=max_candidate_gap_s,
            max_candidate_speed_mps=max_candidate_speed_mps,
            min_candidate_observations=min_candidate_observations,
            min_court_support_ratio=min_court_support_ratio,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "near_court_margin_m": self.near_court_margin_m,
            "boundary_uncertainty_m": self.boundary_uncertainty_m,
            "side_uncertainty_m": self.side_uncertainty_m,
            "max_candidate_gap_s": self.max_candidate_gap_s,
            "max_candidate_speed_mps": self.max_candidate_speed_mps,
            "min_candidate_observations": self.min_candidate_observations,
            "min_court_support_ratio": self.min_court_support_ratio,
        }


@dataclass(frozen=True, slots=True)
class PlayerTrackingSettings:
    """Validated ByteTrack and conservative logical-identity settings."""

    track_high_threshold: float = 0.25
    track_low_threshold: float = 0.10
    new_track_threshold: float = 0.25
    match_threshold: float = 0.80
    track_buffer_seconds: float = 1.0
    max_identity_gap_seconds: float = 3.0
    max_player_speed_mps: float = 8.0
    minimum_identity_score: float = 0.45
    suspected_switch_score: float = 0.65
    appearance_weight: float = 0.47
    minimum_appearance_similarity: float = 0.55
    minimum_appearance_margin: float = -0.05
    appearance_prototype_window_seconds: float = 3.0
    long_gap_appearance_similarity: float = 0.70
    long_gap_minimum_appearance_margin: float = 0.03

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> PlayerTrackingSettings:
        """Load tracker and resolver settings from the shared environment boundary."""

        source = os.environ if environ is None else environ
        defaults = cls()
        values = {
            "track_high_threshold": _float_setting(
                source, "TRACKING_HIGH_THRESHOLD", defaults.track_high_threshold
            ),
            "track_low_threshold": _float_setting(
                source, "TRACKING_LOW_THRESHOLD", defaults.track_low_threshold
            ),
            "new_track_threshold": _float_setting(
                source, "TRACKING_NEW_THRESHOLD", defaults.new_track_threshold
            ),
            "match_threshold": _float_setting(
                source, "TRACKING_MATCH_THRESHOLD", defaults.match_threshold
            ),
            "track_buffer_seconds": _float_setting(
                source, "TRACKING_BUFFER_SECONDS", defaults.track_buffer_seconds
            ),
            "max_identity_gap_seconds": _float_setting(
                source, "TRACKING_MAX_IDENTITY_GAP_SECONDS", defaults.max_identity_gap_seconds
            ),
            "max_player_speed_mps": _float_setting(
                source, "TRACKING_MAX_PLAYER_SPEED_MPS", defaults.max_player_speed_mps
            ),
            "minimum_identity_score": _float_setting(
                source, "TRACKING_MINIMUM_IDENTITY_SCORE", defaults.minimum_identity_score
            ),
            "suspected_switch_score": _float_setting(
                source, "TRACKING_SUSPECTED_SWITCH_SCORE", defaults.suspected_switch_score
            ),
            "appearance_weight": _float_setting(
                source, "TRACKING_APPEARANCE_WEIGHT", defaults.appearance_weight
            ),
            "minimum_appearance_similarity": _float_setting(
                source,
                "TRACKING_MINIMUM_APPEARANCE_SIMILARITY",
                defaults.minimum_appearance_similarity,
            ),
            "minimum_appearance_margin": _float_setting(
                source,
                "TRACKING_MINIMUM_APPEARANCE_MARGIN",
                defaults.minimum_appearance_margin,
            ),
            "appearance_prototype_window_seconds": _float_setting(
                source,
                "TRACKING_APPEARANCE_PROTOTYPE_WINDOW_SECONDS",
                defaults.appearance_prototype_window_seconds,
            ),
            "long_gap_appearance_similarity": _float_setting(
                source,
                "TRACKING_LONG_GAP_APPEARANCE_SIMILARITY",
                defaults.long_gap_appearance_similarity,
            ),
            "long_gap_minimum_appearance_margin": _float_setting(
                source,
                "TRACKING_LONG_GAP_MINIMUM_APPEARANCE_MARGIN",
                defaults.long_gap_minimum_appearance_margin,
            ),
        }
        validations = (
            (0 <= values["track_low_threshold"] <= 1, "TRACKING_LOW_THRESHOLD"),
            (0 <= values["track_high_threshold"] <= 1, "TRACKING_HIGH_THRESHOLD"),
            (0 <= values["new_track_threshold"] <= 1, "TRACKING_NEW_THRESHOLD"),
            (0 <= values["match_threshold"] <= 1, "TRACKING_MATCH_THRESHOLD"),
            (values["track_buffer_seconds"] > 0, "TRACKING_BUFFER_SECONDS"),
            (values["max_identity_gap_seconds"] > 0, "TRACKING_MAX_IDENTITY_GAP_SECONDS"),
            (values["max_player_speed_mps"] > 0, "TRACKING_MAX_PLAYER_SPEED_MPS"),
            (0 <= values["minimum_identity_score"] <= 1, "TRACKING_MINIMUM_IDENTITY_SCORE"),
            (0 <= values["suspected_switch_score"] <= 1, "TRACKING_SUSPECTED_SWITCH_SCORE"),
            (0 <= values["appearance_weight"] <= 1, "TRACKING_APPEARANCE_WEIGHT"),
            (
                0 <= values["minimum_appearance_similarity"] <= 1,
                "TRACKING_MINIMUM_APPEARANCE_SIMILARITY",
            ),
            (
                -1 <= values["minimum_appearance_margin"] <= 1,
                "TRACKING_MINIMUM_APPEARANCE_MARGIN",
            ),
            (
                values["appearance_prototype_window_seconds"] > 0,
                "TRACKING_APPEARANCE_PROTOTYPE_WINDOW_SECONDS",
            ),
            (
                0 <= values["long_gap_appearance_similarity"] <= 1,
                "TRACKING_LONG_GAP_APPEARANCE_SIMILARITY",
            ),
            (
                -1 <= values["long_gap_minimum_appearance_margin"] <= 1,
                "TRACKING_LONG_GAP_MINIMUM_APPEARANCE_MARGIN",
            ),
        )
        for valid, suffix in validations:
            if not valid:
                setting = f"{ENV_PREFIX}{suffix}"
                raise ConfigurationError(
                    f"{setting} must be positive or between 0 and 1 as appropriate",
                    setting=setting,
                )
        if values["track_low_threshold"] > values["track_high_threshold"]:
            setting = f"{ENV_PREFIX}TRACKING_LOW_THRESHOLD"
            raise ConfigurationError(
                f"{setting} must not exceed {ENV_PREFIX}TRACKING_HIGH_THRESHOLD",
                setting=setting,
            )
        return cls(**values)

    def as_dict(self) -> dict[str, object]:
        return {
            "track_high_threshold": self.track_high_threshold,
            "track_low_threshold": self.track_low_threshold,
            "new_track_threshold": self.new_track_threshold,
            "match_threshold": self.match_threshold,
            "track_buffer_seconds": self.track_buffer_seconds,
            "max_identity_gap_seconds": self.max_identity_gap_seconds,
            "max_player_speed_mps": self.max_player_speed_mps,
            "minimum_identity_score": self.minimum_identity_score,
            "suspected_switch_score": self.suspected_switch_score,
            "appearance_weight": self.appearance_weight,
            "minimum_appearance_similarity": self.minimum_appearance_similarity,
            "minimum_appearance_margin": self.minimum_appearance_margin,
            "appearance_prototype_window_seconds": self.appearance_prototype_window_seconds,
            "long_gap_appearance_similarity": self.long_gap_appearance_similarity,
            "long_gap_minimum_appearance_margin": self.long_gap_minimum_appearance_margin,
        }


@dataclass(frozen=True, slots=True)
class PlayerAnalysisSettings:
    """Validated Release 0.1 smoothing and metric quality gates."""

    minimum_tracking_confidence: float = 0.45
    smoothing_window_frames: int = 5
    maximum_smoothing_adjustment_m: float = 0.30
    maximum_step_gap_seconds: float = 0.20
    maximum_step_speed_mps: float = 8.0
    transition_zone_depth_m: float = 2.1336
    topdown_trail_seconds: float = 2.0

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> PlayerAnalysisSettings:
        """Load player-position analysis settings from prefixed variables."""

        source = os.environ if environ is None else environ
        defaults = cls()
        values: dict[str, float | int] = {
            "minimum_tracking_confidence": _float_setting(
                source,
                "ANALYSIS_MINIMUM_TRACKING_CONFIDENCE",
                defaults.minimum_tracking_confidence,
            ),
            "smoothing_window_frames": _int_setting(
                source,
                "ANALYSIS_SMOOTHING_WINDOW_FRAMES",
                defaults.smoothing_window_frames,
            ),
            "maximum_smoothing_adjustment_m": _float_setting(
                source,
                "ANALYSIS_MAXIMUM_SMOOTHING_ADJUSTMENT_METERS",
                defaults.maximum_smoothing_adjustment_m,
            ),
            "maximum_step_gap_seconds": _float_setting(
                source,
                "ANALYSIS_MAXIMUM_STEP_GAP_SECONDS",
                defaults.maximum_step_gap_seconds,
            ),
            "maximum_step_speed_mps": _float_setting(
                source,
                "ANALYSIS_MAXIMUM_STEP_SPEED_MPS",
                defaults.maximum_step_speed_mps,
            ),
            "transition_zone_depth_m": _float_setting(
                source,
                "ANALYSIS_TRANSITION_ZONE_DEPTH_METERS",
                defaults.transition_zone_depth_m,
            ),
            "topdown_trail_seconds": _float_setting(
                source,
                "ANALYSIS_TOPDOWN_TRAIL_SECONDS",
                defaults.topdown_trail_seconds,
            ),
        }
        confidence = float(values["minimum_tracking_confidence"])
        window = int(values["smoothing_window_frames"])
        validations = (
            (
                0 <= confidence <= 1,
                "ANALYSIS_MINIMUM_TRACKING_CONFIDENCE",
                "must be between 0 and 1 inclusive",
            ),
            (
                window >= 3 and window % 2 == 1,
                "ANALYSIS_SMOOTHING_WINDOW_FRAMES",
                "must be an odd integer of at least 3",
            ),
            (
                float(values["maximum_smoothing_adjustment_m"]) > 0,
                "ANALYSIS_MAXIMUM_SMOOTHING_ADJUSTMENT_METERS",
                "must be positive",
            ),
            (
                float(values["maximum_step_gap_seconds"]) > 0,
                "ANALYSIS_MAXIMUM_STEP_GAP_SECONDS",
                "must be positive",
            ),
            (
                float(values["maximum_step_speed_mps"]) > 0,
                "ANALYSIS_MAXIMUM_STEP_SPEED_MPS",
                "must be positive",
            ),
            (
                float(values["transition_zone_depth_m"]) > 0,
                "ANALYSIS_TRANSITION_ZONE_DEPTH_METERS",
                "must be positive",
            ),
            (
                float(values["topdown_trail_seconds"]) > 0,
                "ANALYSIS_TOPDOWN_TRAIL_SECONDS",
                "must be positive",
            ),
        )
        for valid, suffix, reason in validations:
            if not valid:
                setting = f"{ENV_PREFIX}{suffix}"
                raise ConfigurationError(f"{setting} {reason}", setting=setting)
        return cls(
            minimum_tracking_confidence=confidence,
            smoothing_window_frames=window,
            maximum_smoothing_adjustment_m=float(values["maximum_smoothing_adjustment_m"]),
            maximum_step_gap_seconds=float(values["maximum_step_gap_seconds"]),
            maximum_step_speed_mps=float(values["maximum_step_speed_mps"]),
            transition_zone_depth_m=float(values["transition_zone_depth_m"]),
            topdown_trail_seconds=float(values["topdown_trail_seconds"]),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "minimum_tracking_confidence": self.minimum_tracking_confidence,
            "smoothing": {
                "method": "centered_confidence_weighted_component_median",
                "window_frames": self.smoothing_window_frames,
                "maximum_adjustment_m": self.maximum_smoothing_adjustment_m,
                "interpolates_missing_frames": False,
            },
            "maximum_step_gap_seconds": self.maximum_step_gap_seconds,
            "maximum_step_speed_mps": self.maximum_step_speed_mps,
            "transition_zone_depth_m": self.transition_zone_depth_m,
            "topdown_trail_seconds": self.topdown_trail_seconds,
        }


@dataclass(frozen=True, slots=True)
class BallTrackingSettings:
    """Validated image-space association, interpolation, and smoothing settings."""

    max_association_gap_seconds: float = 0.20
    max_interpolation_gap_seconds: float = 0.10
    maximum_speed_diagonals_per_second: float = 3.0
    maximum_acceleration_diagonals_per_second_squared: float = 80.0
    association_base_gate_diagonal_fraction: float = 0.012
    primary_court_side_margin_fraction: float = 0.12
    primary_court_air_margin_fraction: float = 0.50
    primary_court_bottom_margin_fraction: float = 0.06
    minimum_start_score: float = 0.40
    minimum_association_score: float = 0.32
    minimum_segment_observations: int = 2
    smoothing_window_frames: int = 5
    maximum_smoothing_adjustment_diagonal_fraction: float = 0.015
    debug_trail_seconds: float = 0.75

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> BallTrackingSettings:
        """Load conservative ball-trajectory settings from prefixed variables."""

        source = os.environ if environ is None else environ
        defaults = cls()
        values: dict[str, float | int] = {
            "max_association_gap_seconds": _float_setting(
                source,
                "BALL_TRACKING_MAX_ASSOCIATION_GAP_SECONDS",
                defaults.max_association_gap_seconds,
            ),
            "max_interpolation_gap_seconds": _float_setting(
                source,
                "BALL_TRACKING_MAX_INTERPOLATION_GAP_SECONDS",
                defaults.max_interpolation_gap_seconds,
            ),
            "maximum_speed_diagonals_per_second": _float_setting(
                source,
                "BALL_TRACKING_MAX_SPEED_DIAGONALS_PER_SECOND",
                defaults.maximum_speed_diagonals_per_second,
            ),
            "maximum_acceleration_diagonals_per_second_squared": _float_setting(
                source,
                "BALL_TRACKING_MAX_ACCELERATION_DIAGONALS_PER_SECOND_SQUARED",
                defaults.maximum_acceleration_diagonals_per_second_squared,
            ),
            "association_base_gate_diagonal_fraction": _float_setting(
                source,
                "BALL_TRACKING_BASE_GATE_DIAGONAL_FRACTION",
                defaults.association_base_gate_diagonal_fraction,
            ),
            "primary_court_side_margin_fraction": _float_setting(
                source,
                "BALL_TRACKING_COURT_SIDE_MARGIN_FRACTION",
                defaults.primary_court_side_margin_fraction,
            ),
            "primary_court_air_margin_fraction": _float_setting(
                source,
                "BALL_TRACKING_COURT_AIR_MARGIN_FRACTION",
                defaults.primary_court_air_margin_fraction,
            ),
            "primary_court_bottom_margin_fraction": _float_setting(
                source,
                "BALL_TRACKING_COURT_BOTTOM_MARGIN_FRACTION",
                defaults.primary_court_bottom_margin_fraction,
            ),
            "minimum_start_score": _float_setting(
                source,
                "BALL_TRACKING_MINIMUM_START_SCORE",
                defaults.minimum_start_score,
            ),
            "minimum_association_score": _float_setting(
                source,
                "BALL_TRACKING_MINIMUM_ASSOCIATION_SCORE",
                defaults.minimum_association_score,
            ),
            "minimum_segment_observations": _int_setting(
                source,
                "BALL_TRACKING_MINIMUM_SEGMENT_OBSERVATIONS",
                defaults.minimum_segment_observations,
            ),
            "smoothing_window_frames": _int_setting(
                source,
                "BALL_TRACKING_SMOOTHING_WINDOW_FRAMES",
                defaults.smoothing_window_frames,
            ),
            "maximum_smoothing_adjustment_diagonal_fraction": _float_setting(
                source,
                "BALL_TRACKING_MAXIMUM_SMOOTHING_ADJUSTMENT_DIAGONAL_FRACTION",
                defaults.maximum_smoothing_adjustment_diagonal_fraction,
            ),
            "debug_trail_seconds": _float_setting(
                source,
                "BALL_TRACKING_DEBUG_TRAIL_SECONDS",
                defaults.debug_trail_seconds,
            ),
        }
        positive_fields = (
            "max_association_gap_seconds",
            "max_interpolation_gap_seconds",
            "maximum_speed_diagonals_per_second",
            "maximum_acceleration_diagonals_per_second_squared",
            "association_base_gate_diagonal_fraction",
            "primary_court_side_margin_fraction",
            "primary_court_air_margin_fraction",
            "primary_court_bottom_margin_fraction",
            "maximum_smoothing_adjustment_diagonal_fraction",
            "debug_trail_seconds",
        )
        for field_name in positive_fields:
            if float(values[field_name]) <= 0:
                suffix = {
                    "max_association_gap_seconds": "BALL_TRACKING_MAX_ASSOCIATION_GAP_SECONDS",
                    "max_interpolation_gap_seconds": "BALL_TRACKING_MAX_INTERPOLATION_GAP_SECONDS",
                    "maximum_speed_diagonals_per_second": (
                        "BALL_TRACKING_MAX_SPEED_DIAGONALS_PER_SECOND"
                    ),
                    "maximum_acceleration_diagonals_per_second_squared": (
                        "BALL_TRACKING_MAX_ACCELERATION_DIAGONALS_PER_SECOND_SQUARED"
                    ),
                    "association_base_gate_diagonal_fraction": (
                        "BALL_TRACKING_BASE_GATE_DIAGONAL_FRACTION"
                    ),
                    "primary_court_side_margin_fraction": (
                        "BALL_TRACKING_COURT_SIDE_MARGIN_FRACTION"
                    ),
                    "primary_court_air_margin_fraction": (
                        "BALL_TRACKING_COURT_AIR_MARGIN_FRACTION"
                    ),
                    "primary_court_bottom_margin_fraction": (
                        "BALL_TRACKING_COURT_BOTTOM_MARGIN_FRACTION"
                    ),
                    "maximum_smoothing_adjustment_diagonal_fraction": (
                        "BALL_TRACKING_MAXIMUM_SMOOTHING_ADJUSTMENT_DIAGONAL_FRACTION"
                    ),
                    "debug_trail_seconds": "BALL_TRACKING_DEBUG_TRAIL_SECONDS",
                }[field_name]
                setting = f"{ENV_PREFIX}{suffix}"
                raise ConfigurationError(f"{setting} must be positive", setting=setting)
        for field_name, suffix in (
            ("minimum_start_score", "BALL_TRACKING_MINIMUM_START_SCORE"),
            ("minimum_association_score", "BALL_TRACKING_MINIMUM_ASSOCIATION_SCORE"),
        ):
            if not 0 <= float(values[field_name]) <= 1:
                setting = f"{ENV_PREFIX}{suffix}"
                raise ConfigurationError(
                    f"{setting} must be between 0 and 1 inclusive",
                    setting=setting,
                )
        minimum_observations = int(values["minimum_segment_observations"])
        if minimum_observations < 2:
            setting = f"{ENV_PREFIX}BALL_TRACKING_MINIMUM_SEGMENT_OBSERVATIONS"
            raise ConfigurationError(f"{setting} must be at least 2", setting=setting)
        smoothing_window = int(values["smoothing_window_frames"])
        if smoothing_window < 3 or smoothing_window % 2 == 0:
            setting = f"{ENV_PREFIX}BALL_TRACKING_SMOOTHING_WINDOW_FRAMES"
            raise ConfigurationError(
                f"{setting} must be an odd integer of at least 3",
                setting=setting,
            )
        if float(values["max_interpolation_gap_seconds"]) > float(
            values["max_association_gap_seconds"]
        ):
            setting = f"{ENV_PREFIX}BALL_TRACKING_MAX_INTERPOLATION_GAP_SECONDS"
            raise ConfigurationError(
                f"{setting} must not exceed {ENV_PREFIX}BALL_TRACKING_MAX_ASSOCIATION_GAP_SECONDS",
                setting=setting,
            )
        return cls(
            max_association_gap_seconds=float(values["max_association_gap_seconds"]),
            max_interpolation_gap_seconds=float(values["max_interpolation_gap_seconds"]),
            maximum_speed_diagonals_per_second=float(values["maximum_speed_diagonals_per_second"]),
            maximum_acceleration_diagonals_per_second_squared=float(
                values["maximum_acceleration_diagonals_per_second_squared"]
            ),
            association_base_gate_diagonal_fraction=float(
                values["association_base_gate_diagonal_fraction"]
            ),
            primary_court_side_margin_fraction=float(values["primary_court_side_margin_fraction"]),
            primary_court_air_margin_fraction=float(values["primary_court_air_margin_fraction"]),
            primary_court_bottom_margin_fraction=float(
                values["primary_court_bottom_margin_fraction"]
            ),
            minimum_start_score=float(values["minimum_start_score"]),
            minimum_association_score=float(values["minimum_association_score"]),
            minimum_segment_observations=minimum_observations,
            smoothing_window_frames=smoothing_window,
            maximum_smoothing_adjustment_diagonal_fraction=float(
                values["maximum_smoothing_adjustment_diagonal_fraction"]
            ),
            debug_trail_seconds=float(values["debug_trail_seconds"]),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "max_association_gap_seconds": self.max_association_gap_seconds,
            "max_interpolation_gap_seconds": self.max_interpolation_gap_seconds,
            "maximum_speed_diagonals_per_second": self.maximum_speed_diagonals_per_second,
            "maximum_acceleration_diagonals_per_second_squared": (
                self.maximum_acceleration_diagonals_per_second_squared
            ),
            "association_base_gate_diagonal_fraction": (
                self.association_base_gate_diagonal_fraction
            ),
            "primary_court_image_envelope": {
                "side_margin_frame_fraction": self.primary_court_side_margin_fraction,
                "air_margin_frame_fraction": self.primary_court_air_margin_fraction,
                "bottom_margin_frame_fraction": self.primary_court_bottom_margin_fraction,
                "airborne_points_projected_through_homography": False,
            },
            "minimum_start_score": self.minimum_start_score,
            "minimum_association_score": self.minimum_association_score,
            "minimum_segment_observations": self.minimum_segment_observations,
            "smoothing": {
                "method": "bounded_centered_confidence_weighted_mean",
                "window_frames": self.smoothing_window_frames,
                "maximum_adjustment_diagonal_fraction": (
                    self.maximum_smoothing_adjustment_diagonal_fraction
                ),
                "overwrites_raw_observations": False,
            },
            "debug_trail_seconds": self.debug_trail_seconds,
        }


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated settings loaded once at an executable boundary."""

    environment: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.JSON
    output_dir: Path = Path("output")
    media: MediaSettings = field(default_factory=MediaSettings)
    audio_analysis: AudioAnalysisSettings = field(default_factory=AudioAnalysisSettings)
    person_detection: PersonDetectionSettings = field(default_factory=PersonDetectionSettings)
    player_isolation: PlayerIsolationSettings = field(default_factory=PlayerIsolationSettings)
    player_tracking: PlayerTrackingSettings = field(default_factory=PlayerTrackingSettings)
    player_analysis: PlayerAnalysisSettings = field(default_factory=PlayerAnalysisSettings)
    ball_tracking: BallTrackingSettings = field(default_factory=BallTrackingSettings)

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
            media=MediaSettings.from_env(source),
            audio_analysis=AudioAnalysisSettings.from_env(source),
            person_detection=PersonDetectionSettings.from_env(source),
            player_isolation=PlayerIsolationSettings.from_env(source),
            player_tracking=PlayerTrackingSettings.from_env(source),
            player_analysis=PlayerAnalysisSettings.from_env(source),
            ball_tracking=BallTrackingSettings.from_env(source),
        )

    def public_values(self) -> dict[str, object]:
        """Return non-secret settings suitable for diagnostics and logs."""

        return {
            "environment": self.environment.value,
            "log_level": self.log_level,
            "log_format": self.log_format.value,
            "output_dir": str(self.output_dir),
            "media": self.media.as_dict(),
            "audio_analysis": self.audio_analysis.as_dict(),
            "person_detection": self.person_detection.as_dict(),
            "player_isolation": self.player_isolation.as_dict(),
            "player_tracking": self.player_tracking.as_dict(),
            "player_analysis": self.player_analysis.as_dict(),
            "ball_tracking": self.ball_tracking.as_dict(),
        }
