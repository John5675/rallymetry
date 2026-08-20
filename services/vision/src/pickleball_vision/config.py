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


class ArtifactBackend(StrEnum):
    """Configured artifact provider selected at an application boundary."""

    LOCAL = "local"
    VERCEL_BLOB = "vercel_blob"


@dataclass(frozen=True, slots=True)
class MediaSettings:
    """Canonical media-timeline settings shared by future A/V fusion stages."""

    audio_video_offset_ms: float = 0.0
    fusion_tolerance_ms: float = 90.0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> MediaSettings:
        """Load the configured correction applied to audio timestamps."""

        source = os.environ if environ is None else environ
        offset_ms = _float_setting(source, "AUDIO_VIDEO_OFFSET_MS", cls().audio_video_offset_ms)
        tolerance_ms = _float_setting(
            source,
            "FUSION_TOLERANCE_MS",
            cls().fusion_tolerance_ms,
        )
        if not math.isfinite(offset_ms):
            setting = f"{ENV_PREFIX}AUDIO_VIDEO_OFFSET_MS"
            raise ConfigurationError(f"{setting} must be finite", setting=setting)
        if not math.isfinite(tolerance_ms) or tolerance_ms <= 0:
            setting = f"{ENV_PREFIX}FUSION_TOLERANCE_MS"
            raise ConfigurationError(f"{setting} must be finite and positive", setting=setting)
        return cls(audio_video_offset_ms=offset_ms, fusion_tolerance_ms=tolerance_ms)

    def as_dict(self) -> dict[str, object]:
        """Return non-secret timing configuration for provenance."""

        return {
            "audioVideoOffsetMs": self.audio_video_offset_ms,
            "fusionToleranceMs": self.fusion_tolerance_ms,
        }


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
class RallySegmentationSettings:
    """Validated thresholds for inspectable, signal-based rally segmentation."""

    minimum_motion_speed_diagonals_per_second: float = 0.10
    motion_link_gap_seconds: float = 0.20
    motion_support_window_seconds: float = 0.30
    minimum_motion_support_fraction: float = 0.25
    serve_minimum_speed_diagonals_per_second: float = 0.18
    serve_speed_surge_ratio: float = 1.60
    serve_baseline_window_seconds: float = 0.40
    serve_confirmation_seconds: float = 1.20
    serve_minimum_displacement_diagonal_fraction: float = 0.06
    serve_minimum_motion_fraction: float = 0.20
    minimum_rally_duration_seconds: float = 0.75
    maximum_rally_duration_seconds: float = 45.0
    end_quiet_seconds: float = 0.90
    end_tail_grace_seconds: float = 0.25
    minimum_between_rallies_seconds: float = 1.0
    restart_quiet_seconds: float = 0.50
    restart_minimum_elapsed_seconds: float = 2.50
    dead_ball_handoff_window_seconds: float = 2.25
    dead_ball_handoff_minimum_quality_margin: float = 0.05
    dead_ball_handoff_full_duration_seconds: float = 4.0
    player_reset_window_seconds: float = 1.0
    player_reset_maximum_speed_mps: float = 1.25
    audio_support_tolerance_seconds: float = 0.12
    evaluation_minimum_iou: float = 0.25
    evaluation_boundary_tolerance_seconds: float = 1.50
    sparse_evaluation_margin_seconds: float = 2.0

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> RallySegmentationSettings:
        """Load rally thresholds from prefixed environment variables."""

        source = os.environ if environ is None else environ
        defaults = cls()
        suffixes = {
            "minimum_motion_speed_diagonals_per_second": (
                "RALLY_MINIMUM_MOTION_SPEED_DIAGONALS_PER_SECOND"
            ),
            "motion_link_gap_seconds": "RALLY_MOTION_LINK_GAP_SECONDS",
            "motion_support_window_seconds": "RALLY_MOTION_SUPPORT_WINDOW_SECONDS",
            "minimum_motion_support_fraction": "RALLY_MINIMUM_MOTION_SUPPORT_FRACTION",
            "serve_minimum_speed_diagonals_per_second": (
                "RALLY_SERVE_MINIMUM_SPEED_DIAGONALS_PER_SECOND"
            ),
            "serve_speed_surge_ratio": "RALLY_SERVE_SPEED_SURGE_RATIO",
            "serve_baseline_window_seconds": "RALLY_SERVE_BASELINE_WINDOW_SECONDS",
            "serve_confirmation_seconds": "RALLY_SERVE_CONFIRMATION_SECONDS",
            "serve_minimum_displacement_diagonal_fraction": (
                "RALLY_SERVE_MINIMUM_DISPLACEMENT_DIAGONAL_FRACTION"
            ),
            "serve_minimum_motion_fraction": "RALLY_SERVE_MINIMUM_MOTION_FRACTION",
            "minimum_rally_duration_seconds": "RALLY_MINIMUM_DURATION_SECONDS",
            "maximum_rally_duration_seconds": "RALLY_MAXIMUM_DURATION_SECONDS",
            "end_quiet_seconds": "RALLY_END_QUIET_SECONDS",
            "end_tail_grace_seconds": "RALLY_END_TAIL_GRACE_SECONDS",
            "minimum_between_rallies_seconds": "RALLY_MINIMUM_BETWEEN_SECONDS",
            "restart_quiet_seconds": "RALLY_RESTART_QUIET_SECONDS",
            "restart_minimum_elapsed_seconds": "RALLY_RESTART_MINIMUM_ELAPSED_SECONDS",
            "dead_ball_handoff_window_seconds": "RALLY_DEAD_BALL_HANDOFF_WINDOW_SECONDS",
            "dead_ball_handoff_minimum_quality_margin": (
                "RALLY_DEAD_BALL_HANDOFF_MINIMUM_QUALITY_MARGIN"
            ),
            "dead_ball_handoff_full_duration_seconds": (
                "RALLY_DEAD_BALL_HANDOFF_FULL_DURATION_SECONDS"
            ),
            "player_reset_window_seconds": "RALLY_PLAYER_RESET_WINDOW_SECONDS",
            "player_reset_maximum_speed_mps": "RALLY_PLAYER_RESET_MAXIMUM_SPEED_MPS",
            "audio_support_tolerance_seconds": "RALLY_AUDIO_SUPPORT_TOLERANCE_SECONDS",
            "evaluation_minimum_iou": "RALLY_EVALUATION_MINIMUM_IOU",
            "evaluation_boundary_tolerance_seconds": (
                "RALLY_EVALUATION_BOUNDARY_TOLERANCE_SECONDS"
            ),
            "sparse_evaluation_margin_seconds": "RALLY_SPARSE_EVALUATION_MARGIN_SECONDS",
        }
        values = {
            field_name: _float_setting(source, suffix, getattr(defaults, field_name))
            for field_name, suffix in suffixes.items()
        }
        fraction_fields = (
            "minimum_motion_support_fraction",
            "serve_minimum_displacement_diagonal_fraction",
            "serve_minimum_motion_fraction",
            "dead_ball_handoff_minimum_quality_margin",
            "evaluation_minimum_iou",
        )
        for field_name in fraction_fields:
            value = values[field_name]
            if not 0 <= value <= 1:
                setting = f"{ENV_PREFIX}{suffixes[field_name]}"
                raise ConfigurationError(
                    f"{setting} must be between 0 and 1 inclusive",
                    setting=setting,
                )
        for field_name, value in values.items():
            if field_name in fraction_fields:
                continue
            if not math.isfinite(value) or value <= 0:
                setting = f"{ENV_PREFIX}{suffixes[field_name]}"
                raise ConfigurationError(f"{setting} must be finite and positive", setting=setting)
        if values["maximum_rally_duration_seconds"] <= values["minimum_rally_duration_seconds"]:
            setting = f"{ENV_PREFIX}RALLY_MAXIMUM_DURATION_SECONDS"
            raise ConfigurationError(
                f"{setting} must exceed {ENV_PREFIX}RALLY_MINIMUM_DURATION_SECONDS",
                setting=setting,
            )
        if values["restart_quiet_seconds"] >= values["end_quiet_seconds"]:
            setting = f"{ENV_PREFIX}RALLY_RESTART_QUIET_SECONDS"
            raise ConfigurationError(
                f"{setting} must be less than {ENV_PREFIX}RALLY_END_QUIET_SECONDS",
                setting=setting,
            )
        return cls(**values)

    def as_dict(self) -> dict[str, object]:
        """Return the complete non-secret segmentation configuration."""

        return {
            "minimum_motion_speed_diagonals_per_second": (
                self.minimum_motion_speed_diagonals_per_second
            ),
            "motion_link_gap_seconds": self.motion_link_gap_seconds,
            "motion_support_window_seconds": self.motion_support_window_seconds,
            "minimum_motion_support_fraction": self.minimum_motion_support_fraction,
            "serve_like_sequence": {
                "minimum_speed_diagonals_per_second": (
                    self.serve_minimum_speed_diagonals_per_second
                ),
                "speed_surge_ratio": self.serve_speed_surge_ratio,
                "baseline_window_seconds": self.serve_baseline_window_seconds,
                "confirmation_seconds": self.serve_confirmation_seconds,
                "minimum_displacement_diagonal_fraction": (
                    self.serve_minimum_displacement_diagonal_fraction
                ),
                "minimum_motion_fraction": self.serve_minimum_motion_fraction,
            },
            "minimum_rally_duration_seconds": self.minimum_rally_duration_seconds,
            "maximum_rally_duration_seconds": self.maximum_rally_duration_seconds,
            "end_quiet_seconds": self.end_quiet_seconds,
            "end_tail_grace_seconds": self.end_tail_grace_seconds,
            "minimum_between_rallies_seconds": self.minimum_between_rallies_seconds,
            "restart_quiet_seconds": self.restart_quiet_seconds,
            "restart_minimum_elapsed_seconds": self.restart_minimum_elapsed_seconds,
            "dead_ball_handoff_filter": {
                "adjacent_burst_window_seconds": self.dead_ball_handoff_window_seconds,
                "minimum_quality_margin": self.dead_ball_handoff_minimum_quality_margin,
                "full_duration_seconds": self.dead_ball_handoff_full_duration_seconds,
                "semantic_classification": False,
            },
            "player_reset_window_seconds": self.player_reset_window_seconds,
            "player_reset_maximum_speed_mps": self.player_reset_maximum_speed_mps,
            "audio_support_tolerance_seconds": self.audio_support_tolerance_seconds,
            "evaluation": {
                "minimum_interval_iou": self.evaluation_minimum_iou,
                "boundary_tolerance_seconds": self.evaluation_boundary_tolerance_seconds,
                "sparse_annotation_margin_seconds": self.sparse_evaluation_margin_seconds,
                "automatic_threshold_tuning": False,
            },
        }


@dataclass(frozen=True, slots=True)
class BounceDetectionSettings:
    """Validated visual-first bounce-candidate and optional fusion thresholds."""

    trajectory_window_seconds: float = 0.20
    minimum_observations_each_side: int = 2
    minimum_vertical_speed_diagonals_per_second: float = 0.018
    minimum_vertical_reversal_diagonals_per_second: float = 0.040
    minimum_shape_prominence_diagonal_fraction: float = 0.0015
    minimum_continuity_fraction: float = 0.65
    minimum_visual_candidate_confidence: float = 0.35
    accepted_confidence: float = 0.80
    plane_projection_minimum_visual_confidence: float = 0.55
    minimum_between_bounces_seconds: float = 0.18
    audio_confidence_weight: float = 0.20
    rally_sequence_confidence_boost: float = 0.05
    evaluation_tolerance_ms: float = 120.0
    sparse_evaluation_margin_seconds: float = 0.30

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> BounceDetectionSettings:
        """Load bounce thresholds from prefixed environment variables."""

        source = os.environ if environ is None else environ
        defaults = cls()
        float_suffixes = {
            "trajectory_window_seconds": "BOUNCE_TRAJECTORY_WINDOW_SECONDS",
            "minimum_vertical_speed_diagonals_per_second": (
                "BOUNCE_MINIMUM_VERTICAL_SPEED_DIAGONALS_PER_SECOND"
            ),
            "minimum_vertical_reversal_diagonals_per_second": (
                "BOUNCE_MINIMUM_VERTICAL_REVERSAL_DIAGONALS_PER_SECOND"
            ),
            "minimum_shape_prominence_diagonal_fraction": (
                "BOUNCE_MINIMUM_SHAPE_PROMINENCE_DIAGONAL_FRACTION"
            ),
            "minimum_continuity_fraction": "BOUNCE_MINIMUM_CONTINUITY_FRACTION",
            "minimum_visual_candidate_confidence": ("BOUNCE_MINIMUM_VISUAL_CANDIDATE_CONFIDENCE"),
            "accepted_confidence": "BOUNCE_ACCEPTED_CONFIDENCE",
            "plane_projection_minimum_visual_confidence": (
                "BOUNCE_PLANE_PROJECTION_MINIMUM_VISUAL_CONFIDENCE"
            ),
            "minimum_between_bounces_seconds": "BOUNCE_MINIMUM_BETWEEN_SECONDS",
            "audio_confidence_weight": "BOUNCE_AUDIO_CONFIDENCE_WEIGHT",
            "rally_sequence_confidence_boost": "BOUNCE_RALLY_SEQUENCE_CONFIDENCE_BOOST",
            "evaluation_tolerance_ms": "BOUNCE_EVALUATION_TOLERANCE_MS",
            "sparse_evaluation_margin_seconds": "BOUNCE_SPARSE_EVALUATION_MARGIN_SECONDS",
        }
        values: dict[str, float | int] = {
            name: _float_setting(source, suffix, getattr(defaults, name))
            for name, suffix in float_suffixes.items()
        }
        observation_suffix = "BOUNCE_MINIMUM_OBSERVATIONS_EACH_SIDE"
        observations = _int_setting(
            source,
            observation_suffix,
            defaults.minimum_observations_each_side,
        )
        if observations < 2:
            setting = f"{ENV_PREFIX}{observation_suffix}"
            raise ConfigurationError(f"{setting} must be at least 2", setting=setting)
        values["minimum_observations_each_side"] = observations
        fraction_fields = (
            "minimum_shape_prominence_diagonal_fraction",
            "minimum_continuity_fraction",
            "minimum_visual_candidate_confidence",
            "accepted_confidence",
            "plane_projection_minimum_visual_confidence",
            "audio_confidence_weight",
            "rally_sequence_confidence_boost",
        )
        for name in fraction_fields:
            value = float(values[name])
            if not 0 <= value <= 1:
                setting = f"{ENV_PREFIX}{float_suffixes[name]}"
                raise ConfigurationError(
                    f"{setting} must be between 0 and 1 inclusive",
                    setting=setting,
                )
        for name, suffix in float_suffixes.items():
            if name in fraction_fields:
                continue
            value = float(values[name])
            if not math.isfinite(value) or value <= 0:
                setting = f"{ENV_PREFIX}{suffix}"
                raise ConfigurationError(f"{setting} must be finite and positive", setting=setting)
        if values["accepted_confidence"] < values["minimum_visual_candidate_confidence"]:
            setting = f"{ENV_PREFIX}BOUNCE_ACCEPTED_CONFIDENCE"
            raise ConfigurationError(
                f"{setting} must be at least "
                f"{ENV_PREFIX}BOUNCE_MINIMUM_VISUAL_CANDIDATE_CONFIDENCE",
                setting=setting,
            )
        if (
            values["plane_projection_minimum_visual_confidence"]
            < values["minimum_visual_candidate_confidence"]
        ):
            setting = f"{ENV_PREFIX}BOUNCE_PLANE_PROJECTION_MINIMUM_VISUAL_CONFIDENCE"
            raise ConfigurationError(
                f"{setting} must be at least "
                f"{ENV_PREFIX}BOUNCE_MINIMUM_VISUAL_CANDIDATE_CONFIDENCE",
                setting=setting,
            )
        return cls(
            trajectory_window_seconds=float(values["trajectory_window_seconds"]),
            minimum_observations_each_side=int(values["minimum_observations_each_side"]),
            minimum_vertical_speed_diagonals_per_second=float(
                values["minimum_vertical_speed_diagonals_per_second"]
            ),
            minimum_vertical_reversal_diagonals_per_second=float(
                values["minimum_vertical_reversal_diagonals_per_second"]
            ),
            minimum_shape_prominence_diagonal_fraction=float(
                values["minimum_shape_prominence_diagonal_fraction"]
            ),
            minimum_continuity_fraction=float(values["minimum_continuity_fraction"]),
            minimum_visual_candidate_confidence=float(
                values["minimum_visual_candidate_confidence"]
            ),
            accepted_confidence=float(values["accepted_confidence"]),
            plane_projection_minimum_visual_confidence=float(
                values["plane_projection_minimum_visual_confidence"]
            ),
            minimum_between_bounces_seconds=float(values["minimum_between_bounces_seconds"]),
            audio_confidence_weight=float(values["audio_confidence_weight"]),
            rally_sequence_confidence_boost=float(values["rally_sequence_confidence_boost"]),
            evaluation_tolerance_ms=float(values["evaluation_tolerance_ms"]),
            sparse_evaluation_margin_seconds=float(values["sparse_evaluation_margin_seconds"]),
        )

    def as_dict(self) -> dict[str, object]:
        """Return complete, non-secret visual and fusion configuration."""

        return {
            "trajectoryWindowSeconds": self.trajectory_window_seconds,
            "minimumObservationsEachSide": self.minimum_observations_each_side,
            "minimumVerticalSpeedDiagonalsPerSecond": (
                self.minimum_vertical_speed_diagonals_per_second
            ),
            "minimumVerticalReversalDiagonalsPerSecond": (
                self.minimum_vertical_reversal_diagonals_per_second
            ),
            "minimumShapeProminenceDiagonalFraction": (
                self.minimum_shape_prominence_diagonal_fraction
            ),
            "minimumContinuityFraction": self.minimum_continuity_fraction,
            "minimumVisualCandidateConfidence": self.minimum_visual_candidate_confidence,
            "acceptedConfidence": self.accepted_confidence,
            "planeProjectionMinimumVisualConfidence": (
                self.plane_projection_minimum_visual_confidence
            ),
            "minimumBetweenBouncesSeconds": self.minimum_between_bounces_seconds,
            "audioConfidenceWeight": self.audio_confidence_weight,
            "rallySequenceConfidenceBoost": self.rally_sequence_confidence_boost,
            "evaluationToleranceMs": self.evaluation_tolerance_ms,
            "sparseEvaluationMarginSeconds": self.sparse_evaluation_margin_seconds,
        }


@dataclass(frozen=True, slots=True)
class ContactDetectionSettings:
    """Validated visual-first paddle-contact and optional fusion thresholds."""

    trajectory_window_seconds: float = 0.16
    minimum_observations_each_side: int = 2
    minimum_velocity_change_diagonals_per_second: float = 0.060
    minimum_direction_change_degrees: float = 22.0
    minimum_speed_change_ratio: float = 1.35
    minimum_continuity_fraction: float = 0.65
    maximum_player_proximity_diagonal_fraction: float = 0.12
    minimum_visual_candidate_confidence: float = 0.40
    accepted_confidence: float = 0.78
    minimum_between_contacts_seconds: float = 0.12
    bounce_exclusion_window_seconds: float = 0.08
    maximum_previous_bounce_gap_seconds: float = 3.0
    audio_confidence_weight: float = 0.20
    rally_sequence_confidence_boost: float = 0.04
    previous_bounce_confidence_boost: float = 0.03
    evaluation_tolerance_ms: float = 100.0
    sparse_evaluation_margin_seconds: float = 0.30

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> ContactDetectionSettings:
        """Load contact thresholds from prefixed environment variables."""

        source = os.environ if environ is None else environ
        defaults = cls()
        float_suffixes = {
            "trajectory_window_seconds": "CONTACT_TRAJECTORY_WINDOW_SECONDS",
            "minimum_velocity_change_diagonals_per_second": (
                "CONTACT_MINIMUM_VELOCITY_CHANGE_DIAGONALS_PER_SECOND"
            ),
            "minimum_direction_change_degrees": ("CONTACT_MINIMUM_DIRECTION_CHANGE_DEGREES"),
            "minimum_speed_change_ratio": "CONTACT_MINIMUM_SPEED_CHANGE_RATIO",
            "minimum_continuity_fraction": "CONTACT_MINIMUM_CONTINUITY_FRACTION",
            "maximum_player_proximity_diagonal_fraction": (
                "CONTACT_MAXIMUM_PLAYER_PROXIMITY_DIAGONAL_FRACTION"
            ),
            "minimum_visual_candidate_confidence": ("CONTACT_MINIMUM_VISUAL_CANDIDATE_CONFIDENCE"),
            "accepted_confidence": "CONTACT_ACCEPTED_CONFIDENCE",
            "minimum_between_contacts_seconds": "CONTACT_MINIMUM_BETWEEN_SECONDS",
            "bounce_exclusion_window_seconds": "CONTACT_BOUNCE_EXCLUSION_WINDOW_SECONDS",
            "maximum_previous_bounce_gap_seconds": ("CONTACT_MAXIMUM_PREVIOUS_BOUNCE_GAP_SECONDS"),
            "audio_confidence_weight": "CONTACT_AUDIO_CONFIDENCE_WEIGHT",
            "rally_sequence_confidence_boost": "CONTACT_RALLY_SEQUENCE_CONFIDENCE_BOOST",
            "previous_bounce_confidence_boost": ("CONTACT_PREVIOUS_BOUNCE_CONFIDENCE_BOOST"),
            "evaluation_tolerance_ms": "CONTACT_EVALUATION_TOLERANCE_MS",
            "sparse_evaluation_margin_seconds": ("CONTACT_SPARSE_EVALUATION_MARGIN_SECONDS"),
        }
        values: dict[str, float | int] = {
            name: _float_setting(source, suffix, getattr(defaults, name))
            for name, suffix in float_suffixes.items()
        }
        observation_suffix = "CONTACT_MINIMUM_OBSERVATIONS_EACH_SIDE"
        observations = _int_setting(
            source,
            observation_suffix,
            defaults.minimum_observations_each_side,
        )
        if observations < 2:
            setting = f"{ENV_PREFIX}{observation_suffix}"
            raise ConfigurationError(f"{setting} must be at least 2", setting=setting)
        values["minimum_observations_each_side"] = observations
        fraction_fields = (
            "minimum_continuity_fraction",
            "maximum_player_proximity_diagonal_fraction",
            "minimum_visual_candidate_confidence",
            "accepted_confidence",
            "audio_confidence_weight",
            "rally_sequence_confidence_boost",
            "previous_bounce_confidence_boost",
        )
        for name in fraction_fields:
            value = float(values[name])
            if not 0 <= value <= 1:
                setting = f"{ENV_PREFIX}{float_suffixes[name]}"
                raise ConfigurationError(
                    f"{setting} must be between 0 and 1 inclusive",
                    setting=setting,
                )
        for name, suffix in float_suffixes.items():
            if name in fraction_fields:
                continue
            value = float(values[name])
            if not math.isfinite(value) or value <= 0:
                setting = f"{ENV_PREFIX}{suffix}"
                raise ConfigurationError(
                    f"{setting} must be finite and positive",
                    setting=setting,
                )
        if values["maximum_player_proximity_diagonal_fraction"] == 0:
            setting = f"{ENV_PREFIX}CONTACT_MAXIMUM_PLAYER_PROXIMITY_DIAGONAL_FRACTION"
            raise ConfigurationError(f"{setting} must be greater than zero", setting=setting)
        if values["minimum_direction_change_degrees"] > 180:
            setting = f"{ENV_PREFIX}CONTACT_MINIMUM_DIRECTION_CHANGE_DEGREES"
            raise ConfigurationError(f"{setting} must not exceed 180", setting=setting)
        if values["minimum_speed_change_ratio"] <= 1:
            setting = f"{ENV_PREFIX}CONTACT_MINIMUM_SPEED_CHANGE_RATIO"
            raise ConfigurationError(f"{setting} must be greater than 1", setting=setting)
        if values["accepted_confidence"] < values["minimum_visual_candidate_confidence"]:
            setting = f"{ENV_PREFIX}CONTACT_ACCEPTED_CONFIDENCE"
            raise ConfigurationError(
                f"{setting} must be at least "
                f"{ENV_PREFIX}CONTACT_MINIMUM_VISUAL_CANDIDATE_CONFIDENCE",
                setting=setting,
            )
        return cls(
            trajectory_window_seconds=float(values["trajectory_window_seconds"]),
            minimum_observations_each_side=int(values["minimum_observations_each_side"]),
            minimum_velocity_change_diagonals_per_second=float(
                values["minimum_velocity_change_diagonals_per_second"]
            ),
            minimum_direction_change_degrees=float(values["minimum_direction_change_degrees"]),
            minimum_speed_change_ratio=float(values["minimum_speed_change_ratio"]),
            minimum_continuity_fraction=float(values["minimum_continuity_fraction"]),
            maximum_player_proximity_diagonal_fraction=float(
                values["maximum_player_proximity_diagonal_fraction"]
            ),
            minimum_visual_candidate_confidence=float(
                values["minimum_visual_candidate_confidence"]
            ),
            accepted_confidence=float(values["accepted_confidence"]),
            minimum_between_contacts_seconds=float(values["minimum_between_contacts_seconds"]),
            bounce_exclusion_window_seconds=float(values["bounce_exclusion_window_seconds"]),
            maximum_previous_bounce_gap_seconds=float(
                values["maximum_previous_bounce_gap_seconds"]
            ),
            audio_confidence_weight=float(values["audio_confidence_weight"]),
            rally_sequence_confidence_boost=float(values["rally_sequence_confidence_boost"]),
            previous_bounce_confidence_boost=float(values["previous_bounce_confidence_boost"]),
            evaluation_tolerance_ms=float(values["evaluation_tolerance_ms"]),
            sparse_evaluation_margin_seconds=float(values["sparse_evaluation_margin_seconds"]),
        )

    def as_dict(self) -> dict[str, object]:
        """Return complete, non-secret visual and fusion configuration."""

        return {
            "trajectoryWindowSeconds": self.trajectory_window_seconds,
            "minimumObservationsEachSide": self.minimum_observations_each_side,
            "minimumVelocityChangeDiagonalsPerSecond": (
                self.minimum_velocity_change_diagonals_per_second
            ),
            "minimumDirectionChangeDegrees": self.minimum_direction_change_degrees,
            "minimumSpeedChangeRatio": self.minimum_speed_change_ratio,
            "minimumContinuityFraction": self.minimum_continuity_fraction,
            "maximumPlayerProximityDiagonalFraction": (
                self.maximum_player_proximity_diagonal_fraction
            ),
            "minimumVisualCandidateConfidence": self.minimum_visual_candidate_confidence,
            "acceptedConfidence": self.accepted_confidence,
            "minimumBetweenContactsSeconds": self.minimum_between_contacts_seconds,
            "bounceExclusionWindowSeconds": self.bounce_exclusion_window_seconds,
            "maximumPreviousBounceGapSeconds": self.maximum_previous_bounce_gap_seconds,
            "audioConfidenceWeight": self.audio_confidence_weight,
            "rallySequenceConfidenceBoost": self.rally_sequence_confidence_boost,
            "previousBounceConfidenceBoost": self.previous_bounce_confidence_boost,
            "evaluationToleranceMs": self.evaluation_tolerance_ms,
            "sparseEvaluationMarginSeconds": self.sparse_evaluation_margin_seconds,
        }


@dataclass(frozen=True, slots=True)
class HitterIdentificationSettings:
    """Validated visual/player evidence thresholds for hitter resolution."""

    minimum_contact_confidence: float = 0.78
    minimum_assignment_confidence: float = 0.62
    minimum_assignment_margin: float = 0.08
    maximum_player_distance_diagonal_fraction: float = 0.12
    minimum_tracking_confidence: float = 0.45
    minimum_direction_speed_diagonal_fraction_per_second: float = 0.015
    previous_hitter_minimum_confidence: float = 0.70
    maximum_sequence_gap_seconds: float = 4.0
    evaluation_tolerance_ms: float = 100.0
    proximity_weight: float = 0.35
    tracking_weight: float = 0.17
    direction_weight: float = 0.18
    contact_weight: float = 0.12
    court_context_weight: float = 0.08
    sequence_weight: float = 0.10

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> HitterIdentificationSettings:
        """Load hitter-decision thresholds and evidence weights from the environment."""

        source = os.environ if environ is None else environ
        defaults = cls()
        suffixes = {
            "minimum_contact_confidence": "HITTER_MINIMUM_CONTACT_CONFIDENCE",
            "minimum_assignment_confidence": "HITTER_MINIMUM_ASSIGNMENT_CONFIDENCE",
            "minimum_assignment_margin": "HITTER_MINIMUM_ASSIGNMENT_MARGIN",
            "maximum_player_distance_diagonal_fraction": (
                "HITTER_MAXIMUM_PLAYER_DISTANCE_DIAGONAL_FRACTION"
            ),
            "minimum_tracking_confidence": "HITTER_MINIMUM_TRACKING_CONFIDENCE",
            "minimum_direction_speed_diagonal_fraction_per_second": (
                "HITTER_MINIMUM_DIRECTION_SPEED_DIAGONAL_FRACTION_PER_SECOND"
            ),
            "previous_hitter_minimum_confidence": ("HITTER_PREVIOUS_HITTER_MINIMUM_CONFIDENCE"),
            "maximum_sequence_gap_seconds": "HITTER_MAXIMUM_SEQUENCE_GAP_SECONDS",
            "evaluation_tolerance_ms": "HITTER_EVALUATION_TOLERANCE_MS",
            "proximity_weight": "HITTER_PROXIMITY_WEIGHT",
            "tracking_weight": "HITTER_TRACKING_WEIGHT",
            "direction_weight": "HITTER_DIRECTION_WEIGHT",
            "contact_weight": "HITTER_CONTACT_WEIGHT",
            "court_context_weight": "HITTER_COURT_CONTEXT_WEIGHT",
            "sequence_weight": "HITTER_SEQUENCE_WEIGHT",
        }
        values = {
            name: _float_setting(source, suffix, getattr(defaults, name))
            for name, suffix in suffixes.items()
        }
        fraction_fields = (
            "minimum_contact_confidence",
            "minimum_assignment_confidence",
            "minimum_assignment_margin",
            "maximum_player_distance_diagonal_fraction",
            "minimum_tracking_confidence",
            "previous_hitter_minimum_confidence",
        )
        for name in fraction_fields:
            value = values[name]
            if not math.isfinite(value) or not 0 <= value <= 1:
                setting = f"{ENV_PREFIX}{suffixes[name]}"
                raise ConfigurationError(
                    f"{setting} must be between 0 and 1 inclusive",
                    setting=setting,
                )
        positive_fields = (
            "minimum_contact_confidence",
            "minimum_assignment_confidence",
            "minimum_assignment_margin",
            "maximum_player_distance_diagonal_fraction",
            "minimum_tracking_confidence",
            "minimum_direction_speed_diagonal_fraction_per_second",
            "previous_hitter_minimum_confidence",
            "maximum_sequence_gap_seconds",
            "evaluation_tolerance_ms",
        )
        for name in positive_fields:
            value = values[name]
            if not math.isfinite(value) or value <= 0:
                setting = f"{ENV_PREFIX}{suffixes[name]}"
                raise ConfigurationError(
                    f"{setting} must be finite and positive",
                    setting=setting,
                )
        weight_fields = (
            "proximity_weight",
            "tracking_weight",
            "direction_weight",
            "contact_weight",
            "court_context_weight",
            "sequence_weight",
        )
        for name in weight_fields:
            value = values[name]
            if not math.isfinite(value) or value < 0:
                setting = f"{ENV_PREFIX}{suffixes[name]}"
                raise ConfigurationError(
                    f"{setting} must be finite and nonnegative",
                    setting=setting,
                )
        if sum(values[name] for name in weight_fields) <= 0:
            setting = f"{ENV_PREFIX}HITTER_PROXIMITY_WEIGHT"
            raise ConfigurationError(
                "At least one hitter-identification evidence weight must be positive",
                setting=setting,
            )
        return cls(**values)

    def as_dict(self) -> dict[str, object]:
        """Return complete, non-secret hitter configuration for provenance."""

        return {
            "minimumContactConfidence": self.minimum_contact_confidence,
            "minimumAssignmentConfidence": self.minimum_assignment_confidence,
            "minimumAssignmentMargin": self.minimum_assignment_margin,
            "maximumPlayerDistanceDiagonalFraction": (
                self.maximum_player_distance_diagonal_fraction
            ),
            "minimumTrackingConfidence": self.minimum_tracking_confidence,
            "minimumDirectionSpeedDiagonalFractionPerSecond": (
                self.minimum_direction_speed_diagonal_fraction_per_second
            ),
            "previousHitterMinimumConfidence": self.previous_hitter_minimum_confidence,
            "maximumSequenceGapSeconds": self.maximum_sequence_gap_seconds,
            "evaluationToleranceMs": self.evaluation_tolerance_ms,
            "evidenceWeights": {
                "proximity": self.proximity_weight,
                "tracking": self.tracking_weight,
                "trajectoryDirection": self.direction_weight,
                "contactConfidence": self.contact_weight,
                "courtContext": self.court_context_weight,
                "sequence": self.sequence_weight,
            },
        }


@dataclass(frozen=True, slots=True)
class ShotClassificationSettings:
    """Validated thresholds for structured shot reconstruction and rule classification."""

    minimum_hitter_confidence: float = 0.62
    minimum_trajectory_coverage: float = 0.50
    minimum_known_trajectory_points: int = 3
    serve_minimum_backcourt_distance_m: float = 1.20
    kitchen_proximity_m: float = 0.90
    drop_minimum_backcourt_distance_m: float = 0.90
    dink_maximum_speed_diagonals_per_second: float = 0.28
    drop_maximum_speed_diagonals_per_second: float = 0.38
    drive_minimum_speed_diagonals_per_second: float = 0.45
    overhead_minimum_speed_diagonals_per_second: float = 0.35
    overhead_maximum_contact_height_ratio: float = 0.45
    evaluation_tolerance_ms: float = 120.0
    debug_trail_seconds: float = 0.75

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> ShotClassificationSettings:
        """Load documented shot-rule thresholds from prefixed environment variables."""

        source = os.environ if environ is None else environ
        defaults = cls()
        float_suffixes = {
            "minimum_hitter_confidence": "SHOT_MINIMUM_HITTER_CONFIDENCE",
            "minimum_trajectory_coverage": "SHOT_MINIMUM_TRAJECTORY_COVERAGE",
            "serve_minimum_backcourt_distance_m": ("SHOT_SERVE_MINIMUM_BACKCOURT_DISTANCE_METERS"),
            "kitchen_proximity_m": "SHOT_KITCHEN_PROXIMITY_METERS",
            "drop_minimum_backcourt_distance_m": ("SHOT_DROP_MINIMUM_BACKCOURT_DISTANCE_METERS"),
            "dink_maximum_speed_diagonals_per_second": (
                "SHOT_DINK_MAXIMUM_SPEED_DIAGONALS_PER_SECOND"
            ),
            "drop_maximum_speed_diagonals_per_second": (
                "SHOT_DROP_MAXIMUM_SPEED_DIAGONALS_PER_SECOND"
            ),
            "drive_minimum_speed_diagonals_per_second": (
                "SHOT_DRIVE_MINIMUM_SPEED_DIAGONALS_PER_SECOND"
            ),
            "overhead_minimum_speed_diagonals_per_second": (
                "SHOT_OVERHEAD_MINIMUM_SPEED_DIAGONALS_PER_SECOND"
            ),
            "overhead_maximum_contact_height_ratio": ("SHOT_OVERHEAD_MAXIMUM_CONTACT_HEIGHT_RATIO"),
            "evaluation_tolerance_ms": "SHOT_EVALUATION_TOLERANCE_MS",
            "debug_trail_seconds": "SHOT_DEBUG_TRAIL_SECONDS",
        }
        values = {
            name: _float_setting(source, suffix, getattr(defaults, name))
            for name, suffix in float_suffixes.items()
        }
        fraction_fields = (
            "minimum_hitter_confidence",
            "minimum_trajectory_coverage",
            "overhead_maximum_contact_height_ratio",
        )
        for name in fraction_fields:
            value = values[name]
            if not math.isfinite(value) or not 0 < value <= 1:
                setting = f"{ENV_PREFIX}{float_suffixes[name]}"
                raise ConfigurationError(
                    f"{setting} must be greater than zero and at most 1",
                    setting=setting,
                )
        for name, suffix in float_suffixes.items():
            if name in fraction_fields:
                continue
            value = values[name]
            if not math.isfinite(value) or value <= 0:
                setting = f"{ENV_PREFIX}{suffix}"
                raise ConfigurationError(
                    f"{setting} must be finite and positive",
                    setting=setting,
                )
        points_suffix = "SHOT_MINIMUM_KNOWN_TRAJECTORY_POINTS"
        points = _int_setting(
            source,
            points_suffix,
            defaults.minimum_known_trajectory_points,
        )
        if points < 2:
            setting = f"{ENV_PREFIX}{points_suffix}"
            raise ConfigurationError(f"{setting} must be at least 2", setting=setting)
        if (
            values["dink_maximum_speed_diagonals_per_second"]
            > values["drop_maximum_speed_diagonals_per_second"]
        ):
            setting = f"{ENV_PREFIX}SHOT_DINK_MAXIMUM_SPEED_DIAGONALS_PER_SECOND"
            raise ConfigurationError(
                f"{setting} must not exceed the drop maximum speed",
                setting=setting,
            )
        return cls(
            minimum_hitter_confidence=values["minimum_hitter_confidence"],
            minimum_trajectory_coverage=values["minimum_trajectory_coverage"],
            minimum_known_trajectory_points=points,
            serve_minimum_backcourt_distance_m=values["serve_minimum_backcourt_distance_m"],
            kitchen_proximity_m=values["kitchen_proximity_m"],
            drop_minimum_backcourt_distance_m=values["drop_minimum_backcourt_distance_m"],
            dink_maximum_speed_diagonals_per_second=values[
                "dink_maximum_speed_diagonals_per_second"
            ],
            drop_maximum_speed_diagonals_per_second=values[
                "drop_maximum_speed_diagonals_per_second"
            ],
            drive_minimum_speed_diagonals_per_second=values[
                "drive_minimum_speed_diagonals_per_second"
            ],
            overhead_minimum_speed_diagonals_per_second=values[
                "overhead_minimum_speed_diagonals_per_second"
            ],
            overhead_maximum_contact_height_ratio=values["overhead_maximum_contact_height_ratio"],
            evaluation_tolerance_ms=values["evaluation_tolerance_ms"],
            debug_trail_seconds=values["debug_trail_seconds"],
        )

    def as_dict(self) -> dict[str, object]:
        """Return the complete rule configuration for artifact provenance."""

        return {
            "minimumHitterConfidence": self.minimum_hitter_confidence,
            "minimumTrajectoryCoverage": self.minimum_trajectory_coverage,
            "minimumKnownTrajectoryPoints": self.minimum_known_trajectory_points,
            "serveMinimumBackcourtDistanceMeters": self.serve_minimum_backcourt_distance_m,
            "kitchenProximityMeters": self.kitchen_proximity_m,
            "dropMinimumBackcourtDistanceMeters": self.drop_minimum_backcourt_distance_m,
            "dinkMaximumSpeedDiagonalsPerSecond": (self.dink_maximum_speed_diagonals_per_second),
            "dropMaximumSpeedDiagonalsPerSecond": (self.drop_maximum_speed_diagonals_per_second),
            "driveMinimumSpeedDiagonalsPerSecond": (self.drive_minimum_speed_diagonals_per_second),
            "overheadMinimumSpeedDiagonalsPerSecond": (
                self.overhead_minimum_speed_diagonals_per_second
            ),
            "overheadMaximumContactHeightRatio": self.overhead_maximum_contact_height_ratio,
            "evaluationToleranceMs": self.evaluation_tolerance_ms,
            "debugTrailSeconds": self.debug_trail_seconds,
        }


@dataclass(frozen=True, slots=True)
class MatchAnalyticsSettings:
    """Validated tactical definitions unique to deterministic match analytics."""

    kitchen_arrival_distance_m: float = 0.90
    minimum_kitchen_arrival_joint_coverage_ratio: float = 0.50

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> MatchAnalyticsSettings:
        """Load match-analytics definitions from prefixed environment variables."""

        source = os.environ if environ is None else environ
        defaults = cls()
        distance_m = _float_setting(
            source,
            "MATCH_ANALYTICS_KITCHEN_ARRIVAL_DISTANCE_METERS",
            defaults.kitchen_arrival_distance_m,
        )
        coverage = _float_setting(
            source,
            "MATCH_ANALYTICS_MINIMUM_KITCHEN_ARRIVAL_JOINT_COVERAGE_RATIO",
            defaults.minimum_kitchen_arrival_joint_coverage_ratio,
        )
        if not math.isfinite(distance_m) or distance_m < 0:
            setting = f"{ENV_PREFIX}MATCH_ANALYTICS_KITCHEN_ARRIVAL_DISTANCE_METERS"
            raise ConfigurationError(f"{setting} must be finite and nonnegative", setting=setting)
        if not math.isfinite(coverage) or not 0 <= coverage <= 1:
            setting = f"{ENV_PREFIX}MATCH_ANALYTICS_MINIMUM_KITCHEN_ARRIVAL_JOINT_COVERAGE_RATIO"
            raise ConfigurationError(f"{setting} must be between 0 and 1", setting=setting)
        return cls(distance_m, coverage)

    def as_dict(self) -> dict[str, object]:
        return {
            "kitchenArrivalDistanceMeters": self.kitchen_arrival_distance_m,
            "minimumKitchenArrivalJointCoverageRatio": (
                self.minimum_kitchen_arrival_joint_coverage_ratio
            ),
        }


@dataclass(frozen=True, slots=True)
class PersistenceSettings:
    """Hosted-adapter configuration whose public form never contains secrets."""

    mongodb_url: str | None = field(default=None, repr=False)
    mongodb_database: str = "pickleball_vision"
    artifact_backend: ArtifactBackend = ArtifactBackend.LOCAL
    local_artifact_root: Path = Path("output/artifacts")
    vercel_blob_token: str | None = field(default=None, repr=False)
    vercel_blob_public_token: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> PersistenceSettings:
        """Load exact provider variables while allowing a credential-free local default."""

        source = os.environ if environ is None else environ
        mongodb_url_raw = source.get("MONGODB_URL", "").strip()
        mongodb_url = mongodb_url_raw or None
        if mongodb_url is not None and not mongodb_url.startswith(("mongodb://", "mongodb+srv://")):
            raise ConfigurationError(
                "MONGODB_URL must use the mongodb:// or mongodb+srv:// scheme",
                setting="MONGODB_URL",
            )
        mongodb_database = source.get("MONGODB_DATABASE", cls().mongodb_database).strip()
        if re.fullmatch(r"[A-Za-z0-9_-]{1,63}", mongodb_database) is None:
            raise ConfigurationError(
                "MONGODB_DATABASE must contain 1-63 letters, numbers, underscores, or hyphens",
                setting="MONGODB_DATABASE",
            )
        backend_raw = source.get(
            f"{ENV_PREFIX}ARTIFACT_BACKEND",
            ArtifactBackend.LOCAL.value,
        ).strip()
        try:
            backend = ArtifactBackend(backend_raw.lower())
        except ValueError as error:
            setting = f"{ENV_PREFIX}ARTIFACT_BACKEND"
            raise ConfigurationError(
                f"{setting} must be local or vercel_blob",
                setting=setting,
            ) from error
        root_raw = source.get(
            f"{ENV_PREFIX}LOCAL_ARTIFACT_ROOT",
            str(cls().local_artifact_root),
        ).strip()
        if not root_raw:
            setting = f"{ENV_PREFIX}LOCAL_ARTIFACT_ROOT"
            raise ConfigurationError(f"{setting} must not be empty", setting=setting)
        token_raw = source.get("BLOB_READ_WRITE_TOKEN", "").strip()
        token = token_raw or None
        public_token_raw = source.get("PUBLIC_BLOB_READ_WRITE_TOKEN", "").strip()
        public_token = public_token_raw or None
        if backend is ArtifactBackend.VERCEL_BLOB and token is None:
            raise ConfigurationError(
                "BLOB_READ_WRITE_TOKEN is required when the Vercel Blob backend is selected",
                setting="BLOB_READ_WRITE_TOKEN",
            )
        return cls(
            mongodb_url=mongodb_url,
            mongodb_database=mongodb_database,
            artifact_backend=backend,
            local_artifact_root=Path(root_raw).expanduser(),
            vercel_blob_token=token,
            vercel_blob_public_token=public_token,
        )

    def public_values(self) -> dict[str, object]:
        """Return adapter readiness without returning either credential."""

        return {
            "mongodbConfigured": self.mongodb_url is not None,
            "mongodbDatabase": self.mongodb_database,
            "artifactBackend": self.artifact_backend.value,
            "localArtifactRoot": str(self.local_artifact_root),
            "vercelBlobConfigured": self.vercel_blob_token is not None,
            "vercelBlobPublicConfigured": self.vercel_blob_public_token is not None,
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
    rally_segmentation: RallySegmentationSettings = field(default_factory=RallySegmentationSettings)
    bounce_detection: BounceDetectionSettings = field(default_factory=BounceDetectionSettings)
    contact_detection: ContactDetectionSettings = field(default_factory=ContactDetectionSettings)
    hitter_identification: HitterIdentificationSettings = field(
        default_factory=HitterIdentificationSettings
    )
    shot_classification: ShotClassificationSettings = field(
        default_factory=ShotClassificationSettings
    )
    match_analytics: MatchAnalyticsSettings = field(default_factory=MatchAnalyticsSettings)
    persistence: PersistenceSettings = field(default_factory=PersistenceSettings)

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
            rally_segmentation=RallySegmentationSettings.from_env(source),
            bounce_detection=BounceDetectionSettings.from_env(source),
            contact_detection=ContactDetectionSettings.from_env(source),
            hitter_identification=HitterIdentificationSettings.from_env(source),
            shot_classification=ShotClassificationSettings.from_env(source),
            match_analytics=MatchAnalyticsSettings.from_env(source),
            persistence=PersistenceSettings.from_env(source),
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
            "rally_segmentation": self.rally_segmentation.as_dict(),
            "bounce_detection": self.bounce_detection.as_dict(),
            "contact_detection": self.contact_detection.as_dict(),
            "hitter_identification": self.hitter_identification.as_dict(),
            "shot_classification": self.shot_classification.as_dict(),
            "match_analytics": self.match_analytics.as_dict(),
            "persistence": self.persistence.public_values(),
        }
