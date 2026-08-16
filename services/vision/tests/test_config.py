from pathlib import Path

import pytest

from pickleball_vision.config import (
    AudioAnalysisChannelMode,
    AudioAnalysisSettings,
    BallTrackingSettings,
    Environment,
    LogFormat,
    MediaSettings,
    PersonDetectionSettings,
    PlayerAnalysisSettings,
    PlayerIsolationSettings,
    PlayerTrackingSettings,
    Settings,
)
from pickleball_vision.errors import ConfigurationError, ErrorCode


def test_settings_have_safe_development_defaults() -> None:
    settings = Settings.from_env({})

    assert settings.environment is Environment.DEVELOPMENT
    assert settings.log_level == "INFO"
    assert settings.log_format is LogFormat.JSON
    assert settings.output_dir == Path("output")
    assert settings.media == MediaSettings()
    assert settings.audio_analysis == AudioAnalysisSettings()
    assert settings.person_detection == PersonDetectionSettings()
    assert settings.player_isolation == PlayerIsolationSettings()
    assert settings.player_tracking == PlayerTrackingSettings()
    assert settings.player_analysis == PlayerAnalysisSettings()
    assert settings.ball_tracking == BallTrackingSettings()


def test_settings_load_prefixed_environment_values() -> None:
    settings = Settings.from_env(
        {
            "PICKLEBALL_VISION_ENVIRONMENT": "production",
            "PICKLEBALL_VISION_LOG_LEVEL": "warning",
            "PICKLEBALL_VISION_LOG_FORMAT": "console",
            "PICKLEBALL_VISION_OUTPUT_DIR": "~/pickleball-output",
            "PICKLEBALL_VISION_AUDIO_VIDEO_OFFSET_MS": "-35.5",
            "PICKLEBALL_VISION_AUDIO_ANALYSIS_SAMPLE_RATE_HZ": "24000",
            "PICKLEBALL_VISION_AUDIO_ANALYSIS_ONSET_SENSITIVITY": "5.5",
            "PICKLEBALL_VISION_AUDIO_ANALYSIS_MINIMUM_EVENT_SEPARATION_MS": "120",
            "PICKLEBALL_VISION_AUDIO_ANALYSIS_CHANNEL_MODE": "per_channel",
            "PICKLEBALL_VISION_AUDIO_ANALYSIS_FRAME_DURATION_MS": "40",
            "PICKLEBALL_VISION_AUDIO_ANALYSIS_HOP_DURATION_MS": "12.5",
            "PICKLEBALL_VISION_PERSON_MODEL": "custom-person.pt",
            "PICKLEBALL_VISION_PERSON_DEVICE": "CPU",
            "PICKLEBALL_VISION_PERSON_MIN_CONFIDENCE": "0.15",
            "PICKLEBALL_VISION_PERSON_IMAGE_SIZE": "960",
            "PICKLEBALL_VISION_PERSON_IOU_THRESHOLD": "0.6",
            "PICKLEBALL_VISION_PERSON_MAX_DETECTIONS": "250",
            "PICKLEBALL_VISION_ISOLATION_NEAR_MARGIN_METERS": "2.0",
            "PICKLEBALL_VISION_ISOLATION_BOUNDARY_UNCERTAINTY_METERS": "0.3",
            "PICKLEBALL_VISION_ISOLATION_SIDE_UNCERTAINTY_METERS": "0.4",
            "PICKLEBALL_VISION_ISOLATION_MAX_CANDIDATE_GAP_SECONDS": "1.5",
            "PICKLEBALL_VISION_ISOLATION_MAX_CANDIDATE_SPEED_MPS": "7.0",
            "PICKLEBALL_VISION_ISOLATION_MIN_CANDIDATE_OBSERVATIONS": "20",
            "PICKLEBALL_VISION_ISOLATION_MIN_COURT_SUPPORT_RATIO": "0.75",
            "PICKLEBALL_VISION_TRACKING_BUFFER_SECONDS": "1.5",
            "PICKLEBALL_VISION_TRACKING_MAX_IDENTITY_GAP_SECONDS": "2.5",
            "PICKLEBALL_VISION_TRACKING_LONG_GAP_APPEARANCE_SIMILARITY": "0.8",
            "PICKLEBALL_VISION_TRACKING_LONG_GAP_MINIMUM_APPEARANCE_MARGIN": "0.1",
            "PICKLEBALL_VISION_ANALYSIS_MINIMUM_TRACKING_CONFIDENCE": "0.6",
            "PICKLEBALL_VISION_ANALYSIS_SMOOTHING_WINDOW_FRAMES": "7",
            "PICKLEBALL_VISION_ANALYSIS_MAXIMUM_STEP_GAP_SECONDS": "0.3",
            "PICKLEBALL_VISION_BALL_TRACKING_MAX_ASSOCIATION_GAP_SECONDS": "0.25",
            "PICKLEBALL_VISION_BALL_TRACKING_MAX_INTERPOLATION_GAP_SECONDS": "0.05",
            "PICKLEBALL_VISION_BALL_TRACKING_MINIMUM_SEGMENT_OBSERVATIONS": "3",
            "PICKLEBALL_VISION_BALL_TRACKING_SMOOTHING_WINDOW_FRAMES": "7",
        }
    )

    assert settings.environment is Environment.PRODUCTION
    assert settings.log_level == "WARNING"
    assert settings.log_format is LogFormat.CONSOLE
    assert settings.output_dir == Path("~/pickleball-output").expanduser()
    assert settings.media == MediaSettings(audio_video_offset_ms=-35.5)
    assert settings.audio_analysis == AudioAnalysisSettings(
        analysis_sample_rate_hz=24000,
        onset_sensitivity=5.5,
        minimum_event_separation_ms=120.0,
        channel_mode=AudioAnalysisChannelMode.PER_CHANNEL,
        frame_duration_ms=40.0,
        hop_duration_ms=12.5,
    )
    assert settings.person_detection == PersonDetectionSettings(
        model="custom-person.pt",
        device="cpu",
        min_confidence=0.15,
        image_size=960,
        iou_threshold=0.6,
        max_detections=250,
    )
    assert settings.player_isolation == PlayerIsolationSettings(
        near_court_margin_m=2.0,
        boundary_uncertainty_m=0.3,
        side_uncertainty_m=0.4,
        max_candidate_gap_s=1.5,
        max_candidate_speed_mps=7.0,
        min_candidate_observations=20,
        min_court_support_ratio=0.75,
    )
    assert settings.player_tracking == PlayerTrackingSettings(
        track_buffer_seconds=1.5,
        max_identity_gap_seconds=2.5,
        long_gap_appearance_similarity=0.8,
        long_gap_minimum_appearance_margin=0.1,
    )
    assert settings.player_analysis == PlayerAnalysisSettings(
        minimum_tracking_confidence=0.6,
        smoothing_window_frames=7,
        maximum_step_gap_seconds=0.3,
    )
    assert settings.ball_tracking == BallTrackingSettings(
        max_association_gap_seconds=0.25,
        max_interpolation_gap_seconds=0.05,
        minimum_segment_observations=3,
        smoothing_window_frames=7,
    )


@pytest.mark.parametrize(
    ("environment", "setting"),
    [
        ({"PICKLEBALL_VISION_ENVIRONMENT": "staging"}, "PICKLEBALL_VISION_ENVIRONMENT"),
        ({"PICKLEBALL_VISION_LOG_LEVEL": "verbose"}, "PICKLEBALL_VISION_LOG_LEVEL"),
        ({"PICKLEBALL_VISION_LOG_FORMAT": "xml"}, "PICKLEBALL_VISION_LOG_FORMAT"),
        ({"PICKLEBALL_VISION_OUTPUT_DIR": "  "}, "PICKLEBALL_VISION_OUTPUT_DIR"),
        (
            {"PICKLEBALL_VISION_AUDIO_VIDEO_OFFSET_MS": "nan"},
            "PICKLEBALL_VISION_AUDIO_VIDEO_OFFSET_MS",
        ),
        (
            {"PICKLEBALL_VISION_AUDIO_ANALYSIS_SAMPLE_RATE_HZ": "4000"},
            "PICKLEBALL_VISION_AUDIO_ANALYSIS_SAMPLE_RATE_HZ",
        ),
        (
            {"PICKLEBALL_VISION_AUDIO_ANALYSIS_ONSET_SENSITIVITY": "nan"},
            "PICKLEBALL_VISION_AUDIO_ANALYSIS_ONSET_SENSITIVITY",
        ),
        (
            {"PICKLEBALL_VISION_AUDIO_ANALYSIS_CHANNEL_MODE": "stereo"},
            "PICKLEBALL_VISION_AUDIO_ANALYSIS_CHANNEL_MODE",
        ),
        (
            {
                "PICKLEBALL_VISION_AUDIO_ANALYSIS_FRAME_DURATION_MS": "10",
                "PICKLEBALL_VISION_AUDIO_ANALYSIS_HOP_DURATION_MS": "20",
            },
            "PICKLEBALL_VISION_AUDIO_ANALYSIS_HOP_DURATION_MS",
        ),
        ({"PICKLEBALL_VISION_PERSON_MODEL": "  "}, "PICKLEBALL_VISION_PERSON_MODEL"),
        ({"PICKLEBALL_VISION_PERSON_DEVICE": "tpu"}, "PICKLEBALL_VISION_PERSON_DEVICE"),
        (
            {"PICKLEBALL_VISION_PERSON_MIN_CONFIDENCE": "1.1"},
            "PICKLEBALL_VISION_PERSON_MIN_CONFIDENCE",
        ),
        (
            {"PICKLEBALL_VISION_PERSON_IMAGE_SIZE": "tiny"},
            "PICKLEBALL_VISION_PERSON_IMAGE_SIZE",
        ),
        (
            {"PICKLEBALL_VISION_PERSON_IOU_THRESHOLD": "-0.1"},
            "PICKLEBALL_VISION_PERSON_IOU_THRESHOLD",
        ),
        (
            {"PICKLEBALL_VISION_PERSON_MAX_DETECTIONS": "0"},
            "PICKLEBALL_VISION_PERSON_MAX_DETECTIONS",
        ),
        (
            {"PICKLEBALL_VISION_ISOLATION_NEAR_MARGIN_METERS": "0"},
            "PICKLEBALL_VISION_ISOLATION_NEAR_MARGIN_METERS",
        ),
        (
            {"PICKLEBALL_VISION_ISOLATION_MAX_CANDIDATE_GAP_SECONDS": "never"},
            "PICKLEBALL_VISION_ISOLATION_MAX_CANDIDATE_GAP_SECONDS",
        ),
        (
            {"PICKLEBALL_VISION_ISOLATION_MIN_CANDIDATE_OBSERVATIONS": "1"},
            "PICKLEBALL_VISION_ISOLATION_MIN_CANDIDATE_OBSERVATIONS",
        ),
        (
            {"PICKLEBALL_VISION_ISOLATION_MIN_COURT_SUPPORT_RATIO": "1.1"},
            "PICKLEBALL_VISION_ISOLATION_MIN_COURT_SUPPORT_RATIO",
        ),
        (
            {
                "PICKLEBALL_VISION_TRACKING_LOW_THRESHOLD": "0.8",
                "PICKLEBALL_VISION_TRACKING_HIGH_THRESHOLD": "0.2",
            },
            "PICKLEBALL_VISION_TRACKING_LOW_THRESHOLD",
        ),
        (
            {"PICKLEBALL_VISION_TRACKING_LONG_GAP_APPEARANCE_SIMILARITY": "1.1"},
            "PICKLEBALL_VISION_TRACKING_LONG_GAP_APPEARANCE_SIMILARITY",
        ),
        (
            {"PICKLEBALL_VISION_ANALYSIS_SMOOTHING_WINDOW_FRAMES": "4"},
            "PICKLEBALL_VISION_ANALYSIS_SMOOTHING_WINDOW_FRAMES",
        ),
        (
            {
                "PICKLEBALL_VISION_BALL_TRACKING_MAX_ASSOCIATION_GAP_SECONDS": "0.1",
                "PICKLEBALL_VISION_BALL_TRACKING_MAX_INTERPOLATION_GAP_SECONDS": "0.2",
            },
            "PICKLEBALL_VISION_BALL_TRACKING_MAX_INTERPOLATION_GAP_SECONDS",
        ),
        (
            {"PICKLEBALL_VISION_BALL_TRACKING_SMOOTHING_WINDOW_FRAMES": "4"},
            "PICKLEBALL_VISION_BALL_TRACKING_SMOOTHING_WINDOW_FRAMES",
        ),
    ],
)
def test_invalid_settings_raise_typed_errors(
    environment: dict[str, str],
    setting: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        Settings.from_env(environment)

    assert raised.value.code is ErrorCode.CONFIGURATION
    assert raised.value.details == {"setting": setting}
