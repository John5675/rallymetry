from pathlib import Path

import pytest

from pickleball_vision.config import (
    ArtifactBackend,
    AudioAnalysisChannelMode,
    AudioAnalysisSettings,
    BallTrackingSettings,
    BounceDetectionSettings,
    ContactDetectionSettings,
    Environment,
    HitterIdentificationSettings,
    LogFormat,
    MatchAnalyticsSettings,
    MediaSettings,
    PersistenceSettings,
    PersonDetectionSettings,
    PlayerAnalysisSettings,
    PlayerIsolationSettings,
    PlayerTrackingSettings,
    RallySegmentationSettings,
    Settings,
    ShotClassificationSettings,
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
    assert settings.rally_segmentation == RallySegmentationSettings()
    assert settings.bounce_detection == BounceDetectionSettings()
    assert settings.contact_detection == ContactDetectionSettings()
    assert settings.hitter_identification == HitterIdentificationSettings()
    assert settings.shot_classification == ShotClassificationSettings()
    assert settings.match_analytics == MatchAnalyticsSettings()
    assert settings.persistence == PersistenceSettings()


def test_settings_load_prefixed_environment_values() -> None:
    settings = Settings.from_env(
        {
            "PICKLEBALL_VISION_ENVIRONMENT": "production",
            "PICKLEBALL_VISION_LOG_LEVEL": "warning",
            "PICKLEBALL_VISION_LOG_FORMAT": "console",
            "PICKLEBALL_VISION_OUTPUT_DIR": "~/pickleball-output",
            "PICKLEBALL_VISION_AUDIO_VIDEO_OFFSET_MS": "-35.5",
            "PICKLEBALL_VISION_FUSION_TOLERANCE_MS": "75",
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
            "PICKLEBALL_VISION_RALLY_END_QUIET_SECONDS": "1.2",
            "PICKLEBALL_VISION_RALLY_RESTART_QUIET_SECONDS": "0.6",
            "PICKLEBALL_VISION_RALLY_DEAD_BALL_HANDOFF_WINDOW_SECONDS": "2.8",
            "PICKLEBALL_VISION_RALLY_DEAD_BALL_HANDOFF_MINIMUM_QUALITY_MARGIN": "0.08",
            "PICKLEBALL_VISION_RALLY_EVALUATION_MINIMUM_IOU": "0.4",
            "PICKLEBALL_VISION_BOUNCE_AUDIO_CONFIDENCE_WEIGHT": "0.25",
            "PICKLEBALL_VISION_BOUNCE_EVALUATION_TOLERANCE_MS": "100",
            "PICKLEBALL_VISION_CONTACT_ACCEPTED_CONFIDENCE": "0.82",
            "PICKLEBALL_VISION_CONTACT_AUDIO_CONFIDENCE_WEIGHT": "0.3",
            "PICKLEBALL_VISION_HITTER_MINIMUM_ASSIGNMENT_CONFIDENCE": "0.7",
            "PICKLEBALL_VISION_HITTER_DIRECTION_WEIGHT": "0.2",
            "PICKLEBALL_VISION_SHOT_MINIMUM_TRAJECTORY_COVERAGE": "0.65",
            "PICKLEBALL_VISION_SHOT_MINIMUM_KNOWN_TRAJECTORY_POINTS": "5",
            "PICKLEBALL_VISION_SHOT_DRIVE_MINIMUM_SPEED_DIAGONALS_PER_SECOND": "0.55",
            "PICKLEBALL_VISION_SHOT_DEBUG_TRAIL_SECONDS": "1.25",
            "PICKLEBALL_VISION_MATCH_ANALYTICS_KITCHEN_ARRIVAL_DISTANCE_METERS": "0.75",
            (
                "PICKLEBALL_VISION_MATCH_ANALYTICS_MINIMUM_KITCHEN_ARRIVAL_JOINT_COVERAGE_RATIO"
            ): "0.6",
            "MONGODB_URL": "mongodb+srv://user:secret@example.test/",
            "MONGODB_DATABASE": "pickleball_test",
            "PICKLEBALL_VISION_ARTIFACT_BACKEND": "vercel_blob",
            "PICKLEBALL_VISION_LOCAL_ARTIFACT_ROOT": "~/pickleball-artifacts",
            "BLOB_READ_WRITE_TOKEN": "blob-secret",
            "PUBLIC_BLOB_READ_WRITE_TOKEN": "public-blob-secret",
        }
    )

    assert settings.environment is Environment.PRODUCTION
    assert settings.log_level == "WARNING"
    assert settings.log_format is LogFormat.CONSOLE
    assert settings.output_dir == Path("~/pickleball-output").expanduser()
    assert settings.media == MediaSettings(
        audio_video_offset_ms=-35.5,
        fusion_tolerance_ms=75.0,
    )
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
    assert settings.rally_segmentation == RallySegmentationSettings(
        end_quiet_seconds=1.2,
        restart_quiet_seconds=0.6,
        dead_ball_handoff_window_seconds=2.8,
        dead_ball_handoff_minimum_quality_margin=0.08,
        evaluation_minimum_iou=0.4,
    )
    assert settings.bounce_detection == BounceDetectionSettings(
        audio_confidence_weight=0.25,
        evaluation_tolerance_ms=100.0,
    )
    assert settings.contact_detection == ContactDetectionSettings(
        accepted_confidence=0.82,
        audio_confidence_weight=0.3,
    )
    assert settings.hitter_identification == HitterIdentificationSettings(
        minimum_assignment_confidence=0.7,
        direction_weight=0.2,
    )
    assert settings.shot_classification == ShotClassificationSettings(
        minimum_trajectory_coverage=0.65,
        minimum_known_trajectory_points=5,
        drive_minimum_speed_diagonals_per_second=0.55,
        debug_trail_seconds=1.25,
    )
    assert settings.match_analytics == MatchAnalyticsSettings(
        kitchen_arrival_distance_m=0.75,
        minimum_kitchen_arrival_joint_coverage_ratio=0.6,
    )
    assert settings.persistence == PersistenceSettings(
        mongodb_url="mongodb+srv://user:secret@example.test/",
        mongodb_database="pickleball_test",
        artifact_backend=ArtifactBackend.VERCEL_BLOB,
        local_artifact_root=Path("~/pickleball-artifacts").expanduser(),
        vercel_blob_token="blob-secret",
        vercel_blob_public_token="public-blob-secret",
    )
    public = settings.public_values()["persistence"]
    assert isinstance(public, dict)
    assert public == {
        "mongodbConfigured": True,
        "mongodbDatabase": "pickleball_test",
        "artifactBackend": "vercel_blob",
        "localArtifactRoot": str(Path("~/pickleball-artifacts").expanduser()),
        "vercelBlobConfigured": True,
        "vercelBlobPublicConfigured": True,
    }
    assert "secret" not in repr(public)


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
        (
            {"PICKLEBALL_VISION_RALLY_EVALUATION_MINIMUM_IOU": "1.1"},
            "PICKLEBALL_VISION_RALLY_EVALUATION_MINIMUM_IOU",
        ),
        (
            {"PICKLEBALL_VISION_RALLY_DEAD_BALL_HANDOFF_MINIMUM_QUALITY_MARGIN": "1.1"},
            "PICKLEBALL_VISION_RALLY_DEAD_BALL_HANDOFF_MINIMUM_QUALITY_MARGIN",
        ),
        (
            {"PICKLEBALL_VISION_BOUNCE_AUDIO_CONFIDENCE_WEIGHT": "1.1"},
            "PICKLEBALL_VISION_BOUNCE_AUDIO_CONFIDENCE_WEIGHT",
        ),
        (
            {"PICKLEBALL_VISION_BOUNCE_MINIMUM_OBSERVATIONS_EACH_SIDE": "1"},
            "PICKLEBALL_VISION_BOUNCE_MINIMUM_OBSERVATIONS_EACH_SIDE",
        ),
        (
            {"PICKLEBALL_VISION_CONTACT_AUDIO_CONFIDENCE_WEIGHT": "1.1"},
            "PICKLEBALL_VISION_CONTACT_AUDIO_CONFIDENCE_WEIGHT",
        ),
        (
            {"PICKLEBALL_VISION_CONTACT_MINIMUM_OBSERVATIONS_EACH_SIDE": "1"},
            "PICKLEBALL_VISION_CONTACT_MINIMUM_OBSERVATIONS_EACH_SIDE",
        ),
        (
            {"PICKLEBALL_VISION_HITTER_MINIMUM_ASSIGNMENT_MARGIN": "0"},
            "PICKLEBALL_VISION_HITTER_MINIMUM_ASSIGNMENT_MARGIN",
        ),
        (
            {"PICKLEBALL_VISION_HITTER_DIRECTION_WEIGHT": "-0.1"},
            "PICKLEBALL_VISION_HITTER_DIRECTION_WEIGHT",
        ),
        (
            {"PICKLEBALL_VISION_SHOT_MINIMUM_KNOWN_TRAJECTORY_POINTS": "1"},
            "PICKLEBALL_VISION_SHOT_MINIMUM_KNOWN_TRAJECTORY_POINTS",
        ),
        (
            {
                "PICKLEBALL_VISION_SHOT_DINK_MAXIMUM_SPEED_DIAGONALS_PER_SECOND": "0.5",
                "PICKLEBALL_VISION_SHOT_DROP_MAXIMUM_SPEED_DIAGONALS_PER_SECOND": "0.4",
            },
            "PICKLEBALL_VISION_SHOT_DINK_MAXIMUM_SPEED_DIAGONALS_PER_SECOND",
        ),
        (
            {"PICKLEBALL_VISION_SHOT_DEBUG_TRAIL_SECONDS": "0"},
            "PICKLEBALL_VISION_SHOT_DEBUG_TRAIL_SECONDS",
        ),
        (
            {
                "PICKLEBALL_VISION_MATCH_ANALYTICS_"
                "MINIMUM_KITCHEN_ARRIVAL_JOINT_COVERAGE_RATIO": "1.1"
            },
            ("PICKLEBALL_VISION_MATCH_ANALYTICS_MINIMUM_KITCHEN_ARRIVAL_JOINT_COVERAGE_RATIO"),
        ),
        (
            {"PICKLEBALL_VISION_FUSION_TOLERANCE_MS": "0"},
            "PICKLEBALL_VISION_FUSION_TOLERANCE_MS",
        ),
        (
            {
                "PICKLEBALL_VISION_RALLY_END_QUIET_SECONDS": "0.5",
                "PICKLEBALL_VISION_RALLY_RESTART_QUIET_SECONDS": "0.6",
            },
            "PICKLEBALL_VISION_RALLY_RESTART_QUIET_SECONDS",
        ),
        ({"MONGODB_URL": "https://example.test"}, "MONGODB_URL"),
        ({"MONGODB_DATABASE": "bad database"}, "MONGODB_DATABASE"),
        (
            {"PICKLEBALL_VISION_ARTIFACT_BACKEND": "s3"},
            "PICKLEBALL_VISION_ARTIFACT_BACKEND",
        ),
        (
            {"PICKLEBALL_VISION_LOCAL_ARTIFACT_ROOT": ""},
            "PICKLEBALL_VISION_LOCAL_ARTIFACT_ROOT",
        ),
        (
            {"PICKLEBALL_VISION_ARTIFACT_BACKEND": "vercel_blob"},
            "BLOB_READ_WRITE_TOKEN",
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
