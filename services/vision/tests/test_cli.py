import json
from pathlib import Path
from typing import cast

import pytest

from pickleball_vision.bounce_detection_workflow import BounceDetectionArtifacts
from pickleball_vision.calibration import CalibrationCorrespondence
from pickleball_vision.cli import EXIT_OK, EXIT_USAGE_ERROR, main
from pickleball_vision.config import (
    ENV_PREFIX,
    PersonDetectionSettings,
    PlayerAnalysisSettings,
    PlayerIsolationSettings,
)
from pickleball_vision.contact_detection_workflow import ContactDetectionArtifacts
from pickleball_vision.court import CourtDimensions, ImagePoint, court_landmarks
from pickleball_vision.match_annotation import MatchAnnotationArtifacts
from pickleball_vision.media import MediaTimeline
from pickleball_vision.person_detection_pipeline import PersonDetectionArtifacts
from pickleball_vision.player_analysis_workflow import PlayerAnalysisArtifacts
from pickleball_vision.player_isolation import LOGICAL_PLAYER_ROLES
from pickleball_vision.player_isolation_workflow import PlayerIsolationArtifacts
from pickleball_vision.rally_segmentation_workflow import RallySegmentationArtifacts


def _clear_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for suffix in (
        "ENVIRONMENT",
        "LOG_LEVEL",
        "LOG_FORMAT",
        "OUTPUT_DIR",
        "AUDIO_VIDEO_OFFSET_MS",
        "FUSION_TOLERANCE_MS",
        "AUDIO_ANALYSIS_SAMPLE_RATE_HZ",
        "AUDIO_ANALYSIS_ONSET_SENSITIVITY",
        "AUDIO_ANALYSIS_MINIMUM_EVENT_SEPARATION_MS",
        "AUDIO_ANALYSIS_CHANNEL_MODE",
        "AUDIO_ANALYSIS_FRAME_DURATION_MS",
        "AUDIO_ANALYSIS_HOP_DURATION_MS",
        "PERSON_MODEL",
        "PERSON_DEVICE",
        "PERSON_MIN_CONFIDENCE",
        "PERSON_IMAGE_SIZE",
        "PERSON_IOU_THRESHOLD",
        "PERSON_MAX_DETECTIONS",
        "ISOLATION_NEAR_MARGIN_METERS",
        "ISOLATION_BOUNDARY_UNCERTAINTY_METERS",
        "ISOLATION_SIDE_UNCERTAINTY_METERS",
        "ISOLATION_MAX_CANDIDATE_GAP_SECONDS",
        "ISOLATION_MAX_CANDIDATE_SPEED_MPS",
        "ISOLATION_MIN_CANDIDATE_OBSERVATIONS",
        "ISOLATION_MIN_COURT_SUPPORT_RATIO",
        "TRACKING_HIGH_THRESHOLD",
        "TRACKING_LOW_THRESHOLD",
        "TRACKING_NEW_THRESHOLD",
        "TRACKING_MATCH_THRESHOLD",
        "TRACKING_BUFFER_SECONDS",
        "TRACKING_MAX_IDENTITY_GAP_SECONDS",
        "TRACKING_MAX_PLAYER_SPEED_MPS",
        "TRACKING_MINIMUM_IDENTITY_SCORE",
        "TRACKING_SUSPECTED_SWITCH_SCORE",
        "TRACKING_APPEARANCE_WEIGHT",
        "TRACKING_MINIMUM_APPEARANCE_SIMILARITY",
        "TRACKING_MINIMUM_APPEARANCE_MARGIN",
        "TRACKING_APPEARANCE_PROTOTYPE_WINDOW_SECONDS",
        "TRACKING_LONG_GAP_APPEARANCE_SIMILARITY",
        "TRACKING_LONG_GAP_MINIMUM_APPEARANCE_MARGIN",
        "ANALYSIS_MINIMUM_TRACKING_CONFIDENCE",
        "ANALYSIS_SMOOTHING_WINDOW_FRAMES",
        "ANALYSIS_MAXIMUM_SMOOTHING_ADJUSTMENT_METERS",
        "ANALYSIS_MAXIMUM_STEP_GAP_SECONDS",
        "ANALYSIS_MAXIMUM_STEP_SPEED_MPS",
        "ANALYSIS_TRANSITION_ZONE_DEPTH_METERS",
        "ANALYSIS_TOPDOWN_TRAIL_SECONDS",
        "BALL_TRACKING_MAX_ASSOCIATION_GAP_SECONDS",
        "BALL_TRACKING_MAX_INTERPOLATION_GAP_SECONDS",
        "BALL_TRACKING_MAX_SPEED_DIAGONALS_PER_SECOND",
        "BALL_TRACKING_MAX_ACCELERATION_DIAGONALS_PER_SECOND_SQUARED",
        "BALL_TRACKING_BASE_GATE_DIAGONAL_FRACTION",
        "BALL_TRACKING_COURT_SIDE_MARGIN_FRACTION",
        "BALL_TRACKING_COURT_AIR_MARGIN_FRACTION",
        "BALL_TRACKING_COURT_BOTTOM_MARGIN_FRACTION",
        "BALL_TRACKING_MINIMUM_START_SCORE",
        "BALL_TRACKING_MINIMUM_ASSOCIATION_SCORE",
        "BALL_TRACKING_MINIMUM_SEGMENT_OBSERVATIONS",
        "BALL_TRACKING_SMOOTHING_WINDOW_FRAMES",
        "BALL_TRACKING_MAXIMUM_SMOOTHING_ADJUSTMENT_DIAGONAL_FRACTION",
        "BALL_TRACKING_DEBUG_TRAIL_SECONDS",
        "RALLY_MINIMUM_MOTION_SPEED_DIAGONALS_PER_SECOND",
        "RALLY_MOTION_LINK_GAP_SECONDS",
        "RALLY_MOTION_SUPPORT_WINDOW_SECONDS",
        "RALLY_MINIMUM_MOTION_SUPPORT_FRACTION",
        "RALLY_SERVE_MINIMUM_SPEED_DIAGONALS_PER_SECOND",
        "RALLY_SERVE_SPEED_SURGE_RATIO",
        "RALLY_SERVE_BASELINE_WINDOW_SECONDS",
        "RALLY_SERVE_CONFIRMATION_SECONDS",
        "RALLY_SERVE_MINIMUM_DISPLACEMENT_DIAGONAL_FRACTION",
        "RALLY_SERVE_MINIMUM_MOTION_FRACTION",
        "RALLY_MINIMUM_DURATION_SECONDS",
        "RALLY_MAXIMUM_DURATION_SECONDS",
        "RALLY_END_QUIET_SECONDS",
        "RALLY_END_TAIL_GRACE_SECONDS",
        "RALLY_MINIMUM_BETWEEN_SECONDS",
        "RALLY_RESTART_QUIET_SECONDS",
        "RALLY_RESTART_MINIMUM_ELAPSED_SECONDS",
        "RALLY_DEAD_BALL_HANDOFF_WINDOW_SECONDS",
        "RALLY_DEAD_BALL_HANDOFF_MINIMUM_QUALITY_MARGIN",
        "RALLY_DEAD_BALL_HANDOFF_FULL_DURATION_SECONDS",
        "RALLY_PLAYER_RESET_WINDOW_SECONDS",
        "RALLY_PLAYER_RESET_MAXIMUM_SPEED_MPS",
        "RALLY_AUDIO_SUPPORT_TOLERANCE_SECONDS",
        "RALLY_EVALUATION_MINIMUM_IOU",
        "RALLY_EVALUATION_BOUNDARY_TOLERANCE_SECONDS",
        "RALLY_SPARSE_EVALUATION_MARGIN_SECONDS",
        "BOUNCE_TRAJECTORY_WINDOW_SECONDS",
        "BOUNCE_MINIMUM_OBSERVATIONS_EACH_SIDE",
        "BOUNCE_MINIMUM_VERTICAL_SPEED_DIAGONALS_PER_SECOND",
        "BOUNCE_MINIMUM_VERTICAL_REVERSAL_DIAGONALS_PER_SECOND",
        "BOUNCE_MINIMUM_SHAPE_PROMINENCE_DIAGONAL_FRACTION",
        "BOUNCE_MINIMUM_CONTINUITY_FRACTION",
        "BOUNCE_MINIMUM_VISUAL_CANDIDATE_CONFIDENCE",
        "BOUNCE_ACCEPTED_CONFIDENCE",
        "BOUNCE_PLANE_PROJECTION_MINIMUM_VISUAL_CONFIDENCE",
        "BOUNCE_MINIMUM_BETWEEN_SECONDS",
        "BOUNCE_AUDIO_CONFIDENCE_WEIGHT",
        "BOUNCE_RALLY_SEQUENCE_CONFIDENCE_BOOST",
        "BOUNCE_EVALUATION_TOLERANCE_MS",
        "BOUNCE_SPARSE_EVALUATION_MARGIN_SECONDS",
    ):
        monkeypatch.delenv(f"{ENV_PREFIX}{suffix}", raising=False)


def test_doctor_reports_valid_foundation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)

    exit_code = main(["doctor"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    log_record = json.loads(captured.err)
    assert exit_code == EXIT_OK
    assert report["service"] == "pickleball-vision"
    assert report["status"] == "ok"
    assert report["configuration"]["environment"] == "development"
    assert log_record["event"] == "foundation_check_complete"


def test_invalid_configuration_has_stable_error_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)
    monkeypatch.setenv("PICKLEBALL_VISION_LOG_FORMAT", "invalid")

    exit_code = main(["doctor"])

    captured = capsys.readouterr()
    assert exit_code == EXIT_USAGE_ERROR
    assert captured.out == ""
    assert "error [configuration_error]" in captured.err


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == EXIT_OK
    assert "doctor" in captured.out


def test_ball_annotation_template_command_preserves_unreviewed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "ball_dataset_split_assignments",
                "frames": [
                    {"record_id": "source:frame:1"},
                    {"record_id": "source:frame:2"},
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "annotations.json"

    exit_code = main(
        [
            "ball",
            "create-annotation-template",
            str(split_path),
            "--dataset-version",
            "dataset-v1",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    annotations = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == EXIT_OK
    assert report["frame_count"] == 2
    assert annotations["dataset_version"] == "dataset-v1"
    assert all(frame["review_status"] == "unreviewed" for frame in annotations["frames"])


def test_inspect_command_outputs_video_metadata(
    synthetic_video: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)

    exit_code = main(["inspect", str(synthetic_video)])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    log_record = json.loads(captured.err)
    assert exit_code == EXIT_OK
    assert report["filename"] == "synthetic.avi"
    assert report["path"] == str(synthetic_video.resolve())
    assert report["width"] == 96
    assert report["height"] == 64
    assert report["fps"] == pytest.approx(7.5)
    assert report["frame_count"] == 12
    assert report["duration"] == pytest.approx(1.6)
    assert "codec" in report
    assert report["hasAudio"] is False
    assert report["audioCodec"] is None
    assert report["audioSampleRate"] is None
    assert report["audioChannels"] is None
    assert report["audioDuration"] is None
    assert report["audioStartTime"] is None
    assert "videoStartTime" in report
    assert report["audioVideoOffsetMs"] == 0.0
    assert log_record["event"] == "video_inspected"


def test_inspect_and_extract_audio_commands_report_synchronized_metadata(
    synthetic_media_with_audio: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)
    monkeypatch.setenv("PICKLEBALL_VISION_AUDIO_VIDEO_OFFSET_MS", "25")

    inspect_exit = main(["inspect", str(synthetic_media_with_audio)])
    inspect_capture = capsys.readouterr()
    inspect_report = json.loads(inspect_capture.out)
    assert inspect_exit == EXIT_OK
    assert inspect_report["hasAudio"] is True
    assert inspect_report["audioSampleRate"] == 48000
    assert inspect_report["audioChannels"] == 1
    assert inspect_report["audioVideoOffsetMs"] == 25.0

    output = tmp_path / "audio" / "analysis.wav"
    extract_exit = main(
        [
            "extract-audio",
            str(synthetic_media_with_audio),
            "--output",
            str(output),
            "--sample-rate",
            "24000",
            "--channels",
            "2",
        ]
    )
    extract_capture = capsys.readouterr()
    extract_report = json.loads(extract_capture.out)
    assert extract_exit == EXIT_OK
    assert output.is_file()
    assert Path(extract_report["metadataPath"]).is_file()
    assert extract_report["analysisAudio"]["sampleRate"] == 24000
    assert extract_report["analysisAudio"]["channels"] == 2
    assert extract_report["timeline"]["audioVideoOffsetMs"] == 25.0


def test_extract_audio_command_reports_video_without_audio(
    synthetic_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)

    exit_code = main(
        [
            "extract-audio",
            str(synthetic_video),
            "--output",
            str(tmp_path / "audio.wav"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == EXIT_USAGE_ERROR
    assert captured.out == ""
    assert "error [audio_stream_not_found]" in captured.err


def test_extract_frame_command_writes_requested_image(
    synthetic_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)
    output_path = tmp_path / "extracted" / "frame.jpg"

    exit_code = main(
        [
            "extract-frame",
            str(synthetic_video),
            "--timestamp",
            "0.8",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == EXIT_OK
    assert output_path.is_file()
    assert report["requested_timestamp"] == 0.8
    assert report["frame"]["frame_index"] == 6
    assert report["frame"]["width"] == 96
    assert report["frame"]["height"] == 64


def test_sample_frames_command_spans_source_duration(
    synthetic_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)
    output_dir = tmp_path / "samples"

    exit_code = main(
        [
            "sample-frames",
            str(synthetic_video),
            "--count",
            "4",
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == EXIT_OK
    assert report["count"] == 4
    assert [frame["frame_index"] for frame in report["frames"]] == [0, 4, 7, 11]
    assert len(tuple(output_dir.glob("*.jpg"))) == 4


def test_video_command_returns_useful_error_for_missing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)

    exit_code = main(["inspect", str(tmp_path / "missing.mp4")])

    captured = capsys.readouterr()
    assert exit_code == EXIT_USAGE_ERROR
    assert captured.out == ""
    assert "error [video_not_found]" in captured.err
    assert "Video file does not exist" in captured.err


def test_calibrate_command_writes_json_and_debug_artifacts_without_automatic_detection(
    synthetic_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)
    landmarks = court_landmarks(CourtDimensions())
    image_points = (
        ImagePoint(5, 58),
        ImagePoint(91, 58),
        ImagePoint(30, 5),
        ImagePoint(66, 5),
    )
    selected_landmarks = (landmarks[0], landmarks[1], landmarks[8], landmarks[9])
    selections = tuple(
        CalibrationCorrespondence(
            landmark=landmark.name,
            label=landmark.label,
            image_point=image_point,
            court_point=landmark.court_point,
        )
        for landmark, image_point in zip(selected_landmarks, image_points, strict=True)
    )
    monkeypatch.setattr(
        "pickleball_vision.calibration_workflow.select_court_landmarks",
        lambda _frame, _dimensions: selections,
    )
    output_path = tmp_path / "calibration" / "calibration.json"

    exit_code = main(
        [
            "calibrate",
            str(synthetic_video),
            "--timestamp",
            "0.8",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == EXIT_OK
    assert report["correspondence_count"] == 4
    assert report["inlier_count"] == 4
    assert report["fit_method"] == "direct_four_point"
    assert report["quality"]["status"] == "warning"
    assert output_path.is_file()
    assert (output_path.parent / "calibration-overlay.jpg").is_file()
    assert (output_path.parent / "court-topdown.jpg").is_file()


def test_calibrate_command_rejects_invalid_timestamp_before_opening_ui(
    synthetic_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)

    exit_code = main(
        [
            "calibrate",
            str(synthetic_video),
            "--timestamp",
            "1000",
            "--output",
            str(tmp_path / "calibration.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == EXIT_USAGE_ERROR
    assert captured.out == ""
    assert "error [invalid_timestamp]" in captured.err


def test_detect_people_command_dispatches_with_external_configuration(
    synthetic_video: Path,
    synthetic_calibration: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)
    monkeypatch.setenv("PICKLEBALL_VISION_PERSON_MIN_CONFIDENCE", "0.12")
    output_dir = tmp_path / "detections"
    received: dict[str, object] = {}

    def fake_detect_people(
        video_path: Path,
        *,
        calibration_path: Path,
        output_dir: Path,
        settings: PersonDetectionSettings,
    ) -> PersonDetectionArtifacts:
        received.update(
            {
                "video_path": video_path,
                "calibration_path": calibration_path,
                "output_dir": output_dir,
                "settings": settings,
            }
        )
        return PersonDetectionArtifacts(
            detections_path=output_dir / "detections.json",
            annotated_video_path=output_dir / "annotated.mp4",
            summary_path=output_dir / "summary.json",
            processed_frame_count=12,
            detection_count=25,
        )

    monkeypatch.setattr("pickleball_vision.cli.detect_people_in_video", fake_detect_people)

    exit_code = main(
        [
            "detect-people",
            str(synthetic_video),
            "--calibration",
            str(synthetic_calibration),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == EXIT_OK
    assert report["processed_frame_count"] == 12
    assert report["detection_count"] == 25
    assert received["video_path"] == synthetic_video
    assert received["calibration_path"] == synthetic_calibration
    assert received["output_dir"] == output_dir
    received_settings = cast(PersonDetectionSettings, received["settings"])
    assert received_settings.min_confidence == pytest.approx(0.12)


def test_isolate_players_command_dispatches_manual_workflow(
    synthetic_video: Path,
    synthetic_calibration: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)
    detections_path = tmp_path / "detections.json"
    assignments_path = tmp_path / "existing-assignments.json"
    output_dir = tmp_path / "isolation"
    received: dict[str, object] = {}

    def fake_isolate(
        video_path: Path,
        *,
        detections_path: Path,
        calibration_path: Path,
        selection_timestamp_s: float,
        output_dir: Path,
        settings: PlayerIsolationSettings,
        existing_assignments_path: Path | None,
    ) -> PlayerIsolationArtifacts:
        received.update(locals())
        return PlayerIsolationArtifacts(
            candidates_path=output_dir / "player-candidates.json",
            assignments_path=output_dir / "player-assignments.json",
            debug_video_path=output_dir / "primary-player-debug.mp4",
            summary_path=output_dir / "primary-player-summary.json",
            candidate_count=12,
            eligible_candidate_count=6,
        )

    monkeypatch.setattr("pickleball_vision.cli.isolate_primary_players", fake_isolate)

    exit_code = main(
        [
            "isolate-players",
            str(synthetic_video),
            "--detections",
            str(detections_path),
            "--calibration",
            str(synthetic_calibration),
            "--timestamp",
            "0.5",
            "--output-dir",
            str(output_dir),
            "--assignments",
            str(assignments_path),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == EXIT_OK
    assert report["candidate_count"] == 12
    assert report["eligible_candidate_count"] == 6
    assert received["video_path"] == synthetic_video
    assert received["detections_path"] == detections_path
    assert received["calibration_path"] == synthetic_calibration
    assert received["selection_timestamp_s"] == pytest.approx(0.5)
    assert received["output_dir"] == output_dir
    assert received["existing_assignments_path"] == assignments_path


def test_analyze_players_command_dispatches_release_workflow(
    synthetic_video: Path,
    synthetic_calibration: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)
    monkeypatch.setenv("PICKLEBALL_VISION_ANALYSIS_SMOOTHING_WINDOW_FRAMES", "7")
    tracks_path = tmp_path / "tracks.json"
    corrections_path = tmp_path / "player-position-corrections.json"
    output_dir = tmp_path / "analysis"
    received: dict[str, object] = {}

    def fake_analyze(
        video_path: Path,
        *,
        calibration_path: Path,
        output_dir: Path,
        settings: PlayerAnalysisSettings,
        tracks_path: Path | None,
        position_corrections_path: Path | None,
    ) -> PlayerAnalysisArtifacts:
        received.update(locals())
        return PlayerAnalysisArtifacts(
            output_dir / "player_positions.json",
            output_dir / "summary.json",
            output_dir / "annotated.mp4",
            output_dir / "topdown.mp4",
            {role: output_dir / f"heatmap-{role.value}.png" for role in LOGICAL_PLAYER_ROLES},
            12,
        )

    monkeypatch.setattr("pickleball_vision.cli.analyze_players_in_video", fake_analyze)

    exit_code = main(
        [
            "analyze-players",
            str(synthetic_video),
            "--calibration",
            str(synthetic_calibration),
            "--output-dir",
            str(output_dir),
            "--tracks",
            str(tracks_path),
            "--position-corrections",
            str(corrections_path),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == EXIT_OK
    assert report["frames_processed"] == 12
    assert report["release_version"] == "0.1"
    assert received["video_path"] == synthetic_video
    assert received["calibration_path"] == synthetic_calibration
    assert received["output_dir"] == output_dir
    assert received["tracks_path"] == tracks_path
    assert received["position_corrections_path"] == corrections_path
    received_settings = cast(PlayerAnalysisSettings, received["settings"])
    assert received_settings.smoothing_window_frames == 7


def test_dataset_commands_extract_and_split_without_model_inference(
    synthetic_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)
    output_dir = tmp_path / "dataset"

    extract_exit = main(
        [
            "dataset",
            "extract-frames",
            str(synthetic_video),
            "--output-dir",
            str(output_dir),
            "--every",
            "4",
            "--label-group",
            "positive",
            "--group-id",
            "rally-001",
        ]
    )

    extract_report = json.loads(capsys.readouterr().out)
    assert extract_exit == EXIT_OK
    assert extract_report["frame_count"] == 3
    assert extract_report["label_group_counts"]["positive"] == 3

    split_path = tmp_path / "split.json"
    split_exit = main(
        [
            "dataset",
            "split",
            str(output_dir / "dataset-manifest.json"),
            "--output",
            str(split_path),
            "--by",
            "group",
            "--train",
            "1",
            "--validation",
            "0",
            "--test",
            "0",
        ]
    )

    split_report = json.loads(capsys.readouterr().out)
    assert split_exit == EXIT_OK
    assert split_report["frame_count"] == 3
    assert split_report["unit_count"] == 1
    assert split_report["split_counts"] == {"test": 0, "train": 3, "validation": 0}


def test_analyze_audio_command_gracefully_reports_video_without_audio(
    synthetic_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)
    output_dir = tmp_path / "audio-analysis"

    exit_code = main(
        [
            "analyze-audio",
            str(synthetic_video),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    log_record = json.loads(captured.err)
    assert exit_code == EXIT_OK
    assert report["audioAnalysisAvailable"] is False
    assert report["analysisAudioPath"] is None
    assert Path(report["eventsPath"]).is_file()
    assert Path(report["summaryPath"]).is_file()
    assert Path(report["waveformPath"]).is_file()
    assert Path(report["eventsImagePath"]).is_file()
    assert log_record["event"] == "audio_analyzed"


def test_annotate_match_command_routes_optional_audio_context(
    synthetic_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)
    monkeypatch.setenv("PICKLEBALL_VISION_AUDIO_VIDEO_OFFSET_MS", "30")
    output_path = tmp_path / "annotations.json"
    audio_events_path = tmp_path / "audio-events.json"
    received: dict[str, object] = {}

    def fake_serve_match_annotation(
        video_path: Path,
        **kwargs: object,
    ) -> MatchAnnotationArtifacts:
        received["video_path"] = video_path
        received.update(kwargs)
        return MatchAnnotationArtifacts(
            url="http://127.0.0.1:54321/",
            annotations_path=output_path,
            event_count=7,
            audio_context_available=True,
        )

    monkeypatch.setattr(
        "pickleball_vision.cli.serve_match_annotation",
        fake_serve_match_annotation,
    )

    exit_code = main(
        [
            "annotate-match",
            str(synthetic_video),
            "--output",
            str(output_path),
            "--audio-events",
            str(audio_events_path),
            "--port",
            "0",
            "--no-open",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    log_record = json.loads(captured.err)
    assert exit_code == EXIT_OK
    assert report["eventCount"] == 7
    assert report["audioContextAvailable"] is True
    assert received["video_path"] == synthetic_video
    assert received["output_path"] == output_path
    assert received["audio_events_path"] == audio_events_path
    assert received["port"] == 0
    assert received["open_browser"] is False
    timeline = cast(MediaTimeline, received["timeline"])
    assert timeline.audio_video_offset_ms == 30.0
    assert log_record["event"] == "match_annotation_stopped"


def test_segment_rallies_command_routes_structured_inputs(
    synthetic_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)
    ball_tracks = tmp_path / "ball_tracks.json"
    player_tracks = tmp_path / "tracks.json"
    audio_events = tmp_path / "audio-events.json"
    annotations = tmp_path / "annotations.json"
    output_dir = tmp_path / "rallies"
    received: dict[str, object] = {}

    def fake_segment_rallies(video_path: Path, **kwargs: object) -> RallySegmentationArtifacts:
        received["video_path"] = video_path
        received.update(kwargs)
        return RallySegmentationArtifacts(
            rallies_path=output_dir / "rallies.json",
            debug_video_path=output_dir / "rally-debug.mp4",
            evaluation_path=output_dir / "rally-evaluation.json",
            rally_count=8,
            matched_rally_count=4,
            missed_rally_count=1,
            false_rally_count=2,
            rejected_adjacent_burst_count=3,
        )

    monkeypatch.setattr(
        "pickleball_vision.cli.segment_rallies_in_video",
        fake_segment_rallies,
    )
    exit_code = main(
        [
            "segment-rallies",
            str(synthetic_video),
            "--ball-tracks",
            str(ball_tracks),
            "--player-tracks",
            str(player_tracks),
            "--audio-events",
            str(audio_events),
            "--annotations",
            str(annotations),
            "--annotations-complete",
            "--evaluation-partition",
            "test",
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    log_record = json.loads(captured.err)
    assert exit_code == EXIT_OK
    assert report["rallyCount"] == 8
    assert report["matchedRallyCount"] == 4
    assert report["rejectedAdjacentBurstCount"] == 3
    assert received["video_path"] == synthetic_video
    assert received["ball_tracks_path"] == ball_tracks
    assert received["player_tracks_path"] == player_tracks
    assert received["audio_events_path"] == audio_events
    assert received["annotations_path"] == annotations
    assert received["annotations_complete"] is True
    assert received["evaluation_partition"] == "test"
    assert log_record["event"] == "rallies_segmented"


def test_detect_bounces_command_routes_visual_and_optional_multimodal_inputs(
    synthetic_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)
    monkeypatch.setenv("PICKLEBALL_VISION_AUDIO_VIDEO_OFFSET_MS", "45")
    ball_tracks = tmp_path / "ball_tracks.json"
    calibration = tmp_path / "calibration.json"
    rallies = tmp_path / "rallies.json"
    audio_events = tmp_path / "audio-events.json"
    annotations = tmp_path / "annotations.json"
    output_dir = tmp_path / "bounces"
    received: dict[str, object] = {}

    def fake_detect_bounces(video_path: Path, **kwargs: object) -> BounceDetectionArtifacts:
        received["video_path"] = video_path
        received.update(kwargs)
        return BounceDetectionArtifacts(
            bounces_path=output_dir / "bounces.json",
            debug_video_path=output_dir / "bounce-debug.mp4",
            evaluation_path=output_dir / "bounce-evaluation.json",
            visual_candidate_count=12,
            accepted_bounce_count=8,
            fused_candidate_count=5,
        )

    monkeypatch.setattr("pickleball_vision.cli.detect_bounces_in_video", fake_detect_bounces)
    exit_code = main(
        [
            "detect-bounces",
            str(synthetic_video),
            "--ball-tracks",
            str(ball_tracks),
            "--calibration",
            str(calibration),
            "--rallies",
            str(rallies),
            "--audio-events",
            str(audio_events),
            "--annotations",
            str(annotations),
            "--annotations-complete",
            "--evaluation-partition",
            "test",
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    log_record = json.loads(captured.err)
    assert exit_code == EXIT_OK
    assert report["acceptedBounceCount"] == 8
    assert received["video_path"] == synthetic_video
    assert received["ball_tracks_path"] == ball_tracks
    assert received["calibration_path"] == calibration
    assert received["rallies_path"] == rallies
    assert received["audio_events_path"] == audio_events
    assert received["annotations_path"] == annotations
    assert received["annotations_complete"] is True
    assert received["evaluation_partition"] == "test"
    timeline = cast(MediaTimeline, received["timeline"])
    assert timeline.audio_video_offset_ms == 45.0
    assert log_record["event"] == "bounces_detected"


def test_detect_contacts_command_routes_visual_and_optional_multimodal_inputs(
    synthetic_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_settings(monkeypatch)
    monkeypatch.setenv("PICKLEBALL_VISION_AUDIO_VIDEO_OFFSET_MS", "35")
    ball_tracks = tmp_path / "ball_tracks.json"
    player_tracks = tmp_path / "tracks.json"
    rallies = tmp_path / "rallies.json"
    bounces = tmp_path / "bounces.json"
    audio_events = tmp_path / "audio-events.json"
    annotations = tmp_path / "annotations.json"
    output_dir = tmp_path / "contacts"
    received: dict[str, object] = {}

    def fake_detect_contacts(video_path: Path, **kwargs: object) -> ContactDetectionArtifacts:
        received["video_path"] = video_path
        received.update(kwargs)
        return ContactDetectionArtifacts(
            contacts_path=output_dir / "contacts.json",
            debug_video_path=output_dir / "contact-debug.mp4",
            evaluation_path=output_dir / "contact-evaluation.json",
            visual_candidate_count=14,
            accepted_contact_count=9,
            fused_candidate_count=6,
        )

    monkeypatch.setattr("pickleball_vision.cli.detect_contacts_in_video", fake_detect_contacts)
    exit_code = main(
        [
            "detect-contacts",
            str(synthetic_video),
            "--ball-tracks",
            str(ball_tracks),
            "--player-tracks",
            str(player_tracks),
            "--rallies",
            str(rallies),
            "--bounces",
            str(bounces),
            "--audio-events",
            str(audio_events),
            "--annotations",
            str(annotations),
            "--annotations-complete",
            "--evaluation-partition",
            "test",
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    log_record = json.loads(captured.err)
    assert exit_code == EXIT_OK
    assert report["acceptedContactCount"] == 9
    assert received["video_path"] == synthetic_video
    assert received["ball_tracks_path"] == ball_tracks
    assert received["player_tracks_path"] == player_tracks
    assert received["rallies_path"] == rallies
    assert received["bounces_path"] == bounces
    assert received["audio_events_path"] == audio_events
    assert received["annotations_path"] == annotations
    assert received["annotations_complete"] is True
    assert received["evaluation_partition"] == "test"
    timeline = cast(MediaTimeline, received["timeline"])
    assert timeline.audio_video_offset_ms == 35.0
    assert log_record["event"] == "contacts_detected"
