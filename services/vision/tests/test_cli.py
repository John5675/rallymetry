import json
from pathlib import Path
from typing import cast

import pytest

from pickleball_vision.calibration import CalibrationCorrespondence
from pickleball_vision.cli import EXIT_OK, EXIT_USAGE_ERROR, main
from pickleball_vision.config import (
    ENV_PREFIX,
    PersonDetectionSettings,
    PlayerIsolationSettings,
)
from pickleball_vision.court import CourtDimensions, ImagePoint, court_landmarks
from pickleball_vision.person_detection_pipeline import PersonDetectionArtifacts
from pickleball_vision.player_isolation_workflow import PlayerIsolationArtifacts


def _clear_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for suffix in (
        "ENVIRONMENT",
        "LOG_LEVEL",
        "LOG_FORMAT",
        "OUTPUT_DIR",
        "AUDIO_VIDEO_OFFSET_MS",
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
