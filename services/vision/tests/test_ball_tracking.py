import json
from dataclasses import replace
from pathlib import Path

import cv2
import pytest

from pickleball_vision.ball_tracking import (
    BallTrackingCandidate,
    BallTrajectoryStatus,
    ball_box_center,
    build_court_image_envelope,
    reconstruct_ball_trajectory,
    trajectory_summary,
)
from pickleball_vision.ball_tracking_workflow import (
    _validate_calibration_provenance,
    load_ball_detection_candidates,
    track_ball_in_video,
)
from pickleball_vision.calibration import load_calibration
from pickleball_vision.config import BallTrackingSettings
from pickleball_vision.court import ImagePoint
from pickleball_vision.errors import BallTrackingInputError
from pickleball_vision.person_detection import BoundingBox
from pickleball_vision.video import inspect_video


def _candidate(
    frame_number: int,
    x_px: float,
    y_px: float,
    *,
    confidence: float = 0.70,
    relevance: float = 1.0,
    detection_id: str | None = None,
    fps: float = 30.0,
) -> BallTrackingCandidate:
    box = BoundingBox(x_px - 2, y_px - 2, x_px + 2, y_px + 2)
    return BallTrackingCandidate(
        detection_id=detection_id or f"ball-{frame_number}-{x_px}",
        frame_number=frame_number,
        timestamp_s=frame_number / fps,
        bounding_box=box,
        confidence=confidence,
        image_point=ball_box_center(box),
        primary_court_relevance=relevance,
    )


def test_association_prefers_continuity_over_higher_confidence_distractor() -> None:
    frames: list[tuple[BallTrackingCandidate, ...]] = []
    for frame_number in range(8):
        primary = _candidate(frame_number, 100 + frame_number * 5, 200 - frame_number * 2)
        distractors = (
            (
                _candidate(
                    frame_number,
                    500 + frame_number,
                    100,
                    confidence=0.99,
                    detection_id=f"neighbor-{frame_number}",
                ),
            )
            if frame_number
            else ()
        )
        frames.append((primary, *distractors))

    trajectory = reconstruct_ball_trajectory(
        frames,
        fps=30,
        frame_width_px=640,
        frame_height_px=480,
        settings=BallTrackingSettings(),
    )

    assert all(frame.status is BallTrajectoryStatus.OBSERVED for frame in trajectory.frames)
    assert all(
        frame.source_detection_id is not None and frame.source_detection_id.startswith("ball-")
        for frame in trajectory.frames
    )
    assert trajectory.rejected_candidate_count == 7


def test_court_relevance_uses_an_image_envelope_without_ball_projection(
    synthetic_video: Path,
    synthetic_calibration: Path,
) -> None:
    metadata = inspect_video(synthetic_video)
    envelope = build_court_image_envelope(
        load_calibration(synthetic_calibration),
        frame_width_px=metadata.width,
        frame_height_px=metadata.height,
        settings=BallTrackingSettings(),
    )

    assert envelope.relevance(ImagePoint(48, 30)) == 1.0
    assert envelope.relevance(ImagePoint(500, 30)) == 0.0
    assert envelope.as_dict()["airborne_ball_homography_projection"] is False


def test_calibration_provenance_tolerates_small_container_fps_rounding(
    synthetic_video: Path,
    synthetic_calibration: Path,
) -> None:
    metadata = inspect_video(synthetic_video)
    calibration = load_calibration(synthetic_calibration)
    rounded = replace(
        calibration,
        source=replace(calibration.source, fps=metadata.fps * 0.999),
    )

    _validate_calibration_provenance(rounded, metadata)


def test_short_gap_is_interpolated_but_long_gap_remains_unknown() -> None:
    frames: list[tuple[BallTrackingCandidate, ...]] = [tuple() for _ in range(13)]
    for frame_number in (0, 1, 3, 11, 12):
        frames[frame_number] = (
            _candidate(frame_number, 100 + frame_number * 4, 200 - frame_number),
        )

    trajectory = reconstruct_ball_trajectory(
        frames,
        fps=30,
        frame_width_px=640,
        frame_height_px=480,
        settings=BallTrackingSettings(),
    )

    assert trajectory.frames[2].status is BallTrajectoryStatus.INTERPOLATED
    assert trajectory.frames[2].raw_image_point is None
    assert trajectory.frames[2].interpolated_image_point is not None
    assert all(
        trajectory.frames[index].status is BallTrajectoryStatus.UNKNOWN for index in range(4, 11)
    )
    summary = trajectory_summary(trajectory, fps=30)
    assert summary["longest_missing_interval"] == {
        "start_frame": 4,
        "end_frame": 10,
        "frame_count": 7,
        "duration_seconds": 7 / 30,
    }
    assert summary["interpolated_fraction"] == pytest.approx(1 / 6)


def test_acceleration_outlier_is_rejected_and_raw_points_survive_smoothing() -> None:
    frames = [
        (_candidate(0, 100, 200),),
        (_candidate(1, 105, 200),),
        (
            _candidate(2, 110, 200),
            _candidate(2, 250, 200, confidence=0.99, detection_id="acceleration-outlier"),
        ),
        (_candidate(3, 115, 200),),
        (_candidate(4, 120, 200),),
    ]
    settings = replace(
        BallTrackingSettings(),
        maximum_speed_diagonals_per_second=10,
        maximum_acceleration_diagonals_per_second_squared=20,
    )

    trajectory = reconstruct_ball_trajectory(
        frames,
        fps=30,
        frame_width_px=640,
        frame_height_px=480,
        settings=settings,
    )

    frame_two = trajectory.frames[2]
    assert frame_two.source_detection_id != "acceleration-outlier"
    assert "acceleration-outlier" in frame_two.rejected_detection_ids
    assert frame_two.raw_image_point is not None
    assert frame_two.raw_image_point.x_px == 110
    assert frame_two.smoothed_image_point is not None
    maximum_adjustment = 800 * settings.maximum_smoothing_adjustment_diagonal_fraction
    assert abs(frame_two.smoothed_image_point.x_px - frame_two.raw_image_point.x_px) <= (
        maximum_adjustment
    )


def _raw_detection_payload(video_path: Path) -> dict[str, object]:
    metadata = inspect_video(video_path)
    frames = []
    for frame_number in range(metadata.frame_count):
        left_px = 40 + frame_number
        frames.append(
            {
                "frame_number": frame_number,
                "timestamp_s": frame_number / metadata.fps,
                "regions": [],
                "region_predictions": [],
                "detections": [
                    {
                        "detection_id": f"frame-{frame_number:09d}-ball-0000",
                        "frame_number": frame_number,
                        "timestamp_s": frame_number / metadata.fps,
                        "bounding_box": {
                            "left_px": left_px,
                            "top_px": 28,
                            "right_px": left_px + 4,
                            "bottom_px": 32,
                        },
                        "confidence": 0.85,
                        "supporting_prediction_ids": [f"proposal-{frame_number}"],
                        "observation_status": "observed",
                        "temporal_track_id": None,
                    }
                ],
            }
        )
    return {
        "schema_version": 1,
        "record_type": "raw_pickleball_detections",
        "created_at_utc": "2026-08-15T00:00:00+00:00",
        "source": metadata.as_dict(),
        "detector": {"model_version": "synthetic-v1"},
        "strategy": {"name": "synthetic-full-frame"},
        "calibration": {"path": None, "usage": "not_used"},
        "coordinate_system": {"unit": "source_frame_pixels"},
        "temporal_processing": {
            "tracking": False,
            "interpolation": False,
            "events": False,
        },
        "frames": frames,
    }


def test_workflow_validates_provenance_and_writes_all_artifacts(
    synthetic_video: Path,
    synthetic_calibration: Path,
    tmp_path: Path,
) -> None:
    detections_path = tmp_path / "detections.json"
    detections_path.write_text(
        json.dumps(_raw_detection_payload(synthetic_video)),
        encoding="utf-8",
    )
    output = tmp_path / "tracking"

    artifacts = track_ball_in_video(
        synthetic_video,
        detections_path=detections_path,
        calibration_path=synthetic_calibration,
        output_dir=output,
        settings=BallTrackingSettings(),
    )

    assert artifacts.tracks_path.is_file()
    assert artifacts.debug_video_path.is_file()
    assert artifacts.summary_path.is_file()
    tracks = json.loads(artifacts.tracks_path.read_text(encoding="utf-8"))
    assert tracks["record_type"] == "primary_match_ball_trajectory"
    assert tracks["coordinate_system"]["court_coordinates"] is None
    assert tracks["inputs"]["court_calibration"]["airborne_ball_homography_projection"] is False
    assert {frame["status"] for frame in tracks["frames"]} == {"OBSERVED"}
    capture = cv2.VideoCapture(str(artifacts.debug_video_path))
    try:
        assert capture.isOpened()
        assert round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == artifacts.frames_processed
    finally:
        capture.release()


def test_raw_loader_rejects_already_tracked_input(
    synthetic_video: Path,
    synthetic_calibration: Path,
    tmp_path: Path,
) -> None:
    payload = _raw_detection_payload(synthetic_video)
    temporal = payload["temporal_processing"]
    assert isinstance(temporal, dict)
    temporal["tracking"] = True
    detections_path = tmp_path / "tracked.json"
    detections_path.write_text(json.dumps(payload), encoding="utf-8")
    metadata = inspect_video(synthetic_video)
    envelope = build_court_image_envelope(
        load_calibration(synthetic_calibration),
        frame_width_px=metadata.width,
        frame_height_px=metadata.height,
        settings=BallTrackingSettings(),
    )

    with pytest.raises(BallTrackingInputError, match="raw frame-local detections"):
        load_ball_detection_candidates(detections_path, envelope=envelope)
