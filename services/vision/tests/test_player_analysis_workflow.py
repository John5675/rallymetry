import json
from pathlib import Path

import cv2
import pytest

from pickleball_vision.calibration import load_calibration
from pickleball_vision.config import PlayerAnalysisSettings
from pickleball_vision.court import CourtPoint
from pickleball_vision.errors import PlayerAnalysisInputError
from pickleball_vision.player_analysis_workflow import (
    POSITION_CORRECTIONS_NAME,
    analyze_players_in_video,
    load_position_corrections,
    load_tracking_positions,
)
from pickleball_vision.player_isolation import LOGICAL_PLAYER_ROLES
from pickleball_vision.video import inspect_video


def _write_tracks(
    path: Path,
    *,
    video_path: Path,
    calibration_path: Path,
) -> Path:
    source = inspect_video(video_path)
    calibration = load_calibration(calibration_path)
    base_points = ((1.0, 2.0), (5.0, 2.0), (1.0, 11.0), (5.0, 11.0))
    names = ("John", "Denny", "Oksana", "Diana")
    logical_layer: dict[str, object] = {}
    for role_index, role in enumerate(LOGICAL_PLAYER_ROLES):
        frames: list[dict[str, object]] = []
        for frame_number in range(source.frame_count):
            x_m, y_m = base_points[role_index]
            court_point = CourtPoint(x_m + frame_number * 0.01, y_m)
            image_point = calibration.court_to_image(court_point)
            frames.append(
                {
                    "frame_number": frame_number,
                    "timestamp_s": frame_number / source.fps,
                    "tracking_state": "observed",
                    "tracking_confidence": 0.9,
                    "tracker_id": role_index + 1,
                    "raw_tracker_observation_id": f"raw-{role.value}-{frame_number}",
                    "ground_contact": {
                        "image_point": image_point.as_dict(),
                        "court_point": court_point.as_dict(),
                        "method": "bounding_box_bottom_center",
                        "projection_status": "projected_bottom_center_estimate",
                        "court_region": "inside",
                        "court_region_confidence": 1.0,
                        "court_region_boundary_ambiguous": False,
                        "court_side": "near_side" if role_index < 2 else "far_side",
                        "court_side_confidence": 1.0,
                    },
                    "identity_resolution": {"method": "test"},
                }
            )
        logical_layer[role.value] = frames
    payload = {
        "schema_version": 1,
        "record_type": "persistent_logical_player_tracks",
        "source": source.as_dict(),
        "inputs": {"court_calibration": str(calibration_path.resolve())},
        "player_names": {
            role.value: names[index] for index, role in enumerate(LOGICAL_PLAYER_ROLES)
        },
        "logical_identity_layer": logical_layer,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_analysis_workflow_writes_release_artifacts_without_mutating_tracks(
    synthetic_video: Path,
    synthetic_calibration: Path,
    tmp_path: Path,
) -> None:
    tracks_path = _write_tracks(
        tmp_path / "tracks.json",
        video_path=synthetic_video,
        calibration_path=synthetic_calibration,
    )
    original_tracks = tracks_path.read_bytes()
    corrections_path = tmp_path / POSITION_CORRECTIONS_NAME
    corrections_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "coordinate_space": "canonical_court_meters",
                "corrections": {
                    "OPPONENT_1": {
                        "x_offset_m": 0.0,
                        "y_offset_m": 0.15,
                        "reason": "synthetic alignment correction",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    artifacts = analyze_players_in_video(
        synthetic_video,
        calibration_path=synthetic_calibration,
        output_dir=tmp_path / "analysis",
        settings=PlayerAnalysisSettings(maximum_step_gap_seconds=0.2),
        tracks_path=tracks_path,
    )

    assert tracks_path.read_bytes() == original_tracks
    assert artifacts.positions_path.is_file()
    assert artifacts.summary_path.is_file()
    assert artifacts.annotated_video_path.is_file()
    assert artifacts.topdown_video_path.is_file()
    assert all(path.is_file() for path in artifacts.heatmap_paths.values())
    positions = json.loads(artifacts.positions_path.read_text(encoding="utf-8"))
    assert positions["release_version"] == "0.1"
    assert positions["position_contract"]["raw_coordinates_are_immutable"] is True
    me_frame = positions["players"]["ME"][0]
    assert me_frame["ground_point_method"] == "bounding_box_bottom_center"
    assert me_frame["raw_image_ground_point"] is not None
    assert me_frame["raw_court_coordinate"] == {"x_m": 1.0, "y_m": 2.0}
    assert me_frame["smoothed_court_coordinate"] is not None
    opponent_frame = positions["players"]["OPPONENT_1"][0]
    assert opponent_frame["raw_court_coordinate"] == {"x_m": 1.0, "y_m": 11.0}
    assert opponent_frame["corrected_court_coordinate"] == {"x_m": 1.0, "y_m": 11.15}
    assert positions["inputs"]["manual_position_corrections"] == str(corrections_path.resolve())
    assert positions["configuration"]["manual_court_position_corrections"]["ME"]["applied"] is False
    assert (
        positions["configuration"]["manual_court_position_corrections"]["OPPONENT_1"]["applied"]
        is True
    )
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["release_version"] == "0.1"
    assert summary["inputs"]["manual_position_corrections"] == str(corrections_path.resolve())
    assert summary["players"]["OPPONENT_1"]["manual_court_position_correction"][
        "y_offset_m"
    ] == pytest.approx(0.15)
    assert summary["players"]["ME"]["display_name"] == "John"
    assert summary["teams"]["ME_PARTNER"]["average_partner_spacing"]["value_m"]

    for video_path in (artifacts.annotated_video_path, artifacts.topdown_video_path):
        capture = cv2.VideoCapture(str(video_path))
        try:
            assert capture.isOpened()
            assert round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
        finally:
            capture.release()


def test_tracking_loader_rejects_non_ground_position_method(
    synthetic_video: Path,
    synthetic_calibration: Path,
    tmp_path: Path,
) -> None:
    tracks_path = _write_tracks(
        tmp_path / "tracks.json",
        video_path=synthetic_video,
        calibration_path=synthetic_calibration,
    )
    payload = json.loads(tracks_path.read_text(encoding="utf-8"))
    payload["logical_identity_layer"]["ME"][0]["ground_contact"]["method"] = "bounding_box_center"
    tracks_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PlayerAnalysisInputError, match="bottom_center"):
        load_tracking_positions(tracks_path)


def test_position_correction_loader_rejects_large_offset(tmp_path: Path) -> None:
    path = tmp_path / "corrections.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "coordinate_space": "canonical_court_meters",
                "corrections": {
                    "OPPONENT_1": {
                        "y_offset_m": 0.75,
                        "reason": "too large",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PlayerAnalysisInputError, match=r"must not exceed 0\.50 m"):
        load_position_corrections(path)
