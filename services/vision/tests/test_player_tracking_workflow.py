import json
from pathlib import Path

import cv2
import pytest

from pickleball_vision.calibration import load_calibration
from pickleball_vision.config import (
    PersonDetectionSettings,
    PlayerIsolationSettings,
    PlayerTrackingSettings,
)
from pickleball_vision.court import CourtPoint
from pickleball_vision.person_detection import (
    BoundingBox,
    DetectorMetadata,
    PersonDetection,
    PersonDetectionRun,
)
from pickleball_vision.player_isolation import (
    LOGICAL_PLAYER_ROLES,
    LogicalPlayerAssignments,
    ManualPlayerAssignment,
    assess_ground_contact,
    save_player_assignments,
)
from pickleball_vision.player_tracking import (
    IndexedDetection,
    RawTrackerObservation,
    TrackerMetadata,
)
from pickleball_vision.player_tracking_workflow import (
    PlayerProfileMismatchError,
    _rebind_manual_anchors,
    track_players_in_video,
    validate_portable_player_profile,
)
from pickleball_vision.video import inspect_video


class _DeterministicTracker:
    def __init__(self) -> None:
        self._metadata = TrackerMetadata("test_tracker", "test", "test", "1", {})

    @property
    def metadata(self) -> TrackerMetadata:
        return self._metadata

    def update(
        self,
        *,
        frame_number: int,
        timestamp_s: float,
        detections: tuple[IndexedDetection, ...],
        frame_width_px: int,
        frame_height_px: int,
    ) -> tuple[RawTrackerObservation, ...]:
        del frame_width_px, frame_height_px
        return tuple(
            RawTrackerObservation(
                f"raw-{frame_number}-{item.raw_detection_index}",
                item.raw_detection_index % 4 + 1,
                item.raw_detection_index,
                frame_number,
                timestamp_s,
                item.detection.bounding_box,
                item.detection.confidence,
                item.detection.confidence,
            )
            for item in detections
        )


class _AnchorDetector:
    def __init__(self, detections: tuple[PersonDetection, ...]) -> None:
        self._detections = detections
        self._metadata = DetectorMetadata("test", "test", "cpu", "test", "1")

    @property
    def metadata(self) -> DetectorMetadata:
        return self._metadata

    def detect(
        self,
        frame: object,
        *,
        frame_number: int,
        timestamp_s: float,
    ) -> tuple[PersonDetection, ...]:
        del frame
        return tuple(
            PersonDetection(
                item.bounding_box,
                item.confidence,
                frame_number,
                timestamp_s,
            )
            for item in self._detections
        )


def test_tracking_workflow_writes_separate_raw_and_logical_artifacts(
    synthetic_video: Path,
    synthetic_calibration: Path,
    tmp_path: Path,
) -> None:
    source = inspect_video(synthetic_video)
    calibration = load_calibration(synthetic_calibration)
    points = (CourtPoint(1, 2), CourtPoint(5, 2), CourtPoint(1, 11), CourtPoint(5, 11))
    detections: list[PersonDetection] = []
    for frame_number in range(source.frame_count):
        for point in points:
            image = calibration.court_to_image(point)
            detections.append(
                PersonDetection(
                    BoundingBox(
                        image.x_px - 3,
                        max(0, image.y_px - 10),
                        image.x_px + 3,
                        image.y_px,
                    ),
                    0.9,
                    frame_number,
                    frame_number / source.fps,
                )
            )
    run = PersonDetectionRun(
        "2026-08-14T00:00:00+00:00",
        source,
        str(synthetic_calibration),
        calibration.schema_version,
        DetectorMetadata("test", "test", "cpu", "test", "1"),
        {},
        tuple(detections),
    )
    detections_path = tmp_path / "detections.json"
    detections_path.write_text(json.dumps(run.as_dict()), encoding="utf-8")
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "primary_player_candidates",
                "inputs": {"raw_person_detections": str(detections_path)},
                "candidates": [
                    {
                        "candidate_id": f"candidate-{role_index}",
                        "observations": [
                            {"raw_detection": {"index": frame * 4 + role_index}}
                            for frame in range(source.frame_count)
                        ],
                    }
                    for role_index in range(4)
                ],
            }
        ),
        encoding="utf-8",
    )
    anchor_frame = 5
    isolation_settings = PlayerIsolationSettings(min_candidate_observations=2)
    assignments = LogicalPlayerAssignments(
        "2026-08-14T00:00:00+00:00",
        str(candidates_path),
        str(detections_path),
        tuple(
            ManualPlayerAssignment(
                role,
                f"candidate-{index}",
                anchor_frame * 4 + index,
                anchor_frame,
                anchor_frame / source.fps,
                assess_ground_contact(
                    detections[anchor_frame * 4 + index],
                    calibration=calibration,
                    frame_height_px=source.height,
                    settings=isolation_settings,
                ).side,
                assess_ground_contact(
                    detections[anchor_frame * 4 + index],
                    calibration=calibration,
                    frame_height_px=source.height,
                    settings=isolation_settings,
                ).image_point,
            )
            for index, role in enumerate(LOGICAL_PLAYER_ROLES)
        ),
    )
    assignments_path = save_player_assignments(assignments, tmp_path / "assignments.json")
    (tmp_path / "player-names.json").write_text(
        json.dumps(
            {
                "ME": "John",
                "PARTNER": "Denny",
                "OPPONENT_1": "Oksana",
                "OPPONENT_2": "Diana",
            }
        ),
        encoding="utf-8",
    )
    original_detections = detections_path.read_bytes()

    artifacts = track_players_in_video(
        synthetic_video,
        calibration_path=synthetic_calibration,
        output_dir=tmp_path / "tracking",
        tracking_settings=PlayerTrackingSettings(),
        isolation_settings=isolation_settings,
        detections_path=detections_path,
        assignments_path=assignments_path,
        tracker=_DeterministicTracker(),
    )

    assert detections_path.read_bytes() == original_detections
    assert artifacts.tracks_path.is_file()
    assert artifacts.annotated_video_path.is_file()
    assert artifacts.summary_path.is_file()
    tracks = json.loads(artifacts.tracks_path.read_text(encoding="utf-8"))
    assert tracks["raw_tracker_layer"]["record_type"] == "raw_transient_tracker_observations"
    assert set(tracks["logical_identity_layer"]) == {role.value for role in LOGICAL_PLAYER_ROLES}
    assert tracks["player_names"]["ME"] == "John"
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["frames_processed"] == source.frame_count
    assert summary["players"]["OPPONENT_1"]["display_name"] == "Oksana"
    capture = cv2.VideoCapture(str(artifacts.annotated_video_path))
    try:
        assert capture.isOpened()
        assert round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == source.frame_count
    finally:
        capture.release()


def test_portable_manual_anchors_rebind_by_image_position(
    synthetic_video: Path,
    synthetic_calibration: Path,
) -> None:
    source = inspect_video(synthetic_video)
    calibration = load_calibration(synthetic_calibration)
    settings = PlayerIsolationSettings(min_candidate_observations=2)
    court_points = (CourtPoint(1, 2), CourtPoint(5, 2), CourtPoint(1, 11), CourtPoint(5, 11))
    original: list[PersonDetection] = []
    for point in court_points:
        image = calibration.court_to_image(point)
        original.append(
            PersonDetection(
                BoundingBox(
                    image.x_px - 3,
                    max(0, image.y_px - 10),
                    image.x_px + 3,
                    image.y_px,
                ),
                0.9,
                5,
                5 / source.fps,
            )
        )
    assignments = LogicalPlayerAssignments(
        "2026-08-14T00:00:00+00:00",
        "/original/player-candidates.json",
        "/original/detections.json",
        tuple(
            ManualPlayerAssignment(
                role,
                f"candidate-{index}",
                9000 + index,
                5,
                5 / source.fps,
                assess_ground_contact(
                    original[index],
                    calibration=calibration,
                    frame_height_px=source.height,
                    settings=settings,
                ).side,
                assess_ground_contact(
                    original[index],
                    calibration=calibration,
                    frame_height_px=source.height,
                    settings=settings,
                ).image_point,
            )
            for index, role in enumerate(LOGICAL_PLAYER_ROLES)
        ),
    )
    fresh = tuple(reversed(original))

    rebound = _rebind_manual_anchors(
        assignments,
        detections=fresh,
        calibration_path=synthetic_calibration,
        source=source,
        isolation_settings=settings,
    )

    assert [item.anchor_detection_index for item in rebound.assignments] == [3, 2, 1, 0]
    assert all(item.anchor_image_point is not None for item in rebound.assignments)


def test_portable_profile_preflight_checks_only_anchor_frame(
    synthetic_video: Path,
    synthetic_calibration: Path,
    tmp_path: Path,
) -> None:
    source = inspect_video(synthetic_video)
    calibration = load_calibration(synthetic_calibration)
    settings = PlayerIsolationSettings(min_candidate_observations=2)
    anchor_frame = 5
    court_points = (CourtPoint(1, 2), CourtPoint(5, 2), CourtPoint(1, 11), CourtPoint(5, 11))
    detections = tuple(
        PersonDetection(
            BoundingBox(
                (image := calibration.court_to_image(point)).x_px - 3,
                max(0, image.y_px - 10),
                image.x_px + 3,
                image.y_px,
            ),
            0.9,
            anchor_frame,
            anchor_frame / source.fps,
        )
        for point in court_points
    )
    assignments = LogicalPlayerAssignments(
        "2026-08-14T00:00:00+00:00",
        "/reviewed/player-candidates.json",
        "/reviewed/detections.json",
        tuple(
            ManualPlayerAssignment(
                role,
                f"candidate-{index}",
                index,
                anchor_frame,
                anchor_frame / source.fps,
                assess_ground_contact(
                    detections[index],
                    calibration=calibration,
                    frame_height_px=source.height,
                    settings=settings,
                ).side,
                assess_ground_contact(
                    detections[index],
                    calibration=calibration,
                    frame_height_px=source.height,
                    settings=settings,
                ).image_point,
            )
            for index, role in enumerate(LOGICAL_PLAYER_ROLES)
        ),
    )
    assignments_path = save_player_assignments(assignments, tmp_path / "assignments.json")

    result = validate_portable_player_profile(
        synthetic_video,
        calibration_path=synthetic_calibration,
        assignments_path=assignments_path,
        person_settings=PersonDetectionSettings(device="cpu"),
        isolation_settings=settings,
        detector=_AnchorDetector(tuple(reversed(detections))),
    )

    assert result["compatible"] is True
    assert result["anchor_frame_count"] == 1
    assert result["anchor_detection_count"] == 4


def test_portable_profile_preflight_rejects_different_player_layout(
    synthetic_video: Path,
    synthetic_calibration: Path,
    tmp_path: Path,
) -> None:
    source = inspect_video(synthetic_video)
    calibration = load_calibration(synthetic_calibration)
    settings = PlayerIsolationSettings(min_candidate_observations=2)
    anchor_frame = 5
    detections: list[PersonDetection] = []
    for point in (CourtPoint(1, 2), CourtPoint(5, 2), CourtPoint(1, 11), CourtPoint(5, 11)):
        image = calibration.court_to_image(point)
        detections.append(
            PersonDetection(
                BoundingBox(
                    image.x_px - 3,
                    max(0, image.y_px - 10),
                    image.x_px + 3,
                    image.y_px,
                ),
                0.9,
                anchor_frame,
                anchor_frame / source.fps,
            )
        )
    assignments = LogicalPlayerAssignments(
        "2026-08-14T00:00:00+00:00",
        "/reviewed/player-candidates.json",
        "/reviewed/detections.json",
        tuple(
            ManualPlayerAssignment(
                role,
                f"candidate-{index}",
                index,
                anchor_frame,
                anchor_frame / source.fps,
                assess_ground_contact(
                    detections[index],
                    calibration=calibration,
                    frame_height_px=source.height,
                    settings=settings,
                ).side,
                assess_ground_contact(
                    detections[index],
                    calibration=calibration,
                    frame_height_px=source.height,
                    settings=settings,
                ).image_point,
            )
            for index, role in enumerate(LOGICAL_PLAYER_ROLES)
        ),
    )
    assignments_path = save_player_assignments(assignments, tmp_path / "assignments.json")
    shifted = tuple(
        PersonDetection(
            BoundingBox(
                min(source.width - 7, item.bounding_box.left_px + 20),
                item.bounding_box.top_px,
                min(source.width - 1, item.bounding_box.right_px + 20),
                item.bounding_box.bottom_px,
            ),
            item.confidence,
            item.frame_number,
            item.timestamp_s,
        )
        for item in detections
    )

    with pytest.raises(PlayerProfileMismatchError):
        validate_portable_player_profile(
            synthetic_video,
            calibration_path=synthetic_calibration,
            assignments_path=assignments_path,
            person_settings=PersonDetectionSettings(device="cpu"),
            isolation_settings=settings,
            detector=_AnchorDetector(shifted),
        )
