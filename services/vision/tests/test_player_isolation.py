import json
from pathlib import Path

import pytest

from pickleball_vision.calibration import (
    CalibrationCorrespondence,
    CalibrationSource,
    CourtCalibration,
    fit_calibration,
)
from pickleball_vision.config import PlayerIsolationSettings
from pickleball_vision.court import CourtDimensions, CourtPoint, ImagePoint, court_landmarks
from pickleball_vision.errors import PlayerIsolationInputError
from pickleball_vision.person_detection import (
    BoundingBox,
    DetectorMetadata,
    PersonDetection,
    PersonDetectionRun,
)
from pickleball_vision.player_isolation import (
    LOGICAL_PLAYER_ROLES,
    CourtRegionState,
    CourtSide,
    GroundProjectionStatus,
    LogicalPlayerRole,
    assess_ground_contact,
    assignment_selections_for_candidates,
    build_logical_player_assignments,
    build_player_candidates,
    ground_contact_point,
    load_player_assignments,
    save_player_assignments,
)
from pickleball_vision.video import VideoMetadata

FRAME_WIDTH = 120
FRAME_HEIGHT = 100
FPS = 10.0


@pytest.fixture
def simple_calibration() -> CourtCalibration:
    court = CourtDimensions()
    landmarks = court_landmarks(court)
    selected = (landmarks[0], landmarks[1], landmarks[8], landmarks[9])
    correspondences = tuple(
        CalibrationCorrespondence(
            landmark=landmark.name,
            label=landmark.label,
            image_point=ImagePoint(
                x_px=10 + landmark.court_point.x_m * 10,
                y_px=10 + landmark.court_point.y_m * 5,
            ),
            court_point=landmark.court_point,
        )
        for landmark in selected
    )
    return fit_calibration(
        source=CalibrationSource(
            video_path=Path("/video/synthetic.mp4"),
            requested_timestamp_s=0,
            frame_index=0,
            frame_timestamp_s=0,
            frame_width_px=FRAME_WIDTH,
            frame_height_px=FRAME_HEIGHT,
            fps=FPS,
        ),
        court=court,
        correspondences=correspondences,
    )


def _detection_at_court(
    calibration: CourtCalibration,
    court_point: CourtPoint,
    *,
    frame_number: int = 0,
    confidence: float = 0.8,
) -> PersonDetection:
    image = calibration.court_to_image(court_point)
    return PersonDetection(
        bounding_box=BoundingBox(
            left_px=image.x_px - 2,
            top_px=max(0, image.y_px - 8),
            right_px=image.x_px + 2,
            bottom_px=image.y_px,
        ),
        confidence=confidence,
        frame_number=frame_number,
        timestamp_s=frame_number / FPS,
    )


def _run(detections: tuple[PersonDetection, ...], *, frame_count: int) -> PersonDetectionRun:
    return PersonDetectionRun(
        created_at_utc="2026-08-13T12:00:00+00:00",
        source=VideoMetadata(
            filename="synthetic.mp4",
            path=Path("/video/synthetic.mp4"),
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            fps=FPS,
            frame_count=frame_count,
            duration=frame_count / FPS,
            codec="mp4v",
        ),
        calibration_path="/output/calibration.json",
        calibration_schema_version=2,
        detector=DetectorMetadata("test", "test", "cpu", "test", "1"),
        configuration={"minimum_confidence": 0.1},
        detections=detections,
    )


def test_ground_contact_uses_bottom_center_not_person_box_center() -> None:
    detection = PersonDetection(BoundingBox(10, 20, 30, 80), 0.9, 3, 0.3)

    ground = ground_contact_point(detection)

    assert ground == ImagePoint(x_px=20, y_px=80)
    assert ground.y_px != pytest.approx(50)


@pytest.mark.parametrize(
    ("point", "expected_region", "expected_side"),
    [
        (CourtPoint(3, 2), CourtRegionState.INSIDE, CourtSide.NEAR),
        (CourtPoint(3, 11), CourtRegionState.INSIDE, CourtSide.FAR),
        (CourtPoint(-0.5, 3), CourtRegionState.NEAR, CourtSide.NEAR),
        (CourtPoint(8, 3), CourtRegionState.OUTSIDE, CourtSide.NEAR),
        (CourtPoint(3, CourtDimensions().net_y_m), CourtRegionState.INSIDE, CourtSide.AMBIGUOUS),
    ],
)
def test_court_region_and_side_filtering_use_projected_bottom_center(
    simple_calibration: CourtCalibration,
    point: CourtPoint,
    expected_region: CourtRegionState,
    expected_side: CourtSide,
) -> None:
    assessment = assess_ground_contact(
        _detection_at_court(simple_calibration, point),
        calibration=simple_calibration,
        frame_height_px=FRAME_HEIGHT,
        settings=PlayerIsolationSettings(),
    )

    assert assessment.projection_status is GroundProjectionStatus.PROJECTED
    assert assessment.court_point is not None
    assert assessment.court_point.x_m == pytest.approx(point.x_m, abs=1e-6)
    assert assessment.court_point.y_m == pytest.approx(point.y_m, abs=1e-6)
    assert assessment.region_state is expected_region
    assert assessment.side is expected_side


def test_boundary_and_frame_edge_uncertainty_are_preserved(
    simple_calibration: CourtCalibration,
) -> None:
    boundary = assess_ground_contact(
        _detection_at_court(simple_calibration, CourtPoint(0.1, 3)),
        calibration=simple_calibration,
        frame_height_px=FRAME_HEIGHT,
        settings=PlayerIsolationSettings(boundary_uncertainty_m=0.25),
    )
    clipped = PersonDetection(
        bounding_box=BoundingBox(20, 30, 40, FRAME_HEIGHT),
        confidence=0.9,
        frame_number=0,
        timestamp_s=0,
    )
    clipped_assessment = assess_ground_contact(
        clipped,
        calibration=simple_calibration,
        frame_height_px=FRAME_HEIGHT,
        settings=PlayerIsolationSettings(),
    )

    assert boundary.region_state is CourtRegionState.INSIDE
    assert boundary.region_boundary_ambiguous is True
    assert boundary.region_confidence < 1
    assert clipped_assessment.projection_status is GroundProjectionStatus.FRAME_EDGE_CLIPPED
    assert clipped_assessment.region_state is CourtRegionState.AMBIGUOUS
    assert clipped_assessment.court_point is None


def test_candidate_selection_survives_one_missed_frame_and_ignores_confidence_ranking(
    simple_calibration: CourtCalibration,
    tmp_path: Path,
) -> None:
    detections = (
        _detection_at_court(simple_calibration, CourtPoint(2, 2), frame_number=0, confidence=0.35),
        _detection_at_court(simple_calibration, CourtPoint(8, 2), frame_number=0, confidence=0.99),
        _detection_at_court(
            simple_calibration, CourtPoint(2.1, 2.1), frame_number=2, confidence=0.30
        ),
        _detection_at_court(
            simple_calibration, CourtPoint(8.1, 2.1), frame_number=2, confidence=0.98
        ),
    )
    candidates = build_player_candidates(
        _run(detections, frame_count=3),
        calibration=simple_calibration,
        detections_path=tmp_path / "detections.json",
        calibration_path=tmp_path / "calibration.json",
        settings=PlayerIsolationSettings(
            min_candidate_observations=2,
            min_court_support_ratio=0.6,
        ),
    )

    assert len(candidates.candidates) == 2
    inside_candidate = next(candidate for candidate in candidates.candidates if candidate.eligible)
    outside_candidate = next(
        candidate for candidate in candidates.candidates if not candidate.eligible
    )
    assert [
        observation.detection.frame_number for observation in inside_candidate.observations
    ] == [0, 2]
    assert inside_candidate.observed_frame_ratio == pytest.approx(2 / 3)
    assert inside_candidate.court_support_ratio == pytest.approx(1)
    assert outside_candidate.court_support_ratio == pytest.approx(0)
    assert max(
        observation.detection.confidence for observation in outside_candidate.observations
    ) > max(observation.detection.confidence for observation in inside_candidate.observations)


def test_logical_roles_are_distinct_and_persist_separately_from_candidate_ids(
    simple_calibration: CourtCalibration,
    tmp_path: Path,
) -> None:
    court_points = (CourtPoint(1, 2), CourtPoint(5, 2), CourtPoint(1, 11), CourtPoint(5, 11))
    detections = tuple(
        _detection_at_court(simple_calibration, point, frame_number=frame_number)
        for frame_number in range(2)
        for point in court_points
    )
    candidates = build_player_candidates(
        _run(detections, frame_count=2),
        calibration=simple_calibration,
        detections_path=tmp_path / "detections.json",
        calibration_path=tmp_path / "calibration.json",
        settings=PlayerIsolationSettings(min_candidate_observations=2),
    )
    selections = {
        role: candidate.observations[0]
        for role, candidate in zip(LOGICAL_PLAYER_ROLES, candidates.candidates, strict=True)
    }
    assignments = build_logical_player_assignments(
        selections,
        candidates_path=tmp_path / "player-candidates.json",
        detections_path=tmp_path / "detections.json",
    )
    output_path = save_player_assignments(assignments, tmp_path / "player-assignments.json")
    loaded = load_player_assignments(output_path)
    serialized = json.loads(output_path.read_text(encoding="utf-8"))

    assert set(loaded.by_role()) == set(LOGICAL_PLAYER_ROLES)
    assert len({assignment.candidate_id for assignment in loaded.assignments}) == 4
    assert {item["logical_player"] for item in serialized["assignments"]} == {
        "ME",
        "PARTNER",
        "OPPONENT_1",
        "OPPONENT_2",
    }
    assert serialized["identity_contract"].startswith("logical_roles_are_human_owned")

    corrected_selections = assignment_selections_for_candidates(loaded, candidates)
    corrected = build_logical_player_assignments(
        corrected_selections,
        candidates_path=tmp_path / "player-candidates.json",
        detections_path=tmp_path / "detections.json",
        corrected_from_path=output_path,
    )
    assert corrected.corrected_from_path == str(output_path.resolve())

    duplicate = dict(selections)
    duplicate[LogicalPlayerRole.PARTNER] = duplicate[LogicalPlayerRole.ME]
    with pytest.raises(PlayerIsolationInputError, match="distinct candidate"):
        build_logical_player_assignments(
            duplicate,
            candidates_path=tmp_path / "player-candidates.json",
            detections_path=tmp_path / "detections.json",
        )
