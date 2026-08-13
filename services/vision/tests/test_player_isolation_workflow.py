import json
from pathlib import Path

import cv2
import pytest

from pickleball_vision.calibration import load_calibration
from pickleball_vision.config import PlayerIsolationSettings
from pickleball_vision.court import CourtPoint
from pickleball_vision.person_detection import (
    BoundingBox,
    DetectorMetadata,
    PersonDetection,
    PersonDetectionRun,
)
from pickleball_vision.player_isolation import (
    LOGICAL_PLAYER_ROLES,
    CandidateObservation,
    LogicalPlayerRole,
    PlayerCandidateCollection,
)
from pickleball_vision.player_isolation_workflow import isolate_primary_players
from pickleball_vision.video import inspect_video


def test_isolation_workflow_preserves_raw_detections_and_writes_debug_artifacts(
    synthetic_video: Path,
    synthetic_calibration: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = inspect_video(synthetic_video)
    calibration = load_calibration(synthetic_calibration)
    court_points = (CourtPoint(1, 2), CourtPoint(5, 2), CourtPoint(1, 11), CourtPoint(5, 11))
    detections: list[PersonDetection] = []
    for frame_number in range(metadata.frame_count):
        for point in court_points:
            image = calibration.court_to_image(point)
            detections.append(
                PersonDetection(
                    bounding_box=BoundingBox(
                        left_px=image.x_px - 1,
                        top_px=max(0, image.y_px - 3),
                        right_px=image.x_px + 1,
                        bottom_px=image.y_px,
                    ),
                    confidence=0.7,
                    frame_number=frame_number,
                    timestamp_s=frame_number / metadata.fps,
                )
            )
    run = PersonDetectionRun(
        created_at_utc="2026-08-13T12:00:00+00:00",
        source=metadata,
        calibration_path=str(synthetic_calibration),
        calibration_schema_version=calibration.schema_version,
        detector=DetectorMetadata("test", "test", "cpu", "test", "1"),
        configuration={},
        detections=tuple(detections),
    )
    detections_path = tmp_path / "detections.json"
    detections_path.write_text(json.dumps(run.as_dict()), encoding="utf-8")
    original_raw_detections = detections_path.read_bytes()

    def fake_select(
        _video_path: Path,
        *,
        candidates: PlayerCandidateCollection,
        **_kwargs: object,
    ) -> dict[LogicalPlayerRole, CandidateObservation]:
        assert len(candidates.candidates) == 4
        return {
            role: candidate.observations[0]
            for role, candidate in zip(LOGICAL_PLAYER_ROLES, candidates.candidates, strict=True)
        }

    monkeypatch.setattr(
        "pickleball_vision.player_isolation_workflow.select_logical_players",
        fake_select,
    )
    output_dir = tmp_path / "isolation"

    artifacts = isolate_primary_players(
        synthetic_video,
        detections_path=detections_path,
        calibration_path=synthetic_calibration,
        selection_timestamp_s=0.5,
        output_dir=output_dir,
        settings=PlayerIsolationSettings(min_candidate_observations=2),
    )

    assert detections_path.read_bytes() == original_raw_detections
    assert artifacts.candidate_count == 4
    assert artifacts.eligible_candidate_count == 4
    assert artifacts.candidates_path.is_file()
    assert artifacts.assignments_path.is_file()
    assert artifacts.debug_video_path.is_file()
    assert artifacts.summary_path.is_file()

    assignments = json.loads(artifacts.assignments_path.read_text(encoding="utf-8"))
    assert len(assignments["assignments"]) == 4
    assert assignments["record_type"] == "logical_player_assignments"
    candidates = json.loads(artifacts.candidates_path.read_text(encoding="utf-8"))
    assert candidates["eligible_candidate_count"] == 4
    assert candidates["candidates"][0]["identity_scope"].startswith("ephemeral")

    capture = cv2.VideoCapture(str(artifacts.debug_video_path))
    try:
        assert capture.isOpened()
        assert round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == metadata.frame_count
        assert round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == metadata.width
        assert round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == metadata.height
    finally:
        capture.release()
