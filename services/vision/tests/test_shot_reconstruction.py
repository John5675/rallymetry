import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from pickleball_vision.config import ShotClassificationSettings
from pickleball_vision.contact_detection import (
    ContactCandidate,
    ContactCandidatePlayer,
    ContactEvidenceMode,
)
from pickleball_vision.court import CourtDimensions, CourtPoint, ImagePoint
from pickleball_vision.errors import ShotReconstructionInputError
from pickleball_vision.media import inspect_media
from pickleball_vision.rally_segmentation import BallEvidenceStatus, RallyBallFrame
from pickleball_vision.shot_evaluation import GroundTruthShot, evaluate_shots
from pickleball_vision.shot_reconstruction import (
    Shot,
    ShotBounce,
    ShotHitterDecision,
    ShotPlayerPosition,
    ShotRally,
    ShotTrajectorySegment,
    ShotType,
    classify_shot,
    reconstruct_shots,
)
from pickleball_vision.shot_reconstruction_render import render_shot_frame
from pickleball_vision.shot_reconstruction_workflow import load_shot_hitters
from pickleball_vision.video import inspect_video

WIDTH = 100
HEIGHT = 100
FPS = 10.0
SETTINGS = ShotClassificationSettings()


def _contact(
    contact_id: str,
    frame: int,
    role: str,
    *,
    ball_y: float = 42.0,
    accepted: bool = True,
) -> ContactCandidate:
    player = ContactCandidatePlayer(
        role=role,
        display_name=role.title(),
        rank=1,
        bounding_box=(40.0, 20.0, 60.0, 70.0),
        ground_image_position=ImagePoint(50.0, 70.0),
        tracking_confidence=0.95,
        tracking_state="observed",
        court_side="near_side" if role in {"ME", "PARTNER"} else "far_side",
        court_region="inside",
        distance_px=0.0,
        distance_diagonal_fraction=0.0,
        proximity_confidence=1.0,
        ball_inside_person_box=True,
    )
    return ContactCandidate(
        contact_id=contact_id,
        frame=frame,
        timestamp_seconds=frame / FPS,
        media_timestamp_seconds=frame / FPS,
        ball_image_position=ImagePoint(50.0, ball_y),
        trajectory_status=BallEvidenceStatus.OBSERVED,
        candidate_players=(player,),
        visual_confidence=0.95,
        audio_confidence=0.0,
        fused_confidence=0.95,
        matched_audio_event_id=None,
        evidence_mode=ContactEvidenceMode.VISUAL_ONLY,
        accepted_vision_only=accepted,
        accepted_fused=accepted,
        supporting_signals={},
    )


def _position(
    role: str,
    *,
    y_m: float,
) -> ShotPlayerPosition:
    return ShotPlayerPosition(
        player_id=role,
        image_ground_point=ImagePoint(50.0, 70.0),
        court_point=CourtPoint(3.0, y_m),
        confidence=0.95,
        tracking_state="observed",
        court_side="near_side" if role in {"ME", "PARTNER"} else "far_side",
        court_region="inside",
    )


def _trajectory(speed: float = 0.30) -> ShotTrajectorySegment:
    return ShotTrajectorySegment(
        start_frame=5,
        end_frame=15,
        segment_ids=("segment-1",),
        observed_count=11,
        interpolated_count=0,
        unknown_count=0,
        known_fraction=1.0,
        initial_image_position=ImagePoint(50.0, 42.0),
        final_image_position=ImagePoint(60.0, 50.0),
        initial_speed_diagonals_per_second=speed,
        peak_speed_diagonals_per_second=speed,
    )


def _previous(shot_type: ShotType, *, bounced: bool) -> Shot:
    return Shot(
        shot_id="previous",
        rally_id="rally-1",
        shot_index=1,
        hitter_id="OPPONENT_1",
        hitter_confidence=0.9,
        contact_id="previous-contact",
        contact_frame=1,
        contact_timestamp_seconds=0.1,
        trajectory_segment=_trajectory(),
        bounce_id="bounce-previous" if bounced else None,
        landing_court_position=CourtPoint(3.0, 5.0) if bounced else None,
        hitter_court_position=_position("OPPONENT_1", y_m=11.0),
        shot_type=shot_type,
        confidence=0.9,
        classification_evidence={},
    )


def _classify(
    *,
    shot_index: int,
    role: str = "ME",
    position_y_m: float = 4.3,
    speed: float = 0.30,
    bounce: ShotBounce | None = None,
    previous: Shot | None = None,
    ball_y: float = 42.0,
    hitter_confidence: float = 0.9,
) -> tuple[ShotType, float, dict[str, object]]:
    contact = _contact("contact", 5, role, ball_y=ball_y)
    position = _position(role, y_m=position_y_m)
    return classify_shot(
        shot_index=shot_index,
        contact=contact,
        hitter=ShotHitterDecision("contact", role, hitter_confidence),
        hitter_position=position,
        trajectory=_trajectory(speed),
        bounce=bounce,
        previous_shot=previous,
        positions_at_contact={role: position},
        settings=SETTINGS,
        court=CourtDimensions(),
    )


def _bounce(y_m: float = 7.2) -> ShotBounce:
    return ShotBounce("bounce-1", 10, 1.0, CourtPoint(3.0, y_m), 0.9)


@pytest.mark.parametrize(
    ("expected", "kwargs"),
    [
        (ShotType.SERVE, {"shot_index": 1, "position_y_m": 2.0, "bounce": _bounce()}),
        (
            ShotType.RETURN,
            {
                "shot_index": 2,
                "position_y_m": 2.0,
                "previous": _previous(ShotType.SERVE, bounced=True),
            },
        ),
        (
            ShotType.DINK,
            {"shot_index": 3, "position_y_m": 4.3, "speed": 0.12, "bounce": _bounce()},
        ),
        (
            ShotType.DROP,
            {"shot_index": 3, "position_y_m": 2.0, "speed": 0.20, "bounce": _bounce()},
        ),
        (
            ShotType.DRIVE,
            {
                "shot_index": 3,
                "position_y_m": 2.0,
                "speed": 0.70,
                "previous": _previous(ShotType.OTHER, bounced=True),
            },
        ),
        (
            ShotType.VOLLEY,
            {"shot_index": 3, "position_y_m": 4.3, "speed": 0.30},
        ),
        (
            ShotType.OVERHEAD,
            {"shot_index": 3, "position_y_m": 4.3, "speed": 0.60, "ball_y": 28.0},
        ),
        (
            ShotType.OTHER,
            {
                "shot_index": 3,
                "position_y_m": 2.0,
                "speed": 0.30,
                "previous": _previous(ShotType.OTHER, bounced=True),
            },
        ),
    ],
)
def test_documented_rule_families(expected: ShotType, kwargs: dict[str, object]) -> None:
    shot_type, confidence, evidence = _classify(**kwargs)  # type: ignore[arg-type]

    assert shot_type is expected
    assert 0 < confidence <= 1
    assert evidence["selectedRule"] != "UNKNOWN_INSUFFICIENT_EVIDENCE"
    assert evidence["newNeuralNetworkUsed"] is False


def test_unknown_is_retained_when_hitter_or_trajectory_evidence_is_missing() -> None:
    contact = _contact("contact", 5, "ME")
    position = _position("ME", y_m=4.3)
    sparse = ShotTrajectorySegment(
        5,
        15,
        (),
        1,
        0,
        10,
        1 / 11,
        ImagePoint(50.0, 42.0),
        ImagePoint(50.0, 42.0),
        None,
        None,
    )

    shot_type, confidence, evidence = classify_shot(
        shot_index=3,
        contact=contact,
        hitter=ShotHitterDecision("contact", "UNKNOWN", 0.95),
        hitter_position=position,
        trajectory=sparse,
        bounce=None,
        previous_shot=None,
        positions_at_contact={"ME": position},
        settings=SETTINGS,
        court=CourtDimensions(),
    )

    assert shot_type is ShotType.UNKNOWN
    assert confidence == 1.0
    assert "knownHitter" in cast(list[str], evidence["failedEvidenceGates"])
    assert evidence["airborneBallProjectedThroughHomography"] is False


def _ball_frames(count: int = 31) -> tuple[RallyBallFrame, ...]:
    return tuple(
        RallyBallFrame(
            frame,
            frame / FPS,
            BallEvidenceStatus.OBSERVED,
            "segment-1",
            ImagePoint(10.0 + frame, 20.0 + frame * 0.5),
            0.9,
        )
        for frame in range(count)
    )


def test_reconstruction_links_rally_contacts_trajectory_bounce_and_ground_position() -> None:
    frames = _ball_frames()
    contacts = (
        _contact("contact-1", 5, "ME"),
        _contact("contact-2", 20, "OPPONENT_1"),
        _contact("outside", 30, "ME"),
    )
    positions = tuple(
        {
            "ME": _position("ME", y_m=2.0),
            "OPPONENT_1": _position("OPPONENT_1", y_m=11.0),
        }
        for _ in frames
    )

    result = reconstruct_shots(
        frames=frames,
        rallies=(ShotRally("rally-1", 0, 25, 0.9),),
        contacts=contacts,
        bounces=(_bounce(),),
        hitters_by_contact={
            "contact-1": ShotHitterDecision("contact-1", "ME", 0.9),
            "contact-2": ShotHitterDecision("contact-2", "OPPONENT_1", 0.9),
            "outside": ShotHitterDecision("outside", "ME", 0.9),
        },
        player_positions_by_frame=positions,
        frame_width_px=WIDTH,
        frame_height_px=HEIGHT,
        settings=SETTINGS,
    )

    assert len(result.shots) == 2
    first, second = result.shots
    assert (first.rally_id, first.shot_index, first.shot_type) == (
        "rally-1",
        1,
        ShotType.SERVE,
    )
    assert first.trajectory_segment.start_frame == 5
    assert first.trajectory_segment.end_frame == 20
    assert first.bounce_id == "bounce-1"
    assert first.landing_court_position == CourtPoint(3.0, 7.2)
    assert first.hitter_court_position is not None
    assert first.hitter_court_position.image_ground_point == ImagePoint(50.0, 70.0)
    assert second.shot_type is ShotType.RETURN
    assert result.contact_outside_rally_count == 1


def test_evaluation_reports_accuracy_per_class_confusion_and_unknown_rate() -> None:
    base = _previous(ShotType.SERVE, bounced=True)
    serve = replace(
        base,
        shot_id="serve",
        contact_id="serve-contact",
        contact_timestamp_seconds=1.0,
    )
    unknown = replace(
        base,
        shot_id="unknown",
        contact_id="unknown-contact",
        contact_timestamp_seconds=2.0,
        shot_type=ShotType.UNKNOWN,
    )

    evaluation = evaluate_shots(
        (serve, unknown),
        (
            GroundTruthShot("human-serve", 10, 1.0, ShotType.SERVE),
            GroundTruthShot("human-drive", 20, 2.0, ShotType.DRIVE),
            GroundTruthShot("human-return", 100, 10.0, ShotType.RETURN),
        ),
        settings=SETTINGS,
    )

    assert evaluation["accuracy"] == 0.5
    assert evaluation["unknownRate"] == 0.5
    assert evaluation["matchCoverage"] == pytest.approx(2 / 3)
    assert evaluation["missedGroundTruthShotCount"] == 1
    per_class = cast(dict[str, dict[str, object]], evaluation["perClass"])
    assert per_class["SERVE"]["precision"] == 1.0
    assert per_class["DRIVE"]["recall"] == 0.0
    assert per_class["RETURN"]["support"] == 1
    assert per_class["RETURN"]["matchedSupport"] == 0
    matrix = cast(dict[str, object], evaluation["confusionMatrix"])
    counts = cast(dict[str, dict[str, int]], matrix["counts"])
    assert counts["DRIVE"]["UNKNOWN"] == 1


def test_render_overlay_marks_classification_without_modifying_input() -> None:
    shot = _previous(ShotType.DRIVE, bounced=True)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    rendered = render_shot_frame(
        frame,
        frame_number=shot.contact_frame,
        shot=shot,
        recent_shot=None,
        trajectory_point=_ball_frames()[shot.contact_frame],
        recent_trail=_ball_frames()[max(0, shot.contact_frame - 4) : shot.contact_frame + 1],
    )

    assert np.count_nonzero(frame) == 0
    assert np.count_nonzero(rendered) > 0


def test_render_trail_does_not_connect_across_unknown_gap() -> None:
    frames = [
        RallyBallFrame(
            0, 0.0, BallEvidenceStatus.OBSERVED, "segment-1", ImagePoint(30.0, 100.0), 0.9
        ),
        RallyBallFrame(
            1, 0.1, BallEvidenceStatus.OBSERVED, "segment-1", ImagePoint(40.0, 100.0), 0.9
        ),
        RallyBallFrame(2, 0.2, BallEvidenceStatus.UNKNOWN, None, None, None),
        RallyBallFrame(
            3,
            0.3,
            BallEvidenceStatus.OBSERVED,
            "segment-2",
            ImagePoint(260.0, 200.0),
            0.9,
        ),
        RallyBallFrame(
            4,
            0.4,
            BallEvidenceStatus.INTERPOLATED,
            "segment-2",
            ImagePoint(270.0, 200.0),
            0.7,
        ),
    ]
    frame = np.zeros((300, 300, 3), dtype=np.uint8)

    rendered = render_shot_frame(
        frame,
        frame_number=4,
        shot=None,
        recent_shot=None,
        trajectory_point=frames[4],
        recent_trail=frames,
    )

    assert np.count_nonzero(rendered[194:207, 254:278]) > 0
    assert np.count_nonzero(rendered[145:155, 145:155]) == 0


def test_hitter_loader_rejects_incompatible_contact_provenance(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    media = inspect_media(synthetic_video)
    artifact = tmp_path / "hitters.json"
    artifact.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "recordType": "logical_hitter_identifications",
                "source": media.as_dict(),
                "inputs": {
                    "contacts": {"sha256": "wrong"},
                    "playerTracks": {"sha256": "tracks"},
                },
                "hitterIdentifications": [
                    {"contactId": "contact-1", "playerId": "ME", "confidence": 0.9}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ShotReconstructionInputError, match="different contact bytes"):
        load_shot_hitters(
            artifact,
            source=inspect_video(synthetic_video),
            expected_contacts_sha256="contacts",
            expected_player_tracks_sha256="tracks",
            expected_contact_ids={"contact-1"},
        )
