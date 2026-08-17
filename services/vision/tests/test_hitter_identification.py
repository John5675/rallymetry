import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import cv2

from pickleball_vision.config import HitterIdentificationSettings
from pickleball_vision.contact_detection import (
    ContactCandidate,
    ContactCandidatePlayer,
    ContactEvidenceMode,
)
from pickleball_vision.contact_evaluation import GroundTruthContact
from pickleball_vision.court import ImagePoint
from pickleball_vision.hitter_evaluation import evaluate_hitters
from pickleball_vision.hitter_identification import UNKNOWN_PLAYER_ID, identify_hitters
from pickleball_vision.hitter_identification_workflow import identify_hitters_in_video
from pickleball_vision.match_annotation import MatchAnnotationStore
from pickleball_vision.media import inspect_media
from pickleball_vision.rally_segmentation import BallEvidenceStatus
from pickleball_vision.video import inspect_video

WIDTH = 100
HEIGHT = 100
FPS = 10.0


def _player(
    role: str,
    *,
    distance_fraction: float,
    proximity: float,
    side: str,
    tracking: float = 0.9,
    state: str = "observed",
    box: tuple[float, float, float, float] = (40.0, 25.0, 60.0, 60.0),
) -> ContactCandidatePlayer:
    return ContactCandidatePlayer(
        role=role,
        display_name=role.title(),
        rank=1,
        bounding_box=box,
        ground_image_position=ImagePoint((box[0] + box[2]) / 2, box[3]),
        tracking_confidence=tracking,
        tracking_state=state,
        court_side=side,
        court_region="inside",
        distance_px=distance_fraction * (WIDTH**2 + HEIGHT**2) ** 0.5,
        distance_diagonal_fraction=distance_fraction,
        proximity_confidence=proximity,
        ball_inside_person_box=distance_fraction == 0,
    )


def _contact(
    contact_id: str,
    frame: int,
    players: tuple[ContactCandidatePlayer, ...],
    *,
    visual_confidence: float = 0.95,
    before_y: float = 60.0,
    after_y: float = -60.0,
    rally_id: str | None = "rally-1",
) -> ContactCandidate:
    ranked = tuple(
        ContactCandidatePlayer(
            role=item.role,
            display_name=item.display_name,
            rank=index,
            bounding_box=item.bounding_box,
            ground_image_position=item.ground_image_position,
            tracking_confidence=item.tracking_confidence,
            tracking_state=item.tracking_state,
            court_side=item.court_side,
            court_region=item.court_region,
            distance_px=item.distance_px,
            distance_diagonal_fraction=item.distance_diagonal_fraction,
            proximity_confidence=item.proximity_confidence,
            ball_inside_person_box=item.ball_inside_person_box,
        )
        for index, item in enumerate(players, start=1)
    )
    return ContactCandidate(
        contact_id=contact_id,
        frame=frame,
        timestamp_seconds=frame / FPS,
        media_timestamp_seconds=frame / FPS,
        ball_image_position=ImagePoint(50.0, 45.0),
        trajectory_status=BallEvidenceStatus.OBSERVED,
        candidate_players=ranked,
        visual_confidence=visual_confidence,
        audio_confidence=1.0,
        fused_confidence=max(visual_confidence, 0.99),
        matched_audio_event_id="audio-ignored",
        evidence_mode=ContactEvidenceMode.VISUAL_PLUS_AUDIO,
        accepted_vision_only=visual_confidence >= 0.78,
        accepted_fused=True,
        supporting_signals={
            "trajectoryVelocityDiscontinuity": {
                "beforeVelocityPixelsPerSecond": {"x": 5.0, "y": before_y},
                "afterVelocityPixelsPerSecond": {"x": -5.0, "y": after_y},
            },
            "plausibleRallySequence": {"rallyId": rally_id},
        },
    )


def test_clear_proximity_assigns_logical_hitter_without_using_audio() -> None:
    contact = _contact(
        "contact-1",
        10,
        (
            _player("ME", distance_fraction=0.0, proximity=1.0, side="near_side"),
            _player(
                "OPPONENT_1",
                distance_fraction=0.10,
                proximity=0.1,
                side="far_side",
            ),
        ),
    )

    result = identify_hitters(
        (contact,),
        frame_width_px=WIDTH,
        frame_height_px=HEIGHT,
        settings=HitterIdentificationSettings(),
    ).identifications[0]

    assert result.player_id == "ME"
    assert result.confidence >= 0.62
    evidence = cast(dict[str, object], result.supporting_signals["contactEvidence"])
    assert evidence["confidenceUsedForIdentity"] == "visualConfidence"
    assert evidence["audioIdentityContribution"] == 0.0


def test_ambiguous_players_and_weak_inputs_remain_unknown() -> None:
    equal = (
        _player("ME", distance_fraction=0.03, proximity=0.75, side="near_side"),
        _player("PARTNER", distance_fraction=0.03, proximity=0.75, side="near_side"),
    )
    contacts = (
        _contact("ambiguous", 10, equal, before_y=0.0, after_y=0.0),
        _contact("low-contact", 20, equal, visual_confidence=0.40),
        _contact(
            "low-tracking",
            30,
            (_player("ME", distance_fraction=0.0, proximity=1.0, side="near_side", tracking=0.2),),
        ),
    )

    results = identify_hitters(
        contacts,
        frame_width_px=WIDTH,
        frame_height_px=HEIGHT,
        settings=HitterIdentificationSettings(),
    ).identifications

    assert [item.player_id for item in results] == [UNKNOWN_PLAYER_ID] * 3
    decisions = [cast(dict[str, object], item.supporting_signals["decision"]) for item in results]
    assert "assignmentMargin" in cast(list[str], decisions[0]["failedGates"])
    assert "contactConfidence" in cast(list[str], decisions[1]["failedGates"])
    assert "trackingConfidence" in cast(list[str], decisions[2]["failedGates"])


def test_direction_and_previous_hitter_support_opposite_court_side() -> None:
    near = _player("ME", distance_fraction=0.0, proximity=1.0, side="near_side")
    far = _player("OPPONENT_1", distance_fraction=0.03, proximity=0.75, side="far_side")
    equal_near = _player("ME", distance_fraction=0.03, proximity=0.75, side="near_side")
    first = _contact("contact-1", 10, (near, far))
    second = _contact(
        "contact-2",
        20,
        (equal_near, far),
        before_y=-60.0,
        after_y=60.0,
    )

    results = identify_hitters(
        (first, second),
        frame_width_px=WIDTH,
        frame_height_px=HEIGHT,
        settings=HitterIdentificationSettings(),
    ).identifications

    assert results[0].player_id == "ME"
    assert results[1].player_id == "OPPONENT_1"
    previous = cast(dict[str, object], results[1].supporting_signals["previousHitter"])
    assert previous["applied"] is True
    assert previous["playerId"] == "ME"


def test_unknown_contact_resets_previous_hitter_context() -> None:
    near = _player("ME", distance_fraction=0.0, proximity=1.0, side="near_side")
    same_side_equal = (
        _player("ME", distance_fraction=0.03, proximity=0.75, side="near_side"),
        _player("PARTNER", distance_fraction=0.03, proximity=0.75, side="near_side"),
    )
    cross_side_equal = (
        _player("ME", distance_fraction=0.03, proximity=0.75, side="near_side"),
        _player("OPPONENT_1", distance_fraction=0.03, proximity=0.75, side="far_side"),
    )
    contacts = (
        _contact("first", 10, (near,)),
        _contact("unknown", 20, same_side_equal, before_y=0.0, after_y=0.0),
        _contact("after-unknown", 30, cross_side_equal, before_y=0.0, after_y=0.0),
    )

    results = identify_hitters(
        contacts,
        frame_width_px=WIDTH,
        frame_height_px=HEIGHT,
        settings=HitterIdentificationSettings(),
    ).identifications

    assert results[1].player_id == UNKNOWN_PLAYER_ID
    assert results[2].player_id == UNKNOWN_PLAYER_ID
    previous = cast(dict[str, object], results[2].supporting_signals["previousHitter"])
    assert previous["applied"] is False


def test_evaluation_reports_overall_and_observed_near_far_accuracy() -> None:
    contacts = (
        _contact(
            "near-contact",
            10,
            (_player("ME", distance_fraction=0.0, proximity=1.0, side="near_side"),),
        ),
        _contact(
            "far-contact",
            20,
            (
                _player(
                    "OPPONENT_1",
                    distance_fraction=0.0,
                    proximity=1.0,
                    side="far_side",
                ),
            ),
            before_y=-60.0,
            after_y=60.0,
        ),
    )
    results = identify_hitters(
        contacts,
        frame_width_px=WIDTH,
        frame_height_px=HEIGHT,
        settings=HitterIdentificationSettings(),
    ).identifications

    evaluation = evaluate_hitters(
        results,
        (
            GroundTruthContact("human-near", "PADDLE_CONTACT", 10, 1.0, "ME"),
            GroundTruthContact("human-far", "PADDLE_CONTACT", 20, 2.0, "OPPONENT_1"),
        ),
        settings=HitterIdentificationSettings(),
    )

    overall = cast(dict[str, object], evaluation["overall"])
    by_side = cast(dict[str, object], evaluation["byObservedPlayerCourtSide"])
    assert overall["accuracy"] == 1.0
    assert cast(dict[str, object], by_side["near_side"])["accuracy"] == 1.0
    assert cast(dict[str, object], by_side["far_side"])["accuracy"] == 1.0


def _player_tracks(video: Path, output: Path) -> None:
    metadata = inspect_video(video)
    boxes = {
        "ME": (38.0, 20.0, 58.0, 55.0),
        "PARTNER": (5.0, 20.0, 20.0, 55.0),
        "OPPONENT_1": (65.0, 10.0, 78.0, 42.0),
        "OPPONENT_2": (80.0, 10.0, 94.0, 42.0),
    }
    raw: list[dict[str, object]] = []
    logical: dict[str, list[dict[str, object]]] = {role: [] for role in boxes}
    for frame in range(metadata.frame_count):
        for role, box in boxes.items():
            observation_id = f"tracker-{frame}-{role}"
            left, top, right, bottom = box
            raw.append(
                {
                    "observation_id": observation_id,
                    "frame_number": frame,
                    "tracker_bounding_box": {
                        "left_px": left,
                        "top_px": top,
                        "right_px": right,
                        "bottom_px": bottom,
                    },
                }
            )
            logical[role].append(
                {
                    "frame_number": frame,
                    "tracking_confidence": 0.9,
                    "tracking_state": "observed",
                    "raw_tracker_observation_id": observation_id,
                    "ground_contact": {
                        "method": "bounding_box_bottom_center",
                        "image_point": {"x_px": (left + right) / 2, "y_px": bottom},
                        "court_region": "inside",
                        "court_side": "near_side" if role in {"ME", "PARTNER"} else "far_side",
                    },
                }
            )
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "persistent_logical_player_tracks",
                "source": metadata.as_dict(),
                "player_names": {"ME": "John"},
                "raw_tracker_layer": {"observations": raw},
                "logical_identity_layer": logical,
            }
        ),
        encoding="utf-8",
    )


def test_workflow_writes_separate_hitter_artifacts_and_debug_video(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    metadata = inspect_video(synthetic_video)
    player_tracks = tmp_path / "tracks.json"
    _player_tracks(synthetic_video, player_tracks)
    player_hash = hashlib.sha256(player_tracks.read_bytes()).hexdigest()
    frame = metadata.frame_count // 2
    contact = replace(
        _contact(
            "contact-1",
            frame,
            (_player("ME", distance_fraction=0.0, proximity=1.0, side="near_side"),),
        ),
        timestamp_seconds=frame / metadata.fps,
        media_timestamp_seconds=frame / metadata.fps,
    )
    contacts = tmp_path / "contacts.json"
    contacts.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "recordType": "multimodal_paddle_contact_candidates",
                "createdAtUtc": "2026-01-01T00:00:00+00:00",
                "source": inspect_media(synthetic_video).as_dict(),
                "inputs": {"playerTracks": {"sha256": player_hash}},
                "contracts": {
                    "candidatePlayersAreNotHitterAssignments": True,
                    "hitterIdentificationImplemented": False,
                },
                "contactCandidates": [contact.as_dict()],
            }
        ),
        encoding="utf-8",
    )
    annotations = tmp_path / "annotations.json"
    store = MatchAnnotationStore(synthetic_video, output_path=annotations)
    store.add_event({"type": "SERVE_CONTACT", "frame": frame, "playerId": "ME"})

    artifacts = identify_hitters_in_video(
        synthetic_video,
        contacts_path=contacts,
        player_tracks_path=player_tracks,
        annotations_path=annotations,
        output_dir=tmp_path / "hitter-output",
        settings=HitterIdentificationSettings(),
    )

    hitters = json.loads(artifacts.hitters_path.read_text(encoding="utf-8"))
    evaluation = json.loads(artifacts.evaluation_path.read_text(encoding="utf-8"))
    assert hitters["recordType"] == "logical_hitter_identifications"
    assert hitters["contracts"]["audioUsedForIdentity"] is False
    assert hitters["hitterIdentifications"][0]["playerId"] == "ME"
    assert evaluation["overall"]["accuracy"] == 1.0
    capture = cv2.VideoCapture(str(artifacts.debug_video_path))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == metadata.frame_count
    finally:
        capture.release()
