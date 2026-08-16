import json
from pathlib import Path
from typing import cast

import cv2
import pytest

from pickleball_vision.config import ContactDetectionSettings
from pickleball_vision.contact_detection import (
    ContactAudioTransient,
    ContactEvidenceMode,
    ContactPlayerObservation,
    ContactRallyInterval,
    PriorBounce,
    detect_contact_candidates,
)
from pickleball_vision.contact_detection_workflow import (
    detect_contacts_in_video,
    load_contact_audio,
)
from pickleball_vision.contact_evaluation import GroundTruthContact, evaluate_contacts
from pickleball_vision.court import ImagePoint
from pickleball_vision.errors import ContactDetectionInputError
from pickleball_vision.match_annotation import MatchAnnotationStore
from pickleball_vision.media import MediaTimeline, inspect_media
from pickleball_vision.rally_segmentation import BallEvidenceStatus, RallyBallFrame
from pickleball_vision.video import inspect_video

FPS = 10.0
WIDTH = 100
HEIGHT = 100
CONTACT_FRAME = 20


def _contact_timeline(
    *,
    frame_count: int = 41,
    unknown: bool = False,
    interpolated: bool = False,
) -> tuple[RallyBallFrame, ...]:
    frames: list[RallyBallFrame] = []
    for frame in range(frame_count):
        if unknown:
            frames.append(
                RallyBallFrame(
                    frame,
                    frame / FPS,
                    BallEvidenceStatus.UNKNOWN,
                    None,
                    None,
                    None,
                )
            )
            continue
        offset = frame - CONTACT_FRAME
        point = (
            ImagePoint(45 + offset * 2.0, 45 + offset * 0.2)
            if offset <= 0
            else ImagePoint(45 + offset * 0.2, 45 + offset * 2.0)
        )
        frames.append(
            RallyBallFrame(
                frame,
                frame / FPS,
                BallEvidenceStatus.INTERPOLATED if interpolated else BallEvidenceStatus.OBSERVED,
                "ball-segment-1",
                point,
                0.9,
            )
        )
    return tuple(frames)


def _players(
    *,
    frame_count: int = 41,
) -> tuple[tuple[ContactPlayerObservation, ...], ...]:
    return tuple(
        (
            ContactPlayerObservation(
                role="ME",
                display_name="John",
                frame=frame,
                bounding_box=(35.0, 25.0, 55.0, 55.0),
                ground_image_position=ImagePoint(45.0, 55.0),
                tracking_confidence=0.9,
                tracking_state="observed",
                court_side="near_side",
                court_region="inside",
            ),
            ContactPlayerObservation(
                role="OPPONENT_1",
                display_name="Oksana",
                frame=frame,
                bounding_box=(75.0, 20.0, 90.0, 50.0),
                ground_image_position=ImagePoint(82.5, 50.0),
                tracking_confidence=0.85,
                tracking_state="observed",
                court_side="far_side",
                court_region="inside",
            ),
        )
        for frame in range(frame_count)
    )


def test_visual_discontinuity_creates_ranked_players_without_assigning_hitter() -> None:
    result = detect_contact_candidates(
        _contact_timeline(),
        players_by_frame=_players(),
        fps=FPS,
        frame_width_px=WIDTH,
        frame_height_px=HEIGHT,
        settings=ContactDetectionSettings(),
        rallies=(ContactRallyInterval("rally-1", 10, 30, 0.9),),
    )

    assert len(result.candidates) == 1
    contact = result.candidates[0]
    assert contact.frame == CONTACT_FRAME
    assert contact.evidence_mode is ContactEvidenceMode.VISUAL_ONLY
    assert contact.candidate_players[0].role == "ME"
    assert contact.candidate_players[0].rank == 1
    assert contact.candidate_players[1].role == "OPPONENT_1"
    payload = contact.as_dict()
    assert payload["assignedHitter"] is None
    assert all(
        item["isAssignedHitter"] is False
        for item in cast(list[dict[str, object]], payload["candidatePlayers"])
    )
    court = cast(dict[str, object], contact.supporting_signals["courtSideContext"])
    assert court["airborneBallProjectedThroughHomography"] is False
    rally = cast(dict[str, object], contact.supporting_signals["plausibleRallySequence"])
    assert rally["rallyId"] == "rally-1"
    assert rally["canCreateContact"] is False


def test_audio_cannot_create_contact_but_can_support_visual_candidate() -> None:
    audio = (ContactAudioTransient("audio-1", 2.03, 0.95),)
    visual = detect_contact_candidates(
        _contact_timeline(),
        players_by_frame=_players(),
        fps=FPS,
        frame_width_px=WIDTH,
        frame_height_px=HEIGHT,
        settings=ContactDetectionSettings(),
        audio_transients=audio,
    )
    no_visual = detect_contact_candidates(
        _contact_timeline(unknown=True),
        players_by_frame=_players(),
        fps=FPS,
        frame_width_px=WIDTH,
        frame_height_px=HEIGHT,
        settings=ContactDetectionSettings(),
        audio_transients=audio,
    )

    assert no_visual.candidates == ()
    contact = visual.candidates[0]
    assert contact.matched_audio_event_id == "audio-1"
    assert contact.fused_confidence >= contact.visual_confidence
    assert contact.evidence_mode is ContactEvidenceMode.VISUAL_PLUS_AUDIO
    audio_signal = cast(dict[str, object], contact.supporting_signals["audioFusion"])
    assert audio_signal["canCreateContact"] is False


def test_prior_bounce_state_excludes_coincident_reversal_and_retains_previous_state() -> None:
    coincident = detect_contact_candidates(
        _contact_timeline(),
        players_by_frame=_players(),
        fps=FPS,
        frame_width_px=WIDTH,
        frame_height_px=HEIGHT,
        settings=ContactDetectionSettings(),
        prior_bounces=(PriorBounce("bounce-now", CONTACT_FRAME, 0.95),),
    )
    prior = detect_contact_candidates(
        _contact_timeline(),
        players_by_frame=_players(),
        fps=FPS,
        frame_width_px=WIDTH,
        frame_height_px=HEIGHT,
        settings=ContactDetectionSettings(),
        prior_bounces=(PriorBounce("bounce-prior", CONTACT_FRAME - 5, 0.9),),
    )

    assert coincident.candidates == ()
    assert coincident.bounce_excluded_candidate_count >= 1
    state = cast(dict[str, object], prior.candidates[0].supporting_signals["previousBounceState"])
    assert state["previousBounceId"] == "bounce-prior"
    assert state["canCreateContact"] is False


def test_evaluation_compares_same_visual_candidates_with_and_without_audio() -> None:
    settings = ContactDetectionSettings(
        accepted_confidence=0.96,
        audio_confidence_weight=1.0,
    )
    result = detect_contact_candidates(
        _contact_timeline(interpolated=True),
        players_by_frame=_players(),
        fps=FPS,
        frame_width_px=WIDTH,
        frame_height_px=HEIGHT,
        settings=settings,
        audio_transients=(ContactAudioTransient("audio-1", 2.0, 1.0),),
    )
    evaluation = evaluate_contacts(
        result.candidates,
        (GroundTruthContact("human-contact", "PADDLE_CONTACT", 20, 2.0, "ME"),),
        fps=FPS,
        settings=settings,
        annotations_complete=True,
    )

    vision = cast(dict[str, object], evaluation["visionOnly"])
    fused = cast(dict[str, object], evaluation["visionPlusAudio"])
    comparison = cast(dict[str, object], evaluation["comparison"])
    assert vision["recall"] == 0.0
    assert fused["recall"] == 1.0
    assert comparison["sameVisualCandidateSet"] is True
    assert comparison["audioCreatedVisualCandidates"] is False


def test_audio_loader_reapplies_nonzero_configured_offset(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    media = inspect_media(synthetic_video)
    artifact = tmp_path / "audio-events.json"
    artifact.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "recordType": "audio_analysis_observations",
                "audioAnalysisAvailable": True,
                "sourceMedia": media.as_dict(),
                "configuration": {"audioVideoOffsetMs": -20.0},
                "timeline": {"audioStartTimeSeconds": 0.0},
                "audioEventCandidates": [
                    {
                        "id": "audio-1",
                        "candidateType": "TRANSIENT",
                        "semanticClassification": None,
                        "source": "AUDIO",
                        "confidence": 0.8,
                        "analysisTimestampSeconds": 0.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_contact_audio(
        artifact,
        media=media,
        timeline=MediaTimeline(audio_video_offset_ms=50.0),
    )

    assert loaded.stored_offset_ms == -20.0
    assert loaded.applied_offset_ms == 50.0
    assert loaded.transients[0].video_timestamp_seconds == pytest.approx(0.55)


def _ball_tracks_artifact(video: Path, output: Path) -> int:
    metadata = inspect_video(video)
    contact_frame = metadata.frame_count // 2
    frames = []
    for frame in range(metadata.frame_count):
        offset = frame - contact_frame
        point = (
            {
                "x_px": metadata.width * 0.5 + offset * 3.0,
                "y_px": metadata.height * 0.5 + offset * 0.2,
            }
            if offset <= 0
            else {
                "x_px": metadata.width * 0.5 + offset * 0.2,
                "y_px": metadata.height * 0.5 + offset * 3.0,
            }
        )
        frames.append(
            {
                "frame_number": frame,
                "timestamp_s": frame / metadata.fps,
                "status": "OBSERVED",
                "segment_id": "synthetic-contact",
                "source_detection_id": f"detection-{frame}",
                "raw_image_point_px": point,
                "interpolated_image_point_px": None,
                "smoothed_image_point_px": point,
                "confidence": 0.9,
                "detection_confidence": 0.9,
                "primary_court_relevance": 1.0,
                "temporal_support": 1.0,
                "candidate_count": 1,
                "rejected_detection_ids": [],
            }
        )
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "primary_match_ball_trajectory",
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                "source": metadata.as_dict(),
                "statistics": {},
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    return contact_frame


def _player_tracks_artifact(video: Path, output: Path) -> None:
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
                    "timestamp_s": frame / metadata.fps,
                    "tracker_id": 1,
                    "tracker_confidence": 0.9,
                    "detection_confidence": 0.9,
                    "raw_person_detection": {"index": len(raw)},
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
                    "timestamp_s": frame / metadata.fps,
                    "tracker_id": 1,
                    "raw_tracker_observation_id": observation_id,
                    "tracking_confidence": 0.9,
                    "tracking_state": "observed",
                    "ground_contact": {
                        "image_point": {"x_px": (left + right) / 2, "y_px": bottom},
                        "court_point": {"x_m": 2.0, "y_m": 3.0},
                        "method": "bounding_box_bottom_center",
                        "projection_status": "projected_bottom_center_estimate",
                        "court_region": "inside",
                        "court_region_confidence": 0.9,
                        "court_region_boundary_ambiguous": False,
                        "court_side": ("near_side" if role in {"ME", "PARTNER"} else "far_side"),
                        "court_side_confidence": 0.9,
                    },
                    "identity_resolution": {"logical_identity_independent_of_tracker_id": True},
                }
            )
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "persistent_logical_player_tracks",
                "source": metadata.as_dict(),
                "player_names": {"ME": "John"},
                "raw_tracker_layer": {
                    "record_type": "raw_transient_tracker_observations",
                    "observations": raw,
                },
                "logical_identity_layer": logical,
            }
        ),
        encoding="utf-8",
    )


def test_workflow_writes_contacts_debug_video_and_comparative_evaluation(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    ball_tracks = tmp_path / "ball_tracks.json"
    player_tracks = tmp_path / "tracks.json"
    contact_frame = _ball_tracks_artifact(synthetic_video, ball_tracks)
    _player_tracks_artifact(synthetic_video, player_tracks)
    annotations = tmp_path / "annotations.json"
    store = MatchAnnotationStore(synthetic_video, output_path=annotations)
    store.add_event({"type": "RALLY_START", "frame": 1})
    store.add_event({"type": "SERVE_CONTACT", "frame": contact_frame, "playerId": "ME"})
    store.add_event({"type": "RALLY_END", "frame": inspect_video(synthetic_video).frame_count - 2})

    artifacts = detect_contacts_in_video(
        synthetic_video,
        ball_tracks_path=ball_tracks,
        player_tracks_path=player_tracks,
        annotations_path=annotations,
        output_dir=tmp_path / "contact-output",
        settings=ContactDetectionSettings(
            minimum_velocity_change_diagonals_per_second=0.02,
        ),
        timeline=MediaTimeline(),
    )

    assert artifacts.visual_candidate_count >= 1
    contacts = json.loads(artifacts.contacts_path.read_text(encoding="utf-8"))
    evaluation = json.loads(artifacts.evaluation_path.read_text(encoding="utf-8"))
    assert contacts["recordType"] == "multimodal_paddle_contact_candidates"
    assert contacts["contracts"]["audioCanCreateContact"] is False
    assert contacts["contracts"]["hitterIdentificationImplemented"] is False
    assert all(item["assignedHitter"] is None for item in contacts["contactCandidates"])
    assert evaluation["evaluationAvailable"] is True
    assert "visionOnly" in evaluation and "visionPlusAudio" in evaluation
    capture = cv2.VideoCapture(str(artifacts.debug_video_path))
    try:
        assert (
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == inspect_video(synthetic_video).frame_count
        )
    finally:
        capture.release()


def test_workflow_rejects_player_tracks_from_incompatible_source(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    ball_tracks = tmp_path / "ball_tracks.json"
    player_tracks = tmp_path / "tracks.json"
    _ball_tracks_artifact(synthetic_video, ball_tracks)
    _player_tracks_artifact(synthetic_video, player_tracks)
    payload = json.loads(player_tracks.read_text(encoding="utf-8"))
    payload["source"]["frame_count"] -= 1
    player_tracks.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ContactDetectionInputError, match="frame_count"):
        detect_contacts_in_video(
            synthetic_video,
            ball_tracks_path=ball_tracks,
            player_tracks_path=player_tracks,
            output_dir=tmp_path / "contact-output",
            settings=ContactDetectionSettings(),
            timeline=MediaTimeline(),
        )


def test_player_ground_point_may_differ_from_transient_tracker_box(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    ball_tracks = tmp_path / "ball_tracks.json"
    player_tracks = tmp_path / "tracks.json"
    _ball_tracks_artifact(synthetic_video, ball_tracks)
    _player_tracks_artifact(synthetic_video, player_tracks)
    payload = json.loads(player_tracks.read_text(encoding="utf-8"))
    payload["raw_tracker_layer"]["observations"][0]["tracker_bounding_box"]["bottom_px"] -= 3
    player_tracks.write_text(json.dumps(payload), encoding="utf-8")

    artifacts = detect_contacts_in_video(
        synthetic_video,
        ball_tracks_path=ball_tracks,
        player_tracks_path=player_tracks,
        output_dir=tmp_path / "contact-output",
        settings=ContactDetectionSettings(
            minimum_velocity_change_diagonals_per_second=0.02,
        ),
        timeline=MediaTimeline(),
    )

    assert artifacts.contacts_path.is_file()
