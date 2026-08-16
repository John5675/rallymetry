import json
from pathlib import Path
from typing import cast

import cv2
import pytest

from pickleball_vision.bounce_detection import (
    BounceAudioTransient,
    BounceEvidenceMode,
    detect_bounce_candidates,
)
from pickleball_vision.bounce_detection_workflow import (
    detect_bounces_in_video,
    load_bounce_audio,
)
from pickleball_vision.bounce_evaluation import GroundTruthBounce, evaluate_bounces
from pickleball_vision.calibration import (
    CalibrationCorrespondence,
    CalibrationSource,
    CourtCalibration,
    fit_calibration,
)
from pickleball_vision.config import BounceDetectionSettings
from pickleball_vision.court import CourtDimensions, ImagePoint, court_landmarks
from pickleball_vision.match_annotation import MatchAnnotationStore
from pickleball_vision.media import MediaTimeline, inspect_media
from pickleball_vision.rally_segmentation import BallEvidenceStatus, RallyBallFrame
from pickleball_vision.video import inspect_video

FPS = 10.0
WIDTH = 100
HEIGHT = 100


def _calibration(video_path: Path = Path("/video/synthetic.mp4")) -> CourtCalibration:
    court = CourtDimensions()
    landmarks = court_landmarks(court)
    selected = (landmarks[0], landmarks[1], landmarks[8], landmarks[9])
    correspondences = tuple(
        CalibrationCorrespondence(
            landmark=landmark.name,
            label=landmark.label,
            image_point=ImagePoint(
                15 + landmark.court_point.x_m * 10,
                15 + landmark.court_point.y_m * 5,
            ),
            court_point=landmark.court_point,
        )
        for landmark in selected
    )
    return fit_calibration(
        source=CalibrationSource(
            video_path=video_path,
            requested_timestamp_s=0.0,
            frame_index=0,
            frame_timestamp_s=0.0,
            frame_width_px=WIDTH,
            frame_height_px=HEIGHT,
            fps=FPS,
        ),
        court=court,
        correspondences=correspondences,
    )


def _bounce_timeline(
    *,
    frame_count: int = 41,
    bounce_frame: int = 20,
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
        offset = frame - bounce_frame
        frames.append(
            RallyBallFrame(
                frame,
                frame / FPS,
                (BallEvidenceStatus.INTERPOLATED if interpolated else BallEvidenceStatus.OBSERVED),
                "ball-segment-1",
                ImagePoint(45 + offset * 0.4, 55 - 1.5 * offset * offset),
                0.9,
            )
        )
    return tuple(frames)


def test_visual_reversal_creates_bounce_and_only_then_projects_to_court() -> None:
    result = detect_bounce_candidates(
        _bounce_timeline(),
        fps=FPS,
        frame_width_px=WIDTH,
        frame_height_px=HEIGHT,
        calibration=_calibration(),
        settings=BounceDetectionSettings(),
    )

    assert len(result.candidates) == 1
    bounce = result.candidates[0]
    assert bounce.frame == 20
    assert bounce.evidence_mode is BounceEvidenceMode.VISUAL_ONLY
    assert bounce.accepted_fused is True
    assert bounce.matched_audio_event_id is None
    assert bounce.court_position is not None
    projection = cast(dict[str, object], bounce.supporting_signals["courtProjection"])
    assert projection["appliedOnlyAfterVisualPlaneContactPlausibility"] is True
    assert projection["airborneBallProjected"] is False


def test_audio_cannot_create_candidate_but_can_support_visual_candidate() -> None:
    audio = (BounceAudioTransient("audio-1", 2.03, 0.95),)
    visual = detect_bounce_candidates(
        _bounce_timeline(),
        fps=FPS,
        frame_width_px=WIDTH,
        frame_height_px=HEIGHT,
        calibration=_calibration(),
        settings=BounceDetectionSettings(),
        audio_transients=audio,
    )
    no_visual = detect_bounce_candidates(
        _bounce_timeline(unknown=True),
        fps=FPS,
        frame_width_px=WIDTH,
        frame_height_px=HEIGHT,
        calibration=_calibration(),
        settings=BounceDetectionSettings(),
        audio_transients=audio,
    )

    assert no_visual.candidates == ()
    bounce = visual.candidates[0]
    assert bounce.matched_audio_event_id == "audio-1"
    assert bounce.audio_confidence > 0
    assert bounce.fused_confidence >= bounce.visual_confidence
    assert bounce.evidence_mode is BounceEvidenceMode.VISUAL_PLUS_AUDIO
    audio_signal = cast(dict[str, object], bounce.supporting_signals["audioFusion"])
    assert audio_signal["canCreateBounce"] is False


def test_evaluation_compares_same_candidates_with_and_without_audio() -> None:
    settings = BounceDetectionSettings(accepted_confidence=0.90, audio_confidence_weight=0.8)
    result = detect_bounce_candidates(
        _bounce_timeline(interpolated=True),
        fps=FPS,
        frame_width_px=WIDTH,
        frame_height_px=HEIGHT,
        calibration=_calibration(),
        settings=settings,
        audio_transients=(BounceAudioTransient("audio-1", 2.0, 1.0),),
    )
    annotated = (GroundTruthBounce("human-bounce", 20, 2.0),)
    evaluation = evaluate_bounces(
        result.candidates,
        annotated,
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

    loaded = load_bounce_audio(
        artifact,
        media=media,
        timeline=MediaTimeline(audio_video_offset_ms=50.0),
    )

    assert loaded.stored_offset_ms == -20.0
    assert loaded.applied_offset_ms == 50.0
    assert loaded.transients[0].video_timestamp_seconds == pytest.approx(0.55)


def _ball_tracks_artifact(video: Path, output: Path) -> None:
    metadata = inspect_video(video)
    bounce_frame = metadata.frame_count // 2
    frames = []
    for frame in range(metadata.frame_count):
        offset = frame - bounce_frame
        point = {
            "x_px": metadata.width * 0.5 + offset * 0.2,
            "y_px": metadata.height * 0.70 - 1.2 * offset * offset,
        }
        frames.append(
            {
                "frame_number": frame,
                "timestamp_s": frame / metadata.fps,
                "status": "OBSERVED",
                "segment_id": "synthetic-bounce",
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


def test_workflow_writes_candidates_debug_video_and_comparative_evaluation(
    synthetic_video: Path,
    synthetic_calibration: Path,
    tmp_path: Path,
) -> None:
    ball_tracks = tmp_path / "ball_tracks.json"
    _ball_tracks_artifact(synthetic_video, ball_tracks)
    annotations = tmp_path / "annotations.json"
    store = MatchAnnotationStore(synthetic_video, output_path=annotations)
    bounce_frame = inspect_video(synthetic_video).frame_count // 2
    store.add_event({"type": "RALLY_START", "frame": 1})
    store.add_event({"type": "BOUNCE", "frame": bounce_frame})
    store.add_event({"type": "RALLY_END", "frame": inspect_video(synthetic_video).frame_count - 2})
    output = tmp_path / "bounce-output"

    artifacts = detect_bounces_in_video(
        synthetic_video,
        ball_tracks_path=ball_tracks,
        calibration_path=synthetic_calibration,
        annotations_path=annotations,
        output_dir=output,
        settings=BounceDetectionSettings(
            minimum_shape_prominence_diagonal_fraction=0.0005,
        ),
        timeline=MediaTimeline(),
    )

    assert artifacts.visual_candidate_count >= 1
    bounces = json.loads(artifacts.bounces_path.read_text(encoding="utf-8"))
    evaluation = json.loads(artifacts.evaluation_path.read_text(encoding="utf-8"))
    assert bounces["recordType"] == "multimodal_bounce_candidates"
    assert bounces["contracts"]["audioCanCreateBounce"] is False
    assert bounces["contracts"]["airborneBallProjectedThroughHomography"] is False
    assert evaluation["evaluationAvailable"] is True
    assert "visionOnly" in evaluation and "visionPlusAudio" in evaluation
    capture = cv2.VideoCapture(str(artifacts.debug_video_path))
    try:
        assert (
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == inspect_video(synthetic_video).frame_count
        )
    finally:
        capture.release()
