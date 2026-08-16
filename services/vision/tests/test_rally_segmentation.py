import json
from pathlib import Path
from typing import cast

import cv2
import pytest

from pickleball_vision.config import RallySegmentationSettings
from pickleball_vision.court import ImagePoint
from pickleball_vision.match_annotation import MatchAnnotationStore
from pickleball_vision.rally_evaluation import GroundTruthRally, evaluate_rallies
from pickleball_vision.rally_segmentation import (
    AudioTransientEvidence,
    BallEvidenceStatus,
    RallyBallFrame,
    RallyPrediction,
    segment_rallies,
)
from pickleball_vision.rally_segmentation_workflow import segment_rallies_in_video
from pickleball_vision.video import inspect_video


def _timeline(
    *,
    frame_count: int = 200,
    fps: float = 10.0,
    intervals: tuple[tuple[int, int, str], ...] = (
        (20, 50, "segment-1"),
        (100, 145, "segment-2"),
    ),
    unknown_frames: frozenset[int] = frozenset({35, 36, 37, 38}),
) -> tuple[RallyBallFrame, ...]:
    frames: list[RallyBallFrame] = []
    for frame_number in range(frame_count):
        match = next(
            (item for item in intervals if item[0] <= frame_number <= item[1]),
            None,
        )
        if match is None or frame_number in unknown_frames:
            frames.append(
                RallyBallFrame(
                    frame_number,
                    frame_number / fps,
                    BallEvidenceStatus.UNKNOWN,
                    None,
                    None,
                    None,
                )
            )
            continue
        start, _end, segment_id = match
        offset = frame_number - start
        x_px = 10.0 + offset * 2.0 + (8.0 if offset >= 1 else 0.0)
        frames.append(
            RallyBallFrame(
                frame_number,
                frame_number / fps,
                BallEvidenceStatus.OBSERVED,
                segment_id,
                ImagePoint(x_px=x_px, y_px=30.0 + offset),
                0.85,
            )
        )
    return tuple(frames)


def test_structured_ball_activity_segments_two_rallies_and_preserves_long_gap() -> None:
    result = segment_rallies(
        _timeline(),
        fps=10.0,
        frame_width_px=100,
        frame_height_px=100,
        settings=RallySegmentationSettings(),
    )

    assert len(result.rallies) == 2
    first, second = result.rallies
    assert first.start_frame <= 22
    assert first.end_frame <= 55
    assert second.start_frame >= 98
    assert second.end_frame <= 150
    assert first.end_frame < second.start_frame
    serve_signal = cast(dict[str, object], first.supporting_signals["serveLikeSequence"])
    audio_signal = cast(dict[str, object], first.supporting_signals["audioSupport"])
    assert serve_signal["detected"] is True
    assert audio_signal["effect"] == "none_vision_only"


def test_short_unknown_gap_does_not_end_rally_but_long_gap_remains_between_rallies() -> None:
    result = segment_rallies(
        _timeline(unknown_frames=frozenset({32, 33, 34, 35, 36})),
        fps=10.0,
        frame_width_px=100,
        frame_height_px=100,
        settings=RallySegmentationSettings(),
    )

    assert len(result.rallies) == 2
    assert result.rallies[0].start_frame < 32 < result.rallies[0].end_frame
    assert result.rallies[0].end_frame < 100


def test_weaker_adjacent_dead_ball_handoff_burst_is_retained_but_not_a_rally() -> None:
    result = segment_rallies(
        _timeline(
            frame_count=100,
            intervals=((20, 50, "rally"), (60, 70, "dead-ball-handoff")),
            unknown_frames=frozenset(),
        ),
        fps=10.0,
        frame_width_px=100,
        frame_height_px=100,
        settings=RallySegmentationSettings(),
    )

    assert len(result.rallies) == 1
    assert result.rallies[0].start_frame <= 22
    assert len(result.rejected_adjacent_bursts) == 1
    rejection = result.rejected_adjacent_bursts[0]
    assert rejection.candidate.start_frame >= 58
    assert rejection.gap_seconds <= 2.25
    assert rejection.quality_margin >= 0.05
    serialized = rejection.as_dict()
    assert serialized["possibleDeadBallHandoff"] is True
    assert serialized["semanticClassification"] is None
    candidate = cast(dict[str, object], serialized["candidate"])
    signals = cast(dict[str, object], candidate["supportingSignals"])
    assessment = cast(dict[str, object], signals["deadBallHandoffAssessment"])
    assert assessment["possibleDeadBallHandoff"] is True
    assert assessment["rejectedAsRally"] is True


def test_audio_cannot_create_rally_and_only_increases_existing_confidence() -> None:
    audio = (AudioTransientEvidence("audio-1", 2.0, 0.99),)
    empty = tuple(
        RallyBallFrame(
            frame,
            frame / 10,
            BallEvidenceStatus.UNKNOWN,
            None,
            None,
            None,
        )
        for frame in range(100)
    )
    no_visual_result = segment_rallies(
        empty,
        fps=10.0,
        frame_width_px=100,
        frame_height_px=100,
        settings=RallySegmentationSettings(),
        audio_transients=audio,
    )
    assert no_visual_result.rallies == ()

    visual_only = segment_rallies(
        _timeline(),
        fps=10.0,
        frame_width_px=100,
        frame_height_px=100,
        settings=RallySegmentationSettings(),
    )
    supported = segment_rallies(
        _timeline(),
        fps=10.0,
        frame_width_px=100,
        frame_height_px=100,
        settings=RallySegmentationSettings(),
        player_reset_scores=tuple(1.0 for _ in range(200)),
        audio_transients=audio,
    )
    assert [item.start_frame for item in supported.rallies] == [
        item.start_frame for item in visual_only.rallies
    ]
    assert supported.rallies[0].confidence > visual_only.rallies[0].confidence
    audio_signal = cast(
        dict[str, object],
        supported.rallies[0].supporting_signals["audioSupport"],
    )
    assert audio_signal["canCreateBoundary"] is False


def _prediction(rally_id: str, start: int, end: int, fps: float = 10.0) -> RallyPrediction:
    return RallyPrediction(rally_id, start, end, start / fps, end / fps, 0.8, {})


def _annotation(rally_id: str, start: int, end: int, fps: float = 10.0) -> GroundTruthRally:
    return GroundTruthRally(
        rally_id,
        f"{rally_id}-start",
        f"{rally_id}-end",
        start,
        end,
        start / fps,
        end / fps,
    )


def test_evaluation_reports_one_to_one_metrics_and_sparse_coverage() -> None:
    predictions = (
        _prediction("prediction-1", 21, 51),
        _prediction("prediction-2", 101, 140),
        _prediction("outside-reviewed-time", 170, 185),
    )
    annotations = (
        _annotation("truth-1", 20, 50),
        _annotation("truth-2", 100, 145),
        _annotation("missed", 60, 75),
    )

    sparse = evaluate_rallies(
        predictions,
        annotations,
        fps=10.0,
        settings=RallySegmentationSettings(sparse_evaluation_margin_seconds=1.0),
        annotations_complete=False,
        evaluation_partition="validation",
    )
    assert sparse["matchedRallyCount"] == 2
    assert sparse["missedRallyCount"] == 1
    assert sparse["falseRallyCount"] == 0
    assert sparse["precision"] == 1.0
    assert sparse["recall"] == pytest.approx(2 / 3)
    coverage = cast(dict[str, object], sparse["annotationCoverage"])
    assert coverage["ignoredPredictionIds"] == ["outside-reviewed-time"]
    assert sparse["thresholdTuningPerformed"] is False

    complete = evaluate_rallies(
        predictions,
        annotations,
        fps=10.0,
        settings=RallySegmentationSettings(),
        annotations_complete=True,
        evaluation_partition="test",
    )
    assert complete["falseRallyCount"] == 1
    assert complete["heldOutThresholdTuningAllowed"] is False


def _ball_tracks_artifact(video: Path, output: Path) -> None:
    metadata = inspect_video(video)
    frames = []
    for frame_number in range(metadata.frame_count):
        point = {
            "x_px": 8.0 + frame_number * 6.0,
            "y_px": 20.0 + frame_number * 2.0,
        }
        frames.append(
            {
                "frame_number": frame_number,
                "timestamp_s": frame_number / metadata.fps,
                "status": "OBSERVED",
                "segment_id": "synthetic-rally",
                "source_detection_id": f"detection-{frame_number}",
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


def test_workflow_writes_rallies_debug_video_and_evaluation(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    ball_tracks = tmp_path / "ball_tracks.json"
    _ball_tracks_artifact(synthetic_video, ball_tracks)
    annotations_path = tmp_path / "annotations.json"
    store = MatchAnnotationStore(synthetic_video, output_path=annotations_path)
    store.add_event({"type": "RALLY_START", "frame": 1})
    store.add_event({"type": "RALLY_END", "frame": 11})
    output = tmp_path / "rallies"

    artifacts = segment_rallies_in_video(
        synthetic_video,
        ball_tracks_path=ball_tracks,
        annotations_path=annotations_path,
        output_dir=output,
        settings=RallySegmentationSettings(
            serve_minimum_displacement_diagonal_fraction=0.04,
            minimum_rally_duration_seconds=0.5,
        ),
    )

    assert artifacts.rally_count >= 1
    assert artifacts.rallies_path.is_file()
    assert artifacts.debug_video_path.is_file()
    assert artifacts.evaluation_path.is_file()
    capture = cv2.VideoCapture(str(artifacts.debug_video_path))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    finally:
        capture.release()
    rallies = json.loads(artifacts.rallies_path.read_text(encoding="utf-8"))
    evaluation = json.loads(artifacts.evaluation_path.read_text(encoding="utf-8"))
    assert rallies["recordType"] == "automatic_rally_segments"
    assert isinstance(rallies["rejectedCandidates"], list)
    assert rallies["statistics"]["rejectedAdjacentBurstCount"] == len(rallies["rejectedCandidates"])
    assert rallies["contracts"]["annotationsUsedForInference"] is False
    assert rallies["contracts"]["audioCanCreateBoundary"] is False
    assert evaluation["evaluationAvailable"] is True
    assert evaluation["thresholdTuningPerformed"] is False
