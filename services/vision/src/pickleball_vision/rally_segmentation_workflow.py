"""Artifact loading, evaluation, persistence, and rendering for rally segmentation."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar, cast

from pickleball_vision.ball_tracking import BALL_TRACKING_SCHEMA_VERSION
from pickleball_vision.config import RallySegmentationSettings
from pickleball_vision.court import ImagePoint
from pickleball_vision.errors import OutputWriteError, RallySegmentationInputError
from pickleball_vision.match_annotation import (
    MATCH_ANNOTATION_RECORD_TYPE,
    MATCH_ANNOTATION_VERSION,
)
from pickleball_vision.rally_evaluation import (
    GroundTruthRally,
    evaluate_rallies,
    unavailable_evaluation,
)
from pickleball_vision.rally_segmentation import (
    RALLY_SEGMENTATION_SCHEMA_VERSION,
    AudioTransientEvidence,
    BallEvidenceStatus,
    RallyBallFrame,
    RallyPrediction,
    segment_rallies,
)
from pickleball_vision.rally_segmentation_render import render_rally_segmentation_frame
from pickleball_vision.video import VideoMetadata, inspect_video, iter_video_frames
from pickleball_vision.video_output import CompressedVideoWriter

RALLIES_NAME = "rallies.json"
RALLY_DEBUG_NAME = "rally-debug.mp4"
RALLY_EVALUATION_NAME = "rally-evaluation.json"
IntervalT = TypeVar("IntervalT", RallyPrediction, GroundTruthRally)


@dataclass(frozen=True, slots=True)
class LoadedBallTrajectory:
    """Validated ball-trajectory input and provenance."""

    path: Path
    sha256: str
    source: VideoMetadata
    created_at_utc: str
    frames: tuple[RallyBallFrame, ...]
    statistics: dict[str, object]


@dataclass(frozen=True, slots=True)
class LoadedAudioEvidence:
    """Optional generic transient evidence kept separate from rally boundaries."""

    requested: bool
    available: bool
    path: Path | None
    sha256: str | None
    transients: tuple[AudioTransientEvidence, ...]
    reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "audioAnalysisAvailable": self.available,
            "path": str(self.path) if self.path is not None else None,
            "sha256": self.sha256,
            "transientCandidateCount": len(self.transients),
            "usage": "confidence_support_only",
            "canCreateRallyBoundary": False,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class LoadedPlayerResetEvidence:
    """Optional source-compatible player reset evidence."""

    requested: bool
    available: bool
    path: Path | None
    sha256: str | None
    scores: tuple[float | None, ...] | None
    reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "available": self.available,
            "path": str(self.path) if self.path is not None else None,
            "sha256": self.sha256,
            "usage": "confidence_support_only",
            "canCreateRallyBoundary": False,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class LoadedAnnotations:
    """Optional paired human rally intervals used after inference only."""

    requested: bool
    path: Path | None
    sha256: str | None
    rallies: tuple[GroundTruthRally, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "path": str(self.path) if self.path is not None else None,
            "sha256": self.sha256,
            "annotatedRallyCount": len(self.rallies),
            "usedForInference": False,
            "usage": "post_inference_evaluation_only",
        }


@dataclass(frozen=True, slots=True)
class RallySegmentationArtifacts:
    """Generated paths and high-level counts."""

    rallies_path: Path
    debug_video_path: Path
    evaluation_path: Path
    rally_count: int
    matched_rally_count: int | None
    missed_rally_count: int | None
    false_rally_count: int | None
    rejected_adjacent_burst_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "ralliesPath": str(self.rallies_path),
            "debugVideoPath": str(self.debug_video_path),
            "evaluationPath": str(self.evaluation_path),
            "rallyCount": self.rally_count,
            "matchedRallyCount": self.matched_rally_count,
            "missedRallyCount": self.missed_rally_count,
            "falseRallyCount": self.false_rally_count,
            "rejectedAdjacentBurstCount": self.rejected_adjacent_burst_count,
        }


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return cast(list[object], value)


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _integer(value: object, field: str) -> int:
    number = _number(value, field)
    if not number.is_integer():
        raise ValueError(f"{field} must be an integer")
    return int(number)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field)


def _read_root(path: Path, kind: str) -> dict[str, object]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), "root")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RallySegmentationInputError(f"unable to load {kind} {path}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RallySegmentationInputError(f"unable to hash {path}: {error}") from error
    return digest.hexdigest()


def _source_metadata(value: object) -> VideoMetadata:
    source = _object(value, "source")
    return VideoMetadata(
        filename=_string(source.get("filename"), "source.filename"),
        path=Path(_string(source.get("path"), "source.path")).expanduser().resolve(),
        width=_integer(source.get("width"), "source.width"),
        height=_integer(source.get("height"), "source.height"),
        fps=_number(source.get("fps"), "source.fps"),
        frame_count=_integer(source.get("frame_count"), "source.frame_count"),
        duration=_number(source.get("duration"), "source.duration"),
        codec=_optional_string(source.get("codec"), "source.codec"),
    )


def _validate_source(actual: VideoMetadata, recorded: VideoMetadata, kind: str) -> None:
    if recorded.path != actual.path:
        raise RallySegmentationInputError(f"{kind} belongs to a different source video path")
    if (
        recorded.width != actual.width
        or recorded.height != actual.height
        or recorded.frame_count != actual.frame_count
        or not math.isclose(recorded.fps, actual.fps, rel_tol=1e-6)
    ):
        raise RallySegmentationInputError(f"{kind} source metadata does not match the video")


def _image_point(value: object, field: str) -> ImagePoint | None:
    if value is None:
        return None
    point = _object(value, field)
    return ImagePoint(
        x_px=_number(point.get("x_px"), f"{field}.x_px"),
        y_px=_number(point.get("y_px"), f"{field}.y_px"),
    )


def load_ball_trajectory(path: Path, *, source: VideoMetadata) -> LoadedBallTrajectory:
    """Load the conservative frame-complete ball trajectory."""

    resolved = path.expanduser().resolve()
    root = _read_root(resolved, "ball trajectory")
    try:
        if root.get("schema_version") != BALL_TRACKING_SCHEMA_VERSION:
            raise ValueError("unsupported ball trajectory schema_version")
        if root.get("record_type") != "primary_match_ball_trajectory":
            raise ValueError("record_type must be primary_match_ball_trajectory")
        recorded_source = _source_metadata(root.get("source"))
        _validate_source(source, recorded_source, "ball trajectory")
        raw_frames = _array(root.get("frames"), "frames")
        if len(raw_frames) != source.frame_count:
            raise ValueError("ball trajectory must contain every source frame")
        frames: list[RallyBallFrame] = []
        for expected_frame, value in enumerate(raw_frames):
            field = f"frames[{expected_frame}]"
            raw = _object(value, field)
            frame_number = _integer(raw.get("frame_number"), f"{field}.frame_number")
            if frame_number != expected_frame:
                raise ValueError(f"{field}.frame_number must be {expected_frame}")
            timestamp = _number(raw.get("timestamp_s"), f"{field}.timestamp_s")
            if not math.isclose(timestamp, frame_number / source.fps, abs_tol=1e-6):
                raise ValueError(f"{field}.timestamp_s differs from the video time base")
            try:
                status = BallEvidenceStatus(_string(raw.get("status"), f"{field}.status"))
            except ValueError as error:
                raise ValueError(f"{field}.status is unsupported") from error
            point = _image_point(
                raw.get("smoothed_image_point_px"),
                f"{field}.smoothed_image_point_px",
            )
            if point is None:
                point = _image_point(raw.get("raw_image_point_px"), f"{field}.raw_image_point_px")
            if point is None:
                point = _image_point(
                    raw.get("interpolated_image_point_px"),
                    f"{field}.interpolated_image_point_px",
                )
            confidence_value = raw.get("confidence")
            confidence = (
                _number(confidence_value, f"{field}.confidence")
                if confidence_value is not None
                else None
            )
            if confidence is not None and not 0 <= confidence <= 1:
                raise ValueError(f"{field}.confidence must be in [0, 1]")
            if status is BallEvidenceStatus.UNKNOWN and point is not None:
                raise ValueError(f"{field} UNKNOWN status cannot contain a trajectory point")
            if status is not BallEvidenceStatus.UNKNOWN and point is None:
                raise ValueError(f"{field} known status must contain a trajectory point")
            frames.append(
                RallyBallFrame(
                    frame_number=frame_number,
                    timestamp_seconds=timestamp,
                    status=status,
                    segment_id=_optional_string(raw.get("segment_id"), f"{field}.segment_id"),
                    point=point,
                    confidence=confidence,
                )
            )
        return LoadedBallTrajectory(
            path=resolved,
            sha256=_sha256(resolved),
            source=recorded_source,
            created_at_utc=_string(root.get("created_at_utc"), "created_at_utc"),
            frames=tuple(frames),
            statistics=_object(root.get("statistics", {}), "statistics"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RallySegmentationInputError(f"invalid ball trajectory {resolved}: {error}") from error


def _court_point(value: object, field: str) -> tuple[float, float] | None:
    if value is None:
        return None
    point = _object(value, field)
    return (
        _number(point.get("x_m"), f"{field}.x_m"),
        _number(point.get("y_m"), f"{field}.y_m"),
    )


def load_player_reset_evidence(
    path: Path | None,
    *,
    source: VideoMetadata,
    settings: RallySegmentationSettings,
) -> LoadedPlayerResetEvidence:
    """Load compatible logical-player tracks and derive pre-frame reset scores."""

    if path is None:
        return LoadedPlayerResetEvidence(False, False, None, None, None, "not supplied")
    resolved = path.expanduser().resolve()
    root = _read_root(resolved, "player tracks")
    try:
        if root.get("schema_version") != 1 or root.get("record_type") != (
            "persistent_logical_player_tracks"
        ):
            raise ValueError("unsupported player tracks schema or record_type")
        recorded_source = _source_metadata(root.get("source"))
        _validate_source(source, recorded_source, "player tracks")
        layer = _object(root.get("logical_identity_layer"), "logical_identity_layer")
        roles = ("ME", "PARTNER", "OPPONENT_1", "OPPONENT_2")
        role_speeds: dict[str, list[float | None]] = {}
        for role in roles:
            records = _array(layer.get(role), f"logical_identity_layer.{role}")
            if len(records) != source.frame_count:
                raise ValueError(f"logical_identity_layer.{role} must contain every frame")
            speeds: list[float | None] = [None] * source.frame_count
            previous: tuple[int, tuple[float, float]] | None = None
            for expected_frame, value in enumerate(records):
                field = f"logical_identity_layer.{role}[{expected_frame}]"
                record = _object(value, field)
                if _integer(record.get("frame_number"), f"{field}.frame_number") != expected_frame:
                    raise ValueError(f"{field}.frame_number is out of sequence")
                ground = record.get("ground_contact")
                point = None
                if ground is not None:
                    ground_object = _object(ground, f"{field}.ground_contact")
                    point = _court_point(
                        ground_object.get("court_point"),
                        f"{field}.ground_contact.court_point",
                    )
                if point is not None and previous is not None:
                    previous_frame, previous_point = previous
                    elapsed = (expected_frame - previous_frame) / source.fps
                    if elapsed <= 2 / source.fps:
                        speeds[expected_frame] = (
                            math.hypot(
                                point[0] - previous_point[0],
                                point[1] - previous_point[1],
                            )
                            / elapsed
                        )
                if point is not None:
                    previous = (expected_frame, point)
            role_speeds[role] = speeds
        window_frames = max(2, round(settings.player_reset_window_seconds * source.fps))
        scores: list[float | None] = []
        for frame_number in range(source.frame_count):
            start = max(0, frame_number - window_frames)
            role_medians = tuple(
                statistics.median(values)
                for role in roles
                if (
                    values := tuple(
                        value
                        for value in role_speeds[role][start : frame_number + 1]
                        if value is not None
                    )
                )
            )
            if len(role_medians) < 3:
                scores.append(None)
            else:
                slow = sum(
                    value <= settings.player_reset_maximum_speed_mps for value in role_medians
                )
                scores.append(slow / len(role_medians))
        return LoadedPlayerResetEvidence(
            True,
            True,
            resolved,
            _sha256(resolved),
            tuple(scores),
            None,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RallySegmentationInputError(f"invalid player tracks {resolved}: {error}") from error


def load_audio_evidence(
    path: Path | None,
    *,
    source: VideoMetadata,
) -> LoadedAudioEvidence:
    """Load generic transients and map canonical media time onto video-relative time."""

    if path is None:
        return LoadedAudioEvidence(False, False, None, None, (), "not supplied")
    resolved = path.expanduser().resolve()
    root = _read_root(resolved, "audio events")
    try:
        if root.get("schemaVersion") != 1 or root.get("recordType") != (
            "audio_analysis_observations"
        ):
            raise ValueError("unsupported audio events schema or recordType")
        source_media = _object(root.get("sourceMedia"), "sourceMedia")
        source_path = Path(_string(source_media.get("path"), "sourceMedia.path"))
        if source_path.expanduser().resolve() != source.path:
            raise ValueError("audio events belong to a different source video")
        for key, expected in (
            ("width", source.width),
            ("height", source.height),
            ("frame_count", source.frame_count),
        ):
            if _integer(source_media.get(key), f"sourceMedia.{key}") != expected:
                raise ValueError(f"sourceMedia.{key} does not match the video")
        if not math.isclose(
            _number(source_media.get("fps"), "sourceMedia.fps"),
            source.fps,
            rel_tol=1e-6,
        ):
            raise ValueError("sourceMedia.fps does not match the video")
        available = root.get("audioAnalysisAvailable")
        if not isinstance(available, bool):
            raise ValueError("audioAnalysisAvailable must be boolean")
        video_start = _number(source_media.get("videoStartTime", 0.0), "videoStartTime")
        transients: list[AudioTransientEvidence] = []
        raw_candidates = _array(
            root.get("audioEventCandidates", []),
            "audioEventCandidates",
        )
        for index, value in enumerate(raw_candidates):
            field = f"audioEventCandidates[{index}]"
            candidate = _object(value, field)
            if (
                candidate.get("candidateType") != "TRANSIENT"
                or candidate.get("semanticClassification") is not None
                or candidate.get("source") != "AUDIO"
            ):
                raise ValueError(f"{field} must be a generic non-semantic audio transient")
            confidence = _number(candidate.get("confidence"), f"{field}.confidence")
            if not 0 <= confidence <= 1:
                raise ValueError(f"{field}.confidence must be in [0, 1]")
            video_time = (
                _number(
                    candidate.get("mediaTimestampSeconds"),
                    f"{field}.mediaTimestampSeconds",
                )
                - video_start
            )
            if -1 / source.fps <= video_time <= source.duration + 1 / source.fps:
                transients.append(
                    AudioTransientEvidence(
                        candidate_id=_string(candidate.get("id"), f"{field}.id"),
                        video_timestamp_seconds=max(0.0, video_time),
                        confidence=confidence,
                    )
                )
        return LoadedAudioEvidence(
            True,
            available,
            resolved,
            _sha256(resolved),
            tuple(transients) if available else (),
            None if available else "audio analysis unavailable for this source",
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RallySegmentationInputError(f"invalid audio events {resolved}: {error}") from error


def load_ground_truth_annotations(
    path: Path | None,
    *,
    source: VideoMetadata,
    source_sha256: str | None,
) -> LoadedAnnotations:
    """Pair human rally boundaries without exposing them to inference."""

    if path is None:
        return LoadedAnnotations(False, None, None, ())
    resolved = path.expanduser().resolve()
    root = _read_root(resolved, "match annotations")
    try:
        if (
            root.get("annotationVersion") != MATCH_ANNOTATION_VERSION
            or root.get("recordType") != MATCH_ANNOTATION_RECORD_TYPE
        ):
            raise ValueError("unsupported annotationVersion or recordType")
        video = _object(root.get("video"), "video")
        video_path = Path(_string(video.get("path"), "video.path")).expanduser().resolve()
        if video_path != source.path:
            raise ValueError("annotations belong to a different source video")
        for key, expected in (
            ("width", source.width),
            ("height", source.height),
            ("frame_count", source.frame_count),
        ):
            if _integer(video.get(key), f"video.{key}") != expected:
                raise ValueError(f"video.{key} does not match the source")
        if not math.isclose(_number(video.get("fps"), "video.fps"), source.fps, rel_tol=1e-6):
            raise ValueError("video.fps does not match the source")
        if source_sha256 is not None and video.get("contentSha256") != source_sha256:
            raise ValueError("annotations belong to different source-media bytes")
        active: tuple[str, int] | None = None
        rallies: list[GroundTruthRally] = []
        events = _array(root.get("events"), "events")
        previous_frame = -1
        for index, value in enumerate(events):
            field = f"events[{index}]"
            event = _object(value, field)
            frame = _integer(event.get("frame"), f"{field}.frame")
            if frame < previous_frame:
                raise ValueError("annotation events must be chronological")
            previous_frame = frame
            event_type = _string(event.get("type"), f"{field}.type")
            event_id = _string(event.get("id"), f"{field}.id")
            if event_type == "RALLY_START":
                if active is not None:
                    raise ValueError("RALLY_START occurs before the prior RALLY_END")
                active = (event_id, frame)
            elif event_type == "RALLY_END":
                if active is None:
                    raise ValueError("RALLY_END has no preceding RALLY_START")
                start_event_id, start_frame = active
                if frame < start_frame:
                    raise ValueError("RALLY_END precedes RALLY_START")
                rallies.append(
                    GroundTruthRally(
                        rally_id=f"annotated-rally-{len(rallies) + 1:05d}",
                        start_event_id=start_event_id,
                        end_event_id=event_id,
                        start_frame=start_frame,
                        end_frame=frame,
                        start_timestamp_seconds=start_frame / source.fps,
                        end_timestamp_seconds=frame / source.fps,
                    )
                )
                active = None
        if active is not None:
            raise ValueError("final RALLY_START has no RALLY_END")
        if not rallies:
            raise ValueError("annotations contain no complete rally boundary pairs")
        return LoadedAnnotations(True, resolved, _sha256(resolved), tuple(rallies))
    except (KeyError, TypeError, ValueError) as error:
        raise RallySegmentationInputError(f"invalid annotations {resolved}: {error}") from error


def _prepare_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise OutputWriteError(str(output), reason="path is not a directory")
    for name in (RALLIES_NAME, RALLY_DEBUG_NAME, RALLY_EVALUATION_NAME):
        artifact = output / name
        if artifact.exists():
            raise OutputWriteError(str(artifact), reason="rally-segmentation output already exists")
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OutputWriteError(str(output), reason=str(error)) from error
    return output


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except (OSError, TypeError, ValueError) as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise OutputWriteError(str(path), reason=str(error)) from error


def _open_writer(path: Path, source: VideoMetadata) -> CompressedVideoWriter:
    return CompressedVideoWriter(
        path,
        fps=source.fps,
        dimensions=(source.width, source.height),
    )


def _index_intervals(
    intervals: tuple[IntervalT, ...],
    *,
    frame_count: int,
) -> list[IntervalT | None]:
    indexed: list[IntervalT | None] = [None] * frame_count
    for item in intervals:
        for frame_number in range(item.start_frame, item.end_frame + 1):
            indexed[frame_number] = item
    return indexed


def _write_debug_video(
    video_path: Path,
    output_path: Path,
    *,
    source: VideoMetadata,
    ball_frames: tuple[RallyBallFrame, ...],
    speeds: tuple[float | None, ...],
    motion_supported: tuple[bool, ...],
    predictions: tuple[RallyPrediction, ...],
    annotations: tuple[GroundTruthRally, ...],
) -> None:
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    writer = _open_writer(temporary, source)
    predicted_by_frame = _index_intervals(predictions, frame_count=source.frame_count)
    annotated_by_frame = _index_intervals(annotations, frame_count=source.frame_count)
    predicted_starts = {item.start_frame for item in predictions}
    predicted_ends = {item.end_frame for item in predictions}
    logger = logging.getLogger("pickleball_vision.rally_segmentation")
    processed = 0
    try:
        for decoded in iter_video_frames(video_path):
            frame_number = decoded.frame_index
            writer.write(
                render_rally_segmentation_frame(
                    decoded.image,
                    frame_number=frame_number,
                    ball=ball_frames[frame_number],
                    speed_diagonals_per_second=speeds[frame_number],
                    motion_supported=motion_supported[frame_number],
                    predicted_rally=predicted_by_frame[frame_number],
                    annotated_rally=annotated_by_frame[frame_number],
                    predicted_start=frame_number in predicted_starts,
                    predicted_end=frame_number in predicted_ends,
                )
            )
            processed += 1
            if processed % 1000 == 0:
                logger.info(
                    "rally_debug_progress",
                    extra={"context": {"processed_frames": processed}},
                )
    except Exception:
        writer.abort()
        temporary.unlink(missing_ok=True)
        raise
    writer.release()
    if processed != source.frame_count:
        temporary.unlink(missing_ok=True)
        raise RallySegmentationInputError(
            f"decoded {processed} frames but trajectory contains {source.frame_count}"
        )
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise OutputWriteError(str(output_path), reason="OpenCV wrote an empty debug video")
    temporary.replace(output_path)


def segment_rallies_in_video(
    video_path: Path,
    *,
    ball_tracks_path: Path,
    output_dir: Path,
    settings: RallySegmentationSettings,
    player_tracks_path: Path | None = None,
    audio_events_path: Path | None = None,
    annotations_path: Path | None = None,
    annotations_complete: bool = False,
    evaluation_partition: str = "validation",
) -> RallySegmentationArtifacts:
    """Run inference first, then optional evaluation and source-space rendering."""

    source = inspect_video(video_path)
    ball = load_ball_trajectory(ball_tracks_path, source=source)
    players = load_player_reset_evidence(
        player_tracks_path,
        source=source,
        settings=settings,
    )
    audio = load_audio_evidence(audio_events_path, source=source)
    source_hash = _sha256(source.path) if annotations_path is not None else None
    annotations = load_ground_truth_annotations(
        annotations_path,
        source=source,
        source_sha256=source_hash,
    )
    result = segment_rallies(
        ball.frames,
        fps=source.fps,
        frame_width_px=source.width,
        frame_height_px=source.height,
        settings=settings,
        player_reset_scores=players.scores,
        audio_transients=audio.transients,
    )
    evaluation = (
        evaluate_rallies(
            result.rallies,
            annotations.rallies,
            fps=source.fps,
            settings=settings,
            annotations_complete=annotations_complete,
            evaluation_partition=evaluation_partition,
        )
        if annotations.requested
        else unavailable_evaluation(evaluation_partition=evaluation_partition)
    )
    output = _prepare_output(output_dir)
    rallies_path = output / RALLIES_NAME
    debug_path = output / RALLY_DEBUG_NAME
    evaluation_path = output / RALLY_EVALUATION_NAME
    if debug_path == source.path:
        raise OutputWriteError(str(debug_path), reason="output would overwrite source video")
    _write_debug_video(
        source.path,
        debug_path,
        source=source,
        ball_frames=ball.frames,
        speeds=result.speeds_diagonals_per_second,
        motion_supported=result.motion_supported,
        predictions=result.rallies,
        annotations=annotations.rallies,
    )
    created_at_utc = datetime.now(UTC).isoformat()
    inputs = {
        "ballTrajectory": {
            "path": str(ball.path),
            "sha256": ball.sha256,
            "schemaVersion": BALL_TRACKING_SCHEMA_VERSION,
            "createdAtUtc": ball.created_at_utc,
            "statistics": ball.statistics,
        },
        "playerTracks": players.as_dict(),
        "audioEvents": audio.as_dict(),
        "groundTruthAnnotations": annotations.as_dict(),
    }
    durations = tuple(
        (item.end_frame - item.start_frame + 1) / source.fps for item in result.rallies
    )
    statistics_payload = {
        "rallyCount": len(result.rallies),
        "serveLikeCandidateCount": result.serve_candidate_count,
        "totalPredictedRallySeconds": sum(durations),
        "meanRallyDurationSeconds": statistics.fmean(durations) if durations else None,
        "minimumRallyDurationSeconds": min(durations, default=None),
        "maximumRallyDurationSeconds": max(durations, default=None),
        "rejectedAdjacentBurstCount": len(result.rejected_adjacent_bursts),
    }
    _write_json(
        rallies_path,
        {
            "schemaVersion": RALLY_SEGMENTATION_SCHEMA_VERSION,
            "recordType": "automatic_rally_segments",
            "createdAtUtc": created_at_utc,
            "source": source.as_dict(),
            "timeline": {
                "timestampUnit": "seconds",
                "frameIndexing": "zero_based",
                "mapping": "timestamp = frame / fps",
            },
            "inputs": inputs,
            "configuration": settings.as_dict(),
            "contracts": {
                "structuredSignalsOnly": True,
                "annotationsUsedForInference": False,
                "audioCanCreateBoundary": False,
                "audioUsage": "optional_confidence_support_only",
                "bounceDetectionImplemented": False,
                "contactDetectionImplemented": False,
            },
            "statistics": statistics_payload,
            "rallies": [item.as_dict() for item in result.rallies],
            "rejectedCandidates": [item.as_dict() for item in result.rejected_adjacent_bursts],
        },
    )
    _write_json(
        evaluation_path,
        {
            "schemaVersion": RALLY_SEGMENTATION_SCHEMA_VERSION,
            "recordType": "rally_segmentation_evaluation",
            "createdAtUtc": created_at_utc,
            "source": source.as_dict(),
            "inputs": inputs,
            "configuration": settings.as_dict()["evaluation"],
            **evaluation,
            "artifacts": {
                "rallies": str(rallies_path),
                "debugVideo": str(debug_path),
                "evaluation": str(evaluation_path),
            },
        },
    )
    return RallySegmentationArtifacts(
        rallies_path=rallies_path,
        debug_video_path=debug_path,
        evaluation_path=evaluation_path,
        rally_count=len(result.rallies),
        matched_rally_count=cast(int | None, evaluation.get("matchedRallyCount")),
        missed_rally_count=cast(int | None, evaluation.get("missedRallyCount")),
        false_rally_count=cast(int | None, evaluation.get("falseRallyCount")),
        rejected_adjacent_burst_count=len(result.rejected_adjacent_bursts),
    )
