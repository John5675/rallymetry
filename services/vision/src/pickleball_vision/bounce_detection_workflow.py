"""Artifact loading, A/V timing, persistence, evaluation, and bounce debug video."""

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
from typing import cast

from pickleball_vision.bounce_detection import (
    BOUNCE_DETECTION_SCHEMA_VERSION,
    BounceAudioTransient,
    BounceCandidate,
    BounceDetectionResult,
    BounceEvidenceMode,
    BounceRallyInterval,
    detect_bounce_candidates,
)
from pickleball_vision.bounce_detection_render import render_bounce_detection_frame
from pickleball_vision.bounce_evaluation import (
    GroundTruthBounce,
    ReviewedInterval,
    evaluate_bounces,
    unavailable_bounce_evaluation,
)
from pickleball_vision.calibration import CourtCalibration, load_calibration
from pickleball_vision.config import BounceDetectionSettings
from pickleball_vision.errors import BounceDetectionInputError, OutputWriteError
from pickleball_vision.match_annotation import (
    MATCH_ANNOTATION_RECORD_TYPE,
    MATCH_ANNOTATION_VERSION,
)
from pickleball_vision.media import MediaMetadata, MediaTimeline, inspect_media
from pickleball_vision.rally_segmentation_workflow import (
    LoadedBallTrajectory,
    load_ball_trajectory,
)
from pickleball_vision.video import VideoMetadata, iter_video_frames
from pickleball_vision.video_output import CompressedVideoWriter

BOUNCES_NAME = "bounces.json"
BOUNCE_DEBUG_NAME = "bounce-debug.mp4"
BOUNCE_EVALUATION_NAME = "bounce-evaluation.json"


@dataclass(frozen=True, slots=True)
class LoadedBounceAudio:
    """Optional generic transients remapped with the run's configured A/V offset."""

    requested: bool
    available: bool
    path: Path | None
    sha256: str | None
    transients: tuple[BounceAudioTransient, ...]
    stored_offset_ms: float | None
    applied_offset_ms: float
    reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "audioAnalysisAvailable": self.available,
            "path": str(self.path) if self.path is not None else None,
            "sha256": self.sha256,
            "transientCandidateCount": len(self.transients),
            "storedAudioVideoOffsetMs": self.stored_offset_ms,
            "appliedAudioVideoOffsetMs": self.applied_offset_ms,
            "timestampsRemappedFromAnalysisTime": self.available,
            "usage": "confidence_support_only",
            "canCreateBounce": False,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class LoadedBounceRallies:
    """Optional source-compatible rally intervals used only as confidence support."""

    requested: bool
    path: Path | None
    sha256: str | None
    intervals: tuple[BounceRallyInterval, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "path": str(self.path) if self.path is not None else None,
            "sha256": self.sha256,
            "rallyCount": len(self.intervals),
            "usage": "visual_confidence_support_only",
            "canCreateBounce": False,
        }


@dataclass(frozen=True, slots=True)
class LoadedBounceAnnotations:
    """Human bounce ground truth isolated from inference."""

    requested: bool
    path: Path | None
    sha256: str | None
    bounces: tuple[GroundTruthBounce, ...]
    reviewed_intervals: tuple[ReviewedInterval, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "path": str(self.path) if self.path is not None else None,
            "sha256": self.sha256,
            "annotatedBounceCount": len(self.bounces),
            "reviewedRallyIntervalCount": len(self.reviewed_intervals),
            "usedForInference": False,
            "usage": "post_inference_evaluation_only",
        }


@dataclass(frozen=True, slots=True)
class BounceDetectionArtifacts:
    """Generated output paths and high-level candidate counts."""

    bounces_path: Path
    debug_video_path: Path
    evaluation_path: Path
    visual_candidate_count: int
    accepted_bounce_count: int
    fused_candidate_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "bouncesPath": str(self.bounces_path),
            "debugVideoPath": str(self.debug_video_path),
            "evaluationPath": str(self.evaluation_path),
            "visualCandidateCount": self.visual_candidate_count,
            "acceptedBounceCount": self.accepted_bounce_count,
            "fusedCandidateCount": self.fused_candidate_count,
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


def _read_root(path: Path, kind: str) -> dict[str, object]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), "root")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise BounceDetectionInputError(f"unable to load {kind} {path}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise BounceDetectionInputError(f"unable to hash {path}: {error}") from error
    return digest.hexdigest()


def _validate_video_source(value: object, source: VideoMetadata, *, field: str) -> None:
    raw = _object(value, field)
    path = Path(_string(raw.get("path"), f"{field}.path")).expanduser().resolve()
    if path != source.path:
        raise ValueError(f"{field} belongs to a different source video path")
    for key, expected in (
        ("width", source.width),
        ("height", source.height),
        ("frame_count", source.frame_count),
    ):
        if _integer(raw.get(key), f"{field}.{key}") != expected:
            raise ValueError(f"{field}.{key} does not match the video")
    if not math.isclose(_number(raw.get("fps"), f"{field}.fps"), source.fps, rel_tol=1e-6):
        raise ValueError(f"{field}.fps does not match the video")


def _validate_calibration(calibration: CourtCalibration, source: VideoMetadata) -> None:
    calibration_path = calibration.source.video_path.expanduser().resolve()
    if calibration_path != source.path:
        raise BounceDetectionInputError("calibration belongs to a different source video")
    if (
        calibration.source.frame_width_px != source.width
        or calibration.source.frame_height_px != source.height
        or not math.isclose(calibration.source.fps, source.fps, rel_tol=5e-3)
    ):
        raise BounceDetectionInputError("calibration source metadata does not match the video")


def load_bounce_audio(
    path: Path | None,
    *,
    media: MediaMetadata,
    timeline: MediaTimeline,
) -> LoadedBounceAudio:
    """Load generic transients and reapply the current configurable A/V offset."""

    if path is None:
        return LoadedBounceAudio(
            False,
            False,
            None,
            None,
            (),
            None,
            timeline.audio_video_offset_ms,
            "not supplied",
        )
    resolved = path.expanduser().resolve()
    root = _read_root(resolved, "audio events")
    try:
        if root.get("schemaVersion") != 1 or root.get("recordType") != (
            "audio_analysis_observations"
        ):
            raise ValueError("unsupported audio-events schema or recordType")
        _validate_video_source(root.get("sourceMedia"), media.video, field="sourceMedia")
        available = root.get("audioAnalysisAvailable")
        if not isinstance(available, bool):
            raise ValueError("audioAnalysisAvailable must be boolean")
        configuration = _object(root.get("configuration", {}), "configuration")
        stored_offset = _number(
            configuration.get("audioVideoOffsetMs", 0.0),
            "configuration.audioVideoOffsetMs",
        )
        timeline_payload = _object(root.get("timeline", {}), "timeline")
        audio_start = _number(
            timeline_payload.get("audioStartTimeSeconds", 0.0) or 0.0,
            "timeline.audioStartTimeSeconds",
        )
        video_start = media.video_start_time_seconds or 0.0
        transients: list[BounceAudioTransient] = []
        for index, item in enumerate(
            _array(root.get("audioEventCandidates", []), "audioEventCandidates")
        ):
            field = f"audioEventCandidates[{index}]"
            candidate = _object(item, field)
            if (
                candidate.get("candidateType") != "TRANSIENT"
                or candidate.get("semanticClassification") is not None
                or candidate.get("source") != "AUDIO"
            ):
                raise ValueError(f"{field} must be a generic non-semantic audio transient")
            confidence = _number(candidate.get("confidence"), f"{field}.confidence")
            if not 0 <= confidence <= 1:
                raise ValueError(f"{field}.confidence must be in [0, 1]")
            analysis_time = _number(
                candidate.get("analysisTimestampSeconds"),
                f"{field}.analysisTimestampSeconds",
            )
            video_time = (
                audio_start + analysis_time + timeline.audio_video_offset_ms / 1000.0 - video_start
            )
            if -1 / media.video.fps <= video_time <= media.video.duration + (1 / media.video.fps):
                transients.append(
                    BounceAudioTransient(
                        candidate_id=_string(candidate.get("id"), f"{field}.id"),
                        video_timestamp_seconds=max(0.0, video_time),
                        confidence=confidence,
                    )
                )
        return LoadedBounceAudio(
            True,
            available,
            resolved,
            _sha256(resolved),
            tuple(transients) if available else (),
            stored_offset,
            timeline.audio_video_offset_ms,
            None if available else "audio analysis unavailable for this source",
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BounceDetectionInputError(f"invalid audio events {resolved}: {error}") from error


def load_bounce_rallies(
    path: Path | None,
    *,
    source: VideoMetadata,
) -> LoadedBounceRallies:
    """Load optional automatic rally intervals as non-creating sequence support."""

    if path is None:
        return LoadedBounceRallies(False, None, None, ())
    resolved = path.expanduser().resolve()
    root = _read_root(resolved, "rallies")
    try:
        if root.get("schemaVersion") != 1 or root.get("recordType") != ("automatic_rally_segments"):
            raise ValueError("unsupported rallies schema or recordType")
        _validate_video_source(root.get("source"), source, field="source")
        intervals: list[BounceRallyInterval] = []
        for index, item in enumerate(_array(root.get("rallies"), "rallies")):
            field = f"rallies[{index}]"
            raw = _object(item, field)
            start = _integer(raw.get("startFrame"), f"{field}.startFrame")
            end = _integer(raw.get("endFrame"), f"{field}.endFrame")
            confidence = _number(raw.get("confidence"), f"{field}.confidence")
            if start < 0 or end < start or end >= source.frame_count:
                raise ValueError(f"{field} frame interval is invalid")
            if not 0 <= confidence <= 1:
                raise ValueError(f"{field}.confidence must be in [0, 1]")
            intervals.append(
                BounceRallyInterval(
                    rally_id=_string(raw.get("rallyId"), f"{field}.rallyId"),
                    start_frame=start,
                    end_frame=end,
                    confidence=confidence,
                )
            )
        return LoadedBounceRallies(True, resolved, _sha256(resolved), tuple(intervals))
    except (KeyError, TypeError, ValueError) as error:
        raise BounceDetectionInputError(f"invalid rallies {resolved}: {error}") from error


def load_bounce_annotations(
    path: Path | None,
    *,
    source: VideoMetadata,
    source_sha256: str | None,
) -> LoadedBounceAnnotations:
    """Load human BOUNCE events and reviewed rally windows for evaluation only."""

    if path is None:
        return LoadedBounceAnnotations(False, None, None, (), ())
    resolved = path.expanduser().resolve()
    root = _read_root(resolved, "match annotations")
    try:
        if (
            root.get("annotationVersion") != MATCH_ANNOTATION_VERSION
            or root.get("recordType") != MATCH_ANNOTATION_RECORD_TYPE
        ):
            raise ValueError("unsupported annotationVersion or recordType")
        video = _object(root.get("video"), "video")
        _validate_video_source(video, source, field="video")
        if source_sha256 is not None and video.get("contentSha256") != source_sha256:
            raise ValueError("annotations belong to different source-media bytes")
        bounces: list[GroundTruthBounce] = []
        intervals: list[ReviewedInterval] = []
        active: tuple[str, int] | None = None
        previous_frame = -1
        for index, item in enumerate(_array(root.get("events"), "events")):
            field = f"events[{index}]"
            event = _object(item, field)
            frame = _integer(event.get("frame"), f"{field}.frame")
            if frame < previous_frame:
                raise ValueError("annotation events must be chronological")
            if frame < 0 or frame >= source.frame_count:
                raise ValueError(f"{field}.frame is outside the source video")
            previous_frame = frame
            event_id = _string(event.get("id"), f"{field}.id")
            event_type = _string(event.get("type"), f"{field}.type")
            if event_type == "BOUNCE":
                bounces.append(GroundTruthBounce(event_id, frame, frame / source.fps))
            elif event_type == "RALLY_START":
                if active is not None:
                    raise ValueError("RALLY_START occurs before the prior RALLY_END")
                active = (event_id, frame)
            elif event_type == "RALLY_END":
                if active is None:
                    raise ValueError("RALLY_END has no preceding RALLY_START")
                start_id, start_frame = active
                intervals.append(
                    ReviewedInterval(
                        interval_id=f"{start_id}:{event_id}",
                        start_frame=start_frame,
                        end_frame=frame,
                    )
                )
                active = None
        if active is not None:
            raise ValueError("final RALLY_START has no RALLY_END")
        return LoadedBounceAnnotations(
            True,
            resolved,
            _sha256(resolved),
            tuple(bounces),
            tuple(intervals),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BounceDetectionInputError(f"invalid annotations {resolved}: {error}") from error


def _prepare_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise OutputWriteError(str(output), reason="path is not a directory")
    for name in (BOUNCES_NAME, BOUNCE_DEBUG_NAME, BOUNCE_EVALUATION_NAME):
        artifact = output / name
        if artifact.exists():
            raise OutputWriteError(str(artifact), reason="bounce-detection output already exists")
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


def _write_debug_video(
    video_path: Path,
    output_path: Path,
    *,
    source: VideoMetadata,
    candidates: tuple[BounceCandidate, ...],
    audio_available: bool,
) -> None:
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    writer = _open_writer(temporary, source)
    by_frame = {item.frame: item for item in candidates}
    recent_frames = max(1, round(0.15 * source.fps))
    last: BounceCandidate | None = None
    logger = logging.getLogger("pickleball_vision.bounce_detection")
    processed = 0
    try:
        for decoded in iter_video_frames(video_path):
            candidate = by_frame.get(decoded.frame_index)
            if candidate is not None:
                last = candidate
            recent = (
                last
                if last is not None and 0 < decoded.frame_index - last.frame <= recent_frames
                else None
            )
            writer.write(
                render_bounce_detection_frame(
                    decoded.image,
                    frame_number=decoded.frame_index,
                    candidate=candidate,
                    recent_candidate=recent,
                    audio_available=audio_available,
                )
            )
            processed += 1
            if processed % 1000 == 0:
                logger.info(
                    "bounce_debug_progress",
                    extra={"context": {"processed_frames": processed}},
                )
    except Exception:
        writer.abort()
        temporary.unlink(missing_ok=True)
        raise
    writer.release()
    if processed != source.frame_count:
        temporary.unlink(missing_ok=True)
        raise BounceDetectionInputError(
            f"decoded {processed} frames but trajectory contains {source.frame_count}"
        )
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise OutputWriteError(str(output_path), reason="OpenCV wrote an empty debug video")
    temporary.replace(output_path)


def _statistics(result: BounceDetectionResult) -> dict[str, object]:
    accepted = tuple(item for item in result.candidates if item.accepted_fused)
    modes = {
        mode.value: sum(item.evidence_mode is mode for item in result.candidates)
        for mode in BounceEvidenceMode
    }
    fused = tuple(item for item in result.candidates if item.matched_audio_event_id is not None)
    return {
        "rawVisualCandidateCount": result.raw_visual_candidate_count,
        "visualCandidateCountAfterTemporalSuppression": len(result.candidates),
        "suppressedVisualCandidateCount": result.suppressed_visual_candidate_count,
        "acceptedBounceCount": len(accepted),
        "lowConfidenceCandidateCount": len(result.candidates) - len(accepted),
        "matchedAudioCandidateCount": result.matched_audio_candidate_count,
        "courtPositionCount": sum(item.court_position is not None for item in result.candidates),
        "evidenceModeCounts": modes,
        "meanVisualConfidence": (
            statistics.fmean(item.visual_confidence for item in result.candidates)
            if result.candidates
            else None
        ),
        "meanFusedConfidence": (
            statistics.fmean(item.fused_confidence for item in result.candidates)
            if result.candidates
            else None
        ),
        "audioMatchedFraction": (len(fused) / len(result.candidates) if result.candidates else 0.0),
    }


def detect_bounces_in_video(
    video_path: Path,
    *,
    ball_tracks_path: Path,
    calibration_path: Path,
    output_dir: Path,
    settings: BounceDetectionSettings,
    timeline: MediaTimeline,
    audio_events_path: Path | None = None,
    rallies_path: Path | None = None,
    annotations_path: Path | None = None,
    annotations_complete: bool = False,
    evaluation_partition: str = "validation",
) -> BounceDetectionArtifacts:
    """Run visual inference first, then optional audio fusion and evaluation."""

    media = inspect_media(video_path, timeline=timeline)
    source = media.video
    ball: LoadedBallTrajectory = load_ball_trajectory(ball_tracks_path, source=source)
    calibration = load_calibration(calibration_path)
    _validate_calibration(calibration, source)
    audio = load_bounce_audio(audio_events_path, media=media, timeline=timeline)
    rallies = load_bounce_rallies(rallies_path, source=source)
    source_hash = _sha256(source.path) if annotations_path is not None else None
    annotations = load_bounce_annotations(
        annotations_path,
        source=source,
        source_sha256=source_hash,
    )
    result = detect_bounce_candidates(
        ball.frames,
        fps=source.fps,
        frame_width_px=source.width,
        frame_height_px=source.height,
        calibration=calibration,
        settings=settings,
        audio_transients=audio.transients,
        rallies=rallies.intervals,
        video_start_time_seconds=media.video_start_time_seconds or 0.0,
        fusion_tolerance_ms=timeline.fusion_tolerance_ms,
    )
    evaluation = (
        evaluate_bounces(
            result.candidates,
            annotations.bounces,
            fps=source.fps,
            settings=settings,
            annotations_complete=annotations_complete,
            reviewed_intervals=annotations.reviewed_intervals,
            evaluation_partition=evaluation_partition,
        )
        if annotations.bounces
        else unavailable_bounce_evaluation(evaluation_partition=evaluation_partition)
    )
    output = _prepare_output(output_dir)
    bounces_path = output / BOUNCES_NAME
    debug_path = output / BOUNCE_DEBUG_NAME
    evaluation_path = output / BOUNCE_EVALUATION_NAME
    if debug_path == source.path:
        raise OutputWriteError(str(debug_path), reason="output would overwrite source video")
    _write_debug_video(
        source.path,
        debug_path,
        source=source,
        candidates=result.candidates,
        audio_available=audio.available,
    )
    created_at = datetime.now(UTC).isoformat()
    inputs = {
        "ballTrajectory": {
            "path": str(ball.path),
            "sha256": ball.sha256,
            "createdAtUtc": ball.created_at_utc,
            "statistics": ball.statistics,
            "mutated": False,
        },
        "courtCalibration": {
            "path": str(calibration_path.expanduser().resolve()),
            "sha256": _sha256(calibration_path.expanduser().resolve()),
            "schemaVersion": calibration.schema_version,
            "homographyUsage": "candidate court projection only after visual plane contact",
        },
        "rallies": rallies.as_dict(),
        "audioEvents": audio.as_dict(),
        "groundTruthAnnotations": annotations.as_dict(),
    }
    configuration = {
        **settings.as_dict(),
        "audioVideoOffsetMs": timeline.audio_video_offset_ms,
        "fusionToleranceMs": timeline.fusion_tolerance_ms,
    }
    _write_json(
        bounces_path,
        {
            "schemaVersion": BOUNCE_DETECTION_SCHEMA_VERSION,
            "recordType": "multimodal_bounce_candidates",
            "createdAtUtc": created_at,
            "source": media.as_dict(),
            "timeline": {
                "timestampUnit": "seconds",
                "frameIndexing": "zero_based",
                "videoMapping": "timestamp = frame / fps",
                "canonicalMediaMapping": ("mediaTimestamp = (videoStartTime or 0) + frame / fps"),
                "audioVideoOffsetMs": timeline.audio_video_offset_ms,
                "fusionToleranceMs": timeline.fusion_tolerance_ms,
            },
            "inputs": inputs,
            "configuration": configuration,
            "contracts": {
                "visualEvidenceRequired": True,
                "audioCanCreateBounce": False,
                "neighboringCourtSoundsMayBePresent": True,
                "homographyAppliedOnlyAfterPlaneContactPlausibility": True,
                "airborneBallProjectedThroughHomography": False,
                "true3dPositionInferred": False,
                "lineCallingImplemented": False,
                "rawInputsMutated": False,
            },
            "statistics": _statistics(result),
            "bounceCandidates": [item.as_dict() for item in result.candidates],
        },
    )
    _write_json(
        evaluation_path,
        {
            "schemaVersion": BOUNCE_DETECTION_SCHEMA_VERSION,
            "recordType": "multimodal_bounce_evaluation",
            "createdAtUtc": created_at,
            "source": media.as_dict(),
            "inputs": inputs,
            "configuration": {
                "acceptedConfidence": settings.accepted_confidence,
                "evaluationToleranceMs": settings.evaluation_tolerance_ms,
                "sparseEvaluationMarginSeconds": settings.sparse_evaluation_margin_seconds,
                "audioVideoOffsetMs": timeline.audio_video_offset_ms,
                "fusionToleranceMs": timeline.fusion_tolerance_ms,
            },
            **evaluation,
            "artifacts": {
                "bounces": str(bounces_path),
                "debugVideo": str(debug_path),
                "evaluation": str(evaluation_path),
            },
        },
    )
    return BounceDetectionArtifacts(
        bounces_path=bounces_path,
        debug_video_path=debug_path,
        evaluation_path=evaluation_path,
        visual_candidate_count=len(result.candidates),
        accepted_bounce_count=sum(item.accepted_fused for item in result.candidates),
        fused_candidate_count=sum(
            item.matched_audio_event_id is not None for item in result.candidates
        ),
    )
