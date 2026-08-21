"""Artifact loading, A/V fusion, persistence, evaluation, and contact debug video."""

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

from pickleball_vision.config import ContactDetectionSettings
from pickleball_vision.contact_detection import (
    CONTACT_DETECTION_SCHEMA_VERSION,
    ContactAudioTransient,
    ContactCandidate,
    ContactDetectionResult,
    ContactEvidenceMode,
    ContactPlayerObservation,
    ContactRallyInterval,
    PriorBounce,
    detect_contact_candidates,
)
from pickleball_vision.contact_detection_render import render_contact_detection_frame
from pickleball_vision.contact_evaluation import (
    ContactReviewedInterval,
    GroundTruthContact,
    evaluate_contacts,
    unavailable_contact_evaluation,
)
from pickleball_vision.court import ImagePoint
from pickleball_vision.errors import (
    ContactDetectionInputError,
    OutputWriteError,
    RallySegmentationInputError,
)
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

CONTACTS_NAME = "contacts.json"
CONTACT_DEBUG_NAME = "contact-debug.mp4"
CONTACT_EVALUATION_NAME = "contact-evaluation.json"
LOGICAL_ROLES = ("ME", "PARTNER", "OPPONENT_1", "OPPONENT_2")


@dataclass(frozen=True, slots=True)
class LoadedContactPlayerTracks:
    """Validated logical-player visual observations and provenance."""

    path: Path
    sha256: str
    observations_by_frame: tuple[tuple[ContactPlayerObservation, ...], ...]
    player_names: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "logicalRoles": list(LOGICAL_ROLES),
            "playerNames": self.player_names,
            "usage": "candidate_player_proximity_and_court_side_context",
            "assignsHitter": False,
            "mutated": False,
        }


@dataclass(frozen=True, slots=True)
class LoadedContactAudio:
    """Optional generic transients remapped with the run's configured A/V offset."""

    requested: bool
    available: bool
    path: Path | None
    sha256: str | None
    transients: tuple[ContactAudioTransient, ...]
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
            "canCreateContact": False,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class LoadedContactRallies:
    """Optional source-compatible rally intervals used only as sequence support."""

    requested: bool
    path: Path | None
    sha256: str | None
    intervals: tuple[ContactRallyInterval, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "path": str(self.path) if self.path is not None else None,
            "sha256": self.sha256,
            "rallyCount": len(self.intervals),
            "usage": "visual_confidence_support_only",
            "canCreateContact": False,
        }


@dataclass(frozen=True, slots=True)
class LoadedPriorBounces:
    """Optional accepted bounce state used for sequence support and exclusion."""

    requested: bool
    path: Path | None
    sha256: str | None
    bounces: tuple[PriorBounce, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "path": str(self.path) if self.path is not None else None,
            "sha256": self.sha256,
            "acceptedBounceCount": len(self.bounces),
            "usage": "event_sequence_context_and_coincident_bounce_exclusion",
            "canCreateContact": False,
        }


@dataclass(frozen=True, slots=True)
class LoadedContactAnnotations:
    """Human contact ground truth isolated from inference."""

    requested: bool
    path: Path | None
    sha256: str | None
    contacts: tuple[GroundTruthContact, ...]
    reviewed_intervals: tuple[ContactReviewedInterval, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "path": str(self.path) if self.path is not None else None,
            "sha256": self.sha256,
            "annotatedContactCount": len(self.contacts),
            "serveContactCount": sum(item.event_type == "SERVE_CONTACT" for item in self.contacts),
            "paddleContactCount": sum(
                item.event_type == "PADDLE_CONTACT" for item in self.contacts
            ),
            "reviewedRallyIntervalCount": len(self.reviewed_intervals),
            "usedForInference": False,
            "usage": "post_inference_evaluation_only",
        }


@dataclass(frozen=True, slots=True)
class ContactDetectionArtifacts:
    """Generated output paths and high-level candidate counts."""

    contacts_path: Path
    debug_video_path: Path
    evaluation_path: Path
    visual_candidate_count: int
    accepted_contact_count: int
    fused_candidate_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "contactsPath": str(self.contacts_path),
            "debugVideoPath": str(self.debug_video_path),
            "evaluationPath": str(self.evaluation_path),
            "visualCandidateCount": self.visual_candidate_count,
            "acceptedContactCount": self.accepted_contact_count,
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


def _optional_string(value: object, field: str) -> str | None:
    return None if value is None else _string(value, field)


def _read_root(path: Path, kind: str) -> dict[str, object]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), "root")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ContactDetectionInputError(f"unable to load {kind} {path}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ContactDetectionInputError(f"unable to hash {path}: {error}") from error
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
    if not math.isclose(_number(raw.get("fps"), f"{field}.fps"), source.fps, rel_tol=5e-3):
        raise ValueError(f"{field}.fps does not match the video")


def load_contact_player_tracks(
    path: Path,
    *,
    source: VideoMetadata,
) -> LoadedContactPlayerTracks:
    """Load frame-complete logical tracks and their linked person boxes."""

    resolved = path.expanduser().resolve()
    root = _read_root(resolved, "player tracks")
    try:
        if root.get("schema_version") != 1 or root.get("record_type") != (
            "persistent_logical_player_tracks"
        ):
            raise ValueError("unsupported player tracks schema or record_type")
        _validate_video_source(root.get("source"), source, field="source")
        raw_layer = _object(root.get("raw_tracker_layer"), "raw_tracker_layer")
        raw_observations: dict[str, tuple[int, tuple[float, float, float, float]]] = {}
        for index, value in enumerate(
            _array(raw_layer.get("observations"), "raw_tracker_layer.observations")
        ):
            field = f"raw_tracker_layer.observations[{index}]"
            item = _object(value, field)
            raw_observation_id = _string(item.get("observation_id"), f"{field}.observation_id")
            if raw_observation_id in raw_observations:
                raise ValueError(f"duplicate raw tracker observation ID {raw_observation_id}")
            frame = _integer(item.get("frame_number"), f"{field}.frame_number")
            if not 0 <= frame < source.frame_count:
                raise ValueError(f"{field}.frame_number is outside the source video")
            tracker_box_payload = _object(
                item.get("tracker_bounding_box"), f"{field}.tracker_bounding_box"
            )
            bounding_box = (
                _number(
                    tracker_box_payload.get("left_px"),
                    f"{field}.tracker_bounding_box.left_px",
                ),
                _number(
                    tracker_box_payload.get("top_px"),
                    f"{field}.tracker_bounding_box.top_px",
                ),
                _number(
                    tracker_box_payload.get("right_px"),
                    f"{field}.tracker_bounding_box.right_px",
                ),
                _number(
                    tracker_box_payload.get("bottom_px"),
                    f"{field}.tracker_bounding_box.bottom_px",
                ),
            )
            left, top, right, bottom = bounding_box
            if not (0 <= left < right <= source.width and 0 <= top < bottom <= source.height):
                raise ValueError(f"{field}.tracker_bounding_box is invalid")
            raw_observations[raw_observation_id] = (frame, bounding_box)
        raw_names = _object(root.get("player_names", {}), "player_names")
        player_names = {
            role: name
            for role in LOGICAL_ROLES
            if (name := raw_names.get(role)) is not None and isinstance(name, str) and name.strip()
        }
        layer = _object(root.get("logical_identity_layer"), "logical_identity_layer")
        by_frame: list[list[ContactPlayerObservation]] = [[] for _ in range(source.frame_count)]
        for role in LOGICAL_ROLES:
            records = _array(layer.get(role), f"logical_identity_layer.{role}")
            if len(records) != source.frame_count:
                raise ValueError(f"logical_identity_layer.{role} must contain every frame")
            for expected_frame, value in enumerate(records):
                field = f"logical_identity_layer.{role}[{expected_frame}]"
                record = _object(value, field)
                frame = _integer(record.get("frame_number"), f"{field}.frame_number")
                if frame != expected_frame:
                    raise ValueError(f"{field}.frame_number is out of sequence")
                confidence = _number(
                    record.get("tracking_confidence"),
                    f"{field}.tracking_confidence",
                )
                if not 0 <= confidence <= 1:
                    raise ValueError(f"{field}.tracking_confidence must be in [0, 1]")
                linked_observation_id = _optional_string(
                    record.get("raw_tracker_observation_id"),
                    f"{field}.raw_tracker_observation_id",
                )
                ground_value = record.get("ground_contact")
                if linked_observation_id is None or ground_value is None:
                    continue
                if linked_observation_id not in raw_observations:
                    raise ValueError(f"{field} references an unknown tracker observation")
                raw_frame, linked_box = raw_observations[linked_observation_id]
                if raw_frame != frame:
                    raise ValueError(f"{field} links a tracker observation from another frame")
                ground = _object(ground_value, f"{field}.ground_contact")
                if ground.get("method") != "bounding_box_bottom_center":
                    raise ValueError(f"{field}.ground_contact uses an unsupported method")
                image = _object(
                    ground.get("image_point"),
                    f"{field}.ground_contact.image_point",
                )
                image_point = ImagePoint(
                    _number(image.get("x_px"), f"{field}.ground_contact.image_point.x_px"),
                    _number(image.get("y_px"), f"{field}.ground_contact.image_point.y_px"),
                )
                if not (
                    0 <= image_point.x_px <= source.width and 0 <= image_point.y_px <= source.height
                ):
                    raise ValueError(f"{field}.ground_contact.image_point is outside the frame")
                tracking_state = _string(
                    record.get("tracking_state"),
                    f"{field}.tracking_state",
                )
                if tracking_state not in {
                    "observed",
                    "reacquired",
                    "temporarily_missing",
                    "suspected_identity_switch",
                }:
                    raise ValueError(f"{field}.tracking_state is unsupported")
                court_side = _optional_string(
                    ground.get("court_side"),
                    f"{field}.ground_contact.court_side",
                )
                if court_side not in {None, "near_side", "far_side", "ambiguous"}:
                    raise ValueError(f"{field}.ground_contact.court_side is unsupported")
                court_region = _optional_string(
                    ground.get("court_region"),
                    f"{field}.ground_contact.court_region",
                )
                if court_region not in {None, "inside", "near", "outside", "ambiguous"}:
                    raise ValueError(f"{field}.ground_contact.court_region is unsupported")
                by_frame[frame].append(
                    ContactPlayerObservation(
                        role=role,
                        display_name=player_names.get(role),
                        frame=frame,
                        bounding_box=linked_box,
                        ground_image_position=image_point,
                        tracking_confidence=confidence,
                        tracking_state=tracking_state,
                        court_side=court_side,
                        court_region=court_region,
                    )
                )
        return LoadedContactPlayerTracks(
            resolved,
            _sha256(resolved),
            tuple(tuple(items) for items in by_frame),
            player_names,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ContactDetectionInputError(f"invalid player tracks {resolved}: {error}") from error


def load_contact_audio(
    path: Path | None,
    *,
    media: MediaMetadata,
    timeline: MediaTimeline,
) -> LoadedContactAudio:
    """Load generic transients and reapply the current configurable A/V offset."""

    if path is None:
        return LoadedContactAudio(
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
        transients: list[ContactAudioTransient] = []
        for index, value in enumerate(
            _array(root.get("audioEventCandidates", []), "audioEventCandidates")
        ):
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
            analysis_time = _number(
                candidate.get("analysisTimestampSeconds"),
                f"{field}.analysisTimestampSeconds",
            )
            video_time = (
                audio_start + analysis_time + timeline.audio_video_offset_ms / 1000.0 - video_start
            )
            if -1 / media.video.fps <= video_time <= media.video.duration + 1 / media.video.fps:
                transients.append(
                    ContactAudioTransient(
                        _string(candidate.get("id"), f"{field}.id"),
                        max(0.0, video_time),
                        confidence,
                    )
                )
        return LoadedContactAudio(
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
        raise ContactDetectionInputError(f"invalid audio events {resolved}: {error}") from error


def load_contact_rallies(
    path: Path | None,
    *,
    source: VideoMetadata,
) -> LoadedContactRallies:
    """Load optional rally intervals as non-creating sequence support."""

    if path is None:
        return LoadedContactRallies(False, None, None, ())
    resolved = path.expanduser().resolve()
    root = _read_root(resolved, "rallies")
    try:
        if root.get("schemaVersion") != 1 or root.get("recordType") != ("automatic_rally_segments"):
            raise ValueError("unsupported rallies schema or recordType")
        _validate_video_source(root.get("source"), source, field="source")
        intervals: list[ContactRallyInterval] = []
        for index, value in enumerate(_array(root.get("rallies"), "rallies")):
            field = f"rallies[{index}]"
            item = _object(value, field)
            start = _integer(item.get("startFrame"), f"{field}.startFrame")
            end = _integer(item.get("endFrame"), f"{field}.endFrame")
            confidence = _number(item.get("confidence"), f"{field}.confidence")
            if start < 0 or end < start or end >= source.frame_count:
                raise ValueError(f"{field} frame interval is invalid")
            if not 0 <= confidence <= 1:
                raise ValueError(f"{field}.confidence must be in [0, 1]")
            intervals.append(
                ContactRallyInterval(
                    _string(item.get("rallyId"), f"{field}.rallyId"),
                    start,
                    end,
                    confidence,
                )
            )
        return LoadedContactRallies(True, resolved, _sha256(resolved), tuple(intervals))
    except (KeyError, TypeError, ValueError) as error:
        raise ContactDetectionInputError(f"invalid rallies {resolved}: {error}") from error


def load_prior_bounces(
    path: Path | None,
    *,
    source: VideoMetadata,
) -> LoadedPriorBounces:
    """Load accepted prior-stage bounces without reinterpreting low-confidence records."""

    if path is None:
        return LoadedPriorBounces(False, None, None, ())
    resolved = path.expanduser().resolve()
    root = _read_root(resolved, "bounce candidates")
    try:
        if root.get("schemaVersion") != 1 or root.get("recordType") != (
            "multimodal_bounce_candidates"
        ):
            raise ValueError("unsupported bounces schema or recordType")
        _validate_video_source(root.get("source"), source, field="source")
        bounces: list[PriorBounce] = []
        for index, value in enumerate(_array(root.get("bounceCandidates"), "bounceCandidates")):
            field = f"bounceCandidates[{index}]"
            item = _object(value, field)
            accepted = item.get("acceptedFused")
            if not isinstance(accepted, bool):
                raise ValueError(f"{field}.acceptedFused must be boolean")
            if not accepted:
                continue
            frame = _integer(item.get("frame"), f"{field}.frame")
            confidence = _number(item.get("fusedConfidence"), f"{field}.fusedConfidence")
            if not 0 <= frame < source.frame_count or not 0 <= confidence <= 1:
                raise ValueError(f"{field} has invalid frame or confidence")
            bounces.append(
                PriorBounce(
                    _string(item.get("bounceId"), f"{field}.bounceId"),
                    frame,
                    confidence,
                )
            )
        return LoadedPriorBounces(True, resolved, _sha256(resolved), tuple(bounces))
    except (KeyError, TypeError, ValueError) as error:
        raise ContactDetectionInputError(f"invalid bounces {resolved}: {error}") from error


def load_contact_annotations(
    path: Path | None,
    *,
    source: VideoMetadata,
    source_sha256: str | None,
) -> LoadedContactAnnotations:
    """Load human serve/paddle contacts and reviewed rally windows for evaluation."""

    if path is None:
        return LoadedContactAnnotations(False, None, None, (), ())
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
        contacts: list[GroundTruthContact] = []
        intervals: list[ContactReviewedInterval] = []
        active: tuple[str, int] | None = None
        previous_frame = -1
        for index, value in enumerate(_array(root.get("events"), "events")):
            field = f"events[{index}]"
            event = _object(value, field)
            frame = _integer(event.get("frame"), f"{field}.frame")
            if frame < previous_frame:
                raise ValueError("annotation events must be chronological")
            if frame < 0 or frame >= source.frame_count:
                raise ValueError(f"{field}.frame is outside the source video")
            previous_frame = frame
            event_id = _string(event.get("id"), f"{field}.id")
            event_type = _string(event.get("type"), f"{field}.type")
            if event_type in {"SERVE_CONTACT", "PADDLE_CONTACT"}:
                contacts.append(
                    GroundTruthContact(
                        event_id,
                        event_type,
                        frame,
                        frame / source.fps,
                        _optional_string(event.get("playerId"), f"{field}.playerId"),
                    )
                )
            elif event_type == "RALLY_START":
                if active is not None:
                    raise ValueError("RALLY_START occurs before the prior RALLY_END")
                active = (event_id, frame)
            elif event_type == "RALLY_END":
                if active is None:
                    raise ValueError("RALLY_END has no preceding RALLY_START")
                start_id, start_frame = active
                intervals.append(
                    ContactReviewedInterval(
                        f"{start_id}:{event_id}",
                        start_frame,
                        frame,
                    )
                )
                active = None
        if active is not None:
            raise ValueError("final RALLY_START has no RALLY_END")
        return LoadedContactAnnotations(
            True,
            resolved,
            _sha256(resolved),
            tuple(contacts),
            tuple(intervals),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ContactDetectionInputError(f"invalid annotations {resolved}: {error}") from error


def _prepare_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise OutputWriteError(str(output), reason="path is not a directory")
    for name in (CONTACTS_NAME, CONTACT_DEBUG_NAME, CONTACT_EVALUATION_NAME):
        artifact = output / name
        if artifact.exists():
            raise OutputWriteError(str(artifact), reason="contact-detection output already exists")
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
    candidates: tuple[ContactCandidate, ...],
    audio_available: bool,
) -> None:
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    writer = _open_writer(temporary, source)
    by_frame = {item.frame: item for item in candidates}
    recent_frames = max(1, round(0.15 * source.fps))
    last: ContactCandidate | None = None
    logger = logging.getLogger("pickleball_vision.contact_detection")
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
                render_contact_detection_frame(
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
                    "contact_debug_progress",
                    extra={"context": {"processed_frames": processed}},
                )
    except Exception:
        writer.abort()
        temporary.unlink(missing_ok=True)
        raise
    writer.release()
    if processed != source.frame_count:
        temporary.unlink(missing_ok=True)
        raise ContactDetectionInputError(
            f"decoded {processed} frames but trajectory contains {source.frame_count}"
        )
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise OutputWriteError(str(output_path), reason="OpenCV wrote an empty debug video")
    temporary.replace(output_path)


def _statistics(result: ContactDetectionResult) -> dict[str, object]:
    accepted = tuple(item for item in result.candidates if item.accepted_fused)
    modes = {
        mode.value: sum(item.evidence_mode is mode for item in result.candidates)
        for mode in ContactEvidenceMode
    }
    return {
        "rawVisualCandidateCount": result.raw_visual_candidate_count,
        "visualCandidateCountAfterTemporalSuppression": len(result.candidates),
        "suppressedVisualCandidateCount": result.suppressed_visual_candidate_count,
        "bounceExcludedVisualCandidateCount": result.bounce_excluded_candidate_count,
        "acceptedContactCount": len(accepted),
        "lowConfidenceCandidateCount": len(result.candidates) - len(accepted),
        "matchedAudioCandidateCount": result.matched_audio_candidate_count,
        "candidateWithTrackedPlayersCount": sum(
            bool(item.candidate_players) for item in result.candidates
        ),
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
    }


def detect_contacts_in_video(
    video_path: Path,
    *,
    ball_tracks_path: Path,
    player_tracks_path: Path,
    output_dir: Path,
    settings: ContactDetectionSettings,
    timeline: MediaTimeline,
    rallies_path: Path | None = None,
    bounces_path: Path | None = None,
    audio_events_path: Path | None = None,
    annotations_path: Path | None = None,
    annotations_complete: bool = False,
    evaluation_partition: str = "validation",
) -> ContactDetectionArtifacts:
    """Run visual inference first, then optional audio fusion and evaluation."""

    media = inspect_media(video_path, timeline=timeline)
    source = media.video
    try:
        ball: LoadedBallTrajectory = load_ball_trajectory(ball_tracks_path, source=source)
    except RallySegmentationInputError as error:
        raise ContactDetectionInputError(str(error)) from error
    players = load_contact_player_tracks(player_tracks_path, source=source)
    rallies = load_contact_rallies(rallies_path, source=source)
    bounces = load_prior_bounces(bounces_path, source=source)
    audio = load_contact_audio(audio_events_path, media=media, timeline=timeline)
    source_hash = _sha256(source.path) if annotations_path is not None else None
    annotations = load_contact_annotations(
        annotations_path,
        source=source,
        source_sha256=source_hash,
    )
    result = detect_contact_candidates(
        ball.frames,
        players_by_frame=players.observations_by_frame,
        fps=source.fps,
        frame_width_px=source.width,
        frame_height_px=source.height,
        settings=settings,
        audio_transients=audio.transients,
        rallies=rallies.intervals,
        prior_bounces=bounces.bounces,
        video_start_time_seconds=media.video_start_time_seconds or 0.0,
        fusion_tolerance_ms=timeline.fusion_tolerance_ms,
    )
    evaluation = (
        evaluate_contacts(
            result.candidates,
            annotations.contacts,
            fps=source.fps,
            settings=settings,
            annotations_complete=annotations_complete,
            reviewed_intervals=annotations.reviewed_intervals,
            evaluation_partition=evaluation_partition,
        )
        if annotations.contacts
        else unavailable_contact_evaluation(evaluation_partition=evaluation_partition)
    )
    output = _prepare_output(output_dir)
    contacts_path = output / CONTACTS_NAME
    debug_path = output / CONTACT_DEBUG_NAME
    evaluation_path = output / CONTACT_EVALUATION_NAME
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
        "playerTracks": players.as_dict(),
        "rallies": rallies.as_dict(),
        "priorBounces": bounces.as_dict(),
        "audioEvents": audio.as_dict(),
        "groundTruthAnnotations": annotations.as_dict(),
    }
    configuration = {
        **settings.as_dict(),
        "audioVideoOffsetMs": timeline.audio_video_offset_ms,
        "fusionToleranceMs": timeline.fusion_tolerance_ms,
    }
    _write_json(
        contacts_path,
        {
            "schemaVersion": CONTACT_DETECTION_SCHEMA_VERSION,
            "recordType": "multimodal_paddle_contact_candidates",
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
                "audioCanCreateContact": False,
                "neighboringCourtSoundsMayBePresent": True,
                "logicalPlayerIdentityIndependentOfTrackerId": True,
                "candidatePlayersAreNotHitterAssignments": True,
                "hitterIdentificationImplemented": False,
                "airborneBallProjectedThroughHomography": False,
                "rawInputsMutated": False,
            },
            "statistics": _statistics(result),
            "contactCandidates": [item.as_dict() for item in result.candidates],
        },
    )
    _write_json(
        evaluation_path,
        {
            "schemaVersion": CONTACT_DETECTION_SCHEMA_VERSION,
            "recordType": "multimodal_paddle_contact_evaluation",
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
                "contacts": str(contacts_path),
                "debugVideo": str(debug_path),
                "evaluation": str(evaluation_path),
            },
        },
    )
    return ContactDetectionArtifacts(
        contacts_path,
        debug_path,
        evaluation_path,
        len(result.candidates),
        sum(item.accepted_fused for item in result.candidates),
        sum(item.matched_audio_event_id is not None for item in result.candidates),
    )
