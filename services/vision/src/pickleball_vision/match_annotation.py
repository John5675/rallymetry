"""Versioned human ground-truth match events and crash-resistant editing."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import cast

from pickleball_vision.errors import MatchAnnotationInputError, MatchAnnotationIoError
from pickleball_vision.media import MediaMetadata, MediaTimeline, inspect_media

MATCH_ANNOTATION_VERSION = 1
MATCH_ANNOTATION_RECORD_TYPE = "multimodal_match_ground_truth"
EVENT_ID_PATTERN = re.compile(r"^match-event-(\d{7})$")
MAXIMUM_NOTES_LENGTH = 4_000
MAXIMUM_LABEL_LENGTH = 120


class MatchEventType(StrEnum):
    """Human-authored event types supported before automatic event inference."""

    RALLY_START = "RALLY_START"
    RALLY_END = "RALLY_END"
    SERVE_CONTACT = "SERVE_CONTACT"
    PADDLE_CONTACT = "PADDLE_CONTACT"
    BOUNCE = "BOUNCE"
    RALLY_WINNER = "RALLY_WINNER"
    SHOT_TYPE = "SHOT_TYPE"


class AudioAnnotationLabel(StrEnum):
    """Optional human judgment about synchronized audio at an annotated event."""

    PRIMARY_EVENT_AUDIBLE = "PRIMARY_EVENT_AUDIBLE"
    PRIMARY_EVENT_NOT_AUDIBLE = "PRIMARY_EVENT_NOT_AUDIBLE"
    OTHER_COURT_TRANSIENT = "OTHER_COURT_TRANSIENT"
    AMBIGUOUS_AUDIO = "AMBIGUOUS_AUDIO"


@dataclass(frozen=True, slots=True)
class CourtPosition:
    """Human-authored point on the canonical court plane, in meters."""

    x_meters: float
    y_meters: float

    def as_dict(self) -> dict[str, object]:
        return {
            "xMeters": self.x_meters,
            "yMeters": self.y_meters,
            "coordinateSystem": "canonical_pickleball_court",
            "source": "HUMAN_ANNOTATION",
        }


@dataclass(frozen=True, slots=True)
class MatchAnnotationEvent:
    """One human-authored event tied to an exact source-video frame."""

    event_id: str
    event_type: MatchEventType
    frame: int
    video_timestamp_seconds: float
    media_timestamp_seconds: float
    player_id: str | None
    team: str | None
    shot_type: str | None
    court_position: CourtPosition | None
    audio_label: AudioAnnotationLabel | None
    notes: str | None
    annotation_confidence: float
    annotator: str
    created_at_utc: str
    updated_at_utc: str

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.event_id,
            "type": self.event_type.value,
            "frame": self.frame,
            "videoTimestampSeconds": self.video_timestamp_seconds,
            "mediaTimestampSeconds": self.media_timestamp_seconds,
            "playerId": self.player_id,
            "team": self.team,
            "shotType": self.shot_type,
            "courtPosition": (
                self.court_position.as_dict() if self.court_position is not None else None
            ),
            "audioLabel": self.audio_label.value if self.audio_label is not None else None,
            "notes": self.notes,
            "annotationConfidence": self.annotation_confidence,
            "annotator": self.annotator,
            "createdAtUtc": self.created_at_utc,
            "updatedAtUtc": self.updated_at_utc,
            "source": "HUMAN",
        }


@dataclass(frozen=True, slots=True)
class AudioTransientMarker:
    """Non-semantic Prompt 10 candidate shown as optional annotation context."""

    candidate_id: str
    media_timestamp_seconds: float
    confidence: float

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.candidate_id,
            "mediaTimestampSeconds": self.media_timestamp_seconds,
            "confidence": self.confidence,
            "candidateType": "TRANSIENT",
            "semanticEvent": False,
        }


@dataclass(frozen=True, slots=True)
class MatchAudioContext:
    """Optional raw audio-analysis context; never imported as ground truth."""

    requested: bool
    analysis_available: bool
    source_artifact_path: Path | None
    source_artifact_sha256: str | None
    waveform_path: Path | None
    markers: tuple[AudioTransientMarker, ...]
    audio_video_offset_ms: float

    def document_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "audioAnalysisAvailable": self.analysis_available,
            "sourceArtifact": (
                str(self.source_artifact_path) if self.source_artifact_path is not None else None
            ),
            "sourceArtifactSha256": self.source_artifact_sha256,
            "waveformAvailable": self.waveform_path is not None,
            "transientCandidateCount": len(self.markers),
            "audioVideoOffsetMs": self.audio_video_offset_ms,
            "candidatesImportedAsGroundTruth": False,
        }


@dataclass(frozen=True, slots=True)
class MatchAnnotationArtifacts:
    """Paths and event count returned after a local annotation session."""

    url: str
    annotations_path: Path
    event_count: int
    audio_context_available: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "url": self.url,
            "annotationsPath": str(self.annotations_path),
            "eventCount": self.event_count,
            "audioContextAvailable": self.audio_context_available,
        }


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return cast(list[object], value)


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _string(value: object, field: str, *, maximum: int = MAXIMUM_LABEL_LENGTH) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    result = value.strip()
    if len(result) > maximum:
        raise ValueError(f"{field} must contain at most {maximum} characters")
    return result


def _optional_string(
    value: object,
    field: str,
    *,
    maximum: int = MAXIMUM_LABEL_LENGTH,
) -> str | None:
    if value is None or value == "":
        return None
    return _string(value, field, maximum=maximum)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise MatchAnnotationIoError(str(path), reason=str(error)) from error
    return digest.hexdigest()


def _read_json(path: Path, kind: str) -> dict[str, object]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), "root")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise MatchAnnotationIoError(str(path), reason=f"unable to load {kind}: {error}") from error


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except (OSError, TypeError, ValueError) as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise MatchAnnotationIoError(str(path), reason=str(error)) from error


def _court_position(value: object, field: str) -> CourtPosition | None:
    if value is None:
        return None
    raw = _object(value, field)
    allowed = {"xMeters", "yMeters", "coordinateSystem", "source"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{field} contains unsupported fields: {sorted(unknown)}")
    if raw.get("coordinateSystem", "canonical_pickleball_court") != ("canonical_pickleball_court"):
        raise ValueError(f"{field}.coordinateSystem is unsupported")
    if raw.get("source", "HUMAN_ANNOTATION") != "HUMAN_ANNOTATION":
        raise ValueError(f"{field}.source must be HUMAN_ANNOTATION")
    return CourtPosition(
        x_meters=_number(raw.get("xMeters"), f"{field}.xMeters"),
        y_meters=_number(raw.get("yMeters"), f"{field}.yMeters"),
    )


def _audio_context(
    path: Path | None,
    *,
    media: MediaMetadata,
) -> MatchAudioContext:
    if path is None:
        return MatchAudioContext(False, False, None, None, None, (), 0.0)
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise MatchAnnotationIoError(str(resolved), reason="audio-events artifact does not exist")
    root = _read_json(resolved, "audio event context")
    try:
        if root.get("schemaVersion") != 1 or root.get("recordType") != (
            "audio_analysis_observations"
        ):
            raise ValueError("audio context has an unsupported schema or recordType")
        source = _object(root.get("sourceMedia"), "sourceMedia")
        source_path = Path(_string(source.get("path"), "sourceMedia.path", maximum=4_096))
        if source_path.expanduser().resolve() != media.video.path:
            raise ValueError("audio context belongs to a different source video")
        available = root.get("audioAnalysisAvailable")
        if not isinstance(available, bool):
            raise ValueError("audioAnalysisAvailable must be boolean")
        if available and media.audio is None:
            raise ValueError("audio context is available but source video has no audio stream")
        markers: list[AudioTransientMarker] = []
        for index, item in enumerate(
            _array(root.get("audioEventCandidates", []), "audioEventCandidates")
        ):
            field = f"audioEventCandidates[{index}]"
            candidate = _object(item, field)
            if (
                candidate.get("candidateType") != "TRANSIENT"
                or candidate.get("semanticClassification") is not None
            ):
                raise ValueError(f"{field} must be a generic non-semantic TRANSIENT")
            if candidate.get("source") != "AUDIO":
                raise ValueError(f"{field}.source must be AUDIO")
            confidence = _number(candidate.get("confidence"), f"{field}.confidence")
            if not 0 <= confidence <= 1:
                raise ValueError(f"{field}.confidence must be in [0, 1]")
            markers.append(
                AudioTransientMarker(
                    candidate_id=_string(candidate.get("id"), f"{field}.id"),
                    media_timestamp_seconds=_number(
                        candidate.get("mediaTimestampSeconds"),
                        f"{field}.mediaTimestampSeconds",
                    ),
                    confidence=confidence,
                )
            )
        configuration = _object(root.get("configuration", {}), "configuration")
        offset = _number(configuration.get("audioVideoOffsetMs", 0.0), "audioVideoOffsetMs")
        artifacts = _object(root.get("artifacts", {}), "artifacts")
        waveform_value = artifacts.get("waveform")
        waveform_path = (
            Path(_string(waveform_value, "artifacts.waveform", maximum=4_096))
            .expanduser()
            .resolve()
            if waveform_value is not None
            else None
        )
        if waveform_path is not None and not waveform_path.is_file():
            waveform_path = None
        return MatchAudioContext(
            requested=True,
            analysis_available=available,
            source_artifact_path=resolved,
            source_artifact_sha256=_sha256(resolved),
            waveform_path=waveform_path,
            markers=tuple(markers),
            audio_video_offset_ms=offset,
        )
    except ValueError as error:
        raise MatchAnnotationInputError(str(error)) from error


class MatchAnnotationStore:
    """Thread-safe annotation document with validated atomic add/edit/delete."""

    def __init__(
        self,
        video_path: Path,
        *,
        output_path: Path,
        timeline: MediaTimeline | None = None,
        audio_events_path: Path | None = None,
    ) -> None:
        self.timeline = timeline or MediaTimeline()
        self.media = inspect_media(video_path, timeline=self.timeline)
        self.video_path = self.media.video.path
        self.output_path = output_path.expanduser().resolve()
        if self.output_path == self.video_path:
            raise MatchAnnotationInputError("output would overwrite the source recording")
        if self.output_path.exists() and not self.output_path.is_file():
            raise MatchAnnotationIoError(str(self.output_path), reason="path is not a regular file")
        self.source_sha256 = _sha256(self.video_path)
        self._lock = threading.RLock()
        if self.output_path.exists():
            self._root = _read_json(self.output_path, "match annotations")
            recovered_audio_path = audio_events_path
            if recovered_audio_path is None:
                with suppress(ValueError):
                    stored_audio = _object(self._root.get("audioContext"), "audioContext")
                    stored_path = stored_audio.get("sourceArtifact")
                    if isinstance(stored_path, str) and Path(stored_path).is_file():
                        recovered_audio_path = Path(stored_path)
            self.audio_context = _audio_context(recovered_audio_path, media=self.media)
            self._validate_document()
            if audio_events_path is not None:
                self._root["audioContext"] = self.audio_context.document_dict()
                self._root["updatedAtUtc"] = _now_utc()
                self._persist()
        else:
            self.audio_context = _audio_context(audio_events_path, media=self.media)
            now = _now_utc()
            self._root = {
                "annotationVersion": MATCH_ANNOTATION_VERSION,
                "recordType": MATCH_ANNOTATION_RECORD_TYPE,
                "createdAtUtc": now,
                "updatedAtUtc": now,
                "video": {
                    **self.media.as_dict(),
                    "contentSha256": self.source_sha256,
                    "canonicalTimeline": "source_media_presentation_time_seconds",
                    "frameTimestampMapping": (
                        "mediaTimestampSeconds = (videoStartTime or 0) + frame / fps"
                    ),
                },
                "audioContext": self.audio_context.document_dict(),
                "events": [],
                "contracts": {
                    "humanGroundTruth": True,
                    "automaticEventInference": False,
                    "audioLabelsOptional": True,
                    "audioTransientsAreSemanticEvents": False,
                    "sourceMediaModified": False,
                },
            }
            self._persist()

    def _frame_timestamps(self, frame: int) -> tuple[float, float]:
        if frame < 0 or frame >= self.media.video.frame_count:
            raise ValueError(
                f"frame must be in [0, {self.media.video.frame_count - 1}]; received {frame}"
            )
        video_timestamp = frame / self.media.video.fps
        media_timestamp = self.timeline.video_timestamp_to_media_time(
            video_timestamp,
            video_start_time_seconds=self.media.video_start_time_seconds,
        )
        return video_timestamp, media_timestamp

    def _event_from_dict(self, value: object, *, field: str) -> MatchAnnotationEvent:
        raw = _object(value, field)
        event_id = _string(raw.get("id"), f"{field}.id")
        if EVENT_ID_PATTERN.fullmatch(event_id) is None:
            raise ValueError(f"{field}.id has an unsupported format")
        try:
            event_type = MatchEventType(_string(raw.get("type"), f"{field}.type"))
        except ValueError as error:
            raise ValueError(f"{field}.type is unsupported") from error
        frame = _integer(raw.get("frame"), f"{field}.frame")
        expected_video_time, expected_media_time = self._frame_timestamps(frame)
        video_time = _number(raw.get("videoTimestampSeconds"), f"{field}.videoTimestampSeconds")
        media_time = _number(raw.get("mediaTimestampSeconds"), f"{field}.mediaTimestampSeconds")
        tolerance = max(1e-9, 1 / self.media.video.fps / 1000)
        if abs(video_time - expected_video_time) > tolerance:
            raise ValueError(f"{field}.videoTimestampSeconds is inconsistent with frame")
        if abs(media_time - expected_media_time) > tolerance:
            raise ValueError(f"{field}.mediaTimestampSeconds is inconsistent with frame")
        confidence = _number(raw.get("annotationConfidence"), f"{field}.annotationConfidence")
        if not 0 <= confidence <= 1:
            raise ValueError(f"{field}.annotationConfidence must be in [0, 1]")
        audio_value = raw.get("audioLabel")
        try:
            audio_label = (
                AudioAnnotationLabel(_string(audio_value, f"{field}.audioLabel"))
                if audio_value is not None
                else None
            )
        except ValueError as error:
            raise ValueError(f"{field}.audioLabel is unsupported") from error
        if raw.get("source", "HUMAN") != "HUMAN":
            raise ValueError(f"{field}.source must be HUMAN")
        return MatchAnnotationEvent(
            event_id=event_id,
            event_type=event_type,
            frame=frame,
            video_timestamp_seconds=expected_video_time,
            media_timestamp_seconds=expected_media_time,
            player_id=_optional_string(raw.get("playerId"), f"{field}.playerId"),
            team=_optional_string(raw.get("team"), f"{field}.team"),
            shot_type=_optional_string(raw.get("shotType"), f"{field}.shotType"),
            court_position=_court_position(raw.get("courtPosition"), f"{field}.courtPosition"),
            audio_label=audio_label,
            notes=_optional_string(
                raw.get("notes"),
                f"{field}.notes",
                maximum=MAXIMUM_NOTES_LENGTH,
            ),
            annotation_confidence=confidence,
            annotator=_string(raw.get("annotator"), f"{field}.annotator"),
            created_at_utc=_string(raw.get("createdAtUtc"), f"{field}.createdAtUtc", maximum=100),
            updated_at_utc=_string(raw.get("updatedAtUtc"), f"{field}.updatedAtUtc", maximum=100),
        )

    def _validate_document(self) -> None:
        try:
            if self._root.get("annotationVersion") != MATCH_ANNOTATION_VERSION:
                raise ValueError("annotationVersion is unsupported")
            if self._root.get("recordType") != MATCH_ANNOTATION_RECORD_TYPE:
                raise ValueError("recordType is unsupported")
            video = _object(self._root.get("video"), "video")
            if video.get("contentSha256") != self.source_sha256:
                raise ValueError("annotation file belongs to different source-media bytes")
            for key, expected in (
                ("width", self.media.video.width),
                ("height", self.media.video.height),
                ("frame_count", self.media.video.frame_count),
            ):
                if video.get(key) != expected:
                    raise ValueError(f"video.{key} differs from the current source")
            if not math.isclose(
                _number(video.get("fps"), "video.fps"),
                self.media.video.fps,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError("video.fps differs from the current source")
            events = _array(self._root.get("events"), "events")
            parsed = [
                self._event_from_dict(value, field=f"events[{index}]")
                for index, value in enumerate(events)
            ]
            ids = [event.event_id for event in parsed]
            if len(ids) != len(set(ids)):
                raise ValueError("events contain duplicate IDs")
        except ValueError as error:
            raise MatchAnnotationInputError(str(error)) from error

    def _persist(self) -> None:
        _write_json_atomic(self.output_path, self._root)

    def _events(self) -> list[dict[str, object]]:
        return [
            _object(item, f"events[{index}]")
            for index, item in enumerate(_array(self._root.get("events"), "events"))
        ]

    def _next_id(self) -> str:
        maximum = 0
        for event in self._events():
            match = EVENT_ID_PATTERN.fullmatch(cast(str, event["id"]))
            if match is not None:
                maximum = max(maximum, int(match.group(1)))
        return f"match-event-{maximum + 1:07d}"

    def _event_from_input(
        self,
        payload: Mapping[str, object],
        *,
        event_id: str,
        created_at_utc: str,
    ) -> MatchAnnotationEvent:
        allowed = {
            "type",
            "frame",
            "playerId",
            "team",
            "shotType",
            "courtPosition",
            "audioLabel",
            "notes",
            "annotationConfidence",
            "annotator",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"event update contains unsupported fields: {sorted(unknown)}")
        frame = _integer(payload.get("frame"), "frame")
        video_timestamp, media_timestamp = self._frame_timestamps(frame)
        now = _now_utc()
        candidate = {
            "id": event_id,
            **dict(payload),
            "videoTimestampSeconds": video_timestamp,
            "mediaTimestampSeconds": media_timestamp,
            "annotationConfidence": payload.get("annotationConfidence", 1.0),
            "annotator": payload.get("annotator", "local-annotator"),
            "createdAtUtc": created_at_utc,
            "updatedAtUtc": now,
            "source": "HUMAN",
        }
        return self._event_from_dict(candidate, field="event")

    def _sort_events(self) -> None:
        events = self._events()
        events.sort(key=lambda event: (_integer(event["frame"], "frame"), cast(str, event["id"])))
        self._root["events"] = events

    def add_event(self, payload: object) -> dict[str, object]:
        """Validate, append, and atomically persist one human event."""

        with self._lock:
            try:
                raw = _object(payload, "event")
                now = _now_utc()
                event = self._event_from_input(
                    raw,
                    event_id=self._next_id(),
                    created_at_utc=now,
                )
                events = self._events()
                events.append(event.as_dict())
                self._root["events"] = events
                self._sort_events()
                self._root["updatedAtUtc"] = now
                self._persist()
                return event.as_dict()
            except ValueError as error:
                raise MatchAnnotationInputError(str(error)) from error

    def update_event(self, event_id: str, payload: object) -> dict[str, object]:
        """Patch an existing event while retaining its ID and creation provenance."""

        with self._lock:
            try:
                raw = _object(payload, "event update")
                events = self._events()
                index = next(
                    (
                        position
                        for position, event in enumerate(events)
                        if event.get("id") == event_id
                    ),
                    None,
                )
                if index is None:
                    raise ValueError(f"event {event_id!r} does not exist")
                existing = events[index]
                mutable_fields = {
                    key: existing.get(key)
                    for key in (
                        "type",
                        "frame",
                        "playerId",
                        "team",
                        "shotType",
                        "courtPosition",
                        "audioLabel",
                        "notes",
                        "annotationConfidence",
                        "annotator",
                    )
                }
                unknown = set(raw) - set(mutable_fields)
                if unknown:
                    raise ValueError(f"event update contains unsupported fields: {sorted(unknown)}")
                mutable_fields.update(raw)
                event = self._event_from_input(
                    mutable_fields,
                    event_id=event_id,
                    created_at_utc=_string(existing.get("createdAtUtc"), "createdAtUtc"),
                )
                events[index] = event.as_dict()
                self._root["events"] = events
                self._sort_events()
                self._root["updatedAtUtc"] = event.updated_at_utc
                self._persist()
                return event.as_dict()
            except ValueError as error:
                raise MatchAnnotationInputError(str(error)) from error

    def delete_event(self, event_id: str) -> dict[str, object]:
        """Delete one event atomically without renumbering remaining stable IDs."""

        with self._lock:
            try:
                events = self._events()
                index = next(
                    (
                        position
                        for position, event in enumerate(events)
                        if event.get("id") == event_id
                    ),
                    None,
                )
                if index is None:
                    raise ValueError(f"event {event_id!r} does not exist")
                deleted = events.pop(index)
                self._root["events"] = events
                self._root["updatedAtUtc"] = _now_utc()
                self._persist()
                return deleted
            except ValueError as error:
                raise MatchAnnotationInputError(str(error)) from error

    def document_payload(self) -> dict[str, object]:
        """Return a detached JSON-compatible annotation document."""

        with self._lock:
            return cast(dict[str, object], json.loads(json.dumps(self._root, allow_nan=False)))

    def session_payload(self) -> dict[str, object]:
        """Return UI data while keeping transient context outside ground-truth events."""

        with self._lock:
            events = self._events()
            counts = {event_type.value: 0 for event_type in MatchEventType}
            for event in events:
                counts[cast(str, event["type"])] += 1
            video_start = self.media.video_start_time_seconds or 0.0
            return {
                "annotationVersion": MATCH_ANNOTATION_VERSION,
                "video": {
                    **self.media.as_dict(),
                    "videoUrl": "/media/video",
                    "mediaTimelineStartSeconds": video_start,
                },
                "events": json.loads(json.dumps(events, allow_nan=False)),
                "eventTypes": [event_type.value for event_type in MatchEventType],
                "audioLabels": [label.value for label in AudioAnnotationLabel],
                "counts": {"total": len(events), "byType": counts},
                "audioContext": {
                    **self.audio_context.document_dict(),
                    "waveformUrl": (
                        "/media/waveform" if self.audio_context.waveform_path is not None else None
                    ),
                    "transientMarkers": [marker.as_dict() for marker in self.audio_context.markers],
                },
                "contracts": {
                    "humanGroundTruthOnly": True,
                    "automaticEvents": False,
                    "audioLabelsOptional": True,
                },
            }
