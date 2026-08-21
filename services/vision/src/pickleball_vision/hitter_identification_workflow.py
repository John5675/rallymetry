"""Input validation, persistence, evaluation, and debug video for hitter identity."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import cast

from pickleball_vision.config import HitterIdentificationSettings
from pickleball_vision.contact_detection import (
    ContactCandidate,
    ContactCandidatePlayer,
    ContactEvidenceMode,
)
from pickleball_vision.contact_detection_workflow import (
    load_contact_annotations,
    load_contact_player_tracks,
)
from pickleball_vision.court import ImagePoint
from pickleball_vision.errors import (
    ContactDetectionInputError,
    HitterIdentificationInputError,
    OutputWriteError,
)
from pickleball_vision.hitter_evaluation import (
    evaluate_hitters,
    unavailable_hitter_evaluation,
)
from pickleball_vision.hitter_identification import (
    HITTER_IDENTIFICATION_SCHEMA_VERSION,
    LOGICAL_PLAYER_IDS,
    UNKNOWN_PLAYER_ID,
    HitterIdentification,
    identify_hitters,
)
from pickleball_vision.hitter_identification_render import (
    render_hitter_identification_frame,
)
from pickleball_vision.media import inspect_media
from pickleball_vision.rally_segmentation import BallEvidenceStatus
from pickleball_vision.video import VideoMetadata, iter_video_frames
from pickleball_vision.video_output import CompressedVideoWriter

HITTERS_NAME = "hitters.json"
HITTER_DEBUG_NAME = "hitter-debug.mp4"
HITTER_EVALUATION_NAME = "hitter-evaluation.json"


@dataclass(frozen=True, slots=True)
class LoadedHitterContacts:
    """Validated immutable contact candidates and provenance."""

    path: Path
    sha256: str
    created_at_utc: str | None
    candidates: tuple[ContactCandidate, ...]
    expected_player_tracks_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "createdAtUtc": self.created_at_utc,
            "contactCandidateCount": len(self.candidates),
            "usage": "immutable_visual_contact_and_candidate_player_evidence",
            "mutated": False,
        }


@dataclass(frozen=True, slots=True)
class HitterIdentificationArtifacts:
    """Paths and high-level counts for one completed hitter-identification run."""

    hitters_path: Path
    debug_video_path: Path
    evaluation_path: Path
    contact_count: int
    assigned_count: int
    unknown_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "hittersPath": str(self.hitters_path),
            "debugVideoPath": str(self.debug_video_path),
            "evaluationPath": str(self.evaluation_path),
            "contactCount": self.contact_count,
            "assignedCount": self.assigned_count,
            "unknownCount": self.unknown_count,
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


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise HitterIdentificationInputError(f"unable to hash {path}: {error}") from error
    return digest.hexdigest()


def _read_root(path: Path) -> dict[str, object]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), "root")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise HitterIdentificationInputError(
            f"unable to load contact candidates {path}: {error}"
        ) from error


def _validate_source(value: object, source: VideoMetadata) -> None:
    payload = _object(value, "source")
    path = Path(_string(payload.get("path"), "source.path")).expanduser().resolve()
    if path != source.path:
        raise ValueError("contacts belong to a different source video path")
    for key, expected in (
        ("width", source.width),
        ("height", source.height),
        ("frame_count", source.frame_count),
    ):
        if _integer(payload.get(key), f"source.{key}") != expected:
            raise ValueError(f"source.{key} does not match the video")
    if not math.isclose(_number(payload.get("fps"), "source.fps"), source.fps, rel_tol=5e-3):
        raise ValueError("source.fps does not match the video")


def _point(value: object, field: str, source: VideoMetadata) -> ImagePoint:
    payload = _object(value, field)
    point = ImagePoint(
        _number(payload.get("x_px"), f"{field}.x_px"),
        _number(payload.get("y_px"), f"{field}.y_px"),
    )
    if not 0 <= point.x_px <= source.width or not 0 <= point.y_px <= source.height:
        raise ValueError(f"{field} is outside the source frame")
    return point


def _candidate_player(
    value: object,
    *,
    field: str,
    source: VideoMetadata,
) -> ContactCandidatePlayer:
    payload = _object(value, field)
    role = _string(payload.get("playerId"), f"{field}.playerId")
    if role not in LOGICAL_PLAYER_IDS:
        raise ValueError(f"{field}.playerId is not a logical match role")
    box = _object(payload.get("trackerBoundingBox"), f"{field}.trackerBoundingBox")
    bounding_box = (
        _number(box.get("left_px"), f"{field}.trackerBoundingBox.left_px"),
        _number(box.get("top_px"), f"{field}.trackerBoundingBox.top_px"),
        _number(box.get("right_px"), f"{field}.trackerBoundingBox.right_px"),
        _number(box.get("bottom_px"), f"{field}.trackerBoundingBox.bottom_px"),
    )
    left, top, right, bottom = bounding_box
    if not (0 <= left < right <= source.width and 0 <= top < bottom <= source.height):
        raise ValueError(f"{field}.trackerBoundingBox is invalid")
    ground = _object(
        payload.get("groundContactImagePosition"),
        f"{field}.groundContactImagePosition",
    )
    if (
        ground.get("method") != "bounding_box_bottom_center"
        or ground.get("usedAsPersonPhysicalCourtPosition") is not True
    ):
        raise ValueError(f"{field}.groundContactImagePosition violates the ground-point contract")
    tracking_confidence = _number(payload.get("trackingConfidence"), f"{field}.trackingConfidence")
    proximity = _number(payload.get("proximityConfidence"), f"{field}.proximityConfidence")
    distance_px = _number(
        payload.get("ballToBoundingBoxDistancePx"),
        f"{field}.ballToBoundingBoxDistancePx",
    )
    distance_fraction = _number(
        payload.get("ballToBoundingBoxDistanceDiagonalFraction"),
        f"{field}.ballToBoundingBoxDistanceDiagonalFraction",
    )
    if not 0 <= tracking_confidence <= 1 or not 0 <= proximity <= 1:
        raise ValueError(f"{field} confidence fields must be in [0, 1]")
    if distance_px < 0 or distance_fraction < 0:
        raise ValueError(f"{field} distance fields must be nonnegative")
    if payload.get("isAssignedHitter") is not False:
        raise ValueError(f"{field}.isAssignedHitter must remain false in the source contact")
    court_side = _optional_string(payload.get("courtSide"), f"{field}.courtSide")
    if court_side not in {None, "near_side", "far_side", "ambiguous"}:
        raise ValueError(f"{field}.courtSide is unsupported")
    court_region = _optional_string(payload.get("courtRegion"), f"{field}.courtRegion")
    if court_region not in {None, "inside", "near", "outside", "ambiguous"}:
        raise ValueError(f"{field}.courtRegion is unsupported")
    return ContactCandidatePlayer(
        role=role,
        display_name=_optional_string(payload.get("displayName"), f"{field}.displayName"),
        rank=_integer(payload.get("rank"), f"{field}.rank"),
        bounding_box=bounding_box,
        ground_image_position=_point(ground, f"{field}.groundContactImagePosition", source),
        tracking_confidence=tracking_confidence,
        tracking_state=_string(payload.get("trackingState"), f"{field}.trackingState"),
        court_side=court_side,
        court_region=court_region,
        distance_px=distance_px,
        distance_diagonal_fraction=distance_fraction,
        proximity_confidence=proximity,
        ball_inside_person_box=_boolean(
            payload.get("ballInsidePersonBoundingBox"),
            f"{field}.ballInsidePersonBoundingBox",
        ),
    )


def _candidate(
    value: object,
    *,
    field: str,
    source: VideoMetadata,
) -> ContactCandidate:
    payload = _object(value, field)
    if payload.get("assignedHitter") is not None:
        raise ValueError(f"{field}.assignedHitter must be null in the source contact artifact")
    frame = _integer(payload.get("frame"), f"{field}.frame")
    if not 0 <= frame < source.frame_count:
        raise ValueError(f"{field}.frame is outside the source video")
    candidates = tuple(
        _candidate_player(item, field=f"{field}.candidatePlayers[{index}]", source=source)
        for index, item in enumerate(
            _array(payload.get("candidatePlayers"), f"{field}.candidatePlayers")
        )
    )
    if len({item.role for item in candidates}) != len(candidates):
        raise ValueError(f"{field}.candidatePlayers contains duplicate logical roles")
    if tuple(item.rank for item in candidates) != tuple(range(1, len(candidates) + 1)):
        raise ValueError(f"{field}.candidatePlayers ranks must be consecutive")
    confidences = {
        name: _number(payload.get(name), f"{field}.{name}")
        for name in ("visualConfidence", "audioConfidence", "fusedConfidence")
    }
    if any(not 0 <= value <= 1 for value in confidences.values()):
        raise ValueError(f"{field} confidence fields must be in [0, 1]")
    ball = _object(payload.get("ballImagePosition"), f"{field}.ballImagePosition")
    try:
        trajectory_status = BallEvidenceStatus(
            _string(ball.get("trajectoryStatus"), f"{field}.ballImagePosition.trajectoryStatus")
        )
        mode = ContactEvidenceMode(_string(payload.get("evidenceMode"), f"{field}.evidenceMode"))
    except ValueError as error:
        raise ValueError(f"{field} has an unsupported enum value") from error
    return ContactCandidate(
        contact_id=_string(payload.get("contactId"), f"{field}.contactId"),
        frame=frame,
        timestamp_seconds=_number(payload.get("timestamp"), f"{field}.timestamp"),
        media_timestamp_seconds=_number(payload.get("mediaTimestamp"), f"{field}.mediaTimestamp"),
        ball_image_position=_point(ball, f"{field}.ballImagePosition", source),
        trajectory_status=trajectory_status,
        candidate_players=candidates,
        visual_confidence=confidences["visualConfidence"],
        audio_confidence=confidences["audioConfidence"],
        fused_confidence=confidences["fusedConfidence"],
        matched_audio_event_id=_optional_string(
            payload.get("matchedAudioEventId"), f"{field}.matchedAudioEventId"
        ),
        evidence_mode=mode,
        accepted_vision_only=_boolean(
            payload.get("acceptedVisionOnly"), f"{field}.acceptedVisionOnly"
        ),
        accepted_fused=_boolean(payload.get("acceptedFused"), f"{field}.acceptedFused"),
        supporting_signals=_object(payload.get("supportingSignals"), f"{field}.supportingSignals"),
    )


def load_hitter_contacts(path: Path, *, source: VideoMetadata) -> LoadedHitterContacts:
    """Load the Prompt 15 artifact without changing its candidate-player semantics."""

    resolved = path.expanduser().resolve()
    root = _read_root(resolved)
    try:
        if root.get("schemaVersion") != 1 or root.get("recordType") != (
            "multimodal_paddle_contact_candidates"
        ):
            raise ValueError("unsupported contact schemaVersion or recordType")
        _validate_source(root.get("source"), source)
        contracts = _object(root.get("contracts"), "contracts")
        if contracts.get("candidatePlayersAreNotHitterAssignments") is not True:
            raise ValueError("source contacts do not preserve candidate-player separation")
        if contracts.get("hitterIdentificationImplemented") is not False:
            raise ValueError("source contacts already claim hitter identification")
        inputs = _object(root.get("inputs"), "inputs")
        player_tracks = _object(inputs.get("playerTracks"), "inputs.playerTracks")
        expected_tracks_hash = _string(player_tracks.get("sha256"), "inputs.playerTracks.sha256")
        candidates = tuple(
            _candidate(item, field=f"contactCandidates[{index}]", source=source)
            for index, item in enumerate(_array(root.get("contactCandidates"), "contactCandidates"))
        )
        if any(current.frame >= following.frame for current, following in pairwise(candidates)):
            raise ValueError("contactCandidates must be strictly chronological")
        if len({item.contact_id for item in candidates}) != len(candidates):
            raise ValueError("contactCandidates contains duplicate contact IDs")
        created = root.get("createdAtUtc")
        return LoadedHitterContacts(
            resolved,
            _sha256(resolved),
            created if isinstance(created, str) else None,
            candidates,
            expected_tracks_hash,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise HitterIdentificationInputError(f"invalid contacts {resolved}: {error}") from error


def _prepare_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise OutputWriteError(str(output), reason="path is not a directory")
    for name in (HITTERS_NAME, HITTER_DEBUG_NAME, HITTER_EVALUATION_NAME):
        artifact = output / name
        if artifact.exists():
            raise OutputWriteError(str(artifact), reason="hitter-identification output exists")
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
    identifications: tuple[HitterIdentification, ...],
) -> None:
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    writer = _open_writer(temporary, source)
    by_frame = {item.frame: item for item in identifications}
    recent_frames = max(1, round(0.20 * source.fps))
    last: HitterIdentification | None = None
    processed = 0
    logger = logging.getLogger("pickleball_vision.hitter_identification")
    try:
        for decoded in iter_video_frames(video_path):
            identification = by_frame.get(decoded.frame_index)
            if identification is not None:
                last = identification
            recent = (
                last
                if last is not None and 0 < decoded.frame_index - last.frame <= recent_frames
                else None
            )
            writer.write(
                render_hitter_identification_frame(
                    decoded.image,
                    frame_number=decoded.frame_index,
                    identification=identification,
                    recent_identification=recent,
                )
            )
            processed += 1
            if processed % 1000 == 0:
                logger.info(
                    "hitter_debug_progress",
                    extra={"context": {"processed_frames": processed}},
                )
    except Exception:
        writer.abort()
        temporary.unlink(missing_ok=True)
        raise
    writer.release()
    if processed != source.frame_count:
        temporary.unlink(missing_ok=True)
        raise HitterIdentificationInputError(
            f"decoded {processed} frames but source metadata contains {source.frame_count}"
        )
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise OutputWriteError(str(output_path), reason="OpenCV wrote an empty debug video")
    temporary.replace(output_path)


def _statistics(identifications: tuple[HitterIdentification, ...]) -> dict[str, object]:
    assigned = tuple(item for item in identifications if item.player_id != UNKNOWN_PLAYER_ID)
    return {
        "contactCount": len(identifications),
        "eligibleVisualContactCount": sum(item.source_contact_eligible for item in identifications),
        "assignedCount": len(assigned),
        "unknownCount": len(identifications) - len(assigned),
        "assignmentCountsByPlayer": {
            player_id: sum(item.player_id == player_id for item in identifications)
            for player_id in LOGICAL_PLAYER_IDS
        },
        "assignmentCoverage": len(assigned) / len(identifications) if identifications else 0.0,
        "meanAssignedConfidence": (
            statistics.fmean(item.confidence for item in assigned) if assigned else None
        ),
    }


def identify_hitters_in_video(
    video_path: Path,
    *,
    contacts_path: Path,
    player_tracks_path: Path,
    output_dir: Path,
    settings: HitterIdentificationSettings,
    annotations_path: Path | None = None,
    evaluation_partition: str = "validation",
) -> HitterIdentificationArtifacts:
    """Validate immutable inputs, infer hitters, evaluate, and render a debug video."""

    media = inspect_media(video_path)
    source = media.video
    contacts = load_hitter_contacts(contacts_path, source=source)
    try:
        player_tracks = load_contact_player_tracks(player_tracks_path, source=source)
    except ContactDetectionInputError as error:
        raise HitterIdentificationInputError(str(error)) from error
    if contacts.expected_player_tracks_sha256 != player_tracks.sha256:
        raise HitterIdentificationInputError(
            "contacts were generated from different player-track artifact bytes"
        )
    source_hash = _sha256(source.path) if annotations_path is not None else None
    try:
        annotations = load_contact_annotations(
            annotations_path,
            source=source,
            source_sha256=source_hash,
        )
    except ContactDetectionInputError as error:
        raise HitterIdentificationInputError(str(error)) from error
    result = identify_hitters(
        contacts.candidates,
        frame_width_px=source.width,
        frame_height_px=source.height,
        settings=settings,
    )
    evaluation = (
        evaluate_hitters(
            result.identifications,
            annotations.contacts,
            settings=settings,
            evaluation_partition=evaluation_partition,
        )
        if annotations.contacts
        else unavailable_hitter_evaluation(evaluation_partition=evaluation_partition)
    )
    output = _prepare_output(output_dir)
    hitters_path = output / HITTERS_NAME
    debug_path = output / HITTER_DEBUG_NAME
    evaluation_path = output / HITTER_EVALUATION_NAME
    if debug_path == source.path:
        raise OutputWriteError(str(debug_path), reason="output would overwrite source video")
    _write_debug_video(
        source.path,
        debug_path,
        source=source,
        identifications=result.identifications,
    )
    created_at = datetime.now(UTC).isoformat()
    inputs = {
        "contacts": contacts.as_dict(),
        "playerTracks": {
            **player_tracks.as_dict(),
            "usage": "validated_logical_player_position_and_identity_provenance",
            "assignsHitter": False,
        },
        "groundTruthAnnotations": annotations.as_dict(),
    }
    statistics_payload = _statistics(result.identifications)
    _write_json(
        hitters_path,
        {
            "schemaVersion": HITTER_IDENTIFICATION_SCHEMA_VERSION,
            "recordType": "logical_hitter_identifications",
            "createdAtUtc": created_at,
            "source": media.as_dict(),
            "inputs": inputs,
            "configuration": settings.as_dict(),
            "contracts": {
                "sourceContactsMutated": False,
                "playerTracksMutated": False,
                "logicalIdentityIndependentOfTrackerId": True,
                "audioUsedForIdentity": False,
                "unknownSupported": True,
                "airborneBallProjectedThroughHomography": False,
                "shotClassificationImplemented": False,
            },
            "statistics": statistics_payload,
            "hitterIdentifications": [item.as_dict() for item in result.identifications],
        },
    )
    _write_json(
        evaluation_path,
        {
            "schemaVersion": HITTER_IDENTIFICATION_SCHEMA_VERSION,
            "recordType": "logical_hitter_identification_evaluation",
            "createdAtUtc": created_at,
            "source": media.as_dict(),
            "inputs": inputs,
            "configuration": {
                "evaluationToleranceMs": settings.evaluation_tolerance_ms,
                "minimumContactConfidence": settings.minimum_contact_confidence,
            },
            **evaluation,
            "artifacts": {
                "hitters": str(hitters_path),
                "debugVideo": str(debug_path),
                "evaluation": str(evaluation_path),
            },
        },
    )
    assigned_count = sum(item.player_id != UNKNOWN_PLAYER_ID for item in result.identifications)
    return HitterIdentificationArtifacts(
        hitters_path,
        debug_path,
        evaluation_path,
        len(result.identifications),
        assigned_count,
        len(result.identifications) - assigned_count,
    )
