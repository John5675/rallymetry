"""Artifact validation, persistence, evaluation, and debug video for shots."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import cv2

from pickleball_vision.bounce_detection_workflow import load_bounce_rallies
from pickleball_vision.config import ShotClassificationSettings
from pickleball_vision.contact_detection_workflow import load_contact_player_tracks
from pickleball_vision.court import CourtPoint, ImagePoint
from pickleball_vision.errors import (
    BounceDetectionInputError,
    ContactDetectionInputError,
    HitterIdentificationInputError,
    OutputWriteError,
    RallySegmentationInputError,
    ShotReconstructionInputError,
)
from pickleball_vision.hitter_identification import (
    LOGICAL_PLAYER_IDS,
    UNKNOWN_PLAYER_ID,
)
from pickleball_vision.hitter_identification_workflow import load_hitter_contacts
from pickleball_vision.match_annotation import (
    MATCH_ANNOTATION_RECORD_TYPE,
    MATCH_ANNOTATION_VERSION,
)
from pickleball_vision.media import MediaMetadata, inspect_media
from pickleball_vision.rally_segmentation import RallyBallFrame
from pickleball_vision.rally_segmentation_workflow import load_ball_trajectory
from pickleball_vision.shot_evaluation import (
    GroundTruthShot,
    evaluate_shots,
    unavailable_shot_evaluation,
)
from pickleball_vision.shot_reconstruction import (
    SHOT_RECONSTRUCTION_SCHEMA_VERSION,
    Shot,
    ShotBounce,
    ShotHitterDecision,
    ShotPlayerPosition,
    ShotRally,
    ShotType,
    reconstruct_shots,
)
from pickleball_vision.shot_reconstruction_render import render_shot_frame
from pickleball_vision.video import VideoMetadata, iter_video_frames

SHOTS_NAME = "shots.json"
SHOT_DEBUG_NAME = "shot-debug.mp4"
SHOT_EVALUATION_NAME = "shot-evaluation.json"
DEBUG_VIDEO_CODEC = "mp4v"


@dataclass(frozen=True, slots=True)
class LoadedShotBounces:
    requested: bool
    path: Path | None
    sha256: str | None
    bounces: tuple[ShotBounce, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "path": str(self.path) if self.path is not None else None,
            "sha256": self.sha256,
            "acceptedBounceCount": len(self.bounces),
            "usage": "first_accepted_bounce_and_plane_gated_landing_linkage",
            "mutated": False,
        }


@dataclass(frozen=True, slots=True)
class LoadedShotHitters:
    path: Path
    sha256: str
    decisions: dict[str, ShotHitterDecision]

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "decisionCount": len(self.decisions),
            "usage": "logical_hitter_identity_and_confidence",
            "mutated": False,
        }


@dataclass(frozen=True, slots=True)
class LoadedShotAnnotations:
    requested: bool
    path: Path | None
    sha256: str | None
    shots: tuple[GroundTruthShot, ...]
    unsupported_label_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "path": str(self.path) if self.path is not None else None,
            "sha256": self.sha256,
            "supportedShotLabelCount": len(self.shots),
            "unsupportedShotLabelCount": self.unsupported_label_count,
            "usedForInference": False,
            "usage": "post_inference_evaluation_only",
        }


@dataclass(frozen=True, slots=True)
class ShotReconstructionArtifacts:
    shots_path: Path
    debug_video_path: Path
    evaluation_path: Path
    shot_count: int
    unknown_count: int
    outside_rally_contact_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "shotsPath": str(self.shots_path),
            "debugVideoPath": str(self.debug_video_path),
            "evaluationPath": str(self.evaluation_path),
            "shotCount": self.shot_count,
            "unknownCount": self.unknown_count,
            "outsideRallyContactCount": self.outside_rally_contact_count,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ShotReconstructionInputError(f"unable to hash {path}: {error}") from error
    return digest.hexdigest()


def _read_root(path: Path, kind: str) -> dict[str, object]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), "root")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ShotReconstructionInputError(f"unable to load {kind} {path}: {error}") from error


def _validate_source(value: object, source: VideoMetadata, *, field: str = "source") -> None:
    payload = _object(value, field)
    path = Path(_string(payload.get("path"), f"{field}.path")).expanduser().resolve()
    if path != source.path:
        raise ValueError(f"{field} belongs to a different source video path")
    for key, expected in (
        ("width", source.width),
        ("height", source.height),
        ("frame_count", source.frame_count),
    ):
        if _integer(payload.get(key), f"{field}.{key}") != expected:
            raise ValueError(f"{field}.{key} does not match the video")
    if not math.isclose(_number(payload.get("fps"), f"{field}.fps"), source.fps, rel_tol=5e-3):
        raise ValueError(f"{field}.fps does not match the video")


def _expected_contact_ball_hash(path: Path) -> str:
    root = _read_root(path, "contacts")
    try:
        inputs = _object(root.get("inputs"), "inputs")
        ball = _object(inputs.get("ballTrajectory"), "inputs.ballTrajectory")
        return _string(ball.get("sha256"), "inputs.ballTrajectory.sha256")
    except (KeyError, TypeError, ValueError) as error:
        raise ShotReconstructionInputError(f"invalid contact provenance: {error}") from error


def _expected_rally_ball_hash(path: Path) -> str:
    root = _read_root(path, "rallies")
    try:
        inputs = _object(root.get("inputs"), "inputs")
        ball = _object(inputs.get("ballTrajectory"), "inputs.ballTrajectory")
        return _string(ball.get("sha256"), "inputs.ballTrajectory.sha256")
    except (KeyError, TypeError, ValueError) as error:
        raise ShotReconstructionInputError(f"invalid rally provenance: {error}") from error


def load_shot_bounces(
    path: Path | None,
    *,
    source: VideoMetadata,
    expected_ball_sha256: str,
) -> LoadedShotBounces:
    if path is None:
        return LoadedShotBounces(False, None, None, ())
    resolved = path.expanduser().resolve()
    root = _read_root(resolved, "bounces")
    try:
        if root.get("schemaVersion") != 1 or root.get("recordType") != (
            "multimodal_bounce_candidates"
        ):
            raise ValueError("unsupported bounce schemaVersion or recordType")
        _validate_source(root.get("source"), source)
        inputs = _object(root.get("inputs"), "inputs")
        ball = _object(inputs.get("ballTrajectory"), "inputs.ballTrajectory")
        if _string(ball.get("sha256"), "inputs.ballTrajectory.sha256") != expected_ball_sha256:
            raise ValueError("bounces were generated from different ball-trajectory bytes")
        bounces: list[ShotBounce] = []
        for index, value in enumerate(_array(root.get("bounceCandidates"), "bounceCandidates")):
            field = f"bounceCandidates[{index}]"
            item = _object(value, field)
            if item.get("acceptedFused") is not True:
                continue
            frame = _integer(item.get("frame"), f"{field}.frame")
            if not 0 <= frame < source.frame_count:
                raise ValueError(f"{field}.frame is outside the source video")
            confidence = _number(item.get("fusedConfidence"), f"{field}.fusedConfidence")
            if not 0 <= confidence <= 1:
                raise ValueError(f"{field}.fusedConfidence must be in [0, 1]")
            court_value = item.get("courtPosition")
            court_position = None
            if court_value is not None:
                court = _object(court_value, f"{field}.courtPosition")
                court_position = CourtPoint(
                    _number(court.get("x_m"), f"{field}.courtPosition.x_m"),
                    _number(court.get("y_m"), f"{field}.courtPosition.y_m"),
                )
            bounces.append(
                ShotBounce(
                    _string(item.get("bounceId"), f"{field}.bounceId"),
                    frame,
                    _number(item.get("timestamp"), f"{field}.timestamp"),
                    court_position,
                    confidence,
                )
            )
        return LoadedShotBounces(True, resolved, _sha256(resolved), tuple(bounces))
    except (KeyError, TypeError, ValueError) as error:
        raise ShotReconstructionInputError(f"invalid bounces {resolved}: {error}") from error


def load_shot_hitters(
    path: Path,
    *,
    source: VideoMetadata,
    expected_contacts_sha256: str,
    expected_player_tracks_sha256: str,
    expected_contact_ids: set[str],
) -> LoadedShotHitters:
    resolved = path.expanduser().resolve()
    root = _read_root(resolved, "hitters")
    try:
        if root.get("schemaVersion") != 1 or root.get("recordType") != (
            "logical_hitter_identifications"
        ):
            raise ValueError("unsupported hitter schemaVersion or recordType")
        _validate_source(root.get("source"), source)
        inputs = _object(root.get("inputs"), "inputs")
        contacts = _object(inputs.get("contacts"), "inputs.contacts")
        tracks = _object(inputs.get("playerTracks"), "inputs.playerTracks")
        if _string(contacts.get("sha256"), "inputs.contacts.sha256") != expected_contacts_sha256:
            raise ValueError("hitters were generated from different contact bytes")
        if _string(tracks.get("sha256"), "inputs.playerTracks.sha256") != (
            expected_player_tracks_sha256
        ):
            raise ValueError("hitters were generated from different player-track bytes")
        decisions: dict[str, ShotHitterDecision] = {}
        for index, value in enumerate(
            _array(root.get("hitterIdentifications"), "hitterIdentifications")
        ):
            field = f"hitterIdentifications[{index}]"
            item = _object(value, field)
            contact_id = _string(item.get("contactId"), f"{field}.contactId")
            player_id = _string(item.get("playerId"), f"{field}.playerId")
            if player_id not in {*LOGICAL_PLAYER_IDS, UNKNOWN_PLAYER_ID}:
                raise ValueError(f"{field}.playerId is unsupported")
            confidence = _number(item.get("confidence"), f"{field}.confidence")
            if not 0 <= confidence <= 1:
                raise ValueError(f"{field}.confidence must be in [0, 1]")
            if contact_id in decisions:
                raise ValueError(f"duplicate hitter contact ID {contact_id}")
            decisions[contact_id] = ShotHitterDecision(contact_id, player_id, confidence)
        if set(decisions) != expected_contact_ids:
            raise ValueError("hitter decisions must correspond exactly to source contacts")
        return LoadedShotHitters(resolved, _sha256(resolved), decisions)
    except (KeyError, TypeError, ValueError) as error:
        raise ShotReconstructionInputError(f"invalid hitters {resolved}: {error}") from error


def load_shot_player_positions(
    path: Path,
    *,
    source: VideoMetadata,
) -> tuple[str, tuple[dict[str, ShotPlayerPosition], ...]]:
    try:
        validated = load_contact_player_tracks(path, source=source)
    except ContactDetectionInputError as error:
        raise ShotReconstructionInputError(str(error)) from error
    root = _read_root(validated.path, "player tracks")
    try:
        layer = _object(root.get("logical_identity_layer"), "logical_identity_layer")
        by_frame: list[dict[str, ShotPlayerPosition]] = [{} for _ in range(source.frame_count)]
        for role in LOGICAL_PLAYER_IDS:
            records = _array(layer.get(role), f"logical_identity_layer.{role}")
            if len(records) != source.frame_count:
                raise ValueError(f"logical_identity_layer.{role} must contain every frame")
            for expected_frame, value in enumerate(records):
                field = f"logical_identity_layer.{role}[{expected_frame}]"
                record = _object(value, field)
                if _integer(record.get("frame_number"), f"{field}.frame_number") != expected_frame:
                    raise ValueError(f"{field}.frame_number is out of sequence")
                ground_value = record.get("ground_contact")
                if ground_value is None:
                    continue
                ground = _object(ground_value, f"{field}.ground_contact")
                if ground.get("method") != "bounding_box_bottom_center":
                    raise ValueError(f"{field}.ground_contact method is unsupported")
                image = _object(ground.get("image_point"), f"{field}.ground_contact.image_point")
                image_point = ImagePoint(
                    _number(image.get("x_px"), f"{field}.ground_contact.image_point.x_px"),
                    _number(image.get("y_px"), f"{field}.ground_contact.image_point.y_px"),
                )
                court_value = ground.get("court_point")
                court_point = None
                if court_value is not None:
                    court = _object(court_value, f"{field}.ground_contact.court_point")
                    court_point = CourtPoint(
                        _number(court.get("x_m"), f"{field}.ground_contact.court_point.x_m"),
                        _number(court.get("y_m"), f"{field}.ground_contact.court_point.y_m"),
                    )
                confidence = _number(
                    record.get("tracking_confidence"), f"{field}.tracking_confidence"
                )
                if not 0 <= confidence <= 1:
                    raise ValueError(f"{field}.tracking_confidence must be in [0, 1]")
                by_frame[expected_frame][role] = ShotPlayerPosition(
                    role,
                    image_point,
                    court_point,
                    confidence,
                    _string(record.get("tracking_state"), f"{field}.tracking_state"),
                    _optional_string(
                        ground.get("court_side"), f"{field}.ground_contact.court_side"
                    ),
                    _optional_string(
                        ground.get("court_region"), f"{field}.ground_contact.court_region"
                    ),
                )
        return validated.sha256, tuple(by_frame)
    except (KeyError, TypeError, ValueError) as error:
        raise ShotReconstructionInputError(
            f"invalid player positions {validated.path}: {error}"
        ) from error


def load_shot_annotations(
    path: Path | None,
    *,
    media: MediaMetadata,
    source_sha256: str | None,
) -> LoadedShotAnnotations:
    if path is None:
        return LoadedShotAnnotations(False, None, None, (), 0)
    resolved = path.expanduser().resolve()
    root = _read_root(resolved, "match annotations")
    try:
        if (
            root.get("annotationVersion") != MATCH_ANNOTATION_VERSION
            or root.get("recordType") != MATCH_ANNOTATION_RECORD_TYPE
        ):
            raise ValueError("unsupported annotationVersion or recordType")
        video = _object(root.get("video"), "video")
        _validate_source(video, media.video, field="video")
        if source_sha256 is not None and video.get("contentSha256") != source_sha256:
            raise ValueError("annotations belong to different source-media bytes")
        by_frame: dict[int, tuple[int, GroundTruthShot]] = {}
        unsupported = 0
        for index, value in enumerate(_array(root.get("events"), "events")):
            field = f"events[{index}]"
            event = _object(value, field)
            event_type = _string(event.get("type"), f"{field}.type")
            if event_type not in {"SHOT_TYPE", "SERVE_CONTACT", "PADDLE_CONTACT"}:
                continue
            raw_type = event.get("shotType")
            if raw_type is None:
                continue
            try:
                shot_type = ShotType(_string(raw_type, f"{field}.shotType").upper())
            except ValueError:
                unsupported += 1
                continue
            frame = _integer(event.get("frame"), f"{field}.frame")
            if not 0 <= frame < media.video.frame_count:
                raise ValueError(f"{field}.frame is outside the source video")
            priority = 2 if event_type == "SHOT_TYPE" else 1
            candidate = GroundTruthShot(
                _string(event.get("id"), f"{field}.id"),
                frame,
                frame / media.video.fps,
                shot_type,
            )
            existing = by_frame.get(frame)
            if existing is None or priority > existing[0]:
                by_frame[frame] = (priority, candidate)
        return LoadedShotAnnotations(
            True,
            resolved,
            _sha256(resolved),
            tuple(item[1] for item in sorted(by_frame.values(), key=lambda value: value[1].frame)),
            unsupported,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ShotReconstructionInputError(f"invalid annotations {resolved}: {error}") from error


def _prepare_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise OutputWriteError(str(output), reason="path is not a directory")
    for name in (SHOTS_NAME, SHOT_DEBUG_NAME, SHOT_EVALUATION_NAME):
        artifact = output / name
        if artifact.exists():
            raise OutputWriteError(str(artifact), reason="shot-reconstruction output exists")
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


def _open_writer(path: Path, source: VideoMetadata) -> cv2.VideoWriter:
    try:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter.fourcc(*DEBUG_VIDEO_CODEC),
            source.fps,
            (source.width, source.height),
        )
    except cv2.error as error:
        raise OutputWriteError(str(path), reason=str(error)) from error
    if not writer.isOpened():
        writer.release()
        raise OutputWriteError(str(path), reason="OpenCV MP4 writer could not be opened")
    return writer


def _write_debug_video(
    video_path: Path,
    output_path: Path,
    *,
    source: VideoMetadata,
    shots: tuple[Shot, ...],
    trajectory_frames: tuple[RallyBallFrame, ...],
    trail_seconds: float,
) -> None:
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    writer = _open_writer(temporary, source)
    by_frame = {item.contact_frame: item for item in shots}
    recent_frames = max(1, round(0.35 * source.fps))
    trail_frames = max(1, round(trail_seconds * source.fps))
    last: Shot | None = None
    processed = 0
    logger = logging.getLogger("pickleball_vision.shot_reconstruction")
    try:
        for decoded in iter_video_frames(video_path):
            trajectory_point = trajectory_frames[decoded.frame_index]
            trail_start = max(0, decoded.frame_index - trail_frames + 1)
            shot = by_frame.get(decoded.frame_index)
            if shot is not None:
                last = shot
            recent = (
                last
                if last is not None
                and 0 < decoded.frame_index - last.contact_frame <= recent_frames
                else None
            )
            writer.write(
                render_shot_frame(
                    decoded.image,
                    frame_number=decoded.frame_index,
                    shot=shot,
                    recent_shot=recent,
                    trajectory_point=trajectory_point,
                    recent_trail=trajectory_frames[trail_start : decoded.frame_index + 1],
                )
            )
            processed += 1
            if processed % 1000 == 0:
                logger.info(
                    "shot_debug_progress",
                    extra={"context": {"processed_frames": processed}},
                )
    except Exception:
        writer.release()
        temporary.unlink(missing_ok=True)
        raise
    writer.release()
    if processed != source.frame_count:
        temporary.unlink(missing_ok=True)
        raise ShotReconstructionInputError(
            f"decoded {processed} frames but source metadata contains {source.frame_count}"
        )
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise OutputWriteError(str(output_path), reason="OpenCV wrote an empty debug video")
    temporary.replace(output_path)


def _shot_statistics(shots: tuple[Shot, ...]) -> dict[str, object]:
    return {
        "shotCount": len(shots),
        "unknownCount": sum(item.shot_type is ShotType.UNKNOWN for item in shots),
        "unknownRate": (
            sum(item.shot_type is ShotType.UNKNOWN for item in shots) / len(shots) if shots else 0.0
        ),
        "shotTypeCounts": {
            shot_type.value: sum(item.shot_type is shot_type for item in shots)
            for shot_type in ShotType
        },
        "shotWithBounceCount": sum(item.bounce_id is not None for item in shots),
        "shotWithLandingCourtPositionCount": sum(
            item.landing_court_position is not None for item in shots
        ),
        "shotWithHitterCourtPositionCount": sum(
            item.hitter_court_position is not None
            and item.hitter_court_position.court_point is not None
            for item in shots
        ),
    }


def reconstruct_shots_in_video(
    video_path: Path,
    *,
    ball_tracks_path: Path,
    rallies_path: Path,
    contacts_path: Path,
    hitters_path: Path,
    player_tracks_path: Path,
    output_dir: Path,
    settings: ShotClassificationSettings,
    bounces_path: Path | None = None,
    annotations_path: Path | None = None,
    evaluation_partition: str = "validation",
) -> ShotReconstructionArtifacts:
    """Validate current domain artifacts, reconstruct shots, evaluate, and render."""

    media = inspect_media(video_path)
    source = media.video
    try:
        ball = load_ball_trajectory(ball_tracks_path, source=source)
    except RallySegmentationInputError as error:
        raise ShotReconstructionInputError(str(error)) from error
    try:
        contacts = load_hitter_contacts(contacts_path, source=source)
    except HitterIdentificationInputError as error:
        raise ShotReconstructionInputError(str(error)) from error
    if _expected_contact_ball_hash(contacts.path) != ball.sha256:
        raise ShotReconstructionInputError(
            "contacts were generated from different ball-trajectory bytes"
        )
    player_tracks_sha256, player_positions = load_shot_player_positions(
        player_tracks_path,
        source=source,
    )
    if contacts.expected_player_tracks_sha256 != player_tracks_sha256:
        raise ShotReconstructionInputError(
            "contacts were generated from different player-track bytes"
        )
    hitters = load_shot_hitters(
        hitters_path,
        source=source,
        expected_contacts_sha256=contacts.sha256,
        expected_player_tracks_sha256=player_tracks_sha256,
        expected_contact_ids={item.contact_id for item in contacts.candidates},
    )
    try:
        loaded_rallies = load_bounce_rallies(rallies_path, source=source)
    except BounceDetectionInputError as error:
        raise ShotReconstructionInputError(str(error)) from error
    if loaded_rallies.path is None or _expected_rally_ball_hash(loaded_rallies.path) != ball.sha256:
        raise ShotReconstructionInputError(
            "rallies were generated from different ball-trajectory bytes"
        )
    rallies = tuple(
        ShotRally(item.rally_id, item.start_frame, item.end_frame, item.confidence)
        for item in loaded_rallies.intervals
    )
    bounces = load_shot_bounces(
        bounces_path,
        source=source,
        expected_ball_sha256=ball.sha256,
    )
    source_hash = _sha256(source.path) if annotations_path is not None else None
    annotations = load_shot_annotations(
        annotations_path,
        media=media,
        source_sha256=source_hash,
    )
    try:
        result = reconstruct_shots(
            frames=ball.frames,
            rallies=rallies,
            contacts=contacts.candidates,
            bounces=bounces.bounces,
            hitters_by_contact=hitters.decisions,
            player_positions_by_frame=player_positions,
            frame_width_px=source.width,
            frame_height_px=source.height,
            settings=settings,
        )
    except ValueError as error:
        raise ShotReconstructionInputError(
            f"invalid shot reconstruction inputs: {error}"
        ) from error
    evaluation = (
        evaluate_shots(
            result.shots,
            annotations.shots,
            settings=settings,
            evaluation_partition=evaluation_partition,
            unsupported_human_label_count=annotations.unsupported_label_count,
        )
        if annotations.shots
        else unavailable_shot_evaluation(
            evaluation_partition=evaluation_partition,
            unsupported_human_label_count=annotations.unsupported_label_count,
        )
    )
    output = _prepare_output(output_dir)
    shots_path = output / SHOTS_NAME
    debug_path = output / SHOT_DEBUG_NAME
    evaluation_path = output / SHOT_EVALUATION_NAME
    if debug_path == source.path:
        raise OutputWriteError(str(debug_path), reason="output would overwrite source video")
    _write_debug_video(
        source.path,
        debug_path,
        source=source,
        shots=result.shots,
        trajectory_frames=ball.frames,
        trail_seconds=settings.debug_trail_seconds,
    )
    created_at = datetime.now(UTC).isoformat()
    inputs = {
        "ballTrajectory": {
            "path": str(ball.path),
            "sha256": ball.sha256,
            "createdAtUtc": ball.created_at_utc,
            "mutated": False,
        },
        "rallies": {
            "path": str(loaded_rallies.path),
            "sha256": loaded_rallies.sha256,
            "rallyCount": len(rallies),
            "mutated": False,
        },
        "contacts": contacts.as_dict(),
        "bounces": bounces.as_dict(),
        "hitters": hitters.as_dict(),
        "playerTracks": {
            "path": str(Path(player_tracks_path).expanduser().resolve()),
            "sha256": player_tracks_sha256,
            "usage": "raw_bottom_center_hitter_and_opponent_ground_positions",
            "mutated": False,
        },
        "groundTruthAnnotations": annotations.as_dict(),
    }
    statistics_payload = {
        **_shot_statistics(result.shots),
        "acceptedSourceContactCount": result.accepted_contact_count,
        "acceptedContactOutsideRallyCount": result.contact_outside_rally_count,
    }
    _write_json(
        shots_path,
        {
            "schemaVersion": SHOT_RECONSTRUCTION_SCHEMA_VERSION,
            "recordType": "reconstructed_classified_shots",
            "createdAtUtc": created_at,
            "source": media.as_dict(),
            "inputs": inputs,
            "configuration": settings.as_dict(),
            "supportedShotTypes": [item.value for item in ShotType],
            "contracts": {
                "ruleBasedClassification": True,
                "newNeuralNetworkUsed": False,
                "unknownSupported": True,
                "airborneBallProjectedThroughHomography": False,
                "landingRequiresAcceptedPlaneGatedBounce": True,
                "playerPhysicalPositionUsesBottomCenterGroundPoint": True,
                "sourceArtifactsMutated": False,
                "aiCoachingImplemented": False,
            },
            "statistics": statistics_payload,
            "shots": [item.as_dict() for item in result.shots],
        },
    )
    _write_json(
        evaluation_path,
        {
            "schemaVersion": SHOT_RECONSTRUCTION_SCHEMA_VERSION,
            "recordType": "shot_classification_evaluation",
            "createdAtUtc": created_at,
            "source": media.as_dict(),
            "inputs": inputs,
            "configuration": {
                "evaluationToleranceMs": settings.evaluation_tolerance_ms,
                "supportedShotTypes": [item.value for item in ShotType],
            },
            **evaluation,
            "artifacts": {
                "shots": str(shots_path),
                "debugVideo": str(debug_path),
                "evaluation": str(evaluation_path),
            },
        },
    )
    unknown_count = sum(item.shot_type is ShotType.UNKNOWN for item in result.shots)
    return ShotReconstructionArtifacts(
        shots_path,
        debug_path,
        evaluation_path,
        len(result.shots),
        unknown_count,
        result.contact_outside_rally_count,
    )
