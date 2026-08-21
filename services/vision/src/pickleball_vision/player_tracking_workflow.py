"""Persistent player tracking workflow over existing detection and assignment artifacts."""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path

from pickleball_vision.calibration import load_calibration
from pickleball_vision.config import PlayerIsolationSettings, PlayerTrackingSettings
from pickleball_vision.errors import OutputWriteError, PlayerTrackingInputError
from pickleball_vision.person_detection import PersonDetection, load_person_detection_run
from pickleball_vision.player_appearance import (
    appearance_similarity,
    build_appearance_prototypes,
    extract_tracker_appearance,
)
from pickleball_vision.player_isolation import (
    LOGICAL_PLAYER_ROLES,
    LogicalPlayerAssignments,
    LogicalPlayerRole,
    assess_ground_contact,
    load_player_assignments,
)
from pickleball_vision.player_tracking import (
    IndexedDetection,
    LogicalPlayerFrame,
    MultiObjectTracker,
    RawTrackerObservation,
    build_tracking_run,
    resolve_logical_player_tracks,
    tracking_summary,
)
from pickleball_vision.player_tracking_render import render_player_tracking_frame
from pickleball_vision.trackers import UltralyticsByteTracker
from pickleball_vision.video import VideoMetadata, inspect_video, iter_video_frames
from pickleball_vision.video_output import CompressedVideoWriter

TRACKS_NAME = "tracks.json"
ANNOTATED_VIDEO_NAME = "annotated.mp4"
SUMMARY_NAME = "tracking-summary.json"


@dataclass(frozen=True, slots=True)
class PlayerTrackingArtifacts:
    """Paths and counts returned by the completed tracking workflow."""

    tracks_path: Path
    annotated_video_path: Path
    summary_path: Path
    frames_processed: int
    raw_tracker_observation_count: int
    suspected_identity_switch_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "tracks_path": str(self.tracks_path),
            "annotated_video_path": str(self.annotated_video_path),
            "summary_path": str(self.summary_path),
            "frames_processed": self.frames_processed,
            "raw_tracker_observation_count": self.raw_tracker_observation_count,
            "suspected_identity_switch_count": self.suspected_identity_switch_count,
        }


def _prepare_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and not resolved.is_dir():
        raise OutputWriteError(str(resolved), reason="path is not a directory")
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OutputWriteError(str(resolved), reason=str(error)) from error
    return resolved


def _write_json(path: Path, value: object) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except (OSError, TypeError, ValueError) as error:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)
        raise OutputWriteError(str(path), reason=str(error)) from error


def _validate_video_provenance(metadata: VideoMetadata, source: VideoMetadata) -> None:
    if metadata.path != source.path.expanduser().resolve():
        raise PlayerTrackingInputError(
            "the video path differs from detections.json; rerun person detection for this video"
        )
    if (
        metadata.width != source.width
        or metadata.height != source.height
        or metadata.frame_count != source.frame_count
        or not math.isclose(metadata.fps, source.fps, rel_tol=1e-6)
    ):
        raise PlayerTrackingInputError("video metadata does not match detections.json")


def _group_detections(
    detections: tuple[PersonDetection, ...],
) -> dict[int, tuple[IndexedDetection, ...]]:
    grouped: dict[int, list[IndexedDetection]] = defaultdict(list)
    for index, detection in enumerate(detections):
        grouped[detection.frame_number].append(IndexedDetection(index, detection))
    return {frame: tuple(items) for frame, items in grouped.items()}


def _rebind_manual_anchors(
    assignments: LogicalPlayerAssignments,
    *,
    detections: tuple[PersonDetection, ...],
    calibration_path: Path,
    source: VideoMetadata,
    isolation_settings: PlayerIsolationSettings,
) -> LogicalPlayerAssignments:
    """Match portable manual image anchors to this run's immutable detections."""

    calibration = load_calibration(calibration_path)
    maximum_distance_px = math.hypot(source.width, source.height) * 0.08
    pairs: list[tuple[float, LogicalPlayerRole, int]] = []
    rebound_by_role = assignments.by_role()
    roles_without_image_anchor: set[LogicalPlayerRole] = set()
    for assignment in assignments.assignments:
        if assignment.anchor_image_point is None:
            roles_without_image_anchor.add(assignment.logical_player)
            continue
        for index, detection in enumerate(detections):
            if detection.frame_number != assignment.anchor_frame_number:
                continue
            ground = assess_ground_contact(
                detection,
                calibration=calibration,
                frame_height_px=source.height,
                settings=isolation_settings,
            )
            if ground.side is not assignment.observed_side:
                continue
            distance_px = math.hypot(
                ground.image_point.x_px - assignment.anchor_image_point.x_px,
                ground.image_point.y_px - assignment.anchor_image_point.y_px,
            )
            if distance_px <= maximum_distance_px:
                pairs.append((distance_px, assignment.logical_player, index))

    selected_roles: set[LogicalPlayerRole] = set()
    selected_indices: set[int] = set()
    for _, role, index in sorted(pairs):
        if role in selected_roles or index in selected_indices:
            continue
        detection = detections[index]
        rebound_by_role[role] = replace(
            rebound_by_role[role],
            anchor_detection_index=index,
            anchor_frame_number=detection.frame_number,
            anchor_timestamp_s=detection.timestamp_s,
        )
        selected_roles.add(role)
        selected_indices.add(index)

    for role in roles_without_image_anchor:
        assignment = rebound_by_role[role]
        index = assignment.anchor_detection_index
        if not 0 <= index < len(detections):
            continue
        detection = detections[index]
        if detection.frame_number != assignment.anchor_frame_number:
            continue
        if not math.isclose(
            detection.timestamp_s,
            assignment.anchor_timestamp_s,
            abs_tol=max(1 / source.fps, 1e-3),
        ):
            continue
        selected_roles.add(role)
        selected_indices.add(index)

    missing = set(LOGICAL_PLAYER_ROLES) - selected_roles
    if missing:
        labels = ", ".join(sorted(role.value for role in missing))
        raise PlayerTrackingInputError(
            "portable manual anchors could not be matched to fresh detections for: "
            f"{labels}; create or enrich player assignments from this recording"
        )
    return replace(
        assignments, assignments=tuple(rebound_by_role[role] for role in LOGICAL_PLAYER_ROLES)
    )


def _validate_calibration_metadata(calibration_path: Path, source: VideoMetadata) -> None:
    calibration = load_calibration(calibration_path)
    if (
        calibration.source.frame_width_px != source.width
        or calibration.source.frame_height_px != source.height
        or not math.isclose(calibration.source.fps, source.fps, rel_tol=5e-3)
    ):
        raise PlayerTrackingInputError("the calibration metadata does not match the video")


def _load_candidate_seed_indices(
    path: Path,
    *,
    assignments: LogicalPlayerAssignments,
    detections_path: Path,
    detection_count: int,
) -> dict[LogicalPlayerRole, tuple[int, ...]]:
    """Load ephemeral candidate observations only as soft manual-tracklet seeds."""

    resolved_path = path.expanduser().resolve()
    try:
        root = json.loads(resolved_path.read_text(encoding="utf-8"))
        if not isinstance(root, dict):
            raise ValueError("root must be an object")
        if (
            root.get("schema_version") != 1
            or root.get("record_type") != "primary_player_candidates"
        ):
            raise ValueError("unsupported primary-player candidate artifact")
        inputs = root.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError("inputs must be an object")
        recorded_detections = inputs.get("raw_person_detections")
        if not isinstance(recorded_detections, str):
            raise ValueError("inputs.raw_person_detections must be a path string")
        if Path(recorded_detections).expanduser().resolve() != detections_path:
            raise ValueError("candidate artifact refers to a different detections.json")
        raw_candidates = root.get("candidates")
        if not isinstance(raw_candidates, list):
            raise ValueError("candidates must be an array")
        wanted = {item.candidate_id: item.logical_player for item in assignments.assignments}
        result: dict[LogicalPlayerRole, tuple[int, ...]] = {}
        for value in raw_candidates:
            if not isinstance(value, dict):
                raise ValueError("each candidate must be an object")
            candidate_id = value.get("candidate_id")
            if not isinstance(candidate_id, str):
                raise ValueError("candidate_id must be a string")
            if candidate_id not in wanted:
                continue
            raw_observations = value.get("observations")
            if not isinstance(raw_observations, list):
                raise ValueError(f"candidate {candidate_id} observations must be an array")
            indices: list[int] = []
            for observation in raw_observations:
                if not isinstance(observation, dict):
                    raise ValueError("candidate observation must be an object")
                raw_detection = observation.get("raw_detection")
                if not isinstance(raw_detection, dict):
                    raise ValueError("candidate raw_detection must be an object")
                index = raw_detection.get("index")
                if isinstance(index, bool) or not isinstance(index, int):
                    raise ValueError("candidate raw detection index must be an integer")
                if not 0 <= index < detection_count:
                    raise ValueError("candidate raw detection index is out of range")
                indices.append(index)
            result[wanted[candidate_id]] = tuple(indices)
        missing = set(LOGICAL_PLAYER_ROLES) - set(result)
        if missing:
            labels = ", ".join(sorted(role.value for role in missing))
            raise ValueError(f"candidate observations are missing for: {labels}")
        return result
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise PlayerTrackingInputError(
            f"unable to load primary-player candidate seeds {resolved_path}: {error}"
        ) from error


def _load_player_names(
    path: Path | None,
) -> tuple[dict[LogicalPlayerRole, str], Path | None]:
    """Load optional user-owned display names without changing logical role IDs."""

    if path is None:
        return ({role: role.value for role in LOGICAL_PLAYER_ROLES}, None)
    resolved = path.expanduser().resolve()
    try:
        decoded = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("player names must be a JSON object")
        names: dict[LogicalPlayerRole, str] = {}
        for role in LOGICAL_PLAYER_ROLES:
            value = decoded.get(role.value)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{role.value} must have a nonempty display name")
            names[role] = value.strip()
        if len(set(names.values())) != len(names):
            raise ValueError("display names must be unique")
        return (names, resolved)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise PlayerTrackingInputError(
            f"unable to load player names {resolved}: {error}"
        ) from error


def _run_raw_tracker(
    *,
    source: VideoMetadata,
    detections_by_frame: dict[int, tuple[IndexedDetection, ...]],
    tracker: MultiObjectTracker,
) -> tuple[RawTrackerObservation, ...]:
    observations: list[RawTrackerObservation] = []
    logger = logging.getLogger("pickleball_vision.player_tracking")
    for frame_number in range(source.frame_count):
        observations.extend(
            tracker.update(
                frame_number=frame_number,
                timestamp_s=frame_number / source.fps,
                detections=detections_by_frame.get(frame_number, ()),
                frame_width_px=source.width,
                frame_height_px=source.height,
            )
        )
        if (frame_number + 1) % 1000 == 0:
            logger.info(
                "raw_tracking_progress",
                extra={"context": {"processed_frames": frame_number + 1}},
            )
    return tuple(observations)


def _open_writer(path: Path, source: VideoMetadata) -> CompressedVideoWriter:
    return CompressedVideoWriter(
        path,
        fps=source.fps,
        dimensions=(source.width, source.height),
    )


def _write_annotated_video(
    video_path: Path,
    output_path: Path,
    *,
    source: VideoMetadata,
    detections_by_frame: dict[int, tuple[IndexedDetection, ...]],
    logical_tracks: dict[LogicalPlayerRole, tuple[LogicalPlayerFrame, ...]],
    calibration_path: Path,
    player_names: dict[LogicalPlayerRole, str],
) -> None:
    calibration = load_calibration(calibration_path)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp.mp4")
    writer = _open_writer(temporary_path, source)
    logger = logging.getLogger("pickleball_vision.player_tracking")
    try:
        for decoded in iter_video_frames(video_path):
            logical_frames = tuple(
                logical_tracks[role][decoded.frame_index] for role in LOGICAL_PLAYER_ROLES
            )
            annotated = render_player_tracking_frame(
                decoded.image,
                raw_detections=tuple(
                    item.detection for item in detections_by_frame.get(decoded.frame_index, ())
                ),
                logical_frames=logical_frames,
                calibration=calibration,
                player_names=player_names,
            )
            writer.write(annotated)
            if (decoded.frame_index + 1) % 300 == 0:
                logger.info(
                    "tracking_overlay_progress",
                    extra={"context": {"processed_frames": decoded.frame_index + 1}},
                )
    except Exception:
        writer.abort()
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)
        raise
    else:
        writer.release()
    if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
        raise OutputWriteError(str(output_path), reason="OpenCV wrote an empty annotated video")
    try:
        temporary_path.replace(output_path)
    except OSError as error:
        raise OutputWriteError(str(output_path), reason=str(error)) from error


def track_players_in_video(
    video_path: Path,
    *,
    calibration_path: Path,
    output_dir: Path,
    tracking_settings: PlayerTrackingSettings,
    isolation_settings: PlayerIsolationSettings,
    detections_path: Path | None = None,
    assignments_path: Path | None = None,
    player_names_path: Path | None = None,
    allow_portable_profile: bool = False,
    tracker: MultiObjectTracker | None = None,
) -> PlayerTrackingArtifacts:
    """Track all people, then resolve the existing four logical player assignments."""

    source = inspect_video(video_path)
    resolved_output_dir = _prepare_output_dir(output_dir)
    resolved_assignments_path = (
        assignments_path.expanduser().resolve()
        if assignments_path is not None
        else resolved_output_dir.parent / "player-isolation" / "player-assignments.json"
    )
    assignments = load_player_assignments(resolved_assignments_path)
    default_player_names_path = resolved_assignments_path.parent / "player-names.json"
    effective_player_names_path = (
        player_names_path
        if player_names_path is not None
        else default_player_names_path
        if default_player_names_path.is_file()
        else None
    )
    player_names, resolved_player_names_path = _load_player_names(effective_player_names_path)
    resolved_candidates_path = Path(assignments.candidates_path).expanduser().resolve()
    resolved_detections_path = (
        detections_path.expanduser().resolve()
        if detections_path is not None
        else Path(assignments.detections_path).expanduser().resolve()
    )
    if (
        not allow_portable_profile
        and Path(assignments.detections_path).expanduser().resolve() != resolved_detections_path
    ):
        raise PlayerTrackingInputError(
            "the selected detections.json differs from the manual assignment provenance"
        )
    detection_run = load_person_detection_run(resolved_detections_path)
    _validate_video_provenance(source, detection_run.source)
    calibration = load_calibration(calibration_path)
    resolved_calibration_path = calibration_path.expanduser().resolve()
    if (
        not allow_portable_profile
        and calibration.source.video_path.expanduser().resolve() != source.path
    ):
        raise PlayerTrackingInputError("the calibration was created from a different video")
    _validate_calibration_metadata(resolved_calibration_path, source)
    if allow_portable_profile:
        assignments = _rebind_manual_anchors(
            assignments,
            detections=detection_run.detections,
            calibration_path=resolved_calibration_path,
            source=source,
            isolation_settings=isolation_settings,
        )

    tracks_path = resolved_output_dir / TRACKS_NAME
    annotated_path = resolved_output_dir / ANNOTATED_VIDEO_NAME
    summary_path = resolved_output_dir / SUMMARY_NAME
    if source.path in {tracks_path, annotated_path, summary_path}:
        raise OutputWriteError(
            str(source.path), reason="tracking output would overwrite source video"
        )

    detections_by_frame = _group_detections(detection_run.detections)
    candidate_seed_indices: dict[LogicalPlayerRole, tuple[int, ...]]
    if allow_portable_profile:
        candidate_seed_indices = {
            assignment.logical_player: (assignment.anchor_detection_index,)
            for assignment in assignments.assignments
        }
    else:
        candidate_seed_indices = _load_candidate_seed_indices(
            resolved_candidates_path,
            assignments=assignments,
            detections_path=resolved_detections_path,
            detection_count=len(detection_run.detections),
        )
    effective_tracker = tracker or UltralyticsByteTracker(tracking_settings, fps=source.fps)
    raw_observations = _run_raw_tracker(
        source=source,
        detections_by_frame=detections_by_frame,
        tracker=effective_tracker,
    )
    appearance_descriptors = extract_tracker_appearance(
        source.path,
        detections=detection_run.detections,
        raw_observations=raw_observations,
    )
    try:
        appearance_prototypes = build_appearance_prototypes(
            raw_observations=raw_observations,
            descriptors=appearance_descriptors,
            assignments=assignments,
            window_seconds=tracking_settings.appearance_prototype_window_seconds,
        )
    except ValueError as error:
        raise PlayerTrackingInputError(str(error)) from error
    appearance_similarities = {
        role: {
            observation_id: similarity
            for observation_id, descriptor in appearance_descriptors.items()
            if (
                similarity := appearance_similarity(
                    descriptor,
                    appearance_prototypes[role],
                )
            )
            is not None
        }
        for role in LOGICAL_PLAYER_ROLES
    }
    ground_by_observation_id = {
        observation.observation_id: assess_ground_contact(
            detection_run.detections[observation.raw_detection_index],
            calibration=calibration,
            frame_height_px=source.height,
            settings=isolation_settings,
        )
        for observation in raw_observations
    }
    try:
        logical_tracks, switch_events = resolve_logical_player_tracks(
            source=source,
            raw_observations=raw_observations,
            ground_by_observation_id=ground_by_observation_id,
            assignments=assignments,
            settings=tracking_settings,
            candidate_seed_indices=candidate_seed_indices,
            appearance_similarities=appearance_similarities,
        )
    except ValueError as error:
        raise PlayerTrackingInputError(str(error)) from error
    run = build_tracking_run(
        source=source,
        detections_path=str(resolved_detections_path),
        candidates_path=str(resolved_candidates_path),
        assignments_path=str(resolved_assignments_path),
        calibration_path=str(resolved_calibration_path),
        tracker=effective_tracker.metadata,
        configuration={
            "player_tracking": tracking_settings.as_dict(),
            "ground_contact_geometry": isolation_settings.as_dict(),
            "portable_profile_rebinding": allow_portable_profile,
            "candidate_seed_source": (
                "rebound_manual_image_anchors"
                if allow_portable_profile
                else "primary_player_candidate_tracklets"
            ),
        },
        appearance={
            "method": "two_band_hsv_clothing_histogram",
            "raw_descriptor_count": len(appearance_descriptors),
            "player_names_path": (
                str(resolved_player_names_path) if resolved_player_names_path is not None else None
            ),
            "prototypes": {
                role.value: appearance_prototypes[role].as_dict() for role in LOGICAL_PLAYER_ROLES
            },
        },
        player_names=player_names,
        raw_observations=raw_observations,
        logical_tracks=logical_tracks,
        suspected_identity_switches=switch_events,
    )
    _write_json(tracks_path, run.as_dict())
    _write_annotated_video(
        source.path,
        annotated_path,
        source=source,
        detections_by_frame=detections_by_frame,
        logical_tracks=logical_tracks,
        calibration_path=resolved_calibration_path,
        player_names=player_names,
    )
    artifacts = {
        "tracks": str(tracks_path),
        "annotated_video": str(annotated_path),
        "summary": str(summary_path),
    }
    _write_json(summary_path, tracking_summary(run, artifacts=artifacts))
    return PlayerTrackingArtifacts(
        tracks_path,
        annotated_path,
        summary_path,
        source.frame_count,
        len(raw_observations),
        len(switch_events),
    )
