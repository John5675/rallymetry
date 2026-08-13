"""End-to-end primary-player candidate isolation and manual role assignment."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import cv2

from pickleball_vision.calibration import load_calibration
from pickleball_vision.config import PlayerIsolationSettings
from pickleball_vision.errors import OutputWriteError, PlayerIsolationInputError
from pickleball_vision.person_detection import load_person_detection_run
from pickleball_vision.player_isolation import (
    CandidateObservation,
    LogicalPlayerAssignments,
    LogicalPlayerRole,
    PlayerCandidateCollection,
    assignment_selections_for_candidates,
    build_logical_player_assignments,
    build_player_candidates,
    load_player_assignments,
    save_player_assignments,
)
from pickleball_vision.player_isolation_render import render_player_isolation_frame
from pickleball_vision.player_isolation_ui import select_logical_players
from pickleball_vision.video import VideoMetadata, inspect_video, iter_video_frames

CANDIDATES_NAME = "player-candidates.json"
ASSIGNMENTS_NAME = "player-assignments.json"
DEBUG_VIDEO_NAME = "primary-player-debug.mp4"
SUMMARY_NAME = "primary-player-summary.json"
ANNOTATED_CODEC = "mp4v"


@dataclass(frozen=True, slots=True)
class PlayerIsolationArtifacts:
    """Artifacts produced by a completed primary-player isolation workflow."""

    candidates_path: Path
    assignments_path: Path
    debug_video_path: Path
    summary_path: Path
    candidate_count: int
    eligible_candidate_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "candidates_path": str(self.candidates_path),
            "assignments_path": str(self.assignments_path),
            "debug_video_path": str(self.debug_video_path),
            "summary_path": str(self.summary_path),
            "candidate_count": self.candidate_count,
            "eligible_candidate_count": self.eligible_candidate_count,
        }


def _prepare_output_dir(path: Path) -> Path:
    output_dir = path.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise OutputWriteError(str(output_dir), reason="path is not a directory")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OutputWriteError(str(output_dir), reason=str(error)) from error
    return output_dir


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


def _validate_video_provenance(
    metadata: VideoMetadata,
    detection_source: VideoMetadata,
) -> None:
    if metadata.path != detection_source.path.expanduser().resolve():
        raise PlayerIsolationInputError(
            "the video path differs from the source path in detections.json; rerun detection "
            "for this local video"
        )
    if (
        metadata.width != detection_source.width
        or metadata.height != detection_source.height
        or metadata.frame_count != detection_source.frame_count
        or not math.isclose(metadata.fps, detection_source.fps, rel_tol=1e-6)
    ):
        raise PlayerIsolationInputError("video metadata does not match detections.json provenance")


def _open_debug_writer(path: Path, metadata: VideoMetadata) -> cv2.VideoWriter:
    try:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter.fourcc(*ANNOTATED_CODEC),
            metadata.fps,
            (metadata.width, metadata.height),
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
    candidates: PlayerCandidateCollection,
    assignments: dict[LogicalPlayerRole, CandidateObservation],
    calibration_path: Path,
) -> None:
    calibration = load_calibration(calibration_path)
    observations_by_frame: dict[int, list[CandidateObservation]] = defaultdict(list)
    for observation in candidates.observations:
        observations_by_frame[observation.detection.frame_number].append(observation)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates.candidates}
    metadata = inspect_video(video_path)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp.mp4")
    writer = _open_debug_writer(temporary_path, metadata)
    logger = logging.getLogger("pickleball_vision.player_isolation")
    try:
        for decoded in iter_video_frames(video_path):
            annotated = render_player_isolation_frame(
                decoded.image,
                observations=tuple(observations_by_frame.get(decoded.frame_index, ())),
                candidates=candidate_by_id,
                logical_assignments=assignments,
                calibration=calibration,
            )
            writer.write(annotated)
            if (decoded.frame_index + 1) % 300 == 0:
                logger.info(
                    "player_isolation_debug_progress",
                    extra={
                        "context": {
                            "processed_frames": decoded.frame_index + 1,
                            "source_frames": metadata.frame_count,
                        }
                    },
                )
    except Exception:
        writer.release()
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)
        raise
    else:
        writer.release()
    if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
        raise OutputWriteError(str(output_path), reason="OpenCV wrote an empty debug video")
    try:
        temporary_path.replace(output_path)
    except OSError as error:
        raise OutputWriteError(str(output_path), reason=str(error)) from error


def _summary(
    candidates: PlayerCandidateCollection,
    assignments: LogicalPlayerAssignments,
    *,
    output_dir: Path,
) -> dict[str, object]:
    region_counts = Counter(
        observation.ground_contact.region_state.value for observation in candidates.observations
    )
    side_counts = Counter(
        observation.ground_contact.side.value for observation in candidates.observations
    )
    return {
        "schema_version": 1,
        "record_type": "primary_player_isolation_summary",
        "source": candidates.source.as_dict(),
        "statistics": {
            "raw_detection_count": len(candidates.observations),
            "candidate_count": len(candidates.candidates),
            "eligible_candidate_count": sum(
                candidate.eligible for candidate in candidates.candidates
            ),
            "court_region_observations": dict(region_counts),
            "court_side_observations": dict(side_counts),
            "logical_player_assignments": len(assignments.assignments),
        },
        "artifacts": {
            "candidates": str(output_dir / CANDIDATES_NAME),
            "assignments": str(output_dir / ASSIGNMENTS_NAME),
            "debug_video": str(output_dir / DEBUG_VIDEO_NAME),
            "summary": str(output_dir / SUMMARY_NAME),
        },
    }


def isolate_primary_players(
    video_path: Path,
    *,
    detections_path: Path,
    calibration_path: Path,
    selection_timestamp_s: float,
    output_dir: Path,
    settings: PlayerIsolationSettings,
    existing_assignments_path: Path | None = None,
) -> PlayerIsolationArtifacts:
    """Derive candidates, collect four manual roles, and render inspectable output."""

    metadata = inspect_video(video_path)
    if (
        not math.isfinite(selection_timestamp_s)
        or selection_timestamp_s < 0
        or selection_timestamp_s >= metadata.duration
    ):
        raise PlayerIsolationInputError(
            f"selection timestamp must be in [0, {metadata.duration:.6f}) seconds"
        )
    detections = load_person_detection_run(detections_path)
    _validate_video_provenance(metadata, detections.source)
    calibration = load_calibration(calibration_path)
    if calibration.source.video_path.expanduser().resolve() != metadata.path:
        raise PlayerIsolationInputError("the calibration was created from a different source video")
    candidates = build_player_candidates(
        detections,
        calibration=calibration,
        detections_path=detections_path,
        calibration_path=calibration_path,
        settings=settings,
    )
    resolved_output_dir = _prepare_output_dir(output_dir)
    debug_video_path = resolved_output_dir / DEBUG_VIDEO_NAME
    if debug_video_path == metadata.path:
        raise OutputWriteError(
            str(debug_video_path),
            reason="debug output would overwrite the source video",
        )
    candidates_path = resolved_output_dir / CANDIDATES_NAME
    _write_json(candidates_path, candidates.as_dict())

    existing_selections: dict[LogicalPlayerRole, CandidateObservation] | None = None
    if existing_assignments_path is not None:
        existing_assignments = load_player_assignments(existing_assignments_path)
        existing_selections = assignment_selections_for_candidates(
            existing_assignments,
            candidates,
        )
    initial_frame_number = min(int(selection_timestamp_s * metadata.fps), metadata.frame_count - 1)
    selections = select_logical_players(
        metadata.path,
        candidates=candidates,
        calibration=calibration,
        initial_frame_number=initial_frame_number,
        existing=existing_selections,
    )
    assignments = build_logical_player_assignments(
        selections,
        candidates_path=candidates_path,
        detections_path=detections_path,
        corrected_from_path=existing_assignments_path,
    )
    assignments_path = resolved_output_dir / ASSIGNMENTS_NAME
    save_player_assignments(assignments, assignments_path)
    _write_debug_video(
        metadata.path,
        debug_video_path,
        candidates=candidates,
        assignments=selections,
        calibration_path=calibration_path,
    )
    summary_path = resolved_output_dir / SUMMARY_NAME
    _write_json(summary_path, _summary(candidates, assignments, output_dir=resolved_output_dir))
    return PlayerIsolationArtifacts(
        candidates_path=candidates_path,
        assignments_path=assignments_path,
        debug_video_path=debug_video_path,
        summary_path=summary_path,
        candidate_count=len(candidates.candidates),
        eligible_candidate_count=sum(candidate.eligible for candidate in candidates.candidates),
    )
