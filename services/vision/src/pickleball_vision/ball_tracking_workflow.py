"""Workflow for reconstructing a primary-match ball path from raw detections."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import cv2

from pickleball_vision.ball_detection import BALL_DETECTION_SCHEMA_VERSION
from pickleball_vision.ball_tracking import (
    BALL_TRACKING_SCHEMA_VERSION,
    BallTrackingCandidate,
    BallTrajectoryPoint,
    CourtImageEnvelope,
    ball_box_center,
    build_court_image_envelope,
    reconstruct_ball_trajectory,
    trajectory_summary,
)
from pickleball_vision.ball_tracking_render import render_ball_tracking_frame
from pickleball_vision.calibration import CourtCalibration, load_calibration
from pickleball_vision.config import BallTrackingSettings
from pickleball_vision.errors import BallTrackingInputError, OutputWriteError
from pickleball_vision.person_detection import BoundingBox
from pickleball_vision.video import VideoMetadata, inspect_video, iter_video_frames

BALL_TRACKS_NAME = "ball_tracks.json"
BALL_DEBUG_VIDEO_NAME = "ball-debug.mp4"
BALL_TRACKING_SUMMARY_NAME = "ball-tracking-summary.json"
DEBUG_VIDEO_CODEC = "mp4v"


@dataclass(frozen=True, slots=True)
class LoadedBallDetectionCandidates:
    """Validated raw detector records with derived court relevance only."""

    source: VideoMetadata
    candidates_by_frame: tuple[tuple[BallTrackingCandidate, ...], ...]
    detector: dict[str, object]
    strategy: dict[str, object]
    raw_calibration: dict[str, object]
    created_at_utc: str


@dataclass(frozen=True, slots=True)
class BallTrackingArtifacts:
    """Paths and counts from a completed trajectory reconstruction run."""

    tracks_path: Path
    debug_video_path: Path
    summary_path: Path
    frames_processed: int
    observed_frames: int
    interpolated_frames: int
    unknown_frames: int
    rejected_candidate_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "tracks_path": str(self.tracks_path),
            "debug_video_path": str(self.debug_video_path),
            "summary_path": str(self.summary_path),
            "frames_processed": self.frames_processed,
            "observed_frames": self.observed_frames,
            "interpolated_frames": self.interpolated_frames,
            "unknown_frames": self.unknown_frames,
            "rejected_candidate_count": self.rejected_candidate_count,
        }


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return cast(list[object], value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field)


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _integer(value: object, field: str) -> int:
    result = _number(value, field)
    if not result.is_integer():
        raise ValueError(f"{field} must be an integer")
    return int(result)


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _source_metadata(value: object) -> VideoMetadata:
    source = _object(value, "source")
    metadata = VideoMetadata(
        filename=_string(source.get("filename"), "source.filename"),
        path=Path(_string(source.get("path"), "source.path")),
        width=_integer(source.get("width"), "source.width"),
        height=_integer(source.get("height"), "source.height"),
        fps=_number(source.get("fps"), "source.fps"),
        frame_count=_integer(source.get("frame_count"), "source.frame_count"),
        duration=_number(source.get("duration"), "source.duration"),
        codec=_optional_string(source.get("codec"), "source.codec"),
    )
    if (
        metadata.width < 1
        or metadata.height < 1
        or metadata.fps <= 0
        or metadata.frame_count < 1
        or metadata.duration <= 0
    ):
        raise ValueError("source dimensions, FPS, frame count, and duration must be positive")
    return metadata


def load_ball_detection_candidates(
    path: Path,
    *,
    envelope: CourtImageEnvelope,
) -> LoadedBallDetectionCandidates:
    """Load immutable frame-local ball candidates with strict source-pixel validation."""

    resolved = path.expanduser().resolve()
    try:
        root = _object(json.loads(resolved.read_text(encoding="utf-8")), "root")
        if _integer(root.get("schema_version"), "schema_version") != (
            BALL_DETECTION_SCHEMA_VERSION
        ):
            raise ValueError("unsupported raw ball-detection schema version")
        if _string(root.get("record_type"), "record_type") != "raw_pickleball_detections":
            raise ValueError("record_type must be raw_pickleball_detections")
        temporal = _object(root.get("temporal_processing"), "temporal_processing")
        if _boolean(temporal.get("tracking"), "temporal_processing.tracking"):
            raise ValueError("input must be raw frame-local detections, not an existing track")
        source = _source_metadata(root.get("source"))
        if envelope.frame_width_px != source.width or envelope.frame_height_px != source.height:
            raise ValueError("court envelope dimensions do not match raw detections")
        raw_frames = _array(root.get("frames"), "frames")
        if len(raw_frames) != source.frame_count:
            raise ValueError(
                f"frames must contain all {source.frame_count} source frames; "
                f"found {len(raw_frames)}"
            )
        seen_ids: set[str] = set()
        frames: list[tuple[BallTrackingCandidate, ...]] = []
        for expected_frame, value in enumerate(raw_frames):
            field = f"frames[{expected_frame}]"
            raw_frame = _object(value, field)
            frame_number = _integer(raw_frame.get("frame_number"), f"{field}.frame_number")
            if frame_number != expected_frame:
                raise ValueError(f"{field}.frame_number must be {expected_frame}")
            timestamp_s = _number(raw_frame.get("timestamp_s"), f"{field}.timestamp_s")
            expected_timestamp_s = frame_number / source.fps
            if not math.isclose(
                timestamp_s,
                expected_timestamp_s,
                rel_tol=1e-6,
                abs_tol=max(1e-6, 0.05 / source.fps),
            ):
                raise ValueError(f"{field}.timestamp_s does not match the source time base")
            detections = _array(raw_frame.get("detections"), f"{field}.detections")
            frame_candidates: list[BallTrackingCandidate] = []
            for index, detection_value in enumerate(detections):
                detection_field = f"{field}.detections[{index}]"
                raw_detection = _object(detection_value, detection_field)
                detection_id = _string(
                    raw_detection.get("detection_id"), f"{detection_field}.detection_id"
                )
                if detection_id in seen_ids:
                    raise ValueError(f"duplicate detection_id {detection_id!r}")
                seen_ids.add(detection_id)
                if (
                    _integer(raw_detection.get("frame_number"), f"{detection_field}.frame_number")
                    != frame_number
                ):
                    raise ValueError(f"{detection_field}.frame_number differs from its frame")
                detection_timestamp_s = _number(
                    raw_detection.get("timestamp_s"), f"{detection_field}.timestamp_s"
                )
                if not math.isclose(detection_timestamp_s, timestamp_s, abs_tol=1e-9):
                    raise ValueError(f"{detection_field}.timestamp_s differs from its frame")
                raw_box = _object(
                    raw_detection.get("bounding_box"), f"{detection_field}.bounding_box"
                )
                box = BoundingBox(
                    left_px=_number(
                        raw_box.get("left_px"), f"{detection_field}.bounding_box.left_px"
                    ),
                    top_px=_number(raw_box.get("top_px"), f"{detection_field}.bounding_box.top_px"),
                    right_px=_number(
                        raw_box.get("right_px"), f"{detection_field}.bounding_box.right_px"
                    ),
                    bottom_px=_number(
                        raw_box.get("bottom_px"), f"{detection_field}.bounding_box.bottom_px"
                    ),
                )
                if box.right_px > source.width + 1 or box.bottom_px > source.height + 1:
                    raise ValueError(f"{detection_field}.bounding_box exceeds source dimensions")
                confidence = _number(
                    raw_detection.get("confidence"), f"{detection_field}.confidence"
                )
                if not 0 <= confidence <= 1:
                    raise ValueError(f"{detection_field}.confidence must be in [0, 1]")
                image_point = ball_box_center(box)
                frame_candidates.append(
                    BallTrackingCandidate(
                        detection_id=detection_id,
                        frame_number=frame_number,
                        timestamp_s=timestamp_s,
                        bounding_box=box,
                        confidence=confidence,
                        image_point=image_point,
                        primary_court_relevance=envelope.relevance(image_point),
                    )
                )
            frames.append(tuple(frame_candidates))
        return LoadedBallDetectionCandidates(
            source=source,
            candidates_by_frame=tuple(frames),
            detector=_object(root.get("detector"), "detector"),
            strategy=_object(root.get("strategy"), "strategy"),
            raw_calibration=_object(root.get("calibration"), "calibration"),
            created_at_utc=_string(root.get("created_at_utc"), "created_at_utc"),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise BallTrackingInputError(
            f"unable to load raw detections {resolved}: {error}"
        ) from error


def _validate_video_provenance(metadata: VideoMetadata, source: VideoMetadata) -> None:
    if metadata.path != source.path.expanduser().resolve():
        raise BallTrackingInputError(
            "the video path differs from detections.json; rerun ball detection for this video"
        )
    if (
        metadata.width != source.width
        or metadata.height != source.height
        or metadata.frame_count != source.frame_count
        or not math.isclose(metadata.fps, source.fps, rel_tol=1e-6)
    ):
        raise BallTrackingInputError("video metadata does not match detections.json")


def _validate_calibration_provenance(
    calibration: CourtCalibration,
    metadata: VideoMetadata,
) -> None:
    if calibration.source.video_path.expanduser().resolve() != metadata.path:
        raise BallTrackingInputError("court calibration was created from a different video")
    if (
        calibration.source.frame_width_px != metadata.width
        or calibration.source.frame_height_px != metadata.height
        or not math.isclose(calibration.source.fps, metadata.fps, rel_tol=5e-3)
    ):
        raise BallTrackingInputError("court calibration metadata does not match the video")


def _prepare_output_dir(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise OutputWriteError(str(output), reason="path is not a directory")
    tracks_path = output / BALL_TRACKS_NAME
    if tracks_path.exists():
        raise OutputWriteError(str(tracks_path), reason="ball-tracking output already exists")
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise BallTrackingInputError(f"unable to hash input {path}: {error}") from error
    return digest.hexdigest()


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
    candidates_by_frame: Sequence[tuple[BallTrackingCandidate, ...]],
    trajectory_frames: Sequence[BallTrajectoryPoint],
    trail_frames: int,
) -> None:
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    writer = _open_writer(temporary, source)
    logger = logging.getLogger("pickleball_vision.ball_tracking")
    try:
        processed = 0
        for decoded in iter_video_frames(video_path):
            frame_number = decoded.frame_index
            start = max(0, frame_number - trail_frames + 1)
            writer.write(
                render_ball_tracking_frame(
                    decoded.image,
                    raw_candidates=candidates_by_frame[frame_number],
                    trajectory_point=trajectory_frames[frame_number],
                    recent_trail=trajectory_frames[start : frame_number + 1],
                )
            )
            processed += 1
            if processed % 1000 == 0:
                logger.info(
                    "ball_tracking_debug_progress",
                    extra={"context": {"processed_frames": processed}},
                )
    except Exception:
        writer.release()
        temporary.unlink(missing_ok=True)
        raise
    writer.release()
    if processed != source.frame_count:
        temporary.unlink(missing_ok=True)
        raise BallTrackingInputError(
            f"decoded {processed} video frames but detections contain {source.frame_count}"
        )
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise OutputWriteError(str(output_path), reason="OpenCV wrote an empty debug video")
    temporary.replace(output_path)


def track_ball_in_video(
    video_path: Path,
    *,
    detections_path: Path,
    calibration_path: Path,
    output_dir: Path,
    settings: BallTrackingSettings,
) -> BallTrackingArtifacts:
    """Reconstruct and render a conservative primary-match ball trajectory."""

    metadata = inspect_video(video_path)
    resolved_detections = detections_path.expanduser().resolve()
    resolved_calibration = calibration_path.expanduser().resolve()
    calibration = load_calibration(resolved_calibration)
    _validate_calibration_provenance(calibration, metadata)
    envelope = build_court_image_envelope(
        calibration,
        frame_width_px=metadata.width,
        frame_height_px=metadata.height,
        settings=settings,
    )
    loaded = load_ball_detection_candidates(resolved_detections, envelope=envelope)
    _validate_video_provenance(metadata, loaded.source)
    output = _prepare_output_dir(output_dir)
    tracks_path = output / BALL_TRACKS_NAME
    debug_path = output / BALL_DEBUG_VIDEO_NAME
    summary_path = output / BALL_TRACKING_SUMMARY_NAME
    if debug_path == metadata.path:
        raise OutputWriteError(str(debug_path), reason="output would overwrite source video")

    trajectory = reconstruct_ball_trajectory(
        loaded.candidates_by_frame,
        fps=metadata.fps,
        frame_width_px=metadata.width,
        frame_height_px=metadata.height,
        settings=settings,
    )
    statistics = trajectory_summary(trajectory, fps=metadata.fps)
    trail_frames = max(1, round(settings.debug_trail_seconds * metadata.fps))
    _write_debug_video(
        metadata.path,
        debug_path,
        source=metadata,
        candidates_by_frame=loaded.candidates_by_frame,
        trajectory_frames=trajectory.frames,
        trail_frames=trail_frames,
    )

    created_at_utc = datetime.now(UTC).isoformat()
    input_payload = {
        "raw_ball_detections": {
            "path": str(resolved_detections),
            "sha256": _sha256(resolved_detections),
            "created_at_utc": loaded.created_at_utc,
            "detector": loaded.detector,
            "strategy": loaded.strategy,
            "recorded_calibration": loaded.raw_calibration,
        },
        "court_calibration": {
            "path": str(resolved_calibration),
            "sha256": _sha256(resolved_calibration),
            "usage": "project_known_court_outline_to_image_relevance_envelope_only",
            "airborne_ball_homography_projection": False,
        },
    }
    _write_json(
        tracks_path,
        {
            "schema_version": BALL_TRACKING_SCHEMA_VERSION,
            "record_type": "primary_match_ball_trajectory",
            "created_at_utc": created_at_utc,
            "source": metadata.as_dict(),
            "inputs": input_payload,
            "configuration": settings.as_dict(),
            "coordinate_system": {
                "origin": "top_left",
                "x_axis": "right",
                "y_axis": "down",
                "unit": "source_frame_pixels",
                "court_coordinates": None,
            },
            "court_image_envelope": envelope.as_dict(),
            "status_contract": {
                "OBSERVED": "linked immutable raw detection; raw point retained",
                "INTERPOLATED": "short bounded gap between observations; no raw point",
                "UNKNOWN": "no defensible primary-match ball position",
            },
            "statistics": statistics,
            "frames": [frame.as_dict() for frame in trajectory.frames],
        },
    )
    _write_json(
        summary_path,
        {
            "schema_version": BALL_TRACKING_SCHEMA_VERSION,
            "record_type": "primary_match_ball_tracking_summary",
            "created_at_utc": created_at_utc,
            "source": metadata.as_dict(),
            "inputs": input_payload,
            "configuration": settings.as_dict(),
            "statistics": statistics,
            "artifacts": {
                "ball_tracks": str(tracks_path),
                "debug_video": str(debug_path),
                "summary": str(summary_path),
            },
        },
    )
    return BallTrackingArtifacts(
        tracks_path=tracks_path,
        debug_video_path=debug_path,
        summary_path=summary_path,
        frames_processed=metadata.frame_count,
        observed_frames=cast(int, statistics["observed_frames"]),
        interpolated_frames=cast(int, statistics["interpolated_frames"]),
        unknown_frames=cast(int, statistics["unknown_frames"]),
        rejected_candidate_count=trajectory.rejected_candidate_count,
    )
