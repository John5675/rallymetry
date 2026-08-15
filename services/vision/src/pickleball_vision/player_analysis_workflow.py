"""Release 0.1 player-position analysis workflow over persisted logical tracks."""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2

from pickleball_vision.calibration import load_calibration
from pickleball_vision.config import PlayerAnalysisSettings
from pickleball_vision.court import CourtPoint, ImagePoint
from pickleball_vision.errors import OutputWriteError, PlayerAnalysisInputError
from pickleball_vision.player_analysis import (
    ManualCourtPositionCorrection,
    PlayerPositionFrame,
    RawPlayerPositionFrame,
    build_player_analysis_run,
    build_player_analysis_summary,
    derive_player_positions,
)
from pickleball_vision.player_analysis_render import (
    TOPDOWN_HEIGHT_PX,
    TOPDOWN_WIDTH_PX,
    render_player_heatmap,
    render_source_analysis_frame,
    render_topdown_frame,
)
from pickleball_vision.player_isolation import (
    LOGICAL_PLAYER_ROLES,
    CourtRegionState,
    LogicalPlayerRole,
)
from pickleball_vision.player_tracking import LogicalTrackingState
from pickleball_vision.video import Image, VideoMetadata, inspect_video, iter_video_frames

POSITIONS_NAME = "player_positions.json"
SUMMARY_NAME = "summary.json"
ANNOTATED_VIDEO_NAME = "annotated.mp4"
TOPDOWN_VIDEO_NAME = "topdown.mp4"
VIDEO_CODEC = "mp4v"
POSITION_CORRECTIONS_NAME = "player-position-corrections.json"
MAXIMUM_MANUAL_CORRECTION_M = 0.50
HEATMAP_NAMES = {
    LogicalPlayerRole.ME: "heatmap-me.png",
    LogicalPlayerRole.PARTNER: "heatmap-partner.png",
    LogicalPlayerRole.OPPONENT_1: "heatmap-opponent-1.png",
    LogicalPlayerRole.OPPONENT_2: "heatmap-opponent-2.png",
}


@dataclass(frozen=True, slots=True)
class LoadedTrackingPositions:
    """Validated position-bearing subset of tracks.json."""

    source: VideoMetadata
    calibration_path: Path
    player_names: dict[LogicalPlayerRole, str]
    raw_tracks: dict[LogicalPlayerRole, tuple[RawPlayerPositionFrame, ...]]


@dataclass(frozen=True, slots=True)
class PlayerAnalysisArtifacts:
    """Release 0.1 artifact paths and processed-frame count."""

    positions_path: Path
    summary_path: Path
    annotated_video_path: Path
    topdown_video_path: Path
    heatmap_paths: dict[LogicalPlayerRole, Path]
    frames_processed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "positions_path": str(self.positions_path),
            "summary_path": str(self.summary_path),
            "annotated_video_path": str(self.annotated_video_path),
            "topdown_video_path": str(self.topdown_video_path),
            "heatmap_paths": {
                role.value: str(self.heatmap_paths[role]) for role in LOGICAL_PLAYER_ROLES
            },
            "frames_processed": self.frames_processed,
            "release_version": "0.1",
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


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field)


def _parse_source(value: object) -> VideoMetadata:
    source = _object(value, "source")
    path = Path(_string(source.get("path"), "source.path")).expanduser().resolve()
    width = _integer(source.get("width"), "source.width")
    height = _integer(source.get("height"), "source.height")
    fps = _finite_number(source.get("fps"), "source.fps")
    frame_count = _integer(source.get("frame_count"), "source.frame_count")
    duration = _finite_number(source.get("duration"), "source.duration")
    codec_value = source.get("codec")
    codec = None if codec_value is None else _string(codec_value, "source.codec")
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0 or duration <= 0:
        raise ValueError("source dimensions, FPS, frame count, and duration must be positive")
    return VideoMetadata(
        _string(source.get("filename"), "source.filename"),
        path,
        width,
        height,
        fps,
        frame_count,
        duration,
        codec,
    )


def _parse_image_point(value: object, field: str) -> ImagePoint:
    point = _object(value, field)
    return ImagePoint(
        _finite_number(point.get("x_px"), f"{field}.x_px"),
        _finite_number(point.get("y_px"), f"{field}.y_px"),
    )


def _parse_court_point(value: object, field: str) -> CourtPoint:
    point = _object(value, field)
    return CourtPoint(
        _finite_number(point.get("x_m"), f"{field}.x_m"),
        _finite_number(point.get("y_m"), f"{field}.y_m"),
    )


def _parse_role_frames(
    role: LogicalPlayerRole,
    value: object,
    *,
    source: VideoMetadata,
) -> tuple[RawPlayerPositionFrame, ...]:
    raw_frames = _array(value, f"logical_identity_layer.{role.value}")
    if len(raw_frames) != source.frame_count:
        raise ValueError(
            f"logical_identity_layer.{role.value} must contain exactly {source.frame_count} frames"
        )
    parsed: list[RawPlayerPositionFrame] = []
    for expected_frame, raw_value in enumerate(raw_frames):
        field = f"logical_identity_layer.{role.value}[{expected_frame}]"
        frame = _object(raw_value, field)
        frame_number = _integer(frame.get("frame_number"), f"{field}.frame_number")
        if frame_number != expected_frame:
            raise ValueError(f"{field}.frame_number must be {expected_frame}")
        timestamp_s = _finite_number(frame.get("timestamp_s"), f"{field}.timestamp_s")
        confidence = _finite_number(
            frame.get("tracking_confidence"),
            f"{field}.tracking_confidence",
        )
        if not 0 <= confidence <= 1:
            raise ValueError(f"{field}.tracking_confidence must be between 0 and 1")
        try:
            state = LogicalTrackingState(
                _string(frame.get("tracking_state"), f"{field}.tracking_state")
            )
        except ValueError as error:
            raise ValueError(f"{field}.tracking_state is unsupported") from error
        observation_id = _optional_string(
            frame.get("raw_tracker_observation_id"),
            f"{field}.raw_tracker_observation_id",
        )
        ground_value = frame.get("ground_contact")
        image_point: ImagePoint | None = None
        court_point: CourtPoint | None = None
        court_region = CourtRegionState.AMBIGUOUS
        method: str | None = None
        if ground_value is not None:
            ground = _object(ground_value, f"{field}.ground_contact")
            method = _string(ground.get("method"), f"{field}.ground_contact.method")
            if method != "bounding_box_bottom_center":
                raise ValueError(
                    f"{field}.ground_contact.method must be bounding_box_bottom_center"
                )
            image_point = _parse_image_point(
                ground.get("image_point"),
                f"{field}.ground_contact.image_point",
            )
            if (
                not 0 <= image_point.x_px <= source.width
                or not 0 <= image_point.y_px <= source.height
            ):
                raise ValueError(f"{field}.ground_contact.image_point is outside the frame")
            raw_court_point = ground.get("court_point")
            if raw_court_point is not None:
                court_point = _parse_court_point(
                    raw_court_point,
                    f"{field}.ground_contact.court_point",
                )
            try:
                court_region = CourtRegionState(
                    _string(
                        ground.get("court_region"),
                        f"{field}.ground_contact.court_region",
                    )
                )
            except ValueError as error:
                raise ValueError(f"{field}.ground_contact.court_region is unsupported") from error
        parsed.append(
            RawPlayerPositionFrame(
                role,
                frame_number,
                timestamp_s,
                state,
                confidence,
                image_point,
                court_point,
                court_region,
                observation_id,
                method,
            )
        )
    return tuple(parsed)


def load_tracking_positions(path: Path) -> LoadedTrackingPositions:
    """Load only the structured logical-position layer required by analytics."""

    resolved = path.expanduser().resolve()
    try:
        root = _object(json.loads(resolved.read_text(encoding="utf-8")), "root")
        if root.get("schema_version") != 1:
            raise ValueError("unsupported tracks.json schema_version")
        if root.get("record_type") != "persistent_logical_player_tracks":
            raise ValueError("tracks.json has an unsupported record_type")
        source = _parse_source(root.get("source"))
        inputs = _object(root.get("inputs"), "inputs")
        calibration_path = (
            Path(_string(inputs.get("court_calibration"), "inputs.court_calibration"))
            .expanduser()
            .resolve()
        )
        raw_names = _object(root.get("player_names"), "player_names")
        player_names = {
            role: _string(raw_names.get(role.value), f"player_names.{role.value}")
            for role in LOGICAL_PLAYER_ROLES
        }
        logical_layer = _object(root.get("logical_identity_layer"), "logical_identity_layer")
        raw_tracks = {
            role: _parse_role_frames(role, logical_layer.get(role.value), source=source)
            for role in LOGICAL_PLAYER_ROLES
        }
        return LoadedTrackingPositions(source, calibration_path, player_names, raw_tracks)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise PlayerAnalysisInputError(f"unable to load {resolved}: {error}") from error


def load_position_corrections(
    path: Path,
) -> dict[LogicalPlayerRole, ManualCourtPositionCorrection]:
    """Load bounded, recording-local court offsets without changing raw tracks."""

    resolved = path.expanduser().resolve()
    try:
        root = _object(json.loads(resolved.read_text(encoding="utf-8")), "root")
        if root.get("schema_version") != 1:
            raise ValueError("unsupported position-correction schema_version")
        if root.get("coordinate_space") != "canonical_court_meters":
            raise ValueError("coordinate_space must be canonical_court_meters")
        raw_corrections = _object(root.get("corrections"), "corrections")
        known_roles = {role.value for role in LOGICAL_PLAYER_ROLES}
        unexpected = sorted(set(raw_corrections) - known_roles)
        if unexpected:
            raise ValueError(f"unsupported logical player roles: {', '.join(unexpected)}")
        corrections: dict[LogicalPlayerRole, ManualCourtPositionCorrection] = {}
        for role in LOGICAL_PLAYER_ROLES:
            raw_value = raw_corrections.get(role.value)
            if raw_value is None:
                corrections[role] = ManualCourtPositionCorrection()
                continue
            field = f"corrections.{role.value}"
            raw = _object(raw_value, field)
            x_offset_m = _finite_number(raw.get("x_offset_m", 0.0), f"{field}.x_offset_m")
            y_offset_m = _finite_number(raw.get("y_offset_m", 0.0), f"{field}.y_offset_m")
            magnitude_m = math.hypot(x_offset_m, y_offset_m)
            if magnitude_m > MAXIMUM_MANUAL_CORRECTION_M:
                raise ValueError(
                    f"{field} magnitude must not exceed {MAXIMUM_MANUAL_CORRECTION_M:.2f} m"
                )
            corrections[role] = ManualCourtPositionCorrection(
                x_offset_m=x_offset_m,
                y_offset_m=y_offset_m,
                reason=_string(raw.get("reason"), f"{field}.reason"),
            )
        return corrections
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise PlayerAnalysisInputError(f"unable to load {resolved}: {error}") from error


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
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except (OSError, TypeError, ValueError) as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise OutputWriteError(str(path), reason=str(error)) from error


def _open_writer(path: Path, *, fps: float, dimensions: tuple[int, int]) -> cv2.VideoWriter:
    try:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter.fourcc(*VIDEO_CODEC),
            fps,
            dimensions,
        )
    except cv2.error as error:
        raise OutputWriteError(str(path), reason=str(error)) from error
    if not writer.isOpened():
        writer.release()
        raise OutputWriteError(str(path), reason="OpenCV MP4 writer could not be opened")
    return writer


def _finish_video(writer: cv2.VideoWriter, temporary: Path, output: Path) -> None:
    writer.release()
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise OutputWriteError(str(output), reason="OpenCV wrote an empty video")
    try:
        temporary.replace(output)
    except OSError as error:
        raise OutputWriteError(str(output), reason=str(error)) from error


def _write_analysis_videos(
    video_path: Path,
    *,
    source: VideoMetadata,
    positions: dict[LogicalPlayerRole, tuple[PlayerPositionFrame, ...]],
    calibration_path: Path,
    player_names: dict[LogicalPlayerRole, str],
    settings: PlayerAnalysisSettings,
    annotated_path: Path,
    topdown_path: Path,
) -> None:
    calibration = load_calibration(calibration_path)
    annotated_temporary = annotated_path.with_name(f".{annotated_path.name}.tmp.mp4")
    topdown_temporary = topdown_path.with_name(f".{topdown_path.name}.tmp.mp4")
    annotated_writer = _open_writer(
        annotated_temporary,
        fps=source.fps,
        dimensions=(source.width, source.height),
    )
    topdown_writer = _open_writer(
        topdown_temporary,
        fps=source.fps,
        dimensions=(TOPDOWN_WIDTH_PX, TOPDOWN_HEIGHT_PX),
    )
    trail_length = max(1, round(settings.topdown_trail_seconds * source.fps))
    trails = {role: deque[CourtPoint](maxlen=trail_length) for role in LOGICAL_PLAYER_ROLES}
    logger = logging.getLogger("pickleball_vision.player_analysis")
    try:
        for decoded in iter_video_frames(video_path):
            current = tuple(positions[role][decoded.frame_index] for role in LOGICAL_PLAYER_ROLES)
            for position in current:
                role = position.raw.logical_player
                point = position.smoothed_court_coordinate
                if point is None:
                    trails[role].clear()
                else:
                    trails[role].append(point)
            annotated_writer.write(
                render_source_analysis_frame(
                    decoded.image,
                    positions=current,
                    calibration=calibration,
                    player_names=player_names,
                )
            )
            topdown_writer.write(
                render_topdown_frame(
                    frame_number=decoded.frame_index,
                    timestamp_s=decoded.timestamp,
                    positions=current,
                    trails=trails,
                    court=calibration.court,
                    player_names=player_names,
                )
            )
            if (decoded.frame_index + 1) % 300 == 0:
                logger.info(
                    "player_analysis_render_progress",
                    extra={"context": {"processed_frames": decoded.frame_index + 1}},
                )
    except Exception:
        annotated_writer.release()
        topdown_writer.release()
        with suppress(OSError):
            annotated_temporary.unlink(missing_ok=True)
        with suppress(OSError):
            topdown_temporary.unlink(missing_ok=True)
        raise
    _finish_video(annotated_writer, annotated_temporary, annotated_path)
    _finish_video(topdown_writer, topdown_temporary, topdown_path)


def _write_heatmap(path: Path, image: Image) -> None:
    try:
        written = cv2.imwrite(str(path), image)
    except cv2.error as error:
        raise OutputWriteError(str(path), reason=str(error)) from error
    if not written:
        raise OutputWriteError(str(path), reason="OpenCV did not write the heatmap")


def _validate_video(metadata: VideoMetadata, expected: VideoMetadata) -> None:
    if metadata.path != expected.path:
        raise PlayerAnalysisInputError("the source video differs from tracks.json")
    if (
        metadata.width != expected.width
        or metadata.height != expected.height
        or metadata.frame_count != expected.frame_count
        or not math.isclose(metadata.fps, expected.fps, rel_tol=1e-6)
    ):
        raise PlayerAnalysisInputError("video metadata does not match tracks.json")


def analyze_players_in_video(
    video_path: Path,
    *,
    calibration_path: Path,
    output_dir: Path,
    settings: PlayerAnalysisSettings,
    tracks_path: Path | None = None,
    position_corrections_path: Path | None = None,
) -> PlayerAnalysisArtifacts:
    """Create Release 0.1 positions, visualizations, and qualified metrics."""

    source = inspect_video(video_path)
    resolved_output = _prepare_output_dir(output_dir)
    resolved_tracks = (
        tracks_path.expanduser().resolve()
        if tracks_path is not None
        else resolved_output.parent / "player-tracking" / "tracks.json"
    )
    loaded = load_tracking_positions(resolved_tracks)
    _validate_video(source, loaded.source)
    resolved_calibration = calibration_path.expanduser().resolve()
    if resolved_calibration != loaded.calibration_path:
        raise PlayerAnalysisInputError(
            "the selected calibration differs from the calibration recorded by tracks.json"
        )
    calibration = load_calibration(resolved_calibration)
    if calibration.source.video_path.expanduser().resolve() != source.path:
        raise PlayerAnalysisInputError("the calibration was created from a different video")
    if settings.transition_zone_depth_m > calibration.court.near_kitchen_y_m:
        raise PlayerAnalysisInputError(
            "transition-zone depth exceeds the baseline-to-kitchen distance"
        )

    default_corrections_path = resolved_tracks.parent / POSITION_CORRECTIONS_NAME
    resolved_corrections_path = (
        position_corrections_path.expanduser().resolve()
        if position_corrections_path is not None
        else default_corrections_path.resolve()
        if default_corrections_path.is_file()
        else None
    )
    corrections = (
        load_position_corrections(resolved_corrections_path)
        if resolved_corrections_path is not None
        else {role: ManualCourtPositionCorrection() for role in LOGICAL_PLAYER_ROLES}
    )

    positions_path = resolved_output / POSITIONS_NAME
    summary_path = resolved_output / SUMMARY_NAME
    annotated_path = resolved_output / ANNOTATED_VIDEO_NAME
    topdown_path = resolved_output / TOPDOWN_VIDEO_NAME
    heatmap_paths = {role: resolved_output / HEATMAP_NAMES[role] for role in LOGICAL_PLAYER_ROLES}
    outputs = {positions_path, summary_path, annotated_path, topdown_path, *heatmap_paths.values()}
    if source.path in outputs:
        raise OutputWriteError(
            str(source.path), reason="analysis output would overwrite source video"
        )

    positions = derive_player_positions(
        loaded.raw_tracks,
        settings=settings,
        corrections=corrections,
    )
    run = build_player_analysis_run(
        source=source,
        court=calibration.court,
        tracking_path=str(resolved_tracks),
        calibration_path=str(resolved_calibration),
        settings=settings,
        player_names=loaded.player_names,
        positions=positions,
        position_corrections=corrections,
        position_corrections_path=(
            str(resolved_corrections_path) if resolved_corrections_path is not None else None
        ),
    )
    _write_json(positions_path, run.as_dict())
    _write_analysis_videos(
        source.path,
        source=source,
        positions=positions,
        calibration_path=resolved_calibration,
        player_names=loaded.player_names,
        settings=settings,
        annotated_path=annotated_path,
        topdown_path=topdown_path,
    )
    for role in LOGICAL_PLAYER_ROLES:
        heatmap = render_player_heatmap(
            role,
            positions[role],
            court=calibration.court,
            display_name=loaded.player_names[role],
        )
        _write_heatmap(heatmap_paths[role], heatmap)
    artifact_map: dict[str, object] = {
        "player_positions": str(positions_path),
        "summary": str(summary_path),
        "annotated_video": str(annotated_path),
        "topdown_video": str(topdown_path),
        "heatmaps": {role.value: str(heatmap_paths[role]) for role in LOGICAL_PLAYER_ROLES},
    }
    _write_json(summary_path, build_player_analysis_summary(run, artifacts=artifact_map))
    return PlayerAnalysisArtifacts(
        positions_path,
        summary_path,
        annotated_path,
        topdown_path,
        heatmap_paths,
        source.frame_count,
    )
