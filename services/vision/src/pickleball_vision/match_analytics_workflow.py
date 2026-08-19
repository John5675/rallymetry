"""Validated I/O workflow for deterministic match analytics."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pickleball_vision.config import MatchAnalyticsSettings
from pickleball_vision.court import CourtDimensions, CourtPoint
from pickleball_vision.errors import MatchAnalyticsInputError, OutputWriteError
from pickleball_vision.match_analytics import (
    KNOWN_PLAYER_IDS,
    LOGICAL_PLAYERS,
    MATCH_ANALYTICS_SCHEMA_VERSION,
    AnalyticsPosition,
    AnalyticsRally,
    AnalyticsShot,
    MatchAnalyticsConfiguration,
    PositionMetricConfiguration,
    compute_match_analytics,
)
from pickleball_vision.media import MediaMetadata, inspect_media
from pickleball_vision.shot_reconstruction import ShotType


@dataclass(frozen=True, slots=True)
class MatchAnalyticsArtifacts:
    """Written deterministic match-analytics artifact."""

    analytics_path: Path
    rally_count: int
    shot_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "analyticsPath": str(self.analytics_path),
            "rallyCount": self.rally_count,
            "shotCount": self.shot_count,
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


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _confidence(value: object, field: str) -> float:
    number = _number(value, field)
    if not 0 <= number <= 1:
        raise ValueError(f"{field} must be between 0 and 1 inclusive")
    return number


def _read_json(path: Path, label: str) -> tuple[Path, dict[str, object]]:
    resolved = path.expanduser().resolve()
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MatchAnalyticsInputError(f"unable to read {label} {resolved}: {error}") from error
    try:
        return resolved, _object(raw, label)
    except ValueError as error:
        raise MatchAnalyticsInputError(str(error)) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise MatchAnalyticsInputError(f"unable to hash input {path}: {error}") from error
    return digest.hexdigest()


def _validate_source(payload: dict[str, object], media: MediaMetadata, label: str) -> None:
    source = _object(payload.get("source"), f"{label}.source")
    expected = media.video
    actual_path = Path(_string(source.get("path"), f"{label}.source.path")).expanduser().resolve()
    if actual_path != expected.path:
        raise ValueError(
            f"{label}.source.path is {actual_path}, not requested video {expected.path}"
        )
    exact_fields = {
        "width": expected.width,
        "height": expected.height,
        "frame_count": expected.frame_count,
    }
    for field, expected_value in exact_fields.items():
        if _integer(source.get(field), f"{label}.source.{field}") != expected_value:
            raise ValueError(f"{label}.source.{field} does not match requested video")
    fps = _number(source.get("fps"), f"{label}.source.fps")
    duration = _number(source.get("duration"), f"{label}.source.duration")
    if not math.isclose(fps, expected.fps, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"{label}.source.fps does not match requested video")
    if not math.isclose(duration, expected.duration, rel_tol=1e-6, abs_tol=1e-3):
        raise ValueError(f"{label}.source.duration does not match requested video")


def _load_rallies(
    path: Path,
    *,
    media: MediaMetadata,
) -> tuple[Path, str, tuple[AnalyticsRally, ...]]:
    resolved, payload = _read_json(path, "rallies")
    try:
        if payload.get("recordType") != "automatic_rally_segments":
            raise ValueError("rallies.recordType must be automatic_rally_segments")
        _validate_source(payload, media, "rallies")
        parsed: list[AnalyticsRally] = []
        ids: set[str] = set()
        previous_end = -1
        for index, raw in enumerate(_array(payload.get("rallies"), "rallies.rallies")):
            field = f"rallies.rallies[{index}]"
            item = _object(raw, field)
            rally_id = _string(item.get("rallyId"), f"{field}.rallyId")
            if rally_id in ids:
                raise ValueError(f"{field}.rallyId is duplicated")
            ids.add(rally_id)
            start_frame = _integer(item.get("startFrame"), f"{field}.startFrame")
            end_frame = _integer(item.get("endFrame"), f"{field}.endFrame")
            start_s = _number(item.get("startTimestamp"), f"{field}.startTimestamp")
            end_s = _number(item.get("endTimestamp"), f"{field}.endTimestamp")
            if start_frame < 0 or end_frame < start_frame or end_frame >= media.video.frame_count:
                raise ValueError(f"{field} has an invalid frame interval")
            if start_frame <= previous_end:
                raise ValueError("rallies must be ordered and non-overlapping")
            if start_s < 0 or end_s < start_s or end_s > media.video.duration + 1 / media.video.fps:
                raise ValueError(f"{field} has an invalid timestamp interval")
            parsed.append(
                AnalyticsRally(
                    rally_id,
                    start_s,
                    end_s,
                    start_frame,
                    end_frame,
                    _confidence(item.get("confidence"), f"{field}.confidence"),
                )
            )
            previous_end = end_frame
        return resolved, _sha256(resolved), tuple(parsed)
    except ValueError as error:
        raise MatchAnalyticsInputError(str(error)) from error


def _optional_court_point(value: object, field: str) -> tuple[CourtPoint | None, str | None]:
    if value is None:
        return None, None
    position = _object(value, field)
    raw_point = position.get("courtPoint")
    if raw_point is None:
        return None, None
    point = _object(raw_point, f"{field}.courtPoint")
    court_point = CourtPoint(
        _number(point.get("x_m"), f"{field}.courtPoint.x_m"),
        _number(point.get("y_m"), f"{field}.courtPoint.y_m"),
    )
    region_value = position.get("courtRegion")
    region = None if region_value is None else _string(region_value, f"{field}.courtRegion")
    if region not in {"inside", "near", "outside", "ambiguous"}:
        raise ValueError(f"{field}.courtRegion is unsupported")
    return court_point, region


def _load_shots(
    path: Path,
    *,
    media: MediaMetadata,
    rally_ids: frozenset[str],
    rallies_path: Path,
    rallies_sha256: str,
) -> tuple[Path, str, tuple[AnalyticsShot, ...], Path]:
    resolved, payload = _read_json(path, "shots")
    try:
        if payload.get("recordType") != "reconstructed_classified_shots":
            raise ValueError("shots.recordType must be reconstructed_classified_shots")
        _validate_source(payload, media, "shots")
        inputs = _object(payload.get("inputs"), "shots.inputs")
        rallies_input = _object(inputs.get("rallies"), "shots.inputs.rallies")
        recorded_rallies_path = (
            Path(_string(rallies_input.get("path"), "shots.inputs.rallies.path"))
            .expanduser()
            .resolve()
        )
        if recorded_rallies_path != rallies_path:
            raise ValueError("shots were not reconstructed from the supplied rallies artifact")
        if _string(rallies_input.get("sha256"), "shots.inputs.rallies.sha256") != rallies_sha256:
            raise ValueError("supplied rallies artifact changed after shots were reconstructed")
        tracks_input = _object(inputs.get("playerTracks"), "shots.inputs.playerTracks")
        tracks_path = (
            Path(_string(tracks_input.get("path"), "shots.inputs.playerTracks.path"))
            .expanduser()
            .resolve()
        )
        parsed: list[AnalyticsShot] = []
        ids: set[str] = set()
        expected_index_by_rally = dict.fromkeys(rally_ids, 1)
        for index, raw in enumerate(_array(payload.get("shots"), "shots.shots")):
            field = f"shots.shots[{index}]"
            item = _object(raw, field)
            shot_id = _string(item.get("shotId"), f"{field}.shotId")
            if shot_id in ids:
                raise ValueError(f"{field}.shotId is duplicated")
            ids.add(shot_id)
            rally_id = _string(item.get("rallyId"), f"{field}.rallyId")
            if rally_id not in rally_ids:
                raise ValueError(f"{field}.rallyId does not reference a supplied rally")
            shot_index = _integer(item.get("shotIndex"), f"{field}.shotIndex")
            if shot_index != expected_index_by_rally[rally_id]:
                raise ValueError(f"{field}.shotIndex must be contiguous within its rally")
            expected_index_by_rally[rally_id] += 1
            hitter_id = _string(item.get("hitterId"), f"{field}.hitterId")
            if hitter_id not in KNOWN_PLAYER_IDS | {"UNKNOWN"}:
                raise ValueError(f"{field}.hitterId is unsupported")
            try:
                shot_type = ShotType(_string(item.get("shotType"), f"{field}.shotType"))
            except ValueError as error:
                raise ValueError(f"{field}.shotType is unsupported") from error
            court_point, court_region = _optional_court_point(
                item.get("hitterCourtPosition"),
                f"{field}.hitterCourtPosition",
            )
            parsed.append(
                AnalyticsShot(
                    shot_id,
                    rally_id,
                    shot_index,
                    hitter_id,
                    shot_type,
                    _confidence(item.get("confidence"), f"{field}.confidence"),
                    _confidence(item.get("hitterConfidence"), f"{field}.hitterConfidence"),
                    court_point,
                    court_region,
                )
            )
        return resolved, _sha256(resolved), tuple(parsed), tracks_path
    except ValueError as error:
        raise MatchAnalyticsInputError(str(error)) from error


def _parse_court(payload: dict[str, object]) -> CourtDimensions:
    court = _object(payload.get("court"), "playerPositions.court")
    return CourtDimensions(
        width_m=_number(court.get("width_m"), "playerPositions.court.width_m"),
        length_m=_number(court.get("length_m"), "playerPositions.court.length_m"),
        non_volley_zone_depth_m=_number(
            court.get("non_volley_zone_depth_m"),
            "playerPositions.court.non_volley_zone_depth_m",
        ),
    )


def _load_positions(
    path: Path,
    *,
    media: MediaMetadata,
    expected_tracks_path: Path,
) -> tuple[
    Path,
    str,
    CourtDimensions,
    PositionMetricConfiguration,
    dict[str, str],
    dict[str, tuple[AnalyticsPosition, ...]],
]:
    resolved, payload = _read_json(path, "playerPositions")
    try:
        if payload.get("record_type") != "logical_player_court_positions":
            raise ValueError("playerPositions.record_type must be logical_player_court_positions")
        _validate_source(payload, media, "playerPositions")
        inputs = _object(payload.get("inputs"), "playerPositions.inputs")
        tracks_path = (
            Path(
                _string(
                    inputs.get("persistent_player_tracks"),
                    "playerPositions.inputs.persistent_player_tracks",
                )
            )
            .expanduser()
            .resolve()
        )
        if tracks_path != expected_tracks_path:
            raise ValueError(
                "shots and player positions were not derived from the same persistent tracks"
            )
        court = _parse_court(payload)
        raw_configuration = _object(payload.get("configuration"), "playerPositions.configuration")
        position_configuration = PositionMetricConfiguration(
            maximum_step_gap_seconds=_number(
                raw_configuration.get("maximum_step_gap_seconds"),
                "playerPositions.configuration.maximum_step_gap_seconds",
            ),
            maximum_step_speed_mps=_number(
                raw_configuration.get("maximum_step_speed_mps"),
                "playerPositions.configuration.maximum_step_speed_mps",
            ),
            transition_zone_depth_m=_number(
                raw_configuration.get("transition_zone_depth_m"),
                "playerPositions.configuration.transition_zone_depth_m",
            ),
        )
        if (
            position_configuration.maximum_step_gap_seconds <= 0
            or position_configuration.maximum_step_speed_mps <= 0
            or position_configuration.transition_zone_depth_m <= 0
        ):
            raise ValueError("playerPositions metric configuration must be positive")
        raw_names = _object(payload.get("player_names"), "playerPositions.player_names")
        player_names = {
            player_id: _string(
                raw_names.get(player_id),
                f"playerPositions.player_names.{player_id}",
            )
            for player_id in LOGICAL_PLAYERS
        }
        raw_players = _object(payload.get("players"), "playerPositions.players")
        if set(raw_players) != set(LOGICAL_PLAYERS):
            raise ValueError(
                "playerPositions.players must contain exactly the four logical players"
            )
        positions_by_player: dict[str, tuple[AnalyticsPosition, ...]] = {}
        for player_id in LOGICAL_PLAYERS:
            raw_positions = _array(
                raw_players.get(player_id),
                f"playerPositions.players.{player_id}",
            )
            if len(raw_positions) != media.video.frame_count:
                raise ValueError(
                    f"playerPositions.players.{player_id} must contain one record per source frame"
                )
            parsed: list[AnalyticsPosition] = []
            for expected_frame, raw in enumerate(raw_positions):
                field = f"playerPositions.players.{player_id}[{expected_frame}]"
                item = _object(raw, field)
                frame = _integer(item.get("frame_number"), f"{field}.frame_number")
                if frame != expected_frame:
                    raise ValueError(f"{field}.frame_number must be {expected_frame}")
                point_value = item.get("smoothed_court_coordinate")
                point: CourtPoint | None = None
                if point_value is not None:
                    raw_point = _object(point_value, f"{field}.smoothed_court_coordinate")
                    point = CourtPoint(
                        _number(raw_point.get("x_m"), f"{field}.smoothed_court_coordinate.x_m"),
                        _number(raw_point.get("y_m"), f"{field}.smoothed_court_coordinate.y_m"),
                    )
                region = _string(item.get("raw_court_region"), f"{field}.raw_court_region")
                if region not in {"inside", "near", "outside", "ambiguous"}:
                    raise ValueError(f"{field}.raw_court_region is unsupported")
                timestamp_s = _number(item.get("timestamp_s"), f"{field}.timestamp_s")
                expected_timestamp_s = frame / media.video.fps
                if not math.isclose(
                    timestamp_s,
                    expected_timestamp_s,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ):
                    raise ValueError(f"{field}.timestamp_s does not match its source frame")
                parsed.append(
                    AnalyticsPosition(
                        player_id,
                        frame,
                        timestamp_s,
                        _confidence(item.get("confidence"), f"{field}.confidence"),
                        point,
                        region,
                    )
                )
            positions_by_player[player_id] = tuple(parsed)
        return (
            resolved,
            _sha256(resolved),
            court,
            position_configuration,
            player_names,
            positions_by_player,
        )
    except (ValueError, TypeError) as error:
        raise MatchAnalyticsInputError(str(error)) from error


def _write_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise OutputWriteError(str(path), reason="refusing to overwrite an existing artifact")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except (OSError, ValueError) as error:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.unlink(missing_ok=True)
        raise OutputWriteError(str(path), reason=str(error)) from error


def analyze_match(
    video_path: Path,
    *,
    rallies_path: Path,
    shots_path: Path,
    player_positions_path: Path,
    output_path: Path,
    settings: MatchAnalyticsSettings,
) -> MatchAnalyticsArtifacts:
    """Validate related domain artifacts, calculate metrics, and write one JSON file."""

    media = inspect_media(video_path)
    resolved_rallies, rallies_hash, rallies = _load_rallies(rallies_path, media=media)
    resolved_shots, shots_hash, shots, player_tracks_path = _load_shots(
        shots_path,
        media=media,
        rally_ids=frozenset(rally.rally_id for rally in rallies),
        rallies_path=resolved_rallies,
        rallies_sha256=rallies_hash,
    )
    (
        resolved_positions,
        positions_hash,
        court,
        position_configuration,
        player_names,
        positions_by_player,
    ) = _load_positions(
        player_positions_path,
        media=media,
        expected_tracks_path=player_tracks_path,
    )
    configuration = MatchAnalyticsConfiguration(
        kitchen_arrival_distance_m=settings.kitchen_arrival_distance_m,
        minimum_kitchen_arrival_joint_coverage_ratio=(
            settings.minimum_kitchen_arrival_joint_coverage_ratio
        ),
    )
    metrics = compute_match_analytics(
        rallies=rallies,
        shots=shots,
        positions_by_player=positions_by_player,
        player_names=player_names,
        court=court,
        source_fps=media.video.fps,
        source_frame_count=media.video.frame_count,
        position_configuration=position_configuration,
        configuration=configuration,
    )
    output = output_path.expanduser().resolve()
    if output == media.video.path:
        raise OutputWriteError(str(output), reason="output would overwrite source video")
    inputs = {
        "rallies": {
            "path": str(resolved_rallies),
            "sha256": rallies_hash,
            "recordType": "automatic_rally_segments",
            "mutated": False,
        },
        "shots": {
            "path": str(resolved_shots),
            "sha256": shots_hash,
            "recordType": "reconstructed_classified_shots",
            "mutated": False,
        },
        "playerPositions": {
            "path": str(resolved_positions),
            "sha256": positions_hash,
            "recordType": "logical_player_court_positions",
            "mutated": False,
        },
    }
    payload = {
        "schemaVersion": MATCH_ANALYTICS_SCHEMA_VERSION,
        "recordType": "deterministic_match_analytics",
        "createdAtUtc": datetime.now(UTC).isoformat(),
        "source": media.as_dict(),
        "court": court.as_dict(),
        "inputs": inputs,
        "configuration": {
            **configuration.as_dict(),
            "positionMetricsInheritedFromPlayerPositions": {
                "maximumStepGapSeconds": position_configuration.maximum_step_gap_seconds,
                "maximumStepSpeedMps": position_configuration.maximum_step_speed_mps,
                "transitionZoneDepthMeters": position_configuration.transition_zone_depth_m,
            },
        },
        "contracts": {
            "structuredDomainObjectsOnly": True,
            "rawModelTensorsAccessed": False,
            "rawAudioWaveformsAccessed": False,
            "rawYoloOutputsAccessed": False,
            "unknownValuesPreserved": True,
            "confidenceWeightedStatistics": False,
            "sourceArtifactsMutated": False,
            "inventedStatisticsAllowed": False,
        },
        "metricDefinitions": {
            "document": "docs/analytics-definitions.md",
            "version": metrics["analyticsVersion"],
        },
        **metrics,
        "limitations": [
            (
                "every metric inherits rally, hitter, shot-classification, calibration, "
                "and tracking error"
            ),
            (
                "counts are deterministic summaries of upstream structured predictions, "
                "not ground truth"
            ),
            "confidence is reported as data quality and never used as fractional event weight",
            "UNKNOWN values are excluded only from explicitly documented classified denominators",
            (
                "distance and occupancy use quality-gated smoothed court positions and "
                "remain approximate"
            ),
        ],
    }
    _write_json(output, payload)
    return MatchAnalyticsArtifacts(output, len(rallies), len(shots))
