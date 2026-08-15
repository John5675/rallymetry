"""Release 0.1 player court positions and inspectable movement analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise

from pickleball_vision.config import PlayerAnalysisSettings
from pickleball_vision.court import CourtDimensions, CourtPoint, ImagePoint
from pickleball_vision.player_isolation import (
    LOGICAL_PLAYER_ROLES,
    CourtRegionState,
    LogicalPlayerRole,
)
from pickleball_vision.player_tracking import LogicalTrackingState
from pickleball_vision.video import VideoMetadata

PLAYER_ANALYSIS_SCHEMA_VERSION = 1
PLAYER_ANALYTICS_VERSION = "player_positions_0.1"
RELEASE_VERSION = "0.1"


class PositionSmoothingStatus(StrEnum):
    """Why a frame does or does not have a separate smoothed coordinate."""

    SMOOTHED = "smoothed"
    MISSING_RAW_COURT_COORDINATE = "missing_raw_court_coordinate"
    BELOW_CONFIDENCE_THRESHOLD = "below_confidence_threshold"
    SUSPECTED_IDENTITY_SWITCH = "suspected_identity_switch"


@dataclass(frozen=True, slots=True)
class ManualCourtPositionCorrection:
    """Recording-local, auditable offset applied after raw court projection."""

    x_offset_m: float = 0.0
    y_offset_m: float = 0.0
    reason: str = "no manual correction"

    @property
    def applied(self) -> bool:
        return self.x_offset_m != 0.0 or self.y_offset_m != 0.0

    def apply(self, point: CourtPoint) -> CourtPoint:
        return CourtPoint(
            x_m=point.x_m + self.x_offset_m,
            y_m=point.y_m + self.y_offset_m,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "x_offset_m": self.x_offset_m,
            "y_offset_m": self.y_offset_m,
            "reason": self.reason,
            "applied": self.applied,
        }


@dataclass(frozen=True, slots=True)
class RawPlayerPositionFrame:
    """Immutable tracking-derived position inputs for one logical player/frame."""

    logical_player: LogicalPlayerRole
    frame_number: int
    timestamp_s: float
    tracking_state: LogicalTrackingState
    confidence: float
    raw_image_ground_point: ImagePoint | None
    raw_court_coordinate: CourtPoint | None
    raw_court_region: CourtRegionState
    raw_tracker_observation_id: str | None
    ground_point_method: str | None


@dataclass(frozen=True, slots=True)
class PlayerPositionFrame:
    """Raw, manually corrected, and separately smoothed player position."""

    raw: RawPlayerPositionFrame
    corrected_court_coordinate: CourtPoint | None
    smoothed_court_coordinate: CourtPoint | None
    smoothing_status: PositionSmoothingStatus
    smoothing_support_frames: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_number": self.raw.frame_number,
            "timestamp_s": self.raw.timestamp_s,
            "confidence": self.raw.confidence,
            "tracking_state": self.raw.tracking_state.value,
            "raw_tracker_observation_id": self.raw.raw_tracker_observation_id,
            "ground_point_method": self.raw.ground_point_method,
            "raw_image_ground_point": (
                self.raw.raw_image_ground_point.as_dict()
                if self.raw.raw_image_ground_point is not None
                else None
            ),
            "raw_court_coordinate": (
                self.raw.raw_court_coordinate.as_dict()
                if self.raw.raw_court_coordinate is not None
                else None
            ),
            "raw_court_region": self.raw.raw_court_region.value,
            "corrected_court_coordinate": (
                self.corrected_court_coordinate.as_dict()
                if self.corrected_court_coordinate is not None
                else None
            ),
            "smoothed_court_coordinate": (
                self.smoothed_court_coordinate.as_dict()
                if self.smoothed_court_coordinate is not None
                else None
            ),
            "smoothing": {
                "status": self.smoothing_status.value,
                "method": (
                    "centered_confidence_weighted_component_median"
                    if self.smoothed_court_coordinate is not None
                    else None
                ),
                "support_frame_numbers": list(self.smoothing_support_frames),
                "interpolated": False,
                "raw_coordinate_preserved": True,
            },
        }


@dataclass(frozen=True, slots=True)
class PlayerAnalysisRun:
    """Versioned structured position artifact for all four logical players."""

    source: VideoMetadata
    court: CourtDimensions
    tracking_path: str
    calibration_path: str
    settings: PlayerAnalysisSettings
    player_names: dict[LogicalPlayerRole, str]
    position_corrections: dict[LogicalPlayerRole, ManualCourtPositionCorrection]
    position_corrections_path: str | None
    positions: dict[LogicalPlayerRole, tuple[PlayerPositionFrame, ...]]
    created_at_utc: str
    schema_version: int = PLAYER_ANALYSIS_SCHEMA_VERSION

    def input_provenance(self) -> dict[str, object]:
        return {
            "persistent_player_tracks": self.tracking_path,
            "court_calibration": self.calibration_path,
            "manual_position_corrections": self.position_corrections_path,
        }

    def configuration(self) -> dict[str, object]:
        return {
            **self.settings.as_dict(),
            "manual_court_position_corrections": {
                role.value: self.position_corrections[role].as_dict()
                for role in LOGICAL_PLAYER_ROLES
            },
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "release_version": RELEASE_VERSION,
            "analytics_version": PLAYER_ANALYTICS_VERSION,
            "record_type": "logical_player_court_positions",
            "created_at_utc": self.created_at_utc,
            "source": self.source.as_dict(),
            "court": self.court.as_dict(),
            "inputs": self.input_provenance(),
            "configuration": self.configuration(),
            "position_contract": {
                "physical_position_source": "bounding_box_bottom_center_ground_estimate",
                "raw_coordinates_are_immutable": True,
                "manual_corrections_are_separate": True,
                "smoothed_coordinates_are_separate": True,
                "missing_frames_are_not_interpolated": True,
            },
            "player_names": {role.value: self.player_names[role] for role in LOGICAL_PLAYER_ROLES},
            "players": {
                role.value: [frame.as_dict() for frame in self.positions[role]]
                for role in LOGICAL_PLAYER_ROLES
            },
        }


def _weighted_median(values: list[float], weights: list[float]) -> float:
    ordered = sorted(zip(values, weights, strict=True), key=lambda item: item[0])
    midpoint = sum(weight for _, weight in ordered) / 2
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= midpoint:
            return value
    return ordered[-1][0]


def _smoothing_eligibility(
    frame: RawPlayerPositionFrame,
    settings: PlayerAnalysisSettings,
) -> PositionSmoothingStatus:
    if frame.raw_court_coordinate is None:
        return PositionSmoothingStatus.MISSING_RAW_COURT_COORDINATE
    if frame.tracking_state is LogicalTrackingState.SUSPECTED_IDENTITY_SWITCH:
        return PositionSmoothingStatus.SUSPECTED_IDENTITY_SWITCH
    if frame.confidence < settings.minimum_tracking_confidence:
        return PositionSmoothingStatus.BELOW_CONFIDENCE_THRESHOLD
    return PositionSmoothingStatus.SMOOTHED


def smooth_player_positions(
    frames: tuple[RawPlayerPositionFrame, ...],
    *,
    settings: PlayerAnalysisSettings,
    correction: ManualCourtPositionCorrection | None = None,
) -> tuple[PlayerPositionFrame, ...]:
    """Smooth observed frames only, with bounded changes and no gap interpolation."""

    selected_correction = correction or ManualCourtPositionCorrection()
    radius = settings.smoothing_window_frames // 2
    statuses = tuple(_smoothing_eligibility(frame, settings) for frame in frames)
    corrected_points = tuple(
        selected_correction.apply(frame.raw_court_coordinate)
        if frame.raw_court_coordinate is not None
        else None
        for frame in frames
    )
    output: list[PlayerPositionFrame] = []
    for index, frame in enumerate(frames):
        status = statuses[index]
        corrected_point = corrected_points[index]
        if status is not PositionSmoothingStatus.SMOOTHED or corrected_point is None:
            output.append(PlayerPositionFrame(frame, corrected_point, None, status, ()))
            continue

        support_indices = [index]
        for candidate_index in range(index - 1, max(-1, index - radius - 1), -1):
            if statuses[candidate_index] is not PositionSmoothingStatus.SMOOTHED:
                break
            if frames[candidate_index + 1].frame_number - frames[candidate_index].frame_number != 1:
                break
            support_indices.append(candidate_index)
        for candidate_index in range(index + 1, min(len(frames), index + radius + 1)):
            if statuses[candidate_index] is not PositionSmoothingStatus.SMOOTHED:
                break
            if frames[candidate_index].frame_number - frames[candidate_index - 1].frame_number != 1:
                break
            support_indices.append(candidate_index)
        support_indices.sort()
        support_points = [corrected_points[item] for item in support_indices]
        if any(point is None for point in support_points):  # defensive; statuses guarantee points
            output.append(PlayerPositionFrame(frame, corrected_point, None, status, ()))
            continue
        court_points = [point for point in support_points if point is not None]
        weights = [max(frames[item].confidence, 1e-6) for item in support_indices]
        candidate_x = _weighted_median([point.x_m for point in court_points], weights)
        candidate_y = _weighted_median([point.y_m for point in court_points], weights)
        dx = candidate_x - corrected_point.x_m
        dy = candidate_y - corrected_point.y_m
        adjustment_m = math.hypot(dx, dy)
        if adjustment_m > settings.maximum_smoothing_adjustment_m:
            scale = settings.maximum_smoothing_adjustment_m / adjustment_m
            candidate_x = corrected_point.x_m + dx * scale
            candidate_y = corrected_point.y_m + dy * scale
        output.append(
            PlayerPositionFrame(
                frame,
                corrected_point,
                CourtPoint(candidate_x, candidate_y),
                status,
                tuple(frames[item].frame_number for item in support_indices),
            )
        )
    return tuple(output)


def derive_player_positions(
    raw_tracks: dict[LogicalPlayerRole, tuple[RawPlayerPositionFrame, ...]],
    *,
    settings: PlayerAnalysisSettings,
    corrections: dict[LogicalPlayerRole, ManualCourtPositionCorrection] | None = None,
) -> dict[LogicalPlayerRole, tuple[PlayerPositionFrame, ...]]:
    """Derive a separate smoothed layer without mutating tracking observations."""

    selected_corrections = corrections or {
        role: ManualCourtPositionCorrection() for role in LOGICAL_PLAYER_ROLES
    }
    return {
        role: smooth_player_positions(
            raw_tracks[role],
            settings=settings,
            correction=selected_corrections[role],
        )
        for role in LOGICAL_PLAYER_ROLES
    }


def build_player_analysis_run(
    *,
    source: VideoMetadata,
    court: CourtDimensions,
    tracking_path: str,
    calibration_path: str,
    settings: PlayerAnalysisSettings,
    player_names: dict[LogicalPlayerRole, str],
    positions: dict[LogicalPlayerRole, tuple[PlayerPositionFrame, ...]],
    position_corrections: dict[LogicalPlayerRole, ManualCourtPositionCorrection] | None = None,
    position_corrections_path: str | None = None,
) -> PlayerAnalysisRun:
    selected_corrections = position_corrections or {
        role: ManualCourtPositionCorrection() for role in LOGICAL_PLAYER_ROLES
    }
    return PlayerAnalysisRun(
        source=source,
        court=court,
        tracking_path=tracking_path,
        calibration_path=calibration_path,
        settings=settings,
        player_names=player_names,
        position_corrections=selected_corrections,
        position_corrections_path=position_corrections_path,
        positions=positions,
        created_at_utc=datetime.now(UTC).isoformat(),
    )


def _inside_court(point: CourtPoint, court: CourtDimensions) -> bool:
    return 0 <= point.x_m <= court.width_m and 0 <= point.y_m <= court.length_m


def metric_eligible(frame: PlayerPositionFrame) -> bool:
    """Return whether a frame can contribute to trajectory-style metrics."""

    return frame.smoothed_court_coordinate is not None and frame.raw.raw_court_region in {
        CourtRegionState.INSIDE,
        CourtRegionState.NEAR,
    }


def heatmap_eligible(frame: PlayerPositionFrame, court: CourtDimensions) -> bool:
    point = frame.smoothed_court_coordinate
    return metric_eligible(frame) and point is not None and _inside_court(point, court)


def _frame_ranges(frame_numbers: list[int]) -> list[dict[str, int]]:
    if not frame_numbers:
        return []
    ordered = sorted(set(frame_numbers))
    ranges: list[dict[str, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            ranges.append({"start_frame": start, "end_frame": previous})
            start = value
        previous = value
    ranges.append({"start_frame": start, "end_frame": previous})
    return ranges


def _occupancy_region(
    point: CourtPoint,
    court: CourtDimensions,
    *,
    transition_zone_depth_m: float,
) -> str:
    if court.near_kitchen_y_m <= point.y_m <= court.far_kitchen_y_m:
        return "kitchen"
    if (
        court.near_kitchen_y_m - transition_zone_depth_m <= point.y_m < court.near_kitchen_y_m
        or court.far_kitchen_y_m < point.y_m <= court.far_kitchen_y_m + transition_zone_depth_m
    ):
        return "transition_zone"
    return "backcourt"


def _accepted_steps(
    frames: tuple[PlayerPositionFrame, ...],
    settings: PlayerAnalysisSettings,
) -> tuple[list[tuple[PlayerPositionFrame, PlayerPositionFrame, float, float, float]], int, int]:
    eligible = [frame for frame in frames if metric_eligible(frame)]
    accepted: list[tuple[PlayerPositionFrame, PlayerPositionFrame, float, float, float]] = []
    excluded_gap = 0
    excluded_speed = 0
    for previous, current in pairwise(eligible):
        elapsed_s = current.raw.timestamp_s - previous.raw.timestamp_s
        if (
            current.raw.frame_number - previous.raw.frame_number != 1
            or elapsed_s <= 0
            or elapsed_s > settings.maximum_step_gap_seconds
        ):
            excluded_gap += 1
            continue
        previous_point = previous.smoothed_court_coordinate
        current_point = current.smoothed_court_coordinate
        if previous_point is None or current_point is None:
            continue
        dx_m = current_point.x_m - previous_point.x_m
        dy_m = current_point.y_m - previous_point.y_m
        distance_m = math.hypot(dx_m, dy_m)
        if distance_m / elapsed_s > settings.maximum_step_speed_mps:
            excluded_speed += 1
            continue
        accepted.append((previous, current, distance_m, dx_m, elapsed_s))
    return accepted, excluded_gap, excluded_speed


def _distance_metric(
    frames: tuple[PlayerPositionFrame, ...],
    settings: PlayerAnalysisSettings,
) -> dict[str, object]:
    steps, excluded_gap, excluded_speed = _accepted_steps(frames, settings)
    frame_numbers = [frame.raw.frame_number for step in steps for frame in step[:2]]
    return {
        "value_m": sum(step[2] for step in steps) if steps else None,
        "unit": "meters",
        "calculation_version": PLAYER_ANALYTICS_VERSION,
        "population": "quality-gated consecutive inside-or-near smoothed positions",
        "contributing_step_count": len(steps),
        "contributing_frame_ranges": _frame_ranges(frame_numbers),
        "excluded_gap_step_count": excluded_gap,
        "excluded_speed_step_count": excluded_speed,
        "approximate": True,
    }


def _occupancy_metrics(
    frames: tuple[PlayerPositionFrame, ...],
    *,
    source: VideoMetadata,
    court: CourtDimensions,
    settings: PlayerAnalysisSettings,
) -> dict[str, object]:
    in_court = [frame for frame in frames if heatmap_eligible(frame, court)]
    by_region: dict[str, list[int]] = {
        "kitchen": [],
        "transition_zone": [],
        "backcourt": [],
    }
    for frame in in_court:
        point = frame.smoothed_court_coordinate
        if point is not None:
            by_region[
                _occupancy_region(
                    point,
                    court,
                    transition_zone_depth_m=settings.transition_zone_depth_m,
                )
            ].append(frame.raw.frame_number)
    denominator = len(in_court)
    return {
        region: {
            "frame_count": len(frame_numbers),
            "seconds": len(frame_numbers) / source.fps,
            "share_of_in_court_frames": (len(frame_numbers) / denominator if denominator else None),
            "unit": "frame_share",
            "calculation_version": PLAYER_ANALYTICS_VERSION,
            "contributing_frame_ranges": _frame_ranges(frame_numbers),
        }
        for region, frame_numbers in by_region.items()
    }


def _average_kitchen_distance(
    frames: tuple[PlayerPositionFrame, ...],
    court: CourtDimensions,
) -> dict[str, object]:
    values: list[float] = []
    frame_numbers: list[int] = []
    for frame in frames:
        point = frame.smoothed_court_coordinate
        if not metric_eligible(frame) or point is None or not _inside_court(point, court):
            continue
        if point.y_m <= court.net_y_m:
            distance_m = max(0.0, court.near_kitchen_y_m - point.y_m)
        else:
            distance_m = max(0.0, point.y_m - court.far_kitchen_y_m)
        values.append(distance_m)
        frame_numbers.append(frame.raw.frame_number)
    return {
        "value_m": sum(values) / len(values) if values else None,
        "unit": "meters",
        "calculation_version": PLAYER_ANALYTICS_VERSION,
        "population": "quality-gated in-court smoothed positions",
        "sample_count": len(values),
        "contributing_frame_ranges": _frame_ranges(frame_numbers),
    }


def _lateral_metrics(
    frames: tuple[PlayerPositionFrame, ...],
    settings: PlayerAnalysisSettings,
) -> dict[str, object]:
    eligible = [frame for frame in frames if metric_eligible(frame)]
    points = [frame.smoothed_court_coordinate for frame in eligible]
    x_values = [point.x_m for point in points if point is not None]
    steps, excluded_gap, excluded_speed = _accepted_steps(frames, settings)
    absolute_lateral = [abs(step[3]) for step in steps]
    lateral_speeds = [abs(step[3]) / step[4] for step in steps]
    mean_x = sum(x_values) / len(x_values) if x_values else None
    position_stddev = (
        math.sqrt(sum((value - mean_x) ** 2 for value in x_values) / len(x_values))
        if x_values and mean_x is not None
        else None
    )
    frame_numbers = [frame.raw.frame_number for step in steps for frame in step[:2]]
    return {
        "total_absolute_lateral_distance_m": (sum(absolute_lateral) if absolute_lateral else None),
        "mean_absolute_lateral_speed_mps": (
            sum(absolute_lateral) / sum(step[4] for step in steps) if steps else None
        ),
        "maximum_lateral_speed_mps": max(lateral_speeds) if lateral_speeds else None,
        "lateral_range_m": max(x_values) - min(x_values) if x_values else None,
        "lateral_position_stddev_m": position_stddev,
        "unit": "meters_and_meters_per_second",
        "calculation_version": PLAYER_ANALYTICS_VERSION,
        "contributing_step_count": len(steps),
        "contributing_frame_ranges": _frame_ranges(frame_numbers),
        "excluded_gap_step_count": excluded_gap,
        "excluded_speed_step_count": excluded_speed,
    }


def _partner_spacing(
    first_role: LogicalPlayerRole,
    second_role: LogicalPlayerRole,
    positions: dict[LogicalPlayerRole, tuple[PlayerPositionFrame, ...]],
    *,
    source: VideoMetadata,
) -> dict[str, object]:
    values: list[float] = []
    frame_numbers: list[int] = []
    for first, second in zip(positions[first_role], positions[second_role], strict=True):
        if not metric_eligible(first) or not metric_eligible(second):
            continue
        first_point = first.smoothed_court_coordinate
        second_point = second.smoothed_court_coordinate
        if first_point is None or second_point is None:
            continue
        values.append(
            math.hypot(first_point.x_m - second_point.x_m, first_point.y_m - second_point.y_m)
        )
        frame_numbers.append(first.raw.frame_number)
    return {
        "value_m": sum(values) / len(values) if values else None,
        "unit": "meters",
        "calculation_version": PLAYER_ANALYTICS_VERSION,
        "population": "joint quality-gated inside-or-near smoothed positions",
        "sample_count": len(values),
        "source_frame_coverage_ratio": len(values) / source.frame_count,
        "contributing_frame_ranges": _frame_ranges(frame_numbers),
    }


def build_player_analysis_summary(
    run: PlayerAnalysisRun,
    *,
    artifacts: dict[str, object],
) -> dict[str, object]:
    """Compute Release 0.1 metrics only from structured player position records."""

    team_pairs = (
        (LogicalPlayerRole.ME, LogicalPlayerRole.PARTNER),
        (LogicalPlayerRole.OPPONENT_1, LogicalPlayerRole.OPPONENT_2),
    )
    spacing_by_role: dict[LogicalPlayerRole, dict[str, object]] = {}
    teams: dict[str, object] = {}
    for first_role, second_role in team_pairs:
        spacing = _partner_spacing(first_role, second_role, run.positions, source=run.source)
        spacing_by_role[first_role] = spacing
        spacing_by_role[second_role] = spacing
        teams[f"{first_role.value}_{second_role.value}"] = {
            "players": [first_role.value, second_role.value],
            "average_partner_spacing": spacing,
        }

    per_player: dict[str, object] = {}
    for role in LOGICAL_PLAYER_ROLES:
        frames = run.positions[role]
        raw_image_frames = sum(frame.raw.raw_image_ground_point is not None for frame in frames)
        raw_court_frames = sum(frame.raw.raw_court_coordinate is not None for frame in frames)
        corrected_frames = sum(frame.corrected_court_coordinate is not None for frame in frames)
        smoothed_frames = sum(frame.smoothed_court_coordinate is not None for frame in frames)
        metric_frames = sum(metric_eligible(frame) for frame in frames)
        in_court_frames = sum(heatmap_eligible(frame, run.court) for frame in frames)
        per_player[role.value] = {
            "display_name": run.player_names[role],
            "manual_court_position_correction": run.position_corrections[role].as_dict(),
            "data_quality": {
                "source_frame_count": len(frames),
                "raw_image_ground_frame_count": raw_image_frames,
                "raw_court_coordinate_frame_count": raw_court_frames,
                "corrected_court_coordinate_frame_count": corrected_frames,
                "smoothed_coordinate_frame_count": smoothed_frames,
                "metric_eligible_frame_count": metric_frames,
                "in_court_metric_frame_count": in_court_frames,
                "raw_court_coverage_ratio": raw_court_frames / len(frames) if frames else 0.0,
                "metric_coverage_ratio": metric_frames / len(frames) if frames else 0.0,
            },
            "metrics": {
                "approximate_distance_traveled": _distance_metric(frames, run.settings),
                "court_occupancy": _occupancy_metrics(
                    frames,
                    source=run.source,
                    court=run.court,
                    settings=run.settings,
                ),
                "average_distance_from_kitchen": _average_kitchen_distance(
                    frames,
                    run.court,
                ),
                "average_partner_spacing": spacing_by_role[role],
                "lateral_movement": _lateral_metrics(frames, run.settings),
            },
        }
    return {
        "schema_version": PLAYER_ANALYSIS_SCHEMA_VERSION,
        "release_version": RELEASE_VERSION,
        "analytics_version": PLAYER_ANALYTICS_VERSION,
        "record_type": "player_court_position_analytics_summary",
        "source": run.source.as_dict(),
        "court": run.court.as_dict(),
        "inputs": run.input_provenance(),
        "configuration": run.configuration(),
        "players": per_player,
        "teams": teams,
        "artifacts": artifacts,
        "limitations": [
            "positions inherit calibration and logical-tracking uncertainty",
            "manual court-position corrections are recording-local estimates",
            "distance is quality-gated and approximate, not a survey measurement",
            "missing frames and suspected identity switches are never interpolated",
        ],
    }
