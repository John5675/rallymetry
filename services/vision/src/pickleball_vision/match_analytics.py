"""Deterministic match analytics over validated structured domain records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

from pickleball_vision.court import CourtDimensions, CourtPoint
from pickleball_vision.shot_reconstruction import ShotType

MATCH_ANALYTICS_SCHEMA_VERSION = 1
MATCH_ANALYTICS_VERSION = "match_analytics_1.0"
LOGICAL_PLAYERS = ("ME", "PARTNER", "OPPONENT_1", "OPPONENT_2")
KNOWN_PLAYER_IDS = frozenset(LOGICAL_PLAYERS)
PLAYER_TEAMS = {
    "nearTeam": ("ME", "PARTNER"),
    "farTeam": ("OPPONENT_1", "OPPONENT_2"),
}
COUNTED_SHOT_TYPES = (
    ShotType.DINK,
    ShotType.DROP,
    ShotType.DRIVE,
    ShotType.VOLLEY,
    ShotType.OVERHEAD,
)
POSITION_REGIONS = ("kitchen", "transitionZone", "backcourt")


@dataclass(frozen=True, slots=True)
class AnalyticsRally:
    """The rally fields consumed by deterministic analytics."""

    rally_id: str
    start_timestamp_s: float
    end_timestamp_s: float
    start_frame: int
    end_frame: int
    confidence: float

    @property
    def duration_s(self) -> float:
        return self.end_timestamp_s - self.start_timestamp_s


@dataclass(frozen=True, slots=True)
class AnalyticsShot:
    """The classified-shot fields consumed by deterministic analytics."""

    shot_id: str
    rally_id: str
    shot_index: int
    hitter_id: str
    shot_type: ShotType
    confidence: float
    hitter_confidence: float
    hitter_court_position: CourtPoint | None
    hitter_court_region: str | None


@dataclass(frozen=True, slots=True)
class AnalyticsPosition:
    """One quality-gated player-position input without raw detector details."""

    player_id: str
    frame_number: int
    timestamp_s: float
    confidence: float
    court_point: CourtPoint | None
    court_region: str

    @property
    def metric_eligible(self) -> bool:
        return self.court_point is not None and self.court_region in {"inside", "near"}


@dataclass(frozen=True, slots=True)
class PositionMetricConfiguration:
    """Quality gates inherited from the structured player-position artifact."""

    maximum_step_gap_seconds: float
    maximum_step_speed_mps: float
    transition_zone_depth_m: float


@dataclass(frozen=True, slots=True)
class MatchAnalyticsConfiguration:
    """Externalized definitions unique to deterministic match analytics."""

    kitchen_arrival_distance_m: float = 0.90
    minimum_kitchen_arrival_joint_coverage_ratio: float = 0.50

    def as_dict(self) -> dict[str, object]:
        return {
            "kitchenArrivalDistanceMeters": self.kitchen_arrival_distance_m,
            "minimumKitchenArrivalJointCoverageRatio": (
                self.minimum_kitchen_arrival_joint_coverage_ratio
            ),
        }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rate_metric(
    numerator: int,
    denominator: int,
    *,
    numerator_population: str,
    denominator_population: str,
) -> dict[str, object]:
    return {
        "value": numerator / denominator if denominator else None,
        "unit": "ratio",
        "numerator": numerator,
        "denominator": denominator,
        "numeratorPopulation": numerator_population,
        "denominatorPopulation": denominator_population,
        "calculationVersion": MATCH_ANALYTICS_VERSION,
    }


def _count_metric(value: int, population: str) -> dict[str, object]:
    return {
        "value": value,
        "unit": "count",
        "population": population,
        "calculationVersion": MATCH_ANALYTICS_VERSION,
    }


def _inside_court(point: CourtPoint, court: CourtDimensions) -> bool:
    return 0 <= point.x_m <= court.width_m and 0 <= point.y_m <= court.length_m


def position_region(
    point: CourtPoint,
    court: CourtDimensions,
    *,
    transition_zone_depth_m: float,
) -> str:
    """Classify a court-plane point using the Release 0.1 longitudinal zones."""

    if court.near_kitchen_y_m <= point.y_m <= court.far_kitchen_y_m:
        return "kitchen"
    if (
        court.near_kitchen_y_m - transition_zone_depth_m <= point.y_m < court.near_kitchen_y_m
        or court.far_kitchen_y_m < point.y_m <= court.far_kitchen_y_m + transition_zone_depth_m
    ):
        return "transitionZone"
    return "backcourt"


def _distance_traveled(
    positions: tuple[AnalyticsPosition, ...],
    configuration: PositionMetricConfiguration,
) -> dict[str, object]:
    eligible = [item for item in positions if item.metric_eligible]
    accepted_distances: list[float] = []
    excluded_gap_count = 0
    excluded_speed_count = 0
    for previous, current in pairwise(eligible):
        elapsed_s = current.timestamp_s - previous.timestamp_s
        if (
            current.frame_number != previous.frame_number + 1
            or elapsed_s <= 0
            or elapsed_s > configuration.maximum_step_gap_seconds
        ):
            excluded_gap_count += 1
            continue
        if previous.court_point is None or current.court_point is None:
            continue
        distance_m = math.hypot(
            current.court_point.x_m - previous.court_point.x_m,
            current.court_point.y_m - previous.court_point.y_m,
        )
        if distance_m / elapsed_s > configuration.maximum_step_speed_mps:
            excluded_speed_count += 1
            continue
        accepted_distances.append(distance_m)
    return {
        "value": sum(accepted_distances) if accepted_distances else None,
        "unit": "meters",
        "contributingStepCount": len(accepted_distances),
        "excludedGapStepCount": excluded_gap_count,
        "excludedSpeedStepCount": excluded_speed_count,
        "approximate": True,
        "calculationVersion": MATCH_ANALYTICS_VERSION,
    }


def _occupancy(
    positions: tuple[AnalyticsPosition, ...],
    *,
    court: CourtDimensions,
    configuration: PositionMetricConfiguration,
    fps: float,
) -> dict[str, object]:
    counts = dict.fromkeys(POSITION_REGIONS, 0)
    for item in positions:
        point = item.court_point
        if not item.metric_eligible or point is None or not _inside_court(point, court):
            continue
        region = position_region(
            point,
            court,
            transition_zone_depth_m=configuration.transition_zone_depth_m,
        )
        counts[region] += 1
    denominator = sum(counts.values())
    return {
        region: {
            "frameCount": count,
            "seconds": count / fps,
            "shareOfInCourtFrames": count / denominator if denominator else None,
            "denominatorInCourtFrameCount": denominator,
            "calculationVersion": MATCH_ANALYTICS_VERSION,
        }
        for region, count in counts.items()
    }


def _partner_spacing(
    first: tuple[AnalyticsPosition, ...],
    second: tuple[AnalyticsPosition, ...],
    *,
    source_frame_count: int,
) -> dict[str, object]:
    first_by_frame = {item.frame_number: item for item in first}
    values: list[float] = []
    for second_position in second:
        first_position = first_by_frame.get(second_position.frame_number)
        if (
            first_position is None
            or not first_position.metric_eligible
            or not second_position.metric_eligible
            or first_position.court_point is None
            or second_position.court_point is None
        ):
            continue
        values.append(
            math.hypot(
                first_position.court_point.x_m - second_position.court_point.x_m,
                first_position.court_point.y_m - second_position.court_point.y_m,
            )
        )
    return {
        "value": _mean(values),
        "unit": "meters",
        "sampleCount": len(values),
        "sourceFrameCoverageRatio": len(values) / source_frame_count,
        "calculationVersion": MATCH_ANALYTICS_VERSION,
    }


def _shot_position_region(
    shot: AnalyticsShot,
    court: CourtDimensions,
    configuration: PositionMetricConfiguration,
) -> str | None:
    point = shot.hitter_court_position
    if point is None or shot.hitter_court_region not in {"inside", "near"}:
        return None
    return position_region(
        point,
        court,
        transition_zone_depth_m=configuration.transition_zone_depth_m,
    )


def _at_kitchen(
    player_id: str,
    point: CourtPoint,
    *,
    court: CourtDimensions,
    arrival_distance_m: float,
) -> bool:
    if not _inside_court(point, court):
        return False
    if player_id in {"ME", "PARTNER"}:
        return (
            point.y_m <= court.net_y_m and point.y_m >= court.near_kitchen_y_m - arrival_distance_m
        )
    return point.y_m >= court.net_y_m and point.y_m <= court.far_kitchen_y_m + arrival_distance_m


def _team_kitchen_arrival(
    rallies: tuple[AnalyticsRally, ...],
    positions_by_player: dict[str, tuple[AnalyticsPosition, ...]],
    team: tuple[str, str],
    *,
    court: CourtDimensions,
    source_fps: float,
    configuration: MatchAnalyticsConfiguration,
) -> dict[str, object]:
    first_by_frame = {item.frame_number: item for item in positions_by_player[team[0]]}
    second_by_frame = {item.frame_number: item for item in positions_by_player[team[1]]}
    evaluable_count = 0
    arrival_count = 0
    excluded_low_coverage_count = 0
    rally_results: list[dict[str, object]] = []
    for rally in rallies:
        expected_frames = max(1, rally.end_frame - rally.start_frame + 1)
        joint: list[tuple[AnalyticsPosition, AnalyticsPosition]] = []
        for frame_number in range(rally.start_frame, rally.end_frame + 1):
            first = first_by_frame.get(frame_number)
            second = second_by_frame.get(frame_number)
            if (
                first is not None
                and second is not None
                and first.metric_eligible
                and second.metric_eligible
                and first.court_point is not None
                and second.court_point is not None
            ):
                joint.append((first, second))
        coverage = len(joint) / expected_frames
        evaluable = coverage >= configuration.minimum_kitchen_arrival_joint_coverage_ratio
        arrived = False
        arrival_frame: int | None = None
        if evaluable:
            evaluable_count += 1
            for first, second in joint:
                assert first.court_point is not None
                assert second.court_point is not None
                if _at_kitchen(
                    team[0],
                    first.court_point,
                    court=court,
                    arrival_distance_m=configuration.kitchen_arrival_distance_m,
                ) and _at_kitchen(
                    team[1],
                    second.court_point,
                    court=court,
                    arrival_distance_m=configuration.kitchen_arrival_distance_m,
                ):
                    arrived = True
                    arrival_frame = first.frame_number
                    arrival_count += 1
                    break
        else:
            excluded_low_coverage_count += 1
        rally_results.append(
            {
                "rallyId": rally.rally_id,
                "evaluable": evaluable,
                "jointPositionCoverageRatio": coverage,
                "arrived": arrived if evaluable else None,
                "arrivalFrame": arrival_frame,
                "arrivalTimestamp": (
                    arrival_frame / source_fps if arrival_frame is not None else None
                ),
            }
        )
    return {
        **_rate_metric(
            arrival_count,
            evaluable_count,
            numerator_population="evaluable rallies where both teammates reached the kitchen",
            denominator_population="rallies meeting joint player-position coverage threshold",
        ),
        "excludedLowCoverageRallyCount": excluded_low_coverage_count,
        "rallies": rally_results,
    }


def compute_match_analytics(
    *,
    rallies: tuple[AnalyticsRally, ...],
    shots: tuple[AnalyticsShot, ...],
    positions_by_player: dict[str, tuple[AnalyticsPosition, ...]],
    player_names: dict[str, str],
    court: CourtDimensions,
    source_fps: float,
    source_frame_count: int,
    position_configuration: PositionMetricConfiguration,
    configuration: MatchAnalyticsConfiguration,
) -> dict[str, object]:
    """Compute deterministic metrics without accessing model or waveform data."""

    shots_by_rally: dict[str, list[AnalyticsShot]] = {rally.rally_id: [] for rally in rallies}
    for shot in shots:
        shots_by_rally[shot.rally_id].append(shot)
    longest = max(
        rallies,
        key=lambda rally: (
            len(shots_by_rally[rally.rally_id]),
            rally.duration_s,
            -rally.start_frame,
        ),
        default=None,
    )
    match_metrics: dict[str, object] = {
        "rallyCount": _count_metric(len(rallies), "validated structured Rally records"),
        "shotCount": _count_metric(len(shots), "validated structured Shot records"),
        "averageRallyDuration": {
            "value": _mean([rally.duration_s for rally in rallies]),
            "unit": "seconds",
            "rallyCount": len(rallies),
            "calculationVersion": MATCH_ANALYTICS_VERSION,
        },
        "averageRallyLength": {
            "value": len(shots) / len(rallies) if rallies else None,
            "unit": "shots_per_rally",
            "shotCount": len(shots),
            "rallyCount": len(rallies),
            "calculationVersion": MATCH_ANALYTICS_VERSION,
        },
        "longestRally": (
            {
                "rallyId": longest.rally_id,
                "shotCount": len(shots_by_rally[longest.rally_id]),
                "durationSeconds": longest.duration_s,
                "startFrame": longest.start_frame,
                "endFrame": longest.end_frame,
                "selectionRule": "maximum shot count, then duration, then earliest start",
                "calculationVersion": MATCH_ANALYTICS_VERSION,
            }
            if longest is not None
            else None
        ),
    }

    spacing_by_player: dict[str, dict[str, object]] = {}
    team_metrics: dict[str, object] = {}
    for team_name, team in PLAYER_TEAMS.items():
        spacing = _partner_spacing(
            positions_by_player[team[0]],
            positions_by_player[team[1]],
            source_frame_count=source_frame_count,
        )
        spacing_by_player[team[0]] = spacing
        spacing_by_player[team[1]] = spacing
        team_metrics[team_name] = {
            "players": list(team),
            "averagePartnerSpacing": spacing,
            "kitchenArrivalRate": _team_kitchen_arrival(
                rallies,
                positions_by_player,
                team,
                court=court,
                source_fps=source_fps,
                configuration=configuration,
            ),
        }

    player_metrics: dict[str, object] = {}
    for player_id in LOGICAL_PLAYERS:
        player_shots = [shot for shot in shots if shot.hitter_id == player_id]
        classified = [shot for shot in player_shots if shot.shot_type is not ShotType.UNKNOWN]
        type_metrics: dict[str, object] = {}
        for shot_type in COUNTED_SHOT_TYPES:
            count = sum(shot.shot_type is shot_type for shot in classified)
            type_metrics[shot_type.value] = {
                "count": count,
                "rate": count / len(classified) if classified else None,
                "rateDenominatorClassifiedHits": len(classified),
                "calculationVersion": MATCH_ANALYTICS_VERSION,
            }
        player_positions = positions_by_player[player_id]
        player_metrics[player_id] = {
            "displayName": player_names[player_id],
            "totalHits": _count_metric(
                len(player_shots),
                "Shot records with this logical hitter identity, including UNKNOWN shot type",
            ),
            "classifiedHitCount": len(classified),
            "unknownShotTypeHitCount": len(player_shots) - len(classified),
            "shotTypes": type_metrics,
            "positions": {
                "distanceTraveled": _distance_traveled(
                    player_positions,
                    position_configuration,
                ),
                "courtOccupancy": _occupancy(
                    player_positions,
                    court=court,
                    configuration=position_configuration,
                    fps=source_fps,
                ),
                "averagePartnerSpacing": spacing_by_player[player_id],
            },
        }

    third_shots = [shot for shot in shots if shot.shot_index == 3]
    classified_third_shots = [
        shot for shot in third_shots if shot.shot_type is not ShotType.UNKNOWN
    ]
    drop_count = sum(shot.shot_type is ShotType.DROP for shot in classified_third_shots)
    drive_count = sum(shot.shot_type is ShotType.DRIVE for shot in classified_third_shots)
    selection_counts: dict[str, dict[str, int]] = {
        region: {shot_type.value: 0 for shot_type in ShotType if shot_type is not ShotType.UNKNOWN}
        for region in POSITION_REGIONS
    }
    missing_position_count = 0
    unknown_type_with_position_count = 0
    for shot in shots:
        region = _shot_position_region(shot, court, position_configuration)
        if region is None:
            missing_position_count += 1
            continue
        if shot.shot_type is ShotType.UNKNOWN:
            unknown_type_with_position_count += 1
            continue
        selection_counts[region][shot.shot_type.value] += 1
    shot_selection: dict[str, object] = {}
    for region in POSITION_REGIONS:
        denominator = sum(selection_counts[region].values())
        shot_selection[region] = {
            "classifiedShotCount": denominator,
            "counts": selection_counts[region],
            "rates": {
                shot_type: count / denominator if denominator else None
                for shot_type, count in selection_counts[region].items()
            },
            "calculationVersion": MATCH_ANALYTICS_VERSION,
        }

    return {
        "analyticsVersion": MATCH_ANALYTICS_VERSION,
        "match": match_metrics,
        "players": player_metrics,
        "teams": team_metrics,
        "tactical": {
            "thirdShotDropRate": _rate_metric(
                drop_count,
                len(classified_third_shots),
                numerator_population="classified third shots labeled DROP",
                denominator_population="all third shots whose shot type is not UNKNOWN",
            ),
            "thirdShotDriveRate": _rate_metric(
                drive_count,
                len(classified_third_shots),
                numerator_population="classified third shots labeled DRIVE",
                denominator_population="all third shots whose shot type is not UNKNOWN",
            ),
            "thirdShotDataQuality": {
                "thirdShotCount": len(third_shots),
                "classifiedThirdShotCount": len(classified_third_shots),
                "unknownThirdShotCount": len(third_shots) - len(classified_third_shots),
            },
            "shotSelectionByCourtPosition": {
                "regions": shot_selection,
                "excludedMissingOrAmbiguousPositionCount": missing_position_count,
                "excludedUnknownShotTypeCount": unknown_type_with_position_count,
            },
        },
        "dataQuality": {
            "unknownHitterShotCount": sum(shot.hitter_id == "UNKNOWN" for shot in shots),
            "unknownShotTypeCount": sum(shot.shot_type is ShotType.UNKNOWN for shot in shots),
            "meanRallyConfidence": _mean([rally.confidence for rally in rallies]),
            "meanShotConfidence": _mean([shot.confidence for shot in shots]),
            "meanKnownHitterConfidence": _mean(
                [shot.hitter_confidence for shot in shots if shot.hitter_id in KNOWN_PLAYER_IDS]
            ),
            "playerPositionCoverage": {
                player_id: {
                    "eligibleFrameCount": sum(
                        position.metric_eligible for position in positions_by_player[player_id]
                    ),
                    "sourceFrameCount": source_frame_count,
                    "ratio": sum(
                        position.metric_eligible for position in positions_by_player[player_id]
                    )
                    / source_frame_count,
                }
                for player_id in LOGICAL_PLAYERS
            },
        },
    }
