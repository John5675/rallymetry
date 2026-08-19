from __future__ import annotations

from typing import cast

import pytest

from pickleball_vision.court import CourtDimensions, CourtPoint
from pickleball_vision.match_analytics import (
    LOGICAL_PLAYERS,
    AnalyticsPosition,
    AnalyticsRally,
    AnalyticsShot,
    MatchAnalyticsConfiguration,
    PositionMetricConfiguration,
    compute_match_analytics,
    position_region,
)
from pickleball_vision.shot_reconstruction import ShotType


def _position_series(player_id: str, *, x_m: float, y_m: float) -> tuple[AnalyticsPosition, ...]:
    return tuple(
        AnalyticsPosition(
            player_id=player_id,
            frame_number=frame,
            timestamp_s=frame / 10,
            confidence=0.9,
            court_point=CourtPoint(x_m + frame * 0.1, y_m),
            court_region="inside",
        )
        for frame in range(10)
    )


def _shot(
    shot_id: str,
    rally_id: str,
    index: int,
    hitter: str,
    shot_type: ShotType,
    *,
    point: CourtPoint | None = None,
) -> AnalyticsShot:
    return AnalyticsShot(
        shot_id,
        rally_id,
        index,
        hitter,
        shot_type,
        0.8,
        0.75,
        point,
        "inside" if point is not None else None,
    )


def _analytics() -> dict[str, object]:
    rallies = (
        AnalyticsRally("rally-1", 0.0, 0.4, 0, 4, 0.9),
        AnalyticsRally("rally-2", 0.5, 0.9, 5, 9, 0.8),
    )
    shots = (
        _shot("s1", "rally-1", 1, "ME", ShotType.SERVE, point=CourtPoint(1, 1)),
        _shot("s2", "rally-1", 2, "OPPONENT_1", ShotType.RETURN),
        _shot("s3", "rally-1", 3, "PARTNER", ShotType.DROP, point=CourtPoint(2, 3)),
        _shot("s4", "rally-2", 1, "OPPONENT_2", ShotType.SERVE),
        _shot("s5", "rally-2", 2, "ME", ShotType.DRIVE, point=CourtPoint(2, 4.7)),
    )
    positions: dict[str, tuple[AnalyticsPosition, ...]] = {
        "ME": _position_series("ME", x_m=1, y_m=4.6),
        "PARTNER": _position_series("PARTNER", x_m=3, y_m=4.6),
        "OPPONENT_1": _position_series("OPPONENT_1", x_m=1, y_m=8.8),
        "OPPONENT_2": _position_series("OPPONENT_2", x_m=3, y_m=8.8),
    }
    return compute_match_analytics(
        rallies=rallies,
        shots=shots,
        positions_by_player=positions,
        player_names={player: player.title() for player in LOGICAL_PLAYERS},
        court=CourtDimensions(),
        source_fps=10,
        source_frame_count=10,
        position_configuration=PositionMetricConfiguration(0.2, 8.0, 2.1336),
        configuration=MatchAnalyticsConfiguration(0.9, 0.5),
    )


def test_match_and_player_metrics_are_deterministic() -> None:
    result = _analytics()
    match = cast(dict[str, object], result["match"])
    rally_count = cast(dict[str, object], match["rallyCount"])
    shot_count = cast(dict[str, object], match["shotCount"])
    average_duration = cast(dict[str, object], match["averageRallyDuration"])
    average_length = cast(dict[str, object], match["averageRallyLength"])
    longest = cast(dict[str, object], match["longestRally"])

    assert rally_count["value"] == 2
    assert shot_count["value"] == 5
    assert average_duration["value"] == pytest.approx(0.4)
    assert average_length["value"] == pytest.approx(2.5)
    assert longest["rallyId"] == "rally-1"
    assert longest["shotCount"] == 3

    players = cast(dict[str, dict[str, object]], result["players"])
    me = players["ME"]
    assert cast(dict[str, object], me["totalHits"])["value"] == 2
    types = cast(dict[str, dict[str, object]], me["shotTypes"])
    assert types["DRIVE"]["count"] == 1
    assert types["DRIVE"]["rate"] == pytest.approx(0.5)
    positions = cast(dict[str, object], me["positions"])
    distance = cast(dict[str, object], positions["distanceTraveled"])
    spacing = cast(dict[str, object], positions["averagePartnerSpacing"])
    assert distance["value"] == pytest.approx(0.9)
    assert spacing["value"] == pytest.approx(2.0)


def test_tactical_metrics_have_explicit_denominators() -> None:
    result = _analytics()
    tactical = cast(dict[str, object], result["tactical"])
    drop = cast(dict[str, object], tactical["thirdShotDropRate"])
    drive = cast(dict[str, object], tactical["thirdShotDriveRate"])

    assert drop["numerator"] == 1
    assert drop["denominator"] == 1
    assert drop["value"] == pytest.approx(1.0)
    assert drive["value"] == pytest.approx(0.0)

    teams = cast(dict[str, dict[str, object]], result["teams"])
    near_arrival = cast(dict[str, object], teams["nearTeam"]["kitchenArrivalRate"])
    assert near_arrival["numerator"] == 2
    assert near_arrival["denominator"] == 2
    assert near_arrival["value"] == pytest.approx(1.0)


def test_unknown_values_are_not_fabricated_into_classified_rates() -> None:
    result = compute_match_analytics(
        rallies=(AnalyticsRally("r1", 0, 0.1, 0, 1, 0.7),),
        shots=(_shot("s1", "r1", 1, "UNKNOWN", ShotType.UNKNOWN),),
        positions_by_player={
            player: (
                AnalyticsPosition(player, 0, 0, 0, None, "ambiguous"),
                AnalyticsPosition(player, 1, 0.1, 0, None, "ambiguous"),
            )
            for player in LOGICAL_PLAYERS
        },
        player_names={player: player for player in LOGICAL_PLAYERS},
        court=CourtDimensions(),
        source_fps=10,
        source_frame_count=2,
        position_configuration=PositionMetricConfiguration(0.2, 8, 2.1336),
        configuration=MatchAnalyticsConfiguration(),
    )
    quality = cast(dict[str, object], result["dataQuality"])
    players = cast(dict[str, dict[str, object]], result["players"])
    me_types = cast(dict[str, dict[str, object]], players["ME"]["shotTypes"])

    assert quality["unknownHitterShotCount"] == 1
    assert quality["unknownShotTypeCount"] == 1
    assert me_types["DINK"]["rate"] is None
    assert cast(dict[str, object], players["ME"]["totalHits"])["value"] == 0


def test_empty_populations_return_null_and_retain_confidence_quality() -> None:
    result = compute_match_analytics(
        rallies=(),
        shots=(),
        positions_by_player={player: () for player in LOGICAL_PLAYERS},
        player_names={player: player for player in LOGICAL_PLAYERS},
        court=CourtDimensions(),
        source_fps=10,
        source_frame_count=10,
        position_configuration=PositionMetricConfiguration(0.2, 8, 2.1336),
        configuration=MatchAnalyticsConfiguration(),
    )
    match = cast(dict[str, object], result["match"])
    quality = cast(dict[str, object], result["dataQuality"])

    assert cast(dict[str, object], match["averageRallyDuration"])["value"] is None
    assert cast(dict[str, object], match["averageRallyLength"])["value"] is None
    assert match["longestRally"] is None
    assert quality["meanRallyConfidence"] is None
    assert quality["meanShotConfidence"] is None


def test_distance_rejects_nonconsecutive_and_implausible_steps() -> None:
    positions: dict[str, tuple[AnalyticsPosition, ...]] = {
        player: (
            AnalyticsPosition(player, 0, 0.0, 0.9, CourtPoint(0, 1), "inside"),
            AnalyticsPosition(player, 1, 0.1, 0.9, CourtPoint(2, 1), "inside"),
            AnalyticsPosition(player, 3, 0.3, 0.9, CourtPoint(2.1, 1), "inside"),
        )
        for player in LOGICAL_PLAYERS
    }
    result = compute_match_analytics(
        rallies=(),
        shots=(),
        positions_by_player=positions,
        player_names={player: player for player in LOGICAL_PLAYERS},
        court=CourtDimensions(),
        source_fps=10,
        source_frame_count=4,
        position_configuration=PositionMetricConfiguration(0.2, 8, 2.1336),
        configuration=MatchAnalyticsConfiguration(),
    )
    players = cast(dict[str, dict[str, object]], result["players"])
    position_metrics = cast(dict[str, object], players["ME"]["positions"])
    distance = cast(dict[str, object], position_metrics["distanceTraveled"])

    assert distance["value"] is None
    assert distance["contributingStepCount"] == 0
    assert distance["excludedSpeedStepCount"] == 1
    assert distance["excludedGapStepCount"] == 1


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        (CourtPoint(3, 5), "kitchen"),
        (CourtPoint(3, 3), "transitionZone"),
        (CourtPoint(3, 1), "backcourt"),
        (CourtPoint(3, 10), "transitionZone"),
        (CourtPoint(3, 12), "backcourt"),
    ],
)
def test_position_region_uses_explicit_court_geometry(
    point: CourtPoint,
    expected: str,
) -> None:
    assert (
        position_region(
            point,
            CourtDimensions(),
            transition_zone_depth_m=2.1336,
        )
        == expected
    )
