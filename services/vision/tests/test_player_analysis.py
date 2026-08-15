from pathlib import Path

import pytest

from pickleball_vision.config import PlayerAnalysisSettings
from pickleball_vision.court import CourtDimensions, CourtPoint, ImagePoint
from pickleball_vision.player_analysis import (
    ManualCourtPositionCorrection,
    PlayerPositionFrame,
    PositionSmoothingStatus,
    RawPlayerPositionFrame,
    build_player_analysis_run,
    build_player_analysis_summary,
    heatmap_eligible,
    smooth_player_positions,
)
from pickleball_vision.player_analysis_render import (
    TOPDOWN_HEIGHT_PX,
    TOPDOWN_WIDTH_PX,
    render_player_heatmap,
)
from pickleball_vision.player_isolation import (
    LOGICAL_PLAYER_ROLES,
    CourtRegionState,
    LogicalPlayerRole,
)
from pickleball_vision.player_tracking import LogicalTrackingState
from pickleball_vision.video import VideoMetadata


def _source(frame_count: int, *, fps: float = 1.0) -> VideoMetadata:
    return VideoMetadata(
        "test.mp4",
        Path("/tmp/test.mp4"),
        1920,
        1080,
        fps,
        frame_count,
        frame_count / fps,
        "h264",
    )


def _raw(
    role: LogicalPlayerRole,
    frame_number: int,
    point: CourtPoint | None,
    *,
    confidence: float = 0.9,
    state: LogicalTrackingState = LogicalTrackingState.OBSERVED,
    region: CourtRegionState | None = None,
) -> RawPlayerPositionFrame:
    return RawPlayerPositionFrame(
        role,
        frame_number,
        float(frame_number),
        state,
        confidence,
        ImagePoint(100 + frame_number, 900) if point is not None else None,
        point,
        region
        if region is not None
        else CourtRegionState.INSIDE
        if point is not None
        else CourtRegionState.AMBIGUOUS,
        f"raw-{role.value}-{frame_number}" if point is not None else None,
        "bounding_box_bottom_center" if point is not None else None,
    )


def _position(raw: RawPlayerPositionFrame) -> PlayerPositionFrame:
    return PlayerPositionFrame(
        raw,
        raw.raw_court_coordinate,
        raw.raw_court_coordinate,
        PositionSmoothingStatus.SMOOTHED,
        (raw.frame_number,),
    )


def test_smoothing_preserves_raw_coordinates_and_bounds_adjustment() -> None:
    role = LogicalPlayerRole.ME
    raw = tuple(
        _raw(role, frame, CourtPoint(x_m, 2.0))
        for frame, x_m in enumerate((1.0, 1.0, 5.0, 1.0, 1.0))
    )
    settings = PlayerAnalysisSettings(maximum_smoothing_adjustment_m=0.3)

    smoothed = smooth_player_positions(raw, settings=settings)

    assert smoothed[2].raw.raw_court_coordinate == CourtPoint(5.0, 2.0)
    assert smoothed[2].smoothed_court_coordinate is not None
    assert smoothed[2].smoothed_court_coordinate.x_m == pytest.approx(4.7)
    assert smoothed[2].smoothing_support_frames == (0, 1, 2, 3, 4)
    serialized = smoothed[2].as_dict()
    assert serialized["raw_court_coordinate"] == {"x_m": 5.0, "y_m": 2.0}
    assert serialized["smoothing"]["raw_coordinate_preserved"] is True  # type: ignore[index]


def test_manual_court_correction_is_separate_and_player_specific() -> None:
    role = LogicalPlayerRole.OPPONENT_1
    raw = (_raw(role, 0, CourtPoint(4.0, 10.0)),)
    correction = ManualCourtPositionCorrection(
        y_offset_m=0.15,
        reason="align far-side foot contact",
    )

    corrected = smooth_player_positions(
        raw,
        settings=PlayerAnalysisSettings(),
        correction=correction,
    )[0]

    assert corrected.raw.raw_court_coordinate == CourtPoint(4.0, 10.0)
    assert corrected.corrected_court_coordinate == CourtPoint(4.0, 10.15)
    assert corrected.smoothed_court_coordinate == CourtPoint(4.0, 10.15)
    serialized = corrected.as_dict()
    assert serialized["raw_court_coordinate"] == {"x_m": 4.0, "y_m": 10.0}
    assert serialized["corrected_court_coordinate"] == {"x_m": 4.0, "y_m": 10.15}


def test_smoothing_never_interpolates_missing_or_identity_switch_frames() -> None:
    role = LogicalPlayerRole.ME
    raw = (
        _raw(role, 0, CourtPoint(1, 2)),
        _raw(role, 1, None),
        _raw(
            role,
            2,
            CourtPoint(1.2, 2),
            state=LogicalTrackingState.SUSPECTED_IDENTITY_SWITCH,
        ),
        _raw(role, 3, CourtPoint(1.3, 2)),
    )

    smoothed = smooth_player_positions(raw, settings=PlayerAnalysisSettings())

    assert smoothed[0].smoothing_support_frames == (0,)
    assert smoothed[1].smoothed_court_coordinate is None
    assert smoothed[1].smoothing_status is PositionSmoothingStatus.MISSING_RAW_COURT_COORDINATE
    assert smoothed[2].smoothed_court_coordinate is None
    assert smoothed[2].smoothing_status is PositionSmoothingStatus.SUSPECTED_IDENTITY_SWITCH
    assert smoothed[3].smoothing_support_frames == (3,)


def test_summary_metrics_use_quality_gated_structured_positions() -> None:
    court = CourtDimensions()
    source = _source(4)
    settings = PlayerAnalysisSettings(maximum_step_gap_seconds=1.1)
    coordinates = {
        LogicalPlayerRole.ME: ((1, 4.8), (2, 4.0), (3, 1.0), (4, 1.0)),
        LogicalPlayerRole.PARTNER: ((3, 4.8), (4, 4.0), (5, 1.0), (6, 1.0)),
        LogicalPlayerRole.OPPONENT_1: ((1, 8.6), (2, 9.2), (3, 12.0), (4, 12.0)),
        LogicalPlayerRole.OPPONENT_2: ((3, 8.6), (4, 9.2), (5, 12.0), (6, 12.0)),
    }
    positions = {
        role: tuple(
            _position(_raw(role, frame, CourtPoint(*point)))
            for frame, point in enumerate(coordinates[role])
        )
        for role in LOGICAL_PLAYER_ROLES
    }
    run = build_player_analysis_run(
        source=source,
        court=court,
        tracking_path="/tmp/tracks.json",
        calibration_path="/tmp/calibration.json",
        settings=settings,
        player_names={role: role.value for role in LOGICAL_PLAYER_ROLES},
        positions=positions,
    )

    summary = build_player_analysis_summary(run, artifacts={})
    me = summary["players"]["ME"]  # type: ignore[index]
    metrics = me["metrics"]
    occupancy = metrics["court_occupancy"]
    assert occupancy["kitchen"]["share_of_in_court_frames"] == pytest.approx(0.25)
    assert occupancy["transition_zone"]["share_of_in_court_frames"] == pytest.approx(0.25)
    assert occupancy["backcourt"]["share_of_in_court_frames"] == pytest.approx(0.50)
    assert metrics["approximate_distance_traveled"]["value_m"] == pytest.approx(
        (1.64**0.5) + (10**0.5) + 1
    )
    lateral = metrics["lateral_movement"]
    assert lateral["total_absolute_lateral_distance_m"] == pytest.approx(3.0)
    assert lateral["mean_absolute_lateral_speed_mps"] == pytest.approx(1.0)
    assert lateral["lateral_range_m"] == pytest.approx(3.0)
    assert metrics["average_partner_spacing"]["value_m"] == pytest.approx(2.0)
    assert metrics["average_distance_from_kitchen"]["sample_count"] == 4


def test_heatmap_uses_only_smoothed_in_court_positions() -> None:
    role = LogicalPlayerRole.ME
    frames = tuple(
        _position(_raw(role, frame, CourtPoint(1 + frame, 2 + frame))) for frame in range(3)
    )

    image = render_player_heatmap(
        role,
        frames,
        court=CourtDimensions(),
        display_name="John",
    )

    assert image.shape == (TOPDOWN_HEIGHT_PX, TOPDOWN_WIDTH_PX, 3)
    assert image.max() > image.min()
    excluded = _position(
        _raw(
            role,
            3,
            CourtPoint(2, 3),
            region=CourtRegionState.OUTSIDE,
        )
    )
    assert heatmap_eligible(excluded, CourtDimensions()) is False
