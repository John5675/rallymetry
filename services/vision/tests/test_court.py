from collections.abc import Callable

import pytest

from pickleball_vision.court import (
    CourtDimensions,
    LandmarkName,
    court_landmarks,
    court_line_segments,
)


def test_default_court_geometry_is_explicit() -> None:
    court = CourtDimensions()

    assert court.width_m == pytest.approx(6.096)
    assert court.length_m == pytest.approx(13.4112)
    assert court.non_volley_zone_depth_m == pytest.approx(2.1336)
    assert court.net_y_m == pytest.approx(6.7056)
    assert court.near_kitchen_y_m == pytest.approx(4.572)
    assert court.far_kitchen_y_m == pytest.approx(8.8392)


def test_landmark_catalog_contains_named_multi_point_options() -> None:
    landmarks = court_landmarks(CourtDimensions())

    assert len(landmarks) == 10
    assert landmarks[0].name is LandmarkName.NEAR_BASELINE_LEFT
    assert landmarks[0].court_point.x_m == 0
    assert landmarks[0].court_point.y_m == 0
    assert landmarks[4].name is LandmarkName.NEAR_CENTERLINE_KITCHEN
    assert landmarks[-1].name is LandmarkName.FAR_BASELINE_RIGHT


def test_court_geometry_contains_boundaries_kitchens_net_and_centerlines() -> None:
    segment_names = {name for name, _, _ in court_line_segments(CourtDimensions())}

    assert segment_names == {
        "near_baseline",
        "right_sideline",
        "far_baseline",
        "left_sideline",
        "near_kitchen",
        "net",
        "far_kitchen",
        "near_centerline",
        "far_centerline",
    }


@pytest.mark.parametrize(
    "court",
    [
        lambda: CourtDimensions(width_m=0),
        lambda: CourtDimensions(length_m=float("nan")),
        lambda: CourtDimensions(non_volley_zone_depth_m=7),
    ],
)
def test_invalid_court_dimensions_are_rejected(court: Callable[[], CourtDimensions]) -> None:
    with pytest.raises(ValueError):
        court()
