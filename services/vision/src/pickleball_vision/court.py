"""Explicit canonical doubles-pickleball court geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

COURT_WIDTH_METERS = 6.096
COURT_LENGTH_METERS = 13.4112
NON_VOLLEY_ZONE_DEPTH_METERS = 2.1336


@dataclass(frozen=True, slots=True)
class CourtPoint:
    """A point on the canonical court plane, in meters."""

    x_m: float
    y_m: float

    def as_dict(self) -> dict[str, float]:
        return {"x_m": self.x_m, "y_m": self.y_m}


@dataclass(frozen=True, slots=True)
class ImagePoint:
    """A point in a decoded video frame, in pixels."""

    x_px: float
    y_px: float

    def as_dict(self) -> dict[str, float]:
        return {"x_px": self.x_px, "y_px": self.y_px}


@dataclass(frozen=True, slots=True)
class CourtDimensions:
    """Canonical court dimensions with explicit metric units."""

    width_m: float = COURT_WIDTH_METERS
    length_m: float = COURT_LENGTH_METERS
    non_volley_zone_depth_m: float = NON_VOLLEY_ZONE_DEPTH_METERS

    def __post_init__(self) -> None:
        values = (self.width_m, self.length_m, self.non_volley_zone_depth_m)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("Court dimensions must be finite positive values")
        if self.non_volley_zone_depth_m * 2 >= self.length_m:
            raise ValueError("Non-volley-zone depth must be less than half the court length")

    @property
    def center_x_m(self) -> float:
        return self.width_m / 2

    @property
    def net_y_m(self) -> float:
        return self.length_m / 2

    @property
    def near_kitchen_y_m(self) -> float:
        return self.net_y_m - self.non_volley_zone_depth_m

    @property
    def far_kitchen_y_m(self) -> float:
        return self.net_y_m + self.non_volley_zone_depth_m

    def as_dict(self) -> dict[str, object]:
        return {
            "unit": "meters",
            "width_m": self.width_m,
            "length_m": self.length_m,
            "non_volley_zone_depth_m": self.non_volley_zone_depth_m,
        }


class LandmarkName(StrEnum):
    """Supported named court-plane calibration landmarks."""

    NEAR_BASELINE_LEFT = "near_baseline_left"
    NEAR_BASELINE_RIGHT = "near_baseline_right"
    NEAR_KITCHEN_LEFT = "near_kitchen_left"
    NEAR_KITCHEN_RIGHT = "near_kitchen_right"
    NEAR_CENTERLINE_KITCHEN = "near_centerline_kitchen_intersection"
    FAR_KITCHEN_LEFT = "far_kitchen_left"
    FAR_KITCHEN_RIGHT = "far_kitchen_right"
    FAR_CENTERLINE_KITCHEN = "far_centerline_kitchen_intersection"
    FAR_BASELINE_LEFT = "far_baseline_left"
    FAR_BASELINE_RIGHT = "far_baseline_right"


@dataclass(frozen=True, slots=True)
class CourtLandmark:
    """A named point in canonical court coordinates."""

    name: LandmarkName
    label: str
    court_point: CourtPoint


def court_landmarks(dimensions: CourtDimensions) -> tuple[CourtLandmark, ...]:
    """Return the ordered landmark prompts for manual calibration."""

    left = 0.0
    center = dimensions.center_x_m
    right = dimensions.width_m
    near_baseline = 0.0
    near_kitchen = dimensions.near_kitchen_y_m
    far_kitchen = dimensions.far_kitchen_y_m
    far_baseline = dimensions.length_m
    return (
        CourtLandmark(
            LandmarkName.NEAR_BASELINE_LEFT,
            "near baseline left",
            CourtPoint(left, near_baseline),
        ),
        CourtLandmark(
            LandmarkName.NEAR_BASELINE_RIGHT,
            "near baseline right",
            CourtPoint(right, near_baseline),
        ),
        CourtLandmark(
            LandmarkName.NEAR_KITCHEN_LEFT,
            "near kitchen left",
            CourtPoint(left, near_kitchen),
        ),
        CourtLandmark(
            LandmarkName.NEAR_KITCHEN_RIGHT,
            "near kitchen right",
            CourtPoint(right, near_kitchen),
        ),
        CourtLandmark(
            LandmarkName.NEAR_CENTERLINE_KITCHEN,
            "near centerline/kitchen intersection",
            CourtPoint(center, near_kitchen),
        ),
        CourtLandmark(
            LandmarkName.FAR_KITCHEN_LEFT,
            "far kitchen left",
            CourtPoint(left, far_kitchen),
        ),
        CourtLandmark(
            LandmarkName.FAR_KITCHEN_RIGHT,
            "far kitchen right",
            CourtPoint(right, far_kitchen),
        ),
        CourtLandmark(
            LandmarkName.FAR_CENTERLINE_KITCHEN,
            "far centerline/kitchen intersection",
            CourtPoint(center, far_kitchen),
        ),
        CourtLandmark(
            LandmarkName.FAR_BASELINE_LEFT,
            "far baseline left",
            CourtPoint(left, far_baseline),
        ),
        CourtLandmark(
            LandmarkName.FAR_BASELINE_RIGHT,
            "far baseline right",
            CourtPoint(right, far_baseline),
        ),
    )


def court_line_segments(
    dimensions: CourtDimensions,
) -> tuple[tuple[str, CourtPoint, CourtPoint], ...]:
    """Return canonical court markings as named line segments."""

    width = dimensions.width_m
    length = dimensions.length_m
    center = dimensions.center_x_m
    near_kitchen = dimensions.near_kitchen_y_m
    far_kitchen = dimensions.far_kitchen_y_m
    net = dimensions.net_y_m
    return (
        ("near_baseline", CourtPoint(0, 0), CourtPoint(width, 0)),
        ("right_sideline", CourtPoint(width, 0), CourtPoint(width, length)),
        ("far_baseline", CourtPoint(width, length), CourtPoint(0, length)),
        ("left_sideline", CourtPoint(0, length), CourtPoint(0, 0)),
        ("near_kitchen", CourtPoint(0, near_kitchen), CourtPoint(width, near_kitchen)),
        ("net", CourtPoint(0, net), CourtPoint(width, net)),
        ("far_kitchen", CourtPoint(0, far_kitchen), CourtPoint(width, far_kitchen)),
        ("near_centerline", CourtPoint(center, 0), CourtPoint(center, near_kitchen)),
        ("far_centerline", CourtPoint(center, far_kitchen), CourtPoint(center, length)),
    )
