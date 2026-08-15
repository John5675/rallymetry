"""Source-space, top-down, and heatmap rendering for Release 0.1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from pickleball_vision.calibration import CourtCalibration
from pickleball_vision.court import CourtDimensions, CourtPoint, court_line_segments
from pickleball_vision.player_analysis import PlayerPositionFrame, heatmap_eligible
from pickleball_vision.player_isolation import LOGICAL_PLAYER_ROLES, LogicalPlayerRole
from pickleball_vision.video import Image

TOPDOWN_WIDTH_PX = 720
TOPDOWN_HEIGHT_PX = 1280
TOPDOWN_MARGIN_X_PX = 92
TOPDOWN_MARGIN_Y_PX = 75
BACKGROUND_COLOR = (24, 28, 31)
COURT_COLOR = (54, 88, 82)
KITCHEN_COLOR = (72, 112, 102)
TRANSITION_COLOR = (62, 94, 89)
COURT_LINE_COLOR = (235, 235, 235)
NET_COLOR = (55, 185, 255)
RAW_POINT_COLOR = (190, 190, 190)
ROLE_COLORS = {
    LogicalPlayerRole.ME: (255, 40, 220),
    LogicalPlayerRole.PARTNER: (255, 220, 40),
    LogicalPlayerRole.OPPONENT_1: (30, 120, 255),
    LogicalPlayerRole.OPPONENT_2: (40, 220, 255),
}


def _court_rect() -> tuple[int, int, int, int]:
    return (
        TOPDOWN_MARGIN_X_PX,
        TOPDOWN_MARGIN_Y_PX,
        TOPDOWN_WIDTH_PX - TOPDOWN_MARGIN_X_PX,
        TOPDOWN_HEIGHT_PX - TOPDOWN_MARGIN_Y_PX,
    )


def court_to_topdown(point: CourtPoint, court: CourtDimensions) -> tuple[int, int]:
    """Map canonical meters to a near-baseline-at-bottom top-down canvas."""

    left, top, right, bottom = _court_rect()
    x_px = left + point.x_m / court.width_m * (right - left)
    y_px = bottom - point.y_m / court.length_m * (bottom - top)
    return round(x_px), round(y_px)


def _draw_court_lines(canvas: Image, court: CourtDimensions) -> None:
    left, top, right, bottom = _court_rect()
    for name, start, end in court_line_segments(court):
        color = NET_COLOR if name == "net" else COURT_LINE_COLOR
        thickness = 3 if name == "net" else 2
        cv2.line(
            canvas,
            court_to_topdown(start, court),
            court_to_topdown(end, court),
            color,
            thickness,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "FAR BASELINE",
        (right - 154, top - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COURT_LINE_COLOR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "NEAR BASELINE",
        (left, bottom + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COURT_LINE_COLOR,
        1,
        cv2.LINE_AA,
    )


def _draw_court(canvas: Image, court: CourtDimensions) -> None:
    left, top, right, bottom = _court_rect()
    cv2.rectangle(canvas, (left, top), (right, bottom), COURT_COLOR, -1)
    near_kitchen_y = court_to_topdown(CourtPoint(0, court.near_kitchen_y_m), court)[1]
    far_kitchen_y = court_to_topdown(CourtPoint(0, court.far_kitchen_y_m), court)[1]
    cv2.rectangle(
        canvas,
        (left, far_kitchen_y),
        (right, near_kitchen_y),
        KITCHEN_COLOR,
        -1,
    )
    _draw_court_lines(canvas, court)


def render_source_analysis_frame(
    frame: Image,
    *,
    positions: Sequence[PlayerPositionFrame],
    calibration: CourtCalibration,
    player_names: Mapping[LogicalPlayerRole, str],
) -> Image:
    """Draw immutable raw ground points and separate smoothed projections."""

    canvas = frame.copy()
    scale = max(0.45, min(frame.shape[:2]) / 1500)
    missing_y = 28
    for position in positions:
        role = position.raw.logical_player
        color = ROLE_COLORS[role]
        raw_image = position.raw.raw_image_ground_point
        smooth = position.smoothed_court_coordinate
        if raw_image is None:
            cv2.putText(
                canvas,
                f"{player_names[role]}: POSITION MISSING",
                (12, missing_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (70, 70, 220),
                2,
                cv2.LINE_AA,
            )
            missing_y += 26
            continue
        raw_px = (round(raw_image.x_px), round(raw_image.y_px))
        cv2.circle(canvas, raw_px, 8, color, 2, cv2.LINE_AA)
        label_point = (raw_px[0] + 10, max(18, raw_px[1] - 10))
        if smooth is not None:
            projected = calibration.court_to_image(smooth)
            smooth_px = (round(projected.x_px), round(projected.y_px))
            cv2.line(canvas, raw_px, smooth_px, RAW_POINT_COLOR, 1, cv2.LINE_AA)
            cv2.circle(canvas, smooth_px, 5, color, -1, cv2.LINE_AA)
            label_point = (smooth_px[0] + 10, max(18, smooth_px[1] - 10))
        cv2.putText(
            canvas,
            f"{player_names[role]} | {position.raw.confidence:.2f}",
            label_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "ring = raw bottom-center | dot = corrected + smoothed court position",
        (12, frame.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return np.asarray(canvas, dtype=np.uint8)


def render_topdown_frame(
    *,
    frame_number: int,
    timestamp_s: float,
    positions: Sequence[PlayerPositionFrame],
    trails: Mapping[LogicalPlayerRole, Sequence[CourtPoint]],
    court: CourtDimensions,
    player_names: Mapping[LogicalPlayerRole, str],
) -> Image:
    """Render one canonical court-plane animation frame."""

    canvas = np.full(
        (TOPDOWN_HEIGHT_PX, TOPDOWN_WIDTH_PX, 3),
        BACKGROUND_COLOR,
        dtype=np.uint8,
    )
    _draw_court(canvas, court)
    for role in LOGICAL_PLAYER_ROLES:
        trail = trails.get(role, ())
        if len(trail) >= 2:
            trail_points = np.asarray(
                [court_to_topdown(point, court) for point in trail],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [trail_points], False, ROLE_COLORS[role], 2, cv2.LINE_AA)

    missing: list[str] = []
    for position in positions:
        role = position.raw.logical_player
        point = position.smoothed_court_coordinate
        if point is None:
            missing.append(player_names[role])
            continue
        center = court_to_topdown(point, court)
        cv2.circle(canvas, center, 14, ROLE_COLORS[role], -1, cv2.LINE_AA)
        cv2.circle(canvas, center, 17, COURT_LINE_COLOR, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            player_names[role],
            (center[0] + 20, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            ROLE_COLORS[role],
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        f"frame {frame_number} | {timestamp_s:.2f}s",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        COURT_LINE_COLOR,
        1,
        cv2.LINE_AA,
    )
    if missing:
        cv2.putText(
            canvas,
            "missing: " + ", ".join(missing),
            (18, TOPDOWN_HEIGHT_PX - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 110, 240),
            1,
            cv2.LINE_AA,
        )
    return canvas


def render_player_heatmap(
    role: LogicalPlayerRole,
    frames: Sequence[PlayerPositionFrame],
    *,
    court: CourtDimensions,
    display_name: str,
) -> Image:
    """Render an in-court density map from quality-gated smoothed positions."""

    x_values: list[float] = []
    y_values: list[float] = []
    for frame in frames:
        if not heatmap_eligible(frame, court):
            continue
        point = frame.smoothed_court_coordinate
        if point is not None:
            x_values.append(point.x_m)
            y_values.append(point.y_m)
    raw_histogram, _, _ = np.histogram2d(
        y_values,
        x_values,
        bins=(96, 48),
        range=((0, court.length_m), (0, court.width_m)),
    )
    flipped_histogram = np.flipud(raw_histogram)
    histogram = cv2.GaussianBlur(
        flipped_histogram.astype(np.float32),
        (0, 0),
        sigmaX=2.0,
    )
    maximum = float(histogram.max()) if histogram.size else 0.0
    normalized = (
        np.asarray(np.clip(histogram / maximum * 255, 0, 255), dtype=np.uint8)
        if maximum > 0
        else np.zeros(histogram.shape, dtype=np.uint8)
    )
    left, top, right, bottom = _court_rect()
    density = cv2.resize(normalized, (right - left, bottom - top), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(density, cv2.COLORMAP_TURBO)
    canvas = np.full(
        (TOPDOWN_HEIGHT_PX, TOPDOWN_WIDTH_PX, 3),
        BACKGROUND_COLOR,
        dtype=np.uint8,
    )
    _draw_court(canvas, court)
    if maximum > 0:
        court_region = canvas[top:bottom, left:right]
        alpha = np.asarray(density, dtype=np.float32)[..., None] / 255 * 0.78
        blended = court_region.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha
        canvas[top:bottom, left:right] = np.asarray(np.clip(blended, 0, 255), dtype=np.uint8)
        _draw_court_lines(canvas, court)
    cv2.putText(
        canvas,
        f"{display_name} ({role.value})",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        ROLE_COLORS[role],
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"quality-gated in-court samples: {len(x_values)}",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COURT_LINE_COLOR,
        1,
        cv2.LINE_AA,
    )
    if not x_values:
        cv2.putText(
            canvas,
            "NO ELIGIBLE POSITIONS",
            (180, TOPDOWN_HEIGHT_PX // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (80, 110, 240),
            2,
            cv2.LINE_AA,
        )
    return canvas
