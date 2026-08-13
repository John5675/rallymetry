"""Inspectable image-space and top-down calibration debug artifacts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from pickleball_vision.calibration import CourtCalibration
from pickleball_vision.court import CourtPoint, court_line_segments
from pickleball_vision.errors import OutputWriteError
from pickleball_vision.video import Image

LINE_COLOR = (0, 255, 255)
NET_COLOR = (0, 128, 255)
INLIER_COLOR = (0, 220, 0)
OUTLIER_COLOR = (0, 80, 255)
TOPDOWN_MARGIN_PX = 60
TOPDOWN_PIXELS_PER_METER = 100.0


def _integer_point(x: float, y: float) -> tuple[int, int] | None:
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    limit = 1_000_000
    if abs(x) > limit or abs(y) > limit:
        return None
    return round(x), round(y)


def render_calibration_overlay(frame: Image, calibration: CourtCalibration) -> Image:
    """Draw selected landmarks and projected canonical geometry on a source frame."""

    overlay = frame.copy()
    for name, start, end in court_line_segments(calibration.court):
        start_image = calibration.court_to_image(start)
        end_image = calibration.court_to_image(end)
        start_px = _integer_point(start_image.x_px, start_image.y_px)
        end_px = _integer_point(end_image.x_px, end_image.y_px)
        if start_px is None or end_px is None:
            continue
        color = NET_COLOR if name == "net" else LINE_COLOR
        cv2.line(overlay, start_px, end_px, color, 2, cv2.LINE_AA)

    for correspondence in calibration.correspondences:
        point = _integer_point(
            correspondence.image_point.x_px,
            correspondence.image_point.y_px,
        )
        if point is None:
            continue
        color = INLIER_COLOR if correspondence.inlier else OUTLIER_COLOR
        cv2.circle(overlay, point, 6, color, -1, cv2.LINE_AA)
        label = f"{correspondence.label} ({correspondence.image_error_px:.1f}px)"
        cv2.putText(
            overlay,
            label,
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def _court_to_topdown_matrix(calibration: CourtCalibration) -> np.ndarray:
    scale = TOPDOWN_PIXELS_PER_METER
    margin = float(TOPDOWN_MARGIN_PX)
    return np.asarray(
        [
            [scale, 0.0, margin],
            [0.0, -scale, margin + calibration.court.length_m * scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _topdown_point(calibration: CourtCalibration, point: CourtPoint) -> tuple[int, int]:
    x = TOPDOWN_MARGIN_PX + point.x_m * TOPDOWN_PIXELS_PER_METER
    y = TOPDOWN_MARGIN_PX + (calibration.court.length_m - point.y_m) * TOPDOWN_PIXELS_PER_METER
    return round(x), round(y)


def render_court_topdown(frame: Image, calibration: CourtCalibration) -> Image:
    """Warp the source frame to canonical court meters and draw court geometry."""

    width = round(calibration.court.width_m * TOPDOWN_PIXELS_PER_METER) + 2 * TOPDOWN_MARGIN_PX
    height = round(calibration.court.length_m * TOPDOWN_PIXELS_PER_METER) + 2 * TOPDOWN_MARGIN_PX
    source_to_topdown = _court_to_topdown_matrix(calibration) @ (
        calibration.image_to_court_homography
    )
    warped = cv2.warpPerspective(
        frame,
        source_to_topdown,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(24, 24, 24),
    )
    topdown = cast(Image, np.asarray(warped))
    for name, start, end in court_line_segments(calibration.court):
        color = NET_COLOR if name == "net" else LINE_COLOR
        cv2.line(
            topdown,
            _topdown_point(calibration, start),
            _topdown_point(calibration, end),
            color,
            2,
            cv2.LINE_AA,
        )
    for correspondence in calibration.correspondences:
        color = INLIER_COLOR if correspondence.inlier else OUTLIER_COLOR
        cv2.circle(
            topdown,
            _topdown_point(calibration, correspondence.court_point),
            5,
            color,
            -1,
            cv2.LINE_AA,
        )
    return topdown


def write_debug_image(image: Image, path: Path) -> Path:
    """Write a calibration image artifact with useful application errors."""

    output_path = path.expanduser().resolve()
    if output_path.suffix.lower() not in {".jpeg", ".jpg", ".png"}:
        raise OutputWriteError(str(output_path), reason="debug image must be JPEG or PNG")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        written = cv2.imwrite(str(output_path), image)
    except (OSError, cv2.error) as error:
        raise OutputWriteError(str(output_path), reason=str(error)) from error
    if not written:
        raise OutputWriteError(str(output_path), reason="OpenCV did not write the image")
    return output_path
