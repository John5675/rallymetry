"""Inspectable source-space rendering for primary-match ball reconstruction."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import cv2
import numpy as np

from pickleball_vision.ball_tracking import (
    BallTrackingCandidate,
    BallTrajectoryPoint,
    BallTrajectoryStatus,
)
from pickleball_vision.court import ImagePoint
from pickleball_vision.video import Image

OBSERVED_COLOR = (80, 245, 80)
INTERPOLATED_COLOR = (0, 180, 255)
UNKNOWN_COLOR = (70, 70, 240)
RAW_CANDIDATE_COLOR = (120, 120, 120)
RAW_POINT_COLOR = (230, 230, 230)


def _pixel(point: ImagePoint) -> tuple[int, int]:
    return round(point.x_px), round(point.y_px)


def _draw_interpolated_marker(image: Image, point: tuple[int, int], size: int) -> None:
    x_px, y_px = point
    diamond = np.asarray(
        [
            (x_px, y_px - size),
            (x_px + size, y_px),
            (x_px, y_px + size),
            (x_px - size, y_px),
        ],
        dtype=np.int32,
    )
    cv2.polylines(image, [diamond], True, INTERPOLATED_COLOR, 2, cv2.LINE_AA)
    cv2.line(
        image,
        (x_px - size // 2, y_px),
        (x_px + size // 2, y_px),
        INTERPOLATED_COLOR,
        1,
        cv2.LINE_AA,
    )


def render_ball_tracking_frame(
    frame: Image,
    *,
    raw_candidates: Sequence[BallTrackingCandidate],
    trajectory_point: BallTrajectoryPoint,
    recent_trail: Sequence[BallTrajectoryPoint],
) -> Image:
    """Draw raw candidates subtly and the derived trajectory with explicit status."""

    canvas = frame.copy()
    line_width = max(1, round(min(frame.shape[:2]) / 540))
    font_scale = max(0.42, min(frame.shape[:2]) / 1500)
    for candidate in raw_candidates:
        box = candidate.bounding_box
        top_left = (round(box.left_px), round(box.top_px))
        bottom_right = (round(box.right_px), round(box.bottom_px))
        selected = candidate.detection_id == trajectory_point.source_detection_id
        color = OBSERVED_COLOR if selected else RAW_CANDIDATE_COLOR
        thickness = line_width + 1 if selected else line_width
        cv2.rectangle(canvas, top_left, bottom_right, color, thickness, cv2.LINE_AA)
        if not selected:
            cv2.putText(
                canvas,
                f"candidate {candidate.confidence:.2f}",
                (top_left[0], max(12, top_left[1] - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.75,
                color,
                1,
                cv2.LINE_AA,
            )

    trail_points = [
        item.smoothed_image_point
        for item in recent_trail
        if item.smoothed_image_point is not None and item.segment_id == trajectory_point.segment_id
    ]
    for left, right in pairwise(trail_points):
        assert left is not None and right is not None
        cv2.line(canvas, _pixel(left), _pixel(right), OBSERVED_COLOR, 2, cv2.LINE_AA)

    status = trajectory_point.status
    point = trajectory_point.smoothed_image_point
    if status is BallTrajectoryStatus.OBSERVED and point is not None:
        raw = trajectory_point.raw_image_point
        if raw is not None:
            cv2.circle(canvas, _pixel(raw), 7, RAW_POINT_COLOR, 2, cv2.LINE_AA)
            cv2.line(canvas, _pixel(raw), _pixel(point), RAW_POINT_COLOR, 1, cv2.LINE_AA)
        cv2.circle(canvas, _pixel(point), 6, OBSERVED_COLOR, -1, cv2.LINE_AA)
    elif status is BallTrajectoryStatus.INTERPOLATED and point is not None:
        _draw_interpolated_marker(canvas, _pixel(point), max(7, line_width * 4))

    confidence = (
        f" | confidence {trajectory_point.confidence:.2f}"
        if trajectory_point.confidence is not None
        else ""
    )
    color = {
        BallTrajectoryStatus.OBSERVED: OBSERVED_COLOR,
        BallTrajectoryStatus.INTERPOLATED: INTERPOLATED_COLOR,
        BallTrajectoryStatus.UNKNOWN: UNKNOWN_COLOR,
    }[status]
    cv2.putText(
        canvas,
        f"PRIMARY BALL: {status.value}{confidence}",
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "green=observed | amber diamond=interpolated | gray=raw rejected candidate",
        (16, frame.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.80,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return np.asarray(canvas, dtype=np.uint8)
