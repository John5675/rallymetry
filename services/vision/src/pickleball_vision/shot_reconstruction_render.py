"""Source-space debug rendering for reconstructed and classified shots."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import cv2
import numpy as np

from pickleball_vision.court import ImagePoint
from pickleball_vision.rally_segmentation import BallEvidenceStatus, RallyBallFrame
from pickleball_vision.shot_reconstruction import Shot, ShotType
from pickleball_vision.video import Image

SHOT_COLORS = {
    ShotType.SERVE: (60, 220, 60),
    ShotType.RETURN: (230, 210, 60),
    ShotType.DINK: (220, 140, 50),
    ShotType.DROP: (255, 170, 60),
    ShotType.DRIVE: (40, 80, 255),
    ShotType.VOLLEY: (220, 80, 220),
    ShotType.OVERHEAD: (40, 150, 255),
    ShotType.OTHER: (180, 180, 180),
    ShotType.UNKNOWN: (30, 50, 235),
}
TEXT_COLOR = (255, 255, 255)
OBSERVED_BALL_COLOR = (80, 245, 80)
INTERPOLATED_BALL_COLOR = (0, 180, 255)
UNKNOWN_BALL_COLOR = (70, 70, 240)


def _pixel(point: ImagePoint) -> tuple[int, int]:
    return round(point.x_px), round(point.y_px)


def _draw_interpolated_marker(canvas: Image, point: tuple[int, int]) -> None:
    x_px, y_px = point
    diamond = np.asarray(
        ((x_px, y_px - 8), (x_px + 8, y_px), (x_px, y_px + 8), (x_px - 8, y_px)),
        dtype=np.int32,
    )
    cv2.polylines(canvas, [diamond], True, INTERPOLATED_BALL_COLOR, 2, cv2.LINE_AA)


def _draw_ball_trajectory(
    canvas: Image,
    *,
    trajectory_point: RallyBallFrame,
    recent_trail: Sequence[RallyBallFrame],
) -> None:
    for left, right in pairwise(recent_trail):
        if (
            left.point is None
            or right.point is None
            or left.frame_number + 1 != right.frame_number
            or left.segment_id is None
            or left.segment_id != right.segment_id
        ):
            continue
        color = (
            INTERPOLATED_BALL_COLOR
            if BallEvidenceStatus.INTERPOLATED in {left.status, right.status}
            else OBSERVED_BALL_COLOR
        )
        cv2.line(canvas, _pixel(left.point), _pixel(right.point), color, 3, cv2.LINE_AA)

    if trajectory_point.point is not None:
        if trajectory_point.status is BallEvidenceStatus.OBSERVED:
            cv2.circle(
                canvas,
                _pixel(trajectory_point.point),
                7,
                OBSERVED_BALL_COLOR,
                -1,
                cv2.LINE_AA,
            )
        elif trajectory_point.status is BallEvidenceStatus.INTERPOLATED:
            _draw_interpolated_marker(canvas, _pixel(trajectory_point.point))

    confidence = (
        f" confidence={trajectory_point.confidence:.2f}"
        if trajectory_point.confidence is not None
        else ""
    )
    status_color = {
        BallEvidenceStatus.OBSERVED: OBSERVED_BALL_COLOR,
        BallEvidenceStatus.INTERPOLATED: INTERPOLATED_BALL_COLOR,
        BallEvidenceStatus.UNKNOWN: UNKNOWN_BALL_COLOR,
    }[trajectory_point.status]
    cv2.putText(
        canvas,
        f"BALL: {trajectory_point.status.value}{confidence}",
        (20, canvas.shape[0] - 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "green=observed trail | amber=interpolated | unknown gaps stay disconnected",
        (20, canvas.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def render_shot_frame(
    frame: Image,
    *,
    frame_number: int,
    shot: Shot | None,
    recent_shot: Shot | None,
    trajectory_point: RallyBallFrame,
    recent_trail: Sequence[RallyBallFrame],
) -> Image:
    """Render shot evidence and a gap-safe image-space ball trajectory trail."""

    canvas = frame.copy()
    _draw_ball_trajectory(
        canvas,
        trajectory_point=trajectory_point,
        recent_trail=recent_trail,
    )
    cv2.putText(
        canvas,
        f"frame {frame_number} | rule-based shot reconstruction",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    active = shot or recent_shot
    if active is None:
        return canvas
    color = SHOT_COLORS[active.shot_type]
    contact_point = active.trajectory_segment.initial_image_position
    if contact_point is not None:
        cv2.circle(
            canvas,
            (round(contact_point.x_px), round(contact_point.y_px)),
            14 if shot is not None else 9,
            color,
            3,
            cv2.LINE_AA,
        )
    if active.hitter_court_position is not None:
        ground = active.hitter_court_position.image_ground_point
        cv2.circle(
            canvas,
            (round(ground.x_px), round(ground.y_px)),
            7,
            color,
            -1,
            cv2.LINE_AA,
        )
    suffix = "" if shot is not None else " (recent)"
    cv2.putText(
        canvas,
        f"SHOT: {active.shot_type.value}{suffix} | HIT: {active.hitter_id}",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        color,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (f"rally={active.rally_id} index={active.shot_index} confidence={active.confidence:.3f}"),
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (
            f"contact={active.contact_id} bounce={active.bounce_id or 'none'} "
            f"trajectory coverage={active.trajectory_segment.known_fraction:.2f}"
        ),
        (20, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    return canvas
