"""Inspectability overlay for persistent logical-player tracking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from pickleball_vision.calibration import CourtCalibration
from pickleball_vision.court import court_line_segments
from pickleball_vision.person_detection import PersonDetection
from pickleball_vision.player_isolation import LogicalPlayerRole
from pickleball_vision.player_tracking import LogicalPlayerFrame, LogicalTrackingState
from pickleball_vision.video import Image

SUBTLE_PERSON_COLOR = (105, 105, 105)
COURT_LINE_COLOR = (150, 100, 25)
STATE_MISSING_COLOR = (70, 70, 220)
ROLE_COLORS = {
    LogicalPlayerRole.ME: (255, 40, 220),
    LogicalPlayerRole.PARTNER: (255, 220, 40),
    LogicalPlayerRole.OPPONENT_1: (30, 120, 255),
    LogicalPlayerRole.OPPONENT_2: (40, 220, 255),
}


def _draw_court(canvas: Image, calibration: CourtCalibration) -> None:
    for name, start, end in court_line_segments(calibration.court):
        start_image = calibration.court_to_image(start)
        end_image = calibration.court_to_image(end)
        width = (
            2 if name in {"near_baseline", "far_baseline", "left_sideline", "right_sideline"} else 1
        )
        cv2.line(
            canvas,
            (round(start_image.x_px), round(start_image.y_px)),
            (round(end_image.x_px), round(end_image.y_px)),
            COURT_LINE_COLOR,
            width,
            cv2.LINE_AA,
        )


def render_player_tracking_frame(
    frame: Image,
    *,
    raw_detections: Sequence[PersonDetection],
    logical_frames: Sequence[LogicalPlayerFrame],
    calibration: CourtCalibration,
    player_names: Mapping[LogicalPlayerRole, str],
) -> Image:
    """Show all raw people subtly and four resolved identities distinctly."""

    canvas = frame.copy()
    _draw_court(canvas, calibration)
    line_width = max(1, round(min(frame.shape[:2]) / 360))
    font_scale = max(0.38, min(frame.shape[:2]) / 1450)
    for detection in raw_detections:
        box = detection.bounding_box
        cv2.rectangle(
            canvas,
            (round(box.left_px), round(box.top_px)),
            (round(box.right_px), round(box.bottom_px)),
            SUBTLE_PERSON_COLOR,
            line_width,
        )

    missing_y = 24
    for logical_frame in logical_frames:
        role = logical_frame.logical_player
        display_name = player_names[role]
        color = ROLE_COLORS[role]
        tracked = logical_frame.tracker_observation
        if tracked is None:
            cv2.putText(
                canvas,
                f"{display_name} ({role.value}): MISSING",
                (12, missing_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                STATE_MISSING_COLOR,
                max(1, line_width),
                cv2.LINE_AA,
            )
            missing_y += max(18, round(28 * font_scale))
            continue
        raw = tracked.raw
        box = raw.tracker_bounding_box
        thickness = line_width + 2
        cv2.rectangle(
            canvas,
            (round(box.left_px), round(box.top_px)),
            (round(box.right_px), round(box.bottom_px)),
            color,
            thickness,
        )
        ground = tracked.ground_contact.image_point
        cv2.circle(
            canvas,
            (round(ground.x_px), round(ground.y_px)),
            line_width + 4,
            color,
            -1,
            cv2.LINE_AA,
        )
        state = logical_frame.state.value.upper()
        label = (
            f"{display_name} ({role.value}) | ID {raw.tracker_id} | "
            f"{logical_frame.confidence:.2f} | {state}"
        )
        cv2.putText(
            canvas,
            label,
            (round(box.left_px), max(16, round(box.top_px) - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
        if logical_frame.state is LogicalTrackingState.SUSPECTED_IDENTITY_SWITCH:
            cv2.putText(
                canvas,
                "REVIEW IDENTITY",
                (round(box.left_px), min(frame.shape[0] - 8, round(box.bottom_px) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 255),
                thickness,
                cv2.LINE_AA,
            )
    return np.asarray(canvas, dtype=np.uint8)
