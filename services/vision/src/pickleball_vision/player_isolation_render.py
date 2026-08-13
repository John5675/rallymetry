"""Debug rendering for court-aware primary-player isolation."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

from pickleball_vision.calibration import CourtCalibration
from pickleball_vision.court import court_line_segments
from pickleball_vision.player_isolation import (
    CandidateObservation,
    CourtRegionState,
    LogicalPlayerRole,
    PlayerCandidate,
)
from pickleball_vision.video import Image

SUBTLE_BOX_COLOR = (120, 120, 120)
CANDIDATE_BOX_COLOR = (255, 210, 20)
COURT_LINE_COLOR = (180, 120, 30)
GROUND_COLORS = {
    CourtRegionState.INSIDE: (20, 220, 20),
    CourtRegionState.NEAR: (0, 210, 255),
    CourtRegionState.OUTSIDE: (50, 50, 255),
    CourtRegionState.AMBIGUOUS: (255, 160, 30),
}
ROLE_COLORS = {
    LogicalPlayerRole.ME: (255, 40, 220),
    LogicalPlayerRole.PARTNER: (255, 220, 40),
    LogicalPlayerRole.OPPONENT_1: (30, 120, 255),
    LogicalPlayerRole.OPPONENT_2: (40, 220, 255),
}


def _draw_court_geometry(canvas: Image, calibration: CourtCalibration) -> None:
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


def _role_by_candidate(
    logical_assignments: Mapping[LogicalPlayerRole, CandidateObservation],
) -> dict[str, LogicalPlayerRole]:
    return {observation.candidate_id: role for role, observation in logical_assignments.items()}


def render_player_isolation_frame(
    frame: Image,
    *,
    observations: tuple[CandidateObservation, ...],
    candidates: Mapping[str, PlayerCandidate],
    logical_assignments: Mapping[LogicalPlayerRole, CandidateObservation],
    calibration: CourtCalibration,
    draw_court: bool = True,
) -> Image:
    """Show raw people, candidates, ground points, court state, and logical labels."""

    canvas = frame.copy()
    if draw_court:
        _draw_court_geometry(canvas, calibration)
    role_by_candidate = _role_by_candidate(logical_assignments)
    line_width = max(1, round(min(frame.shape[:2]) / 360))
    font_scale = max(0.38, min(frame.shape[:2]) / 1450)

    for observation in observations:
        detection = observation.detection
        box = detection.bounding_box
        candidate = candidates[observation.candidate_id]
        role = role_by_candidate.get(observation.candidate_id)
        if role is not None:
            box_color = ROLE_COLORS[role]
            thickness = line_width + 2
        elif candidate.eligible:
            box_color = CANDIDATE_BOX_COLOR
            thickness = line_width + 1
        else:
            box_color = SUBTLE_BOX_COLOR
            thickness = line_width
        left_top = (round(box.left_px), round(box.top_px))
        right_bottom = (round(box.right_px), round(box.bottom_px))
        cv2.rectangle(canvas, left_top, right_bottom, box_color, thickness)

        ground = observation.ground_contact
        ground_xy = (round(ground.image_point.x_px), round(ground.image_point.y_px))
        cv2.circle(
            canvas,
            ground_xy,
            line_width + 4,
            GROUND_COLORS[ground.region_state],
            -1,
            cv2.LINE_AA,
        )
        candidate_number = observation.candidate_id.removeprefix("candidate-").lstrip("0") or "0"
        state = ground.region_state.value.upper()
        label_parts = []
        if role is not None:
            label_parts.append(role.value)
        elif candidate.eligible:
            label_parts.append(f"C{candidate_number}")
        label_parts.append(state)
        if ground.region_boundary_ambiguous:
            label_parts.append("UNCERTAIN")
        label = " | ".join(label_parts)
        cv2.putText(
            canvas,
            label,
            (left_top[0], max(16, left_top[1] - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            box_color,
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
    return np.asarray(canvas, dtype=np.uint8)
