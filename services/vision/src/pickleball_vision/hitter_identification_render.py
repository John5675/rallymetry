"""Source-space rendering for inspectable hitter-identification decisions."""

from __future__ import annotations

import cv2

from pickleball_vision.hitter_identification import (
    UNKNOWN_PLAYER_ID,
    HitterIdentification,
)
from pickleball_vision.video import Image

ROLE_COLORS = {
    "ME": (60, 220, 60),
    "PARTNER": (230, 210, 60),
    "OPPONENT_1": (220, 80, 220),
    "OPPONENT_2": (40, 150, 255),
    UNKNOWN_PLAYER_ID: (30, 50, 235),
}
ALTERNATIVE_COLOR = (150, 150, 150)
TEXT_COLOR = (255, 255, 255)


def _player_score(identification: HitterIdentification, player_id: str) -> float | None:
    decision = identification.supporting_signals.get("decision")
    if isinstance(decision, dict) and decision.get("bestPlayerId") == player_id:
        value = decision.get("bestScore")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    for alternative in identification.alternatives:
        if alternative.player_id == player_id:
            return alternative.confidence
    return None


def render_hitter_identification_frame(
    frame: Image,
    *,
    frame_number: int,
    identification: HitterIdentification | None,
    recent_identification: HitterIdentification | None,
) -> Image:
    """Overlay the logical hitter decision without changing the source frame."""

    canvas = frame.copy()
    cv2.putText(
        canvas,
        f"frame {frame_number} | hitter identity uses visual/player evidence only",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    active = identification or recent_identification
    if active is None:
        return canvas
    color = ROLE_COLORS[active.player_id]
    ball = (
        round(active.ball_image_position.x_px),
        round(active.ball_image_position.y_px),
    )
    cv2.circle(canvas, ball, 14 if identification is not None else 9, color, 3, cv2.LINE_AA)
    for player in active.candidate_players:
        selected = player.role == active.player_id
        box_color = color if selected else ALTERNATIVE_COLOR
        thickness = 3 if selected else 1
        left, top, right, bottom = player.bounding_box
        cv2.rectangle(
            canvas,
            (round(left), round(top)),
            (round(right), round(bottom)),
            box_color,
            thickness,
            cv2.LINE_AA,
        )
        ground = (
            round(player.ground_image_position.x_px),
            round(player.ground_image_position.y_px),
        )
        cv2.circle(canvas, ground, 5, box_color, -1, cv2.LINE_AA)
        score = _player_score(active, player.role)
        cv2.putText(
            canvas,
            f"{player.role} score={score:.2f}" if score is not None else player.role,
            (round(left), max(18, round(top) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            box_color,
            1,
            cv2.LINE_AA,
        )
    suffix = "" if identification is not None else " (recent)"
    cv2.putText(
        canvas,
        f"HIT: {active.player_id}{suffix}",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.92,
        color,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (
            f"identity confidence={active.confidence:.3f} | "
            f"visual contact confidence={active.source_visual_contact_confidence:.3f}"
        ),
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "audio identity contribution=0.000 | UNKNOWN retained when gates fail",
        (20, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    return canvas
