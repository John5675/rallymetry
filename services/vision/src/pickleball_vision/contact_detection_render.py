"""Source-space debug rendering for multimodal paddle-contact candidates."""

from __future__ import annotations

import cv2

from pickleball_vision.contact_detection import ContactCandidate, ContactEvidenceMode
from pickleball_vision.video import Image

VISUAL_COLOR = (60, 220, 60)
FUSED_COLOR = (220, 80, 220)
LOW_COLOR = (0, 170, 255)
PLAYER_COLOR = (200, 200, 80)
TEXT_COLOR = (255, 255, 255)


def _candidate_color(candidate: ContactCandidate) -> tuple[int, int, int]:
    if candidate.evidence_mode is ContactEvidenceMode.VISUAL_PLUS_AUDIO:
        return FUSED_COLOR
    if candidate.evidence_mode is ContactEvidenceMode.VISUAL_ONLY:
        return VISUAL_COLOR
    return LOW_COLOR


def render_contact_detection_frame(
    frame: Image,
    *,
    frame_number: int,
    candidate: ContactCandidate | None,
    recent_candidate: ContactCandidate | None,
    audio_available: bool,
) -> Image:
    """Render ball and candidate-player evidence without assigning a hitter."""

    canvas = frame.copy()
    cv2.putText(
        canvas,
        f"frame {frame_number} | audio={'available' if audio_available else 'vision-only'}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    active = candidate or recent_candidate
    if active is None:
        return canvas
    color = _candidate_color(active)
    ball = (
        round(active.ball_image_position.x_px),
        round(active.ball_image_position.y_px),
    )
    cv2.circle(canvas, ball, 13 if candidate is not None else 8, color, 3, cv2.LINE_AA)
    for player in active.candidate_players:
        left, top, right, bottom = player.bounding_box
        box_color = PLAYER_COLOR if player.rank == 1 else (140, 140, 100)
        thickness = 2 if player.rank == 1 else 1
        cv2.rectangle(
            canvas,
            (round(left), round(top)),
            (round(right), round(bottom)),
            box_color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"candidate {player.role} p={player.proximity_confidence:.2f}",
            (round(left), max(18, round(top) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            box_color,
            1,
            cv2.LINE_AA,
        )
    status = "CONTACT CANDIDATE" if active.accepted_fused else "LOW-CONFIDENCE CANDIDATE"
    age = "" if candidate is not None else " (recent)"
    cv2.putText(
        canvas,
        f"{status}{age} | {active.evidence_mode.value}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (
            f"visual={active.visual_confidence:.3f} "
            f"audio={active.audio_confidence:.3f} fused={active.fused_confidence:.3f}"
        ),
        (20, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    nearest = active.candidate_players[0] if active.candidate_players else None
    cv2.putText(
        canvas,
        (
            "hitter not assigned | no tracked player nearby"
            if nearest is None
            else (
                f"hitter not assigned | nearest candidate={nearest.role} "
                f"distance={nearest.distance_px:.1f}px"
            )
        ),
        (20, 121),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    return canvas
