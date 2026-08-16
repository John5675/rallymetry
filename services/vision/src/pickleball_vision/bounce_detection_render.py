"""Source-space debug rendering for multimodal bounce candidates."""

from __future__ import annotations

import cv2

from pickleball_vision.bounce_detection import BounceCandidate, BounceEvidenceMode
from pickleball_vision.video import Image

VISUAL_COLOR = (60, 220, 60)
FUSED_COLOR = (220, 80, 220)
LOW_COLOR = (0, 170, 255)
TEXT_COLOR = (255, 255, 255)


def _candidate_color(candidate: BounceCandidate) -> tuple[int, int, int]:
    if candidate.evidence_mode is BounceEvidenceMode.VISUAL_PLUS_AUDIO:
        return FUSED_COLOR
    if candidate.evidence_mode is BounceEvidenceMode.VISUAL_ONLY:
        return VISUAL_COLOR
    return LOW_COLOR


def render_bounce_detection_frame(
    frame: Image,
    *,
    frame_number: int,
    candidate: BounceCandidate | None,
    recent_candidate: BounceCandidate | None,
    audio_available: bool,
) -> Image:
    """Render candidate evidence without altering the source frame."""

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
    center = (round(active.image_position.x_px), round(active.image_position.y_px))
    cv2.circle(canvas, center, 13 if candidate is not None else 8, color, 3, cv2.LINE_AA)
    status = "BOUNCE" if active.accepted_fused else "LOW-CONFIDENCE CANDIDATE"
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
    court_text = (
        "court position withheld"
        if active.court_position is None
        else f"court=({active.court_position.x_m:.2f}m, {active.court_position.y_m:.2f}m)"
    )
    cv2.putText(
        canvas,
        court_text,
        (20, 121),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    return canvas
