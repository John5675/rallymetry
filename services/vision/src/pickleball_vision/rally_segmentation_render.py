"""Source-space debug rendering for automatic rally segmentation."""

from __future__ import annotations

import cv2
import numpy as np

from pickleball_vision.rally_evaluation import GroundTruthRally
from pickleball_vision.rally_segmentation import RallyBallFrame, RallyPrediction
from pickleball_vision.video import Image

PREDICTED_COLOR = (70, 230, 90)
GROUND_TRUTH_COLOR = (255, 210, 40)
BALL_COLOR = (60, 240, 240)
UNKNOWN_COLOR = (90, 90, 235)


def render_rally_segmentation_frame(
    frame: Image,
    *,
    frame_number: int,
    ball: RallyBallFrame,
    speed_diagonals_per_second: float | None,
    motion_supported: bool,
    predicted_rally: RallyPrediction | None,
    annotated_rally: GroundTruthRally | None,
    predicted_start: bool,
    predicted_end: bool,
) -> Image:
    """Overlay rally state without replacing structured JSON as the source of truth."""

    canvas = frame.copy()
    height, width = canvas.shape[:2]
    font_scale = max(0.46, min(height, width) / 1500)
    if predicted_rally is not None:
        cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), PREDICTED_COLOR, 4)
        cv2.rectangle(canvas, (8, 8), (min(width - 8, 690), 112), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            f"{predicted_rally.rally_id}  confidence {predicted_rally.confidence:.2f}",
            (18, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            PREDICTED_COLOR,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            (f"predicted frames {predicted_rally.start_frame}--{predicted_rally.end_frame}"),
            (18, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.88,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    if annotated_rally is not None:
        cv2.rectangle(canvas, (7, 7), (width - 8, height - 8), GROUND_TRUTH_COLOR, 2)
        cv2.putText(
            canvas,
            f"GROUND TRUTH: {annotated_rally.rally_id}",
            (18, 102 if predicted_rally is not None else 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.88,
            GROUND_TRUTH_COLOR,
            2,
            cv2.LINE_AA,
        )
    if ball.point is not None:
        point = (round(ball.point.x_px), round(ball.point.y_px))
        cv2.circle(canvas, point, 7, BALL_COLOR, 2, cv2.LINE_AA)
    speed_text = (
        "unknown"
        if speed_diagonals_per_second is None
        else (f"{speed_diagonals_per_second:.3f} diagonal/s")
    )
    cv2.putText(
        canvas,
        f"ball {ball.status.value} | motion {speed_text} | sustained={motion_supported}",
        (18, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.78,
        (240, 240, 240) if ball.point is not None else UNKNOWN_COLOR,
        1,
        cv2.LINE_AA,
    )
    if predicted_start or predicted_end:
        label = "PREDICTED RALLY START" if predicted_start else "PREDICTED RALLY END"
        text_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 1.15,
            2,
        )
        x_px = max(14, (width - text_size[0]) // 2)
        y_px = height // 2
        cv2.rectangle(
            canvas,
            (x_px - 10, y_px - text_size[1] - 12),
            (x_px + text_size[0] + 10, y_px + 10),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x_px, y_px),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 1.15,
            PREDICTED_COLOR,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        f"frame {frame_number}",
        (width - max(170, round(width * 0.11)), 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.78,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return np.asarray(canvas, dtype=np.uint8)
