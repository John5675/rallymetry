"""Lightweight OpenCV UI for manual named-landmark selection."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from typing import cast

import cv2
import numpy as np

from pickleball_vision.calibration import CalibrationCorrespondence
from pickleball_vision.court import CourtDimensions, ImagePoint, LandmarkName, court_landmarks
from pickleball_vision.errors import CalibrationCancelledError, InvalidCalibrationError
from pickleball_vision.video import Image

WINDOW_NAME = "Pickleball Vision - Manual Court Calibration"
MAX_DISPLAY_WIDTH = 1600
MAX_DISPLAY_HEIGHT = 900


@dataclass(slots=True)
class _SelectionState:
    prompt_index: int = 0
    selections: dict[LandmarkName, ImagePoint] = field(default_factory=dict)
    pending_click: tuple[int, int] | None = None


def _display_frame(frame: Image) -> tuple[Image, float]:
    height, width = frame.shape[:2]
    scale = min(1.0, MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height)
    if scale == 1.0:
        return frame.copy(), scale
    resized = cv2.resize(
        frame,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return cast(Image, np.asarray(resized)), scale


def _render_ui(
    base: Image,
    state: _SelectionState,
    dimensions: CourtDimensions,
    scale: float,
) -> Image:
    canvas = base.copy()
    landmarks = court_landmarks(dimensions)
    for landmark in landmarks:
        point = state.selections.get(landmark.name)
        if point is None:
            continue
        displayed = round(point.x_px * scale), round(point.y_px * scale)
        cv2.circle(canvas, displayed, 6, (0, 220, 0), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            landmark.label,
            (displayed[0] + 8, displayed[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 82), (20, 20, 20), -1)
    if state.prompt_index < len(landmarks):
        prompt = f"Click: {landmarks[state.prompt_index].label}"
    else:
        prompt = "Landmark review complete"
    cv2.putText(
        canvas,
        prompt,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    controls = (
        f"selected {len(state.selections)}/{len(landmarks)} | left-click select | "
        "s skip | u undo | r restart | Enter finish | q cancel"
    )
    cv2.putText(
        canvas,
        controls,
        (16, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


def select_court_landmarks(
    frame: Image,
    dimensions: CourtDimensions,
) -> tuple[CalibrationCorrespondence, ...]:
    """Interactively associate visible image clicks with canonical landmarks."""

    base, scale = _display_frame(frame)
    state = _SelectionState()
    landmarks = court_landmarks(dimensions)
    original_height, original_width = frame.shape[:2]

    def on_mouse(event: int, x: int, y: int, _flags: int, _parameter: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state.pending_click = (x, y)

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)
        while True:
            if state.pending_click is not None and state.prompt_index < len(landmarks):
                click_x, click_y = state.pending_click
                landmark = landmarks[state.prompt_index]
                state.selections[landmark.name] = ImagePoint(
                    x_px=min(original_width - 1.0, max(0.0, click_x / scale)),
                    y_px=min(original_height - 1.0, max(0.0, click_y / scale)),
                )
                state.prompt_index += 1
                state.pending_click = None

            cv2.imshow(WINDOW_NAME, _render_ui(base, state, dimensions, scale))
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                raise CalibrationCancelledError
            if key == ord("s") and state.prompt_index < len(landmarks):
                state.prompt_index += 1
            elif key == ord("u") and state.prompt_index > 0:
                state.prompt_index -= 1
                state.selections.pop(landmarks[state.prompt_index].name, None)
            elif key == ord("r"):
                state.prompt_index = 0
                state.selections.clear()
            elif key in (10, 13, 32):
                if len(state.selections) < 4:
                    continue
                break

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                raise CalibrationCancelledError
    except cv2.error as error:
        raise InvalidCalibrationError(
            "OpenCV could not create the calibration window; run from a graphical desktop"
        ) from error
    finally:
        with suppress(cv2.error):
            cv2.destroyWindow(WINDOW_NAME)

    return tuple(
        CalibrationCorrespondence(
            landmark=landmark.name,
            label=landmark.label,
            image_point=state.selections[landmark.name],
            court_point=landmark.court_point,
        )
        for landmark in landmarks
        if landmark.name in state.selections
    )
