"""OpenCV workflow for manual logical-player initialization and correction."""

from __future__ import annotations

from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pickleball_vision.calibration import CourtCalibration
from pickleball_vision.errors import (
    PlayerIsolationCancelledError,
    PlayerIsolationInputError,
)
from pickleball_vision.player_isolation import (
    LOGICAL_PLAYER_ROLES,
    CandidateObservation,
    LogicalPlayerRole,
    PlayerCandidateCollection,
)
from pickleball_vision.player_isolation_render import render_player_isolation_frame
from pickleball_vision.video import Image, decode_video_frame

WINDOW_NAME = "Pickleball Vision - Primary Match Players"
MAX_DISPLAY_WIDTH = 1600
MAX_DISPLAY_HEIGHT = 900


@dataclass(slots=True)
class _SelectionState:
    frame_number: int
    current_role: LogicalPlayerRole = LogicalPlayerRole.ME
    selections: dict[LogicalPlayerRole, CandidateObservation] = field(default_factory=dict)
    pending_click: tuple[int, int] | None = None
    message: str = "Select ME"


def _display_image(frame: Image) -> tuple[Image, float]:
    height, width = frame.shape[:2]
    scale = min(1.0, MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height)
    if scale == 1.0:
        return frame.copy(), scale
    resized = cv2.resize(
        frame,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return np.asarray(resized, dtype=np.uint8), scale


def _candidate_at_click(
    observations: tuple[CandidateObservation, ...],
    *,
    x_px: float,
    y_px: float,
) -> CandidateObservation | None:
    containing: list[tuple[float, CandidateObservation]] = []
    for observation in observations:
        box = observation.detection.bounding_box
        if box.left_px <= x_px <= box.right_px and box.top_px <= y_px <= box.bottom_px:
            area = (box.right_px - box.left_px) * (box.bottom_px - box.top_px)
            containing.append((area, observation))
    if not containing:
        return None
    return min(containing, key=lambda item: item[0])[1]


def _advance_role(state: _SelectionState) -> None:
    current_index = LOGICAL_PLAYER_ROLES.index(state.current_role)
    for offset in range(1, len(LOGICAL_PLAYER_ROLES) + 1):
        candidate = LOGICAL_PLAYER_ROLES[(current_index + offset) % len(LOGICAL_PLAYER_ROLES)]
        if candidate not in state.selections:
            state.current_role = candidate
            state.message = f"Select {candidate.value}"
            return
    state.message = "All roles assigned; press Enter to finish or 1-4 to correct"


def _status_panel(canvas: Image, state: _SelectionState) -> Image:
    panel_height = 100
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], panel_height), (15, 15, 15), -1)
    assigned = " | ".join(
        f"{index + 1}:{role.value}="
        + (
            state.selections[role].candidate_id.replace("candidate-", "C")
            if role in state.selections
            else "?"
        )
        for index, role in enumerate(LOGICAL_PLAYER_ROLES)
    )
    cv2.putText(
        canvas,
        state.message,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        assigned,
        (14, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    controls = (
        f"frame {state.frame_number} | click person | 1-4 choose role | a/d +/-frame | "
        "j/l +/-1s | c clear role | r reset | Enter finish | q cancel"
    )
    cv2.putText(
        canvas,
        controls,
        (14, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return canvas


def select_logical_players(
    video_path: Path,
    *,
    candidates: PlayerCandidateCollection,
    calibration: CourtCalibration,
    initial_frame_number: int,
    existing: dict[LogicalPlayerRole, CandidateObservation] | None = None,
) -> dict[LogicalPlayerRole, CandidateObservation]:
    """Assign or correct four logical identities through a local video window."""

    if initial_frame_number < 0 or initial_frame_number >= candidates.source.frame_count:
        raise PlayerIsolationInputError("initial selection frame is outside the video")
    observations_by_frame: dict[int, list[CandidateObservation]] = defaultdict(list)
    for observation in candidates.observations:
        observations_by_frame[observation.detection.frame_number].append(observation)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates.candidates}
    state = _SelectionState(
        frame_number=initial_frame_number,
        selections=dict(existing or {}),
        message=(
            "Review assignments; press 1-4 then click to correct" if existing else "Select ME"
        ),
    )
    decoded_cache: dict[int, Image] = {}

    def frame_image(frame_number: int) -> Image:
        cached = decoded_cache.get(frame_number)
        if cached is None:
            decoded = decode_video_frame(
                video_path,
                timestamp_seconds=frame_number / candidates.source.fps,
            )
            cached = decoded.image
            decoded_cache.clear()
            decoded_cache[frame_number] = cached
        return cached

    def on_mouse(event: int, x: int, y: int, _flags: int, _parameter: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state.pending_click = (x, y)

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)
        while True:
            frame = frame_image(state.frame_number)
            frame_observations = tuple(observations_by_frame.get(state.frame_number, ()))
            rendered = render_player_isolation_frame(
                frame,
                observations=frame_observations,
                candidates=candidate_by_id,
                logical_assignments=state.selections,
                calibration=calibration,
            )
            base, scale = _display_image(rendered)
            if state.pending_click is not None:
                click_x, click_y = state.pending_click
                selected = _candidate_at_click(
                    frame_observations,
                    x_px=click_x / scale,
                    y_px=click_y / scale,
                )
                state.pending_click = None
                if selected is None:
                    state.message = "No person box at that click"
                else:
                    for role, existing_observation in tuple(state.selections.items()):
                        if (
                            role is not state.current_role
                            and existing_observation.candidate_id == selected.candidate_id
                        ):
                            del state.selections[role]
                    state.selections[state.current_role] = selected
                    _advance_role(state)

            cv2.imshow(WINDOW_NAME, _status_panel(base, state))
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                raise PlayerIsolationCancelledError
            if ord("1") <= key <= ord("4"):
                state.current_role = LOGICAL_PLAYER_ROLES[key - ord("1")]
                role_anchor = state.selections.get(state.current_role)
                if role_anchor is not None:
                    state.frame_number = role_anchor.detection.frame_number
                state.message = f"Select {state.current_role.value}"
            elif key == ord("a"):
                state.frame_number = max(0, state.frame_number - 1)
            elif key == ord("d"):
                state.frame_number = min(candidates.source.frame_count - 1, state.frame_number + 1)
            elif key == ord("j"):
                state.frame_number = max(0, state.frame_number - round(candidates.source.fps))
            elif key == ord("l"):
                state.frame_number = min(
                    candidates.source.frame_count - 1,
                    state.frame_number + round(candidates.source.fps),
                )
            elif key == ord("c"):
                state.selections.pop(state.current_role, None)
                state.message = f"Cleared {state.current_role.value}; select a person"
            elif key == ord("r"):
                state.selections.clear()
                state.current_role = LogicalPlayerRole.ME
                state.message = "Assignments reset; select ME"
            elif key in (10, 13, 32):
                if set(state.selections) == set(LOGICAL_PLAYER_ROLES):
                    return dict(state.selections)
                state.message = "Assign all four distinct roles before finishing"
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                raise PlayerIsolationCancelledError
    except cv2.error as error:
        raise PlayerIsolationInputError(
            "OpenCV could not create the player-selection window; run from a graphical desktop"
        ) from error
    finally:
        with suppress(cv2.error):
            cv2.destroyWindow(WINDOW_NAME)
