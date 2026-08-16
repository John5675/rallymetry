"""Dependency-light waveform and transient-timeline visualizations."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from pickleball_vision.audio_analysis import (
    AudioEventCandidate,
    AudioFeatureObservation,
    FloatArray,
)
from pickleball_vision.video import Image

CANVAS_WIDTH = 1800
BACKGROUND = (24, 27, 31)
GRID = (64, 68, 74)
TEXT = (230, 232, 235)
WAVEFORM_COLORS = ((80, 220, 120), (255, 170, 80), (220, 120, 255), (90, 210, 240))
ONSET_COLOR = (60, 190, 255)
EVENT_COLOR = (70, 70, 245)


def _timeline_ticks(
    canvas: Image,
    *,
    left: int,
    right: int,
    top: int,
    bottom: int,
    start_time_s: float,
    end_time_s: float,
) -> None:
    for index in range(11):
        fraction = index / 10
        x_px = round(left + fraction * (right - left))
        cv2.line(canvas, (x_px, top), (x_px, bottom), GRID, 1, cv2.LINE_AA)
        timestamp = start_time_s + fraction * (end_time_s - start_time_s)
        cv2.putText(
            canvas,
            f"{timestamp:.1f}s",
            (x_px - 22, bottom + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            TEXT,
            1,
            cv2.LINE_AA,
        )


def render_no_audio_artifact(*, title: str) -> Image:
    """Render an explicit successful-no-audio artifact rather than omitting output."""

    canvas = np.full((420, CANVAS_WIDTH, 3), BACKGROUND, dtype=np.uint8)
    cv2.putText(
        canvas,
        title,
        (40, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        TEXT,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "audioAnalysisAvailable = false | vision-only fallback remains available",
        (40, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (90, 190, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def render_waveform(
    samples: FloatArray,
    *,
    sample_rate_hz: int,
    media_start_time_s: float,
) -> Image:
    """Render per-channel min/max envelopes across the complete media timeline."""

    channel_count = samples.shape[1]
    height = max(520, 180 * channel_count + 150)
    canvas = np.full((height, CANVAS_WIDTH, 3), BACKGROUND, dtype=np.uint8)
    left, right = 85, CANVAS_WIDTH - 35
    top, bottom = 90, height - 65
    duration_s = samples.shape[0] / sample_rate_hz
    _timeline_ticks(
        canvas,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        start_time_s=media_start_time_s,
        end_time_s=media_start_time_s + duration_s,
    )
    cv2.putText(
        canvas,
        "Analysis waveform by preserved channel",
        (30, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        TEXT,
        2,
        cv2.LINE_AA,
    )
    row_height = (bottom - top) / channel_count
    column_count = right - left + 1
    for channel_index in range(channel_count):
        center_y = round(top + (channel_index + 0.5) * row_height)
        amplitude_height = row_height * 0.42
        cv2.line(canvas, (left, center_y), (right, center_y), GRID, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"CH {channel_index + 1}",
            (18, center_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            WAVEFORM_COLORS[channel_index % len(WAVEFORM_COLORS)],
            1,
            cv2.LINE_AA,
        )
        for column in range(column_count):
            sample_start = column * samples.shape[0] // column_count
            sample_stop = max(sample_start + 1, (column + 1) * samples.shape[0] // column_count)
            values = samples[sample_start:sample_stop, channel_index]
            minimum = float(np.min(values))
            maximum = float(np.max(values))
            cv2.line(
                canvas,
                (left + column, round(center_y - maximum * amplitude_height)),
                (left + column, round(center_y - minimum * amplitude_height)),
                WAVEFORM_COLORS[channel_index % len(WAVEFORM_COLORS)],
                1,
            )
    return canvas


def render_audio_events(
    observations: Sequence[AudioFeatureObservation],
    candidates: Sequence[AudioEventCandidate],
    *,
    media_start_time_s: float,
    duration_seconds: float,
) -> Image:
    """Render onset strength plus generic transient markers on canonical media time."""

    height = 650
    canvas = np.full((height, CANVAS_WIDTH, 3), BACKGROUND, dtype=np.uint8)
    left, right = 85, CANVAS_WIDTH - 35
    top, bottom = 100, height - 70
    end_time_s = media_start_time_s + duration_seconds
    _timeline_ticks(
        canvas,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        start_time_s=media_start_time_s,
        end_time_s=end_time_s,
    )
    cv2.putText(
        canvas,
        "Generic audio transient candidates (not paddle contacts or bounces)",
        (30, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.90,
        TEXT,
        2,
        cv2.LINE_AA,
    )
    onset_values = np.asarray([item.onset_strength for item in observations], dtype=np.float64)
    scale = float(np.quantile(onset_values, 0.995)) if onset_values.size else 1.0
    scale = max(scale, 1e-9)
    column_values = np.zeros(right - left + 1, dtype=np.float64)
    for observation in observations:
        fraction = (observation.media_timestamp_s - media_start_time_s) / max(
            duration_seconds, 1e-9
        )
        column = min(len(column_values) - 1, max(0, round(fraction * (len(column_values) - 1))))
        column_values[column] = max(column_values[column], observation.onset_strength)
    points = np.asarray(
        [
            (
                left + index,
                round(bottom - min(1.0, value / scale) * (bottom - top)),
            )
            for index, value in enumerate(column_values)
        ],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [points], False, ONSET_COLOR, 2, cv2.LINE_AA)
    for candidate in candidates:
        fraction = (candidate.media_timestamp_s - media_start_time_s) / max(duration_seconds, 1e-9)
        x_px = round(left + min(1.0, max(0.0, fraction)) * (right - left))
        marker_bottom = top + 14 + round(candidate.confidence * 24)
        cv2.line(canvas, (x_px, top), (x_px, marker_bottom), EVENT_COLOR, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x_px, marker_bottom), 2, EVENT_COLOR, -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"transients: {len(candidates)} | yellow: onset strength | red: candidate",
        (30, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        TEXT,
        1,
        cv2.LINE_AA,
    )
    return canvas
