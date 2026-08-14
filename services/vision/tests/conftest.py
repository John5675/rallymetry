import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg  # type: ignore[import-untyped]
import numpy as np
import pytest

from pickleball_vision.calibration import (
    CalibrationCorrespondence,
    CalibrationSource,
    fit_calibration,
    save_calibration,
)
from pickleball_vision.court import CourtDimensions, ImagePoint, court_landmarks
from pickleball_vision.video import inspect_video

SYNTHETIC_WIDTH = 96
SYNTHETIC_HEIGHT = 64
SYNTHETIC_FPS = 7.5
SYNTHETIC_FRAME_COUNT = 12


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Generate a small non-integer-FPS video without private test footage."""

    video_path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter.fourcc(*"MJPG"),
        SYNTHETIC_FPS,
        (SYNTHETIC_WIDTH, SYNTHETIC_HEIGHT),
    )
    if not writer.isOpened():
        pytest.fail("OpenCV MJPG codec is unavailable in the test environment")

    try:
        for frame_index in range(SYNTHETIC_FRAME_COUNT):
            frame = np.full(
                (SYNTHETIC_HEIGHT, SYNTHETIC_WIDTH, 3),
                frame_index * 16,
                dtype=np.uint8,
            )
            cv2.putText(
                frame,
                str(frame_index),
                (8, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            writer.write(frame)
    finally:
        writer.release()

    if not video_path.is_file() or video_path.stat().st_size == 0:
        pytest.fail("OpenCV did not create the synthetic test video")
    return video_path


@pytest.fixture
def synthetic_media_with_audio(synthetic_video: Path, tmp_path: Path) -> Path:
    """Mux generated mono PCM audio with the synthetic video via bundled FFmpeg."""

    media_path = tmp_path / "synthetic-with-audio.mkv"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(synthetic_video),
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=880:sample_rate=48000:duration={SYNTHETIC_FRAME_COUNT / SYNTHETIC_FPS}",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "pcm_s16le",
        "-shortest",
        str(media_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        pytest.fail(f"FFmpeg did not create synthetic A/V media: {completed.stderr}")
    if not media_path.is_file() or media_path.stat().st_size == 0:
        pytest.fail("FFmpeg produced no synthetic A/V media")
    return media_path


@pytest.fixture
def synthetic_media_with_audio_gap(synthetic_media_with_audio: Path, tmp_path: Path) -> Path:
    """Create media whose audio packet timestamps contain a 200 ms internal gap."""

    media_path = tmp_path / "synthetic-with-audio-gap.mkv"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(synthetic_media_with_audio),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-c:v",
        "copy",
        "-af",
        "asetpts=PTS+if(gte(T\\,0.8)\\,0.2/TB\\,0)",
        "-c:a",
        "pcm_s16le",
        str(media_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        pytest.fail(f"FFmpeg did not create timestamp-gap media: {completed.stderr}")
    return media_path


@pytest.fixture
def synthetic_calibration(synthetic_video: Path, tmp_path: Path) -> Path:
    """Fit a valid four-corner calibration for the generated test video."""

    metadata = inspect_video(synthetic_video)
    landmarks = court_landmarks(CourtDimensions())
    selected_landmarks = (landmarks[0], landmarks[1], landmarks[8], landmarks[9])
    image_points = (
        ImagePoint(5, 58),
        ImagePoint(91, 58),
        ImagePoint(30, 5),
        ImagePoint(66, 5),
    )
    correspondences = tuple(
        CalibrationCorrespondence(
            landmark=landmark.name,
            label=landmark.label,
            image_point=image_point,
            court_point=landmark.court_point,
        )
        for landmark, image_point in zip(selected_landmarks, image_points, strict=True)
    )
    calibration = fit_calibration(
        source=CalibrationSource(
            video_path=metadata.path,
            requested_timestamp_s=0.0,
            frame_index=0,
            frame_timestamp_s=0.0,
            frame_width_px=metadata.width,
            frame_height_px=metadata.height,
            fps=metadata.fps,
        ),
        court=CourtDimensions(),
        correspondences=correspondences,
    )
    return save_calibration(calibration, tmp_path / "calibration.json")
