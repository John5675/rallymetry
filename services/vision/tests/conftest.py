from pathlib import Path

import cv2
import numpy as np
import pytest

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
