from pathlib import Path

import cv2
import numpy as np
import pytest

from pickleball_vision.errors import OutputWriteError
from pickleball_vision.video_output import CompressedVideoWriter


def test_compressed_video_writer_preserves_dimensions_fps_and_frames(tmp_path: Path) -> None:
    output = tmp_path / "artifact.mp4"
    writer = CompressedVideoWriter(output, fps=29.97, dimensions=(96, 64))
    for index in range(12):
        frame = np.full((64, 96, 3), (index * 17) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        assert round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 96
        assert round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 64
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(29.97, abs=0.02)
        assert round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    finally:
        capture.release()


def test_compressed_video_writer_rejects_wrong_frame_shape(tmp_path: Path) -> None:
    output = tmp_path / "artifact.mp4"
    writer = CompressedVideoWriter(output, fps=30.0, dimensions=(96, 64))

    with pytest.raises(OutputWriteError, match="frame must have shape"):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))

    assert not output.exists()
