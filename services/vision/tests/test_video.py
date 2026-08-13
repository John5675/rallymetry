import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from pickleball_vision.errors import (
    InvalidSampleCountError,
    InvalidTimestampError,
    OutputWriteError,
    VideoNotFoundError,
    VideoUnreadableError,
)
from pickleball_vision.video import (
    extract_frame,
    inspect_video,
    sample_frame_indices,
    sample_frames,
)

SYNTHETIC_WIDTH = 96
SYNTHETIC_HEIGHT = 64
SYNTHETIC_FPS = 7.5
SYNTHETIC_FRAME_COUNT = 12


def test_inspect_video_reports_resolved_metadata(synthetic_video: Path) -> None:
    metadata = inspect_video(synthetic_video)

    assert metadata.filename == "synthetic.avi"
    assert metadata.path == synthetic_video.resolve()
    assert metadata.width == SYNTHETIC_WIDTH
    assert metadata.height == SYNTHETIC_HEIGHT
    assert metadata.fps == pytest.approx(SYNTHETIC_FPS)
    assert metadata.frame_count == SYNTHETIC_FRAME_COUNT
    assert metadata.duration == pytest.approx(SYNTHETIC_FRAME_COUNT / SYNTHETIC_FPS)
    assert metadata.codec is None or isinstance(metadata.codec, str)


def test_inspect_video_rejects_missing_non_file_and_corrupt_inputs(tmp_path: Path) -> None:
    with pytest.raises(VideoNotFoundError):
        inspect_video(tmp_path / "missing.mp4")

    with pytest.raises(VideoUnreadableError, match="not a regular file"):
        inspect_video(tmp_path)

    corrupt_video = tmp_path / "corrupt.mp4"
    corrupt_video.write_bytes(b"not a video")
    with pytest.raises(VideoUnreadableError, match="OpenCV could not open"):
        inspect_video(corrupt_video)


def test_extract_frame_preserves_source_resolution(synthetic_video: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "frame.png"

    artifact = extract_frame(
        synthetic_video,
        timestamp_seconds=0.8,
        output_path=output_path,
    )

    assert artifact.output_path == output_path.resolve()
    assert artifact.frame_index == 6
    assert artifact.timestamp == pytest.approx(0.8)
    assert artifact.width == SYNTHETIC_WIDTH
    assert artifact.height == SYNTHETIC_HEIGHT

    decoded_image = cv2.imread(str(output_path))
    assert decoded_image is not None
    image = np.asarray(decoded_image)
    assert image.shape[:2] == (SYNTHETIC_HEIGHT, SYNTHETIC_WIDTH)


@pytest.mark.parametrize("timestamp", [-0.1, math.nan, 1.6, math.inf])
def test_extract_frame_rejects_invalid_timestamps(
    synthetic_video: Path,
    tmp_path: Path,
    timestamp: float,
) -> None:
    with pytest.raises(InvalidTimestampError):
        extract_frame(
            synthetic_video,
            timestamp_seconds=timestamp,
            output_path=tmp_path / "frame.jpg",
        )


def test_extract_frame_rejects_unsupported_output_extension(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(OutputWriteError, match="unsupported image extension"):
        extract_frame(
            synthetic_video,
            timestamp_seconds=0,
            output_path=tmp_path / "frame.txt",
        )


def test_sample_frame_indices_span_the_video() -> None:
    assert sample_frame_indices(frame_count=12, count=1) == (5,)
    assert sample_frame_indices(frame_count=12, count=5) == (0, 3, 6, 8, 11)
    assert sample_frame_indices(frame_count=4, count=4) == (0, 1, 2, 3)


@pytest.mark.parametrize("count", [0, -1, 13])
def test_sample_frame_indices_reject_invalid_counts(count: int) -> None:
    with pytest.raises(InvalidSampleCountError):
        sample_frame_indices(frame_count=SYNTHETIC_FRAME_COUNT, count=count)


def test_sample_frames_writes_full_resolution_images_across_source(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "samples"

    artifacts = sample_frames(synthetic_video, count=5, output_dir=output_dir)

    assert tuple(artifact.frame_index for artifact in artifacts) == (0, 3, 6, 8, 11)
    assert all(artifact.output_path.is_file() for artifact in artifacts)
    assert all(artifact.width == SYNTHETIC_WIDTH for artifact in artifacts)
    assert all(artifact.height == SYNTHETIC_HEIGHT for artifact in artifacts)
