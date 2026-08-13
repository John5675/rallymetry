"""Reusable OpenCV video inspection and frame-extraction operations."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from pickleball_vision.errors import (
    FrameDecodeError,
    InvalidSampleCountError,
    InvalidTimestampError,
    OutputWriteError,
    VideoNotFoundError,
    VideoUnreadableError,
)

Image = NDArray[np.uint8]
SUPPORTED_IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Validated metadata reported for a readable local video."""

    filename: str
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str | None

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible metadata using CLI field names."""

        return {
            "filename": self.filename,
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration": self.duration,
            "codec": self.codec,
        }


@dataclass(frozen=True, slots=True)
class ExtractedFrame:
    """A decoded frame written to a local image file."""

    output_path: Path
    frame_index: int
    timestamp: float
    width: int
    height: int

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible extraction metadata."""

        return {
            "output_path": str(self.output_path),
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "width": self.width,
            "height": self.height,
        }


def _resolved_video_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.exists():
        raise VideoNotFoundError(str(expanded))

    resolved = expanded.resolve()
    if not resolved.is_file():
        raise VideoUnreadableError(str(resolved), reason="path is not a regular file")
    return resolved


def _codec_name(fourcc_value: float) -> str | None:
    if not math.isfinite(fourcc_value) or fourcc_value <= 0:
        return None

    fourcc = int(fourcc_value)
    characters = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4))
    codec = characters.rstrip("\x00 ")
    if not codec or not all(character.isprintable() for character in codec):
        return None
    return codec


def _validated_metadata(capture: cv2.VideoCapture, path: Path) -> VideoMetadata:
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        raw_frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_value = float(capture.get(cv2.CAP_PROP_FOURCC))
        success, first_frame = capture.read()
    except cv2.error as error:
        raise VideoUnreadableError(
            str(path), reason="OpenCV failed while reading metadata or the first frame"
        ) from error

    if not math.isfinite(fps) or fps <= 0:
        raise VideoUnreadableError(str(path), reason="video reports an invalid FPS")
    if not math.isfinite(raw_frame_count) or raw_frame_count < 1:
        raise VideoUnreadableError(str(path), reason="video reports no decodable frames")

    frame_count = round(raw_frame_count)
    if not success or first_frame is None:
        raise VideoUnreadableError(str(path), reason="the first frame could not be decoded")

    frame = np.asarray(first_frame)
    if frame.ndim < 2 or frame.size == 0:
        raise VideoUnreadableError(str(path), reason="the first frame has invalid dimensions")
    height, width = (int(value) for value in frame.shape[:2])

    return VideoMetadata(
        filename=path.name,
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration=frame_count / fps,
        codec=_codec_name(fourcc_value),
    )


@contextmanager
def _open_video(path: Path) -> Iterator[tuple[cv2.VideoCapture, VideoMetadata]]:
    resolved = _resolved_video_path(path)
    try:
        capture = cv2.VideoCapture(str(resolved))
    except cv2.error as error:
        raise VideoUnreadableError(
            str(resolved), reason="OpenCV could not open the file"
        ) from error

    try:
        if not capture.isOpened():
            raise VideoUnreadableError(str(resolved), reason="OpenCV could not open the file")
        yield capture, _validated_metadata(capture, resolved)
    finally:
        capture.release()


def inspect_video(path: Path) -> VideoMetadata:
    """Inspect a local video and return validated source metadata."""

    with _open_video(path) as (_, metadata):
        return metadata


def _read_frame(capture: cv2.VideoCapture, metadata: VideoMetadata, frame_index: int) -> Image:
    capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    try:
        success, decoded_frame = capture.read()
    except cv2.error as error:
        raise FrameDecodeError(str(metadata.path), frame_index=frame_index) from error

    if not success or decoded_frame is None:
        raise FrameDecodeError(str(metadata.path), frame_index=frame_index)

    frame = np.asarray(decoded_frame)
    if frame.ndim < 2 or frame.size == 0:
        raise FrameDecodeError(str(metadata.path), frame_index=frame_index)
    if frame.dtype != np.uint8:
        raise FrameDecodeError(str(metadata.path), frame_index=frame_index)
    return cast(Image, frame)


def _resolved_output_path(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise OutputWriteError(
            str(output), reason=f"unsupported image extension; use one of {supported}"
        )
    if output.exists() and not output.is_file():
        raise OutputWriteError(str(output), reason="path is not a regular file")

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OutputWriteError(str(output), reason=str(error)) from error
    return output


def _write_frame(frame: Image, output_path: Path) -> None:
    try:
        written = cv2.imwrite(str(output_path), frame)
    except (cv2.error, OSError) as error:
        raise OutputWriteError(str(output_path), reason=str(error)) from error
    if not written:
        raise OutputWriteError(str(output_path), reason="OpenCV did not write the image")


def _frame_artifact(
    frame: Image, output_path: Path, frame_index: int, fps: float
) -> ExtractedFrame:
    height, width = (int(value) for value in frame.shape[:2])
    return ExtractedFrame(
        output_path=output_path,
        frame_index=frame_index,
        timestamp=frame_index / fps,
        width=width,
        height=height,
    )


def extract_frame(path: Path, *, timestamp_seconds: float, output_path: Path) -> ExtractedFrame:
    """Decode and write the frame containing a valid source timestamp."""

    with _open_video(path) as (capture, metadata):
        if (
            not math.isfinite(timestamp_seconds)
            or timestamp_seconds < 0
            or timestamp_seconds >= metadata.duration
        ):
            raise InvalidTimestampError(timestamp_seconds, duration_seconds=metadata.duration)

        frame_index = min(int(timestamp_seconds * metadata.fps), metadata.frame_count - 1)
        frame = _read_frame(capture, metadata, frame_index)
        resolved_output = _resolved_output_path(output_path)
        if resolved_output == metadata.path:
            raise OutputWriteError(
                str(resolved_output), reason="output would overwrite the source video"
            )
        _write_frame(frame, resolved_output)
        return _frame_artifact(frame, resolved_output, frame_index, metadata.fps)


def sample_frame_indices(*, frame_count: int, count: int) -> tuple[int, ...]:
    """Choose unique frame indices uniformly over the inclusive source span."""

    if count < 1 or count > frame_count:
        raise InvalidSampleCountError(count, frame_count=frame_count)
    if count == 1:
        return ((frame_count - 1) // 2,)

    denominator = count - 1
    last_frame = frame_count - 1
    return tuple(
        (sample_number * last_frame + denominator // 2) // denominator
        for sample_number in range(count)
    )


def _resolved_output_dir(path: Path) -> Path:
    output_dir = path.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise OutputWriteError(str(output_dir), reason="path is not a directory")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OutputWriteError(str(output_dir), reason=str(error)) from error
    return output_dir


def sample_frames(path: Path, *, count: int, output_dir: Path) -> tuple[ExtractedFrame, ...]:
    """Decode and write unique frames sampled across the complete source span."""

    with _open_video(path) as (capture, metadata):
        indices = sample_frame_indices(frame_count=metadata.frame_count, count=count)
        resolved_output_dir = _resolved_output_dir(output_dir)
        artifacts: list[ExtractedFrame] = []

        for sample_number, frame_index in enumerate(indices, start=1):
            frame = _read_frame(capture, metadata, frame_index)
            output_path = resolved_output_dir / (
                f"frame_{sample_number:04d}_index_{frame_index:09d}.jpg"
            )
            _write_frame(frame, output_path)
            artifacts.append(_frame_artifact(frame, output_path, frame_index, metadata.fps))

        return tuple(artifacts)
