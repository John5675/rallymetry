"""Space-efficient, browser-compatible video output for generated artifacts."""

from __future__ import annotations

import subprocess
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO

import imageio_ffmpeg  # type: ignore[import-untyped]
import numpy as np

from pickleball_vision.errors import OutputWriteError
from pickleball_vision.video import Image

DEFAULT_VIDEO_CRF = 27
DEFAULT_VIDEO_PRESET = "veryfast"


class CompressedVideoWriter:
    """Stream BGR frames to bundled FFmpeg using H.264 instead of disk-heavy MP4V."""

    def __init__(
        self,
        path: Path,
        *,
        fps: float,
        dimensions: tuple[int, int],
        crf: int = DEFAULT_VIDEO_CRF,
        preset: str = DEFAULT_VIDEO_PRESET,
    ) -> None:
        self._path = path
        self._width, self._height = dimensions
        self._closed = False
        self._stderr_path = path.with_name(f".{path.name}.ffmpeg.stderr")
        try:
            executable = imageio_ffmpeg.get_ffmpeg_exe()
            self._stderr: BinaryIO = self._stderr_path.open("w+b")
            self._process = subprocess.Popen(
                [
                    executable,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-video_size",
                    f"{self._width}x{self._height}",
                    "-framerate",
                    f"{fps:.9f}",
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    preset,
                    "-crf",
                    str(crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr,
            )
        except (OSError, RuntimeError) as error:
            with suppress(AttributeError, OSError):
                self._stderr.close()
            self._stderr_path.unlink(missing_ok=True)
            raise OutputWriteError(str(path), reason=str(error)) from error

    def write(self, frame: Image) -> None:
        """Write one source-sized uint8 BGR frame or surface the FFmpeg failure."""

        if self._closed:
            raise OutputWriteError(str(self._path), reason="video writer is already closed")
        array = np.asarray(frame)
        expected_shape = (self._height, self._width, 3)
        if array.dtype != np.uint8 or array.shape != expected_shape:
            self.abort()
            raise OutputWriteError(
                str(self._path),
                reason=(
                    f"frame must have shape {expected_shape} and dtype uint8; "
                    f"received shape {array.shape} and dtype {array.dtype}"
                ),
            )
        stdin = self._process.stdin
        if stdin is None:
            self.abort()
            raise OutputWriteError(str(self._path), reason="FFmpeg input pipe is unavailable")
        payload = memoryview(np.ascontiguousarray(array).tobytes())
        try:
            while payload:
                written = stdin.write(payload)
                if written is None or written <= 0:
                    raise BrokenPipeError("FFmpeg accepted no frame bytes")
                payload = payload[written:]
        except (BrokenPipeError, OSError) as error:
            reason = self._failure_reason() or str(error)
            self.abort()
            raise OutputWriteError(str(self._path), reason=reason) from error

    def release(self) -> None:
        """Finalize the MP4 and raise a typed error when FFmpeg did not finish cleanly."""

        if self._closed:
            return
        self._closed = True
        stdin = self._process.stdin
        close_error: OSError | None = None
        if stdin is not None:
            try:
                stdin.close()
            except OSError as error:
                close_error = error
        return_code = self._process.wait()
        reason = self._failure_reason()
        self._close_stderr()
        if return_code != 0 or close_error is not None:
            self._path.unlink(missing_ok=True)
            detail = reason or str(close_error) if close_error is not None else reason
            raise OutputWriteError(
                str(self._path),
                reason=detail or f"FFmpeg exited with status {return_code}",
            )

    def abort(self) -> None:
        """Stop a partial encode and remove only its scoped temporary outputs."""

        if self._closed:
            return
        self._closed = True
        stdin = self._process.stdin
        if stdin is not None:
            with suppress(OSError):
                stdin.close()
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._close_stderr()
        self._path.unlink(missing_ok=True)

    def _failure_reason(self) -> str:
        try:
            self._stderr.flush()
            self._stderr.seek(0, 2)
            size = self._stderr.tell()
            self._stderr.seek(max(0, size - 4096))
            return self._stderr.read().decode("utf-8", errors="replace").strip()
        except (AttributeError, OSError):
            return ""

    def _close_stderr(self) -> None:
        with suppress(OSError):
            self._stderr.close()
        self._stderr_path.unlink(missing_ok=True)
