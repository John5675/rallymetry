"""Private PyAV probe executable isolated from OpenCV's bundled FFmpeg libraries."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import av


def _timestamp_seconds(value: int | None, time_base: object | None) -> float | None:
    if value is None or time_base is None:
        return None
    seconds = float(value * time_base)  # type: ignore[operator]
    return seconds if math.isfinite(seconds) else None


def probe_payload(path: Path) -> dict[str, object]:
    """Return JSON-compatible FFmpeg stream facts for the parent media boundary."""

    with av.open(str(path)) as container:
        video_stream = container.streams.video[0] if container.streams.video else None
        audio_stream = container.streams.audio[0] if container.streams.audio else None
        video_start = (
            _timestamp_seconds(video_stream.start_time, video_stream.time_base)
            if video_stream is not None
            else None
        )
        audio: dict[str, object] | None = None
        if audio_stream is not None:
            codec_context = audio_stream.codec_context
            sample_rate = codec_context.sample_rate or None
            duration = _timestamp_seconds(audio_stream.duration, audio_stream.time_base)
            if duration is None and container.duration is not None:
                duration = container.duration / av.time_base
            sample_count = (
                round(duration * sample_rate)
                if duration is not None and sample_rate is not None
                else None
            )
            audio = {
                "stream_index": audio_stream.index,
                "codec": codec_context.name or None,
                "sample_rate_hz": sample_rate,
                "channels": codec_context.channels or None,
                "channel_layout": (
                    codec_context.layout.name if codec_context.layout is not None else None
                ),
                "duration_seconds": duration,
                "start_time_seconds": _timestamp_seconds(
                    audio_stream.start_time,
                    audio_stream.time_base,
                ),
                "sample_count": sample_count,
            }
    return {
        "video_start_time_seconds": video_start,
        "audio": audio,
        "backend_name": "ffmpeg-pyav",
        "backend_version": av.__version__,
    }


def main(argv: list[str] | None = None) -> int:
    """Probe one local path and emit only JSON on standard output."""

    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: python -m pickleball_vision._ffmpeg_probe <media>", file=sys.stderr)
        return 2
    try:
        payload = probe_payload(Path(arguments[0]))
    except (av.FFmpegError, OSError, ValueError, IndexError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
