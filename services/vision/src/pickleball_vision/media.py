"""Audio-aware media inspection, extraction, and canonical timeline mapping."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import imageio_ffmpeg  # type: ignore[import-untyped]

from pickleball_vision.errors import (
    AudioExtractionError,
    AudioStreamNotFoundError,
    ClipExtractionError,
    InvalidAudioConversionError,
    MediaInspectionError,
)
from pickleball_vision.video import VideoMetadata, inspect_video

PCM_WAV_CODEC = "pcm_s16le"
SUPPORTED_EXPLICIT_CHANNEL_COUNTS = frozenset({1, 2})
MEDIA_METADATA_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class AudioStreamMetadata:
    """FFmpeg-reported metadata for the selected source audio stream."""

    stream_index: int
    codec: str | None
    sample_rate_hz: int | None
    channels: int | None
    channel_layout: str | None
    duration_seconds: float | None
    start_time_seconds: float | None
    sample_count: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "streamIndex": self.stream_index,
            "codec": self.codec,
            "sampleRate": self.sample_rate_hz,
            "channels": self.channels,
            "channelLayout": self.channel_layout,
            "duration": self.duration_seconds,
            "startTime": self.start_time_seconds,
            "sampleCount": self.sample_count,
        }


@dataclass(frozen=True, slots=True)
class MediaStreamProbe:
    """Model-independent stream facts returned by a media backend."""

    video_start_time_seconds: float | None
    audio: AudioStreamMetadata | None
    backend_name: str
    backend_version: str


class MediaBackend(Protocol):
    """Boundary that hides FFmpeg/PyAV and executable-specific details."""

    def probe(self, path: Path) -> MediaStreamProbe:
        """Inspect stream timing and audio properties without decoding video frames."""

    def extract_pcm_wav(
        self,
        source_path: Path,
        *,
        audio_stream_index: int,
        output_path: Path,
        sample_rate_hz: int | None,
        channels: int | None,
    ) -> None:
        """Decode one audio stream into a lossless PCM WAV representation."""


class ClipMediaBackend(Protocol):
    """Boundary for synchronized, lossless dataset review-clip extraction."""

    def extract_lossless_clip(
        self,
        source_path: Path,
        *,
        output_path: Path,
        start_time_s: float,
        duration_s: float,
        include_audio: bool,
    ) -> None:
        """Trim video and optional audio onto a zero-based clip timeline."""


def _probe_int(record: dict[str, object], key: str, *, optional: bool = False) -> int | None:
    value = record.get(key)
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"probe field {key!r} must be an integer")
    return value


def _probe_float(record: dict[str, object], key: str) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"probe field {key!r} must be numeric or null")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"probe field {key!r} must be finite")
    return parsed


def _probe_string(record: dict[str, object], key: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"probe field {key!r} must be a string or null")
    return value


def _parse_probe_output(path: Path, output: str) -> MediaStreamProbe:
    try:
        raw_payload = json.loads(output)
        if not isinstance(raw_payload, dict):
            raise ValueError("probe output must be a JSON object")
        payload = cast(dict[str, object], raw_payload)
        backend_name = _probe_string(payload, "backend_name")
        backend_version = _probe_string(payload, "backend_version")
        if backend_name is None or backend_version is None:
            raise ValueError("probe output omitted backend identity")
        raw_audio = payload.get("audio")
        audio = None
        if raw_audio is not None:
            if not isinstance(raw_audio, dict):
                raise ValueError("probe field 'audio' must be an object or null")
            audio_payload = cast(dict[str, object], raw_audio)
            stream_index = _probe_int(audio_payload, "stream_index")
            assert stream_index is not None
            audio = AudioStreamMetadata(
                stream_index=stream_index,
                codec=_probe_string(audio_payload, "codec"),
                sample_rate_hz=_probe_int(audio_payload, "sample_rate_hz", optional=True),
                channels=_probe_int(audio_payload, "channels", optional=True),
                channel_layout=_probe_string(audio_payload, "channel_layout"),
                duration_seconds=_probe_float(audio_payload, "duration_seconds"),
                start_time_seconds=_probe_float(audio_payload, "start_time_seconds"),
                sample_count=_probe_int(audio_payload, "sample_count", optional=True),
            )
        return MediaStreamProbe(
            video_start_time_seconds=_probe_float(payload, "video_start_time_seconds"),
            audio=audio,
            backend_name=backend_name,
            backend_version=backend_version,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise MediaInspectionError(
            str(path), reason=f"invalid FFmpeg probe output: {error}"
        ) from error


class FFmpegMediaBackend:
    """Media backend using PyAV for probing and a bundled FFmpeg for extraction."""

    def probe(self, path: Path) -> MediaStreamProbe:
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pickleball_vision._ffmpeg_probe", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            raise MediaInspectionError(str(path), reason=str(error)) from error
        if completed.returncode != 0:
            reason = completed.stderr.strip() or f"probe exited with status {completed.returncode}"
            raise MediaInspectionError(str(path), reason=reason)
        return _parse_probe_output(path, completed.stdout)

    def extract_pcm_wav(
        self,
        source_path: Path,
        *,
        audio_stream_index: int,
        output_path: Path,
        sample_rate_hz: int | None,
        channels: int | None,
    ) -> None:
        try:
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except RuntimeError as error:
            raise AudioExtractionError(str(output_path), reason=str(error)) from error

        command: list[str] = [
            executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-map",
            f"0:{audio_stream_index}",
            "-vn",
            "-af",
            "asetpts=PTS-STARTPTS,aresample=async=1:min_hard_comp=0:first_pts=0",
            "-c:a",
            PCM_WAV_CODEC,
        ]
        if sample_rate_hz is not None:
            command.extend(("-ar", str(sample_rate_hz)))
        if channels is not None:
            command.extend(("-ac", str(channels)))
        command.extend(("-f", "wav", str(output_path)))

        try:
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as error:
            raise AudioExtractionError(str(output_path), reason=str(error)) from error
        if completed.returncode != 0:
            output_path.unlink(missing_ok=True)
            reason = completed.stderr.strip() or f"FFmpeg exited with status {completed.returncode}"
            raise AudioExtractionError(str(output_path), reason=reason)


class FFmpegClipMediaBackend:
    """Bundled-FFmpeg implementation for lossless Matroska review clips."""

    def extract_lossless_clip(
        self,
        source_path: Path,
        *,
        output_path: Path,
        start_time_s: float,
        duration_s: float,
        include_audio: bool,
    ) -> None:
        try:
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except RuntimeError as error:
            raise ClipExtractionError(str(output_path), reason=str(error)) from error

        command = [
            executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-ss",
            f"{start_time_s:.9f}",
            "-t",
            f"{duration_s:.9f}",
            "-map",
            "0:v:0",
            "-vf",
            "setpts=PTS-STARTPTS",
            "-c:v",
            "ffv1",
            "-level",
            "3",
        ]
        if include_audio:
            command.extend(
                (
                    "-map",
                    "0:a:0",
                    "-af",
                    "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0",
                    "-c:a",
                    PCM_WAV_CODEC,
                )
            )
        command.extend(("-f", "matroska", str(output_path)))
        try:
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as error:
            raise ClipExtractionError(str(output_path), reason=str(error)) from error
        if completed.returncode != 0:
            reason = completed.stderr.strip() or f"FFmpeg exited with status {completed.returncode}"
            raise ClipExtractionError(str(output_path), reason=reason)


@dataclass(frozen=True, slots=True)
class MediaTimeline:
    """Map relative video/audio timestamps onto source-media presentation time."""

    audio_video_offset_ms: float = 0.0
    fusion_tolerance_ms: float = 90.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.audio_video_offset_ms):
            raise ValueError("audio_video_offset_ms must be finite")
        if not math.isfinite(self.fusion_tolerance_ms) or self.fusion_tolerance_ms <= 0:
            raise ValueError("fusion_tolerance_ms must be finite and positive")

    def video_timestamp_to_media_time(
        self,
        video_timestamp_seconds: float,
        *,
        video_start_time_seconds: float | None,
    ) -> float:
        """Map an existing frame-relative timestamp onto the canonical timeline."""

        if not math.isfinite(video_timestamp_seconds) or video_timestamp_seconds < 0:
            raise ValueError("video_timestamp_seconds must be finite and non-negative")
        return (video_start_time_seconds or 0.0) + video_timestamp_seconds

    def audio_sample_timestamp(self, sample_index: int, *, sample_rate_hz: int) -> float:
        """Return the sample's zero-based timestamp within extracted analysis audio."""

        if sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        return sample_index / sample_rate_hz

    def audio_sample_to_media_time(
        self,
        sample_index: int,
        *,
        sample_rate_hz: int,
        audio_start_time_seconds: float | None,
    ) -> float:
        """Map an extracted sample onto corrected source-media presentation time."""

        relative_time = self.audio_sample_timestamp(sample_index, sample_rate_hz=sample_rate_hz)
        return (
            (audio_start_time_seconds or 0.0) + relative_time + self.audio_video_offset_ms / 1000.0
        )


@dataclass(frozen=True, slots=True)
class MediaMetadata:
    """OpenCV video facts plus FFmpeg audio and stream-timing facts."""

    video: VideoMetadata
    video_start_time_seconds: float | None
    audio: AudioStreamMetadata | None
    timeline: MediaTimeline
    backend_name: str
    backend_version: str

    def as_dict(self) -> dict[str, object]:
        audio = self.audio
        return {
            **self.video.as_dict(),
            "hasAudio": audio is not None,
            "audioCodec": audio.codec if audio is not None else None,
            "audioSampleRate": audio.sample_rate_hz if audio is not None else None,
            "audioChannels": audio.channels if audio is not None else None,
            "audioChannelLayout": audio.channel_layout if audio is not None else None,
            "audioDuration": audio.duration_seconds if audio is not None else None,
            "audioStartTime": audio.start_time_seconds if audio is not None else None,
            "videoStartTime": self.video_start_time_seconds,
            "audioVideoOffsetMs": self.timeline.audio_video_offset_ms,
        }


def inspect_media(
    path: Path,
    *,
    timeline: MediaTimeline | None = None,
    backend: MediaBackend | None = None,
) -> MediaMetadata:
    """Inspect video decoding facts and synchronized stream metadata."""

    video = inspect_video(path)
    selected_backend = FFmpegMediaBackend() if backend is None else backend
    probe = selected_backend.probe(video.path)
    return MediaMetadata(
        video=video,
        video_start_time_seconds=probe.video_start_time_seconds,
        audio=probe.audio,
        timeline=timeline or MediaTimeline(),
        backend_name=probe.backend_name,
        backend_version=probe.backend_version,
    )


@dataclass(frozen=True, slots=True)
class AudioExtractionOptions:
    """Optional, explicit analysis-audio conversion request."""

    sample_rate_hz: int | None = None
    channels: int | None = None

    def __post_init__(self) -> None:
        if self.sample_rate_hz is not None and self.sample_rate_hz <= 0:
            raise InvalidAudioConversionError("sample rate must be a positive integer")
        if self.channels is not None and self.channels not in SUPPORTED_EXPLICIT_CHANNEL_COUNTS:
            raise InvalidAudioConversionError(
                "explicit channel conversion supports 1 (mono) or 2 (stereo) channels"
            )


@dataclass(frozen=True, slots=True)
class AudioExtractionArtifact:
    """Extracted PCM WAV and durable provenance/timeline sidecar."""

    source_path: Path
    output_path: Path
    metadata_path: Path
    source_audio: AudioStreamMetadata
    output_audio: AudioStreamMetadata
    video_start_time_seconds: float | None
    timeline: MediaTimeline
    requested: AudioExtractionOptions
    backend_name: str
    backend_version: str

    def as_dict(self) -> dict[str, object]:
        output_rate = self.output_audio.sample_rate_hz
        audio_start = self.source_audio.start_time_seconds
        sample_zero_media_time = (
            self.timeline.audio_sample_to_media_time(
                0,
                sample_rate_hz=output_rate,
                audio_start_time_seconds=audio_start,
            )
            if output_rate is not None
            else None
        )
        return {
            "schemaVersion": MEDIA_METADATA_SCHEMA_VERSION,
            "recordType": "analysis_audio_extraction",
            "sourceMedia": str(self.source_path),
            "outputAudio": str(self.output_path),
            "metadataPath": str(self.metadata_path),
            "sourcePreserved": True,
            "sourceAudio": self.source_audio.as_dict(),
            "analysisAudio": self.output_audio.as_dict(),
            "conversion": {
                "codec": PCM_WAV_CODEC,
                "requestedSampleRate": self.requested.sample_rate_hz,
                "requestedChannels": self.requested.channels,
                "sampleRateConverted": (
                    self.source_audio.sample_rate_hz != self.output_audio.sample_rate_hz
                ),
                "channelCountConverted": self.source_audio.channels != self.output_audio.channels,
                "sourceChannelLayout": self.source_audio.channel_layout,
                "outputChannelLayout": self.output_audio.channel_layout,
                "timestampSynchronization": {
                    "performed": True,
                    "method": (
                        "rebase first audio PTS to WAV sample zero and hard-compensate "
                        "internal timestamp discontinuities"
                    ),
                },
            },
            "timeline": {
                "canonicalTimeline": "source_media_presentation_time_seconds",
                "videoStartTime": self.video_start_time_seconds,
                "audioStartTime": audio_start,
                "audioVideoOffsetMs": self.timeline.audio_video_offset_ms,
                "audioSampleZeroMediaTime": sample_zero_media_time,
                "mapping": (
                    "mediaTime = (audioStartTime or 0) + sampleIndex / sampleRate "
                    "+ audioVideoOffsetMs / 1000"
                ),
            },
            "backend": {"name": self.backend_name, "version": self.backend_version},
        }


@dataclass(frozen=True, slots=True)
class MediaClipArtifact:
    """Lossless review clip with source-time and conversion provenance."""

    source_path: Path
    output_path: Path
    start_time_s: float
    end_time_s: float
    source: MediaMetadata
    output: MediaMetadata

    def as_dict(self) -> dict[str, object]:
        source_audio = self.source.audio
        output_audio = self.output.audio
        return {
            "source_media": str(self.source_path),
            "output_path": str(self.output_path),
            "source_preserved": True,
            "requested_range": {
                "start_time_s": self.start_time_s,
                "end_time_s": self.end_time_s,
                "duration_s": self.end_time_s - self.start_time_s,
                "end_exclusive": True,
            },
            "timeline": {
                "clip_time_zero_source_video_time_s": self.start_time_s,
                "mapping": "sourceVideoTime = clipVideoTimestamp + startTime",
                "audio_video_synchronization_preserved": True,
            },
            "conversion": {
                "container": "matroska",
                "video_codec": "ffv1",
                "video_lossless": True,
                "audio_codec": PCM_WAV_CODEC if source_audio is not None else None,
                "audio_lossless": source_audio is not None,
                "source_audio_channels": (
                    source_audio.channels if source_audio is not None else None
                ),
                "output_audio_channels": (
                    output_audio.channels if output_audio is not None else None
                ),
                "source_audio_sample_rate_hz": (
                    source_audio.sample_rate_hz if source_audio is not None else None
                ),
                "output_audio_sample_rate_hz": (
                    output_audio.sample_rate_hz if output_audio is not None else None
                ),
            },
            "output_media": self.output.as_dict(),
        }


def _resolve_audio_output(source_path: Path, output_path: Path) -> tuple[Path, Path]:
    resolved_output = output_path.expanduser().resolve()
    if resolved_output.suffix.lower() != ".wav":
        raise AudioExtractionError(str(resolved_output), reason="output must use a .wav extension")
    if resolved_output == source_path:
        raise AudioExtractionError(
            str(resolved_output),
            reason="output would destructively overwrite the source media",
        )
    if resolved_output.exists() and not resolved_output.is_file():
        raise AudioExtractionError(str(resolved_output), reason="path is not a regular file")
    metadata_path = Path(f"{resolved_output}.metadata.json")
    if metadata_path.exists() and not metadata_path.is_file():
        raise AudioExtractionError(str(metadata_path), reason="path is not a regular file")
    try:
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise AudioExtractionError(str(resolved_output), reason=str(error)) from error
    return resolved_output, metadata_path


def _write_audio_metadata(artifact: AudioExtractionArtifact) -> None:
    try:
        artifact.metadata_path.write_text(
            json.dumps(artifact.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise AudioExtractionError(str(artifact.metadata_path), reason=str(error)) from error


def extract_audio(
    path: Path,
    *,
    output_path: Path,
    options: AudioExtractionOptions | None = None,
    timeline: MediaTimeline | None = None,
    backend: MediaBackend | None = None,
) -> AudioExtractionArtifact:
    """Extract synchronized PCM WAV audio without modifying the source media."""

    selected_options = options or AudioExtractionOptions()
    selected_timeline = timeline or MediaTimeline()
    selected_backend = FFmpegMediaBackend() if backend is None else backend
    media = inspect_media(path, timeline=selected_timeline, backend=selected_backend)
    if media.audio is None:
        raise AudioStreamNotFoundError(str(media.video.path))
    resolved_output, metadata_path = _resolve_audio_output(media.video.path, output_path)
    selected_backend.extract_pcm_wav(
        media.video.path,
        audio_stream_index=media.audio.stream_index,
        output_path=resolved_output,
        sample_rate_hz=selected_options.sample_rate_hz,
        channels=selected_options.channels,
    )
    output_probe = selected_backend.probe(resolved_output)
    if output_probe.audio is None:
        resolved_output.unlink(missing_ok=True)
        raise AudioExtractionError(str(resolved_output), reason="output WAV has no audio stream")
    artifact = AudioExtractionArtifact(
        source_path=media.video.path,
        output_path=resolved_output,
        metadata_path=metadata_path,
        source_audio=media.audio,
        output_audio=output_probe.audio,
        video_start_time_seconds=media.video_start_time_seconds,
        timeline=selected_timeline,
        requested=selected_options,
        backend_name=output_probe.backend_name,
        backend_version=output_probe.backend_version,
    )
    _write_audio_metadata(artifact)
    return artifact


def extract_lossless_clip(
    path: Path,
    *,
    output_path: Path,
    start_time_s: float,
    end_time_s: float,
    timeline: MediaTimeline | None = None,
    backend: ClipMediaBackend | None = None,
) -> MediaClipArtifact:
    """Create a synchronized lossless MKV review clip without changing its source."""

    selected_timeline = timeline or MediaTimeline()
    source = inspect_media(path, timeline=selected_timeline)
    if (
        not math.isfinite(start_time_s)
        or not math.isfinite(end_time_s)
        or start_time_s < 0
        or end_time_s <= start_time_s
        or end_time_s > source.video.duration
    ):
        raise ClipExtractionError(
            str(output_path.expanduser().resolve()),
            reason=(
                "clip range must be finite and satisfy "
                f"0 <= start < end <= {source.video.duration:.6f} seconds"
            ),
        )

    resolved_output = output_path.expanduser().resolve()
    if resolved_output.suffix.lower() != ".mkv":
        raise ClipExtractionError(str(resolved_output), reason="output must use a .mkv extension")
    if resolved_output == source.video.path:
        raise ClipExtractionError(
            str(resolved_output),
            reason="output would destructively overwrite the source media",
        )
    if resolved_output.exists():
        raise ClipExtractionError(str(resolved_output), reason="output already exists")
    try:
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ClipExtractionError(str(resolved_output), reason=str(error)) from error

    temporary = resolved_output.with_name(f".{resolved_output.stem}.tmp.mkv")
    temporary.unlink(missing_ok=True)
    selected_backend = FFmpegClipMediaBackend() if backend is None else backend
    try:
        selected_backend.extract_lossless_clip(
            source.video.path,
            output_path=temporary,
            start_time_s=start_time_s,
            duration_s=end_time_s - start_time_s,
            include_audio=source.audio is not None,
        )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise ClipExtractionError(str(resolved_output), reason="FFmpeg wrote no clip data")
        temporary.replace(resolved_output)
        output = inspect_media(resolved_output, timeline=selected_timeline)
        if (source.audio is None) != (output.audio is None):
            raise ClipExtractionError(
                str(resolved_output),
                reason="output audio presence does not match the source clip",
            )
        if source.audio is not None and output.audio is not None:
            if source.audio.channels != output.audio.channels:
                raise ClipExtractionError(
                    str(resolved_output),
                    reason="output audio channel count does not match the source",
                )
            if source.audio.sample_rate_hz != output.audio.sample_rate_hz:
                raise ClipExtractionError(
                    str(resolved_output),
                    reason="output audio sample rate does not match the source",
                )
        if output.video.width != source.video.width or output.video.height != source.video.height:
            raise ClipExtractionError(
                str(resolved_output),
                reason="output resolution does not match the source video",
            )
    except (OSError, ClipExtractionError):
        temporary.unlink(missing_ok=True)
        resolved_output.unlink(missing_ok=True)
        raise
    return MediaClipArtifact(
        source_path=source.video.path,
        output_path=resolved_output,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        source=source,
        output=output,
    )
