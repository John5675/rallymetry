import json
from pathlib import Path

import pytest

from pickleball_vision.errors import (
    AudioStreamNotFoundError,
    InvalidAudioConversionError,
    VideoUnreadableError,
)
from pickleball_vision.media import (
    AudioExtractionOptions,
    AudioStreamMetadata,
    FFmpegMediaBackend,
    MediaStreamProbe,
    MediaTimeline,
    extract_audio,
    inspect_media,
)


def test_inspect_media_reports_audio_and_stream_timing(
    synthetic_media_with_audio: Path,
) -> None:
    metadata = inspect_media(synthetic_media_with_audio)

    assert metadata.audio is not None
    assert metadata.audio.codec == "pcm_s16le"
    assert metadata.audio.sample_rate_hz == 48000
    assert metadata.audio.channels == 1
    assert metadata.audio.channel_layout in {"mono", "1 channels"}
    assert metadata.audio.duration_seconds == pytest.approx(metadata.video.duration, abs=0.05)
    assert metadata.as_dict()["hasAudio"] is True
    assert metadata.as_dict()["audioStartTime"] == pytest.approx(0.0)
    assert metadata.as_dict()["videoStartTime"] == pytest.approx(0.0)


def test_inspect_media_reports_optional_audio_as_absent(synthetic_video: Path) -> None:
    report = inspect_media(synthetic_video).as_dict()

    assert report["hasAudio"] is False
    assert report["audioCodec"] is None
    assert report["audioSampleRate"] is None
    assert report["audioChannels"] is None
    assert report["audioDuration"] is None
    assert report["audioStartTime"] is None
    assert "videoStartTime" in report


def test_stream_probe_is_parsed_behind_media_backend(
    synthetic_video: Path,
) -> None:
    class StubBackend:
        def probe(self, _path: Path) -> MediaStreamProbe:
            return MediaStreamProbe(
                video_start_time_seconds=1.25,
                audio=AudioStreamMetadata(
                    stream_index=3,
                    codec="aac",
                    sample_rate_hz=44100,
                    channels=2,
                    channel_layout="stereo",
                    duration_seconds=1.55,
                    start_time_seconds=1.2,
                    sample_count=68355,
                ),
                backend_name="stub-ffmpeg",
                backend_version="test",
            )

        def extract_pcm_wav(
            self,
            source_path: Path,
            *,
            audio_stream_index: int,
            output_path: Path,
            sample_rate_hz: int | None,
            channels: int | None,
        ) -> None:
            raise AssertionError("not used")

    report = inspect_media(synthetic_video, backend=StubBackend()).as_dict()

    assert report["audioCodec"] == "aac"
    assert report["audioSampleRate"] == 44100
    assert report["audioChannels"] == 2
    assert report["audioDuration"] == pytest.approx(1.55)
    assert report["audioStartTime"] == pytest.approx(1.2)
    assert report["videoStartTime"] == pytest.approx(1.25)


def test_inspect_media_rejects_invalid_media(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.mp4"
    invalid.write_bytes(b"not media")

    with pytest.raises(VideoUnreadableError):
        inspect_media(invalid)


def test_extract_audio_preserves_rate_channels_and_timeline(
    synthetic_media_with_audio: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "analysis" / "audio.wav"

    artifact = extract_audio(synthetic_media_with_audio, output_path=output)

    assert artifact.output_path == output.resolve()
    assert artifact.output_audio.codec == "pcm_s16le"
    assert artifact.output_audio.sample_rate_hz == 48000
    assert artifact.output_audio.channels == 1
    assert artifact.output_audio.duration_seconds == pytest.approx(
        artifact.source_audio.duration_seconds,
        abs=1 / 48000,
    )
    sidecar = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    assert sidecar["sourcePreserved"] is True
    assert sidecar["conversion"]["sampleRateConverted"] is False
    assert sidecar["conversion"]["channelCountConverted"] is False
    assert sidecar["conversion"]["timestampSynchronization"]["performed"] is True
    assert sidecar["timeline"]["audioSampleZeroMediaTime"] == pytest.approx(0.0)


def test_extract_audio_records_explicit_rate_and_channel_conversion(
    synthetic_media_with_audio: Path,
    tmp_path: Path,
) -> None:
    artifact = extract_audio(
        synthetic_media_with_audio,
        output_path=tmp_path / "converted.wav",
        options=AudioExtractionOptions(sample_rate_hz=24000, channels=2),
    )

    assert artifact.output_audio.sample_rate_hz == 24000
    assert artifact.output_audio.channels == 2
    report = artifact.as_dict()
    conversion = report["conversion"]
    assert isinstance(conversion, dict)
    assert conversion["sampleRateConverted"] is True
    assert conversion["channelCountConverted"] is True


def test_extract_audio_preserves_internal_timestamp_gaps(
    synthetic_media_with_audio_gap: Path,
    tmp_path: Path,
) -> None:
    artifact = extract_audio(
        synthetic_media_with_audio_gap,
        output_path=tmp_path / "gap-preserved.wav",
    )

    assert artifact.output_audio.duration_seconds is not None
    assert artifact.output_audio.duration_seconds >= 1.79
    assert artifact.source_audio.duration_seconds == pytest.approx(
        artifact.output_audio.duration_seconds,
        abs=0.01,
    )


def test_extract_audio_gracefully_rejects_video_without_audio(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(AudioStreamNotFoundError, match="has no audio stream"):
        extract_audio(synthetic_video, output_path=tmp_path / "audio.wav")


def test_invalid_audio_conversion_is_typed() -> None:
    with pytest.raises(InvalidAudioConversionError):
        AudioExtractionOptions(sample_rate_hz=0)
    with pytest.raises(InvalidAudioConversionError):
        AudioExtractionOptions(channels=6)


def test_media_timeline_maps_video_and_audio_with_non_zero_offset() -> None:
    timeline = MediaTimeline(audio_video_offset_ms=35.0)

    assert timeline.video_timestamp_to_media_time(
        1.0,
        video_start_time_seconds=0.1,
    ) == pytest.approx(1.1)
    assert timeline.audio_sample_timestamp(48000, sample_rate_hz=48000) == pytest.approx(1.0)
    assert timeline.audio_sample_to_media_time(
        48000,
        sample_rate_hz=48000,
        audio_start_time_seconds=0.2,
    ) == pytest.approx(1.235)


def test_ffmpeg_backend_reports_version(synthetic_video: Path) -> None:
    probe = FFmpegMediaBackend().probe(synthetic_video)

    assert probe.backend_name == "ffmpeg-pyav"
    assert probe.backend_version
