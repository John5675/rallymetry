import json
import subprocess
import wave
from pathlib import Path

import cv2
import imageio_ffmpeg  # type: ignore[import-untyped]
import numpy as np
import pytest

from pickleball_vision.audio_analysis import (
    AudioAnalysisResult,
    analyze_audio_samples,
    read_pcm16_wav,
)
from pickleball_vision.audio_analysis_workflow import analyze_audio_in_video
from pickleball_vision.config import (
    AudioAnalysisChannelMode,
    AudioAnalysisSettings,
)
from pickleball_vision.media import MediaTimeline

SAMPLE_RATE_HZ = 16_000


def _noise(duration_s: float, channels: int, *, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(
        0,
        0.0015,
        (round(duration_s * SAMPLE_RATE_HZ), channels),
    ).astype(np.float32)


def _add_impulse(
    samples: np.ndarray,
    timestamp_s: float,
    *,
    channel: int,
    amplitude: float = 0.9,
) -> None:
    start = round(timestamp_s * SAMPLE_RATE_HZ)
    shape = np.asarray((0.20, 0.55, 1.0, 0.55, 0.20), dtype=np.float32)
    samples[start : start + shape.size, channel] += amplitude * shape


def _analyze(
    samples: np.ndarray,
    *,
    settings: AudioAnalysisSettings | None = None,
    offset_ms: float = 0.0,
    audio_start_s: float = 0.0,
) -> AudioAnalysisResult:
    return analyze_audio_samples(
        samples,
        sample_rate_hz=SAMPLE_RATE_HZ,
        audio_start_time_s=audio_start_s,
        timeline=MediaTimeline(audio_video_offset_ms=offset_ms),
        settings=settings or AudioAnalysisSettings(),
    )


def _candidate_times(result: AudioAnalysisResult) -> list[float]:
    return [candidate.media_timestamp_s for candidate in result.candidates]


def _write_pcm_wav(path: Path, samples: np.ndarray) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.round(clipped * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(samples.shape[1])
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE_HZ)
        output.writeframes(pcm.tobytes())


def _mux_audio(video: Path, wav: Path, output: Path) -> None:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-i",
        str(wav),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "pcm_s16le",
        "-shortest",
        str(output),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        pytest.fail(f"FFmpeg could not mux synthetic transient media: {completed.stderr}")


def test_known_impulses_are_detected_amid_noise() -> None:
    samples = _noise(1.5, 1)
    for timestamp_s in (0.25, 0.75, 1.20):
        _add_impulse(samples, timestamp_s, channel=0)

    result = _analyze(samples)

    assert _candidate_times(result) == pytest.approx((0.25, 0.75, 1.20), abs=0.02)
    assert all(candidate.confidence > 0.5 for candidate in result.candidates)
    assert all(
        candidate.features["paddleOrBounceInferred"] is False for candidate in result.candidates
    )


def test_quiet_audio_has_no_transient_candidates() -> None:
    samples = np.zeros((SAMPLE_RATE_HZ, 1), dtype=np.float32)

    result = _analyze(samples)

    assert result.candidates == ()
    assert result.feature_observations


def test_clustered_impulses_respect_minimum_separation() -> None:
    samples = _noise(1.2, 1)
    for timestamp_s in (0.40, 0.45, 0.90):
        _add_impulse(samples, timestamp_s, channel=0)
    settings = AudioAnalysisSettings(minimum_event_separation_ms=100.0)

    result = _analyze(samples, settings=settings)

    times = _candidate_times(result)
    assert len(times) == 2
    assert min(abs(times[0] - expected) for expected in (0.40, 0.45)) < 0.02
    assert times[1] == pytest.approx(0.90, abs=0.02)


def test_per_channel_analysis_retains_stereo_differences() -> None:
    samples = _noise(1.3, 2)
    _add_impulse(samples, 0.30, channel=0)
    _add_impulse(samples, 0.95, channel=1)
    settings = AudioAnalysisSettings(
        channel_mode=AudioAnalysisChannelMode.PER_CHANNEL,
    )

    result = _analyze(samples, settings=settings)

    assert _candidate_times(result) == pytest.approx((0.30, 0.95), abs=0.02)
    first_channels = result.candidates[0].channel_data
    second_channels = result.candidates[1].channel_data
    assert first_channels[0]["triggered"] is True
    assert second_channels[1]["triggered"] is True
    assert result.channel_count == 2


def test_candidate_uses_canonical_timeline_with_nonzero_offset() -> None:
    samples = _noise(0.8, 1)
    _add_impulse(samples, 0.20, channel=0)

    result = _analyze(samples, offset_ms=35.0, audio_start_s=1.5)

    assert _candidate_times(result) == pytest.approx((1.735,), abs=0.02)
    assert result.candidates[0].analysis_timestamp_s == pytest.approx(0.20, abs=0.02)


def test_audio_workflow_writes_separate_observations_and_generic_candidates(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    samples = _noise(1.5, 2)
    _add_impulse(samples, 0.30, channel=0)
    _add_impulse(samples, 0.36, channel=1)
    _add_impulse(samples, 1.05, channel=1)
    wav = tmp_path / "impulses.wav"
    media = tmp_path / "transient-media.mkv"
    _write_pcm_wav(wav, samples)
    _mux_audio(synthetic_video, wav, media)
    output = tmp_path / "analysis"

    artifacts = analyze_audio_in_video(
        media,
        output_dir=output,
        settings=AudioAnalysisSettings(
            analysis_sample_rate_hz=8_000,
            channel_mode=AudioAnalysisChannelMode.PER_CHANNEL,
        ),
        timeline=MediaTimeline(audio_video_offset_ms=20.0),
    )

    events = json.loads(artifacts.events_path.read_text(encoding="utf-8"))
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert artifacts.audio_analysis_available is True
    assert artifacts.analysis_audio_path is not None
    assert read_pcm16_wav(artifacts.analysis_audio_path).channel_count == 2
    assert summary["statistics"]["analysisSampleRateHz"] == 8_000
    assert events["timeline"]["audioVideoOffsetMs"] == 20.0
    assert events["rawAudioFeatureObservations"]
    assert events["audioEventCandidates"]
    assert events["semanticEvents"] == []
    assert all(
        candidate["candidateType"] == "TRANSIENT" and candidate["semanticClassification"] is None
        for candidate in events["audioEventCandidates"]
    )
    for path in (
        artifacts.waveform_path,
        artifacts.events_image_path,
    ):
        image = cv2.imread(str(path))
        assert image is not None and image.size > 0


def test_no_audio_workflow_exits_gracefully_with_required_artifacts(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    artifacts = analyze_audio_in_video(
        synthetic_video,
        output_dir=tmp_path / "no-audio-analysis",
        settings=AudioAnalysisSettings(),
        timeline=MediaTimeline(),
    )

    events = json.loads(artifacts.events_path.read_text(encoding="utf-8"))
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert artifacts.audio_analysis_available is False
    assert artifacts.analysis_audio_path is None
    assert events["audioAnalysisAvailable"] is False
    assert events["rawAudioFeatureObservations"] == []
    assert events["audioEventCandidates"] == []
    assert events["limitations"]["visionOnlyFallbackAvailable"] is True
    assert summary["audioAnalysisAvailable"] is False
    assert summary["limitations"]["visionOnlyFallbackAvailable"] is True
    assert artifacts.waveform_path.is_file()
    assert artifacts.events_image_path.is_file()
