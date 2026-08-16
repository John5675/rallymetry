"""Channel-aware raw audio features and generic transient candidates."""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pickleball_vision.config import AudioAnalysisChannelMode, AudioAnalysisSettings
from pickleball_vision.errors import AudioAnalysisError
from pickleball_vision.media import MediaTimeline

AUDIO_ANALYSIS_SCHEMA_VERSION = 1
FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class FrequencyDomainSummary:
    """Compact frequency evidence for one time-localized signal window."""

    spectral_centroid_hz: float
    dominant_frequency_hz: float
    low_band_energy_fraction: float
    mid_band_energy_fraction: float
    high_band_energy_fraction: float
    zero_crossing_rate: float

    def as_dict(self) -> dict[str, object]:
        return {
            "spectralCentroidHz": self.spectral_centroid_hz,
            "dominantFrequencyHz": self.dominant_frequency_hz,
            "bandEnergyFractions": {
                "lowBelow500Hz": self.low_band_energy_fraction,
                "mid500To4000Hz": self.mid_band_energy_fraction,
                "highAbove4000Hz": self.high_band_energy_fraction,
            },
            "zeroCrossingRate": self.zero_crossing_rate,
        }


@dataclass(frozen=True, slots=True)
class ChannelAudioFeature:
    """One channel's evidence for the same analysis window."""

    channel_index: int
    peak_amplitude: float
    rms_energy: float
    onset_strength: float
    frequency_summary: FrequencyDomainSummary

    def as_dict(self, *, triggered: bool | None = None) -> dict[str, object]:
        value: dict[str, object] = {
            "channelIndex": self.channel_index,
            "peakAmplitude": self.peak_amplitude,
            "rmsEnergy": self.rms_energy,
            "onsetStrength": self.onset_strength,
            "frequencyDomainSummary": self.frequency_summary.as_dict(),
        }
        if triggered is not None:
            value["triggered"] = triggered
        return value


@dataclass(frozen=True, slots=True)
class AudioFeatureObservation:
    """Raw time-localized signal evidence, independent of event candidates."""

    observation_id: str
    sample_start_index: int
    peak_sample_index: int
    analysis_timestamp_s: float
    media_timestamp_s: float
    duration_ms: float
    peak_amplitude: float
    rms_energy: float
    onset_strength: float
    frequency_summary: FrequencyDomainSummary
    channel_data: tuple[ChannelAudioFeature, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.observation_id,
            "sampleStartIndex": self.sample_start_index,
            "peakSampleIndex": self.peak_sample_index,
            "analysisTimestampSeconds": self.analysis_timestamp_s,
            "mediaTimestampSeconds": self.media_timestamp_s,
            "durationMs": self.duration_ms,
            "peakAmplitude": self.peak_amplitude,
            "rmsEnergy": self.rms_energy,
            "onsetStrength": self.onset_strength,
            "frequencyDomainSummary": self.frequency_summary.as_dict(),
            "channelData": [channel.as_dict() for channel in self.channel_data],
        }


@dataclass(frozen=True, slots=True)
class AudioEventCandidate:
    """Generic audio transient evidence with no pickleball semantic classification."""

    candidate_id: str
    media_timestamp_s: float
    analysis_timestamp_s: float
    duration_ms: float
    confidence: float
    peak_amplitude: float
    rms_energy: float
    onset_strength: float
    channel_data: tuple[dict[str, object], ...]
    features: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.candidate_id,
            "mediaTimestampSeconds": self.media_timestamp_s,
            "analysisTimestampSeconds": self.analysis_timestamp_s,
            "durationMs": self.duration_ms,
            "confidence": self.confidence,
            "peakAmplitude": self.peak_amplitude,
            "rmsEnergy": self.rms_energy,
            "onsetStrength": self.onset_strength,
            "channelData": list(self.channel_data),
            "features": self.features,
            "source": "AUDIO",
            "candidateType": "TRANSIENT",
            "semanticClassification": None,
        }


@dataclass(frozen=True, slots=True)
class AudioAnalysisResult:
    """Separate raw feature observations and derived transient candidates."""

    sample_rate_hz: int
    channel_count: int
    sample_count_per_channel: int
    duration_seconds: float
    feature_observations: tuple[AudioFeatureObservation, ...]
    candidates: tuple[AudioEventCandidate, ...]
    onset_thresholds: dict[str, object]


@dataclass(frozen=True, slots=True)
class PcmAudio:
    """Normalized interleaved PCM samples read from the analysis WAV."""

    samples: FloatArray
    sample_rate_hz: int
    channel_count: int


def read_pcm16_wav(path: Path) -> PcmAudio:
    """Read FFmpeg-produced signed 16-bit PCM while retaining every channel."""

    resolved = path.expanduser().resolve()
    try:
        with wave.open(str(resolved), "rb") as source:
            channel_count = source.getnchannels()
            sample_rate_hz = source.getframerate()
            sample_width = source.getsampwidth()
            compression = source.getcomptype()
            sample_count = source.getnframes()
            raw = source.readframes(sample_count)
        if channel_count < 1 or sample_rate_hz < 1:
            raise ValueError("WAV channel count and sample rate must be positive")
        if sample_width != 2 or compression != "NONE":
            raise ValueError("analysis WAV must be uncompressed signed 16-bit PCM")
        decoded = np.frombuffer(raw, dtype="<i2")
        expected_values = sample_count * channel_count
        if decoded.size != expected_values:
            raise ValueError(f"WAV contains {decoded.size} values; expected {expected_values}")
        samples = decoded.reshape(sample_count, channel_count).astype(np.float32) / 32768.0
        return PcmAudio(samples, sample_rate_hz, channel_count)
    except (OSError, EOFError, ValueError, wave.Error) as error:
        raise AudioAnalysisError(str(error), operation="read_analysis_audio") from error


def _frequency_summary(
    magnitudes: NDArray[np.float64],
    frequencies_hz: NDArray[np.float64],
    signal: FloatArray,
) -> FrequencyDomainSummary:
    power = np.square(magnitudes)
    total_power = float(np.sum(power))
    if total_power > 0:
        centroid = float(np.sum(frequencies_hz * power) / total_power)
        dominant_index = int(np.argmax(power[1:]) + 1) if power.size > 1 else 0
        dominant = float(frequencies_hz[dominant_index])
        low = float(np.sum(power[frequencies_hz < 500]) / total_power)
        mid = float(np.sum(power[(frequencies_hz >= 500) & (frequencies_hz < 4000)]) / total_power)
        high = max(0.0, 1.0 - low - mid)
    else:
        centroid = dominant = low = mid = high = 0.0
    if signal.size > 1:
        crossings = np.count_nonzero(np.signbit(signal[1:]) != np.signbit(signal[:-1]))
        zero_crossing_rate: float = float(crossings / (signal.size - 1))
    else:
        zero_crossing_rate = 0.0
    return FrequencyDomainSummary(
        spectral_centroid_hz=centroid,
        dominant_frequency_hz=dominant,
        low_band_energy_fraction=low,
        mid_band_energy_fraction=mid,
        high_band_energy_fraction=high,
        zero_crossing_rate=float(zero_crossing_rate),
    )


def _spectral_flux(
    magnitudes: NDArray[np.float64],
    previous_log_magnitudes: NDArray[np.float64] | None,
) -> tuple[float, NDArray[np.float64]]:
    current = np.log1p(magnitudes)
    if previous_log_magnitudes is None:
        return 0.0, current
    positive_difference = np.maximum(current - previous_log_magnitudes, 0.0)
    return float(np.sqrt(np.mean(np.square(positive_difference)))), current


def _robust_threshold(values: NDArray[np.float64], sensitivity: float) -> dict[str, float]:
    baseline = float(np.median(values)) if values.size else 0.0
    mad = float(np.median(np.abs(values - baseline))) if values.size else 0.0
    robust_scale = max(1.4826 * mad, 1e-6)
    return {
        "baseline": baseline,
        "robustScale": robust_scale,
        "sensitivity": sensitivity,
        "threshold": baseline + sensitivity * robust_scale,
    }


def _window_starts(sample_count: int, frame_length: int, hop_length: int) -> tuple[int, ...]:
    if sample_count <= frame_length:
        return (0,)
    starts = list(range(0, sample_count - frame_length + 1, hop_length))
    final_start = sample_count - frame_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def _candidate_confidence(z_score: float, sensitivity: float, peak_amplitude: float) -> float:
    excess = max(0.0, z_score - sensitivity)
    onset_confidence = 0.50 + 0.50 * (1 - math.exp(-excess / max(sensitivity, 1e-6)))
    amplitude_support = min(1.0, peak_amplitude * 4)
    return min(1.0, 0.85 * onset_confidence + 0.15 * amplitude_support)


def analyze_audio_samples(
    samples: FloatArray,
    *,
    sample_rate_hz: int,
    audio_start_time_s: float | None,
    timeline: MediaTimeline,
    settings: AudioAnalysisSettings,
) -> AudioAnalysisResult:
    """Extract raw window features and generic transient candidates from PCM samples."""

    if samples.ndim != 2 or samples.shape[0] < 1 or samples.shape[1] < 1:
        raise AudioAnalysisError(
            "samples must have shape (sample_count, channel_count)",
            operation="feature_extraction",
        )
    if sample_rate_hz < 1 or not np.all(np.isfinite(samples)):
        raise AudioAnalysisError(
            "sample rate must be positive and samples must be finite",
            operation="feature_extraction",
        )
    frame_length = max(2, round(sample_rate_hz * settings.frame_duration_ms / 1000))
    hop_length = max(1, round(sample_rate_hz * settings.hop_duration_ms / 1000))
    starts = _window_starts(samples.shape[0], frame_length, hop_length)
    window = np.hanning(frame_length).astype(np.float32)
    frequencies_hz = np.asarray(
        np.fft.rfftfreq(frame_length, d=1 / sample_rate_hz),
        dtype=np.float64,
    )
    channel_count = samples.shape[1]
    previous_channels: list[NDArray[np.float64] | None] = [None] * channel_count
    previous_combined: NDArray[np.float64] | None = None
    observations: list[AudioFeatureObservation] = []

    for observation_index, start in enumerate(starts):
        segment = samples[start : start + frame_length]
        if segment.shape[0] < frame_length:
            segment = np.pad(segment, ((0, frame_length - segment.shape[0]), (0, 0)))
        windowed = segment * window[:, np.newaxis]
        channel_magnitudes = np.abs(np.fft.rfft(windowed, axis=0))
        aggregate_magnitudes = np.sqrt(np.mean(np.square(channel_magnitudes), axis=1))
        aggregate_signal = np.mean(segment, axis=1).astype(np.float32)
        aggregate_onset, previous_combined = _spectral_flux(
            aggregate_magnitudes,
            previous_combined,
        )
        channel_features: list[ChannelAudioFeature] = []
        for channel_index in range(channel_count):
            channel_signal = segment[:, channel_index]
            onset, previous_channels[channel_index] = _spectral_flux(
                channel_magnitudes[:, channel_index],
                previous_channels[channel_index],
            )
            channel_features.append(
                ChannelAudioFeature(
                    channel_index=channel_index,
                    peak_amplitude=float(np.max(np.abs(channel_signal))),
                    rms_energy=float(np.sqrt(np.mean(np.square(channel_signal)))),
                    onset_strength=onset,
                    frequency_summary=_frequency_summary(
                        channel_magnitudes[:, channel_index],
                        frequencies_hz,
                        channel_signal,
                    ),
                )
            )
        absolute_channels = np.abs(segment)
        peak_flat_index = int(np.argmax(absolute_channels))
        peak_sample_offset = peak_flat_index // channel_count
        peak_sample_index = min(start + peak_sample_offset, samples.shape[0] - 1)
        observations.append(
            AudioFeatureObservation(
                observation_id=f"audio-feature-{observation_index:09d}",
                sample_start_index=start,
                peak_sample_index=peak_sample_index,
                analysis_timestamp_s=start / sample_rate_hz,
                media_timestamp_s=timeline.audio_sample_to_media_time(
                    start,
                    sample_rate_hz=sample_rate_hz,
                    audio_start_time_seconds=audio_start_time_s,
                ),
                duration_ms=frame_length / sample_rate_hz * 1000,
                peak_amplitude=float(np.max(absolute_channels)),
                rms_energy=float(np.sqrt(np.mean(np.square(segment)))),
                onset_strength=aggregate_onset,
                frequency_summary=_frequency_summary(
                    aggregate_magnitudes,
                    frequencies_hz,
                    aggregate_signal,
                ),
                channel_data=tuple(channel_features),
            )
        )

    combined_values = np.asarray(
        [observation.onset_strength for observation in observations],
        dtype=np.float64,
    )
    channel_values = np.asarray(
        [
            [channel.onset_strength for channel in observation.channel_data]
            for observation in observations
        ],
        dtype=np.float64,
    )
    combined_threshold = _robust_threshold(combined_values, settings.onset_sensitivity)
    channel_thresholds = tuple(
        _robust_threshold(channel_values[:, index], settings.onset_sensitivity)
        for index in range(channel_count)
    )
    if settings.channel_mode is AudioAnalysisChannelMode.COMBINED:
        detection_z_scores = (
            combined_values - combined_threshold["baseline"]
        ) / combined_threshold["robustScale"]
    else:
        channel_z_scores = np.column_stack(
            [
                (channel_values[:, index] - threshold["baseline"]) / threshold["robustScale"]
                for index, threshold in enumerate(channel_thresholds)
            ]
        )
        detection_z_scores = np.max(channel_z_scores, axis=1)

    peak_indices = [
        index
        for index in range(1, len(observations))
        if detection_z_scores[index] >= settings.onset_sensitivity
        and detection_z_scores[index] >= detection_z_scores[index - 1]
        and (
            index == len(observations) - 1
            or detection_z_scores[index] > detection_z_scores[index + 1]
        )
    ]
    minimum_separation_samples = round(sample_rate_hz * settings.minimum_event_separation_ms / 1000)
    selected_indices: list[int] = []
    for index in sorted(peak_indices, key=lambda item: detection_z_scores[item], reverse=True):
        sample_index = observations[index].peak_sample_index
        if any(
            abs(sample_index - observations[selected].peak_sample_index)
            < minimum_separation_samples
            for selected in selected_indices
        ):
            continue
        selected_indices.append(index)
    selected_indices.sort(key=lambda index: observations[index].peak_sample_index)

    candidates: list[AudioEventCandidate] = []
    for candidate_index, observation_index in enumerate(selected_indices, start=1):
        observation = observations[observation_index]
        channel_triggers = tuple(
            bool(
                (
                    channel_values[observation_index, channel_index]
                    - channel_thresholds[channel_index]["baseline"]
                )
                / channel_thresholds[channel_index]["robustScale"]
                >= settings.onset_sensitivity
            )
            for channel_index in range(channel_count)
        )
        media_timestamp = timeline.audio_sample_to_media_time(
            observation.peak_sample_index,
            sample_rate_hz=sample_rate_hz,
            audio_start_time_seconds=audio_start_time_s,
        )
        analysis_timestamp = observation.peak_sample_index / sample_rate_hz
        z_score = float(detection_z_scores[observation_index])
        candidates.append(
            AudioEventCandidate(
                candidate_id=f"audio-transient-{candidate_index:07d}",
                media_timestamp_s=media_timestamp,
                analysis_timestamp_s=analysis_timestamp,
                duration_ms=observation.duration_ms,
                confidence=_candidate_confidence(
                    z_score,
                    settings.onset_sensitivity,
                    observation.peak_amplitude,
                ),
                peak_amplitude=observation.peak_amplitude,
                rms_energy=observation.rms_energy,
                onset_strength=observation.onset_strength,
                channel_data=tuple(
                    channel.as_dict(triggered=channel_triggers[channel_index])
                    for channel_index, channel in enumerate(observation.channel_data)
                ),
                features={
                    "supportingFeatureObservationId": observation.observation_id,
                    "sampleIndex": observation.peak_sample_index,
                    "detectionZScore": z_score,
                    "frequencyDomainSummary": observation.frequency_summary.as_dict(),
                    "genericTransientOnly": True,
                    "paddleOrBounceInferred": False,
                    "primaryMatchAssociationInferred": False,
                },
            )
        )
    return AudioAnalysisResult(
        sample_rate_hz=sample_rate_hz,
        channel_count=channel_count,
        sample_count_per_channel=samples.shape[0],
        duration_seconds=samples.shape[0] / sample_rate_hz,
        feature_observations=tuple(observations),
        candidates=tuple(candidates),
        onset_thresholds={
            "mode": settings.channel_mode.value,
            "combined": combined_threshold,
            "channels": [
                {"channelIndex": index, **threshold}
                for index, threshold in enumerate(channel_thresholds)
            ],
        },
    )
