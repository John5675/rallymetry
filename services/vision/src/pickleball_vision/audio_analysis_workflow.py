"""Synchronized audio feature and generic transient extraction workflow."""

from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2

from pickleball_vision.audio_analysis import (
    AUDIO_ANALYSIS_SCHEMA_VERSION,
    AudioAnalysisResult,
    analyze_audio_samples,
    read_pcm16_wav,
)
from pickleball_vision.audio_analysis_render import (
    render_audio_events,
    render_no_audio_artifact,
    render_waveform,
)
from pickleball_vision.config import AudioAnalysisSettings
from pickleball_vision.errors import AudioAnalysisError, OutputWriteError
from pickleball_vision.media import (
    AudioExtractionOptions,
    MediaMetadata,
    MediaTimeline,
    extract_audio,
    inspect_media,
)
from pickleball_vision.video import Image

AUDIO_EVENTS_NAME = "audio-events.json"
AUDIO_SUMMARY_NAME = "audio-summary.json"
WAVEFORM_NAME = "waveform.png"
AUDIO_EVENTS_IMAGE_NAME = "audio-events.png"
ANALYSIS_AUDIO_NAME = "analysis-audio.wav"


@dataclass(frozen=True, slots=True)
class AudioAnalysisArtifacts:
    """Required artifacts and availability returned by the CLI workflow."""

    events_path: Path
    summary_path: Path
    waveform_path: Path
    events_image_path: Path
    analysis_audio_path: Path | None
    audio_analysis_available: bool
    feature_observation_count: int
    candidate_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "eventsPath": str(self.events_path),
            "summaryPath": str(self.summary_path),
            "waveformPath": str(self.waveform_path),
            "eventsImagePath": str(self.events_image_path),
            "analysisAudioPath": (
                str(self.analysis_audio_path) if self.analysis_audio_path is not None else None
            ),
            "audioAnalysisAvailable": self.audio_analysis_available,
            "featureObservationCount": self.feature_observation_count,
            "candidateCount": self.candidate_count,
        }


def _prepare_output_dir(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise OutputWriteError(str(output), reason="path is not a directory")
    events_path = output / AUDIO_EVENTS_NAME
    if events_path.exists():
        raise OutputWriteError(str(events_path), reason="audio-analysis output already exists")
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OutputWriteError(str(output), reason=str(error)) from error
    return output


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except (OSError, TypeError, ValueError) as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise OutputWriteError(str(path), reason=str(error)) from error


def _write_image(path: Path, image: Image) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        success, encoded = cv2.imencode(path.suffix, image)
        if not success:
            raise ValueError("OpenCV could not encode the visualization")
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(path)
    except (OSError, ValueError, cv2.error) as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise OutputWriteError(str(path), reason=str(error)) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise AudioAnalysisError(str(error), operation="hash_analysis_audio") from error
    return digest.hexdigest()


def _configuration_payload(
    settings: AudioAnalysisSettings,
    timeline: MediaTimeline,
) -> dict[str, object]:
    return {
        **settings.as_dict(),
        "audioVideoOffsetMs": timeline.audio_video_offset_ms,
        "fusionToleranceMs": timeline.fusion_tolerance_ms,
    }


def _statistics(result: AudioAnalysisResult) -> dict[str, object]:
    duration_minutes = result.duration_seconds / 60
    channel_trigger_counts = [0] * result.channel_count
    for candidate in result.candidates:
        for channel in candidate.channel_data:
            channel_index = channel.get("channelIndex")
            if channel.get("triggered") is True and isinstance(channel_index, int):
                channel_trigger_counts[channel_index] += 1
    return {
        "analysisDurationSeconds": result.duration_seconds,
        "analysisSampleRateHz": result.sample_rate_hz,
        "analysisChannels": result.channel_count,
        "sampleCountPerChannel": result.sample_count_per_channel,
        "featureObservationCount": len(result.feature_observations),
        "transientCandidateCount": len(result.candidates),
        "transientCandidatesPerMinute": (
            len(result.candidates) / duration_minutes if duration_minutes > 0 else 0.0
        ),
        "maximumPeakAmplitude": max(
            (observation.peak_amplitude for observation in result.feature_observations),
            default=0.0,
        ),
        "meanRmsEnergy": (
            sum(observation.rms_energy for observation in result.feature_observations)
            / len(result.feature_observations)
            if result.feature_observations
            else 0.0
        ),
        "candidateChannelTriggerCounts": [
            {"channelIndex": index, "count": count}
            for index, count in enumerate(channel_trigger_counts)
        ],
    }


def _artifact_payload(
    *,
    events_path: Path,
    summary_path: Path,
    waveform_path: Path,
    events_image_path: Path,
    analysis_audio_path: Path | None,
) -> dict[str, object]:
    return {
        "audioEvents": str(events_path),
        "audioSummary": str(summary_path),
        "waveform": str(waveform_path),
        "eventVisualization": str(events_image_path),
        "analysisAudio": str(analysis_audio_path) if analysis_audio_path is not None else None,
    }


def _no_audio_payload(
    *,
    media: MediaMetadata,
    configuration: dict[str, object],
    created_at_utc: str,
    artifacts: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    events = {
        "schemaVersion": AUDIO_ANALYSIS_SCHEMA_VERSION,
        "recordType": "audio_analysis_observations",
        "createdAtUtc": created_at_utc,
        "audioAnalysisAvailable": False,
        "sourceMedia": media.as_dict(),
        "configuration": configuration,
        "analysisRepresentation": None,
        "rawAudioFeatureObservations": [],
        "audioEventCandidates": [],
        "semanticEvents": [],
        "limitations": {
            "reason": "source media has no audio stream",
            "visionOnlyFallbackAvailable": True,
        },
        "artifacts": artifacts,
    }
    summary = {
        "schemaVersion": AUDIO_ANALYSIS_SCHEMA_VERSION,
        "recordType": "audio_analysis_summary",
        "createdAtUtc": created_at_utc,
        "audioAnalysisAvailable": False,
        "sourceMedia": media.as_dict(),
        "configuration": configuration,
        "statistics": {
            "featureObservationCount": 0,
            "transientCandidateCount": 0,
        },
        "limitations": {
            "reason": "source media has no audio stream",
            "visionOnlyFallbackAvailable": True,
        },
        "artifacts": artifacts,
    }
    return events, summary


def analyze_audio_in_video(
    video_path: Path,
    *,
    output_dir: Path,
    settings: AudioAnalysisSettings,
    timeline: MediaTimeline,
) -> AudioAnalysisArtifacts:
    """Extract synchronized raw audio features and generic transient candidates."""

    output = _prepare_output_dir(output_dir)
    events_path = output / AUDIO_EVENTS_NAME
    summary_path = output / AUDIO_SUMMARY_NAME
    waveform_path = output / WAVEFORM_NAME
    events_image_path = output / AUDIO_EVENTS_IMAGE_NAME
    media = inspect_media(video_path, timeline=timeline)
    created_at_utc = datetime.now(UTC).isoformat()
    configuration = _configuration_payload(settings, timeline)

    if media.audio is None:
        artifacts = _artifact_payload(
            events_path=events_path,
            summary_path=summary_path,
            waveform_path=waveform_path,
            events_image_path=events_image_path,
            analysis_audio_path=None,
        )
        events_payload, summary_payload = _no_audio_payload(
            media=media,
            configuration=configuration,
            created_at_utc=created_at_utc,
            artifacts=artifacts,
        )
        _write_image(
            waveform_path,
            render_no_audio_artifact(title="Analysis waveform"),
        )
        _write_image(
            events_image_path,
            render_no_audio_artifact(title="Audio transient timeline"),
        )
        _write_json(events_path, events_payload)
        _write_json(summary_path, summary_payload)
        return AudioAnalysisArtifacts(
            events_path=events_path,
            summary_path=summary_path,
            waveform_path=waveform_path,
            events_image_path=events_image_path,
            analysis_audio_path=None,
            audio_analysis_available=False,
            feature_observation_count=0,
            candidate_count=0,
        )

    analysis_audio_path = output / ANALYSIS_AUDIO_NAME
    extraction = extract_audio(
        media.video.path,
        output_path=analysis_audio_path,
        options=AudioExtractionOptions(
            sample_rate_hz=settings.analysis_sample_rate_hz,
            channels=None,
        ),
        timeline=timeline,
    )
    pcm = read_pcm16_wav(extraction.output_path)
    if pcm.sample_rate_hz != settings.analysis_sample_rate_hz:
        raise AudioAnalysisError(
            "extracted WAV sample rate does not match configured analysis rate",
            operation="validate_analysis_audio",
        )
    result = analyze_audio_samples(
        pcm.samples,
        sample_rate_hz=pcm.sample_rate_hz,
        audio_start_time_s=media.audio.start_time_seconds,
        timeline=timeline,
        settings=settings,
    )
    sample_zero_media_time = timeline.audio_sample_to_media_time(
        0,
        sample_rate_hz=result.sample_rate_hz,
        audio_start_time_seconds=media.audio.start_time_seconds,
    )
    _write_image(
        waveform_path,
        render_waveform(
            pcm.samples,
            sample_rate_hz=pcm.sample_rate_hz,
            media_start_time_s=sample_zero_media_time,
        ),
    )
    _write_image(
        events_image_path,
        render_audio_events(
            result.feature_observations,
            result.candidates,
            media_start_time_s=sample_zero_media_time,
            duration_seconds=result.duration_seconds,
        ),
    )
    statistics = _statistics(result)
    artifacts = _artifact_payload(
        events_path=events_path,
        summary_path=summary_path,
        waveform_path=waveform_path,
        events_image_path=events_image_path,
        analysis_audio_path=analysis_audio_path,
    )
    extraction_payload = extraction.as_dict()
    extraction_payload["sha256"] = _sha256(analysis_audio_path)
    events_payload = {
        "schemaVersion": AUDIO_ANALYSIS_SCHEMA_VERSION,
        "recordType": "audio_analysis_observations",
        "createdAtUtc": created_at_utc,
        "audioAnalysisAvailable": True,
        "sourceMedia": media.as_dict(),
        "configuration": configuration,
        "analysisRepresentation": extraction_payload,
        "timeline": {
            "canonicalTimeline": "source_media_presentation_time_seconds",
            "audioStartTimeSeconds": media.audio.start_time_seconds,
            "audioVideoOffsetMs": timeline.audio_video_offset_ms,
            "sampleZeroMediaTimeSeconds": sample_zero_media_time,
            "mapping": (
                "mediaTime = (audioStartTime or 0) + sampleIndex / sampleRate "
                "+ audioVideoOffsetMs / 1000"
            ),
        },
        "onsetThresholds": result.onset_thresholds,
        "rawAudioFeatureObservations": [
            observation.as_dict() for observation in result.feature_observations
        ],
        "audioEventCandidates": [candidate.as_dict() for candidate in result.candidates],
        "semanticEvents": [],
        "limitations": {
            "genericTransientOnly": True,
            "paddleOrBounceInferred": False,
            "primaryMatchAssociationInferred": False,
            "neighboringCourtSoundsMayBePresent": True,
        },
        "artifacts": artifacts,
    }
    summary_payload = {
        "schemaVersion": AUDIO_ANALYSIS_SCHEMA_VERSION,
        "recordType": "audio_analysis_summary",
        "createdAtUtc": created_at_utc,
        "audioAnalysisAvailable": True,
        "sourceMedia": media.as_dict(),
        "configuration": configuration,
        "statistics": statistics,
        "onsetThresholds": result.onset_thresholds,
        "artifacts": artifacts,
        "semanticEventCounts": {
            "paddleContacts": None,
            "bounces": None,
            "reason": "not inferred by the audio-analysis milestone",
        },
    }
    _write_json(events_path, events_payload)
    _write_json(summary_path, summary_payload)
    return AudioAnalysisArtifacts(
        events_path=events_path,
        summary_path=summary_path,
        waveform_path=waveform_path,
        events_image_path=events_image_path,
        analysis_audio_path=analysis_audio_path,
        audio_analysis_available=True,
        feature_observation_count=len(result.feature_observations),
        candidate_count=len(result.candidates),
    )
