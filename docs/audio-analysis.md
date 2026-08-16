# Audio Analysis

Milestone 10 produces synchronized audio evidence that later multimodal stages may
use. It does not infer paddle contacts, bounces, rallies, or primary-court ownership.
A sound from a neighboring court can look identical to a primary-match sound when
considered without video.

## Command and artifacts

From `services/vision`:

```bash
uv run pickleball-vision analyze-audio /absolute/path/to/match.mp4 \
  --output-dir ../../output/audio-analysis
```

The command never modifies the source recording. When audio is present it writes:

- `analysis-audio.wav`: signed 16-bit PCM analysis audio with source channels
  preserved;
- `analysis-audio.wav.metadata.json`: extraction, conversion, stream, and timing
  provenance;
- `audio-events.json`: raw time-window observations and a separate list of generic
  transient candidates;
- `audio-summary.json`: availability, configuration, thresholds, and counts;
- `waveform.png`: per-channel amplitude envelope on the canonical media timeline;
- `audio-events.png`: onset-strength trace and transient markers.

The WAV and its sidecar are additional inspectability artifacts. The four required
JSON/PNG artifacts are always created, including when no audio stream exists.
Generated media and output remain local and must not be committed.

## Observation and candidate boundary

`rawAudioFeatureObservations` contains overlapping time-localized signal windows.
Each observation retains sample indices, analysis-relative and source-media
timestamps, duration, peak amplitude, RMS energy, onset strength, frequency-domain
summary, and channel-specific values.

`audioEventCandidates` is derived from local peaks above a robust spectral-flux
threshold. Candidates are separated conservatively in time and retain their
supporting raw observation, confidence, channel evidence, and timing provenance.
Their only type is `TRANSIENT`; `semanticClassification` is `null`. The output also
contains an empty `semanticEvents` list to make the boundary explicit.

No transient proves a contact or bounce. Speech, shoes, wind, traffic, and adjacent
courts may all create candidates. Later multimodal stages must use visual
plausibility, retain uncertainty, and fall back to vision when audio is absent.

## Timing

Candidate `analysisTimestampSeconds` is relative to sample zero in the extracted
WAV. `mediaTimestampSeconds` is the canonical source-media time:

```text
mediaTimestamp = (audioStartTime or 0)
               + peakSampleIndex / analysisSampleRate
               + audioVideoOffsetMs / 1000
```

The configured offset defaults to zero. Change it only after measuring a consistent
offset against visible events. Both the configured offset and the exact mapping are
persisted in `audio-events.json`.

## Configuration

All settings use the `PICKLEBALL_VISION_` prefix:

| Setting suffix | Default | Meaning |
| --- | ---: | --- |
| `AUDIO_ANALYSIS_SAMPLE_RATE_HZ` | `16000` | PCM analysis rate |
| `AUDIO_ANALYSIS_ONSET_SENSITIVITY` | `4.0` | Robust deviations above the onset baseline |
| `AUDIO_ANALYSIS_MINIMUM_EVENT_SEPARATION_MS` | `80` | Minimum retained transient separation |
| `AUDIO_ANALYSIS_CHANNEL_MODE` | `combined` | Detect from combined evidence or `per_channel` maximum |
| `AUDIO_ANALYSIS_FRAME_DURATION_MS` | `32` | Signal-feature window duration |
| `AUDIO_ANALYSIS_HOP_DURATION_MS` | `10` | Feature-window step |
| `AUDIO_VIDEO_OFFSET_MS` | `0` | Correction applied on the canonical timeline |

Channels are always retained in feature output. `combined` affects candidate
selection only; it does not downmix the stored WAV or discard channel observations.
A higher onset sensitivity normally yields fewer candidates. Tune it against held
out rallies and environmental noise rather than one favorable exchange.

## No-audio behavior

When no audio stream exists, the command succeeds. Both JSON artifacts set
`audioAnalysisAvailable` to `false`, contain no observations or candidates, and
declare that vision-only fallback is available. The PNG artifacts explain that the
source has no audio. No other pipeline stage is invalidated.

## Manual comparison against real rallies

For 5–10 rallies spread across the recording:

1. Note source-video timestamps for several visually plausible paddle contacts and
   bounces, plus a few obvious neighboring-court sounds or noise-only intervals.
2. Compare those times with the red markers in `audio-events.png`. Use
   `audio-events.json` for precise candidate timestamps and confidence.
3. Review video at reduced speed around each marker. Record whether it is aligned
   with primary-court visual evidence, aligned with an unrelated source, or
   uncertain. Do not relabel the candidate as a contact or bounce.
4. If every marker is consistently early or late, measure that shift and rerun with
   `PICKLEBALL_VISION_AUDIO_VIDEO_OFFSET_MS`. If timing is aligned but candidates
   are too dense or sparse, adjust sensitivity separately.
5. Confirm that visually plausible events with no audio candidate remain usable by
   the future vision-only path.

This review evaluates timestamp alignment and candidate usefulness only. Semantic
event accuracy belongs to later multimodal milestones.
