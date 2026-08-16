# Media Timeline and Audio Extraction

## Scope

The local pipeline accepts a source recording with video and optional synchronized
audio. Computer-vision stages continue to work when audio is absent. The media
layer exposes stream facts and lossless analysis audio. The audio-analysis stage may
derive raw signal observations and generic transients, but it does not infer
pickleball semantics.

FFmpeg implementation details are isolated behind the vision service's media
backend. PyAV supplies FFmpeg stream probing, while the project-managed FFmpeg
executable performs PCM WAV extraction. Callers consume typed media records rather
than PyAV stream or FFmpeg process objects.

## Canonical timeline

The canonical timeline is source-media presentation time in seconds. It retains
stream start-time evidence instead of forcing every stream to appear to start at
zero.

For the existing zero-based video timestamp:

```text
sourceMediaTime = (videoStartTime or 0) + frameIndex / fps
```

For an extracted audio sample:

```text
audioSampleTimestamp = sampleIndex / outputSampleRate

sourceMediaTime = (audioStartTime or 0)
                + audioSampleTimestamp
                + audioVideoOffsetMs / 1000
```

`audioVideoOffsetMs` defaults to `0`. Positive values move audio evidence later;
negative values move it earlier. This setting is a correction used for analysis
and event fusion, not a rewrite of source timestamps or media bytes. Future event
fusion must pair the offset with a separately configurable tolerance.

When a stream start time is unavailable, mapping uses zero and the persisted `null`
start value continues to expose that uncertainty.

## Analysis WAV contract

`pickleball-vision extract-audio` selects the first source audio stream and writes
signed 16-bit PCM WAV. PCM avoids adding a lossy analysis codec. The original
recording remains unchanged.

By default, extraction preserves the source sample rate, channel count, and layout
as supported by the WAV/FFmpeg boundary. `--sample-rate` requests explicit
resampling. `--channels 1` or `--channels 2` requests explicit mono or stereo
conversion. The command writes `<output>.metadata.json` containing:

- source and output codec, rate, channel count/layout, duration, and sample count;
- requested conversion and whether rate/channel conversion occurred;
- source audio/video start times and `audioVideoOffsetMs`;
- the formula mapping output sample indices to canonical time; and
- media backend provenance.

The WAV timeline is rebased so sample zero corresponds to the source audio stream's
first presentation time. Internal timestamp discontinuities are hard-compensated by
inserting or dropping samples rather than silently concatenating decoded frames.
This synchronization normalization is recorded in the sidecar. The original source
timestamps and bytes are not changed.

Audio is supporting evidence only. A transient may raise or lower confidence in a
visually plausible bounce or paddle contact, but cannot create the event by itself,
override contradictory video, or be assumed to originate on the primary court.

Milestone 10 audio observations use the same formula without a second timing model.
Each feature window retains its analysis-relative time, while each transient uses its
peak sample to calculate canonical `mediaTimestampSeconds`. See
`audio-analysis.md` for fields and manual synchronization review.
