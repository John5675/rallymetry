# Vision Pipeline Contract

This document describes stable stage boundaries. A stage may be introduced only
when its matching milestone is current. Audio-aware media ingestion, manual court
calibration, broad person detection, primary-player isolation, and persistent
logical-player tracking are currently implemented; all later stages remain
contracts only.

## Intended flow

```text
source media metadata (video + optional audio)
  -> court calibration
  -> person and ball observations
  -> primary-player isolation
  -> persistent player tracks
  -> observed/interpolated ball track segments
  -> optional raw audio observations
  -> rally and multimodal bounce/contact/hitter/shot events
  -> structured match data
  -> analytics
```

## Video source contract

The pipeline accepts a local file path. How bytes arrived locally—camera transfer,
object-store download, or a separately performed download from an unlisted
source—is outside the vision service. Private source URLs and credentials never
enter pipeline output or source control.

Media metadata records the resolved path, pixel dimensions, OpenCV-reported FPS as
a floating-point value, frame count, duration in seconds, and codec FourCC when
available. It combines those decoded-video facts with FFmpeg-reported audio
presence, codec, rate, channels, duration, and available audio/video stream start
times. Audio is optional and no CV stage may require it.

Duration reported in the existing `duration` field remains `frame_count / fps`; it
is an OpenCV/container estimate, not a promise of constant-frame-rate presentation
timestamps. `audioDuration` uses the audio-stream duration when available and the
container duration as a metadata fallback.

Timestamp extraction maps a valid timestamp in `[0, duration)` to the containing
zero-based frame index using the reported FPS. Frames are decoded and written at
their source pixel dimensions. Uniform sampling selects unique indices over the
inclusive range from the first to the last frame; a one-frame sample uses the
middle frame.

## Canonical media timeline contract

The canonical timeline is source-media presentation time in seconds. Existing
video frame timestamps remain relative to the first decoded video frame and map as:

```text
mediaTime = (videoStartTime or 0) + videoFrameTimestamp
```

Extracted WAV sample timestamps are zero-based (`sampleIndex / sampleRate`). They
map back to source-media time as:

```text
mediaTime = (audioStartTime or 0)
          + sampleIndex / sampleRate
          + audioVideoOffsetMs / 1000
```

`audioVideoOffsetMs` defaults to zero and is a configured correction for downstream
fusion; stream start times remain separately visible evidence. A positive offset
makes an audio observation later on the canonical timeline. Every future fusion
stage must also use an explicit timing tolerance and provide a vision-only fallback.

Audio extraction decodes the selected source stream into PCM WAV, preserves source
sample rate and channels by default, and never rewrites the source recording.
Explicit rate/channel conversion is allowed and recorded. The adjacent
`.wav.metadata.json` sidecar stores source and analysis stream properties,
conversion flags, backend provenance, and the sample-to-source-time mapping. Raw
audio timestamp discontinuities are normalized in the WAV with inserted/dropped
samples and that operation is recorded. Raw audio and later raw audio observations
are evidence, not bounce/contact events.

## Court calibration contract

Manual calibration uses named court-plane landmarks defined in
`coordinate-system.md`. At least four geometrically valid image/court
correspondences are required. Additional selected landmarks participate in a
robust fit and retain their inlier/outlier status rather than being silently
discarded.

Fit selection evaluates all-point residuals before falling back to robust outlier
rejection. Quality reporting uses errors across every selected correspondence and
must warn when a tight inlier subset masks poor whole-court alignment.

The stored image-to-court homography applies only to points known to lie on the
court plane. Its inverse maps canonical court points into the calibrated image.
Calibration quality reports forward reprojection error in meters and reverse
reprojection error in pixels. These errors describe the selected plane landmarks;
they do not make homography valid for an airborne ball.

## Observation contract

An observation is evidence tied to a frame/time, model version, configuration,
confidence, and image-space geometry. It is not a pickleball event. Later stages
may link to or supersede an observation but must not silently mutate the raw result.

Person detections include bounding boxes. A player's initial physical-position
estimate is the bottom-center ground-contact point of the selected bounding box.
The bounding-box center is never a physical court position. Selecting primary
players requires temporal and court-aware evidence; taking the four highest person
confidences is not a valid algorithm.

The person-detection baseline stores every accepted COCO `person` observation in
the original source-frame pixel coordinate system. Each observation retains its
confidence, zero-based frame number, and timestamp. The calibration is validated
and recorded as run provenance, but person boxes are not projected through its
homography during this milestone. People on neighboring courts, spectators, and
officials are intentionally not removed; that is primary-player isolation work.

## Primary-player isolation contract

Isolation derives, but never adds fields to, raw person observations. The initial
ground-contact estimate is exactly the bottom-center of each person box. It is
projected through the court homography only when the point is not clipped at the
bottom frame edge. Derived records retain the image point, optional court point,
projection status, inside/near/outside state, boundary ambiguity, court side, and
confidence.

Candidate selection combines court support with short-gap ground-point proximity
and temporal persistence. These candidate IDs are ephemeral selection tracklets,
not persistent physical identities. They may bridge a brief missed detection but
cannot satisfy the persistent tracking contract in Milestone 5. Detection
confidence is retained as evidence and is never a top-four selection rule.

The four human-owned logical roles `ME`, `PARTNER`, `OPPONENT_1`, and
`OPPONENT_2` are stored in a separate manual-assignment artifact. Logical roles are
independent of raw detection indices and ephemeral candidate IDs, even though the
assignment record links to both for provenance. Manual correction supersedes a
prior assignment artifact; it does not rewrite detector observations.

## Persistent-player tracking contract

The tracker boundary consumes persisted person detections, not model tensors.
ByteTrack produces transient motion associations linked to immutable raw detection
indices. Its IDs are evidence only: they are neither logical player names nor match
statistics.

Logical identity resolution begins at the four manual assignment anchors and runs
both earlier and later in the clip. It accepts an observation only when court
membership, expected near/far side, elapsed time, movement distance, and one-to-one
use are defensible. Recording-local clothing appearance and its margin over the
same-side teammate provide additional identity evidence, with stricter thresholds
for reacquisition after long gaps. A player may be temporarily missing and later
reacquired. An immediate or weakly supported tracker-ID transition is stored as a
suspected switch for human review. Clearly outside-court people cannot acquire a
logical identity, even when their detector confidence is high.

`tracks.json` keeps the raw tracker and logical identity layers separate.
`tracking-summary.json` reports coverage, suspected switches, longest missing gaps,
and reacquisition counts per role. `annotated.mp4` is a review aid, not structured
truth.

## Player-position analytics contract

Release 0.1 consumes `tracks.json`, not raw detector tensors. For every player and
source frame, it retains the image-space bottom-center ground estimate, raw court
projection, optional separately recorded manual court-plane correction, separate
conservative smoothed court coordinate, tracking confidence, frame number, and
timestamp. Correction and smoothing never replace raw data or fill a missing frame.

Position metrics operate only on quality-gated structured position records. Each
metric retains calculation version, unit, population, coverage, and contributing
frame ranges. Heatmaps and animations are debug/presentation artifacts; computed JSON
remains the analytics source of truth. Exact formulas and court-region boundaries are
defined in `analytics-definitions.md`.

Ball records distinguish at minimum:

- `observed`: supported directly by image evidence;
- `interpolated`: estimated only between sufficiently close supported points; and
- `missing`: no defensible position.

Interpolation must retain its method, support interval, and confidence. Long gaps
remain missing.

## Geometry contract

Court homography maps points on the physical court plane. It is valid for court
landmarks, player ground-contact points, and a confirmed ball bounce point. An
airborne ball image coordinate cannot be projected as though it lies on the court.
See `coordinate-system.md` for axes and units.

## Event contract

Events are derived records supported by observations and tracks. Every uncertain
ML-derived event retains confidence and provenance. A bounce, paddle contact,
hitter assignment, and shot class are separate inferences; confidence in one does
not imply confidence in another.

## Inspectability

Every stage should eventually support:

- deterministic configuration snapshots;
- schema and producer versions;
- links to input artifacts or record identifiers;
- machine-readable confidence and uncertainty;
- optional debug overlays that do not replace structured output; and
- evaluation against annotations defined in `annotation-guide.md`.
