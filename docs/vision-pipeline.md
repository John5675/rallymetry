# Vision Pipeline Contract

This document describes stable stage boundaries. A stage may be introduced only
when its matching milestone is current. Audio-aware media ingestion, manual court
calibration, broad person detection, primary-player isolation, and persistent
logical-player tracking are implemented. Ball dataset extraction, custom detector
training, raw spatial ball inference, fixed-split evaluation, and conservative
image-space ball trajectory reconstruction are also implemented. Synchronized raw
audio features, generic transient candidates, versioned human multimodal event
ground truth, and structured automatic rally segmentation are implemented. Automatic
visual-first multimodal bounce detection and paddle-contact detection are also
implemented. Conservative logical hitter identification and interpretable initial
shot reconstruction/classification are implemented.
Deterministic match analytics over structured rallies, shots, and player positions
are implemented as the final local analysis stage.

## Intended flow

```text
source media metadata (video + optional audio)
  +-> ball annotation frame/clip datasets
      -> fixed split + reviewed boxes
      -> custom ball training/evaluation
      -> raw frame-local ball observations
  +-> court calibration
      -> person and ball observations
      -> primary-player isolation
      -> persistent player tracks
      -> observed/interpolated ball track segments
      -> optional raw audio feature observations
      -> generic audio transient candidates
      + human multimodal event ground truth
      -> automatic rally intervals
      -> visual-first bounce candidates + optional audio confidence support
      -> visual-first paddle-contact candidates + optional audio confidence support
      -> logical hitter assignment or UNKNOWN
      -> rally-local reconstructed and rule-classified shots
      -> structured match data
      -> deterministic match-analytics.json
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
fusion; `fusionToleranceMs` defaults to 90 milliseconds and is the shared maximum
matching distance for downstream A/V evidence. Stream start times remain separately
visible evidence. A positive offset makes an audio observation later on the
canonical timeline. Every fusion stage must preserve its effective tolerance and
provide a vision-only fallback.

Audio extraction decodes the selected source stream into PCM WAV, preserves source
sample rate and channels by default, and never rewrites the source recording.
Explicit rate/channel conversion is allowed and recorded. The adjacent
`.wav.metadata.json` sidecar stores source and analysis stream properties,
conversion flags, backend provenance, and the sample-to-source-time mapping. Raw
audio timestamp discontinuities are normalized in the WAV with inserted/dropped
samples and that operation is recorded. Raw audio and later raw audio observations
are evidence, not bounce/contact events.

## Audio observation contract

Audio analysis consumes the synchronized PCM representation and emits two distinct
layers. Raw feature observations describe overlapping signal windows with sample and
media timestamps, peak amplitude, RMS energy, spectral-flux onset strength, compact
frequency evidence, and per-channel values. Generic `TRANSIENT` candidates reference
supporting observations and retain confidence, channel triggers, and threshold
provenance.

Candidates have no paddle/bounce classification and no primary-match association.
Neighboring-court sounds, speech, wind, footsteps, and environmental noise remain
possible explanations. Later fusion may adjust confidence only when compatible
visual evidence exists and must never let audio override contradictory video.

When audio is absent, audio analysis emits explicit unavailable JSON/visual artifacts
and no observations. This is a successful vision-only state, not a pipeline failure.
Exact fields, configuration, timing, and manual review are documented in
`audio-analysis.md`.

## Multimodal ground-truth contract

The local match editor creates a separate, versioned human annotation artifact. It
supports explicit rally boundaries, serve and paddle contacts, bounces, rally
winners, and shot-type labels with exact frame/time provenance and optional
metadata. It never mutates source video, raw audio features, transient candidates,
detector outputs, or trajectories.

Optional Prompt 10 waveform and generic transient artifacts are display-only
context. They are kept separate from match events and are never promoted into
semantic annotations. Optional human audio labels describe evidence around a
visually reviewed event; audio remains unnecessary for ordinary annotation. The
schema, semantics, controls, and audit workflow are defined in
`match-annotation.md`.

## Automatic rally-segmentation contract

Rally segmentation consumes the immutable, frame-complete primary-match ball
trajectory. It derives image-space motion in normalized source-frame units and
combines sustained activity, bounded gaps, long quiet periods, serve-like motion
onsets, and time between bursts. A configurable adjacent-burst arbitration step
retains but rejects materially weaker intervals close to stronger rally evidence,
which reduces dead-ball return and handoff false positives. Optional
source-compatible logical-player tracks
provide reset-like low-motion confidence evidence. Incompatible player artifacts
are rejected rather than silently sampled or rescaled.

Optional generic audio transients may increase confidence near an already visual
boundary. Audio never proposes a start or end, and the same inference behavior is
available when audio is absent. A serve-like sequence is an onset heuristic, not a
serve-contact event. The stage produces no bounce, contact, hitter, or shot record.
Likewise, an adjacent rejected interval is only a possible dead-ball handoff; no
semantic event is asserted.

Human `RALLY_START`/`RALLY_END` annotations are loaded only after inference for
one-to-one interval evaluation. Sparse annotation coverage excludes unreviewed time
instead of treating it as negative; complete-video negative coverage requires an
explicit command flag. Evaluation reports interval precision/recall, matched,
missed, and false rallies, plus start/end timing error. The implementation never
tunes thresholds automatically and records whether a run is development,
validation, or test. Exact fields, settings, and commands are documented in
`rally-segmentation.md`.

## Multimodal bounce-detection contract

Bounce detection consumes the immutable, frame-complete ball trajectory and manual
calibration. Image-space direction reversal, vertical motion, local shape, and
same-segment continuity are required to create a visual candidate. Optional rally
intervals provide confidence-only sequence context and cannot create a candidate.

Generic audio transients are remapped from analysis time using the source stream
starts, the configured `audioVideoOffsetMs`, and an explicit `fusionToleranceMs`.
Audio can increase confidence only on an existing visual candidate; one-to-one
matching retains the raw transient ID and never assigns it an independent bounce
meaning. Vision-only operation remains fully supported.

Known court geometry may be projected into the image during visual assessment. A
ball image point is transformed into court coordinates only after the candidate has
already met the visual plane-contact threshold and lies inside the projected court
polygon. Null court positions remain null. The stage never projects an airborne
ball, infers true 3D position, or performs a line call.

Human `BOUNCE` events are isolated to post-inference evaluation. Visual-only and
fused evaluation threshold the same visual candidates and report precision, recall,
F1, and timing error. Exact commands, fields, thresholds, and sparse-review rules
are defined in `bounce-detection.md`.

## Multimodal paddle-contact detection contract

Paddle-contact detection consumes the immutable, frame-complete ball trajectory and
source-compatible logical-player tracks. Abrupt image-space velocity change,
direction or speed discontinuity, and same-segment before/after continuity are
required to create a visual candidate. Player proximity uses distance to the linked
person box while the separately retained physical player position remains the
bottom-center ground-contact estimate.

Optional rally intervals and accepted bounce state provide confidence or exclusion
context but cannot create a contact. Optional generic audio transients are remapped
with the configured A/V offset and tolerance and can increase confidence only on an
existing visual candidate. Audio-free operation remains fully supported.

Candidate-player rankings retain all available logical roles, tracking uncertainty,
court-side context, and proximity evidence. They do not assign a hitter. The stage
does not project an airborne ball through homography or infer a true 3D position.
Human `SERVE_CONTACT` and `PADDLE_CONTACT` events remain post-inference evaluation
inputs. Exact commands and thresholds are defined in `contact-detection.md`.

## Hitter-identification contract

Hitter identification is a separate derived layer over immutable contact candidates
and the exact logical-player track artifact recorded by them. It combines player-box
proximity, bottom-center player ground evidence, tracking confidence/state, observed
player court side, image-space trajectory direction, visual contact confidence,
prior credible hitter, and rally ordering. The airborne ball is never projected
through homography.

Visual contact confidence—not audio-fused confidence—is used for identity scoring
and gating. Audio identity contribution is always zero. Minimum confidence,
distance, tracking, score, and margin gates preserve `UNKNOWN` whenever evidence is
insufficient. An uncertain eligible contact resets sequence context to avoid error
propagation. Human player labels are loaded only after inference for time-matched
overall and observed near/far accuracy. Exact fields and thresholds are defined in
`hitter-identification.md`.

## Shot reconstruction and classification contract

Only accepted contacts inside non-overlapping automatic rallies create shots. Each
shot references its immutable ball-trajectory frame range, accepted contact and
logical hitter, optional first outgoing accepted bounce, and raw bottom-center
hitter ground position. Landing court coordinates are copied only from a bounce that
already passed visual plane-contact projection gates; airborne trajectory points are
never projected through homography.

The initial ordered classifier supports only `SERVE`, `RETURN`, `DINK`, `DROP`,
`DRIVE`, `VOLLEY`, `OVERHEAD`, `OTHER`, and `UNKNOWN`. It uses explicit domain
features and stores every tested rule and threshold; no new neural network is used.
Missing or weak hitter, player-position, or trajectory evidence produces `UNKNOWN`.
Human shot labels are isolated to post-inference evaluation. Exact reconstruction,
rules, configuration, and metrics are defined in `shot-reconstruction.md`.

Milestone 24A adds a separate temporal-model foundation described in
`shot-model.md`. It treats rally phase, contact mechanics, stroke side, and tactical
intent as independent axes; retains the legacy field as a conservative projection;
and never turns an external racket-sport representation dataset into pickleball
semantic ground truth. A `bestGuess` can be populated below threshold, but analytics
continues to use the authoritative value, which remains `UNKNOWN` when evidence is
insufficient.

## Deterministic match analytics contract

`analyze-match` consumes source-compatible `rallies.json`, `shots.json`, and
`player_positions.json`. It validates source metadata, rally content-hash provenance,
and the shared persistent-player-track lineage before calculating anything. Its
domain boundary includes structured Rally, Shot, Contact/Bounce references carried
by a Shot, and PlayerPosition records; it has no dependency on detector adapters,
raw tensors, YOLO records, audio feature windows, or waveforms.

All output is deterministic for identical input contents and configuration. Unknown
hitters/classes remain explicit, zero-denominator rates are `null`, input confidence
is reported as data quality rather than used as fractional counts, and every input
path/hash is retained without mutation. Match totals, per-player shot selection,
quality-gated position metrics, third-shot selection, regional shot selection, and
coverage-gated team kitchen arrival are defined precisely in
`analytics-definitions.md`. `match-analytics.json` completes the local analysis
pipeline and does not introduce hosted persistence or product services.

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

## Ball dataset contract

Dataset extraction preserves source resolution and records the content SHA-256,
source stream metadata, frame index, timestamp/FPS time base, selection method,
label group, and optional clip/rally group. Cadence sampling is deterministic;
random sampling is unique and seed-reproducible. Named ranges are half-open
`[start_time_s, end_time_s)` intervals and may optionally produce synchronized,
lossless MKV review clips without modifying the source recording.

The three frame directories are human curation queues. `positive` means a reviewer
confirmed a visible pickleball, `negative` means a reviewer confirmed none is
visible, and `unlabeled` includes untouched and ambiguous frames. A neighboring-court
ball is still a positive pickleball observation with separate scope metadata.

Split assignment operates on whole source videos, named clips, or rally/group IDs.
It writes references only and never independently randomizes neighboring frames.
This dataset stage contains no detector, model weights, predictions, trajectories, or
events.

## Ball detector contract

The custom detector has exactly one class, `pickleball`. Training consumes fixed
train/validation/test record IDs and explicit human-reviewed annotation records. An
unreviewed frame is never a negative, ambiguous objects are excluded from detector
ground truth, and dataset/model versions plus source/configuration hashes are retained
with every experiment and metric artifact.

Ultralytics-specific loading, training, and result translation remain behind project
adapters. Configured high-resolution inference is independent of person-detector
resolution. Supported spatial modes are full frame, padded primary-court image ROI,
overlapping full-frame tiles, and overlapping court-ROI tiles. Crop-local predictions
are translated back to original source pixels; all proposals are retained before
cross-crop NMS.

Court homography is used only in reverse to project canonical court corners for an
image crop. It is never applied to an airborne ball position. Raw ball detections
retain model/weights identity, configuration, confidence, frame/time, source-pixel
box, crop provenance, and explicit absence of tracking/interpolation/events.

Evaluation performs one-to-one IoU matching on an immutable validation or test
partition and persists precision, recall, false positives, false positives per union
clip minute, positive-frame detection coverage, and near/far recall/coverage based on
human side labels. Strategy comparison verifies identical ordered frame record IDs.

Manual detector-data review uses the fixed split as its queue. The local loopback UI
may display raw detections as suggestions, but it never mutates that raw artifact or
automatically copies a suggestion into ground truth. Human-reviewed positive and
negative frames, draft annotations, per-ball context, and review progress are saved
atomically in the annotation boundary. No temporal ball path or event is inferred.

## Ball trajectory reconstruction contract

Trajectory reconstruction consumes the immutable `raw_pickleball_detections`
artifact and emits a separate, frame-complete `primary_match_ball_trajectory`.
Candidate association combines predicted image location, velocity, acceleration
plausibility, temporal support, detector confidence, and primary-court relevance;
confidence alone never selects the ball. Rejected candidates remain referenced by
detection ID and the detector artifact is not mutated.

Primary-court relevance uses only a projected image outline of known court geometry
with conservative side and airborne margins. This is an image-space association
heuristic, not a ball court coordinate. No observed, interpolated, or smoothed
airborne point is transformed through homography.

Every source frame is `OBSERVED`, `INTERPOLATED`, or `UNKNOWN`. Observed records
retain the raw box-center point and source detection. Interpolation has no raw point
and is limited to configured short gaps inside a single persistent segment. Bounded
smoothing is a separate value and cannot cross unknown periods. Longer or ambiguous
gaps remain unknown. `ball-debug.mp4` visualizes these distinctions, while
`ball_tracks.json` remains the structured source of truth. Exact fields, metrics,
configuration, and review guidance are defined in `ball-tracking.md`.

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

Ball records distinguish:

- `OBSERVED`: supported directly by one linked raw detection;
- `INTERPOLATED`: estimated only between sufficiently close supported points; and
- `UNKNOWN`: no defensible position.

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
