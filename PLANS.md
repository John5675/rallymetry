# Pickleball Vision Milestones

Milestones are intentionally sequential. Exactly one milestone is current. Work
outside the current milestone requires this plan to be deliberately updated first.

## 0. Foundation — complete

Establish repository instructions, architecture documentation, a typed Python
package, an installable CLI shell, environment-based configuration, structured
logging, error conventions, and automated quality checks.

Exit criteria:

- The documented repository structure exists.
- The vision package installs in a clean environment.
- `pickleball-vision doctor` reports a valid Foundation setup.
- Tests, Ruff checks, Ruff formatting, and static type checks pass.
- No video ingestion or computer-vision behavior was implemented.

## 1. Video ingestion — complete

Read local video files through a reusable OpenCV boundary, expose source metadata,
extract a full-resolution frame by timestamp, and sample frames across the full
source duration. All commands validate their inputs and translate expected OpenCV
or filesystem failures into useful application errors.

Exit criteria:

- `pickleball-vision inspect <video>` reports the resolved source path, dimensions,
  non-integer FPS, frame count, duration in seconds, and codec when available.
- `pickleball-vision extract-frame` writes the requested valid frame without
  resizing it.
- `pickleball-vision sample-frames` selects the requested number of unique frames
  across the complete frame-index span rather than only from its beginning.
- Missing, non-file, unreadable, corrupt, invalid-timestamp, invalid-count, decode,
  and image-write failures produce stable, useful errors rather than raw OpenCV
  exceptions.
- Automated tests generate their own synthetic videos; no private footage is
  required or committed.
- Tests, Ruff checks, Ruff formatting, static type checks, and the CLI health check
  pass.
- No court calibration, detection, tracking, or product-service behavior was
  implemented by this milestone.

## 2. Court calibration — complete

Manually associate at least four visible image points with named canonical
pickleball court landmarks, fit an image/court-plane homography using all valid
correspondences, persist the calibration and its provenance, and create inspectable
image-space and top-down debug artifacts.

Exit criteria:

- Court dimensions, axes, units, named landmarks, and line geometry are explicit
  and documented.
- `pickleball-vision calibrate <video> --timestamp <seconds> --output <json>`
  guides the user through visible landmark selection without requiring all outer
  corners.
- Four valid correspondences are required; additional correspondences support
  robust fitting and retain their inlier status.
- Degenerate, duplicated, out-of-frame, non-finite, unreadable, cancelled, and
  non-invertible calibration configurations produce useful typed errors.
- Calibration JSON retains source-frame provenance, correspondences, both
  homography directions, court configuration, and reprojection error in court
  meters and image pixels.
- Reusable APIs transform a court-plane image point to canonical court coordinates
  and perform the reverse transformation.
- `calibration-overlay.jpg` labels selected landmarks and projected court lines;
  `court-topdown.jpg` rectifies the selected frame and draws canonical geometry.
- Unit tests verify transformations, robust multi-point fitting, degeneracy,
  persistence, and debug rendering with synthetic data.
- No automatic court detection or later vision milestone is implemented.

## 3. Person detection — complete

Run a pretrained, model-adapted person detector over every source frame and retain
broad image-space observations for every visible person. Neighboring-court people
remain valid detections at this stage; participant selection is a later milestone.

Exit criteria:

- `pickleball-vision detect-people <video> --calibration <json> --output-dir <dir>`
  validates its local inputs and processes the source video without resizing the
  output coordinate system.
- A detector protocol keeps model-independent observation records and pipeline
  logic separate from the Ultralytics adapter.
- CPU inference works, while automatic or explicit CUDA/MPS acceleration remains
  optional and externally configured.
- Every detection retains a source-pixel bounding box, confidence, zero-based
  frame number, and timestamp; no player identity or court position is inferred.
- Minimum confidence and inference settings are validated, externalized, and
  included in output provenance.
- `detections.json`, `annotated.mp4`, and `summary.json` are inspectable artifacts;
  summary statistics describe detections, not match participants or match events.
- Tests cover serialization, configuration filtering, adapter translation, and a
  synthetic-video pipeline without downloading weights or using private footage.
- Tests, Ruff checks, Ruff formatting, static type checks, and the CLI health check
  pass.
- No primary-player isolation, tracking, player-position estimation, ball
  detection, or product-service behavior is implemented.

## 4. Primary-player isolation — complete

Derive court-aware person candidates from raw detections without mutating detector
output or assuming that confidence identifies match participants. Use estimated
shoe/court contact, calibrated geometry, limited temporal persistence, and manual
four-role initialization to isolate the primary doubles court.

Exit criteria:

- Every raw person detection receives a derived bottom-center ground-contact point;
  the person-box center is never used as a physical position.
- Geometrically appropriate ground points are projected to canonical court meters
  and classified as inside, near, or clearly outside the primary court, with
  ambiguity and classification confidence retained.
- Candidate selection uses court membership, near/far side, and short-gap temporal
  persistence; detection confidence never becomes a top-four selection rule.
- Lightweight candidate association survives isolated missed detections but is
  explicitly not the persistent identity tracker planned for Milestone 5.
- A local manual workflow assigns exactly four independent logical roles: `ME`,
  `PARTNER`, `OPPONENT_1`, and `OPPONENT_2`.
- Manual assignments are persisted separately from raw observations and ephemeral
  candidate identifiers, and an existing assignment file can be corrected.
- Debug output shows every raw person subtly, eligible primary-court candidates
  distinctly, bottom-center ground points, court state, and assigned logical role.
- Synthetic tests cover inside/near/outside/ambiguous court classification,
  near/far side classification, missed-frame persistence, and independent manual
  role serialization.
- Tests, Ruff checks, Ruff formatting, static type checks, and the CLI health check
  pass.
- No full persistent tracking, player movement analytics, ball detection, or later
  product behavior is implemented.

## 4A. Audio-aware media foundation retrofit — complete

Extend local media ingestion with optional synchronized-audio metadata, a canonical
source-media timeline, and lossless analysis-audio extraction without coupling raw
audio to computer-vision observations or semantic pickleball events.

Exit criteria:

- Existing OpenCV video ingestion and all completed CV milestones remain compatible.
- `pickleball-vision inspect` retains its existing video fields and adds audio
  presence, codec, sample rate, channels, duration, and available stream start times.
- FFmpeg-specific probing and extraction details remain behind a reusable media
  boundary rather than leaking into CV or domain-event code.
- `pickleball-vision extract-audio` writes synchronized PCM WAV without changing the
  source; source rate and channels are preserved unless conversion is explicit.
- A durable sidecar records source/output audio properties, conversion details,
  source-time mapping, and the configured `audioVideoOffsetMs`.
- Audio-free video remains a supported vision input and produces a useful typed
  error only when audio extraction is explicitly requested.
- Synthetic tests cover A/V media, video-only media, invalid media, stream parsing,
  extraction, duration consistency, timestamp conversion, and non-zero A/V offset.
- Tests, Ruff checks, Ruff formatting, static type checks, and the CLI health check
  pass.
- No audio-event detection, bounce/contact fusion, or later milestone is implemented.

## 5. Persistent player tracking — complete

Associate raw person detections with an established multi-object tracker, then
resolve the four manually initialized logical player identities independently of
transient tracker identifiers. Use court geometry, court side, temporal continuity,
plausible movement, recording-local appearance evidence, and conservative occlusion
handling to prevent unrelated neighboring-court people from stealing an identity.

Exit criteria:

- `pickleball-vision track-players <video> --calibration <json> --output-dir <dir>`
  consumes the existing raw detections and four-role manual assignments.
- An established tracker is isolated behind a project-owned adapter, and its raw
  observations remain separately inspectable from logical identity resolution.
- `ME`, `PARTNER`, `OPPONENT_1`, and `OPPONENT_2` remain independent of tracker IDs;
  tracker-ID changes can be conservatively reacquired without renaming a player.
- Court membership, expected near/far side, plausible motion, and one-to-one
  assignment guard against identity theft by adjacent-court detections.
- Same-side appearance comparison supports tracker-ID changes, and long-gap
  reacquisition requires stronger appearance evidence than ordinary continuity.
- Every frame retains an observed, reacquired, temporarily missing, or suspected
  switch state with identity confidence; one missed detection never permanently
  removes a logical player.
- Suspected identity switches are surfaced as explicit review events rather than
  silently accepted.
- `tracks.json`, `annotated.mp4`, and `tracking-summary.json` retain provenance and
  report coverage, suspected switches, longest missing intervals, and reacquisition
  counts for all four roles.
- Synthetic tests exercise tracker adaptation, tracker-ID changes, occlusion,
  court/side filtering, neighboring-player rejection, serialization, and summary
  metrics without private footage or downloaded model weights.
- Tests, Ruff checks, Ruff formatting, static type checks, and the CLI health check
  pass.
- No player movement analytics, ball tracking, or audio-event analysis is
  implemented.

## 6. Player court-position analytics — complete (Release 0.1)

Derive inspectable player ground trajectories and aggregate movement/position
analytics from the four logical tracking records. Preserve raw image ground points
and raw homography coordinates, keep recording-local manual position corrections in
an explicit intermediate layer, add conservative bounded smoothing as a separate
layer, and qualify every metric by its contributing coverage.

Exit criteria:

- `pickleball-vision analyze-players <video> --calibration <json> --output-dir <dir>`
  consumes the persisted logical player tracks rather than detector tensors.
- Every logical player/frame retains frame number, timestamp, raw image-space
  bottom-center ground point, raw court coordinate, separate smoothed court
  coordinate, and confidence; missing positions remain explicit.
- Optional bounded per-role court-plane corrections are persisted separately from
  raw evidence and recorded in analysis provenance.
- Smoothing never overwrites raw observations, never bridges long gaps, and records
  its method and parameters.
- Approximate distance traveled, kitchen/transition/backcourt occupancy, average
  distance from the kitchen, average partner spacing, and lateral-movement metrics
  have precise versioned definitions and data-quality coverage.
- `player_positions.json`, `summary.json`, four per-role heatmaps, a source-space
  `annotated.mp4`, and a `topdown.mp4` are generated with input/configuration
  provenance.
- Synthetic tests cover raw preservation, ground-point use, smoothing boundaries,
  region metrics, spacing, distance, lateral movement, serialization, and rendering
  without private footage.
- Tests, Ruff checks, Ruff formatting, static type checks, and the CLI health check
  pass.
- Release 0.1 stops at player position analytics. No ball tracking, audio-event
  analysis, rally inference, or product-service behavior is implemented.

## 7. Ball dataset tooling — complete

Create inspectable, model-free tooling for extracting ball-annotation frames and
clips from local source videos, preserving source/frame provenance, organizing
positive/negative/unlabeled examples, and assigning leakage-safe dataset splits.

Exit criteria:

- `pickleball-vision dataset extract-frames <video> --output-dir <dir> --every
  <frames>` extracts full-resolution frames at a deterministic cadence.
- Random sampling is reproducible from an explicit seed and time-range bounds are
  validated against the source duration.
- Named clip ranges can scope frame extraction and produce source-preserving clip
  artifacts without modifying the original recording.
- A versioned dataset manifest records source metadata, content identity, selection
  method, frame indices, timestamps, clip/rally groups, label group, and relative
  artifact paths.
- Extracted examples are organized as `positive`, `negative`, or `unlabeled` without
  claiming that a bright object is a pickleball.
- Split tooling assigns whole videos, clips, or rally/groups to train, validation,
  and test; frames from the same selected unit can never cross splits.
- Annotation guidance defines visible, partial, blurred, ambiguous,
  neighboring-court, fully occluded, and multiple-ball behavior.
- Synthetic tests cover cadence/random/range/clip extraction, metadata and grouping,
  validation failures, deterministic split assignment, and neighboring-frame
  leakage prevention.
- Tests, Ruff checks, Ruff formatting, static type checks, and the CLI health check
  pass.
- No ball detector training, inference, trajectory reconstruction, or pickleball
  event inference is implemented.

## 8. Ball detector — current

Train and evaluate a custom single-class `pickleball` detector from human-reviewed,
versioned datasets. Preserve raw frame-local observations and compare spatial
inference strategies suitable for tiny near- and far-side balls without introducing
temporal tracking.

Exit criteria:

- A versioned external JSON configuration fixes dataset/model versions, training
  parameters, random seed, validation/test assignments, and named inference
  strategies.
- Training refuses unreviewed or ambiguous ground truth, materializes a reproducible
  single-class dataset, isolates Ultralytics-specific behavior behind an adapter,
  and writes experiment metadata, hashes, model provenance, and metrics beside each
  run.
- Video inference supports configurable high-resolution full-frame, calibrated
  primary-court ROI, tiled, and court-ROI tiled strategies while retaining original
  source-pixel coordinates and all per-crop proposals.
- Court calibration may define an image crop only; airborne detections are never
  projected through court homography.
- Raw detections retain frame number, timestamp, confidence, model/weights version,
  spatial strategy, and proposal provenance; no track, interpolation, bounce,
  contact, or other event is inferred.
- Fixed validation/test evaluation persists precision, recall, false positives,
  false positives per evaluated minute, positive-frame detection coverage, and
  human-annotated near/far recall and coverage.
- Strategy comparison evaluates identical model weights and identical fixed frame
  record IDs, then writes individual metrics plus a machine-readable comparison.
- A loopback-only manual review interface overlays raw model suggestions without
  promoting them automatically, supports drawing/removing boxes plus per-ball
  near/far, scope, visibility, and confidence labels, and atomically persists
  resumable reviewed positives, reviewed negatives, and drafts.
- Synthetic tests cover configuration, annotation validation, dataset preparation,
  model adaptation, crop/tile coordinate restoration, cross-tile deduplication,
  evaluation metrics, experiment persistence, and video raw-detection artifacts.
- Tests, Ruff checks, Ruff formatting, static type checks, and the CLI health check
  pass.
- No ball trajectory reconstruction, audio-event extraction, or semantic event
  inference is implemented.

## Future milestones — not current

9. Ball trajectory reconstruction
10. Audio event extraction
11. Manual event annotation
12. Rally segmentation
13. Multimodal bounce detection
14. Multimodal contact detection
15. Hitter identification
16. Shot reconstruction/classification
17. Analytics
18. Backend
19. Async processing
20. Web dashboard
21. Human corrections
22. AI coach
23. Final audit

Moving a milestone to current should add measurable entry and exit criteria while
preserving completed milestone history. Do not bundle later milestones into the
current one for convenience.
