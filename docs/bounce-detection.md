# Multimodal Bounce Detection

Milestone 14 derives inspectable bounce candidates from the structured primary-ball
trajectory. Visual evidence is mandatory. Optional generic audio transients can
increase confidence but can never create a candidate or override contradictory
visual evidence.

## Run the stage

From `services/vision`:

```bash
uv run pickleball-vision detect-bounces /absolute/path/to/match.mp4 \
  --ball-tracks ../../output/ball-tracking/ball_tracks.json \
  --calibration ../../output/calibration/calibration.json \
  --rallies ../../output/rally-segmentation/rallies.json \
  --audio-events ../../output/audio-analysis/audio-events.json \
  --annotations ../../output/match-annotations.json \
  --output-dir ../../output/bounce-detection
```

Only `--ball-tracks`, `--calibration`, and `--output-dir` are required. Omit
`--rallies`, `--audio-events`, or `--annotations` when they are unavailable. Audio-
free recordings and runs without an audio artifact use the same vision-only path.

The command writes:

- `bounces.json`: all retained visual candidates, including low-confidence ones;
- `bounce-debug.mp4`: source-space markers and confidence/evidence labels; and
- `bounce-evaluation.json`: visual-only versus visual-plus-audio evaluation, or an
  explicit unavailable state when no human `BOUNCE` annotations were supplied.

Existing output artifacts are never overwritten. Use a new output directory for
each configuration experiment.

## Visual-first inference

For each known trajectory point, the detector fits local image-space velocity on
both sides of the point. A visual candidate requires all of the following:

- downward image motion before the point and upward image motion afterward;
- sufficient vertical-reversal strength;
- a local image-space vertical maximum with sufficient prominence;
- enough known points from one trajectory segment on both sides; and
- a minimum combined visual confidence.

Confidence separately records reversal, local shape, vertical speed, continuity,
observed-versus-interpolated support, projected-court image inclusion, and optional
rally-sequence support. Nearby frame candidates are suppressed conservatively so a
single reversal does not become several bounces.

This is an image-space heuristic, not 3D reconstruction. Perspective means image
velocity is not physical ball velocity.

## Court-plane projection gate

Known canonical court corners are projected into the image to test whether the
candidate appears over the court surface. The ball point is not passed through the
image-to-court homography during candidate generation.

Only after the visual candidate reaches the configured plane-contact confidence and
lies within the projected court polygon may its image point be transformed into a
canonical court coordinate. The JSON records whether this happened and why. A null
`courtPosition` means the projection was not defensible; it is not replaced with a
guess. The stage performs no line call and does not infer a true 3D position.

## Audio fusion and timing

The stage reloads each raw audio candidate's analysis-relative timestamp, then maps
it to video-relative time using source stream start times and the current configured
offset:

```text
audioVideoTime = (audioStartTime or 0)
               + analysisTimestamp
               + audioVideoOffsetMs / 1000
               - (videoStartTime or 0)
```

This intentionally reapplies the run's `audioVideoOffsetMs` instead of assuming the
offset stored by an earlier audio-analysis run is still correct. At most one raw
transient is matched to a visual candidate within `fusionToleranceMs`, and matching
is one-to-one. Timing proximity discounts its raw confidence.

Audio fusion can raise `fusedConfidence` on an existing visual candidate. It cannot
create a new candidate. Neighboring courts, speech, footsteps, and other noise remain
possible sources of every matched transient.

Evidence modes are:

- `VISUAL_ONLY`: accepted without a matched transient;
- `VISUAL_PLUS_AUDIO`: accepted with synchronized audio support; and
- `LOW_CONFIDENCE`: a retained visual candidate below the fused acceptance threshold.

## Evaluation

Human `BOUNCE` annotations are loaded only after inference. The evaluator thresholds
the identical visual candidate set twice:

1. `visualConfidence` only; and
2. `fusedConfidence` after optional audio support.

Each mode uses one-to-one matching within the configured timing tolerance and
reports precision, recall, F1, false/missed bounces, and signed/absolute timing
error. The comparison records metric deltas and explicitly declares that audio did
not create the visual candidate set.

Normal annotation files may cover only representative rallies. By default,
evaluation is limited to annotated rally intervals (or small windows around bounce
annotations when no rally pairs exist). Use `--annotations-complete` only after the
entire recording has been reviewed and all bounces have been annotated. Use
`--evaluation-partition development|validation|test` for provenance; the command
never tunes thresholds automatically.

## Configuration

Settings use the `PICKLEBALL_VISION_` prefix:

- `BOUNCE_TRAJECTORY_WINDOW_SECONDS`
- `BOUNCE_MINIMUM_OBSERVATIONS_EACH_SIDE`
- `BOUNCE_MINIMUM_VERTICAL_SPEED_DIAGONALS_PER_SECOND`
- `BOUNCE_MINIMUM_VERTICAL_REVERSAL_DIAGONALS_PER_SECOND`
- `BOUNCE_MINIMUM_SHAPE_PROMINENCE_DIAGONAL_FRACTION`
- `BOUNCE_MINIMUM_CONTINUITY_FRACTION`
- `BOUNCE_MINIMUM_VISUAL_CANDIDATE_CONFIDENCE`
- `BOUNCE_ACCEPTED_CONFIDENCE`
- `BOUNCE_PLANE_PROJECTION_MINIMUM_VISUAL_CONFIDENCE`
- `BOUNCE_MINIMUM_BETWEEN_SECONDS`
- `BOUNCE_AUDIO_CONFIDENCE_WEIGHT`
- `FUSION_TOLERANCE_MS`
- `BOUNCE_RALLY_SEQUENCE_CONFIDENCE_BOOST`
- `BOUNCE_EVALUATION_TOLERANCE_MS`
- `BOUNCE_SPARSE_EVALUATION_MARGIN_SECONDS`
- `AUDIO_VIDEO_OFFSET_MS`

Every effective value is persisted with the output. Tune only on a development
partition, freeze values, and compare unchanged validation/test results.
