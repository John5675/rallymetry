# Automatic Rally Segmentation

Milestone 13 derives rally intervals from structured, inspectable signals. It does
not use an end-to-end event model and does not infer bounces, paddle contacts,
hitters, or shot classes.

## Run the stage

From `services/vision`:

```bash
uv run pickleball-vision segment-rallies /absolute/path/to/match.mp4 \
  --ball-tracks ../../output/ball-tracking/ball_tracks.json \
  --audio-events ../../output/audio-analysis/audio-events.json \
  --annotations ../../output/match-annotations.json \
  --output-dir ../../output/rally-segmentation
```

`--ball-tracks` is required. `--player-tracks`, `--audio-events`, and
`--annotations` are optional. Player tracks must describe the exact same source
path, dimensions, FPS, and frame count; incompatible tracks fail explicitly rather
than influencing a different recording. Audio-free and player-track-free runs use
the same vision-only inference path.

The command writes:

- `rallies.json`: predicted structured intervals and their evidence;
- `rally-debug.mp4`: source video with predicted intervals, boundaries, ball state,
  motion evidence, and optional human intervals; and
- `rally-evaluation.json`: one-to-one evaluation or an explicit unavailable result
  when annotations were not supplied.

An existing output artifact is never overwritten. Choose a new output directory or
remove an obsolete generated run deliberately.

## Inference signals

The segmenter consumes the frame-complete `OBSERVED` / `INTERPOLATED` / `UNKNOWN`
primary-ball timeline. It derives image-speed in source-frame diagonal units and
uses:

- sustained ball motion and trajectory coverage;
- bounded gaps inside an activity burst;
- a serve-like motion onset, confirmed by later motion and image displacement;
- a long quiet period after activity;
- time between bursts;
- adjacent-burst arbitration: when two activity intervals are separated by no more
  than 2.25 seconds, a materially weaker interval is retained as a rejected
  candidate instead of being reported as a rally. This targets dead-ball returns
  and ball handoffs immediately before or after a real rally;
- optional reset-like low movement among at least three compatible logical player
  tracks; and
- optional generic audio transients near an already visual boundary.

The serve-like signal is not a `SERVE_CONTACT` event. It is only an onset pattern
used to propose a rally boundary. Likewise, a trajectory point is never projected
through court homography and no inferred bounce/contact record is emitted.

Player and audio evidence can increase heuristic confidence but cannot create a
boundary. In particular, an audio transient alone always produces zero rallies.
The JSON records this rule for every run and every rally.

The adjacent-burst rule does not claim that a rejected interval is semantically a
dead-ball handoff. It compares motion fraction, trajectory coverage, sustained
motion, and duration, records the evidence score and margin, and emits the original
interval under `rejectedCandidates`. Close intervals with similar evidence are
both preserved as rallies rather than resolved arbitrarily.

## Prediction contract

Each prediction contains:

```json
{
  "rallyId": "predicted-rally-00001",
  "startTimestamp": 42.1,
  "endTimestamp": 48.7,
  "startFrame": 1263,
  "endFrame": 1461,
  "confidence": 0.84,
  "supportingSignals": {
    "ballTrajectoryActivity": {},
    "sustainedBallMotion": {},
    "serveLikeSequence": {},
    "trajectoryGapBoundaries": {},
    "timeBetweenActivityBurstsSeconds": 8.2,
    "playerResetBehavior": {},
    "audioSupport": {}
  }
}
```

Timestamps are video-relative seconds (`frame / fps`) and frames are zero-based.
Confidence is an inspectable, uncalibrated heuristic composition; it is not a
probability. The input hashes, source metadata, complete threshold snapshot, and
confidence components are persisted.

## Evaluation

Human `RALLY_START` and `RALLY_END` events are paired chronologically and loaded
only after predictions have been produced. They are never visible to inference.
Matching is one-to-one and accepts sufficient interval overlap or both boundaries
within the configured tolerance. Evaluation reports precision, recall,
matched/missed/false rallies, and signed/absolute start and end timing error.

Normal annotation files may contain only selected rallies. By default, evaluation
uses reviewed rally windows plus the configured margin; predictions elsewhere are
reported as ignored rather than false. Use `--annotations-complete` only after a
human has reviewed the entire video and confirmed that all rallies were annotated.

Use `--evaluation-partition development|validation|test` to record the split role.
The command never searches or tunes thresholds. Threshold changes must be made on a
development set, frozen, and then evaluated unchanged on validation/test clips.

## Externalized settings

Every threshold is available under the `PICKLEBALL_VISION_` prefix. Important
settings include:

- `RALLY_MINIMUM_MOTION_SPEED_DIAGONALS_PER_SECOND`
- `RALLY_SERVE_MINIMUM_SPEED_DIAGONALS_PER_SECOND`
- `RALLY_SERVE_SPEED_SURGE_RATIO`
- `RALLY_SERVE_CONFIRMATION_SECONDS`
- `RALLY_END_QUIET_SECONDS`
- `RALLY_RESTART_QUIET_SECONDS`
- `RALLY_DEAD_BALL_HANDOFF_WINDOW_SECONDS`
- `RALLY_DEAD_BALL_HANDOFF_MINIMUM_QUALITY_MARGIN`
- `RALLY_DEAD_BALL_HANDOFF_FULL_DURATION_SECONDS`
- `RALLY_MINIMUM_DURATION_SECONDS`
- `RALLY_MAXIMUM_DURATION_SECONDS`
- `RALLY_AUDIO_SUPPORT_TOLERANCE_SECONDS`
- `RALLY_EVALUATION_MINIMUM_IOU`
- `RALLY_EVALUATION_BOUNDARY_TOLERANCE_SECONDS`
- `RALLY_SPARSE_EVALUATION_MARGIN_SECONDS`

The complete defaults and effective values are stored in every output. Threshold
experiments should use separate output directories and retain the corresponding
configuration snapshot.
