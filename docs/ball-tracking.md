# Ball Trajectory Reconstruction

Milestone 9 converts immutable frame-local detector candidates into one conservative
primary-match trajectory in source-image pixels. It does not infer rallies, bounces,
contacts, hitters, shots, or court-plane ball coordinates.

## Run the checkpoint

From `services/vision`:

```bash
uv run pickleball-vision track-ball ../../sample-data/vid.mp4 \
  --detections ../../output/ball-detection/long-13m-v1-suggestions/detections.json \
  --calibration ../../output/calibration/calibration.json \
  --output-dir ../../output/ball-tracking
```

The source path, dimensions, FPS, and decoded frame count must match the raw
detection artifact. The calibration must come from the same video and dimensions.
The command refuses an existing `ball_tracks.json`; choose a new directory when
comparing configurations so a prior run remains inspectable.

## Outputs

- `ball_tracks.json` is the structured, frame-complete trajectory and source of
  truth for later stages.
- `ball-debug.mp4` is a silent review overlay at the source resolution and FPS.
- `ball-tracking-summary.json` repeats run provenance and checkpoint metrics.

Every source frame has exactly one state:

- `OBSERVED`: one raw detection passed association. Its detection ID, raw box-center
  point, detector confidence, association confidence, court relevance, and temporal
  support remain available. Smoothing is a separate field.
- `INTERPOLATED`: a configured short gap lies between two observations in the same
  segment. The raw point and source detection ID are null; the interpolation point,
  separate smoothed point, and reduced confidence are retained.
- `UNKNOWN`: no defensible primary-match position exists. Position and confidence
  remain null, even when rejected raw candidates exist on that frame.

Each frame also lists its candidate count and rejected detection IDs. The raw
`detections.json` is never rewritten.

## Association policy

Starting or continuing a segment is not a highest-confidence decision. Candidate
association combines:

1. predicted source-image position from recent velocity;
2. normalized speed and acceleration gates;
3. distance from the prediction;
4. bidirectional short-window temporal support;
5. detector confidence; and
6. primary-court image relevance.

The known court corners are projected from the court plane into the image to define
an asymmetric relevance envelope. The envelope has side and below-baseline margins
plus a larger upward margin because an airborne ball may appear above the projected
court polygon. This is only an image-space heuristic. No detected, interpolated, or
smoothed ball point is transformed through the court homography.

Segments with fewer than the configured minimum observations are discarded. An
active segment may bridge only the short association window. Linear interpolation
is then allowed only across the smaller interpolation window and never connects two
different segments. Bounded centered smoothing does not overwrite raw points or
cross an unknown interval.

## Debug overlay

- gray boxes: raw candidates rejected from the primary trajectory;
- green box/ring/dot and trail: observed raw evidence plus its separate smoothed
  display point;
- amber diamond: an interpolated point; and
- red `PRIMARY BALL: UNKNOWN`: no defensible position for that frame.

Review the video around fast direction changes, player occlusions, net overlap,
neighboring-court play, and transitions into or out of unknown periods. The overlay
is a diagnostic aid; the JSON retains exact evidence and uncertainty.

## Summary definitions

- **Trajectory coverage:** `(OBSERVED + INTERPOLATED) / all source frames`.
- **Observed coverage:** `OBSERVED / all source frames`.
- **Interpolated fraction:** `INTERPOLATED / (OBSERVED + INTERPOLATED)`.
- **Longest missing interval:** longest consecutive run of `UNKNOWN`, including its
  inclusive frame bounds, frame count, and `frame_count / FPS` duration.
- **Candidate rejection count:** raw candidates not selected as `OBSERVED`, including
  distractors, gated outliers, and detections in discarded short segments.

Coverage is not detector accuracy and an unknown frame is not evidence that no ball
was visible.

## Configuration

Defaults are deliberately conservative and are recorded in both JSON outputs. The
following environment settings use the `PICKLEBALL_VISION_` prefix:

| Suffix | Default | Meaning |
| --- | ---: | --- |
| `BALL_TRACKING_MAX_ASSOCIATION_GAP_SECONDS` | `0.20` | Longest gap eligible for candidate reassociation |
| `BALL_TRACKING_MAX_INTERPOLATION_GAP_SECONDS` | `0.10` | Longest gap eligible for interpolation |
| `BALL_TRACKING_MAX_SPEED_DIAGONALS_PER_SECOND` | `3.0` | Perspective-neutral image-speed gate |
| `BALL_TRACKING_MAX_ACCELERATION_DIAGONALS_PER_SECOND_SQUARED` | `80.0` | Image-acceleration gate |
| `BALL_TRACKING_BASE_GATE_DIAGONAL_FRACTION` | `0.012` | Position-gate allowance for localization noise |
| `BALL_TRACKING_COURT_SIDE_MARGIN_FRACTION` | `0.12` | Horizontal court-envelope margin |
| `BALL_TRACKING_COURT_AIR_MARGIN_FRACTION` | `0.50` | Upward airborne image margin |
| `BALL_TRACKING_COURT_BOTTOM_MARGIN_FRACTION` | `0.06` | Below-baseline image margin |
| `BALL_TRACKING_MINIMUM_START_SCORE` | `0.40` | New-segment evidence threshold |
| `BALL_TRACKING_MINIMUM_ASSOCIATION_SCORE` | `0.32` | Continuing association threshold |
| `BALL_TRACKING_MINIMUM_SEGMENT_OBSERVATIONS` | `2` | Persistence required to retain a segment |
| `BALL_TRACKING_SMOOTHING_WINDOW_FRAMES` | `5` | Odd centered smoothing window |
| `BALL_TRACKING_MAXIMUM_SMOOTHING_ADJUSTMENT_DIAGONAL_FRACTION` | `0.015` | Per-point smoothing bound |
| `BALL_TRACKING_DEBUG_TRAIL_SECONDS` | `0.75` | Recent trail shown in the overlay |

Increase gap limits only after checking that neighboring-court balls are not stealing
the trajectory. A higher coverage number is not automatically a better trajectory.
