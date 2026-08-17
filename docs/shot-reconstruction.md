# Shot reconstruction and initial classification

Milestone 17 reconstructs a separate shot layer from compatible rally, contact,
ball-trajectory, bounce, hitter, and logical-player artifacts. It does not modify
those inputs, project airborne ball observations through the court homography, train
a neural network, or produce coaching advice.

## Command and artifacts

Run from `services/vision`:

```bash
uv run pickleball-vision reconstruct-shots /absolute/path/to/match.mp4 \
  --ball-tracks /absolute/path/to/ball_tracks.json \
  --rallies /absolute/path/to/rallies.json \
  --contacts /absolute/path/to/contacts.json \
  --bounces /absolute/path/to/bounces.json \
  --hitters /absolute/path/to/hitters.json \
  --player-tracks /absolute/path/to/tracks.json \
  --annotations /absolute/path/to/annotations.json \
  --evaluation-partition validation \
  --output-dir /absolute/path/to/shot-reconstruction
```

`--bounces` and `--annotations` are optional. Without bounces, shots have no linked
bounce or landing court position. Without supported human `shotType` annotations,
evaluation is explicitly unavailable.

The command writes:

- `shots.json`: source provenance, effective thresholds, contracts, statistics, and
  chronological structured shots.
- `shot-debug.mp4`: source-resolution overlays showing the inferred hitter, type,
  confidence, rally/index, bounce link, trajectory coverage, and recent ball trail.
  Observed trajectory is green, interpolation is amber, and unknown/segment gaps
  remain disconnected.
- `shot-evaluation.json`: post-inference comparison with human shot labels.

Every input must describe the same source dimensions, frame count, FPS, and path.
Contacts, hitters, tracks, and ball-dependent artifacts additionally retain and
validate content hashes. Existing output artifacts are not overwritten.

## Reconstruction contract

Only fused-accepted contacts inside a non-overlapping predicted rally become shots.
Within each rally, contact order defines the one-based `shotIndex`. A trajectory
segment begins at that contact and ends at the next contact frame or rally end. The
segment references the immutable ball timeline and records observed, interpolated,
and unknown counts; it does not duplicate or rewrite raw trajectory points.

The first accepted bounce strictly after the contact and before the next contact is
linked. For the last shot, a bounce at the rally-end frame is eligible. A landing
court position is copied only when that accepted bounce already contains a
visually-justified, plane-gated court point. An arbitrary airborne ball point is
never mapped to the court.

The hitter court position is the raw player-track ground-contact record at the
contact frame. Its image point must use `bounding_box_bottom_center`. Bounding-box
center is never used as a physical player position. Missing hitter, player, bounce,
or trajectory evidence remains explicit.

## Ordered classification rules

The classifier supports exactly `SERVE`, `RETURN`, `DINK`, `DROP`, `DRIVE`,
`VOLLEY`, `OVERHEAD`, `OTHER`, and `UNKNOWN`. Rules run in the order below; the
first match wins. Each shot persists the tested rule list, feature values, selected
rule, confidence, and threshold configuration.

Before specialized rules run, the result is `UNKNOWN` if any required evidence gate
fails: known hitter, minimum hitter confidence, hitter court position, minimum
trajectory coverage, minimum known trajectory-point count, or initial image-space
speed. `UNKNOWN` is deliberate and is never replaced simply to increase coverage.

1. `SERVE`: rally shot 1, hitter at least the configured backcourt distance from
   the kitchen line, and an accepted outgoing bounce.
2. `RETURN`: rally shot 2 after an inferred `SERVE` whose trajectory linked an
   accepted bounce.
3. `OVERHEAD`: shot 3 or later, no linked bounce on the incoming shot, contact in
   the configured upper portion of the hitter box, and sufficient initial speed.
4. `DINK`: hitter is near the kitchen, outgoing shot lands in the opponent kitchen,
   and initial speed is at or below the dink maximum.
5. `DROP`: hitter is back from the kitchen, outgoing shot lands in the opponent
   kitchen, and initial speed is at or below the drop maximum.
6. `VOLLEY`: shot 3 or later, hitter is near the kitchen, and the incoming shot has
   no linked bounce.
7. `DRIVE`: initial image-space speed meets the configured drive minimum.
8. `OTHER`: all evidence gates pass but no specialized rule matches.

The initial speed is measured in source-frame diagonal fractions per second, making
it resolution-normalized but not a physical 3D speed. The overhead height feature
is an image-space ratio inside the assigned hitter box, not a pose or true ball
height. Opponent bottom-center court positions are retained as inspectable context;
this first rule set does not force a class from opponent position alone.

Default thresholds and their environment variables are:

| Meaning | Default | Environment suffix |
| --- | ---: | --- |
| Minimum hitter confidence | 0.62 | `SHOT_MINIMUM_HITTER_CONFIDENCE` |
| Minimum trajectory coverage | 0.50 | `SHOT_MINIMUM_TRAJECTORY_COVERAGE` |
| Minimum known trajectory points | 3 | `SHOT_MINIMUM_KNOWN_TRAJECTORY_POINTS` |
| Serve backcourt distance | 1.20 m | `SHOT_SERVE_MINIMUM_BACKCOURT_DISTANCE_METERS` |
| Kitchen proximity | 0.90 m | `SHOT_KITCHEN_PROXIMITY_METERS` |
| Drop backcourt distance | 0.90 m | `SHOT_DROP_MINIMUM_BACKCOURT_DISTANCE_METERS` |
| Dink maximum image speed | 0.28 diagonal/s | `SHOT_DINK_MAXIMUM_SPEED_DIAGONALS_PER_SECOND` |
| Drop maximum image speed | 0.38 diagonal/s | `SHOT_DROP_MAXIMUM_SPEED_DIAGONALS_PER_SECOND` |
| Drive minimum image speed | 0.45 diagonal/s | `SHOT_DRIVE_MINIMUM_SPEED_DIAGONALS_PER_SECOND` |
| Overhead minimum image speed | 0.35 diagonal/s | `SHOT_OVERHEAD_MINIMUM_SPEED_DIAGONALS_PER_SECOND` |
| Overhead contact-height ratio | 0.45 | `SHOT_OVERHEAD_MAXIMUM_CONTACT_HEIGHT_RATIO` |
| Evaluation timing tolerance | 120 ms | `SHOT_EVALUATION_TOLERANCE_MS` |
| Debug trajectory trail | 0.75 s | `SHOT_DEBUG_TRAIL_SECONDS` |

Prefix every suffix with `PICKLEBALL_VISION_`.

## Evaluation

Human `shotType` values on `SERVE_CONTACT` or `PADDLE_CONTACT` events are loaded only
after inference. An explicit `SHOT_TYPE` record at the same frame takes precedence
when present in imported annotation JSON. Unsupported labels are counted rather
than coerced into the fixed vocabulary.

Predictions and annotations are matched one-to-one by contact time without using
the class label. Evaluation reports accuracy over matched shots, per-class
precision/recall/F1, the fixed-label confusion matrix, ground-truth match coverage,
unknown rate, and matched unknown rate. The partition label is retained, and the
command never tunes thresholds automatically; validation and test labels must not
be used for threshold tuning.

Because annotation files may cover representative rallies rather than the full
video, per-class precision is conditional on time-matched predictions; unlabeled
predictions are not treated as false positives. Per-class recall uses every
supported ground-truth shot of that class, so missed annotations remain false
negatives. The evaluation artifact records this scope explicitly.

Classification quality cannot exceed upstream rally, contact, bounce, hitter, and
ball-trajectory quality. Review those artifacts and `UNKNOWN` evidence gates before
interpreting class metrics.
