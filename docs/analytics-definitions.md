# Analytics Definitions

Release 0.1 analytics operate on versioned structured logical-player tracks and
player-position records. They never read detector tensors directly and never infer
statistics from prose. The calculation version for this release is
`player_positions_0.1`.

## Evidence and reporting

Every metric records its unit, population, calculation version, contributing frame
ranges or step count, and relevant coverage. A `null` value means the quality-gated
population was insufficient; it is not silently replaced with zero. LLMs may explain
computed values but must never create or repair them.

Release 0.1 metric confidence is represented by input position confidence, coverage,
excluded-step counts, and contributing frames. It does not combine unrelated
confidences into an uncalibrated aggregate score.

## Player position record

There is exactly one position record per logical player per decoded source frame.
Each record retains:

- zero-based `frame_number` and `timestamp_s` on the video timeline;
- the logical tracking state and confidence;
- `raw_image_ground_point`, the source-pixel bottom-center of the person box;
- `raw_court_coordinate`, the homography projection of that ground estimate when
  geometrically valid;
- `corrected_court_coordinate`, the raw court coordinate plus any explicit
  recording-local manual offset, or `null` when the raw coordinate is missing;
- `smoothed_court_coordinate`, a separate derived value or `null`; and
- the raw court-region classification and tracker-observation provenance.

The raw image and court coordinates are immutable evidence. The center of the person
box is never a physical position. A missing or rejected observation remains missing.

### Recording-local court-position corrections

An optional `player-position-corrections.json` may supply a bounded constant `(x, y)`
offset for an individual logical role in canonical court meters. The correction is
applied after homography projection and before smoothing. It never changes
`raw_image_ground_point`, `raw_court_coordinate`, raw tracking, or detector output.

```json
{
  "schema_version": 1,
  "coordinate_space": "canonical_court_meters",
  "corrections": {
    "OPPONENT_1": {
      "x_offset_m": 0.0,
      "y_offset_m": 0.15,
      "reason": "manual source-overlay alignment review"
    }
  }
}
```

Positive x points toward the court's right sideline and positive y points toward the
far baseline. A correction vector is limited to `0.50 m`. The input path, reason,
offset, and whether each role was adjusted are retained in `player_positions.json`.
When `--position-corrections` is omitted, the CLI discovers this filename beside
`tracks.json`; if absent, every offset is zero.

## Conservative smoothing

Method: `centered_confidence_weighted_component_median`.

For an eligible frame, Release 0.1 takes the component-wise confidence-weighted
median of corrected court x and y values in a centered, odd-sized window (default five
frames). Support stops at the first missing, below-threshold, nonconsecutive, or
suspected-identity-switch frame. The displacement from corrected to smoothed
coordinate is clamped to at most `0.30 m` by default.

A frame is smoothing-eligible only when it has a raw court coordinate, tracking
confidence at least `0.45` by default, and is not marked
`suspected_identity_switch`. Smoothing creates a coordinate only for an existing raw
coordinate. It never interpolates a missing frame and never bridges a gap.

Each output stores the exact supporting frame numbers, effective configuration,
`interpolated: false`, and raw, corrected, and smoothed coordinates.

## Metric populations

Trajectory-eligible positions have a smoothed coordinate and a raw court-region
state of `inside` or `near`. This allows a legitimate player just behind a baseline
while excluding clearly unrelated or geometrically ambiguous positions.

In-court positions are the trajectory-eligible subset satisfying:

```text
0 <= x_m <= 6.096
0 <= y_m <= 13.4112
```

Occupancy, distance-from-kitchen, and heatmaps use only in-court positions. Distance
traveled, lateral movement, and partner spacing use trajectory-eligible positions.

## Approximate distance traveled

Unit: meters.

For one player, order trajectory-eligible smoothed coordinates by frame. A step is
accepted only when its endpoints are consecutive source frames, elapsed time is
positive and no greater than `maximum_step_gap_seconds` (default `0.20 s`), and its
implied planar speed is no greater than `maximum_step_speed_mps` (default `8 m/s`).

For accepted endpoints `(x1, y1)` and `(x2, y2)`:

```text
step_distance_m = sqrt((x2 - x1)^2 + (y2 - y1)^2)
distance_traveled_m = sum(step_distance_m)
```

Missing spans are never bridged. The metric reports accepted-step count, contributing
frame ranges, gap exclusions, and speed-gate exclusions. It is approximate because it
inherits box, homography, identity, smoothing, and sampling error.

## Court occupancy

Unit: share of quality-gated in-court frames, with frame count and approximate
seconds also reported. The denominator is that player's total in-court metric frame
count, not the source frame count. If the denominator is zero, every share is `null`.

Default court constants are:

```text
near kitchen line = 4.5720 m
net line          = 6.7056 m
far kitchen line  = 8.8392 m
transition depth  = 2.1336 m
```

Regions are mutually exclusive with kitchen boundary priority:

- **Kitchen occupancy:** `4.5720 <= y_m <= 8.8392`.
- **Transition-zone occupancy:** near side
  `4.5720 - transition_depth <= y_m < 4.5720`, or far side
  `8.8392 < y_m <= 8.8392 + transition_depth`.
- **Backcourt occupancy:** every other in-court y coordinate.

These are explicit Release 0.1 analysis regions, not claims about a player's legal
volley status. A ground point inside the kitchen does not prove a rule violation.

## Average distance from kitchen

Unit: meters. Population: in-court smoothed positions.

This metric is the mean longitudinal setback behind the applicable kitchen line. A
position inside either kitchen has zero setback:

```text
near half (y <= net): max(0, near_kitchen_y - y)
far half  (y >  net): max(0, y - far_kitchen_y)
```

This is distance behind the kitchen line, not Euclidean distance to the kitchen
polygon and not distance to the net.

## Average partner spacing

Unit: meters. Population: frames where both members of a team simultaneously have
trajectory-eligible smoothed coordinates.

For `ME`/`PARTNER` and `OPPONENT_1`/`OPPONENT_2`, spacing at a shared frame is the
Euclidean court-plane distance between the two coordinates. Average partner spacing
is the arithmetic mean of those shared-frame distances. It reports sample count,
contributing frame ranges, and sample count divided by total source frames as source
coverage. Both players on a team receive the same team metric.

## Lateral movement statistics

Lateral means the canonical court x-axis, left-to-right when facing the far baseline.
Statistics use the same accepted consecutive steps and speed/gap gates as approximate
distance traveled.

- **Total absolute lateral distance (m):** `sum(abs(x2 - x1))` over accepted steps.
- **Mean absolute lateral speed (m/s):** total absolute lateral distance divided by
  total accepted-step elapsed time.
- **Maximum lateral speed (m/s):** maximum `abs(x2 - x1) / elapsed_s` among accepted
  steps.
- **Lateral range (m):** maximum x minus minimum x among all trajectory-eligible
  positions.
- **Lateral position standard deviation (m):** population standard deviation of x
  among all trajectory-eligible positions.

The result reports accepted steps, contributing frame ranges, gap exclusions, and
speed exclusions.

## Heatmaps and animations

Each heatmap is a presentation artifact, not an additional statistic. It bins
quality-gated in-court smoothed positions on a `48 x 96` x/y grid, applies a Gaussian
display blur, and normalizes color independently per player. Heatmap color intensity
must not be compared quantitatively across players.

The top-down animation places the far baseline at the top and near baseline at the
bottom. Trails retain only the configured recent duration and are cleared whenever a
player position is missing, so they never draw across a gap. The source-space
animation draws the raw ground point as a ring and the reverse-projected
corrected-and-smoothed court point as a filled dot.

## Deferred event vocabulary

- **Rally:** the interval from a valid serve contact to a rally-ending condition,
  represented by explicit boundary evidence and confidence.
- **Bounce location:** a court-plane location associated with a bounce event; it is
  not the projection of an arbitrary airborne ball observation.
- **Shot:** a rally-local interval anchored by an accepted paddle-contact candidate
  and ending at the next accepted contact frame or rally end. It references the
  immutable ball-trajectory range, may link the first accepted outgoing bounce, and
  retains its logical hitter, evidence, class, and confidence.
- **Hitter:** logical player identity associated with a contact candidate, retaining
  assignment confidence, ranked alternatives, and `UNKNOWN` when evidence gates do
  not pass. Audio does not select the hitter.

These definitions originated in Release 0.1. Rally, bounce, contact, hitter, and
initial shot reconstruction/classification are now separate implemented derived
layers. Shot classes are interpretive labels, not match statistics.
