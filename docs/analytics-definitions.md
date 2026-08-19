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

## Deterministic match analytics 1.0

Milestone 18 adds `match_analytics_1.0`. It consumes only validated, versioned
`Rally`, `Shot`, and `PlayerPosition` records from `rallies.json`, `shots.json`, and
`player_positions.json`. A `Shot` retains its `Contact` link and optional `Bounce`
link, so contact and bounce evidence remain available through the structured domain
model without analytics reading their raw artifacts. The stage never reads model
tensors, YOLO output, raw audio waveforms, or detector observations.

The workflow verifies that all three inputs describe the requested source video,
that the supplied rally file has the content hash recorded by the shot artifact,
and that shots and player positions derive from the same persistent-player track
path. It records input paths and SHA-256 hashes and does not modify an input.

### Shared UNKNOWN and confidence policy

Event counts count complete structured records; confidence never creates a
fractional rally, shot, or hit. Mean upstream rally, shot, and known-hitter
confidence are reported under `dataQuality`, alongside unknown and position-coverage
counts. These values describe uncertainty; they are not calibrated probabilities
that the final statistic is correct.

`UNKNOWN` is never silently assigned to a player or shot class:

- a shot with `hitterId: UNKNOWN` contributes to match shot/rally-length metrics but
  not to any player's hit or shot-type metrics;
- a hit with `shotType: UNKNOWN` contributes to its known hitter's `totalHits`, but
  is excluded from that player's classified shot-type-rate denominator;
- an unknown third-shot type is excluded from third-shot drop/drive denominators;
- a shot with unknown type or missing/ambiguous hitter position is excluded from
  the relevant position-selection rate, with its exclusion count reported; and
- a missing position stays missing. It contributes neither movement nor occupancy,
  and missing spans are not bridged.

When a denominator is zero, the metric value is `null`, not zero. Each rate reports
its integer numerator and denominator so the population is inspectable.

### Match metrics

Inputs for these metrics are validated `Rally` and `Shot` records.

- **Rally count:** `count(Rally)`. A predicted false rally remains a counted rally;
  upstream segmentation error is not repaired here. No `UNKNOWN` category applies.
- **Shot count:** `count(Shot)`. All shot types and hitter identities, including
  `UNKNOWN`, count because the upstream pipeline still asserted a shot event.
- **Average rally duration (seconds):**
  `sum(rally.endTimestamp - rally.startTimestamp) / rally_count`. A zero-rally
  population yields `null`. Boundary uncertainty and false/missed rally segments
  directly bias the result.
- **Average rally length (shots/rally):** `shot_count / rally_count`. Every shot
  references exactly one supplied rally, and rallies with no shots contribute zero
  to the numerator. A zero-rally population yields `null`. Missed/false contacts,
  rally splitting, and rally merging bias this metric.
- **Longest rally:** choose the rally with maximum linked shot count; break ties by
  longer duration and then earlier start frame. The output retains rally ID, shot
  count, duration, and frames. With no rallies the value is `null`. This is longest
  by reconstructed shots, not elapsed duration.

### Player hit and shot-type metrics

Input is each structured `Shot.hitterId` and `Shot.shotType`.

- **Total hits:** for player `p`, `count(shots where hitterId == p)`, including
  `shotType: UNKNOWN`. `hitterId: UNKNOWN` is excluded from all players and counted
  in data quality. Hitter-identification error can transfer a hit between players.
- **Dink/drop/drive/volley/overhead count:** for player `p` and class `c`,
  `count(shots where hitterId == p and shotType == c)`.
- **Dink/drop/drive/volley/overhead rate:** `class_count /
  classified_hit_count`, where `classified_hit_count` is that player's hit count
  excluding `shotType: UNKNOWN`. The denominator includes other supported known
  classes such as serve, return, and other. It is `null` when no hit has a known
  class. Classification and hitter errors affect both numerator and denominator.

These are inferred contact/shot counts, not scorebook totals. Confidence gates and
missed ball observations upstream may reduce coverage non-uniformly between near and
far players.

### Position metrics in match analytics

Inputs are structured smoothed court coordinates and raw `inside`/`near` membership
states from `player_positions.json`. Match analytics deliberately reuses that
artifact's recorded maximum step gap, maximum step speed, and transition-zone depth
instead of changing Release 0.1 definitions.

- **Distance traveled:** identical to [Approximate distance traveled](#approximate-distance-traveled).
  The value is `null` when no step passes quality gates; this distinguishes missing
  evidence from a measured zero-distance trajectory.
- **Kitchen/transition/backcourt occupancy:** identical to
  [Court occupancy](#court-occupancy), including in-court population and boundary
  priority. Each region reports frame count, approximate seconds, denominator, and
  share. Shares are `null` when no in-court position exists.
- **Average partner spacing:** identical to
  [Average partner spacing](#average-partner-spacing). Team members receive the same
  value. It is `null` without a joint quality-gated frame.

These metrics inherit bottom-center ground-point, calibration, tracking,
recording-local correction, and smoothing error. Frame-based occupancy overweights
time spans with better tracking coverage. Distance normally underestimates movement
through missing spans and may retain residual foot-box jitter despite smoothing.

### Third-shot selection

Inputs are `Shot.shotIndex` and `Shot.shotType`. The third shot is strictly the shot
whose rally-local one-based index is `3`; it is not inferred from hitter team.

```text
classified_third_shots = shots where shotIndex == 3 and shotType != UNKNOWN
third_shot_drop_rate = count(classified_third_shots labeled DROP)
                       / count(classified_third_shots)
third_shot_drive_rate = count(classified_third_shots labeled DRIVE)
                        / count(classified_third_shots)
```

The rates do not have to sum to one because another supported class can occupy the
third shot. A zero classified-third-shot denominator yields `null`; unknown third
shots are reported separately. Missed contacts can shift every later `shotIndex`, so
these metrics are reliable only when early-rally contact coverage is manually
validated.

### Shot selection by court position

Inputs are the hitter court point and court-region state retained on each structured
`Shot`, plus `Shot.shotType`. Only `inside` or `near` points are eligible. The point
is assigned to kitchen, transition zone, or backcourt using the same longitudinal
boundaries and transition depth as occupancy.

For region `r` and known class `c`:

```text
selection_count(r, c) = count(eligible shots in r with shotType == c)
selection_rate(r, c) = selection_count(r, c)
                       / count(eligible shots in r with shotType != UNKNOWN)
```

All supported known classes are reported. A zero regional denominator yields
`null` rates. Missing/ambiguous hitter positions and unknown classes are excluded
and counted separately. Contact-time player positioning comes from a bottom-center
ground estimate; it does not locate the paddle, and a `near` point can sit slightly
outside the painted boundary because of calibration uncertainty.

### Team kitchen arrival rate

Inputs are a rally interval and simultaneous structured positions for the team's two
fixed logical players. `ME`/`PARTNER` are the near team and
`OPPONENT_1`/`OPPONENT_2` are the far team for this stationary recording.

A rally is evaluable when both teammates have trajectory-eligible positions on at
least the configured fraction of rally frames (default `0.50`). The team arrives if,
on any one shared eligible frame in that rally, both players are simultaneously on
their expected court half and no farther behind their applicable kitchen line than
`kitchenArrivalDistanceMeters` (default `0.90 m`). A player already inside the
non-volley zone satisfies the distance criterion; this is positional proximity, not
a claim about volley legality.

```text
kitchen_arrival_rate = evaluable rallies where both teammates arrive
                       / rallies meeting joint-position coverage threshold
```

The value is `null` when no rally is evaluable. Per-rally coverage, eligibility,
arrival result, and first arrival frame are retained; low-coverage exclusions are
reported. This metric cannot distinguish a deliberate strategic arrival from a
player merely standing near the line, can miss asynchronous arrivals, assumes fixed
near/far team sides, and inherits rally-boundary and player-position error. For these
reasons it is reported only with explicit coverage evidence.

### Artifact and repeatability contract

Run:

```bash
pickleball-vision analyze-match /absolute/path/to/match.mp4 \
  --rallies /absolute/path/to/rallies.json \
  --shots /absolute/path/to/shots.json \
  --player-positions /absolute/path/to/player_positions.json \
  --output /absolute/path/to/match-analytics.json
```

The command refuses to overwrite an existing output. For identical JSON inputs and
configuration, every metric is deterministic; only the artifact creation timestamp
changes between separate new output paths. Configuration may override the kitchen
arrival distance and minimum joint coverage through
`PICKLEBALL_VISION_MATCH_ANALYTICS_KITCHEN_ARRIVAL_DISTANCE_METERS` and
`PICKLEBALL_VISION_MATCH_ANALYTICS_MINIMUM_KITCHEN_ARRIVAL_JOINT_COVERAGE_RATIO`.
