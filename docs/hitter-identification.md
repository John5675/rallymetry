# Hitter Identification

Milestone 16 derives a logical-player decision from each immutable Prompt 15
paddle-contact candidate. It never changes `contacts.json` or `tracks.json`, and it
does not classify a shot.

## Command

Run from `services/vision`:

```bash
uv run pickleball-vision identify-hitters /absolute/path/to/match.mp4 \
  --contacts ../../output/contact-detection/contacts.json \
  --player-tracks ../../output/player-tracking/tracks.json \
  --annotations ../../output/match-annotations.json \
  --evaluation-partition validation \
  --output-dir ../../output/hitter-identification
```

`--annotations` is optional and is used only after inference. The supplied player
track file must be the exact artifact recorded by `contacts.json`; its content hash
and source-video metadata are validated before work begins.

Outputs are:

- `hitters.json`: every contact-linked logical-player decision;
- `hitter-debug.mp4`: source-resolution review overlay; and
- `hitter-evaluation.json`: optional comparison with human contact player labels.

Existing output files are never overwritten.

## Identity evidence

Player scoring uses only visual contact and logical-player evidence:

- ball-to-player-box distance and whether the ball lies inside the box;
- the separately retained bottom-center ground point;
- player tracking confidence and tracking state;
- player court region and near/far court side;
- incoming/outgoing image-space trajectory direction;
- visual contact confidence;
- the prior credible hitter within the same rally or short sequence; and
- rally contact order.

The trajectory direction check uses image-space vertical velocity only as near/far
consistency evidence. Weak vertical motion is neutral rather than contradictory.
This is not 3D reconstruction, and the airborne ball is never projected through
court homography.

Audio confidence, loudness, channels, and transient location contribute exactly
zero to player identity. The source fused contact confidence remains visible, but
hitter gating and scoring use `visualConfidence` so an audio boost cannot select a
player.

## Conservative `UNKNOWN`

The tool assigns one of `ME`, `PARTNER`, `OPPONENT_1`, or `OPPONENT_2` only when all
of these gates pass:

- minimum visual contact confidence;
- at least one tracked logical-player candidate;
- maximum ball-to-player distance;
- minimum player tracking confidence;
- minimum combined player score; and
- minimum score margin over the runner-up.

If any gate fails, `playerId` is `UNKNOWN`. The failed gates, best and runner-up
scores, score margin, and every per-player score component remain in
`supportingSignals`. An eligible but uncertain contact resets previous-hitter
context, preventing one weak decision from cascading through the rest of a rally.
Context also resets across rally IDs and after the configured maximum time gap.

Every `hitterIdentifications` record contains:

- `contactId`, frame, video timestamp, and media timestamp;
- source-pixel ball position;
- `playerId` and optional display name;
- decision confidence;
- ranked alternatives; and
- inspectable contact, player, direction, previous-hitter, rally-order, score, and
  gate evidence.

## Configuration

Defaults are recorded in every output. They can be changed with:

- `PICKLEBALL_VISION_HITTER_MINIMUM_CONTACT_CONFIDENCE` (`0.78`)
- `PICKLEBALL_VISION_HITTER_MINIMUM_ASSIGNMENT_CONFIDENCE` (`0.62`)
- `PICKLEBALL_VISION_HITTER_MINIMUM_ASSIGNMENT_MARGIN` (`0.08`)
- `PICKLEBALL_VISION_HITTER_MAXIMUM_PLAYER_DISTANCE_DIAGONAL_FRACTION` (`0.12`)
- `PICKLEBALL_VISION_HITTER_MINIMUM_TRACKING_CONFIDENCE` (`0.45`)
- `PICKLEBALL_VISION_HITTER_MINIMUM_DIRECTION_SPEED_DIAGONAL_FRACTION_PER_SECOND`
  (`0.015`)
- `PICKLEBALL_VISION_HITTER_PREVIOUS_HITTER_MINIMUM_CONFIDENCE` (`0.70`)
- `PICKLEBALL_VISION_HITTER_MAXIMUM_SEQUENCE_GAP_SECONDS` (`4.0`)
- `PICKLEBALL_VISION_HITTER_EVALUATION_TOLERANCE_MS` (`100`)
- `PICKLEBALL_VISION_HITTER_PROXIMITY_WEIGHT` (`0.35`)
- `PICKLEBALL_VISION_HITTER_TRACKING_WEIGHT` (`0.17`)
- `PICKLEBALL_VISION_HITTER_DIRECTION_WEIGHT` (`0.18`)
- `PICKLEBALL_VISION_HITTER_CONTACT_WEIGHT` (`0.12`)
- `PICKLEBALL_VISION_HITTER_COURT_CONTEXT_WEIGHT` (`0.08`)
- `PICKLEBALL_VISION_HITTER_SEQUENCE_WEIGHT` (`0.10`)

Weights are normalized by their sum. Thresholds should be tuned only on a declared
development partition, never on validation or test annotations.

## Evaluation

Human `SERVE_CONTACT` and `PADDLE_CONTACT` events must include a logical `playerId`
to participate. Predictions and annotations are first matched one-to-one by time;
player identity is not used during matching. The matched pair is then scored as
correct, incorrect, or `UNKNOWN`.

The report distinguishes:

- overall accuracy: correct assignments divided by all temporally matched contacts,
  including `UNKNOWN` in the denominator;
- assignment coverage: non-`UNKNOWN` decisions divided by matched contacts;
- decisive accuracy: correct assignments divided by non-`UNKNOWN` decisions; and
- contact match coverage: labeled annotations with a nearby eligible visual contact.

Near/far accuracy is reported only when the human-labeled player has a valid tracked
court side at the matched contact. It uses the player's ground observation, not a
projected airborne ball. Missing side evidence is counted separately.

## Manual review

Review `hitter-debug.mp4` chronologically. At every marker, check the ball ring,
selected player box, alternatives, confidence, and whether `UNKNOWN` was safer than
a forced identity. Pay particular attention to occlusions, same-side partners close
together, suspected player-track switches, and contacts near the net. Correct the
human annotation file rather than editing inference output directly.
