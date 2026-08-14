# Persistent Player Tracking

Milestone 5 carries the four human-assigned roles through a representative clip
without treating a transient tracker ID as a person's identity.

## Inputs and command

The standard output layout supports the concise command:

```bash
cd services/vision
uv run pickleball-vision track-players /absolute/path/to/match.mp4 \
  --calibration ../../output/calibration/calibration.json \
  --output-dir ../../output/player-tracking
```

By default, the command reads the sibling
`output/player-isolation/player-assignments.json`, then follows that artifact's
recorded path to the immutable `detections.json`. For custom layouts, pass
`--assignments` and optionally `--detections`. The latter must match the detection
file recorded by the assignments.

Human-readable names are optional and remain independent of the four stable roles.
Put a `player-names.json` beside `player-assignments.json`, or pass it explicitly
with `--player-names`:

```json
{
  "ME": "John",
  "PARTNER": "Denny",
  "OPPONENT_1": "Oksana",
  "OPPONENT_2": "Diana"
}
```

## Identity model

Ultralytics ByteTrack supplies short-term box association and tracker IDs. The
project-owned adapter records these as a raw layer linked to raw person-detection
indices. A separate resolver carries `ME`, `PARTNER`, `OPPONENT_1`, and
`OPPONENT_2` from their manual anchor observations in both time directions.
The selected isolation candidate's short-gap observations are retained as soft
seeds, not promoted into permanent identities; tracker changes around those seeds
still retain uncertainty and can require review.

The resolver considers:

- bottom-center ground-contact estimates, never person-box centers;
- inside/near/outside calibrated court membership;
- the manually observed near or far court side;
- elapsed time and physically plausible court-plane movement;
- a two-band clothing-color descriptor learned around each manual anchor;
- appearance separation from the other player on the same court side;
- transient tracker continuity and short occlusions; and
- one-to-one observation use across the four roles.

Clearly lateral outside-court observations are rejected. Existing tracker continuity
may retain a player beyond a baseline because match players legitimately stand deep;
that exception never seeds a new identity. A new tracker ID can be associated with
the same role after a defensible short gap. After a longer gap, reacquisition
requires stronger, role-distinctive appearance evidence in addition to the geometry
constraints. Weak or immediate ID changes are marked `suspected_identity_switch`;
uncertainty becomes `temporarily_missing` instead of allowing a neighboring player
to steal the role.

Appearance is clothing evidence for this recording, not facial recognition or a
permanent biometric identity. It is stored as inspectable similarity and same-side
margin values. A new recording or wardrobe change requires new manual anchors and
new prototypes.

## Artifacts

- `tracks.json` contains separate raw transient-tracker and logical-identity layers,
  plus configuration, provenance, ground points, confidence, and switch events.
- `annotated.mp4` shows all raw people subtly and resolved roles distinctly with
  role, tracker ID, confidence, ground point, and state.
- `tracking-summary.json` reports frames processed, per-role coverage, suspected ID
  switches, longest missing interval, and reacquisition count.

The annotated video intentionally has no semantic rally, ball, contact, bounce, or
audio analysis.

## Configuration

Defaults are inspectable through `pickleball-vision doctor`. Environment variables
use the `PICKLEBALL_VISION_` prefix:

- `TRACKING_HIGH_THRESHOLD`, `TRACKING_LOW_THRESHOLD`, and
  `TRACKING_NEW_THRESHOLD` control ByteTrack acceptance.
- `TRACKING_MATCH_THRESHOLD` controls transient box association.
- `TRACKING_BUFFER_SECONDS` retains lost ByteTrack tracks through short occlusions.
- `TRACKING_MAX_IDENTITY_GAP_SECONDS` bounds logical reacquisition.
- `TRACKING_MAX_PLAYER_SPEED_MPS` bounds court-plane motion.
- `TRACKING_MINIMUM_IDENTITY_SCORE` rejects weak role associations.
- `TRACKING_SUSPECTED_SWITCH_SCORE` controls when ID changes require review.
- `TRACKING_APPEARANCE_WEIGHT` controls how much clothing evidence contributes to
  identity scoring.
- `TRACKING_MINIMUM_APPEARANCE_SIMILARITY` and
  `TRACKING_MINIMUM_APPEARANCE_MARGIN` reject weak or same-side-ambiguous tracker
  changes.
- `TRACKING_APPEARANCE_PROTOTYPE_WINDOW_SECONDS` controls the evidence window around
  each manual anchor.
- `TRACKING_LONG_GAP_APPEARANCE_SIMILARITY` and
  `TRACKING_LONG_GAP_MINIMUM_APPEARANCE_MARGIN` impose stricter requirements after
  the normal identity gap expires.

Tune only after reviewing representative failures, and retain the effective values
stored in each run.

## Validation review

Use a 2–5 minute clip that includes near/far players, crossings, at least one brief
occlusion, and adjacent-court people. Review `annotated.mp4` at normal speed and
frame-by-frame around every `suspected_identity_switch`. Confirm each role label
stays on the same person, missing states appear instead of identity theft, and
ground dots remain at the bottom-center shoe/court estimate. Then inspect per-role
coverage and gaps in `tracking-summary.json` and the supporting raw observation IDs
in `tracks.json`.
