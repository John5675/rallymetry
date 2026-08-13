# Primary Match Player Isolation

Milestone 4 distinguishes primary-court candidates from the many people visible on
neighboring courts. It does not choose the four highest-confidence detections and
does not implement the persistent tracker planned for Milestone 5.

## Derived geometry and candidate selection

For every raw person box, isolation estimates ground contact as:

```text
x_px = (left_px + right_px) / 2
y_px = bottom_px
```

That bottom-center image point—not the person-box center—is projected to court
meters when it is not clipped at the bottom frame edge. The derived observation
retains projection status, inside/near/outside state, uncertainty, near/far side,
and links to the unchanged raw detection.

Candidate association is deliberately limited. A greedy short-gap association uses
ground-point distance, elapsed time, box-scale similarity, and court side. It can
bridge an isolated missed frame, but its `candidate-*` identifiers are ephemeral
selection aids and may fail through long occlusion, crossings, or end changes.
Court support and observed-frame persistence determine eligibility. Detector
confidence is never used to rank and take four people.

## Run the workflow

From `services/vision`:

```bash
MATCH_VIDEO="/absolute/path/to/match.mp4"
DETECTIONS="../../output/person-detection/detections.json"
CALIBRATION="../../output/calibration/calibration.json"

uv run pickleball-vision isolate-players "$MATCH_VIDEO" \
  --detections "$DETECTIONS" \
  --calibration "$CALIBRATION" \
  --timestamp 30.5 \
  --output-dir ../../output/player-isolation
```

Choose a timestamp where all four match players are visible and separated. The
window starts with `ME`; click the correct person box, then repeat for `PARTNER`,
`OPPONENT_1`, and `OPPONENT_2`.

Controls:

- `1`, `2`, `3`, `4`: select which logical role to assign or correct;
- left-click: assign the current role to that person candidate;
- `a` / `d`: move backward/forward one frame;
- `j` / `l`: move backward/forward approximately one second;
- `c`: clear the current role;
- `r`: clear all assignments;
- Enter or Space: finish after all four distinct roles are assigned; and
- `q` or Escape: cancel without writing assignments.

To correct an existing assignment, rerun with:

```bash
uv run pickleball-vision isolate-players "$MATCH_VIDEO" \
  --detections "$DETECTIONS" \
  --calibration "$CALIBRATION" \
  --timestamp 30.5 \
  --output-dir ../../output/player-isolation-corrected \
  --assignments ../../output/player-isolation/player-assignments.json
```

The prior anchors load into the window. Press a role number, navigate if needed,
and click its corrected person.

## Debug artifacts

- `player-candidates.json`: derived court states and ephemeral candidate tracklets;
- `player-assignments.json`: the four independent manual logical roles;
- `primary-player-debug.mp4`: subtle boxes for every person, distinct eligible or
  assigned candidates, bottom-center dots, court state, projected court lines, and
  logical labels; and
- `primary-player-summary.json`: candidate and court-state diagnostics, not match
  statistics.

Debug colors use green for inside ground points, yellow for near, red for outside,
and orange for ambiguous. Gray boxes remain unrelated or insufficiently supported;
cyan boxes are eligible primary-court candidates; assigned logical players receive
role-specific colors and labels.

## Configuration

All variables use the `PICKLEBALL_VISION_` prefix:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ISOLATION_NEAR_MARGIN_METERS` | `1.5` | Maximum outside distance classified as near |
| `ISOLATION_BOUNDARY_UNCERTAINTY_METERS` | `0.25` | Ambiguity band around region decisions |
| `ISOLATION_SIDE_UNCERTAINTY_METERS` | `0.25` | Near/far ambiguity band around the net |
| `ISOLATION_MAX_CANDIDATE_GAP_SECONDS` | `1.0` | Maximum short gap eligible for association |
| `ISOLATION_MAX_CANDIDATE_SPEED_MPS` | `8.0` | Ground-distance association gate |
| `ISOLATION_MIN_CANDIDATE_OBSERVATIONS` | `15` | Minimum temporal support for eligibility |
| `ISOLATION_MIN_COURT_SUPPORT_RATIO` | `0.60` | Minimum inside/near support for eligibility |

Thresholds are stored with the candidate artifact. Adjust them only after
inspecting the debug video; full identity reliability belongs to persistent
tracking and labeled evaluation rather than increasingly complex Milestone 4
heuristics.
