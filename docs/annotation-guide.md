# Annotation Guide

This is the durable labeling policy for dataset and evaluation milestones. Ball
dataset extraction, leakage-safe splitting, detector training, local detector-data
review, and multimodal match-event annotation are implemented. Model suggestions
are never ground truth until a human explicitly accepts them and marks the frame
reviewed.

## General rules

- Annotate what is visible; do not fill gaps from expectation.
- Preserve source video identity with a non-secret content identifier, frame index,
  timestamp/time base, annotation schema version, and annotator/review status.
- Use explicit `unknown`, `not_visible`, or `not_applicable` states instead of
  guessed values.
- Keep raw annotations separate from adjudicated labels and model predictions.
- Record ambiguous cases for review rather than forcing a class.
- Do not commit source videos, private video URLs, or personal information.

## Court

Label visible canonical landmarks by semantic name, not merely point order. Mark
occluded or cropped landmarks as absent. Never infer a landmark solely to complete
a rectangle. A reviewer should verify corner correspondence and near/far
orientation before calibration becomes evaluation truth.

## People and players

Person boxes should tightly cover the visible person. Occlusion and truncation are
separate flags. Match-player identity is a temporal label and must not be assigned
from a single frame solely by detection confidence.

Where visible, annotate the court ground-contact point between the player's shoes.
Otherwise, the future initial estimate is the bottom-center of the player box and
must be marked as estimated rather than human-observed. Never label the box center
as court position.

Stable player IDs are match-scoped anonymous identifiers such as `near_left` only
when side/role is known; otherwise use neutral identifiers. Side changes must not
create a new physical identity.

The local isolation workflow's `ME`, `PARTNER`, `OPPONENT_1`, and `OPPONENT_2`
values are manually asserted logical roles, not detector classes or detector IDs.
Their assignment anchor, observed side, and source candidate remain provenance.
Correct a mistaken role in a new assignment record rather than editing raw person
detections.

## Ball

The object class is exactly `pickleball`. Label every visible pickleball with a
tight image-space extent around visible evidence and a center when the center is
defensible. Preserve source pixels; do not project an airborne ball through court
homography. Each object should retain:

- source ID, frame number, and timestamp;
- image-space center when defensible and visible extent;
- visibility: `clear`, `partial`, `blurred`, or `ambiguous`;
- truncation and occlusion flags;
- scope: `primary_match`, `neighboring_court`, or `unknown`;
- annotation confidence and annotator/review state; and
- a stable within-clip object reference only when visually supportable.

For detector evaluation, add human-owned `court_side=near|far|unknown`. This describes
which side the visible ball belongs to based on review context; it must not be generated
by projecting an airborne image point through court homography. Unknown is valid and is
included in overall metrics but excluded from near/far subtotals.

Use these case rules:

- **Partially visible ball:** annotate only the visible pixels with
  `visibility=partial` and `truncated=true` when cropped by the frame. Do not invent
  the hidden extent. Store a center only when the full-ball center is defensible;
  otherwise leave it unknown and require review.
- **Blurred ball:** annotate the tight blur footprint, not an idealized circular
  ball. Use `visibility=blurred`, lower confidence as appropriate, and retain the
  blur direction only as optional descriptive evidence—not as a trajectory.
- **Ambiguous ball:** do not force a positive or negative label. Keep the frame in
  `unlabeled`, add an `ambiguous` review record, and adjudicate it later. Lights,
  shoes, court marks, fence holes, clothing, and other bright objects are common
  ambiguities.
- **Neighboring-court ball:** annotate it as class `pickleball` with
  `scope=neighboring_court`. It is a positive visual example for a generic ball
  detector, even though it is not the primary-match ball. Never put it in the
  negative bucket merely because it belongs to another court.
- **Fully occluded ball:** draw no observed ball box or center. A temporal review
  record may say `not_visible`/fully occluded, but it is not an observed positive
  annotation and its expected location must not be guessed.
- **Multiple balls:** annotate every separately visible pickleball as its own
  object. Set scope independently for each. If the primary-match ball cannot be
  distinguished, use `scope=unknown`; do not select one by proximity or confidence
  alone.

Interpolation is a derived artifact, never hand-authored ground truth. Missing
intervals remain missing unless image evidence supports an annotation.

### Dataset grouping

- `positive` means a human has confirmed at least one visible pickleball annotation.
- `negative` means a human has reviewed the frame and confirmed that no pickleball
  is visible. Hard negatives should deliberately include lights, shoes, markings,
  paddles, net/fence patterns, bright clothing, sky, and court backgrounds.
- `unlabeled` means annotation is incomplete, absent, or ambiguous. It must never be
  consumed as a negative class automatically.

Positive/negative directories are curation queues. The annotation record remains
the source of truth, especially for multiple balls and neighboring-court scope.
Detector training requires an explicit `review_status=reviewed` record for every fixed
split frame. An empty objects array is a detector negative only in such a reviewed
record. Ambiguous objects remain review material and are rejected as detector ground
truth.

## Events

The match ground-truth schema supports `RALLY_START`, `RALLY_END`,
`SERVE_CONTACT`, `PADDLE_CONTACT`, `BOUNCE`, `RALLY_WINNER`, and `SHOT_TYPE` as
distinct human labels. Each event retains an exact frame, derived video and
canonical media timestamps, stable ID, and optional player, team, shot type,
court-plane position, notes, and annotation confidence. Unknown or omitted player,
winner, location, and shot class are valid; guessing is not.

Optional audio labels are `PRIMARY_EVENT_AUDIBLE`,
`PRIMARY_EVENT_NOT_AUDIBLE`, `OTHER_COURT_TRANSIENT`, and `AMBIGUOUS_AUDIO`.
They describe human review context for an existing event. A raw transient marker
is not a bounce or contact, must not create an event automatically, and may belong
to a neighboring court. Normal match annotation never requires audio.

The local editor saves after every validated edit, resumes compatible files, and
does not alter source media or audio-analysis artifacts. Exact event semantics,
schema, controls, and the recommended review procedure are documented in
[the multimodal match annotation guide](match-annotation.md).

Automatic rally evaluation pairs human `RALLY_START`/`RALLY_END` events only after
inference. A selected 5–10-rally annotation file is sparse evaluation coverage, not
proof that the rest of the video contains no rallies. Treat unreviewed time as
excluded. Mark complete-video coverage explicitly only after a human has reviewed
the entire recording for missing rally boundaries. Thresholds may be developed on a
development partition but must remain frozen for held-out validation/test clips.

### Score calls and completed games

A doubles score call is recorded in server-first order:

```text
serving-team score, receiving-team score, server number
```

For example, `1-2-2` means the team with one point is serving with its second
server. If that team loses the rally and side-outs, the unchanged team totals are
called `2-1-1` by the new serving team. This reorder is not a point or a
contradiction. The initial serving turn is the special `0-0-2` opening call; later
side-outs normally begin with server one.

Keep the exact heard words and interpreted numbers as separate fields. Players may
misspeak, question the score, or correct themselves. Retain such observations as
`CONTRADICTORY`, `SELF_CORRECTED`, or `UNRESOLVED`; never rewrite the raw call to
make the sequence look consistent.

A completed regulation game has one of these terminal forms:

```text
11-x, where x <= 9
n-(n-2), where n >= 12
```

The second form represents play continuing from 10-10 until one team leads by two,
such as 14-12 or 16-14. A complete-video review must link the terminal state to the
last accepted rally and record the winning side, fixed-team score, source frame and
timestamp, evidence, and confidence. If the final number is not spoken, it may be
marked `INFERRED` only when a defensible pre-rally score plus the visually reviewed
final point uniquely determines it. Recording duration or an expected final score
is never sufficient evidence by itself.

## Quality control

Prefer train/validation/test separation at the whole-video level. Clip or rally/group
splitting is allowed only when those ranges are explicit, non-overlapping, and treated
as indivisible units. Never assign individual neighboring frames independently across
splits. Double-label a representative subset, adjudicate disagreements without
discarding the original labels, and report agreement alongside model metrics.
