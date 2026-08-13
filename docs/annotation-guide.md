# Annotation Guide

This is the durable labeling policy for future dataset milestones. No annotation
tooling is implemented during Foundation.

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

Label the visible ball center and an approximate extent when resolvable. Include
visibility (`clear`, `blurred`, `occluded`, `out_of_frame`, `not_found`) and an
annotation confidence. Do not label an interpolated point as an observed ball.

Interpolation is a derived artifact, never hand-authored ground truth. Missing
intervals remain missing unless image evidence supports an annotation.

## Events

Rally boundaries, bounces, paddle contacts, hitters, and shot classes are distinct
labels. Each event includes a frame interval when exact timing is ambiguous, a
confidence, and links to evidence. “Unknown hitter” and “unknown shot class” are
valid; guessing is not.

## Quality control

Maintain train/validation/test separation at the source-match level to avoid frame
leakage. Double-label a representative subset, adjudicate disagreements without
discarding the original labels, and report agreement alongside model metrics.
