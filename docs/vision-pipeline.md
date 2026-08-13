# Vision Pipeline Contract

This document describes intended stage boundaries, not implemented Foundation
functionality. A stage may be introduced only when its matching milestone is
current.

## Intended flow

```text
video metadata
  -> court calibration
  -> person and ball observations
  -> primary-player isolation
  -> persistent player tracks
  -> observed/interpolated ball track segments
  -> rally, bounce, contact, hitter, and shot events
  -> structured match data
  -> analytics
```

## Observation contract

An observation is evidence tied to a frame/time, model version, configuration,
confidence, and image-space geometry. It is not a pickleball event. Later stages
may link to or supersede an observation but must not silently mutate the raw result.

Person detections include bounding boxes. A player's initial physical-position
estimate is the bottom-center ground-contact point of the selected bounding box.
The bounding-box center is never a physical court position. Selecting primary
players requires temporal and court-aware evidence; taking the four highest person
confidences is not a valid algorithm.

Ball records distinguish at minimum:

- `observed`: supported directly by image evidence;
- `interpolated`: estimated only between sufficiently close supported points; and
- `missing`: no defensible position.

Interpolation must retain its method, support interval, and confidence. Long gaps
remain missing.

## Geometry contract

Court homography maps points on the physical court plane. It is valid for court
landmarks, player ground-contact points, and a confirmed ball bounce point. An
airborne ball image coordinate cannot be projected as though it lies on the court.
See `coordinate-system.md` for axes and units.

## Event contract

Events are derived records supported by observations and tracks. Every uncertain
ML-derived event retains confidence and provenance. A bounce, paddle contact,
hitter assignment, and shot class are separate inferences; confidence in one does
not imply confidence in another.

## Inspectability

Every stage should eventually support:

- deterministic configuration snapshots;
- schema and producer versions;
- links to input artifacts or record identifiers;
- machine-readable confidence and uncertainty;
- optional debug overlays that do not replace structured output; and
- evaluation against annotations defined in `annotation-guide.md`.
