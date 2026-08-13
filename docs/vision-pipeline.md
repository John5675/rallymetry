# Vision Pipeline Contract

This document describes stable stage boundaries. A stage may be introduced only
when its matching milestone is current. Video ingestion and manual court
calibration are currently implemented; all later stages remain contracts only.

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

## Video source contract

The pipeline accepts a local file path. How bytes arrived locally—camera transfer,
object-store download, or a separately performed download from an unlisted
source—is outside the vision service. Private source URLs and credentials never
enter pipeline output or source control.

Video metadata records the resolved path, pixel dimensions, OpenCV-reported FPS
as a floating-point value, frame count, duration in seconds, and codec FourCC when
available. Duration is `frame_count / fps`; it is an OpenCV/container estimate,
not a promise of constant-frame-rate presentation timestamps.

Timestamp extraction maps a valid timestamp in `[0, duration)` to the containing
zero-based frame index using the reported FPS. Frames are decoded and written at
their source pixel dimensions. Uniform sampling selects unique indices over the
inclusive range from the first to the last frame; a one-frame sample uses the
middle frame.

## Court calibration contract

Manual calibration uses named court-plane landmarks defined in
`coordinate-system.md`. At least four geometrically valid image/court
correspondences are required. Additional selected landmarks participate in a
robust fit and retain their inlier/outlier status rather than being silently
discarded.

Fit selection evaluates all-point residuals before falling back to robust outlier
rejection. Quality reporting uses errors across every selected correspondence and
must warn when a tight inlier subset masks poor whole-court alignment.

The stored image-to-court homography applies only to points known to lie on the
court plane. Its inverse maps canonical court points into the calibrated image.
Calibration quality reports forward reprojection error in meters and reverse
reprojection error in pixels. These errors describe the selected plane landmarks;
they do not make homography valid for an airborne ball.

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
