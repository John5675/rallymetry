# Coordinate System

All geometry must name its coordinate space. Tuples such as `(x, y)` are
insufficient at API boundaries without a type or field names that identify the
space and unit.

## Image coordinates

- Origin: top-left pixel of the decoded frame.
- Positive x-axis: right.
- Positive y-axis: down.
- Unit: pixels, represented as floating-point values where subpixel estimates are
  possible.
- Geometry is associated with the decoded frame width, height, and orientation.

A person detection bounding box is `(left_px, top_px, right_px, bottom_px)`. The
initial ground-contact estimate is:

```text
x_px = (left_px + right_px) / 2
y_px = bottom_px
```

Never substitute the bounding-box center for this ground-contact estimate.

## Canonical court-plane coordinates

The canonical doubles court uses meters:

- Origin `(0, 0)`: near-left outer baseline corner when facing from the near end
  toward the far end.
- Positive x-axis: across the court from left to right.
- Positive y-axis: from the near baseline toward the far baseline.
- Bounds: `0 <= x <= 6.096 m`, `0 <= y <= 13.4112 m` (20 ft by 44 ft).
- Net line: `y = 6.7056 m`.

“Near” and “far” are calibration labels relative to the source camera, not player
identities. A calibration record must store the corner correspondence and units so
orientation is recoverable.

## Homography limits

A homography maps between the image and the two-dimensional court plane only.
Valid examples include annotated court landmarks, a player's estimated shoe/court
contact, and a confirmed ball bounce location. An airborne player extremity,
paddle, or ball does not lie on that plane.

Never project an airborne ball through the homography as a court position. Future
3D ball reconstruction must use a camera model and sufficient temporal or
multi-view constraints, and must advertise its uncertainty separately.

## Time coordinates

Keep an integer zero-based decoded `frame_index` and an explicit `timestamp_s` when
available. Record the time-base source (container timestamp, decoded frame rate, or
estimated fallback). Do not assume a constant frame rate without validating it.
