# Manual Court Calibration

The Milestone 2 workflow calibrates a stationary camera from one representative
video frame. It is designed for a low camera near a baseline, strong perspective,
partial cropping, and court lines that may be occluded. Automatic line or court
detection is intentionally absent.

## Command

From `services/vision`:

```bash
uv run pickleball-vision calibrate /absolute/path/to/match.mp4 \
  --timestamp 30.5 \
  --output ../../output/calibration.json
```

Choose a timestamp with a clear court and minimal player occlusion. A window shows
one canonical landmark label at a time. Controls are:

- left click: select the prompted court-line intersection;
- `s`: skip a landmark that is cropped, hidden, or ambiguous;
- `u`: return to the previous landmark and remove its selection;
- `r`: clear all selections and restart;
- Enter or Space: fit the calibration after at least four selections;
- `q` or Escape: cancel without producing a calibration.

Select more than four well-distributed landmarks whenever possible. Favor points
on both sidelines and at multiple distances from the camera. A large number of
points concentrated on one straight line is not a valid calibration.

With exactly four points, the homography can pass through all four even when a
click is inaccurate, so a near-zero reprojection error alone is not proof of
quality. Extra correspondences make residuals and robust outlier rejection
meaningful; always inspect the projected lines in both debug images.

For more than four points, the fitter first checks a least-squares solution using
every selected landmark. It keeps that solution when every residual is coherent
within frame-scaled pixel and court-meter tolerances. RANSAC is used only when the
all-point solution contains a gross outlier. JSON and CLI output report the actual
fit method plus a whole-court quality status and warnings; inlier-only error is not
treated as sufficient evidence of a good calibration.

## Outputs

The output directory receives:

- `calibration.json`: source-frame provenance, explicit court dimensions,
  correspondences and inlier flags, forward/reverse matrices, and reprojection
  metrics;
- `calibration-overlay.jpg`: source frame with clicked labels and projected court
  boundaries, kitchen lines, net, and service centerlines; and
- `court-topdown.jpg`: source frame rectified onto the canonical court plane with
  the same geometry drawn as a visual check.

Inspect both images before trusting a calibration. Projected lines should align
with the painted court across near and far regions. Recalibrate if labels are
swapped, points were clicked on non-court objects, reprojection error is high, or
the top-down court is visibly folded or skewed.

## Scope limit

The homography represents the two-dimensional court plane. It can transform court
landmarks, player ground-contact estimates, and confirmed bounce points. Never use
it to treat an airborne ball as though it were on the court.
