# Pickleball Vision Milestones

Milestones are intentionally sequential. Exactly one milestone is current. Work
outside the current milestone requires this plan to be deliberately updated first.

## 0. Foundation — complete

Establish repository instructions, architecture documentation, a typed Python
package, an installable CLI shell, environment-based configuration, structured
logging, error conventions, and automated quality checks.

Exit criteria:

- The documented repository structure exists.
- The vision package installs in a clean environment.
- `pickleball-vision doctor` reports a valid Foundation setup.
- Tests, Ruff checks, Ruff formatting, and static type checks pass.
- No video ingestion or computer-vision behavior was implemented.

## 1. Video ingestion — complete

Read local video files through a reusable OpenCV boundary, expose source metadata,
extract a full-resolution frame by timestamp, and sample frames across the full
source duration. All commands validate their inputs and translate expected OpenCV
or filesystem failures into useful application errors.

Exit criteria:

- `pickleball-vision inspect <video>` reports the resolved source path, dimensions,
  non-integer FPS, frame count, duration in seconds, and codec when available.
- `pickleball-vision extract-frame` writes the requested valid frame without
  resizing it.
- `pickleball-vision sample-frames` selects the requested number of unique frames
  across the complete frame-index span rather than only from its beginning.
- Missing, non-file, unreadable, corrupt, invalid-timestamp, invalid-count, decode,
  and image-write failures produce stable, useful errors rather than raw OpenCV
  exceptions.
- Automated tests generate their own synthetic videos; no private footage is
  required or committed.
- Tests, Ruff checks, Ruff formatting, static type checks, and the CLI health check
  pass.
- No court calibration, detection, tracking, or product-service behavior was
  implemented by this milestone.

## 2. Court calibration — current

Manually associate at least four visible image points with named canonical
pickleball court landmarks, fit an image/court-plane homography using all valid
correspondences, persist the calibration and its provenance, and create inspectable
image-space and top-down debug artifacts.

Exit criteria:

- Court dimensions, axes, units, named landmarks, and line geometry are explicit
  and documented.
- `pickleball-vision calibrate <video> --timestamp <seconds> --output <json>`
  guides the user through visible landmark selection without requiring all outer
  corners.
- Four valid correspondences are required; additional correspondences support
  robust fitting and retain their inlier status.
- Degenerate, duplicated, out-of-frame, non-finite, unreadable, cancelled, and
  non-invertible calibration configurations produce useful typed errors.
- Calibration JSON retains source-frame provenance, correspondences, both
  homography directions, court configuration, and reprojection error in court
  meters and image pixels.
- Reusable APIs transform a court-plane image point to canonical court coordinates
  and perform the reverse transformation.
- `calibration-overlay.jpg` labels selected landmarks and projected court lines;
  `court-topdown.jpg` rectifies the selected frame and draws canonical geometry.
- Unit tests verify transformations, robust multi-point fitting, degeneracy,
  persistence, and debug rendering with synthetic data.
- No automatic court detection or later vision milestone is implemented.

## Future milestones — not current

3. Person detection
4. Primary-player isolation
5. Persistent player tracking
6. Player court-position analytics
7. Ball dataset tooling
8. Ball detector
9. Ball trajectory reconstruction
10. Rally annotation
11. Rally segmentation
12. Bounce detection
13. Contact detection
14. Hitter identification
15. Shot classification
16. Analytics engine
17. Backend productization
18. Async processing
19. Web dashboard
20. Human correction
21. AI coach

Moving a milestone to current should add measurable entry and exit criteria while
preserving completed milestone history. Do not bundle later milestones into the
current one for convenience.
