# Vision service

This Python package is the local computer-vision pipeline. It currently provides
typed executable infrastructure, reusable OpenCV video ingestion, and manual
multi-point court calibration. Automatic detection, tracking, and analytics remain
unimplemented.

```bash
uv sync --locked --extra dev
uv run pickleball-vision doctor
uv run pickleball-vision inspect /absolute/path/to/match.mp4
uv run pickleball-vision calibrate /absolute/path/to/match.mp4 \
  --timestamp 30.5 \
  --output ../../output/calibration.json
```

Calibration uses a local OpenCV window. See `docs/court-calibration.md` from the
repository root for landmark definitions and interaction controls.

The package uses a `src/` layout so tests exercise the installed package rather
than accidentally importing a checkout-relative module.
