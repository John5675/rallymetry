# Vision service

This Python package is the local computer-vision pipeline. It currently provides
typed executable infrastructure, reusable OpenCV video ingestion, manual
multi-point court calibration, pretrained person detection, and court-aware manual
primary-player isolation. Full persistent tracking and analytics remain
unimplemented.

```bash
uv sync --locked --extra dev
uv run pickleball-vision doctor
uv run pickleball-vision inspect /absolute/path/to/match.mp4
uv run pickleball-vision calibrate /absolute/path/to/match.mp4 \
  --timestamp 30.5 \
  --output ../../output/calibration.json
uv run pickleball-vision detect-people /absolute/path/to/match.mp4 \
  --calibration ../../output/calibration.json \
  --output-dir ../../output/person-detection
uv run pickleball-vision isolate-players /absolute/path/to/match.mp4 \
  --detections ../../output/person-detection/detections.json \
  --calibration ../../output/calibration.json \
  --timestamp 30.5 \
  --output-dir ../../output/player-isolation
```

Calibration uses a local OpenCV window. See `docs/court-calibration.md` from the
repository root for landmark definitions and interaction controls.
Primary-player assignment also uses a local OpenCV window; see
`docs/primary-player-isolation.md` for controls, uncertainty, and artifact
contracts.

The package uses a `src/` layout so tests exercise the installed package rather
than accidentally importing a checkout-relative module.
