# Vision service

This Python package is the local computer-vision pipeline. It currently provides
typed executable infrastructure and reusable OpenCV video ingestion. Detection,
calibration, tracking, and analytics remain unimplemented.

```bash
uv sync --locked --extra dev
uv run pickleball-vision doctor
uv run pickleball-vision inspect /absolute/path/to/match.mp4
```

The package uses a `src/` layout so tests exercise the installed package rather
than accidentally importing a checkout-relative module.
