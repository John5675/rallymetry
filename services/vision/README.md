# Vision service

This Python package is the future local computer-vision pipeline. During the
Foundation milestone it contains only executable infrastructure: typed
environment configuration, structured logging, application errors, and a CLI
health check.

```bash
uv sync --locked --extra dev
uv run pickleball-vision doctor
```

The package uses a `src/` layout so tests exercise the installed package rather
than accidentally importing a checkout-relative module.
