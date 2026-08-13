"""Allow ``python -m pickleball_vision`` to invoke the CLI."""

from pickleball_vision.cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
