"""Command-line interface for Pickleball Vision."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence

from pickleball_vision import __version__
from pickleball_vision.config import Settings
from pickleball_vision.errors import ErrorCode, PickleballVisionError
from pickleball_vision.logging import configure_logging

EXIT_OK = 0
EXIT_INTERNAL_ERROR = 1
EXIT_USAGE_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without reading configuration or performing I/O."""

    parser = argparse.ArgumentParser(
        prog="pickleball-vision",
        description="Local, inspectable doubles-pickleball vision pipeline",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "doctor",
        help="validate Foundation configuration and report service metadata",
    )
    return parser


def _run_doctor(settings: Settings) -> int:
    logger = logging.getLogger("pickleball_vision.cli")
    logger.info(
        "foundation_check_complete",
        extra={"context": {"environment": settings.environment.value, "status": "ok"}},
    )
    report = {
        "service": "pickleball-vision",
        "status": "ok",
        "version": __version__,
        "configuration": settings.public_values(),
    }
    print(json.dumps(report, sort_keys=True))
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command and translate failures into stable process exit codes."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_OK

    try:
        settings = Settings.from_env()
        logger = configure_logging(level=settings.log_level, log_format=settings.log_format)
        if args.command == "doctor":
            return _run_doctor(settings)
        parser.error(f"unsupported command: {args.command}")
    except PickleballVisionError as error:
        print(f"error [{error.code}]: {error}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    except Exception:
        logger = logging.getLogger("pickleball_vision.cli")
        logger.exception("unexpected_error", extra={"context": {"code": ErrorCode.INTERNAL.value}})
        print(
            f"error [{ErrorCode.INTERNAL}]: an unexpected error occurred",
            file=sys.stderr,
        )
        return EXIT_INTERNAL_ERROR
