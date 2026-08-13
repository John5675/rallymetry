"""Structured logging configured explicitly by executable entry points."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TextIO

from pickleball_vision.config import LogFormat


def _json_default(value: object) -> str:
    if isinstance(value, (Enum, Path)):
        return str(value)
    return repr(value)


class JsonFormatter(logging.Formatter):
    """Render one machine-readable JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if isinstance(context, Mapping):
            payload["context"] = dict(context)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=_json_default, sort_keys=True)


class ConsoleFormatter(logging.Formatter):
    """Render structured records in a concise human-readable form."""

    def format(self, record: logging.LogRecord) -> str:
        message = f"{record.levelname:<8} {record.name}: {record.getMessage()}"
        context = getattr(record, "context", None)
        if isinstance(context, Mapping) and context:
            message = (
                f"{message} {json.dumps(dict(context), default=_json_default, sort_keys=True)}"
            )
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        return message


def configure_logging(
    *,
    level: str,
    log_format: LogFormat,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure process logging and return the service logger."""

    handler = logging.StreamHandler(stream)
    formatter: logging.Formatter = (
        JsonFormatter() if log_format is LogFormat.JSON else ConsoleFormatter()
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    return logging.getLogger("pickleball_vision")
