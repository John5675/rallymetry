"""Environment-backed API settings with no provider I/O."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from pickleball_vision.config import PersistenceSettings
from pickleball_vision.errors import ConfigurationError

DEFAULT_CORS_ORIGINS = ("http://localhost:5173",)


def _parse_cors_origins(raw: str) -> tuple[str, ...]:
    origins = tuple(
        dict.fromkeys(item.strip().rstrip("/") for item in raw.split(",") if item.strip())
    )
    if not origins:
        raise ConfigurationError(
            "CORS_ORIGINS must contain at least one comma-separated origin",
            setting="CORS_ORIGINS",
        )
    if "*" in origins and len(origins) > 1:
        raise ConfigurationError(
            "CORS_ORIGINS wildcard must be used alone",
            setting="CORS_ORIGINS",
        )
    for origin in origins:
        if origin == "*":
            continue
        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ConfigurationError(
                f"CORS_ORIGINS contains invalid origin {origin!r}",
                setting="CORS_ORIGINS",
            )
    return origins


@dataclass(frozen=True, slots=True)
class ApiSettings:
    """Validated API configuration and nested hosted-storage settings."""

    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    persistence: PersistenceSettings = field(default_factory=PersistenceSettings)
    render_api_key: str | None = None
    render_workflow_task: str | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ApiSettings:
        source = os.environ if environ is None else environ
        raw_origins = source.get("CORS_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS))
        api_key = source.get("RENDER_API_KEY", "").strip() or None
        workflow_task = source.get("RENDER_WORKFLOW_TASK", "").strip() or None
        if (api_key is None) != (workflow_task is None):
            missing = "RENDER_WORKFLOW_TASK" if workflow_task is None else "RENDER_API_KEY"
            raise ConfigurationError(
                "RENDER_API_KEY and RENDER_WORKFLOW_TASK must be configured together",
                setting=missing,
            )
        if workflow_task is not None and (
            workflow_task.count("/") != 1 or workflow_task.startswith("/")
        ):
            raise ConfigurationError(
                "RENDER_WORKFLOW_TASK must use {workflow-slug}/{task-name}",
                setting="RENDER_WORKFLOW_TASK",
            )
        return cls(
            cors_origins=_parse_cors_origins(raw_origins),
            persistence=PersistenceSettings.from_env(source),
            render_api_key=api_key,
            render_workflow_task=workflow_task,
        )

    def public_values(self) -> dict[str, object]:
        return {
            "corsOrigins": list(self.cors_origins),
            "persistence": self.persistence.public_values(),
            "renderWorkflowConfigured": self.render_api_key is not None,
        }
