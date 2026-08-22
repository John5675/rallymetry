"""Environment-backed settings for on-demand match analysis."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from pickleball_vision.errors import AnalysisConfigurationError

DEFAULT_WORKFLOW_PLAN = "pro"
DEFAULT_WORKFLOW_TIMEOUT_SECONDS = 21_600
DEFAULT_WORKFLOW_MAX_RETRIES = 1
DEFAULT_WORKFLOW_RETRY_WAIT_MS = 30_000
DEFAULT_YOUTUBE_MAX_DURATION_SECONDS = 7_200
DEFAULT_YOUTUBE_MAX_BYTES = 4_000_000_000
RENDER_WORKFLOW_PLANS = frozenset({"flex", "starter", "standard", "pro", "pro_plus", "pro_max"})


def _positive_int(source: Mapping[str, str], key: str, default: int) -> int:
    raw = source.get(key, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise AnalysisConfigurationError(f"{key} must be an integer") from error
    if value < 1:
        raise AnalysisConfigurationError(f"{key} must be positive")
    return value


def _optional_service_url(source: Mapping[str, str], key: str) -> str | None:
    value = source.get(key, "").strip().rstrip("/")
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise AnalysisConfigurationError(f"{key} must be an HTTP(S) service URL")
    if parsed.username is not None or parsed.password is not None:
        raise AnalysisConfigurationError(f"{key} must not contain credentials")
    return value


@dataclass(frozen=True, slots=True)
class WorkflowSettings:
    """Trusted workflow configuration; task inputs contain identifiers only."""

    pipeline_config_path: Path
    temp_root: Path = Path("/tmp/rallymetry")
    model_device: str = "cpu"
    youtube_max_duration_seconds: int = DEFAULT_YOUTUBE_MAX_DURATION_SECONDS
    youtube_max_bytes: int = DEFAULT_YOUTUBE_MAX_BYTES
    youtube_pot_provider_url: str | None = None

    def __post_init__(self) -> None:
        if not self.model_device.strip():
            raise AnalysisConfigurationError("MODEL_DEVICE must not be empty")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> WorkflowSettings:
        source = os.environ if environ is None else environ
        raw_plan = source.get("PIPELINE_CONFIG", "").strip()
        if not raw_plan:
            raise AnalysisConfigurationError("PIPELINE_CONFIG is required")
        raw_root = source.get("WORKFLOW_TEMP_DIR", "/tmp/rallymetry").strip()
        if not raw_root:
            raise AnalysisConfigurationError("WORKFLOW_TEMP_DIR must not be empty")
        return cls(
            pipeline_config_path=Path(raw_plan).expanduser(),
            temp_root=Path(raw_root).expanduser(),
            model_device=source.get("MODEL_DEVICE", "cpu").strip().lower(),
            youtube_max_duration_seconds=_positive_int(
                source,
                "YOUTUBE_MAX_DURATION_SECONDS",
                DEFAULT_YOUTUBE_MAX_DURATION_SECONDS,
            ),
            youtube_max_bytes=_positive_int(
                source,
                "YOUTUBE_MAX_BYTES",
                DEFAULT_YOUTUBE_MAX_BYTES,
            ),
            youtube_pot_provider_url=_optional_service_url(
                source,
                "YOUTUBE_POT_PROVIDER_URL",
            ),
        )


def workflow_task_plan(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = source.get("RENDER_WORKFLOW_PLAN", DEFAULT_WORKFLOW_PLAN).strip()
    if value not in RENDER_WORKFLOW_PLANS:
        raise AnalysisConfigurationError(
            "RENDER_WORKFLOW_PLAN must be a supported Render instance type"
        )
    return value


def workflow_timeout_seconds(environ: Mapping[str, str] | None = None) -> int:
    source = os.environ if environ is None else environ
    value = _positive_int(
        source,
        "RENDER_WORKFLOW_TIMEOUT_SECONDS",
        DEFAULT_WORKFLOW_TIMEOUT_SECONDS,
    )
    if not 30 <= value <= 86_400:
        raise AnalysisConfigurationError(
            "RENDER_WORKFLOW_TIMEOUT_SECONDS must be between 30 and 86400"
        )
    return value
