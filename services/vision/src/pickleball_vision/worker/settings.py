"""Environment-backed settings for one outbound-only analysis worker."""

from __future__ import annotations

import os
import socket
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pickleball_vision.errors import WorkerConfigurationError

_PREFIX = "PICKLEBALL_VISION_WORKER_"


def _positive_float(source: Mapping[str, str], name: str, default: float) -> float:
    raw = source.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise WorkerConfigurationError(f"{name} must be numeric") from error
    if value <= 0:
        raise WorkerConfigurationError(f"{name} must be positive")
    return value


def _positive_int(source: Mapping[str, str], name: str, default: int) -> int:
    raw = source.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise WorkerConfigurationError(f"{name} must be an integer") from error
    if value < 1:
        raise WorkerConfigurationError(f"{name} must be at least 1")
    return value


def default_worker_id() -> str:
    """Generate a non-secret process identity retained for lease auditing."""

    host = socket.gethostname().split(".", maxsplit=1)[0] or "worker"
    return f"{host}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True, slots=True)
class WorkerSettings:
    """Small-scale polling, lease, and trusted-plan configuration."""

    worker_id: str
    pipeline_plan_path: Path
    work_root: Path = Path("output/worker")
    poll_interval_seconds: float = 5.0
    lease_timeout_seconds: float = 180.0
    heartbeat_interval_seconds: float = 20.0
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if not self.worker_id.strip():
            raise WorkerConfigurationError("worker_id must not be empty")
        if self.poll_interval_seconds <= 0:
            raise WorkerConfigurationError("poll interval must be positive")
        if self.lease_timeout_seconds <= 0:
            raise WorkerConfigurationError("lease timeout must be positive")
        if self.heartbeat_interval_seconds <= 0:
            raise WorkerConfigurationError("heartbeat interval must be positive")
        if self.heartbeat_interval_seconds >= self.lease_timeout_seconds / 2:
            raise WorkerConfigurationError(
                "heartbeat interval must be less than half the lease timeout"
            )
        if self.max_attempts < 1:
            raise WorkerConfigurationError("max attempts must be at least 1")

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        worker_id: str | None = None,
        pipeline_plan_path: Path | None = None,
        poll_interval_seconds: float | None = None,
    ) -> WorkerSettings:
        source = os.environ if environ is None else environ
        plan = pipeline_plan_path
        if plan is None:
            raw_plan = source.get(f"{_PREFIX}PIPELINE_PLAN", "").strip()
            if not raw_plan:
                raise WorkerConfigurationError(
                    f"{_PREFIX}PIPELINE_PLAN or --pipeline-plan is required"
                )
            plan = Path(raw_plan).expanduser()
        root_raw = source.get(f"{_PREFIX}WORK_ROOT", "output/worker").strip()
        if not root_raw:
            raise WorkerConfigurationError(f"{_PREFIX}WORK_ROOT must not be empty")
        poll = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else _positive_float(source, f"{_PREFIX}POLL_INTERVAL_SECONDS", 5.0)
        )
        return cls(
            worker_id=worker_id or source.get(f"{_PREFIX}ID", "").strip() or default_worker_id(),
            pipeline_plan_path=plan,
            work_root=Path(root_raw).expanduser(),
            poll_interval_seconds=poll,
            lease_timeout_seconds=_positive_float(
                source,
                f"{_PREFIX}LEASE_TIMEOUT_SECONDS",
                180.0,
            ),
            heartbeat_interval_seconds=_positive_float(
                source,
                f"{_PREFIX}HEARTBEAT_INTERVAL_SECONDS",
                20.0,
            ),
            max_attempts=_positive_int(source, f"{_PREFIX}MAX_ATTEMPTS", 3),
        )
