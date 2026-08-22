"""Single-concurrency MongoDB worker for developer-machine analysis."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from pickleball_vision.config import PersistenceSettings
from pickleball_vision.errors import AnalysisConfigurationError
from pickleball_vision.persistence.models import Document, ProcessingJobStatus
from pickleball_vision.persistence.mongodb import MongoPersistence
from pickleball_vision.workflows.runtime import run_configured_analysis
from pickleball_vision.workflows.settings import WorkflowSettings

LOGGER = logging.getLogger("pickleball_vision.worker")
AnalysisExecutor = Callable[..., Awaitable[Mapping[str, str]]]


class WorkerPersistence(Protocol):
    async def claim_next_processing_job(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_seconds: int,
        max_attempts: int,
    ) -> Document | None: ...

    async def heartbeat_processing_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        now: datetime,
        lease_seconds: int,
    ) -> bool: ...

    async def release_or_fail_processing_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        now: datetime,
        max_attempts: int,
        error_code: str,
        error_message: str,
    ) -> Document | None: ...

    async def fail_exhausted_stale_processing_job(
        self,
        *,
        now: datetime,
        max_attempts: int,
    ) -> Document | None: ...

    async def get_processing_job(self, job_id: str) -> Document | None: ...


def _positive_float(source: Mapping[str, str], key: str, default: float) -> float:
    raw = source.get(key, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise AnalysisConfigurationError(f"{key} must be numeric") from error
    if value <= 0:
        raise AnalysisConfigurationError(f"{key} must be positive")
    return value


def _positive_int(source: Mapping[str, str], key: str, default: int) -> int:
    raw = source.get(key, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise AnalysisConfigurationError(f"{key} must be an integer") from error
    if value < 1:
        raise AnalysisConfigurationError(f"{key} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class LocalWorkerSettings:
    """Bounded queue and lease settings for a single local worker process."""

    worker_id: str
    poll_seconds: float = 10.0
    heartbeat_seconds: float = 30.0
    lease_seconds: int = 180
    max_attempts: int = 2

    def __post_init__(self) -> None:
        if not self.worker_id.strip():
            raise AnalysisConfigurationError("WORKER_ID must not be empty")
        if self.poll_seconds <= 0 or self.heartbeat_seconds <= 0:
            raise AnalysisConfigurationError("worker polling intervals must be positive")
        if self.lease_seconds <= self.heartbeat_seconds * 2:
            raise AnalysisConfigurationError(
                "WORKER_LEASE_SECONDS must exceed twice WORKER_HEARTBEAT_SECONDS"
            )
        if self.max_attempts < 1:
            raise AnalysisConfigurationError("WORKER_MAX_ATTEMPTS must be positive")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> LocalWorkerSettings:
        source = os.environ if environ is None else environ
        default_id = f"{socket.gethostname()}-{os.getpid()}"
        return cls(
            worker_id=source.get("WORKER_ID", default_id).strip(),
            poll_seconds=_positive_float(source, "WORKER_POLL_SECONDS", 10.0),
            heartbeat_seconds=_positive_float(source, "WORKER_HEARTBEAT_SECONDS", 30.0),
            lease_seconds=_positive_int(source, "WORKER_LEASE_SECONDS", 180),
            max_attempts=_positive_int(source, "WORKER_MAX_ATTEMPTS", 2),
        )


class LocalAnalysisWorker:
    """Claim and execute at most one MongoDB-coordinated match at a time."""

    def __init__(
        self,
        persistence: WorkerPersistence,
        *,
        settings: LocalWorkerSettings,
        executor: AnalysisExecutor = run_configured_analysis,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._persistence = persistence
        self._settings = settings
        self._executor = executor
        self._now = now or (lambda: datetime.now(UTC))

    async def run_once(self) -> bool:
        """Run one available job, returning false when the queue is empty."""

        now = self._now()
        await self._persistence.fail_exhausted_stale_processing_job(
            now=now,
            max_attempts=self._settings.max_attempts,
        )
        document = await self._persistence.claim_next_processing_job(
            worker_id=self._settings.worker_id,
            now=now,
            lease_seconds=self._settings.lease_seconds,
            max_attempts=self._settings.max_attempts,
        )
        if document is None:
            return False
        job_id = document.get("jobId")
        match_id = document.get("matchId")
        if not isinstance(job_id, str) or not isinstance(match_id, str):
            raise AnalysisConfigurationError("claimed processing job has invalid identifiers")
        LOGGER.info(
            "local_worker_job_claimed",
            extra={
                "context": {
                    "jobId": job_id,
                    "matchId": match_id,
                    "workerId": self._settings.worker_id,
                }
            },
        )
        stop_heartbeat = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._heartbeat(job_id, stop_heartbeat),
            name=f"heartbeat-{job_id}",
        )
        try:
            result = await self._executor(job_id=job_id, match_id=match_id)
            status = result.get("status")
            if status not in {
                ProcessingJobStatus.COMPLETE.value,
                ProcessingJobStatus.FAILED.value,
            }:
                await self._release_or_fail(
                    job_id,
                    code="WORKER_INCOMPLETE",
                    message="Analysis worker returned without a terminal result",
                )
        except Exception as error:
            LOGGER.exception(
                "local_worker_job_failed",
                extra={"context": {"jobId": job_id, "exceptionType": type(error).__name__}},
            )
            await self._release_or_fail(
                job_id,
                code="WORKER_EXECUTION_FAILED",
                message=f"Local analysis worker failed ({type(error).__name__})",
            )
        finally:
            stop_heartbeat.set()
            await heartbeat
        return True

    async def run_forever(self) -> None:
        """Poll indefinitely while processing jobs serially."""

        LOGGER.info(
            "local_worker_started",
            extra={"context": {"workerId": self._settings.worker_id, "concurrency": 1}},
        )
        while True:
            processed = await self.run_once()
            if not processed:
                await asyncio.sleep(self._settings.poll_seconds)

    async def _heartbeat(self, job_id: str, stop: asyncio.Event) -> None:
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._settings.heartbeat_seconds)
                return
            except TimeoutError:
                retained = await self._persistence.heartbeat_processing_job(
                    job_id,
                    worker_id=self._settings.worker_id,
                    now=self._now(),
                    lease_seconds=self._settings.lease_seconds,
                )
                if not retained:
                    LOGGER.error(
                        "local_worker_lease_lost",
                        extra={"context": {"jobId": job_id}},
                    )
                    return

    async def _release_or_fail(self, job_id: str, *, code: str, message: str) -> None:
        await self._persistence.release_or_fail_processing_job(
            job_id,
            worker_id=self._settings.worker_id,
            now=self._now(),
            max_attempts=self._settings.max_attempts,
            error_code=code,
            error_message=message,
        )


async def run_local_worker(*, once: bool = False) -> int:
    """Assemble hosted adapters and run the developer-machine worker."""

    persistence_settings = PersistenceSettings.from_env()
    WorkflowSettings.from_env()
    worker_settings = LocalWorkerSettings.from_env()
    persistence = await MongoPersistence.connect_from_settings(persistence_settings)
    try:
        await persistence.initialize_indexes()
        worker = LocalAnalysisWorker(persistence, settings=worker_settings)
        if once:
            await worker.run_once()
            return 0
        await worker.run_forever()
    finally:
        await persistence.close()
    return 0
