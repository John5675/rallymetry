"""Single-concurrency worker lifecycle, leases, publication, and failure capture."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from pickleball_vision.errors import WorkerError, WorkerLeaseLostError
from pickleball_vision.persistence.artifacts import ArtifactStore
from pickleball_vision.persistence.job_queue import JobQueue
from pickleball_vision.persistence.models import (
    AnalyticsRecord,
    ArtifactPutRequest,
    ArtifactRecord,
    PlayerRecord,
    ProcessingJobRecord,
    ProcessingJobStatus,
    StructuredCollection,
    StructuredDomainRecord,
)
from pickleball_vision.worker.models import PipelineRunResult
from pickleball_vision.worker.pipeline import PipelineRunner
from pickleball_vision.worker.settings import WorkerSettings
from pickleball_vision.worker.source import SourceMediaStager

logger = logging.getLogger("pickleball_vision.worker")


class WorkerPersistence(Protocol):
    async def save_players(self, records: Sequence[PlayerRecord]) -> None: ...

    async def save_rallies(self, records: Sequence[StructuredDomainRecord]) -> None: ...

    async def save_contacts(self, records: Sequence[StructuredDomainRecord]) -> None: ...

    async def save_bounces(self, records: Sequence[StructuredDomainRecord]) -> None: ...

    async def save_shots(self, records: Sequence[StructuredDomainRecord]) -> None: ...

    async def save_analytics(self, record: AnalyticsRecord) -> None: ...

    async def save_artifact(self, record: ArtifactRecord) -> None: ...


def _now() -> datetime:
    return datetime.now(UTC)


class AnalysisWorker:
    """Process at most one match at a time using an ownership-checked MongoDB lease."""

    def __init__(
        self,
        *,
        settings: WorkerSettings,
        pipeline_version: str,
        queue: JobQueue,
        persistence: WorkerPersistence,
        source_stager: SourceMediaStager,
        pipeline_runner: PipelineRunner,
        artifact_store: ArtifactStore,
    ) -> None:
        self._settings = settings
        self._pipeline_version = pipeline_version
        self._queue = queue
        self._persistence = persistence
        self._source_stager = source_stager
        self._pipeline_runner = pipeline_runner
        self._artifact_store = artifact_store

    async def run_once(self) -> bool:
        """Claim and process one eligible job; return false when the queue is empty."""

        now = _now()
        stale_before = now - timedelta(seconds=self._settings.lease_timeout_seconds)
        recovered = await self._queue.recover_exhausted_stale_jobs(
            stale_before=stale_before,
            now=now,
            max_attempts=self._settings.max_attempts,
        )
        if recovered:
            logger.warning("worker_stale_jobs_exhausted", extra={"context": {"count": recovered}})
        job = await self._queue.claim_next(
            worker_id=self._settings.worker_id,
            stale_before=stale_before,
            now=now,
            max_attempts=self._settings.max_attempts,
        )
        if job is None:
            return False
        logger.info(
            "worker_job_claimed",
            extra={
                "context": {
                    "jobId": job.job_id,
                    "matchId": job.match_id,
                    "attempt": job.attempt_count,
                    "workerId": self._settings.worker_id,
                }
            },
        )
        await self._process_claimed(job)
        return True

    async def run_forever(self, stop: asyncio.Event | None = None) -> None:
        """Poll outbound dependencies serially until stopped."""

        stop_event = stop or asyncio.Event()
        while not stop_event.is_set():
            processed = await self.run_once()
            if processed:
                continue
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self._settings.poll_interval_seconds,
                )

    async def _process_claimed(self, job: ProcessingJobRecord) -> None:
        current_stage = ProcessingJobStatus.CLAIMED.value
        stop_heartbeat = asyncio.Event()
        lease_lost = asyncio.Event()
        heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(job.job_id, stop_heartbeat, lease_lost)
        )
        self._settings.work_root.mkdir(parents=True, exist_ok=True)

        async def on_stage(stage: ProcessingJobStatus, progress: float) -> None:
            nonlocal current_stage
            if lease_lost.is_set():
                raise WorkerLeaseLostError(job.job_id)
            updated = await self._queue.update_stage(
                job.job_id,
                worker_id=self._settings.worker_id,
                stage=stage,
                progress=progress,
                pipeline_version=self._pipeline_version,
                now=_now(),
            )
            if not updated:
                lease_lost.set()
                raise WorkerLeaseLostError(job.job_id)
            current_stage = stage.value

        try:
            initialized = await self._queue.update_stage(
                job.job_id,
                worker_id=self._settings.worker_id,
                stage=ProcessingJobStatus.CLAIMED,
                progress=0.0,
                pipeline_version=self._pipeline_version,
                now=_now(),
            )
            if not initialized:
                raise WorkerLeaseLostError(job.job_id)
            with tempfile.TemporaryDirectory(
                prefix=f"{job.job_id}-",
                dir=self._settings.work_root,
            ) as temporary:
                workspace = Path(temporary)
                source_path = await self._source_stager.stage(job, workspace=workspace)
                result = await self._pipeline_runner.run(
                    job,
                    source_path=source_path,
                    workspace=workspace,
                    on_stage=on_stage,
                )
                await on_stage(ProcessingJobStatus.PUBLISHING, 0.9)
                artifact_ids = await self._publish(job, result)
                if lease_lost.is_set():
                    raise WorkerLeaseLostError(job.job_id)
                completed = await self._queue.complete(
                    job.job_id,
                    worker_id=self._settings.worker_id,
                    result_artifact_ids=artifact_ids,
                    now=_now(),
                )
                if not completed:
                    raise WorkerLeaseLostError(job.job_id)
                logger.info(
                    "worker_job_completed",
                    extra={"context": {"jobId": job.job_id, "artifactCount": len(artifact_ids)}},
                )
        except WorkerLeaseLostError:
            logger.warning(
                "worker_job_lease_lost",
                extra={"context": {"jobId": job.job_id, "stage": current_stage}},
            )
        except Exception as error:
            if isinstance(error, WorkerError):
                error_code = error.job_error_code
                message = str(error)
            else:
                error_code = "WORKER_UNEXPECTED"
                message = f"Unexpected {type(error).__name__} during {current_stage}"
                logger.exception(
                    "worker_job_unexpected_failure",
                    extra={"context": {"jobId": job.job_id, "stage": current_stage}},
                )
            failed = await self._queue.fail(
                job.job_id,
                worker_id=self._settings.worker_id,
                stage=current_stage,
                error_code=error_code,
                error_message=message,
                now=_now(),
            )
            logger.error(
                "worker_job_failed",
                extra={
                    "context": {
                        "jobId": job.job_id,
                        "stage": current_stage,
                        "errorCode": error_code,
                        "failureRecorded": failed,
                    }
                },
            )
        finally:
            stop_heartbeat.set()
            await heartbeat_task

    async def _heartbeat_loop(
        self,
        job_id: str,
        stop: asyncio.Event,
        lease_lost: asyncio.Event,
    ) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=self._settings.heartbeat_interval_seconds,
                )
                continue
            except TimeoutError:
                pass
            try:
                updated = await self._queue.heartbeat(
                    job_id,
                    worker_id=self._settings.worker_id,
                    now=_now(),
                )
            except Exception:
                logger.exception(
                    "worker_heartbeat_failed",
                    extra={"context": {"jobId": job_id}},
                )
                lease_lost.set()
                return
            if not updated:
                lease_lost.set()
                return

    async def _publish(
        self,
        job: ProcessingJobRecord,
        result: PipelineRunResult,
    ) -> tuple[str, ...]:
        artifact_ids: list[str] = []
        for publication in result.artifacts:
            artifact = await self._artifact_store.put(
                ArtifactPutRequest(
                    source_path=publication.source_path,
                    artifact_type=publication.artifact_type,
                    category=publication.category,
                    match_id=job.match_id,
                    access=publication.access,
                    pipeline_version=self._pipeline_version,
                )
            )
            await self._persistence.save_artifact(artifact)
            artifact_ids.append(artifact.artifact_id)
        if result.players:
            await self._persistence.save_players(result.players)
        for collection, records in result.structured.items():
            if collection is StructuredCollection.RALLIES:
                await self._persistence.save_rallies(records)
            elif collection is StructuredCollection.CONTACTS:
                await self._persistence.save_contacts(records)
            elif collection is StructuredCollection.BOUNCES:
                await self._persistence.save_bounces(records)
            elif collection is StructuredCollection.SHOTS:
                await self._persistence.save_shots(records)
        if result.analytics is not None:
            await self._persistence.save_analytics(result.analytics)
        return tuple(artifact_ids)
