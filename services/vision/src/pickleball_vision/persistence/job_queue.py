"""MongoDB-backed leased job coordination for the standalone analysis worker."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

from pymongo import ASCENDING, ReturnDocument
from pymongo.errors import PyMongoError

from pickleball_vision.errors import PersistenceOperationError
from pickleball_vision.persistence.models import (
    Document,
    ProcessingJobRecord,
    ProcessingJobStatus,
    processing_job_from_document,
)

ACTIVE_JOB_STATUSES = (
    ProcessingJobStatus.CLAIMED,
    ProcessingJobStatus.PLAYER_PROCESSING,
    ProcessingJobStatus.BALL_PROCESSING,
    ProcessingJobStatus.AUDIO_PROCESSING,
    ProcessingJobStatus.RALLY_PROCESSING,
    ProcessingJobStatus.EVENT_PROCESSING,
    ProcessingJobStatus.SHOT_PROCESSING,
    ProcessingJobStatus.ANALYTICS,
    ProcessingJobStatus.PUBLISHING,
)
PROCESSING_STAGES = ACTIVE_JOB_STATUSES[1:]


class _UpdateResult(Protocol):
    modified_count: int


class JobCollection(Protocol):
    """Narrow PyMongo collection surface required by the queue adapter."""

    async def find_one_and_update(
        self,
        filter: Mapping[str, object],
        update: Mapping[str, object],
        *,
        sort: Sequence[tuple[str, int]] | None = None,
        return_document: object,
    ) -> Document | None: ...

    async def update_many(
        self,
        filter: Mapping[str, object],
        update: Mapping[str, object],
    ) -> _UpdateResult: ...


class JobQueue(Protocol):
    """Ownership-checked leased queue used by the worker orchestration."""

    async def recover_exhausted_stale_jobs(
        self,
        *,
        stale_before: datetime,
        now: datetime,
        max_attempts: int,
    ) -> int: ...

    async def claim_next(
        self,
        *,
        worker_id: str,
        stale_before: datetime,
        now: datetime,
        max_attempts: int,
    ) -> ProcessingJobRecord | None: ...

    async def heartbeat(self, job_id: str, *, worker_id: str, now: datetime) -> bool: ...

    async def update_stage(
        self,
        job_id: str,
        *,
        worker_id: str,
        stage: ProcessingJobStatus,
        progress: float,
        pipeline_version: str,
        now: datetime,
    ) -> bool: ...

    async def complete(
        self,
        job_id: str,
        *,
        worker_id: str,
        result_artifact_ids: Sequence[str],
        now: datetime,
    ) -> bool: ...

    async def fail(
        self,
        job_id: str,
        *,
        worker_id: str,
        stage: str,
        error_code: str,
        error_message: str,
        now: datetime,
    ) -> bool: ...


def _driver_reason(error: PyMongoError) -> str:
    return f"MongoDB driver failure ({type(error).__name__})"


def _stale_filter(stale_before: datetime) -> Document:
    return {
        "$or": [
            {"heartbeatAt": {"$lte": stale_before}},
            {
                "heartbeatAt": {"$exists": False},
                "claimedAt": {"$lte": stale_before},
            },
        ]
    }


class MongoJobQueue:
    """Atomic single-document claims with bounded stale-lease recovery."""

    def __init__(self, collection: JobCollection) -> None:
        self._collection = collection

    async def recover_exhausted_stale_jobs(
        self,
        *,
        stale_before: datetime,
        now: datetime,
        max_attempts: int,
    ) -> int:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        query: Document = {
            "status": {"$in": [status.value for status in ACTIVE_JOB_STATUSES]},
            "attemptCount": {"$gte": max_attempts},
            **_stale_filter(stale_before),
        }
        update: Document = {
            "$set": {
                "status": ProcessingJobStatus.FAILED.value,
                "completedAt": now,
                "updatedAt": now,
                "errorCode": "WORKER_LEASE_EXHAUSTED",
                "errorMessage": "Worker lease expired and the maximum attempt count was reached",
            }
        }
        try:
            result = await self._collection.update_many(query, update)
        except PyMongoError as error:
            raise PersistenceOperationError(
                "recover_stale_jobs",
                reason=_driver_reason(error),
            ) from error
        return result.modified_count

    async def claim_next(
        self,
        *,
        worker_id: str,
        stale_before: datetime,
        now: datetime,
        max_attempts: int,
    ) -> ProcessingJobRecord | None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        attempt_available: Document = {
            "$or": [
                {"attemptCount": {"$lt": max_attempts}},
                {"attemptCount": {"$exists": False}},
            ]
        }
        query: Document = {
            "$or": [
                {
                    "status": ProcessingJobStatus.QUEUED.value,
                    **attempt_available,
                },
                {
                    "status": {"$in": [status.value for status in ACTIVE_JOB_STATUSES]},
                    "$and": [attempt_available, _stale_filter(stale_before)],
                },
            ]
        }
        update: Document = {
            "$set": {
                "status": ProcessingJobStatus.CLAIMED.value,
                "stage": ProcessingJobStatus.CLAIMED.value,
                "claimedAt": now,
                "startedAt": now,
                "heartbeatAt": now,
                "workerId": worker_id,
                "progress": 0.0,
                "updatedAt": now,
            },
            "$inc": {"attemptCount": 1},
            "$unset": {
                "completedAt": "",
                "errorCode": "",
                "errorMessage": "",
                "resultArtifactIds": "",
            },
        }
        try:
            document = await self._collection.find_one_and_update(
                query,
                update,
                sort=(("createdAt", ASCENDING),),
                return_document=ReturnDocument.AFTER,
            )
        except PyMongoError as error:
            raise PersistenceOperationError("claim_job", reason=_driver_reason(error)) from error
        if document is None:
            return None
        return processing_job_from_document(document)

    async def heartbeat(self, job_id: str, *, worker_id: str, now: datetime) -> bool:
        return await self._owned_update(
            job_id,
            worker_id=worker_id,
            update={"$set": {"heartbeatAt": now, "updatedAt": now}},
            operation="heartbeat_job",
        )

    async def update_stage(
        self,
        job_id: str,
        *,
        worker_id: str,
        stage: ProcessingJobStatus,
        progress: float,
        pipeline_version: str,
        now: datetime,
    ) -> bool:
        if stage not in ACTIVE_JOB_STATUSES:
            raise ValueError(f"{stage.value} is not an active job stage")
        if not 0.0 <= progress < 1.0:
            raise ValueError("in-progress stage progress must be in [0, 1)")
        return await self._owned_update(
            job_id,
            worker_id=worker_id,
            update={
                "$set": {
                    "status": stage.value,
                    "stage": stage.value,
                    "progress": progress,
                    "pipelineVersion": pipeline_version,
                    "heartbeatAt": now,
                    "updatedAt": now,
                }
            },
            operation="update_job_stage",
            extra_filter={"progress": {"$lte": progress}},
        )

    async def complete(
        self,
        job_id: str,
        *,
        worker_id: str,
        result_artifact_ids: Sequence[str],
        now: datetime,
    ) -> bool:
        return await self._owned_update(
            job_id,
            worker_id=worker_id,
            update={
                "$set": {
                    "status": ProcessingJobStatus.COMPLETE.value,
                    "stage": ProcessingJobStatus.COMPLETE.value,
                    "progress": 1.0,
                    "heartbeatAt": now,
                    "completedAt": now,
                    "updatedAt": now,
                    "resultArtifactIds": list(result_artifact_ids),
                },
                "$unset": {"errorCode": "", "errorMessage": ""},
            },
            operation="complete_job",
        )

    async def fail(
        self,
        job_id: str,
        *,
        worker_id: str,
        stage: str,
        error_code: str,
        error_message: str,
        now: datetime,
    ) -> bool:
        safe_message = error_message.strip()[:512] or "Analysis failed"
        return await self._owned_update(
            job_id,
            worker_id=worker_id,
            update={
                "$set": {
                    "status": ProcessingJobStatus.FAILED.value,
                    "stage": stage,
                    "heartbeatAt": now,
                    "completedAt": now,
                    "updatedAt": now,
                    "errorCode": error_code.strip()[:128] or "WORKER_FAILED",
                    "errorMessage": safe_message,
                }
            },
            operation="fail_job",
        )

    async def _owned_update(
        self,
        job_id: str,
        *,
        worker_id: str,
        update: Document,
        operation: str,
        extra_filter: Document | None = None,
    ) -> bool:
        query: Document = {
            "_id": job_id,
            "workerId": worker_id,
            "status": {"$in": [status.value for status in ACTIVE_JOB_STATUSES]},
        }
        if extra_filter is not None:
            query.update(extra_filter)
        try:
            document = await self._collection.find_one_and_update(
                query,
                update,
                return_document=ReturnDocument.AFTER,
            )
        except PyMongoError as error:
            raise PersistenceOperationError(operation, reason=_driver_reason(error)) from error
        return document is not None
