from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime

from pickleball_vision.local_worker import LocalAnalysisWorker, LocalWorkerSettings
from pickleball_vision.persistence.models import Document

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


class FakeWorkerPersistence:
    def __init__(self, jobs: list[Document]) -> None:
        self.jobs = jobs
        self.claimed = False
        self.heartbeats = 0
        self.releases: list[dict[str, object]] = []
        self.exhausted_checks = 0

    async def claim_next_processing_job(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_seconds: int,
        max_attempts: int,
    ) -> Document | None:
        del now, lease_seconds, max_attempts
        if self.claimed or not self.jobs:
            return None
        self.claimed = True
        document = dict(self.jobs[0])
        document["workerId"] = worker_id
        return document

    async def heartbeat_processing_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        now: datetime,
        lease_seconds: int,
    ) -> bool:
        del job_id, worker_id, now, lease_seconds
        self.heartbeats += 1
        return True

    async def release_or_fail_processing_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        now: datetime,
        max_attempts: int,
        error_code: str,
        error_message: str,
    ) -> Document | None:
        self.releases.append(
            {
                "jobId": job_id,
                "workerId": worker_id,
                "now": now,
                "maxAttempts": max_attempts,
                "errorCode": error_code,
                "errorMessage": error_message,
            }
        )
        return dict(self.jobs[0])

    async def fail_exhausted_stale_processing_job(
        self,
        *,
        now: datetime,
        max_attempts: int,
    ) -> Document | None:
        del now, max_attempts
        self.exhausted_checks += 1
        return None

    async def get_processing_job(self, job_id: str) -> Document | None:
        del job_id
        return dict(self.jobs[0]) if self.jobs else None


def _job() -> Document:
    return {"jobId": "job_local", "matchId": "match_local", "status": "QUEUED"}


def test_local_worker_runs_one_claimed_job_and_heartbeats() -> None:
    persistence = FakeWorkerPersistence([_job()])
    calls: list[tuple[str, str]] = []

    async def execute(*, job_id: str, match_id: str) -> Mapping[str, str]:
        calls.append((job_id, match_id))
        await asyncio.sleep(0.03)
        return {"status": "COMPLETE"}

    worker = LocalAnalysisWorker(
        persistence,
        settings=LocalWorkerSettings(
            worker_id="mac-worker",
            heartbeat_seconds=0.01,
            lease_seconds=1,
        ),
        executor=execute,
        now=lambda: NOW,
    )

    processed = asyncio.run(worker.run_once())

    assert processed is True
    assert calls == [("job_local", "match_local")]
    assert persistence.heartbeats >= 1
    assert persistence.releases == []
    assert persistence.exhausted_checks == 1


def test_local_worker_releases_failed_execution_for_bounded_retry() -> None:
    persistence = FakeWorkerPersistence([_job()])

    async def fail(*, job_id: str, match_id: str) -> Mapping[str, str]:
        del job_id, match_id
        raise RuntimeError("private failure detail")

    worker = LocalAnalysisWorker(
        persistence,
        settings=LocalWorkerSettings(worker_id="mac-worker"),
        executor=fail,
        now=lambda: NOW,
    )

    assert asyncio.run(worker.run_once()) is True
    assert persistence.releases[0]["errorCode"] == "WORKER_EXECUTION_FAILED"
    assert "private failure detail" not in str(persistence.releases[0]["errorMessage"])


def test_local_worker_returns_false_for_empty_queue() -> None:
    persistence = FakeWorkerPersistence([])
    worker = LocalAnalysisWorker(
        persistence,
        settings=LocalWorkerSettings(worker_id="mac-worker"),
        now=lambda: NOW,
    )

    assert asyncio.run(worker.run_once()) is False
