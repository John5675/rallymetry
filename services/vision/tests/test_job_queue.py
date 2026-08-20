from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

from pickleball_vision.persistence.job_queue import MongoJobQueue
from pickleball_vision.persistence.models import Document, ProcessingJobRecord, ProcessingJobStatus

NOW = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)


def _matches(document: Document, query: Mapping[str, object]) -> bool:
    for key, expected in query.items():
        if key == "$or":
            assert isinstance(expected, list)
            if not any(_matches(document, item) for item in expected if isinstance(item, Mapping)):
                return False
            continue
        if key == "$and":
            assert isinstance(expected, list)
            if not all(_matches(document, item) for item in expected if isinstance(item, Mapping)):
                return False
            continue
        actual = document.get(key)
        if isinstance(expected, Mapping):
            for operator, operand in expected.items():
                if operator == "$in" and actual not in operand:
                    return False
                if operator == "$lt" and not (isinstance(actual, int) and actual < operand):
                    return False
                if operator == "$lte" and not (actual is not None and actual <= operand):
                    return False
                if operator == "$gte" and not (isinstance(actual, int) and actual >= operand):
                    return False
                if operator == "$exists" and (key in document) is not operand:
                    return False
            continue
        if actual != expected:
            return False
    return True


class UpdateResult:
    def __init__(self, modified_count: int) -> None:
        self.modified_count = modified_count


class AtomicJobCollection:
    def __init__(self, documents: Sequence[Document]) -> None:
        self.documents = {str(document["_id"]): dict(document) for document in documents}
        self.lock = asyncio.Lock()

    @staticmethod
    def _apply(document: Document, update: Mapping[str, object]) -> None:
        set_values = update.get("$set", {})
        increments = update.get("$inc", {})
        unset_values = update.get("$unset", {})
        assert isinstance(set_values, Mapping)
        assert isinstance(increments, Mapping)
        assert isinstance(unset_values, Mapping)
        document.update(set_values)
        for key, value in increments.items():
            assert isinstance(key, str) and isinstance(value, int)
            current = document.get(key, 0)
            assert isinstance(current, int)
            document[key] = current + value
        for key in unset_values:
            assert isinstance(key, str)
            document.pop(key, None)

    async def find_one_and_update(
        self,
        filter: Mapping[str, object],
        update: Mapping[str, object],
        *,
        sort: Sequence[tuple[str, int]] | None = None,
        return_document: object,
    ) -> Document | None:
        del return_document
        async with self.lock:
            candidates = [item for item in self.documents.values() if _matches(item, filter)]
            if sort:
                key, direction = sort[0]
                candidates.sort(key=lambda item: str(item.get(key)), reverse=direction < 0)
            if not candidates:
                return None
            document = candidates[0]
            self._apply(document, update)
            return dict(document)

    async def update_many(
        self,
        filter: Mapping[str, object],
        update: Mapping[str, object],
    ) -> UpdateResult:
        async with self.lock:
            matches = [item for item in self.documents.values() if _matches(item, filter)]
            for document in matches:
                self._apply(document, update)
            return UpdateResult(len(matches))


def queued_job(job_id: str = "job-1") -> ProcessingJobRecord:
    return ProcessingJobRecord(
        job_id=job_id,
        match_id="match-1",
        job_type="analyze_match",
        created_at=NOW,
        updated_at=NOW,
    )


def test_atomic_claim_prevents_two_workers_from_owning_one_job() -> None:
    collection = AtomicJobCollection([queued_job().to_document()])
    queue = MongoJobQueue(collection)

    async def claim_both() -> list[ProcessingJobRecord | None]:
        return list(
            await asyncio.gather(
                queue.claim_next(
                    worker_id="worker-a",
                    stale_before=NOW - timedelta(minutes=5),
                    now=NOW,
                    max_attempts=3,
                ),
                queue.claim_next(
                    worker_id="worker-b",
                    stale_before=NOW - timedelta(minutes=5),
                    now=NOW,
                    max_attempts=3,
                ),
            )
        )

    claims = asyncio.run(claim_both())

    claimed = [claim for claim in claims if claim is not None]
    assert len(claimed) == 1
    assert claimed[0].status is ProcessingJobStatus.CLAIMED
    assert claimed[0].attempt_count == 1
    assert collection.documents["job-1"]["workerId"] in {"worker-a", "worker-b"}


def test_claim_migrates_pre_worker_queued_record_without_attempt_count() -> None:
    document = queued_job().to_document()
    document.pop("attemptCount")
    collection = AtomicJobCollection([document])
    queue = MongoJobQueue(collection)

    claimed = asyncio.run(
        queue.claim_next(
            worker_id="worker-a",
            stale_before=NOW - timedelta(minutes=5),
            now=NOW,
            max_attempts=3,
        )
    )

    assert claimed is not None
    assert claimed.attempt_count == 1


def test_stale_lease_is_reclaimed_and_previous_owner_cannot_write() -> None:
    stale_time = NOW - timedelta(minutes=10)
    document = queued_job().to_document()
    document.update(
        {
            "status": ProcessingJobStatus.BALL_PROCESSING.value,
            "stage": ProcessingJobStatus.BALL_PROCESSING.value,
            "workerId": "crashed-worker",
            "claimedAt": stale_time,
            "heartbeatAt": stale_time,
            "attemptCount": 1,
        }
    )
    collection = AtomicJobCollection([document])
    queue = MongoJobQueue(collection)

    reclaimed = asyncio.run(
        queue.claim_next(
            worker_id="replacement-worker",
            stale_before=NOW - timedelta(minutes=1),
            now=NOW,
            max_attempts=3,
        )
    )
    old_owner_updated = asyncio.run(queue.heartbeat("job-1", worker_id="crashed-worker", now=NOW))

    assert reclaimed is not None
    assert reclaimed.worker_id == "replacement-worker"
    assert reclaimed.attempt_count == 2
    assert old_owner_updated is False


def test_exhausted_stale_job_is_failed_instead_of_retried_forever() -> None:
    stale_time = NOW - timedelta(minutes=10)
    document = queued_job().to_document()
    document.update(
        {
            "status": ProcessingJobStatus.RALLY_PROCESSING.value,
            "stage": ProcessingJobStatus.RALLY_PROCESSING.value,
            "workerId": "crashed-worker",
            "claimedAt": stale_time,
            "heartbeatAt": stale_time,
            "attemptCount": 3,
        }
    )
    collection = AtomicJobCollection([document])
    queue = MongoJobQueue(collection)

    recovered = asyncio.run(
        queue.recover_exhausted_stale_jobs(
            stale_before=NOW - timedelta(minutes=1),
            now=NOW,
            max_attempts=3,
        )
    )
    claim = asyncio.run(
        queue.claim_next(
            worker_id="replacement-worker",
            stale_before=NOW - timedelta(minutes=1),
            now=NOW,
            max_attempts=3,
        )
    )

    assert recovered == 1
    assert claim is None
    assert collection.documents["job-1"]["status"] == "FAILED"
    assert collection.documents["job-1"]["errorCode"] == "WORKER_LEASE_EXHAUSTED"
