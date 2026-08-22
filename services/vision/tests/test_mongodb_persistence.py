from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

import pytest
from pymongo.errors import DuplicateKeyError

from pickleball_vision.config import PersistenceSettings
from pickleball_vision.errors import ErrorCode, PersistenceValidationError
from pickleball_vision.persistence.models import (
    AnalyticsRecord,
    ArtifactAccess,
    ArtifactCategory,
    ArtifactProvider,
    ArtifactRecord,
    CorrectionRecord,
    Document,
    MatchRecord,
    PlayerRecord,
    ProcessingJobRecord,
    ProcessingJobStatus,
    StructuredDomainRecord,
)
from pickleball_vision.persistence.mongodb import (
    COLLECTION_NAMES,
    MongoPersistence,
    validate_compact_document,
)

NOW = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)


class FakeCursor:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.index = 0

    def sort(self, key: str, direction: int) -> FakeCursor:
        def sort_value(item: Document) -> tuple[int, str]:
            value = item.get(key)
            if isinstance(value, datetime):
                return (0, value.isoformat())
            return (1, str(value))

        self.documents.sort(key=sort_value, reverse=direction < 0)
        return self

    def skip(self, count: int) -> FakeCursor:
        self.documents = self.documents[count:]
        return self

    def limit(self, count: int) -> FakeCursor:
        self.documents = self.documents[:count]
        return self

    def __aiter__(self) -> FakeCursor:
        return self

    async def __anext__(self) -> Document:
        if self.index >= len(self.documents):
            raise StopAsyncIteration
        document = self.documents[self.index]
        self.index += 1
        return document


class FakeCollection:
    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.indexes: list[dict[str, object]] = []

    async def create_index(
        self,
        keys: str | Sequence[tuple[str, int]],
        *,
        unique: bool = False,
        sparse: bool = False,
        name: str | None = None,
        partialFilterExpression: Mapping[str, object] | None = None,
    ) -> str:
        self.indexes.append(
            {
                "keys": keys,
                "unique": unique,
                "sparse": sparse,
                "name": name,
                "partialFilterExpression": partialFilterExpression,
            }
        )
        return name or "unnamed"

    async def insert_one(self, document: Mapping[str, object]) -> object:
        if document.get("active") is True:
            for existing in self.documents.values():
                if (
                    existing.get("matchId") == document.get("matchId")
                    and existing.get("active") is True
                ):
                    raise DuplicateKeyError("one active job per match")
        identifier = document.get("_id")
        assert isinstance(identifier, str)
        self.documents[identifier] = dict(document)
        return object()

    async def replace_one(
        self,
        filter: Mapping[str, object],
        replacement: Mapping[str, object],
        *,
        upsert: bool = False,
    ) -> object:
        assert upsert is True
        identifier = filter["_id"]
        assert isinstance(identifier, str)
        self.documents[identifier] = dict(replacement)
        return object()

    async def find_one(self, filter: Mapping[str, object]) -> Document | None:
        for document in self.documents.values():
            if self._matches(document, filter):
                return document.copy()
        return None

    async def find_one_and_update(
        self,
        filter: Mapping[str, object],
        update: Mapping[str, object],
        *,
        return_document: object,
        sort: Sequence[tuple[str, int]] | None = None,
    ) -> Document | None:
        del return_document
        candidates = list(self.documents.values())
        if sort:
            for key, direction in reversed(sort):
                candidates.sort(key=lambda item: str(item.get(key, "")), reverse=direction < 0)
        document = next(
            (item for item in candidates if self._matches(item, filter)),
            None,
        )
        if document is None:
            return None
        set_values = update.get("$set", {})
        unset_values = update.get("$unset", {})
        increment_values = update.get("$inc", {})
        assert isinstance(set_values, Mapping)
        assert isinstance(unset_values, Mapping)
        assert isinstance(increment_values, Mapping)
        document.update(set_values)
        for key, value in increment_values.items():
            assert isinstance(key, str) and isinstance(value, int)
            current = document.get(key, 0)
            assert isinstance(current, int)
            document[key] = current + value
        for key in unset_values:
            assert isinstance(key, str)
            document.pop(key, None)
        return document.copy()

    async def count_documents(self, filter: Mapping[str, object]) -> int:
        return sum(
            all(document.get(key) == value for key, value in filter.items())
            for document in self.documents.values()
        )

    def find(self, filter: Mapping[str, object]) -> FakeCursor:
        documents = [
            document.copy()
            for document in self.documents.values()
            if self._matches(document, filter)
        ]
        return FakeCursor(documents)

    @classmethod
    def _matches(cls, document: Mapping[str, object], filter: Mapping[str, object]) -> bool:
        for key, expected in filter.items():
            if key == "$or":
                if not isinstance(expected, list) or not any(
                    isinstance(item, Mapping) and cls._matches(document, item) for item in expected
                ):
                    return False
                continue
            actual = document.get(key)
            if isinstance(expected, Mapping):
                for operator, operand in expected.items():
                    if operator == "$exists":
                        if (key in document) is not bool(operand):
                            return False
                    elif operator == "$in":
                        if not isinstance(operand, list) or actual not in operand:
                            return False
                    elif operator == "$lt":
                        if actual is None or not actual < operand:
                            return False
                    elif operator == "$lte":
                        if actual is None or not actual <= operand:
                            return False
                    elif operator == "$gte":
                        if actual is None or not actual >= operand:
                            return False
                    else:
                        raise AssertionError(f"unsupported fake query operator {operator}")
                continue
            if actual != expected:
                return False
        return True


class FakeDatabase:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collections.setdefault(name, FakeCollection())


def test_mongodb_adapter_initializes_indexes_and_separate_collections() -> None:
    database = FakeDatabase()
    persistence = MongoPersistence(database)

    asyncio.run(persistence.initialize_indexes())

    assert set(database.collections) == set(COLLECTION_NAMES)
    assert any(
        index["name"] == "uq_players_match_player" and index["unique"] is True
        for index in database["players"].indexes
    )
    assert any(
        index["name"] == "ix_jobs_status_updated" for index in database["processing_jobs"].indexes
    )
    assert any(
        index["name"] == "ix_jobs_worker_claim" for index in database["processing_jobs"].indexes
    )
    assert any(
        index["name"] == "uq_jobs_one_active_match"
        and index["unique"] is True
        and index["partialFilterExpression"] == {"active": True}
        for index in database["processing_jobs"].indexes
    )
    assert any(
        index["name"] == "uq_artifacts_pathname" and index["unique"] is True
        for index in database["artifacts"].indexes
    )
    assert any(
        index["name"] == "uq_corrections_active_target"
        and index["unique"] is True
        and index["partialFilterExpression"] == {"active": True}
        for index in database["corrections"].indexes
    )


def test_worker_claim_is_atomic_heartbeats_and_recovers_after_expired_lease() -> None:
    database = FakeDatabase()
    persistence = MongoPersistence(database)
    job = ProcessingJobRecord(
        job_id="job_claim",
        match_id="match_claim",
        job_type="analyze_match",
        status=ProcessingJobStatus.QUEUED,
        stage=ProcessingJobStatus.QUEUED.value,
        created_at=NOW,
        updated_at=NOW,
    )
    asyncio.run(persistence.save_processing_job(job))

    first = asyncio.run(
        persistence.claim_next_processing_job(
            worker_id="worker-one",
            now=NOW,
            lease_seconds=60,
            max_attempts=2,
        )
    )
    duplicate = asyncio.run(
        persistence.claim_next_processing_job(
            worker_id="worker-two",
            now=NOW + timedelta(seconds=10),
            lease_seconds=60,
            max_attempts=2,
        )
    )
    retained = asyncio.run(
        persistence.heartbeat_processing_job(
            "job_claim",
            worker_id="worker-one",
            now=NOW + timedelta(seconds=30),
            lease_seconds=60,
        )
    )
    recovered = asyncio.run(
        persistence.claim_next_processing_job(
            worker_id="worker-two",
            now=NOW + timedelta(seconds=91),
            lease_seconds=60,
            max_attempts=2,
        )
    )

    assert first is not None
    assert first["workerId"] == "worker-one"
    assert first["attemptCount"] == 1
    assert duplicate is None
    assert retained is True
    assert recovered is not None
    assert recovered["workerId"] == "worker-two"
    assert recovered["attemptCount"] == 2


def test_worker_exhausted_stale_job_is_failed_instead_of_stranded() -> None:
    database = FakeDatabase()
    persistence = MongoPersistence(database)
    stale = ProcessingJobRecord(
        job_id="job_stale",
        match_id="match_stale",
        job_type="analyze_match",
        status=ProcessingJobStatus.PLAYER_PROCESSING,
        stage=ProcessingJobStatus.PLAYER_PROCESSING.value,
        claimed_at=NOW,
        heartbeat_at=NOW,
        lease_expires_at=NOW + timedelta(seconds=30),
        worker_id="worker-gone",
        attempt_count=2,
        created_at=NOW,
        updated_at=NOW,
    )
    asyncio.run(persistence.save_processing_job(stale))

    failed = asyncio.run(
        persistence.fail_exhausted_stale_processing_job(
            now=NOW + timedelta(seconds=31),
            max_attempts=2,
        )
    )

    assert failed is not None
    assert failed["status"] == "FAILED"
    assert failed["active"] is False
    assert failed["errorCode"] == "WORKER_ATTEMPTS_EXHAUSTED"
    assert "workerId" not in failed


def test_worker_failure_requeues_before_bounded_terminal_attempt() -> None:
    database = FakeDatabase()
    persistence = MongoPersistence(database)
    job = ProcessingJobRecord(
        job_id="job_retry",
        match_id="match_retry",
        job_type="analyze_match",
        status=ProcessingJobStatus.QUEUED,
        stage=ProcessingJobStatus.QUEUED.value,
        claimed_at=NOW,
        heartbeat_at=NOW,
        lease_expires_at=NOW + timedelta(seconds=60),
        worker_id="worker-one",
        attempt_count=1,
        created_at=NOW,
        updated_at=NOW,
    )
    asyncio.run(persistence.save_processing_job(job))

    requeued = asyncio.run(
        persistence.release_or_fail_processing_job(
            "job_retry",
            worker_id="worker-one",
            now=NOW + timedelta(seconds=1),
            max_attempts=2,
            error_code="WORKER_EXECUTION_FAILED",
            error_message="safe failure",
        )
    )
    assert requeued is not None
    assert requeued["status"] == "QUEUED"
    assert requeued["active"] is True
    assert "workerId" not in requeued

    claimed = asyncio.run(
        persistence.claim_next_processing_job(
            worker_id="worker-two",
            now=NOW + timedelta(seconds=2),
            lease_seconds=60,
            max_attempts=2,
        )
    )
    assert claimed is not None and claimed["attemptCount"] == 2
    failed = asyncio.run(
        persistence.release_or_fail_processing_job(
            "job_retry",
            worker_id="worker-two",
            now=NOW + timedelta(seconds=3),
            max_attempts=2,
            error_code="WORKER_EXECUTION_FAILED",
            error_message="safe failure",
        )
    )
    assert failed is not None
    assert failed["status"] == "FAILED"
    assert failed["active"] is False
    assert failed["failedAt"] == NOW + timedelta(seconds=3)


def test_mongodb_adapter_persists_compact_records_without_one_huge_match_document() -> None:
    database = FakeDatabase()
    persistence = MongoPersistence(database)
    match = MatchRecord(
        match_id="match-1",
        youtube_video_id="youtube1234",
        source_artifact_id="artifact-source",
        summary={"rallyCount": 1},
        created_at=NOW,
        updated_at=NOW,
    )
    player = PlayerRecord(
        match_id="match-1",
        player_id="JOHN",
        display_name="John",
        logical_identity="ME",
        created_at=NOW,
        updated_at=NOW,
    )
    event = StructuredDomainRecord(
        match_id="match-1",
        record_id="event-1",
        payload={"frame": 30, "timestamp": 1.0},
        confidence=0.8,
        created_at=NOW,
    )
    analytics = AnalyticsRecord(
        match_id="match-1",
        analytics_id="analytics-1",
        calculation_version="v1",
        metrics={"rallyCount": {"value": 1}},
        created_at=NOW,
    )
    job = ProcessingJobRecord(
        job_id="job-1",
        match_id="match-1",
        job_type="analyze_match",
        created_at=NOW,
        updated_at=NOW,
    )
    correction = CorrectionRecord(
        correction_id="correction-1",
        match_id="match-1",
        correction_type="RALLY_BOUNDARY",
        target_collection="rallies",
        target_record_id="rally-1",
        prediction={"startFrame": 10, "endFrame": 100},
        prediction_confidence=0.8,
        prediction_version="rally-v1",
        human_correction={"startFrame": 12, "endFrame": 98},
        created_at=NOW,
        corrected_at=NOW,
        updated_at=NOW,
    )
    artifact = ArtifactRecord(
        artifact_id="artifact-source",
        match_id="match-1",
        artifact_type="source_video",
        category=ArtifactCategory.SOURCE_MEDIA,
        pathname="source_media/match-1/random/source.mp4",
        provider=ArtifactProvider.LOCAL,
        access=ArtifactAccess.LOCAL,
        content_type="video/mp4",
        size_bytes=100,
        created_at=NOW,
    )

    asyncio.run(persistence.save_match(match))
    asyncio.run(persistence.save_players([player]))
    asyncio.run(persistence.save_rallies([event]))
    asyncio.run(persistence.save_contacts([event]))
    asyncio.run(persistence.save_bounces([event]))
    asyncio.run(persistence.save_shots([event]))
    asyncio.run(persistence.save_analytics(analytics))
    asyncio.run(persistence.save_processing_job(job))
    asyncio.run(persistence.save_correction(correction))
    asyncio.run(persistence.save_artifact(artifact))

    match_document = asyncio.run(persistence.get_match("match-1"))
    assert match_document is not None
    assert match_document["youtubeVideoId"] == "youtube1234"
    assert "rallies" not in match_document
    assert database["rallies"].documents["match-1:event-1"]["payload"] == {
        "frame": 30,
        "timestamp": 1.0,
    }
    assert asyncio.run(persistence.get_processing_job("job-1")) is not None
    assert asyncio.run(persistence.get_artifact("artifact-source")) is not None
    artifacts = asyncio.run(persistence.list_match_artifacts("match-1"))
    assert [record["artifactId"] for record in artifacts] == ["artifact-source"]


def test_processing_job_creation_is_atomic_per_active_match() -> None:
    database = FakeDatabase()
    persistence = MongoPersistence(database)
    first = ProcessingJobRecord(
        job_id="job-1",
        match_id="match-1",
        job_type="analyze_match",
        processing_run_id="run-1",
        created_at=NOW,
        updated_at=NOW,
    )
    second = ProcessingJobRecord(
        job_id="job-2",
        match_id="match-1",
        job_type="analyze_match",
        processing_run_id="run-2",
        created_at=NOW,
        updated_at=NOW,
    )

    first_document, first_created = asyncio.run(
        persistence.create_processing_job_if_no_active(first)
    )
    second_document, second_created = asyncio.run(
        persistence.create_processing_job_if_no_active(second)
    )

    assert first_created is True
    assert second_created is False
    assert first_document["jobId"] == second_document["jobId"] == "job-1"


def test_correction_prediction_snapshot_is_immutable_at_persistence_boundary() -> None:
    database = FakeDatabase()
    persistence = MongoPersistence(database)
    original = CorrectionRecord(
        correction_id="correction-1",
        match_id="match-1",
        correction_type="SHOT_TYPE",
        target_collection="shots",
        target_record_id="shot-1",
        prediction={"shotType": "UNKNOWN"},
        prediction_confidence=0.4,
        prediction_version="shots-v1",
        human_correction={"shotType": "DRIVE"},
        created_at=NOW,
        corrected_at=NOW,
        updated_at=NOW,
    )
    corrupted = CorrectionRecord(
        correction_id="correction-1",
        match_id="match-1",
        correction_type="SHOT_TYPE",
        target_collection="shots",
        target_record_id="shot-1",
        prediction={"shotType": "SERVE"},
        prediction_confidence=0.9,
        prediction_version="shots-v2",
        human_correction={"shotType": "DRIVE"},
        revision=2,
        created_at=NOW,
        corrected_at=NOW,
        updated_at=NOW,
    )

    asyncio.run(persistence.save_correction(original))
    with pytest.raises(PersistenceValidationError):
        asyncio.run(persistence.save_correction(corrupted))

    stored = asyncio.run(persistence.get_correction("correction-1"))
    assert stored is not None
    assert stored["prediction"] == {"shotType": "UNKNOWN"}
    assert stored["predictionVersion"] == "shots-v1"


@pytest.mark.parametrize(
    "document",
    [
        {"_id": "bad", "videoBytes": b"not-in-mongodb"},
        {"_id": "bad", "nested": {"rawFrames": [1, 2, 3]}},
        {"_id": "bad", "confidence": float("nan")},
        {"_id": "bad", "payload": {"illegal.key": 1}},
        {"_id": "bad", "payload": "x" * (2 * 1024 * 1024)},
    ],
)
def test_compact_document_validation_rejects_binary_and_unbounded_data(
    document: Document,
) -> None:
    with pytest.raises(PersistenceValidationError) as raised:
        validate_compact_document(document)

    assert raised.value.code is ErrorCode.PERSISTENCE_VALIDATION


def test_duplicate_batch_ids_are_rejected_before_writing() -> None:
    database = FakeDatabase()
    persistence = MongoPersistence(database)
    player = PlayerRecord(match_id="match-1", player_id="JOHN")

    with pytest.raises(PersistenceValidationError):
        asyncio.run(persistence.save_players([player, player]))

    assert not database["players"].documents


def test_mongodb_connection_requires_environment_backed_url() -> None:
    with pytest.raises(PersistenceValidationError):
        asyncio.run(MongoPersistence.connect_from_settings(PersistenceSettings()))


def test_mongodb_adapter_lists_and_patches_api_records() -> None:
    database = FakeDatabase()
    persistence = MongoPersistence(database)
    first = MatchRecord(
        match_id="match-1",
        title="First",
        youtube_video_id="youtube-1",
        created_at=NOW,
        updated_at=NOW,
    )
    second_time = datetime(2026, 8, 19, 13, 0, tzinfo=UTC)
    second = MatchRecord(
        match_id="match-2",
        title="Second",
        created_at=second_time,
        updated_at=second_time,
    )
    player = PlayerRecord(match_id="match-1", player_id="JOHN", created_at=NOW, updated_at=NOW)
    rally = StructuredDomainRecord(
        match_id="match-1",
        record_id="rally-1",
        payload={"startFrame": 10},
        timestamp_seconds=1.0,
        created_at=NOW,
    )
    shot = StructuredDomainRecord(
        match_id="match-1",
        record_id="shot-1",
        payload={"shotType": "SERVE"},
        timestamp_seconds=1.2,
        created_at=NOW,
    )
    contact = StructuredDomainRecord(
        match_id="match-1",
        record_id="contact-1",
        payload={"playerId": "JOHN"},
        timestamp_seconds=1.2,
        created_at=NOW,
    )
    bounce = StructuredDomainRecord(
        match_id="match-1",
        record_id="bounce-1",
        payload={"courtPosition": {"x": 2.0, "y": 8.0}},
        timestamp_seconds=1.8,
        created_at=NOW,
    )
    analytics = AnalyticsRecord(
        match_id="match-1",
        analytics_id="analytics-1",
        calculation_version="v1",
        metrics={"rallyCount": {"value": 1}},
        created_at=NOW,
    )

    asyncio.run(persistence.save_match(first))
    asyncio.run(persistence.save_match(second))
    asyncio.run(persistence.save_players([player]))
    asyncio.run(persistence.save_rallies([rally]))
    asyncio.run(persistence.save_shots([shot]))
    asyncio.run(persistence.save_contacts([contact]))
    asyncio.run(persistence.save_bounces([bounce]))
    asyncio.run(persistence.save_analytics(analytics))

    matches, total = asyncio.run(persistence.list_matches(limit=1, offset=0))
    patched = asyncio.run(
        persistence.patch_match(
            "match-1",
            {"title": "Updated", "youtubeVideoId": None},
            updated_at=second_time,
        )
    )
    players = asyncio.run(persistence.list_match_players("match-1"))
    rallies, rally_total = asyncio.run(
        persistence.list_match_rallies("match-1", limit=50, offset=0)
    )
    shots, shot_total = asyncio.run(persistence.list_match_shots("match-1", limit=50, offset=0))
    contacts, contact_total = asyncio.run(
        persistence.list_match_contacts("match-1", limit=50, offset=0)
    )
    bounces, bounce_total = asyncio.run(
        persistence.list_match_bounces("match-1", limit=50, offset=0)
    )
    latest_analytics = asyncio.run(persistence.get_latest_match_analytics("match-1"))

    assert total == 2
    assert matches[0]["matchId"] == "match-2"
    assert patched is not None
    assert patched["title"] == "Updated"
    assert "youtubeVideoId" not in patched
    assert len(players) == 1
    assert rally_total == 1 and rallies[0]["recordId"] == "rally-1"
    assert shot_total == 1 and shots[0]["recordId"] == "shot-1"
    assert contact_total == 1 and contacts[0]["recordId"] == "contact-1"
    assert bounce_total == 1 and bounces[0]["recordId"] == "bounce-1"
    assert latest_analytics is not None
    assert latest_analytics["analyticsId"] == "analytics-1"
