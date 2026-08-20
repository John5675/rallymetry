from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import pytest

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
    ) -> str:
        self.indexes.append({"keys": keys, "unique": unique, "sparse": sparse, "name": name})
        return name or "unnamed"

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
        identifier = filter.get("_id")
        if not isinstance(identifier, str):
            return None
        return self.documents.get(identifier)

    async def find_one_and_update(
        self,
        filter: Mapping[str, object],
        update: Mapping[str, object],
        *,
        return_document: object,
    ) -> Document | None:
        del return_document
        identifier = filter.get("_id")
        if not isinstance(identifier, str) or identifier not in self.documents:
            return None
        document = self.documents[identifier]
        set_values = update.get("$set", {})
        unset_values = update.get("$unset", {})
        assert isinstance(set_values, Mapping)
        assert isinstance(unset_values, Mapping)
        document.update(set_values)
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
            if all(document.get(key) == value for key, value in filter.items())
        ]
        return FakeCursor(documents)


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
        index["name"] == "uq_artifacts_pathname" and index["unique"] is True
        for index in database["artifacts"].indexes
    )


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
        target_collection="rallies",
        target_record_id="rally-1",
        changes={"winnerTeam": "NEAR"},
        created_at=NOW,
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
    artifacts = asyncio.run(persistence.list_match_artifacts("match-1"))
    assert [record["artifactId"] for record in artifacts] == ["artifact-source"]


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
    latest_analytics = asyncio.run(persistence.get_latest_match_analytics("match-1"))

    assert total == 2
    assert matches[0]["matchId"] == "match-2"
    assert patched is not None
    assert patched["title"] == "Updated"
    assert "youtubeVideoId" not in patched
    assert len(players) == 1
    assert rally_total == 1 and rallies[0]["recordId"] == "rally-1"
    assert shot_total == 1 and shots[0]["recordId"] == "shot-1"
    assert latest_analytics is not None
    assert latest_analytics["analyticsId"] == "analytics-1"
