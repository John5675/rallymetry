"""MongoDB Atlas adapter using the official PyMongo Async API."""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime
from typing import Protocol, cast

from bson import BSON
from pymongo import ASCENDING, AsyncMongoClient
from pymongo.errors import PyMongoError

from pickleball_vision.config import PersistenceSettings
from pickleball_vision.errors import PersistenceOperationError, PersistenceValidationError
from pickleball_vision.persistence.models import (
    AnalyticsRecord,
    ArtifactRecord,
    CorrectionRecord,
    Document,
    MatchRecord,
    PlayerRecord,
    ProcessingJobRecord,
    StructuredCollection,
    StructuredDomainRecord,
)

MAX_COMPACT_DOCUMENT_BYTES = 2 * 1024 * 1024
COLLECTION_NAMES = (
    "matches",
    "players",
    "rallies",
    "contacts",
    "bounces",
    "shots",
    "analytics",
    "processing_jobs",
    "corrections",
    "artifacts",
)
_PROHIBITED_BINARY_FIELDS = {
    "audiowaveform",
    "audiowaveforms",
    "debugartifactbytes",
    "modelweights",
    "mp4bytes",
    "rawdetectionarrays",
    "rawframes",
    "rawvideoframes",
    "videobytes",
}


class _AsyncCursor(Protocol):
    def sort(self, key: str, direction: int) -> _AsyncCursor: ...

    def __aiter__(self) -> AsyncIterator[Document]: ...


class _AsyncCollection(Protocol):
    async def create_index(
        self,
        keys: str | Sequence[tuple[str, int]],
        *,
        unique: bool = False,
        sparse: bool = False,
        name: str | None = None,
    ) -> str: ...

    async def replace_one(
        self,
        filter: Mapping[str, object],
        replacement: Mapping[str, object],
        *,
        upsert: bool = False,
    ) -> object: ...

    async def find_one(self, filter: Mapping[str, object]) -> Document | None: ...

    def find(self, filter: Mapping[str, object]) -> _AsyncCursor: ...


class _AsyncDatabase(Protocol):
    def __getitem__(self, name: str) -> _AsyncCollection: ...


class _AsyncClient(Protocol):
    async def close(self) -> None: ...


def _validate_value(value: object, path: str) -> None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise PersistenceValidationError(
            f"{path} contains binary data; store it in an ArtifactStore and persist a reference"
        )
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str) or not key:
                raise PersistenceValidationError(f"{path} contains a non-string or empty key")
            if key.startswith("$") or "." in key:
                raise PersistenceValidationError(f"{path}.{key} is not a safe MongoDB field name")
            if key.replace("_", "").lower() in _PROHIBITED_BINARY_FIELDS:
                raise PersistenceValidationError(
                    f"{path}.{key} must be stored as an artifact reference, not inline"
                )
            _validate_value(nested, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_value(nested, f"{path}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise PersistenceValidationError(f"{path} must not contain NaN or infinity")
    if value is None or isinstance(value, (str, bool, int, float, datetime)):
        return
    raise PersistenceValidationError(
        f"{path} contains unsupported type {type(value).__name__}; use BSON-safe primitives"
    )


def validate_compact_document(document: Mapping[str, object]) -> None:
    """Reject binary, unsafe, or unbounded records before contacting MongoDB."""

    _validate_value(document, "document")
    try:
        encoded = BSON.encode(dict(document))
    except (TypeError, ValueError) as error:
        raise PersistenceValidationError(f"document is not BSON encodable: {error}") from error
    if len(encoded) > MAX_COMPACT_DOCUMENT_BYTES:
        raise PersistenceValidationError(
            f"document is {len(encoded)} bytes; compact record limit is "
            f"{MAX_COMPACT_DOCUMENT_BYTES} bytes"
        )


def _driver_reason(error: PyMongoError) -> str:
    return f"MongoDB driver failure ({type(error).__name__})"


class MongoPersistence:
    """Compact match persistence separated from analysis-domain implementation."""

    def __init__(
        self,
        database: _AsyncDatabase,
        *,
        client: _AsyncClient | None = None,
    ) -> None:
        self._database = database
        self._client = client

    @classmethod
    async def connect(cls, mongodb_url: str, database_name: str) -> MongoPersistence:
        """Create and verify a hosted client only when explicitly requested."""

        if not mongodb_url.strip():
            raise PersistenceValidationError("MONGODB_URL is required to connect")
        if not database_name.strip():
            raise PersistenceValidationError("MongoDB database name must not be empty")
        client = AsyncMongoClient[Document](mongodb_url, tz_aware=True)
        try:
            await client.admin.command("ping")
        except PyMongoError as error:
            await client.close()
            raise PersistenceOperationError(
                "connect",
                reason=_driver_reason(error),
            ) from error
        database = cast(_AsyncDatabase, client[database_name])
        return cls(database, client=cast(_AsyncClient, client))

    @classmethod
    async def connect_from_settings(cls, settings: PersistenceSettings) -> MongoPersistence:
        """Connect from the environment-backed settings boundary."""

        if settings.mongodb_url is None:
            raise PersistenceValidationError("MONGODB_URL is required to connect")
        return await cls.connect(settings.mongodb_url, settings.mongodb_database)

    async def close(self) -> None:
        """Close the owned client; injected test databases do not require one."""

        if self._client is None:
            return
        try:
            await self._client.close()
        except PyMongoError as error:
            raise PersistenceOperationError("close", reason=_driver_reason(error)) from error

    async def initialize_indexes(self) -> None:
        """Create deterministic indexes for match-scoped access patterns."""

        index_definitions: dict[
            str,
            tuple[tuple[str | Sequence[tuple[str, int]], bool, bool, str], ...],
        ] = {
            "matches": (
                ("youtubeVideoId", True, True, "uq_matches_youtube_video_id"),
                ("updatedAt", False, False, "ix_matches_updated_at"),
            ),
            "players": (
                (
                    (("matchId", ASCENDING), ("playerId", ASCENDING)),
                    True,
                    False,
                    "uq_players_match_player",
                ),
            ),
            "rallies": self._event_indexes("rally"),
            "contacts": self._event_indexes("contact"),
            "bounces": self._event_indexes("bounce"),
            "shots": self._event_indexes("shot"),
            "analytics": (
                (
                    (("matchId", ASCENDING), ("analyticsId", ASCENDING)),
                    True,
                    False,
                    "uq_analytics_match_id",
                ),
            ),
            "processing_jobs": (
                (
                    (("matchId", ASCENDING), ("createdAt", ASCENDING)),
                    False,
                    False,
                    "ix_jobs_match_created",
                ),
                (
                    (("status", ASCENDING), ("updatedAt", ASCENDING)),
                    False,
                    False,
                    "ix_jobs_status_updated",
                ),
            ),
            "corrections": (
                (
                    (
                        ("matchId", ASCENDING),
                        ("targetCollection", ASCENDING),
                        ("targetRecordId", ASCENDING),
                    ),
                    False,
                    False,
                    "ix_corrections_target",
                ),
            ),
            "artifacts": (
                ("pathname", True, False, "uq_artifacts_pathname"),
                (
                    (("matchId", ASCENDING), ("createdAt", ASCENDING)),
                    False,
                    False,
                    "ix_artifacts_match_created",
                ),
                (
                    (("matchId", ASCENDING), ("category", ASCENDING)),
                    False,
                    False,
                    "ix_artifacts_match_category",
                ),
            ),
        }
        try:
            for collection_name, definitions in index_definitions.items():
                collection = self._database[collection_name]
                for keys, unique, sparse, name in definitions:
                    await collection.create_index(
                        keys,
                        unique=unique,
                        sparse=sparse,
                        name=name,
                    )
        except PyMongoError as error:
            raise PersistenceOperationError(
                "initialize_indexes",
                reason=_driver_reason(error),
            ) from error

    @staticmethod
    def _event_indexes(
        kind: str,
    ) -> tuple[tuple[str | Sequence[tuple[str, int]], bool, bool, str], ...]:
        return (
            (
                (("matchId", ASCENDING), ("recordId", ASCENDING)),
                True,
                False,
                f"uq_{kind}_match_record",
            ),
            (
                (("matchId", ASCENDING), ("timestampSeconds", ASCENDING)),
                False,
                True,
                f"ix_{kind}_match_time",
            ),
        )

    async def save_match(self, record: MatchRecord) -> None:
        await self._replace("matches", record.to_document(), "save_match")

    async def save_players(self, records: Sequence[PlayerRecord]) -> None:
        await self._replace_many("players", records, "save_players")

    async def save_rallies(self, records: Sequence[StructuredDomainRecord]) -> None:
        await self._save_structured(StructuredCollection.RALLIES, records)

    async def save_contacts(self, records: Sequence[StructuredDomainRecord]) -> None:
        await self._save_structured(StructuredCollection.CONTACTS, records)

    async def save_bounces(self, records: Sequence[StructuredDomainRecord]) -> None:
        await self._save_structured(StructuredCollection.BOUNCES, records)

    async def save_shots(self, records: Sequence[StructuredDomainRecord]) -> None:
        await self._save_structured(StructuredCollection.SHOTS, records)

    async def save_analytics(self, record: AnalyticsRecord) -> None:
        await self._replace("analytics", record.to_document(), "save_analytics")

    async def save_processing_job(self, record: ProcessingJobRecord) -> None:
        await self._replace("processing_jobs", record.to_document(), "save_processing_job")

    async def save_correction(self, record: CorrectionRecord) -> None:
        await self._replace("corrections", record.to_document(), "save_correction")

    async def save_artifact(self, record: ArtifactRecord) -> None:
        await self._replace("artifacts", record.to_document(), "save_artifact")

    async def get_match(self, match_id: str) -> Document | None:
        return await self._find_one("matches", {"_id": match_id}, "get_match")

    async def get_processing_job(self, job_id: str) -> Document | None:
        return await self._find_one(
            "processing_jobs",
            {"_id": job_id},
            "get_processing_job",
        )

    async def list_match_artifacts(self, match_id: str) -> tuple[Document, ...]:
        try:
            cursor = self._database["artifacts"].find({"matchId": match_id})
            cursor = cursor.sort("createdAt", ASCENDING)
            documents: list[Document] = []
            async for document in cursor:
                documents.append(document)
            return tuple(documents)
        except PyMongoError as error:
            raise PersistenceOperationError(
                "list_match_artifacts",
                reason=_driver_reason(error),
            ) from error

    async def _save_structured(
        self,
        collection: StructuredCollection,
        records: Sequence[StructuredDomainRecord],
    ) -> None:
        await self._replace_many(collection.value, records, f"save_{collection.value}")

    async def _replace_many(
        self,
        collection: str,
        records: Sequence[PlayerRecord | StructuredDomainRecord],
        operation: str,
    ) -> None:
        documents = [record.to_document() for record in records]
        identifiers: list[str] = []
        for document in documents:
            identifier = document.get("_id")
            if not isinstance(identifier, str) or not identifier:
                raise PersistenceValidationError(f"{operation} document requires a string _id")
            identifiers.append(identifier)
        if len(identifiers) != len(set(identifiers)):
            raise PersistenceValidationError(f"{operation} contains duplicate record IDs")
        for document in documents:
            await self._replace(collection, document, operation)

    async def _replace(self, collection: str, document: Document, operation: str) -> None:
        validate_compact_document(document)
        identifier = document.get("_id")
        if not isinstance(identifier, str) or not identifier:
            raise PersistenceValidationError(f"{operation} document requires a string _id")
        try:
            await self._database[collection].replace_one(
                {"_id": identifier},
                document,
                upsert=True,
            )
        except PyMongoError as error:
            raise PersistenceOperationError(operation, reason=_driver_reason(error)) from error

    async def _find_one(
        self,
        collection: str,
        filter: Mapping[str, object],
        operation: str,
    ) -> Document | None:
        try:
            return await self._database[collection].find_one(filter)
        except PyMongoError as error:
            raise PersistenceOperationError(operation, reason=_driver_reason(error)) from error
