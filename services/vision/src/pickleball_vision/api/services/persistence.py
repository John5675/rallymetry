"""Persistence protocol consumed by the HTTP application layer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

from pickleball_vision.persistence.models import (
    Document,
    MatchRecord,
    ProcessingJobRecord,
)


class ApplicationPersistence(Protocol):
    """Small async data contract required by Milestone 20 routes."""

    async def save_match(self, record: MatchRecord) -> None: ...

    async def get_match(self, match_id: str) -> Document | None: ...

    async def list_matches(
        self,
        *,
        limit: int,
        offset: int,
    ) -> tuple[tuple[Document, ...], int]: ...

    async def patch_match(
        self,
        match_id: str,
        fields: Mapping[str, object],
        *,
        updated_at: datetime,
    ) -> Document | None: ...

    async def list_match_players(self, match_id: str) -> tuple[Document, ...]: ...

    async def list_match_rallies(
        self,
        match_id: str,
        *,
        limit: int,
        offset: int,
    ) -> tuple[tuple[Document, ...], int]: ...

    async def list_match_shots(
        self,
        match_id: str,
        *,
        limit: int,
        offset: int,
    ) -> tuple[tuple[Document, ...], int]: ...

    async def list_match_contacts(
        self,
        match_id: str,
        *,
        limit: int,
        offset: int,
    ) -> tuple[tuple[Document, ...], int]: ...

    async def list_match_bounces(
        self,
        match_id: str,
        *,
        limit: int,
        offset: int,
    ) -> tuple[tuple[Document, ...], int]: ...

    async def get_latest_match_analytics(self, match_id: str) -> Document | None: ...

    async def list_match_artifacts(self, match_id: str) -> tuple[Document, ...]: ...

    async def save_processing_job(self, record: ProcessingJobRecord) -> None: ...

    async def get_processing_job(self, job_id: str) -> Document | None: ...

    async def get_artifact(self, artifact_id: str) -> Document | None: ...


def copy_documents(documents: Sequence[Document]) -> tuple[Document, ...]:
    """Copy injected-test documents so API serialization cannot mutate storage."""

    return tuple(dict(document) for document in documents)
