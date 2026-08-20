"""Application orchestration over compact persistence records only."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal, TypeVar

from pydantic import BaseModel, ValidationError

from pickleball_vision.api.errors import ApiError, ResourceNotFoundError
from pickleball_vision.api.schemas.jobs import JobResponse
from pickleball_vision.api.schemas.matches import (
    MatchCreateRequest,
    MatchListResponse,
    MatchPatchRequest,
    MatchResponse,
)
from pickleball_vision.api.schemas.records import (
    AnalyticsResponse,
    ArtifactListResponse,
    ArtifactResponse,
    DomainRecordListResponse,
    DomainRecordResponse,
    PlayerListResponse,
    PlayerResponse,
)
from pickleball_vision.api.services.persistence import ApplicationPersistence
from pickleball_vision.persistence.models import (
    ArtifactCategory,
    ArtifactProvider,
    Document,
    MatchRecord,
    ProcessingJobRecord,
    SourceMediaType,
)

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


def _record(model: type[ResponseModel], document: Document) -> ResponseModel:
    try:
        return model.model_validate(document)
    except ValidationError as error:
        raise ApiError(
            status_code=500,
            code="persisted_record_invalid",
            message="A persisted record does not satisfy the API contract",
            details={"schema": model.__name__},
        ) from error


class MatchApplicationService:
    """HTTP-facing use cases without CV, ML, audio, or analytics execution."""

    def __init__(self, persistence: ApplicationPersistence) -> None:
        self._persistence = persistence

    async def create_match(self, request: MatchCreateRequest) -> MatchResponse:
        now = datetime.now(UTC)
        record = MatchRecord(
            match_id=f"match_{uuid.uuid4().hex}",
            title=request.title,
            youtube_video_id=request.youtube_video_id,
            source_artifact_id=request.source_artifact_id,
            created_at=now,
            updated_at=now,
        )
        await self._persistence.save_match(record)
        return _record(MatchResponse, record.to_document())

    async def list_matches(self, *, limit: int, offset: int) -> MatchListResponse:
        documents, total = await self._persistence.list_matches(limit=limit, offset=offset)
        return MatchListResponse(
            items=[_record(MatchResponse, document) for document in documents],
            total=total,
            limit=limit,
            offset=offset,
        )

    async def get_match(self, match_id: str) -> MatchResponse:
        document = await self._require_match(match_id)
        return _record(MatchResponse, document)

    async def patch_match(
        self,
        match_id: str,
        request: MatchPatchRequest,
    ) -> MatchResponse:
        fields = request.model_dump(mode="python", by_alias=True, exclude_unset=True)
        document = await self._persistence.patch_match(
            match_id,
            fields,
            updated_at=datetime.now(UTC),
        )
        if document is None:
            raise ResourceNotFoundError("match", match_id)
        return _record(MatchResponse, document)

    async def list_players(self, match_id: str) -> PlayerListResponse:
        await self._require_match(match_id)
        documents = await self._persistence.list_match_players(match_id)
        items = [_record(PlayerResponse, document) for document in documents]
        return PlayerListResponse(items=items, total=len(items))

    async def list_rallies(
        self,
        match_id: str,
        *,
        limit: int,
        offset: int,
    ) -> DomainRecordListResponse:
        await self._require_match(match_id)
        documents, total = await self._persistence.list_match_rallies(
            match_id,
            limit=limit,
            offset=offset,
        )
        return self._domain_page(documents, total=total, limit=limit, offset=offset)

    async def list_shots(
        self,
        match_id: str,
        *,
        limit: int,
        offset: int,
    ) -> DomainRecordListResponse:
        await self._require_match(match_id)
        documents, total = await self._persistence.list_match_shots(
            match_id,
            limit=limit,
            offset=offset,
        )
        return self._domain_page(documents, total=total, limit=limit, offset=offset)

    async def list_events(
        self,
        match_id: str,
        *,
        event_type: Literal["contacts", "bounces"],
        limit: int,
        offset: int,
    ) -> DomainRecordListResponse:
        await self._require_match(match_id)
        if event_type == "contacts":
            documents, total = await self._persistence.list_match_contacts(
                match_id,
                limit=limit,
                offset=offset,
            )
        else:
            documents, total = await self._persistence.list_match_bounces(
                match_id,
                limit=limit,
                offset=offset,
            )
        return self._domain_page(documents, total=total, limit=limit, offset=offset)

    async def get_analytics(self, match_id: str) -> AnalyticsResponse:
        await self._require_match(match_id)
        document = await self._persistence.get_latest_match_analytics(match_id)
        if document is None:
            raise ResourceNotFoundError("analytics", match_id)
        return _record(AnalyticsResponse, document)

    async def list_artifacts(self, match_id: str) -> ArtifactListResponse:
        await self._require_match(match_id)
        documents = await self._persistence.list_match_artifacts(match_id)
        items = [_record(ArtifactResponse, document) for document in documents]
        return ArtifactListResponse(items=items, total=len(items))

    async def queue_processing(self, match_id: str) -> JobResponse:
        match = await self._require_match(match_id)
        source_artifact_id = match.get("sourceArtifactId")
        if not isinstance(source_artifact_id, str) or not source_artifact_id:
            raise ApiError(
                status_code=409,
                code="source_media_required",
                message="The match requires a SOURCE_MEDIA artifact before processing",
            )
        artifact = await self._persistence.get_artifact(source_artifact_id)
        if artifact is None or artifact.get("category") != ArtifactCategory.SOURCE_MEDIA.value:
            raise ApiError(
                status_code=409,
                code="source_media_unavailable",
                message="The match source-media artifact is unavailable",
            )
        provider = artifact.get("provider")
        if provider == ArtifactProvider.LOCAL.value:
            source_type = SourceMediaType.LOCAL_PATH
        elif provider == ArtifactProvider.VERCEL_BLOB.value:
            source_type = SourceMediaType.BLOB
        else:
            raise ApiError(
                status_code=409,
                code="source_media_unavailable",
                message="The match source-media provider is unsupported",
            )
        now = datetime.now(UTC)
        record = ProcessingJobRecord(
            job_id=f"job_{uuid.uuid4().hex}",
            match_id=match_id,
            job_type="analyze_match",
            source_type=source_type,
            source_artifact_id=source_artifact_id,
            created_at=now,
            updated_at=now,
        )
        await self._persistence.save_processing_job(record)
        return _record(JobResponse, record.to_document())

    async def get_job(self, job_id: str) -> JobResponse:
        document = await self._persistence.get_processing_job(job_id)
        if document is None:
            raise ResourceNotFoundError("processing job", job_id)
        return _record(JobResponse, document)

    async def _require_match(self, match_id: str) -> Document:
        document = await self._persistence.get_match(match_id)
        if document is None:
            raise ResourceNotFoundError("match", match_id)
        return document

    @staticmethod
    def _domain_page(
        documents: tuple[Document, ...],
        *,
        total: int,
        limit: int,
        offset: int,
    ) -> DomainRecordListResponse:
        return DomainRecordListResponse(
            items=[_record(DomainRecordResponse, document) for document in documents],
            total=total,
            limit=limit,
            offset=offset,
        )
