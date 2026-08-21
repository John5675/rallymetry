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
    YouTubeMatchSubmitRequest,
    YouTubeMatchSubmitResponse,
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
from pickleball_vision.api.services.render_workflows import AnalysisWorkflowClient
from pickleball_vision.api.youtube import parse_youtube_video_id
from pickleball_vision.persistence.models import (
    ArtifactAccess,
    ArtifactCategory,
    ArtifactProvider,
    Document,
    MatchRecord,
    ProcessingJobRecord,
    ProcessingJobStatus,
    SourceMediaType,
    artifact_record_from_document,
)

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
REQUIRED_ANALYSIS_SETUP = frozenset(
    {
        "ballExperimentArtifactId",
        "ballWeightsArtifactId",
        "calibrationArtifactId",
        "playerAssignmentsArtifactId",
    }
)


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

    def __init__(
        self,
        persistence: ApplicationPersistence,
        *,
        workflow_client: AnalysisWorkflowClient | None = None,
        default_analysis_profile_match_id: str | None = None,
    ) -> None:
        self._persistence = persistence
        self._workflow_client = workflow_client
        self._default_analysis_profile_match_id = default_analysis_profile_match_id

    async def create_match(self, request: MatchCreateRequest) -> MatchResponse:
        now = datetime.now(UTC)
        record = MatchRecord(
            match_id=f"match_{uuid.uuid4().hex}",
            title=request.title,
            youtube_video_id=request.youtube_video_id,
            source_artifact_id=request.source_artifact_id,
            analysis_setup=request.analysis_setup,
            created_at=now,
            updated_at=now,
        )
        await self._persistence.save_match(record)
        return _record(MatchResponse, record.to_document())

    async def submit_youtube_match(
        self,
        request: YouTubeMatchSubmitRequest,
    ) -> YouTubeMatchSubmitResponse:
        """Create and queue one authorized YouTube recording without downloading in HTTP."""

        youtube_video_id = parse_youtube_video_id(request.youtube_url)
        existing = await self._persistence.get_match_by_youtube_video_id(youtube_video_id)
        if existing is not None:
            match_id = existing.get("matchId")
            raise ApiError(
                status_code=409,
                code="youtube_match_exists",
                message="This YouTube recording already exists in Rallymetry",
                details={"matchId": match_id if isinstance(match_id, str) else ""},
            )
        profile_match_id = self._default_analysis_profile_match_id
        if profile_match_id is None:
            raise ApiError(
                status_code=503,
                code="analysis_profile_unavailable",
                message="One-click analysis has no configured court and model profile",
            )
        profile = await self._persistence.get_match(profile_match_id)
        if profile is None:
            raise ApiError(
                status_code=503,
                code="analysis_profile_unavailable",
                message="The configured court and model profile is unavailable",
            )
        match_id = f"match_{uuid.uuid4().hex}"
        now = datetime.now(UTC)
        setup = await self._shared_analysis_setup(
            profile=profile,
        )
        record = MatchRecord(
            match_id=match_id,
            title=request.title or f"YouTube match {youtube_video_id}",
            youtube_video_id=youtube_video_id,
            analysis_profile_match_id=profile_match_id,
            analysis_setup=setup,
            summary={"status": ProcessingJobStatus.CREATED.value},
            created_at=now,
            updated_at=now,
        )
        await self._persistence.save_match(record)
        job = await self.queue_processing(match_id)
        return YouTubeMatchSubmitResponse(
            match=_record(MatchResponse, record.to_document()),
            job=job,
        )

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
        raw_youtube_video_id = match.get("youtubeVideoId")
        youtube_video_id = raw_youtube_video_id if isinstance(raw_youtube_video_id, str) else None
        if isinstance(source_artifact_id, str) and source_artifact_id:
            artifact = await self._persistence.get_artifact(source_artifact_id)
            if artifact is None or artifact.get("category") != ArtifactCategory.SOURCE_MEDIA.value:
                raise ApiError(
                    status_code=409,
                    code="source_media_unavailable",
                    message="The match source-media artifact is unavailable",
                )
            provider = artifact.get("provider")
            if provider == ArtifactProvider.VERCEL_BLOB.value:
                source_type = SourceMediaType.BLOB
            else:
                raise ApiError(
                    status_code=409,
                    code="source_media_unavailable",
                    message="On-demand analysis requires a hosted source-media artifact",
                )
        elif youtube_video_id:
            source_type = SourceMediaType.YOUTUBE
            source_artifact_id = None
        else:
            raise ApiError(
                status_code=409,
                code="source_media_required",
                message="The match requires source media or a YouTube video ID before processing",
            )
        setup = match.get("analysisSetup")
        if not isinstance(setup, dict):
            setup = {}
        missing_setup = sorted(REQUIRED_ANALYSIS_SETUP.difference(setup))
        if missing_setup:
            raise ApiError(
                status_code=409,
                code="analysis_setup_required",
                message="The match requires calibration, player assignments, and ball-model setup",
                details={"missing": missing_setup},
            )
        for name in sorted(REQUIRED_ANALYSIS_SETUP):
            setup_artifact_id = setup.get(name)
            if not isinstance(setup_artifact_id, str) or not setup_artifact_id:
                raise ApiError(
                    status_code=409,
                    code="analysis_setup_invalid",
                    message="The match analysis setup contains an invalid artifact reference",
                    details={"field": name},
                )
            setup_artifact = await self._persistence.get_artifact(setup_artifact_id)
            profile_match_id = match.get("analysisProfileMatchId")
            allowed_owners = {match_id}
            if isinstance(profile_match_id, str) and profile_match_id:
                allowed_owners.add(profile_match_id)
            if (
                setup_artifact is None
                or setup_artifact.get("matchId") not in allowed_owners
                or setup_artifact.get("category") != ArtifactCategory.INTERNAL_ARTIFACT.value
                or setup_artifact.get("provider") != ArtifactProvider.VERCEL_BLOB.value
            ):
                raise ApiError(
                    status_code=409,
                    code="analysis_setup_unavailable",
                    message="A required private analysis setup artifact is unavailable",
                    details={"field": name},
                )
        if self._workflow_client is None:
            raise ApiError(
                status_code=503,
                code="workflow_unavailable",
                message="On-demand analysis is not configured",
            )
        now = datetime.now(UTC)
        processing_run_id = f"run_{uuid.uuid4().hex}"
        record = ProcessingJobRecord(
            job_id=f"job_{uuid.uuid4().hex}",
            match_id=match_id,
            job_type="analyze_match",
            status=ProcessingJobStatus.CREATED,
            stage=ProcessingJobStatus.CREATED.value,
            processing_run_id=processing_run_id,
            source_type=source_type,
            source_artifact_id=source_artifact_id,
            youtube_video_id=(youtube_video_id if source_type is SourceMediaType.YOUTUBE else None),
            created_at=now,
            updated_at=now,
        )
        document, created = await self._persistence.create_processing_job_if_no_active(record)
        if not created:
            return _record(JobResponse, document)
        try:
            workflow_run = await self._workflow_client.start_analysis(
                job_id=record.job_id,
                match_id=match_id,
            )
        except Exception as error:
            failed_at = datetime.now(UTC)
            await self._persistence.update_processing_job(
                record.job_id,
                {
                    "status": ProcessingJobStatus.FAILED.value,
                    "stage": ProcessingJobStatus.FAILED.value,
                    "progress": 0.0,
                    "failedAt": failed_at,
                    "failedStage": ProcessingJobStatus.CREATED.value,
                    "errorCode": "RENDER_TRIGGER_FAILED",
                    "errorMessage": "Unable to queue on-demand analysis",
                },
                updated_at=failed_at,
            )
            raise ApiError(
                status_code=503,
                code="workflow_trigger_failed",
                message="Unable to queue on-demand analysis",
                details={"exceptionType": type(error).__name__},
            ) from error
        triggered_at = datetime.now(UTC)
        queued = await self._persistence.update_processing_job(
            record.job_id,
            {
                "status": ProcessingJobStatus.QUEUED.value,
                "stage": ProcessingJobStatus.QUEUED.value,
                "renderTriggeredAt": triggered_at,
                "renderTaskRunId": workflow_run.run_id,
            },
            updated_at=triggered_at,
        )
        if queued is None:
            raise ApiError(
                status_code=503,
                code="workflow_job_update_failed",
                message="Analysis was queued but its task-run ID could not be persisted",
            )
        return _record(JobResponse, queued)

    async def get_job(self, job_id: str) -> JobResponse:
        document = await self._persistence.get_processing_job(job_id)
        if document is None:
            raise ResourceNotFoundError("processing job", job_id)
        return _record(JobResponse, document)

    async def get_latest_match_job(self, match_id: str) -> JobResponse:
        await self._require_match(match_id)
        document = await self._persistence.get_latest_processing_job_for_match(match_id)
        if document is None:
            raise ResourceNotFoundError("processing job", match_id)
        return _record(JobResponse, document)

    async def _shared_analysis_setup(
        self,
        *,
        profile: Document,
    ) -> dict[str, str]:
        raw_setup = profile.get("analysisSetup")
        if not isinstance(raw_setup, dict):
            raw_setup = {}
        missing = sorted(REQUIRED_ANALYSIS_SETUP.difference(raw_setup))
        if missing:
            raise ApiError(
                status_code=503,
                code="analysis_profile_invalid",
                message="The configured analysis profile is incomplete",
                details={"missing": missing},
            )
        shared: dict[str, str] = {}
        for field in sorted(REQUIRED_ANALYSIS_SETUP):
            artifact_id = raw_setup.get(field)
            document = (
                await self._persistence.get_artifact(artifact_id)
                if isinstance(artifact_id, str)
                else None
            )
            if document is None:
                raise ApiError(
                    status_code=503,
                    code="analysis_profile_invalid",
                    message="A configured analysis profile artifact is unavailable",
                    details={"field": field},
                )
            source = artifact_record_from_document(document)
            if (
                source.category is not ArtifactCategory.INTERNAL_ARTIFACT
                or source.provider is not ArtifactProvider.VERCEL_BLOB
                or source.access is not ArtifactAccess.PRIVATE
            ):
                raise ApiError(
                    status_code=503,
                    code="analysis_profile_invalid",
                    message="A configured analysis profile artifact is not private hosted setup",
                    details={"field": field},
                )
            shared[field] = source.artifact_id
        return shared

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
