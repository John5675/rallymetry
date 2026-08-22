"""Application orchestration over compact persistence records only."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal, TypeVar, cast

from pydantic import BaseModel, ValidationError

from pickleball_vision.api.errors import ApiError, ResourceNotFoundError
from pickleball_vision.api.schemas.corrections import (
    CorrectionCreateRequest,
    CorrectionListResponse,
    CorrectionPatchRequest,
    CorrectionResponse,
)
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
from pickleball_vision.correction_analytics import correction_aware_analytics
from pickleball_vision.corrections import (
    TARGET_COLLECTION,
    CorrectionType,
    apply_verified_corrections,
    prediction_snapshot,
    prediction_version,
    validate_human_correction,
)
from pickleball_vision.persistence.models import (
    ArtifactAccess,
    ArtifactCategory,
    ArtifactProvider,
    CorrectionRecord,
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
        corrections = await self._persistence.list_match_corrections(match_id)
        items = [
            _record(PlayerResponse, apply_verified_corrections(document, corrections))
            for document in documents
        ]
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
        corrections = await self._persistence.list_match_corrections(match_id)
        return self._domain_page(
            documents,
            corrections=corrections,
            total=total,
            limit=limit,
            offset=offset,
        )

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
        corrections = await self._persistence.list_match_corrections(match_id)
        return self._domain_page(
            documents,
            corrections=corrections,
            total=total,
            limit=limit,
            offset=offset,
        )

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
        corrections = await self._persistence.list_match_corrections(match_id)
        return self._domain_page(
            documents,
            corrections=corrections,
            total=total,
            limit=limit,
            offset=offset,
        )

    async def list_corrections(self, match_id: str) -> CorrectionListResponse:
        await self._require_match(match_id)
        documents = await self._persistence.list_match_corrections(match_id)
        items = [_record(CorrectionResponse, document) for document in documents]
        return CorrectionListResponse(items=items, total=len(items))

    async def create_correction(
        self,
        match_id: str,
        request: CorrectionCreateRequest,
    ) -> CorrectionResponse:
        await self._require_match(match_id)
        existing = await self._persistence.list_match_corrections(match_id)
        duplicate = next(
            (
                item
                for item in existing
                if item.get("correctionType") == request.correction_type.value
                and item.get("targetRecordId") == request.target_record_id
            ),
            None,
        )
        if duplicate is not None:
            raise ApiError(
                status_code=409,
                code="correction_exists",
                message="An active correction already exists for this prediction",
                details={"correctionId": duplicate.get("correctionId", "")},
            )
        target = await self._require_correction_target(
            match_id,
            request.correction_type,
            request.target_record_id,
        )
        human = validate_human_correction(
            request.correction_type,
            cast(dict[str, object], request.human_correction),
        )
        now = datetime.now(UTC)
        confidence = _prediction_confidence(target)
        record = CorrectionRecord(
            correction_id=f"correction_{uuid.uuid4().hex}",
            match_id=match_id,
            correction_type=request.correction_type.value,
            target_collection=TARGET_COLLECTION[request.correction_type],
            target_record_id=request.target_record_id,
            prediction=prediction_snapshot(request.correction_type, target),
            prediction_confidence=confidence,
            prediction_version=prediction_version(target),
            human_correction=human,
            verified=request.verified,
            reason=request.reason,
            corrected_by=request.corrected_by,
            visual_evidence=request.visual_evidence,
            audio_evidence=request.audio_evidence,
            created_at=now,
            corrected_at=now,
            updated_at=now,
        )
        await self._persistence.save_correction(record)
        return _record(CorrectionResponse, record.to_document())

    async def update_correction(
        self,
        match_id: str,
        correction_id: str,
        request: CorrectionPatchRequest,
    ) -> CorrectionResponse:
        await self._require_match(match_id)
        existing = await self._require_correction(match_id, correction_id)
        if existing.get("active") is not True:
            raise ApiError(
                status_code=409,
                code="correction_removed",
                message="A removed correction cannot be edited",
            )
        correction_type = CorrectionType(str(existing["correctionType"]))
        fields = request.model_fields_set
        raw_human = (
            request.human_correction
            if "human_correction" in fields
            else existing.get("humanCorrection")
        )
        if not isinstance(raw_human, dict):
            raise ApiError(
                status_code=500,
                code="persisted_record_invalid",
                message="The correction record has an invalid semantic value",
            )
        human = validate_human_correction(correction_type, cast(dict[str, object], raw_human))
        now = datetime.now(UTC)
        history = _correction_history(existing, now)
        record = _updated_correction_record(
            existing,
            human_correction=human,
            verified=(
                request.verified
                if "verified" in fields and request.verified is not None
                else bool(existing["verified"])
            ),
            reason=(request.reason if "reason" in fields else _optional_string(existing, "reason")),
            corrected_by=(
                request.corrected_by
                if "corrected_by" in fields
                else _optional_string(existing, "correctedBy")
            ),
            visual_evidence=(
                cast(dict[str, object] | None, request.visual_evidence)
                if "visual_evidence" in fields
                else _optional_mapping(existing, "visualEvidence")
            ),
            audio_evidence=(
                cast(dict[str, object] | None, request.audio_evidence)
                if "audio_evidence" in fields
                else _optional_mapping(existing, "audioEvidence")
            ),
            history=history,
            now=now,
        )
        await self._persistence.save_correction(record)
        return _record(CorrectionResponse, record.to_document())

    async def remove_correction(self, match_id: str, correction_id: str) -> None:
        await self._require_match(match_id)
        existing = await self._require_correction(match_id, correction_id)
        if existing.get("active") is not True:
            raise ResourceNotFoundError("correction", correction_id)
        now = datetime.now(UTC)
        record = _updated_correction_record(
            existing,
            human_correction=cast(dict[str, object], existing["humanCorrection"]),
            verified=bool(existing["verified"]),
            reason=_optional_string(existing, "reason"),
            corrected_by=_optional_string(existing, "correctedBy"),
            visual_evidence=_optional_mapping(existing, "visualEvidence"),
            audio_evidence=_optional_mapping(existing, "audioEvidence"),
            history=_correction_history(existing, now),
            now=now,
            active=False,
            deleted_at=now,
        )
        await self._persistence.save_correction(record)

    async def _require_correction(self, match_id: str, correction_id: str) -> Document:
        document = await self._persistence.get_correction(correction_id)
        if document is None or document.get("matchId") != match_id:
            raise ResourceNotFoundError("correction", correction_id)
        return document

    async def _require_correction_target(
        self,
        match_id: str,
        correction_type: CorrectionType,
        target_record_id: str,
    ) -> Document:
        collection = TARGET_COLLECTION[correction_type]
        if collection == "players":
            document = await self._persistence.get_match_player(match_id, target_record_id)
        else:
            document = await self._persistence.get_match_domain_record(
                collection,
                match_id,
                target_record_id,
            )
        if document is None:
            raise ResourceNotFoundError("correction target", target_record_id)
        return document

    async def get_analytics(self, match_id: str) -> AnalyticsResponse:
        await self._require_match(match_id)
        document = await self._persistence.get_latest_match_analytics(match_id)
        if document is None:
            raise ResourceNotFoundError("analytics", match_id)
        effective = correction_aware_analytics(
            document,
            players=await self._persistence.list_match_players(match_id),
            rallies=await self._all_domain_records(match_id, collection="rallies"),
            shots=await self._all_domain_records(match_id, collection="shots"),
            corrections=await self._persistence.list_match_corrections(match_id),
        )
        return _record(AnalyticsResponse, effective)

    async def _all_domain_records(
        self,
        match_id: str,
        *,
        collection: Literal["rallies", "shots"],
    ) -> tuple[Document, ...]:
        records: list[Document] = []
        offset = 0
        total = 1
        while offset < total:
            if collection == "rallies":
                page, total = await self._persistence.list_match_rallies(
                    match_id,
                    limit=100,
                    offset=offset,
                )
            else:
                page, total = await self._persistence.list_match_shots(
                    match_id,
                    limit=100,
                    offset=offset,
                )
            if not page:
                break
            records.extend(page)
            offset += len(page)
        return tuple(records)

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
        queue_fields: dict[str, object] = {
            "status": ProcessingJobStatus.QUEUED.value,
            "stage": ProcessingJobStatus.QUEUED.value,
        }
        if workflow_run.run_id is not None:
            queue_fields.update(
                {
                    "renderTriggeredAt": triggered_at,
                    "renderTaskRunId": workflow_run.run_id,
                }
            )
        queued = await self._persistence.update_processing_job(
            record.job_id,
            queue_fields,
            updated_at=triggered_at,
        )
        if queued is None:
            raise ApiError(
                status_code=503,
                code="workflow_job_update_failed",
                message="Analysis was queued but its job state could not be persisted",
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
        corrections: tuple[Document, ...],
        total: int,
        limit: int,
        offset: int,
    ) -> DomainRecordListResponse:
        return DomainRecordListResponse(
            items=[
                _record(DomainRecordResponse, apply_verified_corrections(document, corrections))
                for document in documents
            ],
            total=total,
            limit=limit,
            offset=offset,
        )


def _prediction_confidence(document: Document) -> float | None:
    value = document.get("confidence")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        payload = document.get("payload")
        value = payload.get("confidence") if isinstance(payload, dict) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _optional_string(document: Document, key: str) -> str | None:
    value = document.get(key)
    return value if isinstance(value, str) else None


def _optional_mapping(document: Document, key: str) -> dict[str, object] | None:
    value = document.get(key)
    return cast(dict[str, object], value) if isinstance(value, dict) else None


def _correction_history(document: Document, changed_at: datetime) -> tuple[dict[str, object], ...]:
    raw_value = document.get("history", [])
    raw_history = raw_value if isinstance(raw_value, list) else []
    history = [dict(item) for item in raw_history if isinstance(item, dict)]
    history.append(
        {
            "revision": _revision(document),
            "humanCorrection": dict(cast(dict[str, object], document["humanCorrection"])),
            "verified": bool(document.get("verified")),
            "correctedAt": str(document.get("correctedAt")),
            "supersededAt": changed_at.isoformat(),
        }
    )
    return tuple(history)


def _updated_correction_record(
    document: Document,
    *,
    human_correction: dict[str, object],
    verified: bool,
    reason: str | None,
    corrected_by: str | None,
    visual_evidence: dict[str, object] | None,
    audio_evidence: dict[str, object] | None,
    history: tuple[dict[str, object], ...],
    now: datetime,
    active: bool = True,
    deleted_at: datetime | None = None,
) -> CorrectionRecord:
    return CorrectionRecord(
        correction_id=str(document["correctionId"]),
        match_id=str(document["matchId"]),
        correction_type=str(document["correctionType"]),
        target_collection=str(document["targetCollection"]),
        target_record_id=str(document["targetRecordId"]),
        prediction=cast(dict[str, object], document["prediction"]),
        prediction_confidence=cast(float | None, document.get("predictionConfidence")),
        prediction_version=_optional_string(document, "predictionVersion"),
        human_correction=human_correction,
        verified=verified,
        reason=reason,
        corrected_by=corrected_by,
        visual_evidence=visual_evidence,
        audio_evidence=audio_evidence,
        active=active,
        revision=_revision(document) + 1,
        history=history,
        created_at=cast(datetime, document["createdAt"]),
        corrected_at=now,
        updated_at=now,
        deleted_at=deleted_at,
    )


def _revision(document: Document) -> int:
    value = document.get("revision")
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 1
