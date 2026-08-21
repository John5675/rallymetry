"""Shared one-match orchestration used locally and by Render Workflows."""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pickleball_vision.analysis_runtime.models import PipelineRunResult
from pickleball_vision.analysis_runtime.pipeline import PipelineRunner
from pickleball_vision.errors import (
    AnalysisExecutionError,
    AnalysisPipelineError,
    ArtifactStorageError,
    PersistenceOperationError,
)
from pickleball_vision.media import MediaMetadata, inspect_media
from pickleball_vision.persistence.artifacts import ArtifactStore
from pickleball_vision.persistence.models import (
    AnalyticsRecord,
    ArtifactCategory,
    ArtifactPutRequest,
    ArtifactRecord,
    Document,
    PlayerRecord,
    ProcessingJobRecord,
    ProcessingJobStatus,
    StructuredCollection,
    StructuredDomainRecord,
    artifact_record_from_document,
    processing_job_from_document,
)

LOGGER = logging.getLogger("pickleball_vision.workflows")
PUBLISHABLE_ARTIFACT_TYPES = frozenset(
    {
        "annotated_video",
        "ball_tracking_video",
        "court_visualization",
        "heatmap_me",
        "heatmap_opponent_1",
        "heatmap_opponent_2",
        "heatmap_partner",
        "player_topdown_video",
        "player_tracking_video",
        "rally_debug_video",
        "shot_debug_video",
        "shot_landing_map",
        "thumbnail",
        "topdown_video",
    }
)


class WorkflowPersistence(Protocol):
    async def get_match(self, match_id: str) -> Document | None: ...

    async def get_processing_job(self, job_id: str) -> Document | None: ...

    async def get_artifact(self, artifact_id: str) -> Document | None: ...

    async def update_processing_job(
        self,
        job_id: str,
        fields: Mapping[str, object],
        *,
        updated_at: datetime,
    ) -> Document | None: ...


class SourceStager(Protocol):
    async def stage(self, job: ProcessingJobRecord, *, workspace: Path) -> Path: ...


class SetupStager(Protocol):
    async def stage(
        self,
        *,
        match_id: str,
        match_document: Mapping[str, object],
        workspace: Path,
    ) -> Mapping[str, Path]: ...


class WorkflowResultPersistence(WorkflowPersistence, Protocol):
    async def save_players(self, records: Sequence[PlayerRecord]) -> None: ...

    async def save_rallies(self, records: Sequence[StructuredDomainRecord]) -> None: ...

    async def save_contacts(self, records: Sequence[StructuredDomainRecord]) -> None: ...

    async def save_bounces(self, records: Sequence[StructuredDomainRecord]) -> None: ...

    async def save_shots(self, records: Sequence[StructuredDomainRecord]) -> None: ...

    async def save_analytics(self, record: AnalyticsRecord) -> None: ...

    async def save_artifact(self, record: ArtifactRecord) -> None: ...

    async def find_processing_run_artifact(
        self,
        *,
        match_id: str,
        processing_run_id: str,
        artifact_type: str,
    ) -> Document | None: ...


class OnDemandAnalysisOrchestrator:
    """Coordinate existing pipeline stages without duplicating CV/audio algorithms."""

    def __init__(
        self,
        *,
        persistence: WorkflowResultPersistence,
        artifact_store: ArtifactStore,
        source_stager: SourceStager,
        setup_stager: SetupStager | None = None,
        pipeline_runner: PipelineRunner,
        pipeline_version: str,
        temp_root: Path,
        model_device: str,
        max_retries: int = 1,
        media_inspector: Callable[[Path], MediaMetadata] = inspect_media,
    ) -> None:
        self._persistence = persistence
        self._artifact_store = artifact_store
        self._source_stager = source_stager
        self._setup_stager = setup_stager
        self._pipeline_runner = pipeline_runner
        self._pipeline_version = pipeline_version
        self._temp_root = temp_root.expanduser().resolve()
        self._model_device = model_device
        self._max_retries = max_retries
        self._media_inspector = media_inspector

    async def execute(self, *, job_id: str, match_id: str) -> dict[str, str]:
        stage = ProcessingJobStatus.STARTING
        attempt = 1
        workspace = self._workspace(job_id)
        try:
            job, match_document, attempt = await self._load_and_start(
                job_id=job_id,
                match_id=match_id,
            )
            self._prepare_workspace(workspace)
            await self._stage(job_id, ProcessingJobStatus.DOWNLOADING_MEDIA, 0.05)
            stage = ProcessingJobStatus.DOWNLOADING_MEDIA
            source_path = await self._source_stager.stage(job, workspace=workspace)
            await self._stage(job_id, ProcessingJobStatus.PREPARING_MEDIA, 0.10)
            stage = ProcessingJobStatus.PREPARING_MEDIA
            if self._setup_stager is not None:
                await self._setup_stager.stage(
                    match_id=match_id,
                    match_document=match_document,
                    workspace=workspace,
                )
            media = await asyncio.to_thread(self._media_inspector, source_path)
            output_dir = workspace / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            (workspace / "working").mkdir(parents=True, exist_ok=True)

            async def on_stage(next_stage: ProcessingJobStatus, progress: float) -> None:
                nonlocal stage
                stage = next_stage
                await self._stage(job_id, next_stage, progress)

            result = await self._pipeline_runner.run(
                job,
                source_path=source_path,
                workspace=output_dir,
                on_stage=on_stage,
            )
            stage = ProcessingJobStatus.RENDERING_ARTIFACTS
            await self._stage(job_id, stage, 0.94)
            stage = ProcessingJobStatus.UPLOADING_RESULTS
            await self._stage(job_id, stage, 0.97)
            artifact_ids = await self._persist_results(job, result)
            completed_at = datetime.now(UTC)
            summary: dict[str, object] = {
                "artifactCount": len(artifact_ids),
                "audioAnalysisAvailable": media.audio is not None,
                "computeDevice": self._model_device.upper(),
                "processingRunId": job.processing_run_id or "",
            }
            updated = await self._persistence.update_processing_job(
                job_id,
                {
                    "status": ProcessingJobStatus.COMPLETE.value,
                    "stage": ProcessingJobStatus.COMPLETE.value,
                    "progress": 1.0,
                    "completedAt": completed_at,
                    "resultArtifactIds": list(artifact_ids),
                    "resultSummary": summary,
                },
                updated_at=completed_at,
            )
            if updated is None:
                raise AnalysisPipelineError("processing job disappeared before completion")
            return {"job_id": job_id, "match_id": match_id, "status": "COMPLETE"}
        except Exception as error:
            transient = isinstance(error, (ArtifactStorageError, PersistenceOperationError))
            if transient and attempt <= self._max_retries:
                await self._record_retryable_failure(job_id, stage, error)
                raise
            await self._record_terminal_failure(job_id, stage, error)
            return {"job_id": job_id, "match_id": match_id, "status": "FAILED"}
        finally:
            await asyncio.to_thread(self._cleanup_workspace, workspace)

    async def _load_and_start(
        self,
        *,
        job_id: str,
        match_id: str,
    ) -> tuple[ProcessingJobRecord, Document, int]:
        job_document = await self._persistence.get_processing_job(job_id)
        match_document = await self._persistence.get_match(match_id)
        if job_document is None:
            raise AnalysisPipelineError(f"processing job {job_id} does not exist")
        if match_document is None:
            raise AnalysisPipelineError(f"match {match_id} does not exist")
        job = processing_job_from_document(job_document)
        if job.match_id != match_id:
            raise AnalysisPipelineError("workflow identifiers refer to different matches")
        if job.processing_run_id is None:
            raise AnalysisPipelineError("processing job has no stable processingRunId")
        attempt = job.attempt_count + 1
        now = datetime.now(UTC)
        updated = await self._persistence.update_processing_job(
            job_id,
            {
                "status": ProcessingJobStatus.STARTING.value,
                "stage": ProcessingJobStatus.STARTING.value,
                "progress": 0.01,
                "startedAt": now,
                "attemptCount": attempt,
                "pipelineVersion": self._pipeline_version,
                "errorCode": None,
                "errorMessage": None,
            },
            updated_at=now,
        )
        if updated is None:
            raise AnalysisPipelineError(f"processing job {job_id} disappeared")
        return processing_job_from_document(updated), match_document, attempt

    async def _stage(
        self,
        job_id: str,
        stage: ProcessingJobStatus,
        progress: float,
    ) -> None:
        now = datetime.now(UTC)
        updated = await self._persistence.update_processing_job(
            job_id,
            {"status": stage.value, "stage": stage.value, "progress": progress},
            updated_at=now,
        )
        if updated is None:
            raise AnalysisPipelineError(f"processing job {job_id} disappeared during {stage.value}")

    async def _persist_results(
        self,
        job: ProcessingJobRecord,
        result: PipelineRunResult,
    ) -> tuple[str, ...]:
        if result.players:
            await self._persistence.save_players(result.players)
        writers = {
            StructuredCollection.RALLIES: self._persistence.save_rallies,
            StructuredCollection.CONTACTS: self._persistence.save_contacts,
            StructuredCollection.BOUNCES: self._persistence.save_bounces,
            StructuredCollection.SHOTS: self._persistence.save_shots,
        }
        for collection, records in result.structured.items():
            await writers[collection](records)
        if result.analytics is not None:
            await self._persistence.save_analytics(result.analytics)

        artifact_ids: list[str] = []
        run_id = job.processing_run_id
        assert run_id is not None
        for publication in result.artifacts:
            if (
                publication.category is not ArtifactCategory.VIEWABLE_MEDIA
                or publication.artifact_type not in PUBLISHABLE_ARTIFACT_TYPES
            ):
                continue
            previous = await self._persistence.find_processing_run_artifact(
                match_id=job.match_id,
                processing_run_id=run_id,
                artifact_type=publication.artifact_type,
            )
            if previous is not None:
                artifact = artifact_record_from_document(previous)
                if await self._artifact_store.exists(artifact):
                    artifact_ids.append(artifact.artifact_id)
                    continue
            artifact = await self._artifact_store.put(
                ArtifactPutRequest(
                    source_path=publication.source_path,
                    artifact_type=publication.artifact_type,
                    category=publication.category,
                    match_id=job.match_id,
                    access=publication.access,
                    pipeline_version=self._pipeline_version,
                    processing_run_id=run_id,
                )
            )
            await self._persistence.save_artifact(artifact)
            artifact_ids.append(artifact.artifact_id)
        return tuple(artifact_ids)

    async def _record_retryable_failure(
        self,
        job_id: str,
        stage: ProcessingJobStatus,
        error: Exception,
    ) -> None:
        now = datetime.now(UTC)
        await self._persistence.update_processing_job(
            job_id,
            {
                "stage": stage.value,
                "errorCode": "TRANSIENT_WORKFLOW_FAILURE",
                "errorMessage": self._safe_message(error),
            },
            updated_at=now,
        )

    async def _record_terminal_failure(
        self,
        job_id: str,
        stage: ProcessingJobStatus,
        error: Exception,
    ) -> None:
        now = datetime.now(UTC)
        code = (
            error.job_error_code if isinstance(error, AnalysisExecutionError) else "WORKFLOW_FAILED"
        )
        await self._persistence.update_processing_job(
            job_id,
            {
                "status": ProcessingJobStatus.FAILED.value,
                "stage": ProcessingJobStatus.FAILED.value,
                "failedAt": now,
                "failedStage": stage.value,
                "errorCode": code,
                "errorMessage": self._safe_message(error),
            },
            updated_at=now,
        )
        LOGGER.exception(
            "workflow_analysis_failed",
            extra={"context": {"jobId": job_id, "stage": stage.value, "errorCode": code}},
        )

    @staticmethod
    def _safe_message(error: Exception) -> str:
        if isinstance(error, AnalysisExecutionError):
            return str(error)[:512]
        return f"Analysis failed during on-demand workflow ({type(error).__name__})"

    def _workspace(self, job_id: str) -> Path:
        if not job_id.startswith("job_") or not job_id.replace("_", "").isalnum():
            raise AnalysisPipelineError("job ID is unsafe for temporary workspace creation")
        workspace = (self._temp_root / job_id).resolve()
        if not workspace.is_relative_to(self._temp_root):
            raise AnalysisPipelineError("temporary workspace escaped WORKFLOW_TEMP_DIR")
        return workspace

    def _prepare_workspace(self, workspace: Path) -> None:
        self._temp_root.mkdir(parents=True, exist_ok=True)
        self._cleanup_workspace(workspace)
        for name in ("input", "working", "output"):
            (workspace / name).mkdir(parents=True, exist_ok=True)

    def _cleanup_workspace(self, workspace: Path) -> None:
        if workspace == self._temp_root or not workspace.is_relative_to(self._temp_root):
            raise AnalysisPipelineError("refusing to clean an unscoped workflow path")
        shutil.rmtree(workspace, ignore_errors=True)
