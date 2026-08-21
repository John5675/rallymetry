from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pickleball_vision.analysis_runtime.models import ArtifactPublication, PipelineRunResult
from pickleball_vision.analysis_runtime.pipeline import PipelineRunner, StageCallback
from pickleball_vision.errors import ArtifactStorageError
from pickleball_vision.media import MediaMetadata, MediaTimeline
from pickleball_vision.persistence.artifacts import ArtifactStore, LocalArtifactStore
from pickleball_vision.persistence.models import (
    AnalyticsRecord,
    ArtifactAccess,
    ArtifactCategory,
    ArtifactPutRequest,
    ArtifactRecord,
    Document,
    MatchRecord,
    PlayerRecord,
    ProcessingJobRecord,
    ProcessingJobStatus,
    SourceMediaType,
    StructuredCollection,
    StructuredDomainRecord,
)
from pickleball_vision.video import VideoMetadata
from pickleball_vision.workflows.orchestration import OnDemandAnalysisOrchestrator

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


class MemoryWorkflowPersistence:
    def __init__(self, job: ProcessingJobRecord) -> None:
        self.match = MatchRecord(
            match_id=job.match_id,
            created_at=NOW,
            updated_at=NOW,
        ).to_document()
        self.job = job.to_document()
        self.rallies: list[StructuredDomainRecord] = []
        self.analytics: list[AnalyticsRecord] = []
        self.artifacts: list[ArtifactRecord] = []
        self.updates: list[str] = []

    async def get_match(self, match_id: str) -> Document | None:
        return dict(self.match) if self.match["matchId"] == match_id else None

    async def get_processing_job(self, job_id: str) -> Document | None:
        return dict(self.job) if self.job["jobId"] == job_id else None

    async def get_artifact(self, artifact_id: str) -> Document | None:
        del artifact_id
        return None

    async def update_processing_job(
        self,
        job_id: str,
        fields: Mapping[str, object],
        *,
        updated_at: datetime,
    ) -> Document | None:
        if self.job["jobId"] != job_id:
            return None
        for key, value in fields.items():
            if value is None:
                self.job.pop(key, None)
            else:
                self.job[key] = value
        self.job["updatedAt"] = updated_at
        status = self.job.get("status")
        if isinstance(status, str):
            self.updates.append(status)
        return dict(self.job)

    async def save_players(self, records: Sequence[PlayerRecord]) -> None:
        del records

    async def save_rallies(self, records: Sequence[StructuredDomainRecord]) -> None:
        self.rallies = list(records)

    async def save_contacts(self, records: Sequence[StructuredDomainRecord]) -> None:
        del records

    async def save_bounces(self, records: Sequence[StructuredDomainRecord]) -> None:
        del records

    async def save_shots(self, records: Sequence[StructuredDomainRecord]) -> None:
        del records

    async def save_analytics(self, record: AnalyticsRecord) -> None:
        self.analytics = [record]

    async def save_artifact(self, record: ArtifactRecord) -> None:
        self.artifacts = [
            existing for existing in self.artifacts if existing.artifact_id != record.artifact_id
        ]
        self.artifacts.append(record)

    async def find_processing_run_artifact(
        self,
        *,
        match_id: str,
        processing_run_id: str,
        artifact_type: str,
    ) -> Document | None:
        for artifact in self.artifacts:
            if (
                artifact.match_id == match_id
                and artifact.processing_run_id == processing_run_id
                and artifact.artifact_type == artifact_type
            ):
                return artifact.to_document()
        return None


class FakeSourceStager:
    async def stage(self, job: ProcessingJobRecord, *, workspace: Path) -> Path:
        del job
        source = workspace / "input" / "source.mp4"
        source.write_bytes(b"source")
        return source


class SuccessfulPipeline(PipelineRunner):
    async def run(
        self,
        job: ProcessingJobRecord,
        *,
        source_path: Path,
        workspace: Path,
        on_stage: StageCallback,
    ) -> PipelineRunResult:
        assert source_path.parent.name == "input"
        await on_stage(ProcessingJobStatus.PLAYER_PROCESSING, 0.20)
        await on_stage(ProcessingJobStatus.BALL_PROCESSING, 0.40)
        await on_stage(ProcessingJobStatus.AUDIO_PROCESSING, 0.55)
        await on_stage(ProcessingJobStatus.RALLY_PROCESSING, 0.65)
        await on_stage(ProcessingJobStatus.BOUNCE_PROCESSING, 0.70)
        await on_stage(ProcessingJobStatus.CONTACT_PROCESSING, 0.75)
        await on_stage(ProcessingJobStatus.HITTER_PROCESSING, 0.80)
        await on_stage(ProcessingJobStatus.SHOT_PROCESSING, 0.85)
        await on_stage(ProcessingJobStatus.ANALYTICS, 0.90)
        video = workspace / "annotated.mp4"
        video.write_bytes(b"annotated")
        internal = workspace / "raw.json"
        internal.write_text("{}", encoding="utf-8")
        rally = StructuredDomainRecord(
            match_id=job.match_id,
            record_id="rally-1",
            payload={"rallyId": "rally-1"},
            processing_run_id=job.processing_run_id,
        )
        analytics = AnalyticsRecord(
            match_id=job.match_id,
            analytics_id="analytics",
            calculation_version="v1",
            metrics={"rallyCount": {"value": 1}},
            processing_run_id=job.processing_run_id,
        )
        return PipelineRunResult(
            structured={StructuredCollection.RALLIES: (rally,)},
            analytics=analytics,
            artifacts=(
                ArtifactPublication(
                    source_path=video,
                    artifact_type="annotated_video",
                    category=ArtifactCategory.VIEWABLE_MEDIA,
                    access=ArtifactAccess.LOCAL,
                ),
                ArtifactPublication(
                    source_path=internal,
                    artifact_type="raw_detections",
                    category=ArtifactCategory.INTERNAL_ARTIFACT,
                    access=ArtifactAccess.LOCAL,
                ),
            ),
        )


class FailingPipeline(PipelineRunner):
    async def run(
        self,
        job: ProcessingJobRecord,
        *,
        source_path: Path,
        workspace: Path,
        on_stage: StageCallback,
    ) -> PipelineRunResult:
        del job, source_path, workspace
        await on_stage(ProcessingJobStatus.BALL_PROCESSING, 0.40)
        raise ValueError("secret-looking low-level details")


class CountingStore(ArtifactStore):
    def __init__(self, root: Path) -> None:
        self.delegate = LocalArtifactStore(root)
        self.put_count = 0

    async def put(self, request: ArtifactPutRequest) -> ArtifactRecord:
        self.put_count += 1
        return await self.delegate.put(request)

    async def get(self, artifact: ArtifactRecord, destination: Path) -> Path:
        return await self.delegate.get(artifact, destination)

    async def delete(self, artifact: ArtifactRecord) -> None:
        await self.delegate.delete(artifact)

    async def exists(self, artifact: ArtifactRecord) -> bool:
        return await self.delegate.exists(artifact)


class FailOnceStore(CountingStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.failed = False

    async def put(self, request: ArtifactPutRequest) -> ArtifactRecord:
        if not self.failed:
            self.failed = True
            raise ArtifactStorageError("put", reason="synthetic transient outage")
        return await super().put(request)


def _job() -> ProcessingJobRecord:
    return ProcessingJobRecord(
        job_id="job_test",
        match_id="match_test",
        job_type="analyze_match",
        status=ProcessingJobStatus.QUEUED,
        stage=ProcessingJobStatus.QUEUED.value,
        processing_run_id="run_test",
        source_type=SourceMediaType.LOCAL_PATH,
        source_path="/unused/source.mp4",
        created_at=NOW,
        updated_at=NOW,
    )


def _no_audio_metadata(path: Path) -> MediaMetadata:
    return MediaMetadata(
        video=VideoMetadata(
            filename=path.name,
            path=path,
            width=1920,
            height=1080,
            fps=29.97,
            frame_count=300,
            duration=10.0,
            codec="h264",
        ),
        video_start_time_seconds=0.0,
        audio=None,
        timeline=MediaTimeline(),
        backend_name="synthetic",
        backend_version="1",
    )


def _orchestrator(
    tmp_path: Path,
    *,
    persistence: MemoryWorkflowPersistence,
    runner: PipelineRunner,
    store: CountingStore,
) -> OnDemandAnalysisOrchestrator:
    return OnDemandAnalysisOrchestrator(
        persistence=persistence,
        artifact_store=store,
        source_stager=FakeSourceStager(),
        pipeline_runner=runner,
        pipeline_version="pipeline-test",
        temp_root=tmp_path / "rallymetry",
        model_device="cpu",
        media_inspector=_no_audio_metadata,
    )


def test_workflow_persists_results_uploads_allow_list_and_cleans_temp(tmp_path: Path) -> None:
    persistence = MemoryWorkflowPersistence(_job())
    store = CountingStore(tmp_path / "artifacts")
    result = asyncio.run(
        _orchestrator(
            tmp_path,
            persistence=persistence,
            runner=SuccessfulPipeline(),
            store=store,
        ).execute(job_id="job_test", match_id="match_test")
    )

    assert result["status"] == "COMPLETE"
    assert persistence.job["status"] == "COMPLETE"
    assert persistence.updates[-2:] == ["UPLOADING_RESULTS", "COMPLETE"]
    assert persistence.rallies[0].processing_run_id == "run_test"
    assert persistence.analytics[0].processing_run_id == "run_test"
    assert [item.artifact_type for item in persistence.artifacts] == ["annotated_video"]
    assert persistence.job["resultSummary"] == {
        "artifactCount": 1,
        "audioAnalysisAvailable": False,
        "computeDevice": "CPU",
        "processingRunId": "run_test",
    }
    assert not (tmp_path / "rallymetry" / "job_test").exists()


def test_workflow_failure_is_safe_terminal_and_cleans_temp(tmp_path: Path) -> None:
    persistence = MemoryWorkflowPersistence(_job())
    store = CountingStore(tmp_path / "artifacts")
    result = asyncio.run(
        _orchestrator(
            tmp_path,
            persistence=persistence,
            runner=FailingPipeline(),
            store=store,
        ).execute(job_id="job_test", match_id="match_test")
    )

    assert result["status"] == "FAILED"
    assert persistence.job["status"] == "FAILED"
    assert persistence.job["failedStage"] == "BALL_PROCESSING"
    assert "secret-looking" not in str(persistence.job["errorMessage"])
    assert not (tmp_path / "rallymetry" / "job_test").exists()


def test_workflow_retry_is_idempotent_for_structured_and_blob_results(tmp_path: Path) -> None:
    persistence = MemoryWorkflowPersistence(_job())
    store = CountingStore(tmp_path / "artifacts")
    orchestrator = _orchestrator(
        tmp_path,
        persistence=persistence,
        runner=SuccessfulPipeline(),
        store=store,
    )

    asyncio.run(orchestrator.execute(job_id="job_test", match_id="match_test"))
    asyncio.run(orchestrator.execute(job_id="job_test", match_id="match_test"))

    assert len(persistence.rallies) == 1
    assert len(persistence.analytics) == 1
    assert len(persistence.artifacts) == 1
    assert store.put_count == 1
    assert persistence.job["attemptCount"] == 2


def test_transient_upload_failure_retries_same_run_and_completes(tmp_path: Path) -> None:
    persistence = MemoryWorkflowPersistence(_job())
    store = FailOnceStore(tmp_path / "artifacts")
    orchestrator = _orchestrator(
        tmp_path,
        persistence=persistence,
        runner=SuccessfulPipeline(),
        store=store,
    )

    try:
        asyncio.run(orchestrator.execute(job_id="job_test", match_id="match_test"))
    except ArtifactStorageError:
        pass
    else:
        raise AssertionError("the first transient upload must be raised for Render retry")

    result = asyncio.run(orchestrator.execute(job_id="job_test", match_id="match_test"))

    assert result["status"] == "COMPLETE"
    assert persistence.job["processingRunId"] == "run_test"
    assert persistence.job["attemptCount"] == 2
    assert len(persistence.rallies) == len(persistence.artifacts) == 1
    assert not (tmp_path / "rallymetry" / "job_test").exists()
