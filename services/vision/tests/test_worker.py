from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from pickleball_vision.errors import WorkerPipelineError
from pickleball_vision.persistence.artifacts import LocalArtifactStore
from pickleball_vision.persistence.job_queue import MongoJobQueue
from pickleball_vision.persistence.models import (
    AnalyticsRecord,
    ArtifactAccess,
    ArtifactCategory,
    ArtifactProvider,
    ArtifactPutRequest,
    ArtifactRecord,
    Document,
    ProcessingJobRecord,
    ProcessingJobStatus,
    SourceMediaType,
    StructuredCollection,
    StructuredDomainRecord,
)
from pickleball_vision.worker.models import ArtifactPublication, PipelineRunResult
from pickleball_vision.worker.pipeline import PipelineRunner, StageCallback
from pickleball_vision.worker.service import AnalysisWorker
from pickleball_vision.worker.settings import WorkerSettings
from pickleball_vision.worker.source import SourceMediaStager
from test_job_queue import AtomicJobCollection

NOW = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)


class MemoryWorkerPersistence:
    def __init__(self) -> None:
        self.artifacts: list[ArtifactRecord] = []
        self.rallies: list[StructuredDomainRecord] = []
        self.analytics: list[AnalyticsRecord] = []

    async def get_artifact(self, artifact_id: str) -> Document | None:
        for artifact in self.artifacts:
            if artifact.artifact_id == artifact_id:
                return artifact.to_document()
        return None

    async def save_players(self, records: object) -> None:
        del records

    async def save_rallies(self, records: object) -> None:
        assert isinstance(records, tuple)
        self.rallies.extend(records)

    async def save_contacts(self, records: object) -> None:
        del records

    async def save_bounces(self, records: object) -> None:
        del records

    async def save_shots(self, records: object) -> None:
        del records

    async def save_analytics(self, record: AnalyticsRecord) -> None:
        self.analytics.append(record)

    async def save_artifact(self, record: ArtifactRecord) -> None:
        self.artifacts.append(record)


class SuccessfulRunner(PipelineRunner):
    async def run(
        self,
        job: ProcessingJobRecord,
        *,
        source_path: Path,
        workspace: Path,
        on_stage: StageCallback,
    ) -> PipelineRunResult:
        assert source_path.is_file()
        await on_stage(ProcessingJobStatus.PLAYER_PROCESSING, 0.1)
        await on_stage(ProcessingJobStatus.BALL_PROCESSING, 0.3)
        await on_stage(ProcessingJobStatus.AUDIO_PROCESSING, 0.4)
        await on_stage(ProcessingJobStatus.RALLY_PROCESSING, 0.5)
        await on_stage(ProcessingJobStatus.EVENT_PROCESSING, 0.6)
        await on_stage(ProcessingJobStatus.SHOT_PROCESSING, 0.7)
        await on_stage(ProcessingJobStatus.ANALYTICS, 0.8)
        review_video = workspace / "annotated.mp4"
        review_video.write_bytes(b"generated-review")
        rally = StructuredDomainRecord(
            match_id=job.match_id,
            record_id="rally-1",
            payload={"startTimestamp": 1.0, "endTimestamp": 3.0},
            confidence=0.8,
            timestamp_seconds=1.0,
            pipeline_version="pipeline-test",
        )
        analytics = AnalyticsRecord(
            match_id=job.match_id,
            analytics_id="analytics-1",
            calculation_version="analytics-v1",
            metrics={"rallyCount": {"value": 1}},
            pipeline_version="pipeline-test",
        )
        return PipelineRunResult(
            structured={StructuredCollection.RALLIES: (rally,)},
            analytics=analytics,
            artifacts=(
                ArtifactPublication(
                    source_path=review_video,
                    artifact_type="annotated_video",
                    category=ArtifactCategory.VIEWABLE_MEDIA,
                ),
            ),
        )


class FailingRunner(PipelineRunner):
    async def run(
        self,
        job: ProcessingJobRecord,
        *,
        source_path: Path,
        workspace: Path,
        on_stage: StageCallback,
    ) -> PipelineRunResult:
        del job, source_path, workspace
        await on_stage(ProcessingJobStatus.BALL_PROCESSING, 0.3)
        raise WorkerPipelineError("synthetic detector failure", stage="BALL_PROCESSING")


class FakeBlobStore:
    async def put(self, request: ArtifactPutRequest) -> ArtifactRecord:
        del request
        raise AssertionError("put is not used while staging")

    async def get(self, artifact: ArtifactRecord, destination: Path) -> Path:
        assert artifact.provider is ArtifactProvider.VERCEL_BLOB
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"downloaded-blob")
        return destination

    async def delete(self, artifact: ArtifactRecord) -> None:
        del artifact

    async def exists(self, artifact: ArtifactRecord) -> bool:
        del artifact
        return True


def _job(source: Path) -> ProcessingJobRecord:
    return ProcessingJobRecord(
        job_id="job-1",
        match_id="match-1",
        job_type="analyze_match",
        source_type=SourceMediaType.LOCAL_PATH,
        source_path=str(source),
        created_at=NOW,
        updated_at=NOW,
    )


def _worker(
    tmp_path: Path,
    *,
    runner: PipelineRunner,
) -> tuple[AnalysisWorker, AtomicJobCollection, MemoryWorkerPersistence]:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-is-never-modified")
    collection = AtomicJobCollection([_job(source).to_document()])
    queue = MongoJobQueue(collection)
    persistence = MemoryWorkerPersistence()
    store = LocalArtifactStore(tmp_path / "artifacts", token_factory=lambda: "fixed-token")
    settings = WorkerSettings(
        worker_id="worker-test",
        pipeline_plan_path=tmp_path / "unused-plan.json",
        work_root=tmp_path / "work",
        poll_interval_seconds=0.01,
        lease_timeout_seconds=10.0,
        heartbeat_interval_seconds=1.0,
        max_attempts=3,
    )
    worker = AnalysisWorker(
        settings=settings,
        pipeline_version="pipeline-test",
        queue=queue,
        persistence=persistence,
        source_stager=SourceMediaStager(persistence, store),
        pipeline_runner=runner,
        artifact_store=store,
    )
    return worker, collection, persistence


def test_worker_runs_existing_boundaries_publishes_and_completes(tmp_path: Path) -> None:
    worker, collection, persistence = _worker(tmp_path, runner=SuccessfulRunner())

    processed = asyncio.run(worker.run_once())

    document = collection.documents["job-1"]
    assert processed is True
    assert document["status"] == "COMPLETE"
    assert document["stage"] == "COMPLETE"
    assert document["progress"] == 1.0
    assert document["attemptCount"] == 1
    assert document["pipelineVersion"] == "pipeline-test"
    result_artifact_ids = document["resultArtifactIds"]
    assert isinstance(result_artifact_ids, list)
    assert len(result_artifact_ids) == 1
    assert [record.record_id for record in persistence.rallies] == ["rally-1"]
    assert persistence.analytics[0].metrics["rallyCount"] == {"value": 1}
    assert persistence.artifacts[0].category is ArtifactCategory.VIEWABLE_MEDIA
    assert (tmp_path / "source.mp4").read_bytes() == b"source-is-never-modified"


def test_worker_records_stage_error_and_does_not_retry_explicit_failure(tmp_path: Path) -> None:
    worker, collection, persistence = _worker(tmp_path, runner=FailingRunner())

    processed = asyncio.run(worker.run_once())
    second_attempt = asyncio.run(worker.run_once())

    document = collection.documents["job-1"]
    assert processed is True
    assert second_attempt is False
    assert document["status"] == "FAILED"
    assert document["stage"] == "BALL_PROCESSING"
    assert document["attemptCount"] == 1
    assert document["errorCode"] == "WORKER_PIPELINE"
    assert "synthetic detector failure" in str(document["errorMessage"])
    assert not persistence.artifacts


def test_source_stager_materializes_private_blob_without_modifying_manifest(tmp_path: Path) -> None:
    persistence = MemoryWorkerPersistence()
    artifact = ArtifactRecord(
        artifact_id="artifact-source",
        match_id="match-1",
        artifact_type="source_video",
        category=ArtifactCategory.SOURCE_MEDIA,
        pathname="source_media/match-1/random/source.mp4",
        provider=ArtifactProvider.VERCEL_BLOB,
        access=ArtifactAccess.PRIVATE,
        content_type="video/mp4",
        size_bytes=100,
        created_at=NOW,
        url="https://example.test/source.mp4",
    )
    persistence.artifacts.append(artifact)
    job = ProcessingJobRecord(
        job_id="job-blob",
        match_id="match-1",
        job_type="analyze_match",
        source_type=SourceMediaType.BLOB,
        source_artifact_id=artifact.artifact_id,
        created_at=NOW,
        updated_at=NOW,
    )

    staged = asyncio.run(
        SourceMediaStager(persistence, FakeBlobStore()).stage(job, workspace=tmp_path / "work")
    )

    assert staged.read_bytes() == b"downloaded-blob"
    assert artifact.pathname == "source_media/match-1/random/source.mp4"
