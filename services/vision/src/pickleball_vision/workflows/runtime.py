"""Production dependency assembly for one Render Workflow task run."""

from __future__ import annotations

from datetime import UTC, datetime

from pickleball_vision.analysis_runtime.pipeline import (
    PlannedCliPipelineRunner,
    load_pipeline_plan,
)
from pickleball_vision.analysis_runtime.source import SourceMediaStager
from pickleball_vision.analysis_runtime.youtube import YtDlpYouTubeDownloader
from pickleball_vision.config import ArtifactBackend, PersistenceSettings
from pickleball_vision.errors import AnalysisConfigurationError
from pickleball_vision.persistence.artifacts import create_artifact_store
from pickleball_vision.persistence.models import ProcessingJobStatus
from pickleball_vision.persistence.mongodb import MongoPersistence
from pickleball_vision.workflows.orchestration import OnDemandAnalysisOrchestrator
from pickleball_vision.workflows.settings import WorkflowSettings
from pickleball_vision.workflows.setup import MatchSetupStager


async def run_configured_analysis(*, job_id: str, match_id: str) -> dict[str, str]:
    """Run one identifier-addressed analysis using hosted adapters."""

    persistence_settings = PersistenceSettings.from_env()
    persistence = await MongoPersistence.connect_from_settings(persistence_settings)
    try:
        await persistence.initialize_indexes()
        try:
            workflow_settings = WorkflowSettings.from_env()
            if persistence_settings.artifact_backend is not ArtifactBackend.VERCEL_BLOB:
                raise AnalysisConfigurationError(
                    "Render Workflow requires PICKLEBALL_VISION_ARTIFACT_BACKEND=vercel_blob"
                )
            plan = load_pipeline_plan(workflow_settings.pipeline_config_path)
            artifact_store = create_artifact_store(persistence_settings)
        except AnalysisConfigurationError as error:
            now = datetime.now(UTC)
            await persistence.update_processing_job(
                job_id,
                {
                    "status": ProcessingJobStatus.FAILED.value,
                    "stage": ProcessingJobStatus.FAILED.value,
                    "failedAt": now,
                    "failedStage": ProcessingJobStatus.STARTING.value,
                    "errorCode": error.job_error_code,
                    "errorMessage": str(error)[:512],
                },
                updated_at=now,
            )
            return {"job_id": job_id, "match_id": match_id, "status": "FAILED"}
        orchestrator = OnDemandAnalysisOrchestrator(
            persistence=persistence,
            artifact_store=artifact_store,
            source_stager=SourceMediaStager(
                persistence,
                artifact_store,
                youtube_downloader=YtDlpYouTubeDownloader(
                    max_duration_seconds=workflow_settings.youtube_max_duration_seconds,
                    max_bytes=workflow_settings.youtube_max_bytes,
                    pot_provider_url=workflow_settings.youtube_pot_provider_url,
                ),
            ),
            setup_stager=MatchSetupStager(persistence, artifact_store),
            pipeline_runner=PlannedCliPipelineRunner(
                plan,
                environment={
                    "MODEL_DEVICE": workflow_settings.model_device,
                    "PICKLEBALL_VISION_PERSON_DEVICE": workflow_settings.model_device,
                },
            ),
            pipeline_version=plan.pipeline_version,
            temp_root=workflow_settings.temp_root,
            model_device=workflow_settings.model_device,
        )
        return await orchestrator.execute(job_id=job_id, match_id=match_id)
    finally:
        await persistence.close()
