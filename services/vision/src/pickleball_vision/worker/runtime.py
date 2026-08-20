"""Production dependency assembly for the standalone worker executable boundary."""

from __future__ import annotations

from pickleball_vision.config import PersistenceSettings
from pickleball_vision.persistence.artifacts import create_artifact_store
from pickleball_vision.persistence.mongodb import MongoPersistence
from pickleball_vision.worker.pipeline import PlannedCliPipelineRunner, load_pipeline_plan
from pickleball_vision.worker.service import AnalysisWorker
from pickleball_vision.worker.settings import WorkerSettings
from pickleball_vision.worker.source import SourceMediaStager


async def run_configured_worker(
    *,
    worker_settings: WorkerSettings,
    persistence_settings: PersistenceSettings,
    once: bool,
) -> bool:
    """Connect outbound adapters, then process once or poll continuously."""

    plan = load_pipeline_plan(worker_settings.pipeline_plan_path)
    persistence = await MongoPersistence.connect_from_settings(persistence_settings)
    try:
        await persistence.initialize_indexes()
        artifact_store = create_artifact_store(persistence_settings)
        worker = AnalysisWorker(
            settings=worker_settings,
            pipeline_version=plan.pipeline_version,
            queue=persistence.job_queue(),
            persistence=persistence,
            source_stager=SourceMediaStager(persistence, artifact_store),
            pipeline_runner=PlannedCliPipelineRunner(plan),
            artifact_store=artifact_store,
        )
        if once:
            return await worker.run_once()
        await worker.run_forever()
        return True
    finally:
        await persistence.close()
