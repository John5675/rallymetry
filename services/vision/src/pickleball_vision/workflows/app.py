"""Render Workflow task registration entry point."""

from __future__ import annotations

from render_sdk import Retry, Workflows  # type: ignore[import-untyped]

from pickleball_vision.workflows.runtime import run_configured_analysis
from pickleball_vision.workflows.settings import (
    DEFAULT_WORKFLOW_MAX_RETRIES,
    DEFAULT_WORKFLOW_RETRY_WAIT_MS,
    workflow_task_plan,
    workflow_timeout_seconds,
)

tasks = Workflows()


@tasks.task(  # type: ignore[untyped-decorator]
    name="analyze_match",
    retry=Retry(
        max_retries=DEFAULT_WORKFLOW_MAX_RETRIES,
        wait_duration_ms=DEFAULT_WORKFLOW_RETRY_WAIT_MS,
        backoff_scaling=2.0,
    ),
    timeout_seconds=workflow_timeout_seconds(),
    plan=workflow_task_plan(),
)
async def analyze_match(job_id: str, match_id: str) -> dict[str, str]:
    """Process one match; large inputs and outputs stay in MongoDB/Blob."""

    return await run_configured_analysis(job_id=job_id, match_id=match_id)


app = Workflows.from_workflows(tasks)
app.start()
