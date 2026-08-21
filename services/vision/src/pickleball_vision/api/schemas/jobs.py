"""On-demand workflow processing-job API schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from pickleball_vision.api.schemas.common import ApiOutputModel, Identifier


class JobResponse(ApiOutputModel):
    job_id: Identifier
    match_id: Identifier
    job_type: str
    status: str
    progress: float = Field(ge=0, le=1)
    stage: str | None = None
    render_triggered_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    failed_stage: str | None = None
    render_task_run_id: Identifier | None = None
    processing_run_id: Identifier | None = None
    attempt_count: int = Field(ge=0)
    error_code: str | None = None
    error_message: str | None = None
    pipeline_version: str | None = None
    source_type: str | None = None
    source_artifact_id: Identifier | None = None
    result_artifact_ids: list[Identifier] = Field(default_factory=list)
    result_summary: dict[str, object] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
