"""Queued processing-job API schemas."""

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
    claimed_at: datetime | None = None
    started_at: datetime | None = None
    heartbeat_at: datetime | None = None
    completed_at: datetime | None = None
    worker_id: str | None = None
    attempt_count: int = Field(ge=0)
    error_code: str | None = None
    error_message: str | None = None
    pipeline_version: str | None = None
    source_type: str | None = None
    source_artifact_id: Identifier | None = None
    result_artifact_ids: list[Identifier] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
