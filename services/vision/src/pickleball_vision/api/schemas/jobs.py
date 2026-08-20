"""Queued processing-job API schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from pickleball_vision.api.schemas.common import ApiOutputModel, Identifier, JsonObject


class JobResponse(ApiOutputModel):
    job_id: Identifier
    match_id: Identifier
    job_type: str
    status: str
    progress: float = Field(ge=0, le=1)
    stage: str | None = None
    error: JsonObject | None = None
    created_at: datetime
    updated_at: datetime
