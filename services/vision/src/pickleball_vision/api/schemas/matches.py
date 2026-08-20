"""Match request and response schemas independent of BSON."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from pickleball_vision.api.schemas.common import (
    ApiInputModel,
    ApiOutputModel,
    Identifier,
    JsonObject,
    ShortText,
)


class MatchCreateRequest(ApiInputModel):
    title: ShortText | None = None
    youtube_video_id: Identifier | None = None
    source_artifact_id: Identifier | None = None


class MatchPatchRequest(ApiInputModel):
    title: ShortText | None = None
    youtube_video_id: Identifier | None = None
    source_artifact_id: Identifier | None = None

    @model_validator(mode="after")
    def require_change(self) -> MatchPatchRequest:
        if not self.model_fields_set:
            raise ValueError("at least one match field must be provided")
        return self


class MatchResponse(ApiOutputModel):
    match_id: Identifier
    title: str | None = None
    youtube_video_id: str | None = None
    source_artifact_id: str | None = None
    pipeline_version: str | None = None
    model_versions: dict[str, str] = Field(default_factory=dict)
    summary: JsonObject = Field(default_factory=dict)
    artifact_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class MatchListResponse(ApiOutputModel):
    items: list[MatchResponse]
    total: int
    limit: int
    offset: int
