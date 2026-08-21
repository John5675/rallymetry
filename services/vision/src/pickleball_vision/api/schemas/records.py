"""JSON representations of persisted players, events, analytics, and artifacts."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from pickleball_vision.api.schemas.common import ApiOutputModel, Identifier, JsonObject
from pickleball_vision.api.schemas.corrections import CorrectionResponse


class PlayerResponse(ApiOutputModel):
    match_id: Identifier
    player_id: Identifier
    display_name: str | None = None
    logical_identity: str | None = None
    team: str | None = None
    metadata: JsonObject = Field(default_factory=dict)
    effective_player: JsonObject | None = None
    verified_corrections: list[CorrectionResponse] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class PlayerListResponse(ApiOutputModel):
    items: list[PlayerResponse]
    total: int


class DomainRecordResponse(ApiOutputModel):
    match_id: Identifier
    record_id: Identifier
    payload: JsonObject
    effective_payload: JsonObject = Field(default_factory=dict)
    verified_corrections: list[CorrectionResponse] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    timestamp_seconds: float | None = Field(default=None, ge=0)
    pipeline_version: str | None = None
    model_version: str | None = None
    created_at: datetime


class DomainRecordListResponse(ApiOutputModel):
    items: list[DomainRecordResponse]
    total: int
    limit: int
    offset: int


class AnalyticsResponse(ApiOutputModel):
    match_id: Identifier
    analytics_id: Identifier
    calculation_version: str
    metrics: JsonObject
    prediction_metrics: JsonObject | None = None
    applied_correction_ids: list[str] = Field(default_factory=list)
    input_artifact_ids: list[str] = Field(default_factory=list)
    pipeline_version: str | None = None
    created_at: datetime


class ArtifactResponse(ApiOutputModel):
    artifact_id: Identifier
    match_id: Identifier | None = None
    artifact_type: str
    category: str
    pathname: str
    provider: str
    access: str
    content_type: str
    size: int = Field(ge=0)
    created_at: datetime
    pipeline_version: str | None = None
    url: str | None = None
    checksum_sha256: str | None = None


class ArtifactListResponse(ApiOutputModel):
    items: list[ArtifactResponse]
    total: int
