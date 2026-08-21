"""Versioned API schemas for auditable human semantic corrections."""

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
from pickleball_vision.corrections import CorrectionType


class CorrectionCreateRequest(ApiInputModel):
    correction_type: CorrectionType
    target_record_id: Identifier
    human_correction: JsonObject
    verified: bool = True
    reason: ShortText | None = None
    corrected_by: ShortText | None = None
    visual_evidence: JsonObject | None = None
    audio_evidence: JsonObject | None = None


class CorrectionPatchRequest(ApiInputModel):
    human_correction: JsonObject | None = None
    verified: bool | None = None
    reason: ShortText | None = None
    corrected_by: ShortText | None = None
    visual_evidence: JsonObject | None = None
    audio_evidence: JsonObject | None = None

    @model_validator(mode="after")
    def require_change(self) -> CorrectionPatchRequest:
        if not self.model_fields_set:
            raise ValueError("at least one correction field is required")
        return self


class CorrectionResponse(ApiOutputModel):
    correction_id: Identifier
    match_id: Identifier
    correction_type: CorrectionType
    target_collection: str
    target_record_id: Identifier
    prediction: JsonObject
    prediction_confidence: float | None = Field(default=None, ge=0, le=1)
    prediction_version: str | None = None
    human_correction: JsonObject
    verified: bool
    active: bool
    revision: int = Field(ge=1)
    history: list[JsonObject] = Field(default_factory=list)
    reason: str | None = None
    corrected_by: str | None = None
    visual_evidence: JsonObject | None = None
    audio_evidence: JsonObject | None = None
    created_at: datetime
    corrected_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class CorrectionListResponse(ApiOutputModel):
    items: list[CorrectionResponse]
    total: int
