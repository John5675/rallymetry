"""Public JSON schemas for the FastAPI control plane."""

from pickleball_vision.api.schemas.common import ErrorResponse, HealthResponse
from pickleball_vision.api.schemas.jobs import JobResponse
from pickleball_vision.api.schemas.matches import (
    MatchCreateRequest,
    MatchListResponse,
    MatchPatchRequest,
    MatchResponse,
)
from pickleball_vision.api.schemas.records import (
    AnalyticsResponse,
    ArtifactListResponse,
    ArtifactResponse,
    DomainRecordListResponse,
    DomainRecordResponse,
    PlayerListResponse,
    PlayerResponse,
)

__all__ = [
    "AnalyticsResponse",
    "ArtifactListResponse",
    "ArtifactResponse",
    "DomainRecordListResponse",
    "DomainRecordResponse",
    "ErrorResponse",
    "HealthResponse",
    "JobResponse",
    "MatchCreateRequest",
    "MatchListResponse",
    "MatchPatchRequest",
    "MatchResponse",
    "PlayerListResponse",
    "PlayerResponse",
]
