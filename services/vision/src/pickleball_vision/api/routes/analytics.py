"""Read-only deterministic match-analytics route."""

from __future__ import annotations

from fastapi import APIRouter

from pickleball_vision.api.dependencies import MatchServiceDependency
from pickleball_vision.api.routes.parameters import MatchId
from pickleball_vision.api.schemas.records import AnalyticsResponse

router = APIRouter(prefix="/api/matches", tags=["analytics"])


@router.get("/{matchId}/analytics", response_model=AnalyticsResponse)
async def get_analytics(
    match_id: MatchId,
    service: MatchServiceDependency,
) -> AnalyticsResponse:
    return await service.get_analytics(match_id)
