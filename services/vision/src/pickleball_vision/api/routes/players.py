"""Read-only logical-player routes."""

from __future__ import annotations

from fastapi import APIRouter

from pickleball_vision.api.dependencies import MatchServiceDependency
from pickleball_vision.api.routes.parameters import MatchId
from pickleball_vision.api.schemas.records import PlayerListResponse

router = APIRouter(prefix="/api/matches", tags=["players"])


@router.get("/{matchId}/players", response_model=PlayerListResponse)
async def list_players(
    match_id: MatchId,
    service: MatchServiceDependency,
) -> PlayerListResponse:
    return await service.list_players(match_id)
