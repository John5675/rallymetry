"""Read-only reconstructed-shot routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from pickleball_vision.api.dependencies import MatchServiceDependency
from pickleball_vision.api.routes.parameters import MatchId
from pickleball_vision.api.schemas.records import DomainRecordListResponse

router = APIRouter(prefix="/api/matches", tags=["shots"])


@router.get("/{matchId}/shots", response_model=DomainRecordListResponse)
async def list_shots(
    match_id: MatchId,
    service: MatchServiceDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DomainRecordListResponse:
    return await service.list_shots(match_id, limit=limit, offset=offset)
