"""Read-only contact and bounce event routes for application timelines."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from pickleball_vision.api.dependencies import MatchServiceDependency
from pickleball_vision.api.routes.parameters import MatchId
from pickleball_vision.api.schemas.records import DomainRecordListResponse

router = APIRouter(prefix="/api/matches", tags=["events"])


@router.get("/{matchId}/contacts", response_model=DomainRecordListResponse)
async def list_contacts(
    match_id: MatchId,
    service: MatchServiceDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DomainRecordListResponse:
    return await service.list_events(
        match_id,
        event_type="contacts",
        limit=limit,
        offset=offset,
    )


@router.get("/{matchId}/bounces", response_model=DomainRecordListResponse)
async def list_bounces(
    match_id: MatchId,
    service: MatchServiceDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DomainRecordListResponse:
    return await service.list_events(
        match_id,
        event_type="bounces",
        limit=limit,
        offset=offset,
    )
