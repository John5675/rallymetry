"""Match metadata, artifact listing, and asynchronous process submission routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response, status

from pickleball_vision.api.dependencies import MatchServiceDependency
from pickleball_vision.api.routes.parameters import MatchId
from pickleball_vision.api.schemas.jobs import JobResponse
from pickleball_vision.api.schemas.matches import (
    MatchCreateRequest,
    MatchListResponse,
    MatchPatchRequest,
    MatchResponse,
)
from pickleball_vision.api.schemas.records import ArtifactListResponse

router = APIRouter(prefix="/api/matches", tags=["matches"])


@router.post("", response_model=MatchResponse, status_code=status.HTTP_201_CREATED)
async def create_match(
    request: MatchCreateRequest,
    service: MatchServiceDependency,
) -> MatchResponse:
    return await service.create_match(request)


@router.get("", response_model=MatchListResponse)
async def list_matches(
    service: MatchServiceDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> MatchListResponse:
    return await service.list_matches(limit=limit, offset=offset)


@router.get("/{matchId}", response_model=MatchResponse)
async def get_match(match_id: MatchId, service: MatchServiceDependency) -> MatchResponse:
    return await service.get_match(match_id)


@router.patch("/{matchId}", response_model=MatchResponse)
async def patch_match(
    match_id: MatchId,
    request: MatchPatchRequest,
    service: MatchServiceDependency,
) -> MatchResponse:
    return await service.patch_match(match_id, request)


@router.get("/{matchId}/artifacts", response_model=ArtifactListResponse)
async def list_artifacts(
    match_id: MatchId,
    service: MatchServiceDependency,
) -> ArtifactListResponse:
    return await service.list_artifacts(match_id)


@router.post(
    "/{matchId}/process",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def process_match(
    match_id: MatchId,
    service: MatchServiceDependency,
    response: Response,
) -> JobResponse:
    job = await service.queue_processing(match_id)
    response.headers["Location"] = f"/api/jobs/{job.job_id}"
    return job
