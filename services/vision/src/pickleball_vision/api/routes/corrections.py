"""Human correction CRUD routes with immutable prediction snapshots."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from pickleball_vision.api.dependencies import MatchServiceDependency
from pickleball_vision.api.routes.parameters import CorrectionId, MatchId
from pickleball_vision.api.schemas.corrections import (
    CorrectionCreateRequest,
    CorrectionListResponse,
    CorrectionPatchRequest,
    CorrectionResponse,
)

router = APIRouter(prefix="/api/matches", tags=["corrections"])


@router.get("/{matchId}/corrections", response_model=CorrectionListResponse)
async def list_corrections(
    match_id: MatchId,
    service: MatchServiceDependency,
) -> CorrectionListResponse:
    return await service.list_corrections(match_id)


@router.post(
    "/{matchId}/corrections",
    response_model=CorrectionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_correction(
    match_id: MatchId,
    request: CorrectionCreateRequest,
    service: MatchServiceDependency,
) -> CorrectionResponse:
    return await service.create_correction(match_id, request)


@router.patch("/{matchId}/corrections/{correctionId}", response_model=CorrectionResponse)
async def update_correction(
    match_id: MatchId,
    correction_id: CorrectionId,
    request: CorrectionPatchRequest,
    service: MatchServiceDependency,
) -> CorrectionResponse:
    return await service.update_correction(match_id, correction_id, request)


@router.delete(
    "/{matchId}/corrections/{correctionId}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_correction(
    match_id: MatchId,
    correction_id: CorrectionId,
    service: MatchServiceDependency,
) -> Response:
    await service.remove_correction(match_id, correction_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
