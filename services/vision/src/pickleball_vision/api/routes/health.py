"""Process liveness and hosted-persistence readiness route."""

from __future__ import annotations

from fastapi import APIRouter, Request

from pickleball_vision import __version__
from pickleball_vision.api.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.api_settings
    return HealthResponse(
        status="ok" if request.app.state.database_ready else "degraded",
        service="pickleball-vision-api",
        version=__version__,
        database_configured=settings.persistence.mongodb_url is not None,
        database_ready=bool(request.app.state.database_ready),
        artifact_backend=settings.persistence.artifact_backend.value,
    )
