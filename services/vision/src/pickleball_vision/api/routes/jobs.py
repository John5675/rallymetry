"""Durable asynchronous processing-job status routes."""

from __future__ import annotations

from fastapi import APIRouter

from pickleball_vision.api.dependencies import MatchServiceDependency
from pickleball_vision.api.routes.parameters import JobId
from pickleball_vision.api.schemas.jobs import JobResponse

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{jobId}", response_model=JobResponse)
async def get_job(job_id: JobId, service: MatchServiceDependency) -> JobResponse:
    return await service.get_job(job_id)
