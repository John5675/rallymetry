"""FastAPI dependencies for application persistence and services."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from pickleball_vision.api.errors import PersistenceUnavailableError
from pickleball_vision.api.services.matches import MatchApplicationService
from pickleball_vision.api.services.persistence import ApplicationPersistence
from pickleball_vision.api.services.render_workflows import AnalysisWorkflowClient
from pickleball_vision.api.settings import ApiSettings


def get_persistence(request: Request) -> ApplicationPersistence:
    persistence = getattr(request.app.state, "persistence", None)
    if persistence is None:
        raise PersistenceUnavailableError
    return cast(ApplicationPersistence, persistence)


PersistenceDependency = Annotated[ApplicationPersistence, Depends(get_persistence)]


def get_match_service(
    request: Request,
    persistence: PersistenceDependency,
) -> MatchApplicationService:
    workflow_client = cast(
        AnalysisWorkflowClient | None,
        getattr(request.app.state, "workflow_client", None),
    )
    settings = cast(ApiSettings, request.app.state.api_settings)
    return MatchApplicationService(
        persistence,
        workflow_client=workflow_client,
        default_analysis_profile_match_id=settings.default_analysis_profile_match_id,
    )


MatchServiceDependency = Annotated[MatchApplicationService, Depends(get_match_service)]
