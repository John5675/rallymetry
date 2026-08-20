"""FastAPI dependencies for application persistence and services."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from pickleball_vision.api.errors import PersistenceUnavailableError
from pickleball_vision.api.services.matches import MatchApplicationService
from pickleball_vision.api.services.persistence import ApplicationPersistence


def get_persistence(request: Request) -> ApplicationPersistence:
    persistence = getattr(request.app.state, "persistence", None)
    if persistence is None:
        raise PersistenceUnavailableError
    return cast(ApplicationPersistence, persistence)


PersistenceDependency = Annotated[ApplicationPersistence, Depends(get_persistence)]


def get_match_service(persistence: PersistenceDependency) -> MatchApplicationService:
    return MatchApplicationService(persistence)


MatchServiceDependency = Annotated[MatchApplicationService, Depends(get_match_service)]
