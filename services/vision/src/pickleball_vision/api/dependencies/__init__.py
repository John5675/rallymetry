"""Dependency injection boundaries for the FastAPI application."""

from pickleball_vision.api.dependencies.persistence import (
    MatchServiceDependency,
    PersistenceDependency,
    get_match_service,
    get_persistence,
)

__all__ = [
    "MatchServiceDependency",
    "PersistenceDependency",
    "get_match_service",
    "get_persistence",
]
