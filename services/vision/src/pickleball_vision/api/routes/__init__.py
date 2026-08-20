"""Route registration for the FastAPI application."""

from fastapi import FastAPI

from pickleball_vision.api.routes import (
    analytics,
    health,
    jobs,
    matches,
    players,
    rallies,
    shots,
)


def include_routes(app: FastAPI) -> None:
    app.include_router(health.router)
    app.include_router(matches.router)
    app.include_router(players.router)
    app.include_router(rallies.router)
    app.include_router(shots.router)
    app.include_router(analytics.router)
    app.include_router(jobs.router)


__all__ = ["include_routes"]
