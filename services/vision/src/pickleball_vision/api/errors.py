"""Stable application-layer errors translated into JSON by FastAPI."""

from __future__ import annotations

from collections.abc import Mapping


class ApiError(Exception):
    """Expected API failure with a stable HTTP and machine-readable contract."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = dict(details or {})


class ResourceNotFoundError(ApiError):
    def __init__(self, resource: str, identifier: str) -> None:
        super().__init__(
            status_code=404,
            code="resource_not_found",
            message=f"{resource} was not found",
            details={"resource": resource, "id": identifier},
        )


class PersistenceUnavailableError(ApiError):
    def __init__(self) -> None:
        super().__init__(
            status_code=503,
            code="persistence_unavailable",
            message="Hosted persistence is not available",
        )
