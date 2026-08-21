"""FastAPI application factory and ASGI entry point."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import cast

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import JsonValue
from starlette.exceptions import HTTPException as StarletteHttpException

from pickleball_vision.api.errors import ApiError
from pickleball_vision.api.routes import include_routes
from pickleball_vision.api.schemas.common import ErrorDetail, ErrorResponse, JsonObject
from pickleball_vision.api.services.persistence import ApplicationPersistence
from pickleball_vision.api.services.render_workflows import (
    AnalysisWorkflowClient,
    RenderWorkflowClient,
)
from pickleball_vision.api.settings import ApiSettings
from pickleball_vision.errors import PickleballVisionError
from pickleball_vision.persistence.mongodb import MongoPersistence

LOGGER = logging.getLogger("pickleball_vision.api")
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else "unknown"


def _json_details(details: Mapping[str, object]) -> JsonObject:
    encoded = jsonable_encoder(dict(details))
    if not isinstance(encoded, dict):
        return {}
    return cast(dict[str, JsonValue], encoded)


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: Mapping[str, object] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            details=_json_details(details or {}),
            request_id=_request_id(request),
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json", by_alias=True),
        headers=headers,
    )


async def _api_error_handler(request: Request, untyped_error: Exception) -> JSONResponse:
    error = cast(ApiError, untyped_error)
    return _error_response(
        request,
        status_code=error.status_code,
        code=error.code,
        message=error.message,
        details=error.details,
    )


async def _validation_error_handler(
    request: Request,
    untyped_error: Exception,
) -> JSONResponse:
    error = cast(RequestValidationError, untyped_error)
    issues = [
        {
            "location": [str(part) for part in issue.get("loc", ())],
            "message": str(issue.get("msg", "invalid value")),
            "type": str(issue.get("type", "validation_error")),
        }
        for issue in error.errors()
    ]
    return _error_response(
        request,
        status_code=422,
        code="validation_error",
        message="Request validation failed",
        details={"issues": issues},
    )


async def _domain_error_handler(
    request: Request,
    untyped_error: Exception,
) -> JSONResponse:
    error = cast(PickleballVisionError, untyped_error)
    return _error_response(
        request,
        status_code=503,
        code="persistence_error",
        message="A hosted persistence operation failed",
        details={"errorCode": error.code.value},
    )


async def _http_error_handler(
    request: Request,
    untyped_error: Exception,
) -> JSONResponse:
    error = cast(StarletteHttpException, untyped_error)
    return _error_response(
        request,
        status_code=error.status_code,
        code="http_error",
        message=str(error.detail),
        headers=error.headers,
    )


async def _unexpected_error_handler(request: Request, error: Exception) -> JSONResponse:
    LOGGER.error(
        "api_request_failed",
        extra={
            "context": {
                "method": request.method,
                "path": request.url.path,
                "requestId": _request_id(request),
                "exceptionType": type(error).__name__,
            }
        },
    )
    return _error_response(
        request,
        status_code=500,
        code="internal_error",
        message="An unexpected server error occurred",
    )


def _lifespan(
    settings: ApiSettings,
    injected_persistence: ApplicationPersistence | None,
    injected_workflow_client: AnalysisWorkflowClient | None,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned_persistence: MongoPersistence | None = None
        if injected_workflow_client is not None:
            app.state.workflow_client = injected_workflow_client
        elif settings.render_api_key is not None and settings.render_workflow_task is not None:
            app.state.workflow_client = RenderWorkflowClient(
                api_key=settings.render_api_key,
                task_identifier=settings.render_workflow_task,
            )
        if injected_persistence is not None:
            app.state.persistence = injected_persistence
            app.state.database_ready = True
        elif settings.persistence.mongodb_url is not None:
            try:
                owned_persistence = await MongoPersistence.connect_from_settings(
                    settings.persistence
                )
                await owned_persistence.initialize_indexes()
                app.state.persistence = owned_persistence
                app.state.database_ready = True
            except PickleballVisionError as error:
                app.state.persistence = None
                app.state.database_ready = False
                LOGGER.error(
                    "api_persistence_startup_failed",
                    extra={"context": {"errorCode": error.code.value}},
                )
        try:
            yield
        finally:
            app.state.persistence = None
            app.state.database_ready = False
            app.state.workflow_client = None
            if owned_persistence is not None:
                try:
                    await owned_persistence.close()
                except PickleballVisionError as error:
                    LOGGER.error(
                        "api_persistence_shutdown_failed",
                        extra={"context": {"errorCode": error.code.value}},
                    )

    return lifespan


def create_app(
    *,
    settings: ApiSettings | None = None,
    persistence: ApplicationPersistence | None = None,
    workflow_client: AnalysisWorkflowClient | None = None,
) -> FastAPI:
    """Build the control-plane app without opening provider connections."""

    effective_settings = settings or ApiSettings.from_env()
    app = FastAPI(
        title="Pickleball Vision API",
        version="0.1.0",
        lifespan=_lifespan(effective_settings, persistence, workflow_client),
        responses={
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    app.state.api_settings = effective_settings
    app.state.persistence = persistence
    app.state.database_ready = persistence is not None
    app.state.workflow_client = workflow_client

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(effective_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["Location", "X-Request-ID"],
    )

    @app.middleware("http")
    async def log_request(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        supplied_request_id = request.headers.get("X-Request-ID", "")
        request.state.request_id = (
            supplied_request_id
            if _REQUEST_ID_PATTERN.fullmatch(supplied_request_id)
            else uuid.uuid4().hex
        )
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request.state.request_id
            return response
        finally:
            LOGGER.info(
                "api_request_completed",
                extra={
                    "context": {
                        "durationMs": round((time.perf_counter() - started) * 1000, 3),
                        "method": request.method,
                        "path": request.url.path,
                        "requestId": request.state.request_id,
                        "statusCode": status_code,
                    }
                },
            )

    app.add_exception_handler(ApiError, _api_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(PickleballVisionError, _domain_error_handler)
    app.add_exception_handler(StarletteHttpException, _http_error_handler)
    app.add_exception_handler(Exception, _unexpected_error_handler)
    include_routes(app)
    return app


app = create_app()
