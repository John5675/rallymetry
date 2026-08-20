"""Shared JSON-only API schemas and pagination fields."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class ApiInputModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class ApiOutputModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        extra="ignore",
        populate_by_name=True,
    )


Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    ),
]
ShortText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=512),
]
JsonObject = dict[str, JsonValue]


class PaginationParameters(ApiInputModel):
    limit: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class ErrorDetail(ApiOutputModel):
    code: str
    message: str
    details: JsonObject = Field(default_factory=dict)
    request_id: str


class ErrorResponse(ApiOutputModel):
    error: ErrorDetail


class HealthResponse(ApiOutputModel):
    status: str
    service: str
    version: str
    database_configured: bool
    database_ready: bool
    artifact_backend: str
