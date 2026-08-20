"""Typed hosted-persistence records independent of MongoDB and Vercel Blob."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath

from pickleball_vision.errors import PersistenceValidationError

Document = dict[str, object]
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


def _required_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise PersistenceValidationError(f"{field_name} must not be empty")
    if len(normalized) > 512:
        raise PersistenceValidationError(f"{field_name} must not exceed 512 characters")
    return normalized


def _optional_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _utc_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise PersistenceValidationError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _copy_mapping(value: Mapping[str, object], field_name: str) -> dict[str, object]:
    copied = dict(value)
    if any(not isinstance(key, str) or not key for key in copied):
        raise PersistenceValidationError(f"{field_name} keys must be non-empty strings")
    return copied


class ArtifactCategory(StrEnum):
    """Storage and access policy category for one artifact."""

    SOURCE_MEDIA = "SOURCE_MEDIA"
    VIEWABLE_MEDIA = "VIEWABLE_MEDIA"
    INTERNAL_ARTIFACT = "INTERNAL_ARTIFACT"


class ArtifactAccess(StrEnum):
    """How an artifact may be retrieved."""

    LOCAL = "LOCAL"
    PRIVATE = "PRIVATE"
    PUBLIC = "PUBLIC"


class ArtifactProvider(StrEnum):
    """Physical storage provider retained in the manifest."""

    LOCAL = "LOCAL"
    VERCEL_BLOB = "VERCEL_BLOB"


class ProcessingJobStatus(StrEnum):
    """Persisted status only; queue claiming is deferred to Milestone 21."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class StructuredCollection(StrEnum):
    """Separate collections for repeating structured domain records."""

    RALLIES = "rallies"
    CONTACTS = "contacts"
    BOUNCES = "bounces"
    SHOTS = "shots"


@dataclass(frozen=True, slots=True)
class MatchRecord:
    """Compact match metadata; repeating events live in separate collections."""

    match_id: str
    title: str | None = None
    youtube_video_id: str | None = None
    source_artifact_id: str | None = None
    pipeline_version: str | None = None
    model_versions: Mapping[str, str] = field(default_factory=dict)
    summary: Mapping[str, object] = field(default_factory=dict)
    artifact_ids: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "match_id", _required_text(self.match_id, "match_id"))
        object.__setattr__(self, "title", _optional_text(self.title, "title"))
        object.__setattr__(
            self,
            "youtube_video_id",
            _optional_text(self.youtube_video_id, "youtube_video_id"),
        )
        object.__setattr__(
            self,
            "source_artifact_id",
            _optional_text(self.source_artifact_id, "source_artifact_id"),
        )
        object.__setattr__(
            self,
            "pipeline_version",
            _optional_text(self.pipeline_version, "pipeline_version"),
        )
        versions = dict(self.model_versions)
        for name, version in versions.items():
            _required_text(name, "model_versions key")
            _required_text(version, f"model_versions[{name!r}]")
        object.__setattr__(self, "model_versions", versions)
        object.__setattr__(self, "summary", _copy_mapping(self.summary, "summary"))
        object.__setattr__(
            self,
            "artifact_ids",
            tuple(_required_text(item, "artifact_ids item") for item in self.artifact_ids),
        )
        object.__setattr__(self, "created_at", _utc_datetime(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _utc_datetime(self.updated_at, "updated_at"))
        if self.updated_at < self.created_at:
            raise PersistenceValidationError("updated_at must not precede created_at")

    def to_document(self) -> Document:
        document: Document = {
            "_id": self.match_id,
            "matchId": self.match_id,
            "modelVersions": dict(self.model_versions),
            "summary": dict(self.summary),
            "artifactIds": list(self.artifact_ids),
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }
        optional = {
            "title": self.title,
            "youtubeVideoId": self.youtube_video_id,
            "sourceArtifactId": self.source_artifact_id,
            "pipelineVersion": self.pipeline_version,
        }
        document.update({key: value for key, value in optional.items() if value is not None})
        return document


@dataclass(frozen=True, slots=True)
class PlayerRecord:
    """One match-scoped logical player record."""

    match_id: str
    player_id: str
    display_name: str | None = None
    logical_identity: str | None = None
    team: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "match_id", _required_text(self.match_id, "match_id"))
        object.__setattr__(self, "player_id", _required_text(self.player_id, "player_id"))
        object.__setattr__(
            self,
            "display_name",
            _optional_text(self.display_name, "display_name"),
        )
        object.__setattr__(
            self,
            "logical_identity",
            _optional_text(self.logical_identity, "logical_identity"),
        )
        object.__setattr__(self, "team", _optional_text(self.team, "team"))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata, "metadata"))
        object.__setattr__(self, "created_at", _utc_datetime(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _utc_datetime(self.updated_at, "updated_at"))

    def to_document(self) -> Document:
        document: Document = {
            "_id": f"{self.match_id}:{self.player_id}",
            "matchId": self.match_id,
            "playerId": self.player_id,
            "metadata": dict(self.metadata),
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }
        optional = {
            "displayName": self.display_name,
            "logicalIdentity": self.logical_identity,
            "team": self.team,
        }
        document.update({key: value for key, value in optional.items() if value is not None})
        return document


@dataclass(frozen=True, slots=True)
class StructuredDomainRecord:
    """One rally/contact/bounce/shot stored independently from its match."""

    match_id: str
    record_id: str
    payload: Mapping[str, object]
    confidence: float | None = None
    timestamp_seconds: float | None = None
    pipeline_version: str | None = None
    model_version: str | None = None
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "match_id", _required_text(self.match_id, "match_id"))
        object.__setattr__(self, "record_id", _required_text(self.record_id, "record_id"))
        object.__setattr__(self, "payload", _copy_mapping(self.payload, "payload"))
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise PersistenceValidationError("confidence must be between 0 and 1")
        if self.timestamp_seconds is not None and self.timestamp_seconds < 0:
            raise PersistenceValidationError("timestamp_seconds must be nonnegative")
        object.__setattr__(
            self,
            "pipeline_version",
            _optional_text(self.pipeline_version, "pipeline_version"),
        )
        object.__setattr__(
            self,
            "model_version",
            _optional_text(self.model_version, "model_version"),
        )
        object.__setattr__(self, "created_at", _utc_datetime(self.created_at, "created_at"))

    def to_document(self) -> Document:
        document: Document = {
            "_id": f"{self.match_id}:{self.record_id}",
            "matchId": self.match_id,
            "recordId": self.record_id,
            "payload": dict(self.payload),
            "createdAt": self.created_at,
        }
        optional = {
            "confidence": self.confidence,
            "timestampSeconds": self.timestamp_seconds,
            "pipelineVersion": self.pipeline_version,
            "modelVersion": self.model_version,
        }
        document.update({key: value for key, value in optional.items() if value is not None})
        return document


@dataclass(frozen=True, slots=True)
class AnalyticsRecord:
    """Compact deterministic analytics record referencing its structured inputs."""

    match_id: str
    analytics_id: str
    calculation_version: str
    metrics: Mapping[str, object]
    input_artifact_ids: tuple[str, ...] = ()
    pipeline_version: str | None = None
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "match_id", _required_text(self.match_id, "match_id"))
        object.__setattr__(
            self,
            "analytics_id",
            _required_text(self.analytics_id, "analytics_id"),
        )
        object.__setattr__(
            self,
            "calculation_version",
            _required_text(self.calculation_version, "calculation_version"),
        )
        object.__setattr__(self, "metrics", _copy_mapping(self.metrics, "metrics"))
        object.__setattr__(
            self,
            "input_artifact_ids",
            tuple(
                _required_text(item, "input_artifact_ids item") for item in self.input_artifact_ids
            ),
        )
        object.__setattr__(
            self,
            "pipeline_version",
            _optional_text(self.pipeline_version, "pipeline_version"),
        )
        object.__setattr__(self, "created_at", _utc_datetime(self.created_at, "created_at"))

    def to_document(self) -> Document:
        document: Document = {
            "_id": f"{self.match_id}:{self.analytics_id}",
            "matchId": self.match_id,
            "analyticsId": self.analytics_id,
            "calculationVersion": self.calculation_version,
            "metrics": dict(self.metrics),
            "inputArtifactIds": list(self.input_artifact_ids),
            "createdAt": self.created_at,
        }
        if self.pipeline_version is not None:
            document["pipelineVersion"] = self.pipeline_version
        return document


@dataclass(frozen=True, slots=True)
class ProcessingJobRecord:
    """Small processing-status record; queue leases/claims are not implemented here."""

    job_id: str
    match_id: str
    job_type: str
    status: ProcessingJobStatus = ProcessingJobStatus.QUEUED
    progress: float = 0.0
    stage: str | None = None
    error: Mapping[str, object] | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _required_text(self.job_id, "job_id"))
        object.__setattr__(self, "match_id", _required_text(self.match_id, "match_id"))
        object.__setattr__(self, "job_type", _required_text(self.job_type, "job_type"))
        if not 0.0 <= self.progress <= 1.0:
            raise PersistenceValidationError("progress must be between 0 and 1")
        object.__setattr__(self, "stage", _optional_text(self.stage, "stage"))
        if self.error is not None:
            object.__setattr__(self, "error", _copy_mapping(self.error, "error"))
        object.__setattr__(self, "created_at", _utc_datetime(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _utc_datetime(self.updated_at, "updated_at"))

    def to_document(self) -> Document:
        document: Document = {
            "_id": self.job_id,
            "jobId": self.job_id,
            "matchId": self.match_id,
            "jobType": self.job_type,
            "status": self.status.value,
            "progress": self.progress,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }
        if self.stage is not None:
            document["stage"] = self.stage
        if self.error is not None:
            document["error"] = dict(self.error)
        return document


@dataclass(frozen=True, slots=True)
class CorrectionRecord:
    """Persistable correction proposal; the editing workflow is Milestone 24."""

    correction_id: str
    match_id: str
    target_collection: str
    target_record_id: str
    changes: Mapping[str, object]
    reason: str | None = None
    created_by: str | None = None
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "correction_id",
            _required_text(self.correction_id, "correction_id"),
        )
        object.__setattr__(self, "match_id", _required_text(self.match_id, "match_id"))
        object.__setattr__(
            self,
            "target_collection",
            _required_text(self.target_collection, "target_collection"),
        )
        object.__setattr__(
            self,
            "target_record_id",
            _required_text(self.target_record_id, "target_record_id"),
        )
        object.__setattr__(self, "changes", _copy_mapping(self.changes, "changes"))
        object.__setattr__(self, "reason", _optional_text(self.reason, "reason"))
        object.__setattr__(
            self,
            "created_by",
            _optional_text(self.created_by, "created_by"),
        )
        object.__setattr__(self, "created_at", _utc_datetime(self.created_at, "created_at"))

    def to_document(self) -> Document:
        document: Document = {
            "_id": self.correction_id,
            "correctionId": self.correction_id,
            "matchId": self.match_id,
            "targetCollection": self.target_collection,
            "targetRecordId": self.target_record_id,
            "changes": dict(self.changes),
            "createdAt": self.created_at,
        }
        if self.reason is not None:
            document["reason"] = self.reason
        if self.created_by is not None:
            document["createdBy"] = self.created_by
        return document


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """MongoDB-safe metadata for an artifact stored elsewhere."""

    artifact_id: str
    match_id: str | None
    artifact_type: str
    category: ArtifactCategory
    pathname: str
    provider: ArtifactProvider
    access: ArtifactAccess
    content_type: str
    size_bytes: int
    created_at: datetime
    pipeline_version: str | None = None
    url: str | None = None
    checksum_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_id",
            _required_text(self.artifact_id, "artifact_id"),
        )
        object.__setattr__(self, "match_id", _optional_text(self.match_id, "match_id"))
        object.__setattr__(
            self,
            "artifact_type",
            _required_text(self.artifact_type, "artifact_type"),
        )
        pathname = _required_text(self.pathname, "pathname")
        parsed = PurePosixPath(pathname)
        if parsed.is_absolute() or ".." in parsed.parts:
            raise PersistenceValidationError("pathname must be a safe relative POSIX path")
        object.__setattr__(self, "pathname", parsed.as_posix())
        object.__setattr__(
            self,
            "content_type",
            _required_text(self.content_type, "content_type"),
        )
        if self.size_bytes < 0:
            raise PersistenceValidationError("size_bytes must be nonnegative")
        object.__setattr__(self, "created_at", _utc_datetime(self.created_at, "created_at"))
        object.__setattr__(
            self,
            "pipeline_version",
            _optional_text(self.pipeline_version, "pipeline_version"),
        )
        object.__setattr__(self, "url", _optional_text(self.url, "url"))
        if self.checksum_sha256 is not None and not _SHA256_PATTERN.fullmatch(self.checksum_sha256):
            raise PersistenceValidationError("checksum_sha256 must be lowercase SHA-256 hex")
        if self.provider is ArtifactProvider.LOCAL and self.url is not None:
            raise PersistenceValidationError("local artifacts must not expose a hosted URL")
        if self.provider is ArtifactProvider.LOCAL and self.access is not ArtifactAccess.LOCAL:
            raise PersistenceValidationError("local artifacts must use LOCAL access")
        if self.provider is ArtifactProvider.VERCEL_BLOB and self.access is ArtifactAccess.LOCAL:
            raise PersistenceValidationError("Vercel Blob artifacts cannot use LOCAL access")
        if (
            self.category is not ArtifactCategory.VIEWABLE_MEDIA
            and self.access is ArtifactAccess.PUBLIC
        ):
            raise PersistenceValidationError("only VIEWABLE_MEDIA may be intentionally public")

    def to_document(self) -> Document:
        document: Document = {
            "_id": self.artifact_id,
            "artifactId": self.artifact_id,
            "artifactType": self.artifact_type,
            "category": self.category.value,
            "pathname": self.pathname,
            "provider": self.provider.value,
            "access": self.access.value,
            "contentType": self.content_type,
            "size": self.size_bytes,
            "createdAt": self.created_at,
        }
        optional = {
            "matchId": self.match_id,
            "url": self.url,
            "pipelineVersion": self.pipeline_version,
            "checksumSha256": self.checksum_sha256,
        }
        document.update({key: value for key, value in optional.items() if value is not None})
        return document


@dataclass(frozen=True, slots=True)
class ArtifactPutRequest:
    """Provider-neutral request to persist one existing local file."""

    source_path: Path
    artifact_type: str
    category: ArtifactCategory
    match_id: str | None = None
    access: ArtifactAccess | None = None
    content_type: str | None = None
    pipeline_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_type",
            _required_text(self.artifact_type, "artifact_type"),
        )
        object.__setattr__(self, "match_id", _optional_text(self.match_id, "match_id"))
        object.__setattr__(
            self,
            "content_type",
            _optional_text(self.content_type, "content_type"),
        )
        object.__setattr__(
            self,
            "pipeline_version",
            _optional_text(self.pipeline_version, "pipeline_version"),
        )
        if (
            self.category is not ArtifactCategory.VIEWABLE_MEDIA
            and self.access is ArtifactAccess.PUBLIC
        ):
            raise PersistenceValidationError(
                "SOURCE_MEDIA and INTERNAL_ARTIFACT cannot request public access"
            )
