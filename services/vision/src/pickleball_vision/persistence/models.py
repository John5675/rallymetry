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
    """Durable application stages for one on-demand workflow run."""

    CREATED = "CREATED"
    QUEUED = "QUEUED"
    STARTING = "STARTING"
    DOWNLOADING_MEDIA = "DOWNLOADING_MEDIA"
    PREPARING_MEDIA = "PREPARING_MEDIA"
    PLAYER_PROCESSING = "PLAYER_PROCESSING"
    BALL_PROCESSING = "BALL_PROCESSING"
    AUDIO_PROCESSING = "AUDIO_PROCESSING"
    RALLY_PROCESSING = "RALLY_PROCESSING"
    BOUNCE_PROCESSING = "BOUNCE_PROCESSING"
    CONTACT_PROCESSING = "CONTACT_PROCESSING"
    HITTER_PROCESSING = "HITTER_PROCESSING"
    SHOT_PROCESSING = "SHOT_PROCESSING"
    ANALYTICS = "ANALYTICS"
    RENDERING_ARTIFACTS = "RENDERING_ARTIFACTS"
    UPLOADING_RESULTS = "UPLOADING_RESULTS"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELED = "CANCELED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


ACTIVE_PROCESSING_JOB_STATUSES = frozenset(
    status
    for status in ProcessingJobStatus
    if status
    not in {
        ProcessingJobStatus.COMPLETE,
        ProcessingJobStatus.FAILED,
        ProcessingJobStatus.CANCELED,
    }
)


class SourceMediaType(StrEnum):
    """Supported analysis source locations; YouTube is intentionally absent."""

    LOCAL_PATH = "LOCAL_PATH"
    BLOB = "BLOB"


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
    analysis_setup: Mapping[str, str] = field(default_factory=dict)
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
        setup = dict(self.analysis_setup)
        for name, artifact_id in setup.items():
            _required_text(name, "analysis_setup key")
            _required_text(artifact_id, f"analysis_setup[{name!r}]")
        object.__setattr__(self, "analysis_setup", setup)
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
            "analysisSetup": dict(self.analysis_setup),
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
    processing_run_id: str | None = None
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
        object.__setattr__(
            self,
            "processing_run_id",
            _optional_text(self.processing_run_id, "processing_run_id"),
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
            "processingRunId": self.processing_run_id,
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
    processing_run_id: str | None = None
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
        object.__setattr__(
            self,
            "processing_run_id",
            _optional_text(self.processing_run_id, "processing_run_id"),
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
        if self.processing_run_id is not None:
            document["processingRunId"] = self.processing_run_id
        return document


@dataclass(frozen=True, slots=True)
class ProcessingJobRecord:
    """Durable status record for one on-demand Render Workflow run."""

    job_id: str
    match_id: str
    job_type: str
    status: ProcessingJobStatus = ProcessingJobStatus.CREATED
    progress: float = 0.0
    stage: str | None = None
    render_triggered_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    failed_stage: str | None = None
    render_task_run_id: str | None = None
    processing_run_id: str | None = None
    attempt_count: int = 0
    error_code: str | None = None
    error_message: str | None = None
    pipeline_version: str | None = None
    source_type: SourceMediaType | None = None
    source_path: str | None = None
    source_artifact_id: str | None = None
    result_artifact_ids: tuple[str, ...] = ()
    result_summary: Mapping[str, object] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _required_text(self.job_id, "job_id"))
        object.__setattr__(self, "match_id", _required_text(self.match_id, "match_id"))
        object.__setattr__(self, "job_type", _required_text(self.job_type, "job_type"))
        if not 0.0 <= self.progress <= 1.0:
            raise PersistenceValidationError("progress must be between 0 and 1")
        object.__setattr__(self, "stage", _optional_text(self.stage, "stage"))
        for field_name in (
            "render_triggered_at",
            "started_at",
            "completed_at",
            "failed_at",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _utc_datetime(value, field_name))
        object.__setattr__(
            self,
            "failed_stage",
            _optional_text(self.failed_stage, "failed_stage"),
        )
        object.__setattr__(
            self,
            "render_task_run_id",
            _optional_text(self.render_task_run_id, "render_task_run_id"),
        )
        object.__setattr__(
            self,
            "processing_run_id",
            _optional_text(self.processing_run_id, "processing_run_id"),
        )
        if self.attempt_count < 0:
            raise PersistenceValidationError("attempt_count must be nonnegative")
        object.__setattr__(self, "error_code", _optional_text(self.error_code, "error_code"))
        object.__setattr__(
            self,
            "error_message",
            _optional_text(self.error_message, "error_message"),
        )
        object.__setattr__(
            self,
            "pipeline_version",
            _optional_text(self.pipeline_version, "pipeline_version"),
        )
        object.__setattr__(self, "source_path", _optional_text(self.source_path, "source_path"))
        object.__setattr__(
            self,
            "source_artifact_id",
            _optional_text(self.source_artifact_id, "source_artifact_id"),
        )
        object.__setattr__(
            self,
            "result_artifact_ids",
            tuple(
                _required_text(value, "result_artifact_ids item")
                for value in self.result_artifact_ids
            ),
        )
        object.__setattr__(
            self,
            "result_summary",
            _copy_mapping(self.result_summary, "result_summary"),
        )
        object.__setattr__(self, "created_at", _utc_datetime(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _utc_datetime(self.updated_at, "updated_at"))
        if self.updated_at < self.created_at:
            raise PersistenceValidationError("updated_at must not precede created_at")
        if self.source_type is SourceMediaType.LOCAL_PATH:
            if self.source_path is None and self.source_artifact_id is None:
                raise PersistenceValidationError(
                    "LOCAL_PATH jobs require source_path or source_artifact_id"
                )
        elif self.source_type is SourceMediaType.BLOB and self.source_artifact_id is None:
            raise PersistenceValidationError("BLOB jobs require source_artifact_id")
        if self.source_path is not None and self.source_type is not SourceMediaType.LOCAL_PATH:
            raise PersistenceValidationError("source_path is only valid for LOCAL_PATH jobs")
        if self.status is ProcessingJobStatus.COMPLETE and self.progress != 1.0:
            raise PersistenceValidationError("COMPLETE jobs must have progress 1.0")
        if self.status is ProcessingJobStatus.COMPLETE and self.completed_at is None:
            raise PersistenceValidationError("COMPLETE jobs require completed_at")
        if self.status is ProcessingJobStatus.FAILED and self.failed_at is None:
            raise PersistenceValidationError("FAILED jobs require failed_at")

    def to_document(self) -> Document:
        document: Document = {
            "_id": self.job_id,
            "jobId": self.job_id,
            "matchId": self.match_id,
            "jobType": self.job_type,
            "status": self.status.value,
            "progress": self.progress,
            "attemptCount": self.attempt_count,
            "resultArtifactIds": list(self.result_artifact_ids),
            "resultSummary": dict(self.result_summary),
            "active": self.status in ACTIVE_PROCESSING_JOB_STATUSES,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }
        optional: dict[str, object | None] = {
            "stage": self.stage,
            "renderTriggeredAt": self.render_triggered_at,
            "startedAt": self.started_at,
            "completedAt": self.completed_at,
            "failedAt": self.failed_at,
            "failedStage": self.failed_stage,
            "renderTaskRunId": self.render_task_run_id,
            "processingRunId": self.processing_run_id,
            "errorCode": self.error_code,
            "errorMessage": self.error_message,
            "pipelineVersion": self.pipeline_version,
            "sourceType": self.source_type.value if self.source_type is not None else None,
            "sourcePath": self.source_path,
            "sourceArtifactId": self.source_artifact_id,
        }
        document.update({key: value for key, value in optional.items() if value is not None})
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
    processing_run_id: str | None = None

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
        object.__setattr__(
            self,
            "processing_run_id",
            _optional_text(self.processing_run_id, "processing_run_id"),
        )
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
            "processingRunId": self.processing_run_id,
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
    processing_run_id: str | None = None

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
        object.__setattr__(
            self,
            "processing_run_id",
            _optional_text(self.processing_run_id, "processing_run_id"),
        )
        if (
            self.category is not ArtifactCategory.VIEWABLE_MEDIA
            and self.access is ArtifactAccess.PUBLIC
        ):
            raise PersistenceValidationError(
                "SOURCE_MEDIA and INTERNAL_ARTIFACT cannot request public access"
            )


def _document_datetime(
    document: Mapping[str, object],
    key: str,
    *,
    required: bool = False,
) -> datetime | None:
    value = document.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, datetime):
        raise PersistenceValidationError(f"persisted {key} must be a datetime")
    return value


def _document_string(
    document: Mapping[str, object],
    key: str,
    *,
    required: bool = False,
) -> str | None:
    value = document.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise PersistenceValidationError(f"persisted {key} must be a string")
    return value


def processing_job_from_document(document: Mapping[str, object]) -> ProcessingJobRecord:
    """Parse a MongoDB job document without leaking BSON-specific structures."""

    job_id = _document_string(document, "jobId", required=True)
    match_id = _document_string(document, "matchId", required=True)
    job_type = _document_string(document, "jobType", required=True)
    status_raw = _document_string(document, "status", required=True)
    assert job_id is not None and match_id is not None and job_type is not None
    assert status_raw is not None
    try:
        status = ProcessingJobStatus(status_raw)
    except ValueError as error:
        raise PersistenceValidationError(
            f"persisted job status is invalid: {status_raw}"
        ) from error
    source_type_raw = _document_string(document, "sourceType")
    try:
        source_type = SourceMediaType(source_type_raw) if source_type_raw is not None else None
    except ValueError as error:
        raise PersistenceValidationError(
            f"persisted source type is invalid: {source_type_raw}"
        ) from error
    result_ids_raw = document.get("resultArtifactIds", [])
    if not isinstance(result_ids_raw, list) or not all(
        isinstance(value, str) for value in result_ids_raw
    ):
        raise PersistenceValidationError("persisted resultArtifactIds must be a string array")
    progress = document.get("progress", 0.0)
    attempt_count = document.get("attemptCount", 0)
    if not isinstance(progress, (int, float)) or isinstance(progress, bool):
        raise PersistenceValidationError("persisted progress must be numeric")
    if not isinstance(attempt_count, int) or isinstance(attempt_count, bool):
        raise PersistenceValidationError("persisted attemptCount must be an integer")
    created_at = _document_datetime(document, "createdAt", required=True)
    updated_at = _document_datetime(document, "updatedAt", required=True)
    assert created_at is not None and updated_at is not None
    result_summary_raw = document.get("resultSummary", {})
    result_summary = (
        {str(key): value for key, value in result_summary_raw.items()}
        if isinstance(result_summary_raw, Mapping)
        else {}
    )
    return ProcessingJobRecord(
        job_id=job_id,
        match_id=match_id,
        job_type=job_type,
        status=status,
        progress=float(progress),
        stage=_document_string(document, "stage"),
        render_triggered_at=_document_datetime(document, "renderTriggeredAt"),
        started_at=_document_datetime(document, "startedAt"),
        completed_at=_document_datetime(document, "completedAt"),
        failed_at=_document_datetime(document, "failedAt"),
        failed_stage=_document_string(document, "failedStage"),
        render_task_run_id=_document_string(document, "renderTaskRunId"),
        processing_run_id=_document_string(document, "processingRunId"),
        attempt_count=attempt_count,
        error_code=_document_string(document, "errorCode"),
        error_message=_document_string(document, "errorMessage"),
        pipeline_version=_document_string(document, "pipelineVersion"),
        source_type=source_type,
        source_path=_document_string(document, "sourcePath"),
        source_artifact_id=_document_string(document, "sourceArtifactId"),
        result_artifact_ids=tuple(result_ids_raw),
        result_summary=result_summary,
        created_at=created_at,
        updated_at=updated_at,
    )


def artifact_record_from_document(document: Mapping[str, object]) -> ArtifactRecord:
    """Parse provider-neutral artifact metadata loaded from MongoDB."""

    required_strings = {
        key: _document_string(document, key, required=True)
        for key in (
            "artifactId",
            "artifactType",
            "category",
            "pathname",
            "provider",
            "access",
            "contentType",
        )
    }
    size = document.get("size")
    if not isinstance(size, int) or isinstance(size, bool):
        raise PersistenceValidationError("persisted artifact size must be an integer")
    created_at = _document_datetime(document, "createdAt", required=True)
    assert created_at is not None
    try:
        return ArtifactRecord(
            artifact_id=required_strings["artifactId"] or "",
            match_id=_document_string(document, "matchId"),
            artifact_type=required_strings["artifactType"] or "",
            category=ArtifactCategory(required_strings["category"] or ""),
            pathname=required_strings["pathname"] or "",
            provider=ArtifactProvider(required_strings["provider"] or ""),
            access=ArtifactAccess(required_strings["access"] or ""),
            content_type=required_strings["contentType"] or "",
            size_bytes=size,
            created_at=created_at,
            pipeline_version=_document_string(document, "pipelineVersion"),
            url=_document_string(document, "url"),
            checksum_sha256=_document_string(document, "checksumSha256"),
            processing_run_id=_document_string(document, "processingRunId"),
        )
    except ValueError as error:
        raise PersistenceValidationError("persisted artifact enum value is invalid") from error
