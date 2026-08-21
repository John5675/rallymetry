"""Trusted pipeline-plan and publication models for analysis orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from pickleball_vision.persistence.models import (
    AnalyticsRecord,
    ArtifactAccess,
    ArtifactCategory,
    PlayerRecord,
    ProcessingJobStatus,
    StructuredCollection,
    StructuredDomainRecord,
)


@dataclass(frozen=True, slots=True)
class PipelineStagePlan:
    stage: ProcessingJobStatus
    progress: float
    argv: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StructuredResultPlan:
    collection: str
    path: Path
    records_key: str | None = None
    id_field: str | None = None
    timestamp_field: str | None = None
    confidence_field: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactResultPlan:
    path: Path
    artifact_type: str
    category: ArtifactCategory
    access: ArtifactAccess | None = None
    required: bool = True


@dataclass(frozen=True, slots=True)
class PipelinePlan:
    plan_version: str
    pipeline_version: str
    stages: tuple[PipelineStagePlan, ...]
    structured_results: tuple[StructuredResultPlan, ...] = ()
    artifacts: tuple[ArtifactResultPlan, ...] = ()


@dataclass(frozen=True, slots=True)
class ArtifactPublication:
    source_path: Path
    artifact_type: str
    category: ArtifactCategory
    access: ArtifactAccess | None = None


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    """Compact structured records plus references to generated local files."""

    players: tuple[PlayerRecord, ...] = ()
    structured: Mapping[StructuredCollection, tuple[StructuredDomainRecord, ...]] = field(
        default_factory=dict
    )
    analytics: AnalyticsRecord | None = None
    artifacts: tuple[ArtifactPublication, ...] = ()
