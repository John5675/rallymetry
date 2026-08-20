"""Hosted persistence contracts and provider adapters for Milestone 19."""

from pickleball_vision.persistence.artifacts import (
    ArtifactStore,
    LocalArtifactStore,
    VercelBlobArtifactStore,
    create_artifact_store,
)
from pickleball_vision.persistence.models import (
    AnalyticsRecord,
    ArtifactAccess,
    ArtifactCategory,
    ArtifactProvider,
    ArtifactPutRequest,
    ArtifactRecord,
    CorrectionRecord,
    MatchRecord,
    PlayerRecord,
    ProcessingJobRecord,
    ProcessingJobStatus,
    StructuredCollection,
    StructuredDomainRecord,
)
from pickleball_vision.persistence.mongodb import MongoPersistence

__all__ = [
    "AnalyticsRecord",
    "ArtifactAccess",
    "ArtifactCategory",
    "ArtifactProvider",
    "ArtifactPutRequest",
    "ArtifactRecord",
    "ArtifactStore",
    "CorrectionRecord",
    "LocalArtifactStore",
    "MatchRecord",
    "MongoPersistence",
    "PlayerRecord",
    "ProcessingJobRecord",
    "ProcessingJobStatus",
    "StructuredCollection",
    "StructuredDomainRecord",
    "VercelBlobArtifactStore",
    "create_artifact_store",
]
