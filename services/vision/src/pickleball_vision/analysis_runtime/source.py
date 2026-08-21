"""Source-media staging without mutating local or hosted originals."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pickleball_vision.analysis_runtime.youtube import YouTubeDownloader
from pickleball_vision.errors import AnalysisSourceError, ArtifactStorageError
from pickleball_vision.persistence.artifacts import ArtifactStore
from pickleball_vision.persistence.models import (
    ArtifactCategory,
    ArtifactProvider,
    Document,
    ProcessingJobRecord,
    SourceMediaType,
    artifact_record_from_document,
)


class ArtifactLookup(Protocol):
    async def get_artifact(self, artifact_id: str) -> Document | None: ...


class SourceMediaStager:
    """Resolve direct local paths or materialize configured artifact references."""

    def __init__(
        self,
        persistence: ArtifactLookup,
        artifact_store: ArtifactStore,
        *,
        youtube_downloader: YouTubeDownloader | None = None,
    ) -> None:
        self._persistence = persistence
        self._artifact_store = artifact_store
        self._youtube_downloader = youtube_downloader

    async def stage(self, job: ProcessingJobRecord, *, workspace: Path) -> Path:
        if job.source_type is SourceMediaType.LOCAL_PATH and job.source_path is not None:
            return self._validate_local_path(Path(job.source_path))
        if job.source_type is SourceMediaType.YOUTUBE:
            if self._youtube_downloader is None or job.youtube_video_id is None:
                raise AnalysisSourceError("YouTube source download is not configured")
            return await self._youtube_downloader.download(
                job.youtube_video_id,
                destination_dir=workspace / "input",
            )
        artifact_id = job.source_artifact_id
        if artifact_id is None:
            raise AnalysisSourceError("job has no source-media reference")
        document = await self._persistence.get_artifact(artifact_id)
        if document is None:
            raise AnalysisSourceError(f"source artifact {artifact_id} does not exist")
        artifact = artifact_record_from_document(document)
        if artifact.match_id != job.match_id:
            raise AnalysisSourceError("source artifact belongs to a different match")
        if artifact.category is not ArtifactCategory.SOURCE_MEDIA:
            raise AnalysisSourceError("source artifact is not categorized as SOURCE_MEDIA")
        expected_provider = (
            ArtifactProvider.LOCAL
            if job.source_type is SourceMediaType.LOCAL_PATH
            else ArtifactProvider.VERCEL_BLOB
        )
        if artifact.provider is not expected_provider:
            raise AnalysisSourceError("source type does not match the artifact provider")
        suffix = Path(artifact.pathname).suffix or ".media"
        destination = workspace / "input" / f"source{suffix}"
        try:
            return await self._artifact_store.get(artifact, destination)
        except Exception as error:
            if isinstance(error, (ArtifactStorageError, AnalysisSourceError)):
                raise
            raise AnalysisSourceError(
                f"artifact retrieval failed ({type(error).__name__})"
            ) from error

    @staticmethod
    def _validate_local_path(path: Path) -> Path:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            raise AnalysisSourceError(f"local source does not exist: {resolved}")
        if not resolved.is_file():
            raise AnalysisSourceError(f"local source is not a file: {resolved}")
        return resolved
