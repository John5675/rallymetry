from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from pickleball_vision.analysis_runtime.source import SourceMediaStager
from pickleball_vision.persistence.models import (
    ArtifactAccess,
    ArtifactCategory,
    ArtifactProvider,
    ArtifactPutRequest,
    ArtifactRecord,
    Document,
    ProcessingJobRecord,
    SourceMediaType,
)
from pickleball_vision.workflows.setup import (
    SETUP_FILENAMES,
    MatchSetupStager,
    _bind_calibration_to_runtime_source,
)

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


class ArtifactLookup:
    def __init__(self, artifacts: list[ArtifactRecord]) -> None:
        self.documents = {item.artifact_id: item.to_document() for item in artifacts}

    async def get_artifact(self, artifact_id: str) -> Document | None:
        document = self.documents.get(artifact_id)
        return dict(document) if document is not None else None


class DownloadOnlyBlobStore:
    def __init__(self) -> None:
        self.destinations: list[Path] = []

    async def put(self, request: ArtifactPutRequest) -> ArtifactRecord:
        del request
        raise AssertionError("staging must not upload")

    async def get(self, artifact: ArtifactRecord, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(artifact.artifact_id.encode())
        self.destinations.append(destination)
        return destination

    async def delete(self, artifact: ArtifactRecord) -> None:
        del artifact

    async def exists(self, artifact: ArtifactRecord) -> bool:
        del artifact
        return True


class FakeYouTubeDownloader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    async def download(self, video_id: str, *, destination_dir: Path) -> Path:
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / "source.mp4"
        destination.write_bytes(b"youtube")
        self.calls.append((video_id, destination_dir))
        return destination


def _artifact(
    artifact_id: str,
    *,
    category: ArtifactCategory,
    pathname: str,
    match_id: str = "match_test",
) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=artifact_id,
        match_id=match_id,
        artifact_type=artifact_id,
        category=category,
        pathname=pathname,
        provider=ArtifactProvider.VERCEL_BLOB,
        access=ArtifactAccess.PRIVATE,
        content_type="application/octet-stream",
        size_bytes=10,
        created_at=NOW,
        url=f"https://private.example.test/{artifact_id}",
    )


def test_workflow_stages_source_and_required_private_setup_into_job_input(tmp_path: Path) -> None:
    source = _artifact(
        "artifact_source",
        category=ArtifactCategory.SOURCE_MEDIA,
        pathname="source_media/match_test/random/source.mp4",
    )
    setup_records = [
        _artifact(
            f"artifact_{field}",
            category=ArtifactCategory.INTERNAL_ARTIFACT,
            pathname=f"internal_artifact/match_test/random/{filename}",
        )
        for field, filename in SETUP_FILENAMES.items()
    ]
    lookup = ArtifactLookup([source, *setup_records])
    store = DownloadOnlyBlobStore()
    job = ProcessingJobRecord(
        job_id="job_test",
        match_id="match_test",
        job_type="analyze_match",
        source_type=SourceMediaType.BLOB,
        source_artifact_id=source.artifact_id,
        processing_run_id="run_test",
        created_at=NOW,
        updated_at=NOW,
    )
    match_document: Document = {
        "analysisSetup": {field: f"artifact_{field}" for field in SETUP_FILENAMES}
    }
    workspace = tmp_path / "job_test"

    staged_source = asyncio.run(SourceMediaStager(lookup, store).stage(job, workspace=workspace))
    staged_setup = asyncio.run(
        MatchSetupStager(lookup, store).stage(
            match_id="match_test",
            match_document=match_document,
            workspace=workspace,
        )
    )

    assert staged_source == workspace / "input" / "source.mp4"
    assert staged_source.read_bytes() == b"artifact_source"
    assert {path.name for path in staged_setup.values()} == set(SETUP_FILENAMES.values())
    assert len(store.destinations) == 5


def test_workflow_stages_youtube_source_only_inside_job_workspace(tmp_path: Path) -> None:
    lookup = ArtifactLookup([])
    store = DownloadOnlyBlobStore()
    downloader = FakeYouTubeDownloader()
    job = ProcessingJobRecord(
        job_id="job_youtube",
        match_id="match_test",
        job_type="analyze_match",
        source_type=SourceMediaType.YOUTUBE,
        youtube_video_id="_cPF1fTnk0Y",
        processing_run_id="run_test",
        created_at=NOW,
        updated_at=NOW,
    )
    workspace = tmp_path / "job_youtube"

    staged = asyncio.run(
        SourceMediaStager(
            lookup,
            store,
            youtube_downloader=downloader,
        ).stage(job, workspace=workspace)
    )

    assert staged == workspace / "input" / "source.mp4"
    assert staged.read_bytes() == b"youtube"
    assert downloader.calls == [("_cPF1fTnk0Y", workspace / "input")]
    assert store.destinations == []


def test_workflow_accepts_explicit_shared_analysis_profile_owner(tmp_path: Path) -> None:
    setup_records = [
        _artifact(
            f"profile_{field}",
            category=ArtifactCategory.INTERNAL_ARTIFACT,
            pathname=f"internal_artifact/profile/random/{filename}",
            match_id="match_profile",
        )
        for field, filename in SETUP_FILENAMES.items()
    ]
    lookup = ArtifactLookup(setup_records)
    store = DownloadOnlyBlobStore()
    match_document: Document = {
        "analysisProfileMatchId": "match_profile",
        "analysisSetup": {field: f"profile_{field}" for field in SETUP_FILENAMES},
    }

    staged = asyncio.run(
        MatchSetupStager(lookup, store).stage(
            match_id="match_new",
            match_document=match_document,
            workspace=tmp_path / "match_new",
        )
    )

    assert {path.name for path in staged.values()} == set(SETUP_FILENAMES.values())
    assert len(store.destinations) == 4


def test_runtime_calibration_binding_changes_only_the_staged_copy(tmp_path: Path) -> None:
    staged = tmp_path / "input" / "calibration.json"
    staged.parent.mkdir(parents=True)
    staged.write_text(
        '{"schema_version":2,"source":{"video_path":"/original/match.mp4"}}',
        encoding="utf-8",
    )
    runtime_source = tmp_path / "input" / "source.mp4"

    _bind_calibration_to_runtime_source(staged, source_path=runtime_source)

    payload = json.loads(staged.read_text(encoding="utf-8"))
    assert payload["source"]["video_path"] == str(runtime_source.resolve())
    assert payload["runtime_binding"] == {
        "mode": "temporary_hosted_copy",
        "original_video_path": "/original/match.mp4",
        "runtime_video_path": str(runtime_source.resolve()),
        "original_artifact_modified": False,
    }
