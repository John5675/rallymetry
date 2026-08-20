from __future__ import annotations

from datetime import UTC, datetime

import pytest

from pickleball_vision.errors import ErrorCode, PersistenceValidationError
from pickleball_vision.persistence.models import (
    AnalyticsRecord,
    ArtifactAccess,
    ArtifactCategory,
    ArtifactProvider,
    ArtifactRecord,
    MatchRecord,
    ProcessingJobRecord,
    StructuredDomainRecord,
)

NOW = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)


def test_match_record_stores_compact_metadata_and_youtube_id() -> None:
    record = MatchRecord(
        match_id="match-1",
        title="John and Denny versus Oksana and Diana",
        youtube_video_id="abc123xyz89",
        source_artifact_id="artifact-source",
        pipeline_version="0.1.0",
        model_versions={"ball": "ball-v3"},
        summary={"rallyCount": 18},
        artifact_ids=("artifact-source", "artifact-review"),
        created_at=NOW,
        updated_at=NOW,
    )

    document = record.to_document()

    assert document["_id"] == "match-1"
    assert document["youtubeVideoId"] == "abc123xyz89"
    assert document["summary"] == {"rallyCount": 18}
    assert document["artifactIds"] == ["artifact-source", "artifact-review"]
    assert "rallies" not in document
    assert "shots" not in document


def test_structured_events_and_analytics_retain_versions_and_confidence() -> None:
    rally = StructuredDomainRecord(
        match_id="match-1",
        record_id="rally-1",
        payload={"startFrame": 10, "endFrame": 90},
        confidence=0.82,
        timestamp_seconds=1.5,
        pipeline_version="0.1.0",
        model_version="rally-rules-v2",
        created_at=NOW,
    )
    analytics = AnalyticsRecord(
        match_id="match-1",
        analytics_id="analytics-v1",
        calculation_version="match-analytics-v1",
        metrics={"rallyCount": {"value": 1}},
        input_artifact_ids=("rallies-json", "shots-json"),
        pipeline_version="0.1.0",
        created_at=NOW,
    )

    assert rally.to_document()["confidence"] == 0.82
    assert rally.to_document()["payload"] == {"startFrame": 10, "endFrame": 90}
    assert analytics.to_document()["inputArtifactIds"] == ["rallies-json", "shots-json"]


def test_artifact_record_is_a_reference_without_credentials_or_binary_data() -> None:
    record = ArtifactRecord(
        artifact_id="artifact-1",
        match_id="match-1",
        artifact_type="annotated_video",
        category=ArtifactCategory.VIEWABLE_MEDIA,
        pathname="viewable_media/match-1/random/annotated.mp4",
        provider=ArtifactProvider.VERCEL_BLOB,
        access=ArtifactAccess.PUBLIC,
        content_type="video/mp4",
        size_bytes=1234,
        created_at=NOW,
        pipeline_version="0.1.0",
        url="https://store.public.blob.vercel-storage.com/random.mp4",
        checksum_sha256="a" * 64,
    )

    document = record.to_document()

    assert document["size"] == 1234
    assert document["access"] == "PUBLIC"
    assert "token" not in repr(document).lower()
    assert all(not isinstance(value, bytes) for value in document.values())


@pytest.mark.parametrize(
    "factory",
    [
        lambda: StructuredDomainRecord(
            match_id="match-1",
            record_id="rally-1",
            payload={},
            confidence=1.1,
        ),
        lambda: ProcessingJobRecord(
            job_id="job-1",
            match_id="match-1",
            job_type="analyze",
            progress=-0.1,
        ),
        lambda: ArtifactRecord(
            artifact_id="artifact-1",
            match_id="match-1",
            artifact_type="source",
            category=ArtifactCategory.SOURCE_MEDIA,
            pathname="source_media/match-1/random/source.mp4",
            provider=ArtifactProvider.VERCEL_BLOB,
            access=ArtifactAccess.PUBLIC,
            content_type="video/mp4",
            size_bytes=1,
            created_at=NOW,
        ),
        lambda: MatchRecord(
            match_id="match-1",
            created_at=datetime(2026, 1, 1),
            updated_at=NOW,
        ),
    ],
)
def test_invalid_persistence_records_raise_typed_errors(factory: object) -> None:
    with pytest.raises(PersistenceValidationError) as raised:
        assert callable(factory)
        factory()

    assert raised.value.code is ErrorCode.PERSISTENCE_VALIDATION
