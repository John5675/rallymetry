from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from pickleball_vision.api.main import create_app
from pickleball_vision.api.services.render_workflows import WorkflowRun
from pickleball_vision.api.settings import ApiSettings
from pickleball_vision.config import PersistenceSettings
from pickleball_vision.persistence.models import (
    AnalyticsRecord,
    ArtifactAccess,
    ArtifactCategory,
    ArtifactProvider,
    ArtifactRecord,
    Document,
    MatchRecord,
    PlayerRecord,
    ProcessingJobRecord,
    StructuredDomainRecord,
)

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


class InMemoryApplicationPersistence:
    def __init__(self) -> None:
        self.matches: dict[str, Document] = {}
        self.players: list[Document] = []
        self.rallies: list[Document] = []
        self.shots: list[Document] = []
        self.contacts: list[Document] = []
        self.bounces: list[Document] = []
        self.analytics: list[Document] = []
        self.artifacts: list[Document] = []
        self.jobs: dict[str, Document] = {}

    async def save_match(self, record: MatchRecord) -> None:
        self.matches[record.match_id] = record.to_document()

    async def get_match(self, match_id: str) -> Document | None:
        document = self.matches.get(match_id)
        return dict(document) if document is not None else None

    async def list_matches(
        self,
        *,
        limit: int,
        offset: int,
    ) -> tuple[tuple[Document, ...], int]:
        documents = sorted(
            self.matches.values(),
            key=lambda document: str(document["updatedAt"]),
            reverse=True,
        )
        return tuple(dict(item) for item in documents[offset : offset + limit]), len(documents)

    async def patch_match(
        self,
        match_id: str,
        fields: Mapping[str, object],
        *,
        updated_at: datetime,
    ) -> Document | None:
        document = self.matches.get(match_id)
        if document is None:
            return None
        for key, value in fields.items():
            if value is None:
                document.pop(key, None)
            else:
                document[key] = value
        document["updatedAt"] = updated_at
        return dict(document)

    async def list_match_players(self, match_id: str) -> tuple[Document, ...]:
        return tuple(dict(document) for document in self.players if document["matchId"] == match_id)

    async def list_match_rallies(
        self,
        match_id: str,
        *,
        limit: int,
        offset: int,
    ) -> tuple[tuple[Document, ...], int]:
        return self._page(self.rallies, match_id, limit=limit, offset=offset)

    async def list_match_shots(
        self,
        match_id: str,
        *,
        limit: int,
        offset: int,
    ) -> tuple[tuple[Document, ...], int]:
        return self._page(self.shots, match_id, limit=limit, offset=offset)

    async def list_match_contacts(
        self,
        match_id: str,
        *,
        limit: int,
        offset: int,
    ) -> tuple[tuple[Document, ...], int]:
        return self._page(self.contacts, match_id, limit=limit, offset=offset)

    async def list_match_bounces(
        self,
        match_id: str,
        *,
        limit: int,
        offset: int,
    ) -> tuple[tuple[Document, ...], int]:
        return self._page(self.bounces, match_id, limit=limit, offset=offset)

    async def get_latest_match_analytics(self, match_id: str) -> Document | None:
        matches = [item for item in self.analytics if item["matchId"] == match_id]
        if not matches:
            return None
        return dict(matches[-1])

    async def list_match_artifacts(self, match_id: str) -> tuple[Document, ...]:
        return tuple(
            dict(document) for document in self.artifacts if document.get("matchId") == match_id
        )

    async def save_processing_job(self, record: ProcessingJobRecord) -> None:
        self.jobs[record.job_id] = record.to_document()

    async def create_processing_job_if_no_active(
        self,
        record: ProcessingJobRecord,
    ) -> tuple[Document, bool]:
        for document in self.jobs.values():
            if document.get("matchId") == record.match_id and document.get("active") is True:
                return dict(document), False
        document = record.to_document()
        self.jobs[record.job_id] = document
        return dict(document), True

    async def update_processing_job(
        self,
        job_id: str,
        fields: Mapping[str, object],
        *,
        updated_at: datetime,
    ) -> Document | None:
        document = self.jobs.get(job_id)
        if document is None:
            return None
        for key, value in fields.items():
            if value is None:
                document.pop(key, None)
            else:
                document[key] = value
        document["updatedAt"] = updated_at
        if document.get("status") in {"COMPLETE", "FAILED", "CANCELED"}:
            document["active"] = False
        return dict(document)

    async def get_processing_job(self, job_id: str) -> Document | None:
        document = self.jobs.get(job_id)
        return dict(document) if document is not None else None

    async def get_artifact(self, artifact_id: str) -> Document | None:
        for document in self.artifacts:
            if document.get("artifactId") == artifact_id:
                return dict(document)
        return None

    @staticmethod
    def _page(
        documents: list[Document],
        match_id: str,
        *,
        limit: int,
        offset: int,
    ) -> tuple[tuple[Document, ...], int]:
        matches = [item for item in documents if item["matchId"] == match_id]
        return tuple(dict(item) for item in matches[offset : offset + limit]), len(matches)


def seed_persistence() -> InMemoryApplicationPersistence:
    persistence = InMemoryApplicationPersistence()
    match = MatchRecord(
        match_id="match_seed",
        title="Seed match",
        youtube_video_id="abc123XYZ_9",
        source_artifact_id="artifact_source",
        analysis_setup={
            "calibrationArtifactId": "artifact_calibration",
            "playerAssignmentsArtifactId": "artifact_assignments",
            "ballExperimentArtifactId": "artifact_ball_experiment",
            "ballWeightsArtifactId": "artifact_ball_weights",
        },
        summary={"rallyCount": 1},
        created_at=NOW,
        updated_at=NOW,
    )
    player = PlayerRecord(
        match_id="match_seed",
        player_id="JOHN",
        display_name="John",
        logical_identity="ME",
        team="NEAR",
        created_at=NOW,
        updated_at=NOW,
    )
    rally = StructuredDomainRecord(
        match_id="match_seed",
        record_id="rally_1",
        payload={"startFrame": 10, "endFrame": 100},
        confidence=0.8,
        timestamp_seconds=1.0,
        created_at=NOW,
    )
    shot = StructuredDomainRecord(
        match_id="match_seed",
        record_id="shot_1",
        payload={"shotType": "SERVE", "hitterId": "JOHN"},
        confidence=0.75,
        timestamp_seconds=1.2,
        created_at=NOW,
    )
    contact = StructuredDomainRecord(
        match_id="match_seed",
        record_id="contact_1",
        payload={"candidatePlayers": ["JOHN"]},
        confidence=0.72,
        timestamp_seconds=1.2,
        created_at=NOW,
    )
    bounce = StructuredDomainRecord(
        match_id="match_seed",
        record_id="bounce_1",
        payload={"courtPosition": {"x": 2.5, "y": 9.1}},
        confidence=0.68,
        timestamp_seconds=1.8,
        created_at=NOW,
    )
    analytics = AnalyticsRecord(
        match_id="match_seed",
        analytics_id="analytics_v1",
        calculation_version="match-analytics-v1",
        metrics={"rallyCount": {"value": 1}},
        created_at=NOW,
    )
    artifact = ArtifactRecord(
        artifact_id="artifact_review",
        match_id="match_seed",
        artifact_type="annotated_video",
        category=ArtifactCategory.VIEWABLE_MEDIA,
        pathname="viewable_media/match_seed/random/annotated.mp4",
        provider=ArtifactProvider.VERCEL_BLOB,
        access=ArtifactAccess.PRIVATE,
        content_type="video/mp4",
        size_bytes=123,
        created_at=NOW,
        url="https://private.example.test/annotated.mp4",
    )
    source_artifact = ArtifactRecord(
        artifact_id="artifact_source",
        match_id="match_seed",
        artifact_type="source_video",
        category=ArtifactCategory.SOURCE_MEDIA,
        pathname="source_media/match_seed/random/source.mp4",
        provider=ArtifactProvider.VERCEL_BLOB,
        access=ArtifactAccess.PRIVATE,
        content_type="video/mp4",
        size_bytes=456,
        created_at=NOW,
        url="https://private.example.test/source.mp4",
    )
    persistence.matches[match.match_id] = match.to_document()
    persistence.players.append(player.to_document())
    persistence.rallies.append(rally.to_document())
    persistence.shots.append(shot.to_document())
    persistence.contacts.append(contact.to_document())
    persistence.bounces.append(bounce.to_document())
    persistence.analytics.append(analytics.to_document())
    persistence.artifacts.append(artifact.to_document())
    persistence.artifacts.append(source_artifact.to_document())
    for artifact_id, artifact_type, filename in (
        ("artifact_calibration", "court_calibration", "calibration.json"),
        ("artifact_assignments", "player_assignments", "player-assignments.json"),
        ("artifact_ball_experiment", "ball_experiment", "ball-experiment.json"),
        ("artifact_ball_weights", "ball_weights", "ball-weights.pt"),
    ):
        persistence.artifacts.append(
            ArtifactRecord(
                artifact_id=artifact_id,
                match_id="match_seed",
                artifact_type=artifact_type,
                category=ArtifactCategory.INTERNAL_ARTIFACT,
                pathname=f"internal_artifact/match_seed/random/{filename}",
                provider=ArtifactProvider.VERCEL_BLOB,
                access=ArtifactAccess.PRIVATE,
                content_type="application/octet-stream",
                size_bytes=10,
                created_at=NOW,
                url=f"https://private.example.test/{filename}",
            ).to_document()
        )
    return persistence


class FakeWorkflowClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.starts: list[tuple[str, str]] = []

    async def start_analysis(self, *, job_id: str, match_id: str) -> WorkflowRun:
        self.starts.append((job_id, match_id))
        if self.fail:
            raise RuntimeError("synthetic Render outage")
        return WorkflowRun(run_id="trn-test-123", status="pending")

    async def get_run(self, run_id: str) -> WorkflowRun:
        return WorkflowRun(run_id=run_id, status="pending")


def test_health_request_logging_request_id_and_cors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    persistence = seed_persistence()
    settings = ApiSettings(
        cors_origins=("https://friends.example.com",),
        persistence=PersistenceSettings(mongodb_url="mongodb://test.invalid"),
    )
    app = create_app(settings=settings, persistence=persistence)

    with (
        caplog.at_level(logging.INFO, logger="pickleball_vision.api"),
        TestClient(app) as client,
    ):
        response = client.get("/health", headers={"X-Request-ID": "request-123"})
        preflight = client.options(
            "/api/matches",
            headers={
                "Origin": "https://friends.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "pickleball-vision-api",
        "version": "0.1.0",
        "databaseConfigured": True,
        "databaseReady": True,
        "artifactBackend": "local",
    }
    assert response.headers["X-Request-ID"] == "request-123"
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "https://friends.example.com"
    assert any(record.message == "api_request_completed" for record in caplog.records)


def test_match_crud_uses_json_ids_and_consistent_validation_errors() -> None:
    persistence = seed_persistence()
    app = create_app(settings=ApiSettings(), persistence=persistence)

    with TestClient(app) as client:
        created = client.post(
            "/api/matches",
            json={"title": "New match", "youtubeVideoId": "newVideo_12"},
        )
        match_id = created.json()["matchId"]
        listed = client.get("/api/matches?limit=1&offset=0")
        fetched = client.get(f"/api/matches/{match_id}")
        patched = client.patch(f"/api/matches/{match_id}", json={"title": "Updated match"})
        invalid = client.patch(
            f"/api/matches/{match_id}",
            headers={"X-Request-ID": "validation-1"},
            json={"unsupported": True},
        )
        missing = client.get("/api/matches/match_missing")

    assert created.status_code == 201
    assert created.json()["youtubeVideoId"] == "newVideo_12"
    assert "_id" not in created.json()
    assert listed.status_code == 200
    assert listed.json()["total"] == 2
    assert len(listed.json()["items"]) == 1
    assert fetched.json()["matchId"] == match_id
    assert patched.json()["title"] == "Updated match"
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "validation_error"
    assert invalid.json()["error"]["requestId"] == "validation-1"
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "resource_not_found"


def test_match_accepts_youtube_id_starting_with_underscore() -> None:
    persistence = seed_persistence()
    app = create_app(settings=ApiSettings(), persistence=persistence)

    with TestClient(app) as client:
        response = client.post(
            "/api/matches",
            json={
                "title": "Underscore video ID",
                "youtubeVideoId": "_cPF1fTnk0Y",
            },
        )

    assert response.status_code == 201
    assert response.json()["youtubeVideoId"] == "_cPF1fTnk0Y"


def test_match_scoped_structured_endpoints_do_not_expose_mongo_ids() -> None:
    persistence = seed_persistence()
    app = create_app(settings=ApiSettings(), persistence=persistence)

    with TestClient(app) as client:
        players = client.get("/api/matches/match_seed/players")
        rallies = client.get("/api/matches/match_seed/rallies")
        shots = client.get("/api/matches/match_seed/shots")
        contacts = client.get("/api/matches/match_seed/contacts")
        bounces = client.get("/api/matches/match_seed/bounces")
        analytics = client.get("/api/matches/match_seed/analytics")
        artifacts = client.get("/api/matches/match_seed/artifacts")

    assert players.json()["items"][0]["logicalIdentity"] == "ME"
    assert rallies.json()["items"][0]["payload"]["startFrame"] == 10
    assert shots.json()["items"][0]["payload"]["shotType"] == "SERVE"
    assert contacts.json()["items"][0]["recordId"] == "contact_1"
    assert bounces.json()["items"][0]["payload"]["courtPosition"]["y"] == 9.1
    assert analytics.json()["metrics"]["rallyCount"]["value"] == 1
    assert artifacts.json()["items"][0]["access"] == "PRIVATE"
    for response in (players, rallies, shots, contacts, bounces, analytics, artifacts):
        assert response.status_code == 200
        assert "_id" not in response.text


def test_process_endpoint_only_persists_queued_job_and_returns_202() -> None:
    persistence = seed_persistence()
    workflow = FakeWorkflowClient()
    app = create_app(
        settings=ApiSettings(),
        persistence=persistence,
        workflow_client=workflow,
    )

    with TestClient(app) as client:
        queued = client.post("/api/matches/match_seed/process")
        payload = queued.json()
        fetched = client.get(f"/api/jobs/{payload['jobId']}")

    assert queued.status_code == 202
    assert queued.headers["Location"] == f"/api/jobs/{payload['jobId']}"
    assert payload["status"] == "QUEUED"
    assert payload["progress"] == 0.0
    assert payload["sourceType"] == "BLOB"
    assert payload["sourceArtifactId"] == "artifact_source"
    assert payload["renderTaskRunId"] == "trn-test-123"
    assert payload["processingRunId"].startswith("run_")
    assert len(persistence.jobs) == 1
    assert workflow.starts == [(payload["jobId"], "match_seed")]
    assert fetched.status_code == 200
    assert fetched.json() == payload


def test_duplicate_process_request_returns_active_job_without_second_render_run() -> None:
    persistence = seed_persistence()
    workflow = FakeWorkflowClient()
    app = create_app(settings=ApiSettings(), persistence=persistence, workflow_client=workflow)

    with TestClient(app) as client:
        first = client.post("/api/matches/match_seed/process")
        second = client.post("/api/matches/match_seed/process")

    assert first.status_code == second.status_code == 202
    assert second.json()["jobId"] == first.json()["jobId"]
    assert len(workflow.starts) == 1


def test_render_trigger_failure_marks_job_failed_and_returns_safe_503() -> None:
    persistence = seed_persistence()
    app = create_app(
        settings=ApiSettings(),
        persistence=persistence,
        workflow_client=FakeWorkflowClient(fail=True),
    )

    with TestClient(app) as client:
        response = client.post("/api/matches/match_seed/process")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "workflow_trigger_failed"
    job = next(iter(persistence.jobs.values()))
    assert job["status"] == "FAILED"
    assert job["errorCode"] == "RENDER_TRIGGER_FAILED"


def test_process_endpoint_requires_available_source_media() -> None:
    persistence = seed_persistence()
    match = persistence.matches["match_seed"]
    match.pop("sourceArtifactId")
    app = create_app(settings=ApiSettings(), persistence=persistence)

    with TestClient(app) as client:
        response = client.post("/api/matches/match_seed/process")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "source_media_required"
    assert not persistence.jobs


def test_process_endpoint_requires_private_hosted_analysis_setup() -> None:
    persistence = seed_persistence()
    persistence.matches["match_seed"]["analysisSetup"] = {}
    app = create_app(
        settings=ApiSettings(),
        persistence=persistence,
        workflow_client=FakeWorkflowClient(),
    )

    with TestClient(app) as client:
        response = client.post("/api/matches/match_seed/process")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "analysis_setup_required"
    assert not persistence.jobs


def test_api_starts_degraded_without_mongodb_and_data_routes_return_503() -> None:
    app = create_app(settings=ApiSettings())

    with TestClient(app) as client:
        health = client.get("/health")
        matches = client.get("/api/matches")

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["databaseReady"] is False
    assert matches.status_code == 503
    assert matches.json()["error"]["code"] == "persistence_unavailable"
