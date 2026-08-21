from __future__ import annotations

import pytest

from pickleball_vision.api.settings import ApiSettings
from pickleball_vision.config import ArtifactBackend
from pickleball_vision.errors import ConfigurationError


def test_api_settings_load_cors_mongodb_and_storage_environment() -> None:
    settings = ApiSettings.from_env(
        {
            "CORS_ORIGINS": "http://localhost:5173, https://friends.example.com/",
            "MONGODB_URL": "mongodb+srv://user:secret@example.test/",
            "MONGODB_DATABASE": "rallymetry_test",
            "PICKLEBALL_VISION_ARTIFACT_BACKEND": "vercel_blob",
            "BLOB_READ_WRITE_TOKEN": "server-secret",
            "PUBLIC_BLOB_READ_WRITE_TOKEN": "public-server-secret",
            "RENDER_API_KEY": "render-secret",
            "RENDER_WORKFLOW_TASK": "rallymetry-analysis/analyze_match",
            "DEFAULT_ANALYSIS_PROFILE_MATCH_ID": "match_profile",
        }
    )

    assert settings.cors_origins == (
        "http://localhost:5173",
        "https://friends.example.com",
    )
    assert settings.persistence.artifact_backend is ArtifactBackend.VERCEL_BLOB
    public = settings.public_values()
    assert "server-secret" not in repr(public)
    assert "public-server-secret" not in repr(public)
    assert "user:secret" not in repr(public)
    assert "render-secret" not in repr(public)
    assert public["renderWorkflowConfigured"] is True
    assert public["defaultAnalysisProfileConfigured"] is True
    assert settings.default_analysis_profile_match_id == "match_profile"


def test_api_settings_accept_local_vercel_and_custom_frontend_origins() -> None:
    settings = ApiSettings.from_env(
        {
            "CORS_ORIGINS": (
                "http://localhost:5173,https://rallymetry.vercel.app,https://matches.example.com"
            )
        }
    )

    assert settings.cors_origins == (
        "http://localhost:5173",
        "https://rallymetry.vercel.app",
        "https://matches.example.com",
    )


@pytest.mark.parametrize(
    "origins",
    ["", "ftp://example.com", "https://example.com/path", "*,https://example.com"],
)
def test_api_settings_reject_invalid_cors_origins(origins: str) -> None:
    with pytest.raises(ConfigurationError) as raised:
        ApiSettings.from_env({"CORS_ORIGINS": origins})

    assert raised.value.details["setting"] == "CORS_ORIGINS"


def test_api_settings_require_render_trigger_values_together() -> None:
    with pytest.raises(ConfigurationError) as raised:
        ApiSettings.from_env({"RENDER_API_KEY": "secret"})

    assert raised.value.details["setting"] == "RENDER_WORKFLOW_TASK"


def test_api_settings_validate_render_task_slug() -> None:
    with pytest.raises(ConfigurationError):
        ApiSettings.from_env({"RENDER_API_KEY": "secret", "RENDER_WORKFLOW_TASK": "analyze_match"})
