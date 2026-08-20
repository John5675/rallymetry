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
        }
    )

    assert settings.cors_origins == (
        "http://localhost:5173",
        "https://friends.example.com",
    )
    assert settings.persistence.artifact_backend is ArtifactBackend.VERCEL_BLOB
    public = settings.public_values()
    assert "server-secret" not in repr(public)
    assert "user:secret" not in repr(public)


@pytest.mark.parametrize(
    "origins",
    ["", "ftp://example.com", "https://example.com/path", "*,https://example.com"],
)
def test_api_settings_reject_invalid_cors_origins(origins: str) -> None:
    with pytest.raises(ConfigurationError) as raised:
        ApiSettings.from_env({"CORS_ORIGINS": origins})

    assert raised.value.details["setting"] == "CORS_ORIGINS"
