from __future__ import annotations

import json
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
VISION_ROOT = REPOSITORY_ROOT / "services" / "vision"


def test_vercel_configuration_builds_web_app_and_rewrites_spa_routes() -> None:
    configuration = json.loads((REPOSITORY_ROOT / "vercel.json").read_text(encoding="utf-8"))

    assert configuration["framework"] == "vite"
    assert configuration["installCommand"] == "npm --prefix apps/web ci"
    assert configuration["buildCommand"] == "npm --prefix apps/web run build"
    assert configuration["outputDirectory"] == "apps/web/dist"
    assert configuration["rewrites"] == [{"source": "/(.*)", "destination": "/index.html"}]


def test_browser_environment_example_contains_no_server_credentials() -> None:
    example = (REPOSITORY_ROOT / "apps" / "web" / ".env.example").read_text(encoding="utf-8")

    assert "VITE_API_BASE_URL=" in example
    assert "MONGODB" not in example
    assert "BLOB_READ_WRITE_TOKEN" not in example
    assert "PUBLIC_BLOB_READ_WRITE_TOKEN" not in example


def test_api_container_requirements_match_locked_direct_versions() -> None:
    lock = tomllib.loads((VISION_ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked_versions = {package["name"]: package["version"] for package in lock["package"]}
    requirements = {
        name: version
        for line in (VISION_ROOT / "requirements-api.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
        for name, version in [line.split("==", maxsplit=1)]
    }

    assert requirements == {
        "dnspython": locked_versions["dnspython"],
        "fastapi": locked_versions["fastapi"],
        "pymongo": locked_versions["pymongo"],
        "uvicorn": locked_versions["uvicorn"],
        "vercel": locked_versions["vercel"],
    }


def test_worker_publication_plan_never_exposes_internal_artifacts() -> None:
    plan = json.loads(
        (REPOSITORY_ROOT / "docs" / "examples" / "worker-pipeline-plan.json").read_text(
            encoding="utf-8"
        )
    )

    public_artifacts = [item for item in plan["artifacts"] if item.get("access") == "PUBLIC"]
    assert public_artifacts
    assert all(item["category"] == "VIEWABLE_MEDIA" for item in public_artifacts)
    assert all(
        item.get("access") == "PRIVATE"
        for item in plan["artifacts"]
        if item["category"] == "INTERNAL_ARTIFACT"
    )
