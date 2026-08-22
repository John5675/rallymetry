from __future__ import annotations

from pathlib import Path

import pytest

from pickleball_vision.errors import AnalysisConfigurationError
from pickleball_vision.workflows.settings import (
    WorkflowSettings,
    workflow_task_plan,
    workflow_timeout_seconds,
)


def test_workflow_settings_default_to_cpu_and_scoped_temp_root() -> None:
    settings = WorkflowSettings.from_env({"PIPELINE_CONFIG": "plan.json"})

    assert settings.pipeline_config_path == Path("plan.json")
    assert settings.temp_root == Path("/tmp/rallymetry")
    assert settings.model_device == "cpu"
    assert settings.youtube_max_duration_seconds == 7_200
    assert settings.youtube_max_bytes == 4_000_000_000
    assert settings.youtube_pot_provider_url is None
    assert workflow_task_plan({}) == "pro"
    assert workflow_timeout_seconds({}) == 21_600


def test_workflow_settings_accept_private_youtube_challenge_provider() -> None:
    settings = WorkflowSettings.from_env(
        {
            "PIPELINE_CONFIG": "plan.json",
            "YOUTUBE_POT_PROVIDER_URL": "http://youtube-pot-provider:4416/",
        }
    )

    assert settings.youtube_pot_provider_url == "http://youtube-pot-provider:4416"


@pytest.mark.parametrize(
    "provider_url",
    [
        "youtube-pot-provider:4416",
        "file:///tmp/provider",
        "https://user:password@example.com",
    ],
)
def test_workflow_settings_reject_invalid_youtube_challenge_provider(
    provider_url: str,
) -> None:
    with pytest.raises(AnalysisConfigurationError):
        WorkflowSettings.from_env(
            {
                "PIPELINE_CONFIG": "plan.json",
                "YOUTUBE_POT_PROVIDER_URL": provider_url,
            }
        )


@pytest.mark.parametrize(
    "environment",
    [
        {"RENDER_WORKFLOW_PLAN": "unknown"},
        {"RENDER_WORKFLOW_TIMEOUT_SECONDS": "29"},
        {"RENDER_WORKFLOW_TIMEOUT_SECONDS": "86401"},
    ],
)
def test_workflow_task_compute_and_timeout_are_bounded(environment: dict[str, str]) -> None:
    with pytest.raises(AnalysisConfigurationError):
        if "RENDER_WORKFLOW_PLAN" in environment:
            workflow_task_plan(environment)
        else:
            workflow_timeout_seconds(environment)
