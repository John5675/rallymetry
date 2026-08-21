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
    assert workflow_task_plan({}) == "pro"
    assert workflow_timeout_seconds({}) == 21_600


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
