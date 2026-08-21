from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

from render_sdk import RenderAsync  # type: ignore[import-untyped]

from pickleball_vision.api.services.render_workflows import RenderWorkflowClient


class FakeWorkflowsService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def start_task(self, task: str, input_data: object) -> object:
        self.calls.append((task, input_data))
        return SimpleNamespace(id="trn-123", status="pending")

    async def get_task_run(self, run_id: str) -> object:
        return SimpleNamespace(id=run_id, status="running")


def test_render_client_starts_small_identifier_only_payload_without_waiting() -> None:
    workflows = FakeWorkflowsService()
    sdk = SimpleNamespace(workflows=workflows)
    client = RenderWorkflowClient(
        api_key="unused",
        task_identifier="rallymetry-analysis/analyze_match",
        client=cast(RenderAsync, cast(Any, sdk)),
    )

    run = asyncio.run(client.start_analysis(job_id="job-1", match_id="match-1"))

    assert run.run_id == "trn-123"
    assert workflows.calls == [
        (
            "rallymetry-analysis/analyze_match",
            {"job_id": "job-1", "match_id": "match-1"},
        )
    ]
