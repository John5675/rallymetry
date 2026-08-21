"""Render Workflows SDK adapter kept behind a small application interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from render_sdk import RenderAsync  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class WorkflowRun:
    """Initial infrastructure state returned without waiting for task completion."""

    run_id: str
    status: str


class AnalysisWorkflowClient(Protocol):
    async def start_analysis(self, *, job_id: str, match_id: str) -> WorkflowRun: ...

    async def get_run(self, run_id: str) -> WorkflowRun: ...


class RenderWorkflowClient:
    """Trigger and inspect Render task runs without leaking SDK calls into routes."""

    def __init__(
        self,
        *,
        api_key: str,
        task_identifier: str,
        client: RenderAsync | None = None,
    ) -> None:
        self._task_identifier = task_identifier
        self._client = client or RenderAsync(token=api_key)

    async def start_analysis(self, *, job_id: str, match_id: str) -> WorkflowRun:
        initial = await self._client.workflows.start_task(
            self._task_identifier,
            {"job_id": job_id, "match_id": match_id},
        )
        return WorkflowRun(run_id=initial.id, status=str(initial.status))

    async def get_run(self, run_id: str) -> WorkflowRun:
        details = await self._client.workflows.get_task_run(run_id)
        status = getattr(details, "status", "unknown")
        return WorkflowRun(run_id=cast(str, details.id), status=str(status))
