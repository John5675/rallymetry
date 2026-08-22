from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pickleball_vision.analysis_runtime.pipeline import (
    PlannedCliPipelineRunner,
    load_pipeline_plan,
)
from pickleball_vision.errors import (
    AnalysisConfigurationError,
    AnalysisPipelineError,
    AnalysisSetupRequiredError,
)
from pickleball_vision.persistence.models import ProcessingJobRecord, ProcessingJobStatus

NOW = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)


def _write_plan(path: Path, *, command: str = "detect-people") -> None:
    path.write_text(
        json.dumps(
            {
                "planVersion": "1",
                "pipelineVersion": "pipeline-test",
                "stages": [
                    {
                        "stage": "PLAYER_PROCESSING",
                        "progress": 0.2,
                        "argv": [command, "{source}", "--output-dir", "{workspace}/people"],
                        "cleanupPaths": ["people/intermediate.mp4"],
                    }
                ],
                "structuredResults": [
                    {
                        "collection": "rallies",
                        "path": "rallies/rallies.json",
                        "recordsKey": "rallies",
                        "idField": "rallyId",
                        "timestampField": "startTimestamp",
                        "confidenceField": "confidence",
                    }
                ],
                "artifacts": [
                    {
                        "path": "review/annotated.mp4",
                        "artifactType": "annotated_video",
                        "category": "VIEWABLE_MEDIA",
                        "access": "PUBLIC",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_planned_runner_loads_structured_results_and_artifacts(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)
    plan = load_pipeline_plan(plan_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    workspace = tmp_path / "workspace"
    (workspace / "people").mkdir(parents=True)
    (workspace / "people" / "intermediate.mp4").write_bytes(b"discard")
    (workspace / "rallies").mkdir(parents=True)
    (workspace / "rallies" / "rallies.json").write_text(
        json.dumps(
            {"rallies": [{"rallyId": "rally-1", "startTimestamp": 1.25, "confidence": 0.8}]}
        ),
        encoding="utf-8",
    )
    (workspace / "review").mkdir()
    (workspace / "review" / "annotated.mp4").write_bytes(b"review")
    stages: list[tuple[ProcessingJobStatus, float]] = []

    async def on_stage(stage: ProcessingJobStatus, progress: float) -> None:
        stages.append((stage, progress))

    result = asyncio.run(
        PlannedCliPipelineRunner(plan, executable="/usr/bin/true").run(
            ProcessingJobRecord(
                job_id="job-1",
                match_id="match-1",
                job_type="analyze_match",
                created_at=NOW,
                updated_at=NOW,
            ),
            source_path=source,
            workspace=workspace,
            on_stage=on_stage,
        )
    )

    assert stages == [(ProcessingJobStatus.PLAYER_PROCESSING, 0.2)]
    rally = result.structured[next(iter(result.structured))][0]
    assert rally.record_id == "rally-1"
    assert rally.timestamp_seconds == 1.25
    assert result.artifacts[0].source_path.name == "annotated.mp4"
    assert result.artifacts[0].access is not None
    assert result.artifacts[0].access.value == "PUBLIC"
    assert not (workspace / "people" / "intermediate.mp4").exists()


def test_pipeline_plan_rejects_arbitrary_commands(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, command="bash")

    with pytest.raises(AnalysisConfigurationError):
        load_pipeline_plan(plan_path)


def test_shot_processing_allows_only_review_overlay_subcommand(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "planVersion": "1",
                "pipelineVersion": "pipeline-test",
                "stages": [
                    {
                        "stage": "SHOT_PROCESSING",
                        "progress": 0.8,
                        "argv": [
                            "shot-model",
                            "apply-review",
                            "{source}",
                            "--shots",
                            "{workspace}/shots/shots.json",
                            "--output",
                            "{workspace}/shots/reviewed-shots.json",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = load_pipeline_plan(plan_path)
    assert plan.stages[0].argv[:2] == ("shot-model", "apply-review")

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["stages"][0]["argv"] = [
        "shot-model",
        "pretrain-representation",
        "--config",
        "experiment.json",
    ]
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AnalysisConfigurationError, match="not training commands"):
        load_pipeline_plan(plan_path)


def test_pipeline_plan_rejects_public_internal_artifact(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["artifacts"][0]["category"] = "INTERNAL_ARTIFACT"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AnalysisConfigurationError, match="PUBLIC access"):
        load_pipeline_plan(plan_path)


def test_pipeline_plan_rejects_cleanup_of_retained_artifact(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["stages"][0]["cleanupPaths"] = ["review/annotated.mp4"]
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AnalysisConfigurationError, match="retained result"):
        load_pipeline_plan(plan_path)


def test_planned_runner_surfaces_cli_stderr_tail(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)
    executable = tmp_path / "failing-command"
    executable.write_text(
        "#!/bin/sh\necho 'portable anchor mismatch' >&2\nexit 2\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    async def on_stage(stage: ProcessingJobStatus, progress: float) -> None:
        del stage, progress

    with pytest.raises(AnalysisPipelineError, match="portable anchor mismatch"):
        asyncio.run(
            PlannedCliPipelineRunner(
                load_pipeline_plan(plan_path),
                executable=str(executable),
            ).run(
                ProcessingJobRecord(
                    job_id="job-failure",
                    match_id="match-failure",
                    job_type="analyze_match",
                    created_at=NOW,
                    updated_at=NOW,
                ),
                source_path=source,
                workspace=tmp_path / "workspace",
                on_stage=on_stage,
            )
        )


def test_planned_runner_maps_profile_mismatch_to_setup_required(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, command="validate-player-profile")
    executable = tmp_path / "failing-command"
    executable.write_text(
        "#!/bin/sh\necho 'error [player_profile_mismatch]: wrong recording' >&2\nexit 2\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    async def on_stage(stage: ProcessingJobStatus, progress: float) -> None:
        del stage, progress

    with pytest.raises(AnalysisSetupRequiredError) as caught:
        asyncio.run(
            PlannedCliPipelineRunner(
                load_pipeline_plan(plan_path),
                executable=str(executable),
            ).run(
                ProcessingJobRecord(
                    job_id="job-setup",
                    match_id="match-setup",
                    job_type="analyze_match",
                    created_at=NOW,
                    updated_at=NOW,
                ),
                source_path=source,
                workspace=tmp_path / "workspace",
                on_stage=on_stage,
            )
        )

    assert caught.value.job_error_code == "ANALYSIS_SETUP_REQUIRED"
