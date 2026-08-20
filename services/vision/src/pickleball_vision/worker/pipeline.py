"""Trusted CLI-plan adapter over the existing local CV/audio pipeline."""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import TypeAlias

from pickleball_vision.errors import WorkerConfigurationError, WorkerPipelineError
from pickleball_vision.persistence.models import (
    AnalyticsRecord,
    ArtifactAccess,
    ArtifactCategory,
    PlayerRecord,
    ProcessingJobRecord,
    ProcessingJobStatus,
    StructuredCollection,
    StructuredDomainRecord,
)
from pickleball_vision.worker.models import (
    ArtifactPublication,
    ArtifactResultPlan,
    PipelinePlan,
    PipelineRunResult,
    PipelineStagePlan,
    StructuredResultPlan,
)

StageCallback: TypeAlias = Callable[[ProcessingJobStatus, float], Awaitable[None]]

_STAGE_ORDER = {
    ProcessingJobStatus.PLAYER_PROCESSING: 0,
    ProcessingJobStatus.BALL_PROCESSING: 1,
    ProcessingJobStatus.AUDIO_PROCESSING: 2,
    ProcessingJobStatus.RALLY_PROCESSING: 3,
    ProcessingJobStatus.EVENT_PROCESSING: 4,
    ProcessingJobStatus.SHOT_PROCESSING: 5,
    ProcessingJobStatus.ANALYTICS: 6,
}
_ALLOWED_COMMANDS = {
    ProcessingJobStatus.PLAYER_PROCESSING: {
        "detect-people",
        "isolate-players",
        "track-players",
        "analyze-players",
    },
    ProcessingJobStatus.BALL_PROCESSING: {"ball", "track-ball"},
    ProcessingJobStatus.AUDIO_PROCESSING: {"analyze-audio"},
    ProcessingJobStatus.RALLY_PROCESSING: {"segment-rallies"},
    ProcessingJobStatus.EVENT_PROCESSING: {
        "detect-bounces",
        "detect-contacts",
        "identify-hitters",
    },
    ProcessingJobStatus.SHOT_PROCESSING: {"reconstruct-shots"},
    ProcessingJobStatus.ANALYTICS: {"analyze-match"},
}


class PipelineRunner:
    """Structural base for production and test pipeline runners."""

    async def run(
        self,
        job: ProcessingJobRecord,
        *,
        source_path: Path,
        workspace: Path,
        on_stage: StageCallback,
    ) -> PipelineRunResult:
        raise NotImplementedError


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise WorkerConfigurationError(f"{context} must be an object")
    return value


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerConfigurationError(f"{context} must be a non-empty string")
    return value.strip()


def _relative_path(value: object, context: str) -> Path:
    path = Path(_text(value, context))
    if path.is_absolute() or ".." in path.parts:
        raise WorkerConfigurationError(f"{context} must be a safe workspace-relative path")
    return path


def _optional_text(value: object, context: str) -> str | None:
    if value is None:
        return None
    return _text(value, context)


def load_pipeline_plan(path: Path) -> PipelinePlan:
    """Load an operator-controlled plan; queued job documents never contain commands."""

    resolved = path.expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerConfigurationError(
            f"unable to load pipeline plan {resolved}: {error}"
        ) from error
    root = _mapping(payload, "pipeline plan")
    plan_version = _text(root.get("planVersion"), "planVersion")
    pipeline_version = _text(root.get("pipelineVersion"), "pipelineVersion")
    raw_stages = root.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise WorkerConfigurationError("stages must be a non-empty array")
    stages: list[PipelineStagePlan] = []
    previous_order = -1
    previous_progress = 0.0
    for index, raw_stage in enumerate(raw_stages):
        item = _mapping(raw_stage, f"stages[{index}]")
        try:
            stage = ProcessingJobStatus(_text(item.get("stage"), f"stages[{index}].stage"))
        except ValueError as error:
            raise WorkerConfigurationError(f"stages[{index}].stage is unsupported") from error
        if stage not in _STAGE_ORDER:
            raise WorkerConfigurationError(f"{stage.value} is not an executable pipeline stage")
        raw_progress = item.get("progress")
        if not isinstance(raw_progress, (int, float)) or isinstance(raw_progress, bool):
            raise WorkerConfigurationError(f"stages[{index}].progress must be numeric")
        progress = float(raw_progress)
        order = _STAGE_ORDER[stage]
        if order < previous_order:
            raise WorkerConfigurationError("pipeline stages must follow domain processing order")
        if not previous_progress < progress < 0.9:
            raise WorkerConfigurationError(
                "stage progress must increase strictly and remain below 0.9"
            )
        raw_argv = item.get("argv")
        if not isinstance(raw_argv, list) or not raw_argv:
            raise WorkerConfigurationError(f"stages[{index}].argv must be a non-empty array")
        argv = tuple(_text(value, f"stages[{index}].argv") for value in raw_argv)
        if argv[0] not in _ALLOWED_COMMANDS[stage]:
            raise WorkerConfigurationError(
                f"command {argv[0]!r} is not allowed during {stage.value}"
            )
        if argv[0] == "ball" and (len(argv) < 2 or argv[1] != "detect"):
            raise WorkerConfigurationError(
                "BALL_PROCESSING permits `ball detect` inference but not training commands"
            )
        stages.append(PipelineStagePlan(stage=stage, progress=progress, argv=argv))
        previous_order = order
        previous_progress = progress

    structured_results = _load_structured_result_plans(root.get("structuredResults", []))
    artifacts = _load_artifact_result_plans(root.get("artifacts", []))
    return PipelinePlan(
        plan_version=plan_version,
        pipeline_version=pipeline_version,
        stages=tuple(stages),
        structured_results=structured_results,
        artifacts=artifacts,
    )


def _load_structured_result_plans(value: object) -> tuple[StructuredResultPlan, ...]:
    if not isinstance(value, list):
        raise WorkerConfigurationError("structuredResults must be an array")
    allowed = {"players", "analytics", *(collection.value for collection in StructuredCollection)}
    plans: list[StructuredResultPlan] = []
    for index, raw in enumerate(value):
        item = _mapping(raw, f"structuredResults[{index}]")
        collection = _text(item.get("collection"), f"structuredResults[{index}].collection")
        if collection not in allowed:
            raise WorkerConfigurationError(f"unsupported structured collection {collection!r}")
        id_field = _optional_text(item.get("idField"), f"structuredResults[{index}].idField")
        if collection not in {"analytics"} and id_field is None:
            raise WorkerConfigurationError(f"{collection} structured results require idField")
        plans.append(
            StructuredResultPlan(
                collection=collection,
                path=_relative_path(item.get("path"), f"structuredResults[{index}].path"),
                records_key=_optional_text(
                    item.get("recordsKey"),
                    f"structuredResults[{index}].recordsKey",
                ),
                id_field=id_field,
                timestamp_field=_optional_text(
                    item.get("timestampField"),
                    f"structuredResults[{index}].timestampField",
                ),
                confidence_field=_optional_text(
                    item.get("confidenceField"),
                    f"structuredResults[{index}].confidenceField",
                ),
            )
        )
    return tuple(plans)


def _load_artifact_result_plans(value: object) -> tuple[ArtifactResultPlan, ...]:
    if not isinstance(value, list):
        raise WorkerConfigurationError("artifacts must be an array")
    plans: list[ArtifactResultPlan] = []
    for index, raw in enumerate(value):
        item = _mapping(raw, f"artifacts[{index}]")
        try:
            category = ArtifactCategory(_text(item.get("category"), f"artifacts[{index}].category"))
            access_raw = item.get("access")
            access = (
                ArtifactAccess(_text(access_raw, f"artifacts[{index}].access"))
                if access_raw is not None
                else None
            )
        except ValueError as error:
            raise WorkerConfigurationError(f"artifacts[{index}] has an invalid policy") from error
        if access is ArtifactAccess.PUBLIC and category is not ArtifactCategory.VIEWABLE_MEDIA:
            raise WorkerConfigurationError(
                f"artifacts[{index}] may use PUBLIC access only with VIEWABLE_MEDIA"
            )
        required = item.get("required", True)
        if not isinstance(required, bool):
            raise WorkerConfigurationError(f"artifacts[{index}].required must be boolean")
        plans.append(
            ArtifactResultPlan(
                path=_relative_path(item.get("path"), f"artifacts[{index}].path"),
                artifact_type=_text(
                    item.get("artifactType"),
                    f"artifacts[{index}].artifactType",
                ),
                category=category,
                access=access,
                required=required,
            )
        )
    return tuple(plans)


class PlannedCliPipelineRunner(PipelineRunner):
    """Run only `pickleball-vision` subcommands from a trusted local JSON plan."""

    def __init__(self, plan: PipelinePlan, *, executable: str | None = None) -> None:
        self.plan = plan
        self._executable = executable

    async def run(
        self,
        job: ProcessingJobRecord,
        *,
        source_path: Path,
        workspace: Path,
        on_stage: StageCallback,
    ) -> PipelineRunResult:
        workspace.mkdir(parents=True, exist_ok=True)
        executable = self._executable or shutil.which("pickleball-vision")
        if executable is None:
            raise WorkerPipelineError("pickleball-vision executable is unavailable")
        log_dir = workspace / "worker-logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        substitutions = {
            "{source}": str(source_path.resolve()),
            "{workspace}": str(workspace.resolve()),
            "{matchId}": job.match_id,
        }
        for index, stage_plan in enumerate(self.plan.stages):
            await on_stage(stage_plan.stage, stage_plan.progress)
            argv = tuple(self._substitute(argument, substitutions) for argument in stage_plan.argv)
            stdout_path = log_dir / f"{index:02d}-{stage_plan.stage.value}.stdout.log"
            stderr_path = log_dir / f"{index:02d}-{stage_plan.stage.value}.stderr.log"
            try:
                with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                    process = await asyncio.create_subprocess_exec(
                        executable,
                        *argv,
                        cwd=workspace,
                        stdout=stdout,
                        stderr=stderr,
                    )
                    return_code = await process.wait()
            except OSError as error:
                raise WorkerPipelineError(
                    f"unable to launch local CLI: {type(error).__name__}",
                    stage=stage_plan.stage.value,
                ) from error
            if return_code != 0:
                raise WorkerPipelineError(
                    f"local CLI command {argv[0]} exited with status {return_code}",
                    stage=stage_plan.stage.value,
                )
        return self._load_results(job, workspace)

    @staticmethod
    def _substitute(argument: str, substitutions: Mapping[str, str]) -> str:
        expanded = argument
        for placeholder, value in substitutions.items():
            expanded = expanded.replace(placeholder, value)
        return expanded

    def _load_results(self, job: ProcessingJobRecord, workspace: Path) -> PipelineRunResult:
        players: list[PlayerRecord] = []
        structured: dict[StructuredCollection, tuple[StructuredDomainRecord, ...]] = {}
        analytics: AnalyticsRecord | None = None
        for spec in self.plan.structured_results:
            payload = self._load_json_result(workspace, spec.path)
            records = self._extract_records(payload, spec)
            if spec.collection == "players":
                players.extend(self._player_record(job, record, spec) for record in records)
            elif spec.collection == "analytics":
                if len(records) != 1:
                    raise WorkerPipelineError("analytics result must contain exactly one object")
                analytics = self._analytics_record(job, records[0])
            else:
                collection = StructuredCollection(spec.collection)
                structured[collection] = tuple(
                    self._domain_record(job, record, spec) for record in records
                )
        artifacts: list[ArtifactPublication] = []
        for artifact_spec in self.plan.artifacts:
            source = (workspace / artifact_spec.path).resolve()
            if not source.is_relative_to(workspace.resolve()):
                raise WorkerPipelineError("artifact result path escaped the worker workspace")
            if not source.is_file():
                if artifact_spec.required:
                    raise WorkerPipelineError(
                        f"required artifact was not produced: {artifact_spec.path}"
                    )
                continue
            artifacts.append(
                ArtifactPublication(
                    source_path=source,
                    artifact_type=artifact_spec.artifact_type,
                    category=artifact_spec.category,
                    access=artifact_spec.access,
                )
            )
        return PipelineRunResult(
            players=tuple(players),
            structured=structured,
            analytics=analytics,
            artifacts=tuple(artifacts),
        )

    @staticmethod
    def _load_json_result(workspace: Path, relative_path: Path) -> object:
        path = (workspace / relative_path).resolve()
        if not path.is_relative_to(workspace.resolve()):
            raise WorkerPipelineError("structured result path escaped the worker workspace")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise WorkerPipelineError(
                f"unable to load structured result {relative_path}"
            ) from error

    @staticmethod
    def _extract_records(
        payload: object,
        spec: StructuredResultPlan,
    ) -> tuple[Mapping[str, object], ...]:
        selected = payload
        if spec.records_key is not None:
            root = _mapping(payload, f"result {spec.path}")
            selected = root.get(spec.records_key)
        if spec.collection == "analytics" and isinstance(selected, Mapping):
            return (_mapping(selected, f"result {spec.path}"),)
        if not isinstance(selected, list):
            raise WorkerPipelineError(f"result {spec.path} must contain a record array")
        return tuple(_mapping(item, f"record in {spec.path}") for item in selected)

    def _domain_record(
        self,
        job: ProcessingJobRecord,
        payload: Mapping[str, object],
        spec: StructuredResultPlan,
    ) -> StructuredDomainRecord:
        record_id = self._field_text(payload, spec.id_field, spec.path)
        return StructuredDomainRecord(
            match_id=job.match_id,
            record_id=record_id,
            payload=dict(payload),
            confidence=self._optional_number(payload, spec.confidence_field, spec.path),
            timestamp_seconds=self._optional_number(payload, spec.timestamp_field, spec.path),
            pipeline_version=self.plan.pipeline_version,
        )

    def _player_record(
        self,
        job: ProcessingJobRecord,
        payload: Mapping[str, object],
        spec: StructuredResultPlan,
    ) -> PlayerRecord:
        player_id = self._field_text(payload, spec.id_field, spec.path)
        return PlayerRecord(
            match_id=job.match_id,
            player_id=player_id,
            display_name=self._optional_payload_text(payload, "displayName"),
            logical_identity=self._optional_payload_text(payload, "logicalIdentity"),
            team=self._optional_payload_text(payload, "team"),
            metadata=dict(payload),
        )

    def _analytics_record(
        self,
        job: ProcessingJobRecord,
        payload: Mapping[str, object],
    ) -> AnalyticsRecord:
        metrics_value = payload.get("metrics", payload)
        if not isinstance(metrics_value, Mapping):
            raise WorkerPipelineError("analytics result must be an object")
        analytics_id = self._optional_payload_text(payload, "analyticsId") or "match-analytics"
        calculation_version = (
            self._optional_payload_text(payload, "calculationVersion")
            or self._optional_payload_text(payload, "analyticsVersion")
            or self.plan.pipeline_version
        )
        return AnalyticsRecord(
            match_id=job.match_id,
            analytics_id=analytics_id,
            calculation_version=calculation_version,
            metrics=dict(metrics_value),
            pipeline_version=self.plan.pipeline_version,
        )

    @staticmethod
    def _field_text(
        payload: Mapping[str, object],
        field: str | None,
        path: Path,
    ) -> str:
        if field is None:
            raise WorkerPipelineError(f"result {path} is missing an ID-field configuration")
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise WorkerPipelineError(f"result {path} contains an invalid {field}")
        return value

    @staticmethod
    def _optional_payload_text(payload: Mapping[str, object], field: str) -> str | None:
        value = payload.get(field)
        return value if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _optional_number(
        payload: Mapping[str, object],
        field: str | None,
        path: Path,
    ) -> float | None:
        if field is None:
            return None
        value = payload.get(field)
        if value is None:
            return None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise WorkerPipelineError(f"result {path} contains non-numeric {field}")
        return float(value)
