"""Materialize private per-match analysis configuration into temporary storage."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Protocol

from pickleball_vision.errors import AnalysisConfigurationError
from pickleball_vision.persistence.artifacts import ArtifactStore
from pickleball_vision.persistence.models import (
    ArtifactAccess,
    ArtifactCategory,
    ArtifactProvider,
    Document,
    artifact_record_from_document,
)

SETUP_FILENAMES = {
    "calibrationArtifactId": "calibration.json",
    "playerAssignmentsArtifactId": "player-assignments.json",
    "ballExperimentArtifactId": "ball-experiment.json",
    "ballWeightsArtifactId": "ball-weights.pt",
}


class SetupArtifactLookup(Protocol):
    async def get_artifact(self, artifact_id: str) -> Document | None: ...


class MatchSetupStager:
    """Download only explicitly referenced private setup artifacts for one match."""

    def __init__(
        self,
        persistence: SetupArtifactLookup,
        artifact_store: ArtifactStore,
    ) -> None:
        self._persistence = persistence
        self._artifact_store = artifact_store

    async def stage(
        self,
        *,
        match_id: str,
        match_document: Mapping[str, object],
        workspace: Path,
        source_path: Path | None = None,
    ) -> Mapping[str, Path]:
        raw_setup = match_document.get("analysisSetup")
        if not isinstance(raw_setup, Mapping):
            raise AnalysisConfigurationError("match has no analysisSetup object")
        staged: dict[str, Path] = {}
        profile_match_id = match_document.get("analysisProfileMatchId")
        allowed_owners = {match_id}
        if isinstance(profile_match_id, str) and profile_match_id:
            allowed_owners.add(profile_match_id)
        for field, filename in SETUP_FILENAMES.items():
            artifact_id = raw_setup.get(field)
            if not isinstance(artifact_id, str) or not artifact_id:
                raise AnalysisConfigurationError(f"analysisSetup is missing {field}")
            document = await self._persistence.get_artifact(artifact_id)
            if document is None:
                raise AnalysisConfigurationError(f"analysis setup artifact {field} does not exist")
            artifact = artifact_record_from_document(document)
            if artifact.match_id not in allowed_owners:
                raise AnalysisConfigurationError(f"analysis setup artifact {field} has wrong match")
            if artifact.category is not ArtifactCategory.INTERNAL_ARTIFACT:
                raise AnalysisConfigurationError(f"analysis setup artifact {field} is not internal")
            if artifact.provider is not ArtifactProvider.VERCEL_BLOB:
                raise AnalysisConfigurationError(f"analysis setup artifact {field} is not hosted")
            if artifact.access is not ArtifactAccess.PRIVATE:
                raise AnalysisConfigurationError(f"analysis setup artifact {field} is not private")
            destination = workspace / "input" / filename
            staged[field] = await self._artifact_store.get(artifact, destination)
        if source_path is not None:
            _bind_calibration_to_runtime_source(
                staged["calibrationArtifactId"],
                source_path=source_path,
            )
        names = _player_names_from_match(match_document)
        if names is not None:
            names_path = workspace / "input" / "player-names.json"
            _write_json_atomic(names_path, names)
        return staged


def _player_names_from_match(
    match_document: Mapping[str, object],
) -> dict[str, str] | None:
    """Return only explicit reviewed role/name bindings, never title-order guesses."""

    summary = match_document.get("summary")
    if not isinstance(summary, Mapping):
        return None
    raw = summary.get("playerRoleAssignments")
    if not isinstance(raw, Mapping):
        return None
    roles = ("ME", "PARTNER", "OPPONENT_1", "OPPONENT_2")
    names: dict[str, str] = {}
    for role in roles:
        value = raw.get(role)
        if not isinstance(value, str) or not value.strip():
            raise AnalysisConfigurationError(
                "summary.playerRoleAssignments must contain four non-empty reviewed names"
            )
        names[role] = value.strip()
    return names


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise AnalysisConfigurationError(
            f"unable to stage reviewed player names: {error}"
        ) from error


def _bind_calibration_to_runtime_source(path: Path, *, source_path: Path) -> None:
    """Rebind only the temporary calibration copy to the staged source pathname."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("root must be an object")
        source = payload.get("source")
        if not isinstance(source, dict):
            raise ValueError("source must be an object")
        original_path = source.get("video_path")
        if not isinstance(original_path, str) or not original_path:
            raise ValueError("source.video_path must be a non-empty string")
        runtime_path = str(source_path.expanduser().resolve())
        source["video_path"] = runtime_path
        payload["runtime_binding"] = {
            "mode": "temporary_hosted_copy",
            "original_video_path": original_path,
            "runtime_video_path": runtime_path,
            "original_artifact_modified": False,
        }
        temporary = path.with_name(f".{path.name}.runtime.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        with suppress(OSError):
            path.with_name(f".{path.name}.runtime.tmp").unlink(missing_ok=True)
        raise AnalysisConfigurationError(
            f"unable to bind staged calibration to source media: {error}"
        ) from error
