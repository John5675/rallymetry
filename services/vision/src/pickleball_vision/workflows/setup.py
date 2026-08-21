"""Materialize private per-match analysis configuration into temporary storage."""

from __future__ import annotations

from collections.abc import Mapping
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
        return staged
