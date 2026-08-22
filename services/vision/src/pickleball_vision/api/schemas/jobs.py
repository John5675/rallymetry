"""On-demand workflow processing-job API schemas."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, NamedTuple

from pydantic import Field, model_validator

from pickleball_vision.api.schemas.common import ApiOutputModel, Identifier


class _ProgressDetail(NamedTuple):
    stage: str
    progress: float
    key: str
    label: str
    description: str
    step_index: int | None = None
    total_steps: int | None = None


_ANALYSIS_STEP_COUNT = 14
_PROGRESS_DETAILS = (
    _ProgressDetail(
        "CREATED",
        0.0,
        "create-job",
        "Creating the analysis job",
        "Saving the match and its analysis configuration.",
    ),
    _ProgressDetail(
        "QUEUED",
        0.0,
        "wait-for-worker",
        "Waiting for the local worker",
        "The match is safely queued and will start when the analysis Mac is available.",
    ),
    _ProgressDetail(
        "STARTING",
        0.01,
        "start-worker",
        "Starting the analysis run",
        "The worker claimed this match and is preparing an isolated workspace.",
    ),
    _ProgressDetail(
        "DOWNLOADING_MEDIA",
        0.05,
        "download-media",
        "Downloading the source video",
        "The worker is retrieving the YouTube recording through its local connection.",
    ),
    _ProgressDetail(
        "PREPARING_MEDIA",
        0.10,
        "prepare-inputs",
        "Preparing court and model inputs",
        "Calibration, player assignments, and model files are being staged.",
    ),
    _ProgressDetail(
        "PLAYER_PROCESSING",
        0.12,
        "validate-player-profile",
        "Checking recording setup",
        "Confirming that the reviewed court view and player anchors match this recording.",
        1,
        _ANALYSIS_STEP_COUNT,
    ),
    _ProgressDetail(
        "PLAYER_PROCESSING",
        0.15,
        "detect-people",
        "Detecting people",
        "Scanning every video frame for visible people before selecting match players.",
        2,
        _ANALYSIS_STEP_COUNT,
    ),
    _ProgressDetail(
        "PLAYER_PROCESSING",
        0.17,
        "track-players",
        "Tracking the four match players",
        "Linking the four assigned player identities across frames and short occlusions.",
        3,
        _ANALYSIS_STEP_COUNT,
    ),
    _ProgressDetail(
        "PLAYER_PROCESSING",
        0.20,
        "analyze-players",
        "Mapping player movement",
        "Transforming ground-contact positions into court movement and occupancy data.",
        4,
        _ANALYSIS_STEP_COUNT,
    ),
    _ProgressDetail(
        "BALL_PROCESSING",
        0.30,
        "detect-ball",
        "Detecting the pickleball",
        "Running the tiny-object detector across the primary-court view.",
        5,
        _ANALYSIS_STEP_COUNT,
    ),
    _ProgressDetail(
        "BALL_PROCESSING",
        0.40,
        "track-ball",
        "Reconstructing the ball trajectory",
        "Associating frame-level candidates while preserving observed, interpolated, "
        "and unknown periods.",
        6,
        _ANALYSIS_STEP_COUNT,
    ),
    _ProgressDetail(
        "AUDIO_PROCESSING",
        0.55,
        "analyze-audio",
        "Analyzing synchronized audio",
        "Extracting optional transient evidence without creating events from sound alone.",
        7,
        _ANALYSIS_STEP_COUNT,
    ),
    _ProgressDetail(
        "RALLY_PROCESSING",
        0.65,
        "segment-rallies",
        "Segmenting rallies",
        "Combining structured motion and reset signals into candidate rally boundaries.",
        8,
        _ANALYSIS_STEP_COUNT,
    ),
    _ProgressDetail(
        "BOUNCE_PROCESSING",
        0.70,
        "detect-bounces",
        "Detecting bounces",
        "Evaluating visual trajectory reversals with optional synchronized audio support.",
        9,
        _ANALYSIS_STEP_COUNT,
    ),
    _ProgressDetail(
        "CONTACT_PROCESSING",
        0.75,
        "detect-contacts",
        "Detecting paddle contacts",
        "Evaluating ball-direction changes near tracked players.",
        10,
        _ANALYSIS_STEP_COUNT,
    ),
    _ProgressDetail(
        "HITTER_PROCESSING",
        0.80,
        "identify-hitters",
        "Identifying hitters",
        "Resolving each contact to the most defensible logical player or UNKNOWN.",
        11,
        _ANALYSIS_STEP_COUNT,
    ),
    _ProgressDetail(
        "SHOT_PROCESSING",
        0.85,
        "reconstruct-shots",
        "Reconstructing shots",
        "Connecting contacts, trajectories, bounces, hitters, and landing evidence.",
        12,
        _ANALYSIS_STEP_COUNT,
    ),
    _ProgressDetail(
        "SHOT_PROCESSING",
        0.87,
        "apply-shot-review",
        "Applying reviewed shot labels",
        "Applying available reviewed semantics while retaining the original predictions.",
        13,
        _ANALYSIS_STEP_COUNT,
    ),
    _ProgressDetail(
        "ANALYTICS",
        0.90,
        "analyze-match",
        "Calculating match analytics",
        "Producing deterministic metrics from the structured match records.",
        14,
        _ANALYSIS_STEP_COUNT,
    ),
    _ProgressDetail(
        "RENDERING_ARTIFACTS",
        0.94,
        "render-artifacts",
        "Preparing viewable results",
        "Final videos, heatmaps, and summaries are being assembled.",
    ),
    _ProgressDetail(
        "UPLOADING_RESULTS",
        0.97,
        "upload-results",
        "Publishing results",
        "Structured results are being saved and friend-viewable artifacts uploaded.",
    ),
)


def _progress_detail(stage: object, progress: object) -> _ProgressDetail | None:
    if not isinstance(stage, str):
        return None
    numeric_progress = (
        float(progress)
        if isinstance(progress, (int, float)) and not isinstance(progress, bool)
        else 0.0
    )
    candidates = [detail for detail in _PROGRESS_DETAILS if detail.stage == stage]
    if not candidates:
        return None
    eligible = [detail for detail in candidates if detail.progress <= numeric_progress + 1e-9]
    return max(eligible or candidates, key=lambda detail: detail.progress)


class JobResponse(ApiOutputModel):
    job_id: Identifier
    match_id: Identifier
    job_type: str
    status: str
    progress: float = Field(ge=0, le=1)
    stage: str | None = None
    current_step: str | None = None
    current_step_label: str | None = None
    current_step_description: str | None = None
    current_step_index: int | None = Field(default=None, ge=1)
    total_steps: int | None = Field(default=None, ge=1)
    render_triggered_at: datetime | None = None
    claimed_at: datetime | None = None
    heartbeat_at: datetime | None = None
    lease_expires_at: datetime | None = None
    worker_id: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    failed_stage: str | None = None
    render_task_run_id: Identifier | None = None
    processing_run_id: Identifier | None = None
    attempt_count: int = Field(ge=0)
    error_code: str | None = None
    error_message: str | None = None
    pipeline_version: str | None = None
    source_type: str | None = None
    source_artifact_id: Identifier | None = None
    youtube_video_id: str | None = None
    result_artifact_ids: list[Identifier] = Field(default_factory=list)
    result_summary: dict[str, object] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="before")
    @classmethod
    def add_progress_detail(cls, value: Any) -> Any:
        """Add stable presentation metadata for old and new worker job documents."""

        if not isinstance(value, Mapping):
            return value
        enriched = dict(value)
        if (
            enriched.get("currentStepLabel") is not None
            or enriched.get("current_step_label") is not None
        ):
            return enriched
        stage = enriched.get("stage")
        detail = _progress_detail(stage, enriched.get("progress"))
        if detail is None:
            return enriched
        enriched.update(
            {
                "currentStep": detail.key,
                "currentStepLabel": detail.label,
                "currentStepDescription": detail.description,
                "currentStepIndex": detail.step_index,
                "totalSteps": detail.total_steps,
            }
        )
        return enriched
