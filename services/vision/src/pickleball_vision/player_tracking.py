"""Raw multi-object tracks and conservative logical-player identity resolution."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise
from typing import Protocol

from pickleball_vision.config import PlayerTrackingSettings
from pickleball_vision.court import CourtDimensions
from pickleball_vision.person_detection import BoundingBox, PersonDetection
from pickleball_vision.player_isolation import (
    LOGICAL_PLAYER_ROLES,
    CourtRegionState,
    CourtSide,
    GroundContactAssessment,
    LogicalPlayerAssignments,
    LogicalPlayerRole,
)
from pickleball_vision.video import VideoMetadata

PLAYER_TRACKING_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class IndexedDetection:
    """One immutable raw detection plus its index in detections.json."""

    raw_detection_index: int
    detection: PersonDetection


@dataclass(frozen=True, slots=True)
class TrackerMetadata:
    """Effective established-tracker implementation and configuration."""

    adapter: str
    implementation: str
    framework: str
    framework_version: str | None
    configuration: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "adapter": self.adapter,
            "implementation": self.implementation,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "configuration": self.configuration,
        }


@dataclass(frozen=True, slots=True)
class RawTrackerObservation:
    """Transient tracker evidence linked back to one raw person detection."""

    observation_id: str
    tracker_id: int
    raw_detection_index: int
    frame_number: int
    timestamp_s: float
    tracker_bounding_box: BoundingBox
    detection_confidence: float
    tracker_confidence: float

    def as_dict(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "tracker_id": self.tracker_id,
            "raw_person_detection": {"index": self.raw_detection_index},
            "frame_number": self.frame_number,
            "timestamp_s": self.timestamp_s,
            "tracker_bounding_box": self.tracker_bounding_box.as_dict(),
            "detection_confidence": self.detection_confidence,
            "tracker_confidence": self.tracker_confidence,
        }


class MultiObjectTracker(Protocol):
    """Model-neutral boundary around transient multi-object association."""

    @property
    def metadata(self) -> TrackerMetadata: ...

    def update(
        self,
        *,
        frame_number: int,
        timestamp_s: float,
        detections: tuple[IndexedDetection, ...],
        frame_width_px: int,
        frame_height_px: int,
    ) -> tuple[RawTrackerObservation, ...]: ...


@dataclass(frozen=True, slots=True)
class TrackerGroundObservation:
    """Court geometry derived from a raw tracker-linked person observation."""

    raw: RawTrackerObservation
    ground_contact: GroundContactAssessment


class LogicalTrackingState(StrEnum):
    """Per-frame state of one human-owned logical player identity."""

    OBSERVED = "observed"
    REACQUIRED = "reacquired"
    TEMPORARILY_MISSING = "temporarily_missing"
    SUSPECTED_IDENTITY_SWITCH = "suspected_identity_switch"


@dataclass(frozen=True, slots=True)
class LogicalPlayerFrame:
    """One role's resolved state in one source-video frame."""

    logical_player: LogicalPlayerRole
    frame_number: int
    timestamp_s: float
    state: LogicalTrackingState
    confidence: float
    tracker_observation: TrackerGroundObservation | None
    resolution_method: str
    appearance_similarity: float | None = None
    appearance_margin: float | None = None

    def as_dict(self) -> dict[str, object]:
        observation = self.tracker_observation
        return {
            "frame_number": self.frame_number,
            "timestamp_s": self.timestamp_s,
            "tracking_state": self.state.value,
            "tracking_confidence": self.confidence,
            "tracker_id": observation.raw.tracker_id if observation is not None else None,
            "raw_tracker_observation_id": (
                observation.raw.observation_id if observation is not None else None
            ),
            "ground_contact": (
                observation.ground_contact.as_dict() if observation is not None else None
            ),
            "identity_resolution": {
                "method": self.resolution_method,
                "logical_identity_independent_of_tracker_id": True,
                "appearance_similarity": self.appearance_similarity,
                "same_side_appearance_margin": self.appearance_margin,
            },
        }


@dataclass(frozen=True, slots=True)
class SuspectedIdentitySwitch:
    """Explicit review event for a questionable transient-ID transition."""

    logical_player: LogicalPlayerRole
    frame_number: int
    timestamp_s: float
    previous_tracker_id: int
    current_tracker_id: int
    confidence: float
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "logical_player": self.logical_player.value,
            "frame_number": self.frame_number,
            "timestamp_s": self.timestamp_s,
            "previous_tracker_id": self.previous_tracker_id,
            "current_tracker_id": self.current_tracker_id,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class PlayerTrackingRun:
    """Versioned derived tracking artifact with raw and logical layers."""

    source: VideoMetadata
    detections_path: str
    candidates_path: str
    assignments_path: str
    calibration_path: str
    tracker: TrackerMetadata
    configuration: dict[str, object]
    appearance: dict[str, object]
    player_names: dict[LogicalPlayerRole, str]
    raw_tracker_observations: tuple[RawTrackerObservation, ...]
    logical_tracks: dict[LogicalPlayerRole, tuple[LogicalPlayerFrame, ...]]
    suspected_identity_switches: tuple[SuspectedIdentitySwitch, ...]
    created_at_utc: str
    schema_version: int = PLAYER_TRACKING_SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "record_type": "persistent_logical_player_tracks",
            "source": self.source.as_dict(),
            "inputs": {
                "raw_person_detections": self.detections_path,
                "primary_player_candidates": self.candidates_path,
                "logical_player_assignments": self.assignments_path,
                "court_calibration": self.calibration_path,
            },
            "identity_contract": ("logical_player_identity_is_independent_of_transient_tracker_id"),
            "tracker": self.tracker.as_dict(),
            "configuration": self.configuration,
            "appearance": self.appearance,
            "player_names": {role.value: self.player_names[role] for role in LOGICAL_PLAYER_ROLES},
            "raw_tracker_layer": {
                "record_type": "raw_transient_tracker_observations",
                "observations": [item.as_dict() for item in self.raw_tracker_observations],
            },
            "logical_identity_layer": {
                role.value: [frame.as_dict() for frame in self.logical_tracks[role]]
                for role in LOGICAL_PLAYER_ROLES
            },
            "suspected_identity_switches": [
                event.as_dict() for event in self.suspected_identity_switches
            ],
        }


@dataclass(frozen=True, slots=True)
class _ResolvedObservation:
    observation: TrackerGroundObservation
    confidence: float
    state: LogicalTrackingState
    method: str
    appearance_similarity: float | None = None
    appearance_margin: float | None = None


def _distance_m(
    first: GroundContactAssessment,
    second: GroundContactAssessment,
) -> float | None:
    if first.court_point is None or second.court_point is None:
        return None
    return math.hypot(
        first.court_point.x_m - second.court_point.x_m,
        first.court_point.y_m - second.court_point.y_m,
    )


def _identity_score(
    candidate: TrackerGroundObservation,
    previous: TrackerGroundObservation,
    *,
    elapsed_s: float,
    expected_side: CourtSide,
    settings: PlayerTrackingSettings,
    appearance_similarity: float | None,
    appearance_margin: float | None,
) -> float | None:
    ground = candidate.ground_contact
    previous_ground = previous.ground_contact
    same_tracker = candidate.raw.tracker_id == previous.raw.tracker_id
    if elapsed_s <= 0:
        return None
    long_gap = elapsed_s > settings.max_identity_gap_seconds
    if long_gap and not (
        appearance_similarity is not None
        and appearance_similarity >= settings.long_gap_appearance_similarity
        and appearance_margin is not None
        and appearance_margin >= settings.long_gap_minimum_appearance_margin
    ):
        return None
    court = CourtDimensions()
    longitudinal_extension = (
        ground.court_point is not None
        and 0 <= ground.court_point.x_m <= court.width_m
        and (ground.court_point.y_m < 0 or ground.court_point.y_m > court.length_m)
    )
    if ground.region_state is CourtRegionState.OUTSIDE and not (
        same_tracker and longitudinal_extension
    ):
        return None
    if ground.region_state is CourtRegionState.AMBIGUOUS and not same_tracker:
        return None
    if (
        expected_side is not CourtSide.AMBIGUOUS
        and ground.side is not CourtSide.AMBIGUOUS
        and ground.side is not expected_side
    ):
        return None
    if appearance_margin is not None and appearance_margin < settings.minimum_appearance_margin:
        return None
    if (
        not same_tracker
        and appearance_similarity is not None
        and appearance_similarity < settings.minimum_appearance_similarity
    ):
        return None

    distance_m = _distance_m(ground, previous_ground)
    if distance_m is None:
        if not same_tracker:
            return None
        motion_score = 0.35
    else:
        movement_limit_m = 0.50 + settings.max_player_speed_mps * elapsed_s
        if distance_m > movement_limit_m:
            return None
        motion_score = max(0.0, 1.0 - distance_m / movement_limit_m)

    detection_score = candidate.raw.detection_confidence
    region_score = (
        1.0
        if ground.region_state is CourtRegionState.INSIDE
        else 0.65
        if ground.region_state is CourtRegionState.NEAR
        else 0.25
    )
    side_score = 1.0 if ground.side is expected_side else 0.45
    gap_score = max(0.0, 1.0 - elapsed_s / settings.max_identity_gap_seconds)
    if same_tracker:
        score = (
            0.55
            + 0.15 * detection_score
            + 0.10 * region_score
            + 0.05 * side_score
            + 0.15 * motion_score
        )
    else:
        score = (
            0.18
            + 0.12 * detection_score
            + 0.15 * region_score
            + 0.10 * side_score
            + 0.30 * motion_score
            + 0.15 * gap_score
        )
    if appearance_similarity is not None:
        normalized_margin = (
            0.5
            if appearance_margin is None
            else min(1.0, max(0.0, (appearance_margin + 1.0) / 2.0))
        )
        appearance_score = 0.75 * appearance_similarity + 0.25 * normalized_margin
        score = (
            1.0 - settings.appearance_weight
        ) * score + settings.appearance_weight * appearance_score
    return min(1.0, score)


def _appearance_evidence(
    observation_id: str,
    *,
    role: LogicalPlayerRole,
    same_side_roles: tuple[LogicalPlayerRole, ...],
    similarities: dict[LogicalPlayerRole, dict[str, float]],
) -> tuple[float | None, float | None]:
    own_similarity = similarities.get(role, {}).get(observation_id)
    if own_similarity is None:
        return (None, None)
    competitor_similarities = [
        similarities.get(other, {}).get(observation_id)
        for other in same_side_roles
        if other is not role
    ]
    available = [value for value in competitor_similarities if value is not None]
    margin = own_similarity - max(available) if available else None
    return (own_similarity, margin)


def _resolve_role_direction(
    *,
    anchor: TrackerGroundObservation,
    expected_side: CourtSide,
    frame_numbers: range,
    observations_by_frame: dict[int, tuple[TrackerGroundObservation, ...]],
    settings: PlayerTrackingSettings,
    role: LogicalPlayerRole,
    same_side_roles: tuple[LogicalPlayerRole, ...],
    appearance_similarities: dict[LogicalPlayerRole, dict[str, float]],
) -> dict[int, _ResolvedObservation]:
    resolved: dict[int, _ResolvedObservation] = {}
    previous = anchor
    previous_frame = anchor.raw.frame_number
    for frame_number in frame_numbers:
        candidates: list[tuple[float, TrackerGroundObservation, float | None, float | None]] = []
        for candidate in observations_by_frame.get(frame_number, ()):
            candidate_elapsed_s = abs(candidate.raw.timestamp_s - previous.raw.timestamp_s)
            similarity, margin = _appearance_evidence(
                candidate.raw.observation_id,
                role=role,
                same_side_roles=same_side_roles,
                similarities=appearance_similarities,
            )
            score = _identity_score(
                candidate,
                previous,
                elapsed_s=candidate_elapsed_s,
                expected_side=expected_side,
                settings=settings,
                appearance_similarity=similarity,
                appearance_margin=margin,
            )
            if score is not None and score >= settings.minimum_identity_score:
                candidates.append((score, candidate, similarity, margin))
        if not candidates:
            continue
        score, selected, similarity, margin = max(candidates, key=lambda item: item[0])
        tracker_changed = selected.raw.tracker_id != previous.raw.tracker_id
        had_missing_frame = abs(frame_number - previous_frame) > 1
        if tracker_changed and (not had_missing_frame or score < settings.suspected_switch_score):
            state = LogicalTrackingState.SUSPECTED_IDENTITY_SWITCH
            method = "court_motion_tracker_id_change_requires_review"
        elif tracker_changed or had_missing_frame:
            state = LogicalTrackingState.REACQUIRED
            method = "court_motion_reacquisition"
        else:
            state = LogicalTrackingState.OBSERVED
            method = "tracker_continuity_with_court_constraints"
        resolved[frame_number] = _ResolvedObservation(
            selected,
            score,
            state,
            method,
            similarity,
            margin,
        )
        previous = selected
        previous_frame = frame_number
    return resolved


def resolve_logical_player_tracks(
    *,
    source: VideoMetadata,
    raw_observations: tuple[RawTrackerObservation, ...],
    ground_by_observation_id: dict[str, GroundContactAssessment],
    assignments: LogicalPlayerAssignments,
    settings: PlayerTrackingSettings,
    candidate_seed_indices: dict[LogicalPlayerRole, tuple[int, ...]] | None = None,
    appearance_similarities: dict[LogicalPlayerRole, dict[str, float]] | None = None,
) -> tuple[
    dict[LogicalPlayerRole, tuple[LogicalPlayerFrame, ...]],
    tuple[SuspectedIdentitySwitch, ...],
]:
    """Resolve manual logical roles bidirectionally from their trusted anchors."""

    geometry = tuple(
        TrackerGroundObservation(item, ground_by_observation_id[item.observation_id])
        for item in raw_observations
    )
    by_frame_lists: dict[int, list[TrackerGroundObservation]] = defaultdict(list)
    by_detection: dict[int, TrackerGroundObservation] = {}
    for item in geometry:
        by_frame_lists[item.raw.frame_number].append(item)
        by_detection[item.raw.raw_detection_index] = item
    by_frame = {frame: tuple(items) for frame, items in by_frame_lists.items()}
    similarities = appearance_similarities or {}
    side_by_role = {
        assignment.logical_player: assignment.observed_side
        for assignment in assignments.assignments
    }

    selected_by_role: dict[LogicalPlayerRole, dict[int, _ResolvedObservation]] = {}
    for assignment in assignments.assignments:
        anchor = by_detection.get(assignment.anchor_detection_index)
        if anchor is None:
            raise ValueError(
                f"manual anchor for {assignment.logical_player.value} was not emitted by the "
                "tracker; choose a frame where that player is visible in consecutive frames"
            )
        manual_anchor_frame = anchor.raw.frame_number
        same_side_roles = tuple(
            role for role in LOGICAL_PLAYER_ROLES if side_by_role[role] is assignment.observed_side
        )
        seed_observations = [
            by_detection[index]
            for index in (candidate_seed_indices or {}).get(assignment.logical_player, ())
            if index in by_detection
        ]
        defensible_seeds: list[TrackerGroundObservation] = []
        for seed in seed_observations:
            similarity, margin = _appearance_evidence(
                seed.raw.observation_id,
                role=assignment.logical_player,
                same_side_roles=same_side_roles,
                similarities=similarities,
            )
            if similarity is not None and similarity < settings.minimum_appearance_similarity:
                continue
            if margin is not None and margin < settings.minimum_appearance_margin:
                continue
            defensible_seeds.append(seed)
        seed_observations = defensible_seeds
        seed_by_frame = {item.raw.frame_number: item for item in seed_observations}
        seed_by_frame[manual_anchor_frame] = anchor
        ordered_seeds = sorted(seed_by_frame.values(), key=lambda item: item.raw.frame_number)
        selected: dict[int, _ResolvedObservation] = {}
        for item in ordered_seeds:
            similarity, margin = _appearance_evidence(
                item.raw.observation_id,
                role=assignment.logical_player,
                same_side_roles=same_side_roles,
                similarities=similarities,
            )
            selected[item.raw.frame_number] = _ResolvedObservation(
                item,
                1.0 if item.raw.frame_number == manual_anchor_frame else 0.80,
                LogicalTrackingState.OBSERVED,
                (
                    "manual_identity_anchor"
                    if item.raw.frame_number == manual_anchor_frame
                    else "manual_candidate_tracklet_seed"
                ),
                similarity,
                margin,
            )
        first_seed = ordered_seeds[0]
        selected.update(
            _resolve_role_direction(
                anchor=first_seed,
                expected_side=assignment.observed_side,
                frame_numbers=range(first_seed.raw.frame_number - 1, -1, -1),
                observations_by_frame=by_frame,
                settings=settings,
                role=assignment.logical_player,
                same_side_roles=same_side_roles,
                appearance_similarities=similarities,
            )
        )
        for earlier, later in pairwise(ordered_seeds):
            selected.update(
                _resolve_role_direction(
                    anchor=earlier,
                    expected_side=assignment.observed_side,
                    frame_numbers=range(
                        earlier.raw.frame_number + 1,
                        later.raw.frame_number,
                    ),
                    observations_by_frame=by_frame,
                    settings=settings,
                    role=assignment.logical_player,
                    same_side_roles=same_side_roles,
                    appearance_similarities=similarities,
                )
            )
        last_seed = ordered_seeds[-1]
        selected.update(
            _resolve_role_direction(
                anchor=last_seed,
                expected_side=assignment.observed_side,
                frame_numbers=range(last_seed.raw.frame_number + 1, source.frame_count),
                observations_by_frame=by_frame,
                settings=settings,
                role=assignment.logical_player,
                same_side_roles=same_side_roles,
                appearance_similarities=similarities,
            )
        )
        selected_by_role[assignment.logical_player] = selected

    # A tracker observation may support only one person. Keep the strongest role and
    # surface uncertainty as missing rather than letting roles silently collapse.
    for frame_number in range(source.frame_count):
        claims: dict[str, list[tuple[LogicalPlayerRole, _ResolvedObservation]]] = defaultdict(list)
        for role, selected in selected_by_role.items():
            value = selected.get(frame_number)
            if value is not None:
                claims[value.observation.raw.observation_id].append((role, value))
        for duplicate_claims in claims.values():
            if len(duplicate_claims) < 2:
                continue
            keep_role, _ = max(duplicate_claims, key=lambda item: item[1].confidence)
            for role, _ in duplicate_claims:
                if role != keep_role:
                    selected_by_role[role].pop(frame_number, None)

    logical_tracks: dict[LogicalPlayerRole, tuple[LogicalPlayerFrame, ...]] = {}
    switch_events: list[SuspectedIdentitySwitch] = []
    for role in LOGICAL_PLAYER_ROLES:
        frames: list[LogicalPlayerFrame] = []
        previous_tracker_id: int | None = None
        previous_observed_frame: int | None = None
        for frame_number in range(source.frame_count):
            timestamp_s = frame_number / source.fps
            chosen = selected_by_role[role].get(frame_number)
            if chosen is None:
                frames.append(
                    LogicalPlayerFrame(
                        role,
                        frame_number,
                        timestamp_s,
                        LogicalTrackingState.TEMPORARILY_MISSING,
                        0.0,
                        None,
                        "no_defensible_court_aware_match",
                    )
                )
                continue
            tracker_id = chosen.observation.raw.tracker_id
            had_missing_frame = (
                previous_observed_frame is not None and frame_number - previous_observed_frame > 1
            )
            tracker_changed = previous_tracker_id is not None and tracker_id != previous_tracker_id
            if chosen.method == "manual_identity_anchor" or previous_tracker_id is None:
                final_state = LogicalTrackingState.OBSERVED
            elif tracker_changed and (
                not had_missing_frame or chosen.confidence < settings.suspected_switch_score
            ):
                final_state = LogicalTrackingState.SUSPECTED_IDENTITY_SWITCH
            elif tracker_changed or had_missing_frame:
                final_state = LogicalTrackingState.REACQUIRED
            else:
                final_state = LogicalTrackingState.OBSERVED
            frames.append(
                LogicalPlayerFrame(
                    role,
                    frame_number,
                    timestamp_s,
                    final_state,
                    chosen.confidence,
                    chosen.observation,
                    chosen.method,
                    chosen.appearance_similarity,
                    chosen.appearance_margin,
                )
            )
            if (
                final_state is LogicalTrackingState.SUSPECTED_IDENTITY_SWITCH
                and previous_tracker_id is not None
                and tracker_id != previous_tracker_id
            ):
                switch_events.append(
                    SuspectedIdentitySwitch(
                        role,
                        frame_number,
                        timestamp_s,
                        previous_tracker_id,
                        tracker_id,
                        chosen.confidence,
                        "tracker ID changed without a confident occlusion-based reacquisition",
                    )
                )
            previous_tracker_id = tracker_id
            previous_observed_frame = frame_number
        logical_tracks[role] = tuple(frames)
    return logical_tracks, tuple(switch_events)


def build_tracking_run(
    *,
    source: VideoMetadata,
    detections_path: str,
    candidates_path: str,
    assignments_path: str,
    calibration_path: str,
    tracker: TrackerMetadata,
    configuration: dict[str, object],
    appearance: dict[str, object],
    player_names: dict[LogicalPlayerRole, str],
    raw_observations: tuple[RawTrackerObservation, ...],
    logical_tracks: dict[LogicalPlayerRole, tuple[LogicalPlayerFrame, ...]],
    suspected_identity_switches: tuple[SuspectedIdentitySwitch, ...],
) -> PlayerTrackingRun:
    return PlayerTrackingRun(
        source=source,
        detections_path=detections_path,
        candidates_path=candidates_path,
        assignments_path=assignments_path,
        calibration_path=calibration_path,
        tracker=tracker,
        configuration=configuration,
        appearance=appearance,
        player_names=player_names,
        raw_tracker_observations=raw_observations,
        logical_tracks=logical_tracks,
        suspected_identity_switches=suspected_identity_switches,
        created_at_utc=datetime.now(UTC).isoformat(),
    )


def tracking_summary(run: PlayerTrackingRun, *, artifacts: dict[str, str]) -> dict[str, object]:
    """Calculate inspectable coverage and gap metrics for each logical player."""

    per_player: dict[str, object] = {}
    events_by_role: dict[LogicalPlayerRole, int] = defaultdict(int)
    for event in run.suspected_identity_switches:
        events_by_role[event.logical_player] += 1
    for role in LOGICAL_PLAYER_ROLES:
        frames = run.logical_tracks[role]
        observed = sum(frame.tracker_observation is not None for frame in frames)
        reacquisitions = sum(frame.state is LogicalTrackingState.REACQUIRED for frame in frames)
        longest_start: int | None = None
        longest_end: int | None = None
        longest_frames = 0
        current_start: int | None = None
        for frame in frames:
            if frame.tracker_observation is None and current_start is None:
                current_start = frame.frame_number
            if frame.tracker_observation is not None and current_start is not None:
                candidate_end = frame.frame_number - 1
                candidate_frames = candidate_end - current_start + 1
                if candidate_frames > longest_frames:
                    longest_start, longest_end = current_start, candidate_end
                    longest_frames = candidate_frames
                current_start = None
        if current_start is not None:
            candidate_end = len(frames) - 1
            candidate_frames = candidate_end - current_start + 1
            if candidate_frames > longest_frames:
                longest_start, longest_end = current_start, candidate_end
                longest_frames = candidate_frames
        missing_frames = longest_frames
        per_player[role.value] = {
            "display_name": run.player_names[role],
            "observed_frames": observed,
            "coverage_ratio": observed / len(frames) if frames else 0.0,
            "suspected_id_switches": events_by_role[role],
            "reacquisition_count": reacquisitions,
            "longest_missing_interval": {
                "frames": missing_frames,
                "seconds": missing_frames / run.source.fps,
                "start_frame": longest_start,
                "end_frame": longest_end,
            },
        }
    return {
        "schema_version": 1,
        "record_type": "persistent_player_tracking_summary",
        "source": run.source.as_dict(),
        "frames_processed": run.source.frame_count,
        "suspected_id_switches": len(run.suspected_identity_switches),
        "players": per_player,
        "artifacts": artifacts,
    }
