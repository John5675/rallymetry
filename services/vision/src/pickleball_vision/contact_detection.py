"""Visual-first paddle-contact candidates with optional audio confidence support."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from pickleball_vision.config import ContactDetectionSettings
from pickleball_vision.court import ImagePoint
from pickleball_vision.rally_segmentation import BallEvidenceStatus, RallyBallFrame

CONTACT_DETECTION_SCHEMA_VERSION = 1


class ContactEvidenceMode(StrEnum):
    """How a visual contact candidate reached its final confidence state."""

    VISUAL_ONLY = "VISUAL_ONLY"
    VISUAL_PLUS_AUDIO = "VISUAL_PLUS_AUDIO"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


@dataclass(frozen=True, slots=True)
class ContactAudioTransient:
    """A generic non-semantic transient mapped onto video-relative time."""

    candidate_id: str
    video_timestamp_seconds: float
    confidence: float


@dataclass(frozen=True, slots=True)
class ContactRallyInterval:
    """Optional predicted rally interval used only as sequence support."""

    rally_id: str
    start_frame: int
    end_frame: int
    confidence: float


@dataclass(frozen=True, slots=True)
class PriorBounce:
    """An accepted prior-stage bounce used as optional event-state context."""

    bounce_id: str
    frame: int
    confidence: float


@dataclass(frozen=True, slots=True)
class ContactPlayerObservation:
    """One logical player observation kept independent of hitter assignment."""

    role: str
    display_name: str | None
    frame: int
    bounding_box: tuple[float, float, float, float]
    ground_image_position: ImagePoint
    tracking_confidence: float
    tracking_state: str
    court_side: str | None
    court_region: str | None


@dataclass(frozen=True, slots=True)
class ContactCandidatePlayer:
    """Ranked visual proximity evidence for one logical player."""

    role: str
    display_name: str | None
    rank: int
    bounding_box: tuple[float, float, float, float]
    ground_image_position: ImagePoint
    tracking_confidence: float
    tracking_state: str
    court_side: str | None
    court_region: str | None
    distance_px: float
    distance_diagonal_fraction: float
    proximity_confidence: float
    ball_inside_person_box: bool

    def as_dict(self) -> dict[str, object]:
        left, top, right, bottom = self.bounding_box
        return {
            "playerId": self.role,
            "displayName": self.display_name,
            "rank": self.rank,
            "trackingConfidence": self.tracking_confidence,
            "trackingState": self.tracking_state,
            "courtSide": self.court_side,
            "courtRegion": self.court_region,
            "trackerBoundingBox": {
                "left_px": left,
                "top_px": top,
                "right_px": right,
                "bottom_px": bottom,
                "coordinateSystem": "source_frame_pixels_top_left",
            },
            "groundContactImagePosition": {
                **self.ground_image_position.as_dict(),
                "method": "bounding_box_bottom_center",
                "usedAsPersonPhysicalCourtPosition": True,
            },
            "ballToBoundingBoxDistancePx": self.distance_px,
            "ballToBoundingBoxDistanceDiagonalFraction": self.distance_diagonal_fraction,
            "proximityConfidence": self.proximity_confidence,
            "ballInsidePersonBoundingBox": self.ball_inside_person_box,
            "isAssignedHitter": False,
        }


@dataclass(frozen=True, slots=True)
class ContactCandidate:
    """One visually generated contact candidate and optional audio fusion."""

    contact_id: str
    frame: int
    timestamp_seconds: float
    media_timestamp_seconds: float
    ball_image_position: ImagePoint
    trajectory_status: BallEvidenceStatus
    candidate_players: tuple[ContactCandidatePlayer, ...]
    visual_confidence: float
    audio_confidence: float
    fused_confidence: float
    matched_audio_event_id: str | None
    evidence_mode: ContactEvidenceMode
    accepted_vision_only: bool
    accepted_fused: bool
    supporting_signals: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "contactId": self.contact_id,
            "timestamp": self.timestamp_seconds,
            "mediaTimestamp": self.media_timestamp_seconds,
            "frame": self.frame,
            "ballImagePosition": {
                **self.ball_image_position.as_dict(),
                "coordinateSystem": "source_frame_pixels_top_left",
                "trajectoryStatus": self.trajectory_status.value,
            },
            "candidatePlayers": [item.as_dict() for item in self.candidate_players],
            "assignedHitter": None,
            "visualConfidence": self.visual_confidence,
            "audioConfidence": self.audio_confidence,
            "fusedConfidence": self.fused_confidence,
            "matchedAudioEventId": self.matched_audio_event_id,
            "evidenceMode": self.evidence_mode.value,
            "acceptedVisionOnly": self.accepted_vision_only,
            "acceptedFused": self.accepted_fused,
            "supportingSignals": self.supporting_signals,
        }


@dataclass(frozen=True, slots=True)
class ContactDetectionResult:
    """Visual candidate set after temporal suppression and optional fusion."""

    candidates: tuple[ContactCandidate, ...]
    raw_visual_candidate_count: int
    suppressed_visual_candidate_count: int
    matched_audio_candidate_count: int
    bounce_excluded_candidate_count: int


@dataclass(frozen=True, slots=True)
class _VisualCandidate:
    frame: int
    ball_image_position: ImagePoint
    trajectory_status: BallEvidenceStatus
    candidate_players: tuple[ContactCandidatePlayer, ...]
    visual_confidence: float
    supporting_signals: dict[str, object]


def _slope(
    frames: tuple[RallyBallFrame, ...],
    indices: tuple[int, ...],
    *,
    coordinate: str,
) -> float:
    times = np.asarray([frames[index].timestamp_seconds for index in indices], dtype=np.float64)
    coordinates: list[float] = []
    for index in indices:
        point = frames[index].point
        if point is None:
            raise AssertionError("velocity-fit indices must contain known trajectory points")
        coordinates.append(point.x_px if coordinate == "x" else point.y_px)
    values = np.asarray(coordinates, dtype=np.float64)
    centered = times - float(times.mean())
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(centered, values - float(values.mean())) / denominator)


def _angle_degrees(before: tuple[float, float], after: tuple[float, float]) -> float:
    before_norm = math.hypot(*before)
    after_norm = math.hypot(*after)
    if before_norm <= 1e-12 or after_norm <= 1e-12:
        return 0.0
    cosine = (before[0] * after[0] + before[1] * after[1]) / (before_norm * after_norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _point_to_box_distance(
    point: ImagePoint,
    box: tuple[float, float, float, float],
) -> tuple[float, bool]:
    left, top, right, bottom = box
    dx = max(left - point.x_px, 0.0, point.x_px - right)
    dy = max(top - point.y_px, 0.0, point.y_px - bottom)
    return math.hypot(dx, dy), dx == 0 and dy == 0


def _candidate_players(
    ball_point: ImagePoint,
    players: tuple[ContactPlayerObservation, ...],
    *,
    frame_diagonal_px: float,
    maximum_distance_fraction: float,
) -> tuple[ContactCandidatePlayer, ...]:
    maximum_distance_px = maximum_distance_fraction * frame_diagonal_px
    ranked: list[tuple[float, ContactPlayerObservation, float, bool]] = []
    for player in players:
        distance_px, inside = _point_to_box_distance(ball_point, player.bounding_box)
        proximity = max(0.0, 1.0 - distance_px / maximum_distance_px)
        ranked.append((distance_px, player, proximity, inside))
    ranked.sort(key=lambda item: (item[0], item[1].role))
    return tuple(
        ContactCandidatePlayer(
            role=player.role,
            display_name=player.display_name,
            rank=index,
            bounding_box=player.bounding_box,
            ground_image_position=player.ground_image_position,
            tracking_confidence=player.tracking_confidence,
            tracking_state=player.tracking_state,
            court_side=player.court_side,
            court_region=player.court_region,
            distance_px=distance_px,
            distance_diagonal_fraction=distance_px / frame_diagonal_px,
            proximity_confidence=proximity,
            ball_inside_person_box=inside,
        )
        for index, (distance_px, player, proximity, inside) in enumerate(ranked, start=1)
    )


def _rally_support(
    frame: int,
    rallies: tuple[ContactRallyInterval, ...],
) -> ContactRallyInterval | None:
    return next((item for item in rallies if item.start_frame <= frame <= item.end_frame), None)


def _prior_bounce_state(
    frame: int,
    bounces: tuple[PriorBounce, ...],
    *,
    fps: float,
    exclusion_seconds: float,
) -> tuple[PriorBounce | None, float | None, bool]:
    nearest = min(bounces, key=lambda item: abs(item.frame - frame), default=None)
    effective_exclusion_seconds = max(exclusion_seconds, 1.0 / fps)
    if nearest is not None and abs(nearest.frame - frame) / fps <= effective_exclusion_seconds:
        return nearest, (frame - nearest.frame) / fps, True
    previous = max(
        (item for item in bounces if item.frame < frame), key=lambda item: item.frame, default=None
    )
    return previous, ((frame - previous.frame) / fps if previous is not None else None), False


def _candidate_at_frame(
    frames: tuple[RallyBallFrame, ...],
    *,
    frame: int,
    window_frames: int,
    frame_diagonal_px: float,
    players_by_frame: tuple[tuple[ContactPlayerObservation, ...], ...],
    rallies: tuple[ContactRallyInterval, ...],
    bounces: tuple[PriorBounce, ...],
    fps: float,
    settings: ContactDetectionSettings,
) -> tuple[_VisualCandidate | None, bool]:
    center = frames[frame]
    if center.point is None or center.segment_id is None:
        return None, False
    before = tuple(
        index
        for index in range(max(0, frame - window_frames), frame)
        if frames[index].point is not None and frames[index].segment_id == center.segment_id
    )
    after = tuple(
        index
        for index in range(frame + 1, min(len(frames), frame + window_frames + 1))
        if frames[index].point is not None and frames[index].segment_id == center.segment_id
    )
    if (
        len(before) < settings.minimum_observations_each_side
        or len(after) < settings.minimum_observations_each_side
    ):
        return None, False
    before_fit = (*before, frame)
    after_fit = (frame, *after)
    before_velocity = (
        _slope(frames, before_fit, coordinate="x"),
        _slope(frames, before_fit, coordinate="y"),
    )
    after_velocity = (
        _slope(frames, after_fit, coordinate="x"),
        _slope(frames, after_fit, coordinate="y"),
    )
    before_speed = math.hypot(*before_velocity) / frame_diagonal_px
    after_speed = math.hypot(*after_velocity) / frame_diagonal_px
    velocity_change = (
        math.hypot(
            after_velocity[0] - before_velocity[0],
            after_velocity[1] - before_velocity[1],
        )
        / frame_diagonal_px
    )
    if velocity_change < settings.minimum_velocity_change_diagonals_per_second:
        return None, False
    direction_change = _angle_degrees(before_velocity, after_velocity)
    speed_floor = max(1e-6, min(before_speed, after_speed))
    speed_ratio = max(before_speed, after_speed) / speed_floor
    if (
        direction_change < settings.minimum_direction_change_degrees
        and speed_ratio < settings.minimum_speed_change_ratio
    ):
        return None, False
    window_start = max(0, frame - window_frames)
    window_stop = min(len(frames), frame + window_frames + 1)
    window_count = window_stop - window_start
    same_segment = tuple(
        item
        for item in frames[window_start:window_stop]
        if item.point is not None and item.segment_id == center.segment_id
    )
    continuity_fraction = len(same_segment) / window_count
    if continuity_fraction < settings.minimum_continuity_fraction:
        return None, False
    observed_fraction = sum(
        item.status is BallEvidenceStatus.OBSERVED for item in same_segment
    ) / len(same_segment)
    prior_bounce, prior_bounce_gap, excluded_by_bounce = _prior_bounce_state(
        frame,
        bounces,
        fps=fps,
        exclusion_seconds=settings.bounce_exclusion_window_seconds,
    )
    if excluded_by_bounce:
        return None, True
    candidate_players = _candidate_players(
        center.point,
        players_by_frame[frame],
        frame_diagonal_px=frame_diagonal_px,
        maximum_distance_fraction=settings.maximum_player_proximity_diagonal_fraction,
    )
    nearest_player = candidate_players[0] if candidate_players else None
    proximity_quality = (
        nearest_player.proximity_confidence * nearest_player.tracking_confidence
        if nearest_player is not None
        else 0.0
    )
    court_context_quality = (
        1.0
        if nearest_player is not None and nearest_player.court_region == "inside"
        else (0.65 if nearest_player is not None and nearest_player.court_region == "near" else 0.0)
    )
    rally = _rally_support(frame, rallies)
    bounce_sequence_quality = (
        prior_bounce.confidence
        if prior_bounce is not None
        and prior_bounce_gap is not None
        and 0.10 <= prior_bounce_gap <= settings.maximum_previous_bounce_gap_seconds
        else 0.0
    )
    velocity_quality = min(
        1.0,
        velocity_change / (settings.minimum_velocity_change_diagonals_per_second * 3),
    )
    direction_quality = min(1.0, direction_change / 90.0)
    speed_change_quality = min(
        1.0,
        max(0.0, speed_ratio - 1.0) / max(1e-6, settings.minimum_speed_change_ratio),
    )
    discontinuity_quality = max(direction_quality, speed_change_quality)
    center_observation_quality = 1.0 if center.status is BallEvidenceStatus.OBSERVED else 0.55
    rally_boost = settings.rally_sequence_confidence_boost * (
        rally.confidence if rally is not None else 0.0
    )
    bounce_boost = settings.previous_bounce_confidence_boost * bounce_sequence_quality
    visual_confidence = min(
        1.0,
        0.28 * velocity_quality
        + 0.20 * discontinuity_quality
        + 0.14 * continuity_fraction
        + 0.09 * observed_fraction
        + 0.05 * center_observation_quality
        + 0.17 * proximity_quality
        + 0.04 * court_context_quality
        + rally_boost
        + bounce_boost,
    )
    if visual_confidence < settings.minimum_visual_candidate_confidence:
        return None, False
    signals: dict[str, object] = {
        "trajectoryVelocityDiscontinuity": {
            "beforeVelocityPixelsPerSecond": {
                "x": before_velocity[0],
                "y": before_velocity[1],
            },
            "afterVelocityPixelsPerSecond": {
                "x": after_velocity[0],
                "y": after_velocity[1],
            },
            "beforeSpeedDiagonalsPerSecond": before_speed,
            "afterSpeedDiagonalsPerSecond": after_speed,
            "velocityChangeDiagonalsPerSecond": velocity_change,
            "minimumRequired": settings.minimum_velocity_change_diagonals_per_second,
            "detected": True,
        },
        "trajectoryDirectionDiscontinuity": {
            "directionChangeDegrees": direction_change,
            "minimumDirectionChangeDegrees": settings.minimum_direction_change_degrees,
            "speedChangeRatio": speed_ratio,
            "minimumSpeedChangeRatio": settings.minimum_speed_change_ratio,
            "criterionSatisfied": True,
        },
        "trajectoryBeforeAfter": {
            "beforeFrameIndices": list(before),
            "afterFrameIndices": list(after),
            "centerFrame": frame,
            "segmentId": center.segment_id,
        },
        "trajectoryContinuity": {
            "windowFrameCount": window_count,
            "knownSameSegmentFrameCount": len(same_segment),
            "knownFraction": continuity_fraction,
            "observedFractionAmongKnown": observed_fraction,
        },
        "playerProximity": {
            "availableLogicalPlayerCount": len(candidate_players),
            "nearestCandidatePlayerId": (
                nearest_player.role if nearest_player is not None else None
            ),
            "nearestDistancePx": (
                nearest_player.distance_px if nearest_player is not None else None
            ),
            "nearestProximityConfidence": (
                nearest_player.proximity_confidence if nearest_player is not None else 0.0
            ),
            "usedAsVisualSupport": True,
            "assignsHitter": False,
        },
        "courtSideContext": {
            "candidatePlayerCourtSides": sorted(
                {item.court_side for item in candidate_players if item.court_side is not None}
            ),
            "nearestPlayerCourtSide": (
                nearest_player.court_side if nearest_player is not None else None
            ),
            "nearestPlayerCourtRegion": (
                nearest_player.court_region if nearest_player is not None else None
            ),
            "ballCourtSideInferredByHomography": False,
            "airborneBallProjectedThroughHomography": False,
        },
        "plausibleRallySequence": {
            "available": bool(rallies),
            "insidePredictedRally": rally is not None,
            "rallyId": rally.rally_id if rally is not None else None,
            "confidenceBoost": rally_boost,
            "canCreateContact": False,
        },
        "previousBounceState": {
            "available": bool(bounces),
            "previousBounceId": prior_bounce.bounce_id if prior_bounce is not None else None,
            "secondsSincePreviousBounce": prior_bounce_gap,
            "exclusionWindowSeconds": settings.bounce_exclusion_window_seconds,
            "excludedAsCoincidentBounce": False,
            "confidenceBoost": bounce_boost,
            "canCreateContact": False,
        },
        "visualConfidenceComponents": {
            "velocityDiscontinuity": 0.28 * velocity_quality,
            "directionOrSpeedDiscontinuity": 0.20 * discontinuity_quality,
            "continuity": 0.14 * continuity_fraction,
            "observedSupport": 0.09 * observed_fraction,
            "centerObservation": 0.05 * center_observation_quality,
            "playerProximity": 0.17 * proximity_quality,
            "trackedPlayerCourtContext": 0.04 * court_context_quality,
            "rallySequenceBoost": rally_boost,
            "previousBounceBoost": bounce_boost,
            "calibratedProbability": False,
        },
    }
    return (
        _VisualCandidate(
            frame=frame,
            ball_image_position=center.point,
            trajectory_status=center.status,
            candidate_players=candidate_players,
            visual_confidence=visual_confidence,
            supporting_signals=signals,
        ),
        False,
    )


def _suppress_nearby_candidates(
    candidates: tuple[_VisualCandidate, ...],
    *,
    minimum_frames: int,
) -> tuple[_VisualCandidate, ...]:
    accepted: list[_VisualCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.visual_confidence, reverse=True):
        if any(abs(candidate.frame - item.frame) <= minimum_frames for item in accepted):
            continue
        accepted.append(candidate)
    return tuple(sorted(accepted, key=lambda item: item.frame))


def _audio_matches(
    candidates: tuple[_VisualCandidate, ...],
    audio: tuple[ContactAudioTransient, ...],
    *,
    fps: float,
    tolerance_seconds: float,
) -> dict[int, tuple[ContactAudioTransient, float, float]]:
    proposals: list[tuple[float, float, int, ContactAudioTransient]] = []
    for candidate_index, candidate in enumerate(candidates):
        candidate_time = candidate.frame / fps
        for transient in audio:
            delta = abs(transient.video_timestamp_seconds - candidate_time)
            if delta <= tolerance_seconds:
                proposals.append((delta, -transient.confidence, candidate_index, transient))
    proposals.sort(key=lambda item: item[:2])
    used_candidates: set[int] = set()
    used_audio: set[str] = set()
    matches: dict[int, tuple[ContactAudioTransient, float, float]] = {}
    for delta, _negative_confidence, candidate_index, transient in proposals:
        if candidate_index in used_candidates or transient.candidate_id in used_audio:
            continue
        timing_quality = 1.0 - delta / tolerance_seconds if tolerance_seconds > 0 else 1.0
        confidence = transient.confidence * (0.5 + 0.5 * timing_quality)
        matches[candidate_index] = (transient, confidence, delta)
        used_candidates.add(candidate_index)
        used_audio.add(transient.candidate_id)
    return matches


def detect_contact_candidates(
    frames: tuple[RallyBallFrame, ...],
    *,
    players_by_frame: tuple[tuple[ContactPlayerObservation, ...], ...],
    fps: float,
    frame_width_px: int,
    frame_height_px: int,
    settings: ContactDetectionSettings,
    audio_transients: tuple[ContactAudioTransient, ...] = (),
    rallies: tuple[ContactRallyInterval, ...] = (),
    prior_bounces: tuple[PriorBounce, ...] = (),
    video_start_time_seconds: float = 0.0,
    fusion_tolerance_ms: float = 90.0,
) -> ContactDetectionResult:
    """Generate candidates visually, then optionally fuse one nearby transient."""

    if not frames:
        raise ValueError("contact detection requires a nonempty trajectory timeline")
    if len(players_by_frame) != len(frames):
        raise ValueError("player observations must form the same frame-complete timeline")
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError("contact detection FPS must be finite and positive")
    if frame_width_px < 1 or frame_height_px < 1:
        raise ValueError("contact detection dimensions must be positive")
    if not math.isfinite(fusion_tolerance_ms) or fusion_tolerance_ms <= 0:
        raise ValueError("fusion tolerance must be finite and positive")
    if tuple(item.frame_number for item in frames) != tuple(range(len(frames))):
        raise ValueError("contact trajectory frames must form a complete zero-based timeline")
    diagonal = math.hypot(frame_width_px, frame_height_px)
    window_frames = max(
        settings.minimum_observations_each_side,
        round(settings.trajectory_window_seconds * fps),
    )
    raw: list[_VisualCandidate] = []
    bounce_excluded = 0
    for frame in range(window_frames, len(frames) - window_frames):
        candidate, excluded = _candidate_at_frame(
            frames,
            frame=frame,
            window_frames=window_frames,
            frame_diagonal_px=diagonal,
            players_by_frame=players_by_frame,
            rallies=rallies,
            bounces=prior_bounces,
            fps=fps,
            settings=settings,
        )
        bounce_excluded += int(excluded)
        if candidate is not None:
            raw.append(candidate)
    visual = _suppress_nearby_candidates(
        tuple(raw),
        minimum_frames=max(1, round(settings.minimum_between_contacts_seconds * fps)),
    )
    matches = _audio_matches(
        visual,
        audio_transients,
        fps=fps,
        tolerance_seconds=fusion_tolerance_ms / 1000.0,
    )
    candidates: list[ContactCandidate] = []
    for index, candidate in enumerate(visual):
        match = matches.get(index)
        transient = match[0] if match is not None else None
        audio_confidence = match[1] if match is not None else 0.0
        delta_seconds = match[2] if match is not None else None
        fused_confidence = min(
            1.0,
            candidate.visual_confidence
            + settings.audio_confidence_weight
            * (1.0 - candidate.visual_confidence)
            * audio_confidence,
        )
        accepted_vision = candidate.visual_confidence >= settings.accepted_confidence
        accepted_fused = fused_confidence >= settings.accepted_confidence
        mode = (
            ContactEvidenceMode.LOW_CONFIDENCE
            if not accepted_fused
            else (
                ContactEvidenceMode.VISUAL_PLUS_AUDIO
                if transient is not None
                else ContactEvidenceMode.VISUAL_ONLY
            )
        )
        previous = candidates[-1] if candidates else None
        supporting = {
            **candidate.supporting_signals,
            "previousContactCandidateState": {
                "previousContactCandidateId": (
                    previous.contact_id if previous is not None else None
                ),
                "secondsSincePreviousCandidate": (
                    (candidate.frame - previous.frame) / fps if previous is not None else None
                ),
                "usedForHitterAssignment": False,
                "semanticStateInferred": False,
            },
            "audioFusion": {
                "available": bool(audio_transients),
                "matched": transient is not None,
                "matchedAudioEventId": (transient.candidate_id if transient is not None else None),
                "absoluteTimingDeltaMs": (
                    delta_seconds * 1000 if delta_seconds is not None else None
                ),
                "fusionToleranceMs": fusion_tolerance_ms,
                "audioVideoOffsetAppliedUpstream": True,
                "confidenceWeight": settings.audio_confidence_weight,
                "canCreateContact": False,
                "neighboringCourtSoundMayBePresent": True,
            },
        }
        candidates.append(
            ContactCandidate(
                contact_id=f"contact-candidate-{len(candidates) + 1:06d}",
                frame=candidate.frame,
                timestamp_seconds=candidate.frame / fps,
                media_timestamp_seconds=video_start_time_seconds + candidate.frame / fps,
                ball_image_position=candidate.ball_image_position,
                trajectory_status=candidate.trajectory_status,
                candidate_players=candidate.candidate_players,
                visual_confidence=candidate.visual_confidence,
                audio_confidence=audio_confidence,
                fused_confidence=fused_confidence,
                matched_audio_event_id=(transient.candidate_id if transient is not None else None),
                evidence_mode=mode,
                accepted_vision_only=accepted_vision,
                accepted_fused=accepted_fused,
                supporting_signals=supporting,
            )
        )
    return ContactDetectionResult(
        candidates=tuple(candidates),
        raw_visual_candidate_count=len(raw),
        suppressed_visual_candidate_count=len(raw) - len(visual),
        matched_audio_candidate_count=len(matches),
        bounce_excluded_candidate_count=bounce_excluded,
    )
