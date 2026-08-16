"""Structured, inspectable automatic rally segmentation."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, replace
from enum import StrEnum

from pickleball_vision.config import RallySegmentationSettings
from pickleball_vision.court import ImagePoint

RALLY_SEGMENTATION_SCHEMA_VERSION = 1


class BallEvidenceStatus(StrEnum):
    """Accepted status values from the conservative ball trajectory artifact."""

    OBSERVED = "OBSERVED"
    INTERPOLATED = "INTERPOLATED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class RallyBallFrame:
    """Minimal immutable trajectory evidence used by rally inference."""

    frame_number: int
    timestamp_seconds: float
    status: BallEvidenceStatus
    segment_id: str | None
    point: ImagePoint | None
    confidence: float | None


@dataclass(frozen=True, slots=True)
class AudioTransientEvidence:
    """Optional generic audio evidence on the video-relative timeline."""

    candidate_id: str
    video_timestamp_seconds: float
    confidence: float


@dataclass(frozen=True, slots=True)
class ServeLikeSignal:
    """Non-semantic motion onset that is geometrically consistent with a serve sequence."""

    candidate_frame: int
    refined_start_frame: int
    speed_diagonals_per_second: float
    baseline_speed_diagonals_per_second: float
    speed_surge_ratio: float
    confirmation_known_fraction: float
    confirmation_motion_fraction: float
    displacement_diagonal_fraction: float

    def as_dict(self) -> dict[str, object]:
        return {
            "detected": True,
            "candidateFrame": self.candidate_frame,
            "refinedStartFrame": self.refined_start_frame,
            "speedDiagonalsPerSecond": self.speed_diagonals_per_second,
            "baselineSpeedDiagonalsPerSecond": (self.baseline_speed_diagonals_per_second),
            "speedSurgeRatio": self.speed_surge_ratio,
            "confirmationKnownFraction": self.confirmation_known_fraction,
            "confirmationMotionFraction": self.confirmation_motion_fraction,
            "displacementDiagonalFraction": self.displacement_diagonal_fraction,
            "semanticServeContactInferred": False,
        }


@dataclass(frozen=True, slots=True)
class RallyPrediction:
    """One derived rally interval with explicit supporting signals."""

    rally_id: str
    start_frame: int
    end_frame: int
    start_timestamp_seconds: float
    end_timestamp_seconds: float
    confidence: float
    supporting_signals: dict[str, object]
    rally_evidence_quality: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "rallyId": self.rally_id,
            "startTimestamp": self.start_timestamp_seconds,
            "endTimestamp": self.end_timestamp_seconds,
            "startFrame": self.start_frame,
            "endFrame": self.end_frame,
            "confidence": self.confidence,
            "supportingSignals": self.supporting_signals,
        }


@dataclass(frozen=True, slots=True)
class AdjacentBurstRejection:
    """A retained, non-semantic rejection of a weaker adjacent activity burst."""

    candidate: RallyPrediction
    stronger_candidate_id: str
    stronger_evidence_quality: float
    gap_seconds: float
    quality_margin: float

    def as_dict(self) -> dict[str, object]:
        candidate = self.candidate.as_dict()
        supporting = dict(self.candidate.supporting_signals)
        raw_assessment = supporting.get("deadBallHandoffAssessment")
        if not isinstance(raw_assessment, dict):
            raise AssertionError("dead-ball handoff assessment must be an object")
        assessment = {
            **raw_assessment,
            "possibleDeadBallHandoff": True,
            "rejectedAsRally": True,
            "strongerRallyId": self.stronger_candidate_id,
            "gapToStrongerCandidateSeconds": self.gap_seconds,
            "qualityMargin": self.quality_margin,
        }
        candidate["supportingSignals"] = {
            **supporting,
            "deadBallHandoffAssessment": assessment,
        }
        return {
            "candidateId": self.candidate.rally_id,
            "reason": "adjacent_activity_burst_lower_rally_evidence",
            "possibleDeadBallHandoff": True,
            "semanticClassification": None,
            "gapToStrongerCandidateSeconds": self.gap_seconds,
            "rallyEvidenceQuality": self.candidate.rally_evidence_quality,
            "strongerCandidateId": self.stronger_candidate_id,
            "strongerRallyEvidenceQuality": self.stronger_evidence_quality,
            "qualityMargin": self.quality_margin,
            "candidate": candidate,
        }


@dataclass(frozen=True, slots=True)
class RallySegmentationResult:
    """Predictions plus frame-local kinematics used by debug rendering."""

    rallies: tuple[RallyPrediction, ...]
    speeds_diagonals_per_second: tuple[float | None, ...]
    motion_supported: tuple[bool, ...]
    serve_candidate_count: int
    rejected_adjacent_bursts: tuple[AdjacentBurstRejection, ...]


def _distance(left: ImagePoint, right: ImagePoint) -> float:
    return math.hypot(left.x_px - right.x_px, left.y_px - right.y_px)


def _frame_kinematics(
    frames: tuple[RallyBallFrame, ...],
    *,
    fps: float,
    frame_diagonal_px: float,
    settings: RallySegmentationSettings,
) -> tuple[tuple[float | None, ...], tuple[bool, ...], tuple[bool, ...]]:
    speeds: list[float | None] = [None] * len(frames)
    previous_known: int | None = None
    for index, frame in enumerate(frames):
        if frame.point is None:
            continue
        if previous_known is not None:
            previous = frames[previous_known]
            elapsed_seconds = (index - previous_known) / fps
            if (
                frame.segment_id is not None
                and frame.segment_id == previous.segment_id
                and elapsed_seconds <= settings.motion_link_gap_seconds
                and previous.point is not None
            ):
                speeds[index] = _distance(frame.point, previous.point) / (
                    frame_diagonal_px * elapsed_seconds
                )
        previous_known = index
    raw_motion = tuple(
        value is not None and value >= settings.minimum_motion_speed_diagonals_per_second
        for value in speeds
    )
    window_frames = max(3, round(settings.motion_support_window_seconds * fps))
    radius = window_frames // 2
    supported: list[bool] = []
    for index in range(len(frames)):
        start = max(0, index - radius)
        stop = min(len(frames), index + radius + 1)
        fraction = sum(raw_motion[start:stop]) / (stop - start)
        supported.append(fraction >= settings.minimum_motion_support_fraction)
    return tuple(speeds), raw_motion, tuple(supported)


def _maximum_displacement_fraction(
    frames: tuple[RallyBallFrame, ...],
    indices: range,
    *,
    frame_diagonal_px: float,
) -> float:
    points_list: list[ImagePoint] = []
    for index in indices:
        point = frames[index].point
        if point is not None:
            points_list.append(point)
    points = tuple(points_list)
    if len(points) < 2:
        return 0.0
    return max(_distance(left, right) for left in points for right in points) / frame_diagonal_px


def _refined_start_frame(
    candidate_frame: int,
    *,
    speeds: tuple[float | None, ...],
    frames: tuple[RallyBallFrame, ...],
    fps: float,
    settings: RallySegmentationSettings,
) -> int:
    earliest = max(0, candidate_frame - round(settings.serve_baseline_window_seconds * fps))
    refined = candidate_frame
    for index in range(candidate_frame - 1, earliest - 1, -1):
        speed = speeds[index]
        if frames[index].point is None or speed is None:
            break
        if speed < settings.minimum_motion_speed_diagonals_per_second:
            break
        refined = index
    if refined == candidate_frame:
        for index in range(candidate_frame - 1, earliest - 1, -1):
            if frames[index].point is None:
                break
            refined = index
    return refined


def _serve_like_candidates(
    frames: tuple[RallyBallFrame, ...],
    *,
    speeds: tuple[float | None, ...],
    raw_motion: tuple[bool, ...],
    fps: float,
    frame_diagonal_px: float,
    settings: RallySegmentationSettings,
) -> tuple[ServeLikeSignal, ...]:
    baseline_frames = max(1, round(settings.serve_baseline_window_seconds * fps))
    confirmation_frames = max(1, round(settings.serve_confirmation_seconds * fps))
    candidates: list[ServeLikeSignal] = []
    for index, speed in enumerate(speeds):
        if speed is None or speed < settings.serve_minimum_speed_diagonals_per_second:
            continue
        baseline_values = tuple(
            speeds[item] or 0.0 for item in range(max(0, index - baseline_frames), index)
        )
        baseline = statistics.median(baseline_values) if baseline_values else 0.0
        surge_ratio = speed / max(
            baseline,
            settings.minimum_motion_speed_diagonals_per_second / 2,
        )
        if (
            surge_ratio < settings.serve_speed_surge_ratio
            and speed - baseline < settings.minimum_motion_speed_diagonals_per_second
        ):
            continue
        stop = min(len(frames), index + confirmation_frames + 1)
        confirmation_range = range(index, stop)
        sample_count = stop - index
        known_fraction = (
            sum(frames[item].point is not None for item in confirmation_range) / sample_count
        )
        motion_fraction = sum(raw_motion[item] for item in confirmation_range) / sample_count
        displacement = _maximum_displacement_fraction(
            frames,
            confirmation_range,
            frame_diagonal_px=frame_diagonal_px,
        )
        if (
            known_fraction < 0.30
            or motion_fraction < settings.serve_minimum_motion_fraction
            or displacement < settings.serve_minimum_displacement_diagonal_fraction
        ):
            continue
        candidates.append(
            ServeLikeSignal(
                candidate_frame=index,
                refined_start_frame=_refined_start_frame(
                    index,
                    speeds=speeds,
                    frames=frames,
                    fps=fps,
                    settings=settings,
                ),
                speed_diagonals_per_second=speed,
                baseline_speed_diagonals_per_second=baseline,
                speed_surge_ratio=surge_ratio,
                confirmation_known_fraction=known_fraction,
                confirmation_motion_fraction=motion_fraction,
                displacement_diagonal_fraction=displacement,
            )
        )
    return tuple(candidates)


def _has_quiet_run(
    raw_motion: tuple[bool, ...],
    *,
    start: int,
    stop: int,
    required_frames: int,
) -> bool:
    quiet = 0
    for moving in raw_motion[max(0, start) : min(len(raw_motion), stop)]:
        quiet = 0 if moving else quiet + 1
        if quiet >= required_frames:
            return True
    return False


def _last_defensible_frame(
    frames: tuple[RallyBallFrame, ...],
    raw_motion: tuple[bool, ...],
    *,
    start: int,
    quiet_start: int,
    tail_grace_frames: int,
) -> int:
    motion_frames = tuple(index for index in range(start, quiet_start + 1) if raw_motion[index])
    if not motion_frames:
        return start
    last_motion = motion_frames[-1]
    cutoff = min(quiet_start, last_motion + tail_grace_frames)
    known = tuple(
        index for index in range(last_motion, cutoff + 1) if frames[index].point is not None
    )
    return known[-1] if known else last_motion


def _natural_end_frame(
    frames: tuple[RallyBallFrame, ...],
    raw_motion: tuple[bool, ...],
    supported_motion: tuple[bool, ...],
    *,
    start_frame: int,
    fps: float,
    settings: RallySegmentationSettings,
) -> int:
    minimum_end = min(
        len(frames) - 1,
        start_frame + max(1, round(settings.minimum_rally_duration_seconds * fps)),
    )
    maximum_end = min(
        len(frames) - 1,
        start_frame + max(1, round(settings.maximum_rally_duration_seconds * fps)),
    )
    quiet_frames = max(1, round(settings.end_quiet_seconds * fps))
    tail_grace = max(0, round(settings.end_tail_grace_seconds * fps))
    for quiet_start in range(minimum_end, maximum_end + 1):
        quiet_stop = min(len(frames), quiet_start + quiet_frames)
        if quiet_stop - quiet_start < quiet_frames:
            break
        if not any(supported_motion[quiet_start:quiet_stop]):
            return _last_defensible_frame(
                frames,
                raw_motion,
                start=start_frame,
                quiet_start=quiet_start,
                tail_grace_frames=tail_grace,
            )
    return maximum_end


def _audio_support(
    transients: tuple[AudioTransientEvidence, ...],
    *,
    start_seconds: float,
    end_seconds: float,
    tolerance_seconds: float,
) -> dict[str, object]:
    if not transients:
        return {
            "available": False,
            "candidateCountWithinRally": 0,
            "nearestStartTransientMs": None,
            "nearestEndTransientMs": None,
            "effect": "none_vision_only",
            "canCreateBoundary": False,
        }
    inside = tuple(
        item for item in transients if start_seconds <= item.video_timestamp_seconds <= end_seconds
    )
    nearest_start = min(
        (abs(item.video_timestamp_seconds - start_seconds) for item in transients),
        default=math.inf,
    )
    nearest_end = min(
        (abs(item.video_timestamp_seconds - end_seconds) for item in transients),
        default=math.inf,
    )
    supports = nearest_start <= tolerance_seconds or nearest_end <= tolerance_seconds
    return {
        "available": True,
        "candidateCountWithinRally": len(inside),
        "nearestStartTransientMs": nearest_start * 1000,
        "nearestEndTransientMs": nearest_end * 1000,
        "boundaryProximitySupport": supports,
        "effect": "confidence_support_only" if supports else "recorded_no_boundary_support",
        "canCreateBoundary": False,
    }


def _interval_support(
    frames: tuple[RallyBallFrame, ...],
    speeds: tuple[float | None, ...],
    raw_motion: tuple[bool, ...],
    supported_motion: tuple[bool, ...],
    signal: ServeLikeSignal,
    *,
    start_frame: int,
    end_frame: int,
    previous_end_frame: int | None,
    fps: float,
    frame_diagonal_px: float,
    player_reset_scores: tuple[float | None, ...] | None,
    audio_transients: tuple[AudioTransientEvidence, ...],
    settings: RallySegmentationSettings,
) -> tuple[dict[str, object], float, float]:
    indices = range(start_frame, end_frame + 1)
    count = end_frame - start_frame + 1
    observed_count = sum(frames[index].status is BallEvidenceStatus.OBSERVED for index in indices)
    interpolated_count = sum(
        frames[index].status is BallEvidenceStatus.INTERPOLATED for index in indices
    )
    known_count = observed_count + interpolated_count
    motion_fraction = sum(raw_motion[index] for index in indices) / count
    sustained_fraction = sum(supported_motion[index] for index in indices) / count
    known_fraction = known_count / count
    interval_speeds = tuple(value for value in speeds[start_frame : end_frame + 1] if value)
    segment_ids = sorted(
        segment_id for index in indices if (segment_id := frames[index].segment_id) is not None
    )
    preceding_gap_frames = 0
    for index in range(start_frame - 1, -1, -1):
        if frames[index].point is not None:
            break
        preceding_gap_frames += 1
    following_gap_frames = 0
    for index in range(end_frame + 1, len(frames)):
        if frames[index].point is not None:
            break
        following_gap_frames += 1
    player_score = (
        player_reset_scores[start_frame]
        if player_reset_scores is not None and start_frame < len(player_reset_scores)
        else None
    )
    player_support = {
        "available": player_score is not None,
        "resetScoreBeforeStart": player_score,
        "effect": "confidence_support_only" if player_score is not None else "unavailable",
        "canCreateBoundary": False,
    }
    start_seconds = start_frame / fps
    end_seconds = end_frame / fps
    audio = _audio_support(
        audio_transients,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        tolerance_seconds=settings.audio_support_tolerance_seconds,
    )
    activity_quality = min(1.0, motion_fraction / 0.45)
    known_quality = min(1.0, known_fraction / 0.70)
    serve_quality = statistics.fmean(
        (
            min(1.0, signal.speed_surge_ratio / (settings.serve_speed_surge_ratio * 2)),
            min(
                1.0,
                signal.displacement_diagonal_fraction
                / (settings.serve_minimum_displacement_diagonal_fraction * 2),
            ),
            min(
                1.0,
                signal.confirmation_motion_fraction
                / max(settings.serve_minimum_motion_fraction * 2, 0.01),
            ),
        )
    )
    confidence = 0.25 + 0.30 * serve_quality + 0.25 * activity_quality + 0.15 * known_quality
    if player_score is not None:
        confidence += 0.03 * player_score
    if audio.get("boundaryProximitySupport") is True:
        confidence += 0.02
    confidence = min(0.99, max(0.0, confidence))
    duration_seconds = count / fps
    rally_evidence_quality = (
        0.35 * motion_fraction
        + 0.25 * known_fraction
        + 0.20 * sustained_fraction
        + 0.20 * min(1.0, duration_seconds / settings.dead_ball_handoff_full_duration_seconds)
    )
    supporting = {
        "ballTrajectoryActivity": {
            "observedFrameCount": observed_count,
            "interpolatedFrameCount": interpolated_count,
            "unknownFrameCount": count - known_count,
            "knownFraction": known_fraction,
            "motionFrameFraction": motion_fraction,
            "sustainedMotionFraction": sustained_fraction,
            "maximumSpeedDiagonalsPerSecond": max(interval_speeds, default=None),
            "trajectorySegmentIds": segment_ids,
        },
        "sustainedBallMotion": {
            "detected": sustained_fraction >= settings.minimum_motion_support_fraction,
            "supportFraction": sustained_fraction,
        },
        "serveLikeSequence": signal.as_dict(),
        "trajectoryGapBoundaries": {
            "precedingUnknownSeconds": preceding_gap_frames / fps,
            "followingUnknownSeconds": following_gap_frames / fps,
            "endQuietThresholdSeconds": settings.end_quiet_seconds,
        },
        "timeBetweenActivityBurstsSeconds": (
            None
            if previous_end_frame is None
            else max(0.0, (start_frame - previous_end_frame - 1) / fps)
        ),
        "playerResetBehavior": player_support,
        "audioSupport": audio,
        "deadBallHandoffAssessment": {
            "applied": True,
            "possibleDeadBallHandoff": False,
            "semanticClassificationPerformed": False,
            "rallyEvidenceQuality": rally_evidence_quality,
            "adjacentBurstWindowSeconds": settings.dead_ball_handoff_window_seconds,
            "minimumQualityMargin": settings.dead_ball_handoff_minimum_quality_margin,
            "neighborCandidateIds": [],
            "rejectedNeighborCandidateIds": [],
        },
        "confidenceComponents": {
            "serveLikeSequence": serve_quality,
            "ballActivity": activity_quality,
            "trajectoryCoverage": known_quality,
            "playerResetBoost": 0.0 if player_score is None else 0.03 * player_score,
            "audioBoost": 0.02 if audio.get("boundaryProximitySupport") is True else 0.0,
            "calibratedProbability": False,
        },
    }
    return supporting, confidence, rally_evidence_quality


def _gap_seconds(
    left: RallyPrediction,
    right: RallyPrediction,
    *,
    fps: float,
) -> float:
    if left.start_frame > right.start_frame:
        left, right = right, left
    return max(0.0, (right.start_frame - left.end_frame - 1) / fps)


def _resolve_adjacent_activity_bursts(
    predictions: tuple[RallyPrediction, ...],
    *,
    fps: float,
    settings: RallySegmentationSettings,
) -> tuple[tuple[RallyPrediction, ...], tuple[AdjacentBurstRejection, ...]]:
    """Suppress weaker near-adjacent bursts while retaining every decision."""

    ranked = sorted(
        predictions,
        key=lambda item: (
            item.rally_evidence_quality,
            item.end_frame - item.start_frame,
            item.confidence,
        ),
        reverse=True,
    )
    accepted: list[RallyPrediction] = []
    rejected: list[AdjacentBurstRejection] = []
    for candidate in ranked:
        stronger_neighbors = tuple(
            item
            for item in accepted
            if _gap_seconds(item, candidate, fps=fps) <= settings.dead_ball_handoff_window_seconds
            and item.rally_evidence_quality - candidate.rally_evidence_quality
            >= settings.dead_ball_handoff_minimum_quality_margin
        )
        if not stronger_neighbors:
            accepted.append(candidate)
            continue
        strongest = max(
            stronger_neighbors,
            key=lambda item: item.rally_evidence_quality,
        )
        quality_margin = strongest.rally_evidence_quality - candidate.rally_evidence_quality
        rejected.append(
            AdjacentBurstRejection(
                candidate=candidate,
                stronger_candidate_id=strongest.rally_id,
                stronger_evidence_quality=strongest.rally_evidence_quality,
                gap_seconds=_gap_seconds(candidate, strongest, fps=fps),
                quality_margin=quality_margin,
            )
        )

    accepted.sort(key=lambda item: item.start_frame)
    rejected_by_stronger: dict[str, list[str]] = {}
    for item in rejected:
        rejected_by_stronger.setdefault(item.stronger_candidate_id, []).append(
            item.candidate.rally_id
        )
    assessed: list[RallyPrediction] = []
    for candidate in accepted:
        raw_assessment = candidate.supporting_signals["deadBallHandoffAssessment"]
        if not isinstance(raw_assessment, dict):
            raise AssertionError("dead-ball handoff assessment must be an object")
        assessment = dict(raw_assessment)
        assessment["sourceCandidateId"] = candidate.rally_id
        assessment["neighborCandidateIds"] = [
            item.rally_id
            for item in predictions
            if item is not candidate
            and _gap_seconds(item, candidate, fps=fps) <= settings.dead_ball_handoff_window_seconds
        ]
        assessment["rejectedNeighborCandidateIds"] = rejected_by_stronger.get(
            candidate.rally_id,
            [],
        )
        assessed.append(
            replace(
                candidate,
                supporting_signals={
                    **candidate.supporting_signals,
                    "deadBallHandoffAssessment": assessment,
                },
            )
        )
    renumbered = tuple(
        replace(candidate, rally_id=f"predicted-rally-{index:05d}")
        for index, candidate in enumerate(assessed, start=1)
    )
    selected_ids = {
        source.rally_id: selected.rally_id
        for source, selected in zip(assessed, renumbered, strict=True)
    }
    linked_rejections = tuple(
        replace(
            item,
            stronger_candidate_id=selected_ids[item.stronger_candidate_id],
        )
        for item in rejected
    )
    return renumbered, linked_rejections


def segment_rallies(
    frames: tuple[RallyBallFrame, ...],
    *,
    fps: float,
    frame_width_px: int,
    frame_height_px: int,
    settings: RallySegmentationSettings,
    player_reset_scores: tuple[float | None, ...] | None = None,
    audio_transients: tuple[AudioTransientEvidence, ...] = (),
) -> RallySegmentationResult:
    """Infer rallies from ball motion; player/audio evidence may only adjust confidence."""

    if not frames:
        raise ValueError("rally segmentation requires a nonempty ball timeline")
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError("rally segmentation FPS must be finite and positive")
    if frame_width_px < 1 or frame_height_px < 1:
        raise ValueError("rally segmentation dimensions must be positive")
    if tuple(frame.frame_number for frame in frames) != tuple(range(len(frames))):
        raise ValueError("rally ball frames must form a complete zero-based timeline")
    if player_reset_scores is not None and len(player_reset_scores) != len(frames):
        raise ValueError("player reset evidence must align with every ball frame")
    diagonal = math.hypot(frame_width_px, frame_height_px)
    speeds, raw_motion, supported_motion = _frame_kinematics(
        frames,
        fps=fps,
        frame_diagonal_px=diagonal,
        settings=settings,
    )
    candidates = _serve_like_candidates(
        frames,
        speeds=speeds,
        raw_motion=raw_motion,
        fps=fps,
        frame_diagonal_px=diagonal,
        settings=settings,
    )
    minimum_restart_frames = max(1, round(settings.restart_minimum_elapsed_seconds * fps))
    restart_quiet_frames = max(1, round(settings.restart_quiet_seconds * fps))
    cooldown_frames = max(1, round(settings.minimum_between_rallies_seconds * fps))
    predictions: list[RallyPrediction] = []
    cursor = 0
    previous_end: int | None = None
    while True:
        eligible = tuple(item for item in candidates if item.candidate_frame >= cursor)
        if not eligible:
            break
        signal = eligible[0]
        start_frame = signal.refined_start_frame
        natural_end = _natural_end_frame(
            frames,
            raw_motion,
            supported_motion,
            start_frame=start_frame,
            fps=fps,
            settings=settings,
        )
        restart: ServeLikeSignal | None = None
        for candidate in eligible[1:]:
            if candidate.candidate_frame > natural_end:
                break
            if candidate.refined_start_frame - start_frame < minimum_restart_frames:
                continue
            if _has_quiet_run(
                raw_motion,
                start=start_frame,
                stop=candidate.refined_start_frame,
                required_frames=restart_quiet_frames,
            ):
                restart = candidate
                break
        if restart is not None:
            end_frame = _last_defensible_frame(
                frames,
                raw_motion,
                start=start_frame,
                quiet_start=max(start_frame, restart.refined_start_frame - restart_quiet_frames),
                tail_grace_frames=max(0, round(settings.end_tail_grace_seconds * fps)),
            )
        else:
            end_frame = natural_end
        minimum_frames = max(1, round(settings.minimum_rally_duration_seconds * fps))
        if end_frame - start_frame + 1 >= minimum_frames:
            supporting, confidence, rally_evidence_quality = _interval_support(
                frames,
                speeds,
                raw_motion,
                supported_motion,
                signal,
                start_frame=start_frame,
                end_frame=end_frame,
                previous_end_frame=previous_end,
                fps=fps,
                frame_diagonal_px=diagonal,
                player_reset_scores=player_reset_scores,
                audio_transients=audio_transients,
                settings=settings,
            )
            prediction = RallyPrediction(
                rally_id=f"rally-candidate-{len(predictions) + 1:05d}",
                start_frame=start_frame,
                end_frame=end_frame,
                start_timestamp_seconds=start_frame / fps,
                end_timestamp_seconds=end_frame / fps,
                confidence=confidence,
                supporting_signals=supporting,
                rally_evidence_quality=rally_evidence_quality,
            )
            predictions.append(prediction)
            previous_end = end_frame
        if restart is not None:
            cursor = restart.candidate_frame
        else:
            cursor = max(signal.candidate_frame + 1, end_frame + cooldown_frames)
    accepted, rejected = _resolve_adjacent_activity_bursts(
        tuple(predictions),
        fps=fps,
        settings=settings,
    )
    return RallySegmentationResult(
        rallies=accepted,
        speeds_diagonals_per_second=speeds,
        motion_supported=supported_motion,
        serve_candidate_count=len(candidates),
        rejected_adjacent_bursts=rejected,
    )
