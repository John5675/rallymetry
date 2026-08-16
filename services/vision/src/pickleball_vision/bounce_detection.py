"""Visual-first bounce candidates with optional, non-creating audio support."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from pickleball_vision.calibration import CourtCalibration
from pickleball_vision.config import BounceDetectionSettings
from pickleball_vision.court import CourtPoint, ImagePoint
from pickleball_vision.rally_segmentation import BallEvidenceStatus, RallyBallFrame

BOUNCE_DETECTION_SCHEMA_VERSION = 1


class BounceEvidenceMode(StrEnum):
    """How an emitted visual candidate reached its final confidence state."""

    VISUAL_ONLY = "VISUAL_ONLY"
    VISUAL_PLUS_AUDIO = "VISUAL_PLUS_AUDIO"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


@dataclass(frozen=True, slots=True)
class BounceAudioTransient:
    """A generic non-semantic transient mapped onto video-relative time."""

    candidate_id: str
    video_timestamp_seconds: float
    confidence: float


@dataclass(frozen=True, slots=True)
class BounceRallyInterval:
    """Optional predicted rally interval used only as sequence support."""

    rally_id: str
    start_frame: int
    end_frame: int
    confidence: float


@dataclass(frozen=True, slots=True)
class BounceCandidate:
    """One visually generated bounce candidate and its optional audio fusion."""

    bounce_id: str
    frame: int
    timestamp_seconds: float
    media_timestamp_seconds: float
    image_position: ImagePoint
    court_position: CourtPoint | None
    trajectory_status: BallEvidenceStatus
    visual_confidence: float
    audio_confidence: float
    fused_confidence: float
    matched_audio_event_id: str | None
    evidence_mode: BounceEvidenceMode
    accepted_vision_only: bool
    accepted_fused: bool
    supporting_signals: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "bounceId": self.bounce_id,
            "timestamp": self.timestamp_seconds,
            "mediaTimestamp": self.media_timestamp_seconds,
            "frame": self.frame,
            "imagePosition": {
                **self.image_position.as_dict(),
                "coordinateSystem": "source_frame_pixels_top_left",
                "trajectoryStatus": self.trajectory_status.value,
            },
            "courtPosition": (
                {
                    **self.court_position.as_dict(),
                    "coordinateSystem": "canonical_pickleball_court",
                    "projectionJustification": (
                        "visual_plane_contact_plausibility_and_projected_court_inclusion"
                    ),
                }
                if self.court_position is not None
                else None
            ),
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
class BounceDetectionResult:
    """Visual candidate set after temporal suppression and optional fusion."""

    candidates: tuple[BounceCandidate, ...]
    raw_visual_candidate_count: int
    suppressed_visual_candidate_count: int
    matched_audio_candidate_count: int


@dataclass(frozen=True, slots=True)
class _VisualCandidate:
    frame: int
    image_position: ImagePoint
    trajectory_status: BallEvidenceStatus
    visual_confidence: float
    court_position: CourtPoint | None
    supporting_signals: dict[str, object]


def _slope(
    frames: tuple[RallyBallFrame, ...],
    indices: tuple[int, ...],
    *,
    coordinate: str,
) -> float:
    times = np.asarray([frames[index].timestamp_seconds for index in indices], dtype=np.float64)
    coordinate_values: list[float] = []
    for index in indices:
        point = frames[index].point
        if point is None:
            raise AssertionError("slope indices must contain known trajectory points")
        coordinate_values.append(point.x_px if coordinate == "x" else point.y_px)
    values = np.asarray(coordinate_values, dtype=np.float64)
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


def _projected_court_polygon(calibration: CourtCalibration) -> np.ndarray:
    points = (
        CourtPoint(0.0, 0.0),
        CourtPoint(calibration.court.width_m, 0.0),
        CourtPoint(calibration.court.width_m, calibration.court.length_m),
        CourtPoint(0.0, calibration.court.length_m),
    )
    return np.asarray(
        [
            (calibration.court_to_image(point).x_px, calibration.court_to_image(point).y_px)
            for point in points
        ],
        dtype=np.float32,
    )


def _inside_projected_court(point: ImagePoint, polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, (point.x_px, point.y_px), False) >= 0


def _rally_support(
    frame: int,
    rallies: tuple[BounceRallyInterval, ...],
) -> BounceRallyInterval | None:
    return next((item for item in rallies if item.start_frame <= frame <= item.end_frame), None)


def _candidate_at_frame(
    frames: tuple[RallyBallFrame, ...],
    *,
    frame: int,
    window_frames: int,
    frame_diagonal_px: float,
    calibration: CourtCalibration,
    court_polygon: np.ndarray,
    rallies: tuple[BounceRallyInterval, ...],
    settings: BounceDetectionSettings,
) -> _VisualCandidate | None:
    center = frames[frame]
    if center.point is None or center.segment_id is None:
        return None
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
        return None
    before_fit = (*before, frame)
    after_fit = (frame, *after)
    before_x_slope = _slope(frames, before_fit, coordinate="x")
    before_y_slope = _slope(frames, before_fit, coordinate="y")
    after_x_slope = _slope(frames, after_fit, coordinate="x")
    after_y_slope = _slope(frames, after_fit, coordinate="y")
    minimum_speed_px_s = settings.minimum_vertical_speed_diagonals_per_second * frame_diagonal_px
    if before_y_slope < minimum_speed_px_s or after_y_slope > -minimum_speed_px_s:
        return None
    reversal = (before_y_slope - after_y_slope) / frame_diagonal_px
    if reversal < settings.minimum_vertical_reversal_diagonals_per_second:
        return None
    before_y: list[float] = []
    after_y: list[float] = []
    for index in before:
        point = frames[index].point
        if point is None:
            raise AssertionError("before indices must contain known trajectory points")
        before_y.append(point.y_px)
    for index in after:
        point = frames[index].point
        if point is None:
            raise AssertionError("after indices must contain known trajectory points")
        after_y.append(point.y_px)
    prominence = min(center.point.y_px - min(before_y), center.point.y_px - min(after_y))
    prominence_fraction = prominence / frame_diagonal_px
    if prominence_fraction < settings.minimum_shape_prominence_diagonal_fraction:
        return None
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
        return None
    observed_fraction = sum(
        item.status is BallEvidenceStatus.OBSERVED for item in same_segment
    ) / len(same_segment)
    inside_court_image = _inside_projected_court(center.point, court_polygon)
    rally = _rally_support(frame, rallies)
    reversal_quality = min(
        1.0,
        reversal / (settings.minimum_vertical_reversal_diagonals_per_second * 3),
    )
    prominence_quality = min(
        1.0,
        prominence_fraction / (settings.minimum_shape_prominence_diagonal_fraction * 3),
    )
    speed_quality = min(
        1.0,
        min(before_y_slope, -after_y_slope) / (minimum_speed_px_s * 3),
    )
    center_observation_quality = 1.0 if center.status is BallEvidenceStatus.OBSERVED else 0.55
    visual_confidence = (
        0.27 * reversal_quality
        + 0.18 * prominence_quality
        + 0.13 * speed_quality
        + 0.17 * continuity_fraction
        + 0.10 * observed_fraction
        + 0.10 * float(inside_court_image)
        + 0.05 * center_observation_quality
    )
    if rally is not None:
        visual_confidence += settings.rally_sequence_confidence_boost * rally.confidence
    visual_confidence = min(1.0, visual_confidence)
    if visual_confidence < settings.minimum_visual_candidate_confidence:
        return None
    court_position = None
    projection_applied = False
    if (
        inside_court_image
        and visual_confidence >= settings.plane_projection_minimum_visual_confidence
    ):
        projected = calibration.image_to_court(center.point)
        if (
            math.isfinite(projected.x_m)
            and math.isfinite(projected.y_m)
            and 0 <= projected.x_m <= calibration.court.width_m
            and 0 <= projected.y_m <= calibration.court.length_m
        ):
            court_position = projected
            projection_applied = True
    signals: dict[str, object] = {
        "trajectoryDirectionChange": {
            "beforeVelocityPixelsPerSecond": {
                "x": before_x_slope,
                "y": before_y_slope,
            },
            "afterVelocityPixelsPerSecond": {
                "x": after_x_slope,
                "y": after_y_slope,
            },
            "directionChangeDegrees": _angle_degrees(
                (before_x_slope, before_y_slope),
                (after_x_slope, after_y_slope),
            ),
        },
        "imageSpaceVerticalReversal": {
            "detected": True,
            "strengthDiagonalsPerSecond": reversal,
            "beforeSlopeDiagonalsPerSecond": before_y_slope / frame_diagonal_px,
            "afterSlopeDiagonalsPerSecond": after_y_slope / frame_diagonal_px,
        },
        "trajectoryContinuity": {
            "segmentId": center.segment_id,
            "windowFrameCount": window_count,
            "knownSameSegmentFrameCount": len(same_segment),
            "knownFraction": continuity_fraction,
            "observedFractionAmongKnown": observed_fraction,
        },
        "localTrajectoryShape": {
            "centerIsImageSpaceLocalMaximum": True,
            "prominencePixels": prominence,
            "prominenceDiagonalFraction": prominence_fraction,
        },
        "courtSurfaceProximity": {
            "insideProjectedCourtImagePolygon": inside_court_image,
            "homographyUsedForImageInclusionTest": False,
            "courtGeometryProjectedIntoImage": True,
        },
        "plausibleRallySequence": {
            "available": bool(rallies),
            "insidePredictedRally": rally is not None,
            "rallyId": rally.rally_id if rally is not None else None,
            "confidenceBoost": (
                settings.rally_sequence_confidence_boost * rally.confidence
                if rally is not None
                else 0.0
            ),
            "canCreateBounce": False,
        },
        "courtProjection": {
            "applied": projection_applied,
            "appliedOnlyAfterVisualPlaneContactPlausibility": True,
            "minimumVisualConfidence": (settings.plane_projection_minimum_visual_confidence),
            "airborneBallProjected": False,
            "true3dPositionInferred": False,
            "lineCallInferred": False,
        },
        "visualConfidenceComponents": {
            "verticalReversal": 0.27 * reversal_quality,
            "localShape": 0.18 * prominence_quality,
            "verticalSpeed": 0.13 * speed_quality,
            "continuity": 0.17 * continuity_fraction,
            "observedSupport": 0.10 * observed_fraction,
            "projectedCourtImageInclusion": 0.10 * float(inside_court_image),
            "centerObservation": 0.05 * center_observation_quality,
            "rallySequenceBoost": (
                settings.rally_sequence_confidence_boost * rally.confidence
                if rally is not None
                else 0.0
            ),
            "calibratedProbability": False,
        },
    }
    return _VisualCandidate(
        frame=frame,
        image_position=center.point,
        trajectory_status=center.status,
        visual_confidence=visual_confidence,
        court_position=court_position,
        supporting_signals=signals,
    )


def _suppress_nearby_candidates(
    candidates: tuple[_VisualCandidate, ...],
    *,
    minimum_frames: int,
) -> tuple[_VisualCandidate, ...]:
    accepted: list[_VisualCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.visual_confidence, reverse=True):
        if any(abs(candidate.frame - item.frame) < minimum_frames for item in accepted):
            continue
        accepted.append(candidate)
    return tuple(sorted(accepted, key=lambda item: item.frame))


def _audio_matches(
    candidates: tuple[_VisualCandidate, ...],
    audio: tuple[BounceAudioTransient, ...],
    *,
    fps: float,
    tolerance_seconds: float,
) -> dict[int, tuple[BounceAudioTransient, float, float]]:
    proposals: list[tuple[float, float, int, BounceAudioTransient]] = []
    for candidate_index, candidate in enumerate(candidates):
        candidate_time = candidate.frame / fps
        for transient in audio:
            delta = abs(transient.video_timestamp_seconds - candidate_time)
            if delta <= tolerance_seconds:
                proposals.append((delta, -transient.confidence, candidate_index, transient))
    proposals.sort(key=lambda item: item[:2])
    used_candidates: set[int] = set()
    used_audio: set[str] = set()
    matches: dict[int, tuple[BounceAudioTransient, float, float]] = {}
    for delta, _negative_confidence, candidate_index, transient in proposals:
        if candidate_index in used_candidates or transient.candidate_id in used_audio:
            continue
        timing_quality = 1.0 - delta / tolerance_seconds if tolerance_seconds > 0 else 1.0
        audio_confidence = transient.confidence * (0.5 + 0.5 * timing_quality)
        matches[candidate_index] = (transient, audio_confidence, delta)
        used_candidates.add(candidate_index)
        used_audio.add(transient.candidate_id)
    return matches


def detect_bounce_candidates(
    frames: tuple[RallyBallFrame, ...],
    *,
    fps: float,
    frame_width_px: int,
    frame_height_px: int,
    calibration: CourtCalibration,
    settings: BounceDetectionSettings,
    audio_transients: tuple[BounceAudioTransient, ...] = (),
    rallies: tuple[BounceRallyInterval, ...] = (),
    video_start_time_seconds: float = 0.0,
    fusion_tolerance_ms: float = 90.0,
) -> BounceDetectionResult:
    """Generate candidates visually, then optionally fuse one nearby transient."""

    if not frames:
        raise ValueError("bounce detection requires a nonempty trajectory timeline")
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError("bounce detection FPS must be finite and positive")
    if frame_width_px < 1 or frame_height_px < 1:
        raise ValueError("bounce detection dimensions must be positive")
    if not math.isfinite(fusion_tolerance_ms) or fusion_tolerance_ms <= 0:
        raise ValueError("fusion tolerance must be finite and positive")
    if tuple(item.frame_number for item in frames) != tuple(range(len(frames))):
        raise ValueError("bounce trajectory frames must form a complete zero-based timeline")
    diagonal = math.hypot(frame_width_px, frame_height_px)
    window_frames = max(
        settings.minimum_observations_each_side,
        round(settings.trajectory_window_seconds * fps),
    )
    polygon = _projected_court_polygon(calibration)
    raw = tuple(
        candidate
        for frame in range(window_frames, len(frames) - window_frames)
        if (
            candidate := _candidate_at_frame(
                frames,
                frame=frame,
                window_frames=window_frames,
                frame_diagonal_px=diagonal,
                calibration=calibration,
                court_polygon=polygon,
                rallies=rallies,
                settings=settings,
            )
        )
        is not None
    )
    visual = _suppress_nearby_candidates(
        raw,
        minimum_frames=max(1, round(settings.minimum_between_bounces_seconds * fps)),
    )
    tolerance_seconds = fusion_tolerance_ms / 1000.0
    audio_matches = _audio_matches(
        visual,
        audio_transients,
        fps=fps,
        tolerance_seconds=tolerance_seconds,
    )
    candidates: list[BounceCandidate] = []
    for index, candidate in enumerate(visual):
        match = audio_matches.get(index)
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
            BounceEvidenceMode.LOW_CONFIDENCE
            if not accepted_fused
            else (
                BounceEvidenceMode.VISUAL_PLUS_AUDIO
                if transient is not None
                else BounceEvidenceMode.VISUAL_ONLY
            )
        )
        supporting = {
            **candidate.supporting_signals,
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
                "canCreateBounce": False,
                "neighboringCourtSoundMayBePresent": True,
            },
        }
        candidates.append(
            BounceCandidate(
                bounce_id=f"bounce-candidate-{len(candidates) + 1:06d}",
                frame=candidate.frame,
                timestamp_seconds=candidate.frame / fps,
                media_timestamp_seconds=video_start_time_seconds + candidate.frame / fps,
                image_position=candidate.image_position,
                court_position=candidate.court_position,
                trajectory_status=candidate.trajectory_status,
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
    return BounceDetectionResult(
        candidates=tuple(candidates),
        raw_visual_candidate_count=len(raw),
        suppressed_visual_candidate_count=len(raw) - len(visual),
        matched_audio_candidate_count=len(audio_matches),
    )
