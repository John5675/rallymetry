"""Conservative image-space reconstruction of the primary-match ball trajectory."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import pairwise

import cv2
import numpy as np

from pickleball_vision.calibration import CourtCalibration
from pickleball_vision.config import BallTrackingSettings
from pickleball_vision.court import CourtPoint, ImagePoint
from pickleball_vision.person_detection import BoundingBox

BALL_TRACKING_SCHEMA_VERSION = 1


class BallTrajectoryStatus(StrEnum):
    """Evidence status for one source frame on the reconstructed timeline."""

    OBSERVED = "OBSERVED"
    INTERPOLATED = "INTERPOLATED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class BallTrackingCandidate:
    """One immutable raw detection enriched with image-space association evidence."""

    detection_id: str
    frame_number: int
    timestamp_s: float
    bounding_box: BoundingBox
    confidence: float
    image_point: ImagePoint
    primary_court_relevance: float
    temporal_support: float = 0.0

    def __post_init__(self) -> None:
        if self.frame_number < 0:
            raise ValueError("ball candidate frame number must be non-negative")
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("ball candidate timestamp must be finite and non-negative")
        for value, field_name in (
            (self.confidence, "confidence"),
            (self.primary_court_relevance, "primary court relevance"),
            (self.temporal_support, "temporal support"),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"ball candidate {field_name} must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class CourtImageEnvelope:
    """Projected court outline plus asymmetric image margins for airborne relevance."""

    polygon: tuple[ImagePoint, ...]
    frame_width_px: int
    frame_height_px: int
    side_margin_px: float
    air_margin_px: float
    bottom_margin_px: float

    def relevance(self, point: ImagePoint) -> float:
        """Score image relevance without interpreting the point as court-plane contact."""

        polygon = np.asarray(
            [(item.x_px, item.y_px) for item in self.polygon],
            dtype=np.float32,
        )
        if cv2.pointPolygonTest(polygon, (point.x_px, point.y_px), False) >= 0:
            return 1.0
        x_values = tuple(item.x_px for item in self.polygon)
        y_values = tuple(item.y_px for item in self.polygon)
        left, right = min(x_values), max(x_values)
        top, bottom = min(y_values), max(y_values)
        horizontal_distance = max(left - point.x_px, 0.0, point.x_px - right)
        vertical_distance = top - point.y_px if point.y_px < top else point.y_px - bottom
        vertical_margin = self.air_margin_px if point.y_px < top else self.bottom_margin_px
        if horizontal_distance > self.side_margin_px or vertical_distance > vertical_margin:
            return 0.0
        normalized_horizontal = horizontal_distance / self.side_margin_px
        normalized_vertical = max(0.0, vertical_distance) / vertical_margin
        edge_distance = max(normalized_horizontal, normalized_vertical)
        return max(0.45, 0.75 - 0.30 * edge_distance)

    def as_dict(self) -> dict[str, object]:
        return {
            "projected_court_polygon": [point.as_dict() for point in self.polygon],
            "frame_width_px": self.frame_width_px,
            "frame_height_px": self.frame_height_px,
            "side_margin_px": self.side_margin_px,
            "air_margin_px": self.air_margin_px,
            "bottom_margin_px": self.bottom_margin_px,
            "usage": "image_space_relevance_only",
            "airborne_ball_homography_projection": False,
        }


@dataclass(frozen=True, slots=True)
class BallTrajectoryPoint:
    """One explicit observed, interpolated, or unknown source-frame state."""

    frame_number: int
    timestamp_s: float
    status: BallTrajectoryStatus
    segment_id: str | None
    source_detection_id: str | None
    raw_image_point: ImagePoint | None
    interpolated_image_point: ImagePoint | None
    smoothed_image_point: ImagePoint | None
    confidence: float | None
    detection_confidence: float | None
    primary_court_relevance: float | None
    temporal_support: float | None
    candidate_count: int
    rejected_detection_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_number": self.frame_number,
            "timestamp_s": self.timestamp_s,
            "status": self.status.value,
            "segment_id": self.segment_id,
            "source_detection_id": self.source_detection_id,
            "raw_image_point_px": (
                self.raw_image_point.as_dict() if self.raw_image_point is not None else None
            ),
            "interpolated_image_point_px": (
                self.interpolated_image_point.as_dict()
                if self.interpolated_image_point is not None
                else None
            ),
            "smoothed_image_point_px": (
                self.smoothed_image_point.as_dict()
                if self.smoothed_image_point is not None
                else None
            ),
            "confidence": self.confidence,
            "detection_confidence": self.detection_confidence,
            "primary_court_relevance": self.primary_court_relevance,
            "temporal_support": self.temporal_support,
            "candidate_count": self.candidate_count,
            "rejected_detection_ids": list(self.rejected_detection_ids),
        }


@dataclass(frozen=True, slots=True)
class BallTrajectory:
    """Complete primary-match frame timeline and association accounting."""

    frames: tuple[BallTrajectoryPoint, ...]
    total_candidate_count: int
    rejected_candidate_count: int
    segment_count: int


@dataclass(frozen=True, slots=True)
class _SelectedObservation:
    candidate: BallTrackingCandidate
    association_confidence: float


def ball_box_center(box: BoundingBox) -> ImagePoint:
    """Return the raw image-space center of a detected ball box."""

    return ImagePoint(
        x_px=(box.left_px + box.right_px) / 2,
        y_px=(box.top_px + box.bottom_px) / 2,
    )


def build_court_image_envelope(
    calibration: CourtCalibration,
    *,
    frame_width_px: int,
    frame_height_px: int,
    settings: BallTrackingSettings,
) -> CourtImageEnvelope:
    """Project known court corners into the image; never project a ball point."""

    if (
        calibration.source.frame_width_px != frame_width_px
        or calibration.source.frame_height_px != frame_height_px
    ):
        raise ValueError("court calibration dimensions do not match the ball-tracking video")
    court = calibration.court
    polygon = tuple(
        calibration.court_to_image(point)
        for point in (
            CourtPoint(0.0, 0.0),
            CourtPoint(court.width_m, 0.0),
            CourtPoint(court.width_m, court.length_m),
            CourtPoint(0.0, court.length_m),
        )
    )
    return CourtImageEnvelope(
        polygon=polygon,
        frame_width_px=frame_width_px,
        frame_height_px=frame_height_px,
        side_margin_px=frame_width_px * settings.primary_court_side_margin_fraction,
        air_margin_px=frame_height_px * settings.primary_court_air_margin_fraction,
        bottom_margin_px=frame_height_px * settings.primary_court_bottom_margin_fraction,
    )


def _distance(left: ImagePoint, right: ImagePoint) -> float:
    return math.hypot(left.x_px - right.x_px, left.y_px - right.y_px)


def _temporal_support(
    candidate: BallTrackingCandidate,
    candidates_by_frame: Sequence[tuple[BallTrackingCandidate, ...]],
    *,
    maximum_gap_frames: int,
    fps: float,
    frame_diagonal_px: float,
    settings: BallTrackingSettings,
) -> float:
    supported_directions = 0
    for direction in (-1, 1):
        for offset in range(1, maximum_gap_frames + 1):
            frame_number = candidate.frame_number + direction * offset
            if not 0 <= frame_number < len(candidates_by_frame):
                break
            frame_elapsed_s = offset / fps
            gate = frame_diagonal_px * (
                settings.association_base_gate_diagonal_fraction
                + settings.maximum_speed_diagonals_per_second * frame_elapsed_s
            )
            if any(
                _distance(candidate.image_point, other.image_point) <= gate
                for other in candidates_by_frame[frame_number]
            ):
                supported_directions += 1
                break
    return supported_directions / 2


def add_temporal_support(
    candidates_by_frame: Sequence[tuple[BallTrackingCandidate, ...]],
    *,
    fps: float,
    frame_diagonal_px: float,
    settings: BallTrackingSettings,
) -> tuple[tuple[BallTrackingCandidate, ...], ...]:
    """Add bounded bidirectional persistence evidence without mutating candidates."""

    maximum_gap_frames = max(1, math.floor(settings.max_association_gap_seconds * fps))
    enriched: list[tuple[BallTrackingCandidate, ...]] = []
    for frame in candidates_by_frame:
        enriched.append(
            tuple(
                replace(
                    candidate,
                    temporal_support=_temporal_support(
                        candidate,
                        candidates_by_frame,
                        maximum_gap_frames=maximum_gap_frames,
                        fps=fps,
                        frame_diagonal_px=frame_diagonal_px,
                        settings=settings,
                    ),
                )
                for candidate in frame
            )
        )
    return tuple(enriched)


def _start_score(candidate: BallTrackingCandidate) -> float:
    return (
        0.45 * candidate.primary_court_relevance
        + 0.35 * candidate.temporal_support
        + 0.20 * candidate.confidence
    )


def _association_score(
    candidate: BallTrackingCandidate,
    *,
    previous: _SelectedObservation | None,
    last: _SelectedObservation,
    frame_diagonal_px: float,
    settings: BallTrackingSettings,
) -> float | None:
    elapsed_s = candidate.timestamp_s - last.candidate.timestamp_s
    if elapsed_s <= 0:
        return None
    direct_speed = _distance(candidate.image_point, last.candidate.image_point) / (
        elapsed_s * frame_diagonal_px
    )
    if direct_speed > settings.maximum_speed_diagonals_per_second:
        return None

    predicted = last.candidate.image_point
    acceleration_plausibility = 1.0
    if previous is not None:
        previous_elapsed_s = last.candidate.timestamp_s - previous.candidate.timestamp_s
        if previous_elapsed_s > 0:
            velocity_x = (
                last.candidate.image_point.x_px - previous.candidate.image_point.x_px
            ) / previous_elapsed_s
            velocity_y = (
                last.candidate.image_point.y_px - previous.candidate.image_point.y_px
            ) / previous_elapsed_s
            predicted = ImagePoint(
                last.candidate.image_point.x_px + velocity_x * elapsed_s,
                last.candidate.image_point.y_px + velocity_y * elapsed_s,
            )
            next_velocity_x = (
                candidate.image_point.x_px - last.candidate.image_point.x_px
            ) / elapsed_s
            next_velocity_y = (
                candidate.image_point.y_px - last.candidate.image_point.y_px
            ) / elapsed_s
            acceleration = math.hypot(
                next_velocity_x - velocity_x,
                next_velocity_y - velocity_y,
            ) / (elapsed_s * frame_diagonal_px)
            if acceleration > settings.maximum_acceleration_diagonals_per_second_squared:
                return None
            acceleration_plausibility = 1 - (
                acceleration / settings.maximum_acceleration_diagonals_per_second_squared
            )

    gate_px = frame_diagonal_px * (
        settings.association_base_gate_diagonal_fraction
        + settings.maximum_speed_diagonals_per_second * elapsed_s
    )
    prediction_error = _distance(candidate.image_point, predicted)
    if prediction_error > gate_px:
        return None
    continuity = 1 - prediction_error / gate_px
    score = (
        0.42 * continuity
        + 0.18 * acceleration_plausibility
        + 0.18 * candidate.primary_court_relevance
        + 0.12 * candidate.confidence
        + 0.10 * candidate.temporal_support
    )
    return score if score >= settings.minimum_association_score else None


def _associate_segments(
    candidates_by_frame: Sequence[tuple[BallTrackingCandidate, ...]],
    *,
    fps: float,
    frame_diagonal_px: float,
    settings: BallTrackingSettings,
) -> tuple[tuple[_SelectedObservation, ...], ...]:
    maximum_gap_frames = max(1, math.floor(settings.max_association_gap_seconds * fps))
    accepted_segments: list[tuple[_SelectedObservation, ...]] = []
    active: list[_SelectedObservation] = []

    def finish_active() -> None:
        nonlocal active
        if len(active) >= settings.minimum_segment_observations:
            accepted_segments.append(tuple(active))
        active = []

    for frame_number, candidates in enumerate(candidates_by_frame):
        if active:
            previous = active[-2] if len(active) >= 2 else None
            scores = tuple(
                (
                    _association_score(
                        candidate,
                        previous=previous,
                        last=active[-1],
                        frame_diagonal_px=frame_diagonal_px,
                        settings=settings,
                    ),
                    candidate,
                )
                for candidate in candidates
            )
            eligible = tuple((score, item) for score, item in scores if score is not None)
            if eligible:
                score, selected = max(
                    eligible,
                    key=lambda pair: (
                        pair[0],
                        pair[1].confidence,
                        pair[1].detection_id,
                    ),
                )
                assert score is not None
                active.append(_SelectedObservation(selected, score))
                continue
            if frame_number - active[-1].candidate.frame_number <= maximum_gap_frames:
                continue
            finish_active()

        eligible_starts = tuple(
            (score, candidate)
            for candidate in candidates
            if candidate.primary_court_relevance > 0
            and (score := _start_score(candidate)) >= settings.minimum_start_score
        )
        if eligible_starts:
            score, selected = max(
                eligible_starts,
                key=lambda pair: (pair[0], pair[1].confidence, pair[1].detection_id),
            )
            active.append(_SelectedObservation(selected, score))
    finish_active()
    return tuple(accepted_segments)


def _bounded_smoothed_point(
    index: int,
    frames: Sequence[BallTrajectoryPoint],
    *,
    frame_diagonal_px: float,
    settings: BallTrackingSettings,
) -> ImagePoint:
    current = frames[index]
    base = current.raw_image_point or current.interpolated_image_point
    assert base is not None
    radius = settings.smoothing_window_frames // 2
    start = max(0, index - radius)
    stop = min(len(frames), index + radius + 1)
    neighborhood: list[tuple[ImagePoint, float]] = []
    for candidate_index, candidate in enumerate(frames[start:stop], start=start):
        if (
            candidate.segment_id != current.segment_id
            or candidate.status is BallTrajectoryStatus.UNKNOWN
        ):
            continue
        between_start = min(index, candidate_index)
        between_stop = max(index, candidate_index) + 1
        if any(
            item.status is BallTrajectoryStatus.UNKNOWN
            for item in frames[between_start:between_stop]
        ):
            continue
        point = candidate.raw_image_point or candidate.interpolated_image_point
        if point is not None:
            neighborhood.append((point, candidate.confidence or 0.0))
    total_weight = sum(max(weight, 0.05) for _, weight in neighborhood)
    target = ImagePoint(
        sum(point.x_px * max(weight, 0.05) for point, weight in neighborhood) / total_weight,
        sum(point.y_px * max(weight, 0.05) for point, weight in neighborhood) / total_weight,
    )
    adjustment_x = target.x_px - base.x_px
    adjustment_y = target.y_px - base.y_px
    adjustment = math.hypot(adjustment_x, adjustment_y)
    maximum = frame_diagonal_px * settings.maximum_smoothing_adjustment_diagonal_fraction
    if adjustment > maximum:
        scale = maximum / adjustment
        target = ImagePoint(base.x_px + adjustment_x * scale, base.y_px + adjustment_y * scale)
    return target


def reconstruct_ball_trajectory(
    candidates_by_frame: Sequence[tuple[BallTrackingCandidate, ...]],
    *,
    fps: float,
    frame_width_px: int,
    frame_height_px: int,
    settings: BallTrackingSettings,
) -> BallTrajectory:
    """Associate candidates, preserve short interpolation, and leave long gaps unknown."""

    if fps <= 0 or not math.isfinite(fps):
        raise ValueError("ball trajectory FPS must be finite and positive")
    if frame_width_px < 1 or frame_height_px < 1:
        raise ValueError("ball trajectory frame dimensions must be positive")
    frame_diagonal_px = math.hypot(frame_width_px, frame_height_px)
    candidates = add_temporal_support(
        candidates_by_frame,
        fps=fps,
        frame_diagonal_px=frame_diagonal_px,
        settings=settings,
    )
    segments = _associate_segments(
        candidates,
        fps=fps,
        frame_diagonal_px=frame_diagonal_px,
        settings=settings,
    )
    selected: dict[int, tuple[str, _SelectedObservation]] = {}
    for index, segment in enumerate(segments, start=1):
        segment_id = f"ball-segment-{index:05d}"
        for observation in segment:
            selected[observation.candidate.frame_number] = (segment_id, observation)

    frames: list[BallTrajectoryPoint] = []
    for frame_number, frame_candidates in enumerate(candidates):
        selected_value = selected.get(frame_number)
        rejected_ids = tuple(
            item.detection_id
            for item in frame_candidates
            if selected_value is None
            or item.detection_id != selected_value[1].candidate.detection_id
        )
        if selected_value is None:
            frames.append(
                BallTrajectoryPoint(
                    frame_number=frame_number,
                    timestamp_s=frame_number / fps,
                    status=BallTrajectoryStatus.UNKNOWN,
                    segment_id=None,
                    source_detection_id=None,
                    raw_image_point=None,
                    interpolated_image_point=None,
                    smoothed_image_point=None,
                    confidence=None,
                    detection_confidence=None,
                    primary_court_relevance=None,
                    temporal_support=None,
                    candidate_count=len(frame_candidates),
                    rejected_detection_ids=rejected_ids,
                )
            )
            continue
        segment_id, observation = selected_value
        item = observation.candidate
        frames.append(
            BallTrajectoryPoint(
                frame_number=frame_number,
                timestamp_s=item.timestamp_s,
                status=BallTrajectoryStatus.OBSERVED,
                segment_id=segment_id,
                source_detection_id=item.detection_id,
                raw_image_point=item.image_point,
                interpolated_image_point=None,
                smoothed_image_point=None,
                confidence=observation.association_confidence,
                detection_confidence=item.confidence,
                primary_court_relevance=item.primary_court_relevance,
                temporal_support=item.temporal_support,
                candidate_count=len(frame_candidates),
                rejected_detection_ids=rejected_ids,
            )
        )

    maximum_interpolation_gap_frames = max(
        1,
        math.floor(settings.max_interpolation_gap_seconds * fps),
    )
    for left, right in pairwise(sorted(selected)):
        left_value = selected[left]
        right_value = selected[right]
        missing_count = right - left - 1
        if (
            missing_count < 1
            or missing_count > maximum_interpolation_gap_frames
            or left_value[0] != right_value[0]
        ):
            continue
        left_observation = left_value[1]
        right_observation = right_value[1]
        for frame_number in range(left + 1, right):
            fraction = (frame_number - left) / (right - left)
            left_point = left_observation.candidate.image_point
            right_point = right_observation.candidate.image_point
            interpolated = ImagePoint(
                x_px=left_point.x_px + fraction * (right_point.x_px - left_point.x_px),
                y_px=left_point.y_px + fraction * (right_point.y_px - left_point.y_px),
            )
            endpoint_confidence = min(
                left_observation.association_confidence,
                right_observation.association_confidence,
            )
            confidence = endpoint_confidence * (0.85**missing_count)
            frames[frame_number] = replace(
                frames[frame_number],
                status=BallTrajectoryStatus.INTERPOLATED,
                segment_id=left_value[0],
                interpolated_image_point=interpolated,
                confidence=confidence,
            )

    frames = [
        replace(
            frame,
            smoothed_image_point=_bounded_smoothed_point(
                index,
                frames,
                frame_diagonal_px=frame_diagonal_px,
                settings=settings,
            ),
        )
        if frame.status is not BallTrajectoryStatus.UNKNOWN
        else frame
        for index, frame in enumerate(frames)
    ]
    total_candidate_count = sum(len(frame) for frame in candidates)
    observed_count = sum(frame.status is BallTrajectoryStatus.OBSERVED for frame in frames)
    return BallTrajectory(
        frames=tuple(frames),
        total_candidate_count=total_candidate_count,
        rejected_candidate_count=total_candidate_count - observed_count,
        segment_count=len(segments),
    )


def trajectory_summary(trajectory: BallTrajectory, *, fps: float) -> dict[str, object]:
    """Calculate transparent coverage and missing-interval metrics."""

    frame_count = len(trajectory.frames)
    observed_count = sum(
        frame.status is BallTrajectoryStatus.OBSERVED for frame in trajectory.frames
    )
    interpolated_count = sum(
        frame.status is BallTrajectoryStatus.INTERPOLATED for frame in trajectory.frames
    )
    known_count = observed_count + interpolated_count
    longest_start: int | None = None
    longest_count = 0
    active_start: int | None = None
    for frame in trajectory.frames:
        if frame.status is BallTrajectoryStatus.UNKNOWN:
            if active_start is None:
                active_start = frame.frame_number
            continue
        if active_start is not None:
            count = frame.frame_number - active_start
            if count > longest_count:
                longest_start, longest_count = active_start, count
            active_start = None
    if active_start is not None:
        count = frame_count - active_start
        if count > longest_count:
            longest_start, longest_count = active_start, count
    longest_end = longest_start + longest_count - 1 if longest_start is not None else None
    return {
        "frames_processed": frame_count,
        "observed_frames": observed_count,
        "interpolated_frames": interpolated_count,
        "unknown_frames": frame_count - known_count,
        "trajectory_coverage": known_count / frame_count if frame_count else 0.0,
        "observed_coverage": observed_count / frame_count if frame_count else 0.0,
        "interpolated_fraction": interpolated_count / known_count if known_count else 0.0,
        "longest_missing_interval": {
            "start_frame": longest_start,
            "end_frame": longest_end,
            "frame_count": longest_count,
            "duration_seconds": longest_count / fps if fps > 0 else None,
        },
        "candidate_count": trajectory.total_candidate_count,
        "candidate_rejection_count": trajectory.rejected_candidate_count,
        "segment_count": trajectory.segment_count,
    }
