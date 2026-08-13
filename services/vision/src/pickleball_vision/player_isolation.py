"""Court-aware primary-player candidates without persistent identity tracking."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import cast

from pickleball_vision.calibration import CourtCalibration
from pickleball_vision.config import PlayerIsolationSettings
from pickleball_vision.court import CourtDimensions, CourtPoint, ImagePoint
from pickleball_vision.errors import (
    InvalidCalibrationError,
    PlayerAssignmentIoError,
    PlayerIsolationInputError,
)
from pickleball_vision.person_detection import PersonDetection, PersonDetectionRun
from pickleball_vision.video import VideoMetadata

PLAYER_CANDIDATE_SCHEMA_VERSION = 1
PLAYER_ASSIGNMENT_SCHEMA_VERSION = 1
GROUND_CLIP_TOLERANCE_PX = 1.0
ASSOCIATION_BASE_DISTANCE_M = 0.35


class CourtRegionState(StrEnum):
    """Primary-court membership of an estimated ground-contact point."""

    INSIDE = "inside"
    NEAR = "near"
    OUTSIDE = "outside"
    AMBIGUOUS = "ambiguous"


class CourtSide(StrEnum):
    """Camera-relative side of the calibrated net."""

    NEAR = "near_side"
    FAR = "far_side"
    AMBIGUOUS = "ambiguous"


class GroundProjectionStatus(StrEnum):
    """Whether bottom-center was appropriate for court-plane projection."""

    PROJECTED = "projected_bottom_center_estimate"
    FRAME_EDGE_CLIPPED = "frame_edge_clipped"
    TRANSFORM_INVALID = "transform_invalid"


class LogicalPlayerRole(StrEnum):
    """Human-owned match identities, independent of model/candidate identifiers."""

    ME = "ME"
    PARTNER = "PARTNER"
    OPPONENT_1 = "OPPONENT_1"
    OPPONENT_2 = "OPPONENT_2"


LOGICAL_PLAYER_ROLES = tuple(LogicalPlayerRole)


def ground_contact_point(detection: PersonDetection) -> ImagePoint:
    """Estimate shoe/court contact as the person box's bottom-center point."""

    box = detection.bounding_box
    return ImagePoint(
        x_px=(box.left_px + box.right_px) / 2,
        y_px=box.bottom_px,
    )


@dataclass(frozen=True, slots=True)
class GroundContactAssessment:
    """Derived court geometry for one immutable raw person observation."""

    image_point: ImagePoint
    court_point: CourtPoint | None
    projection_status: GroundProjectionStatus
    region_state: CourtRegionState
    region_confidence: float
    region_boundary_ambiguous: bool
    side: CourtSide
    side_confidence: float

    def as_dict(self) -> dict[str, object]:
        return {
            "method": "bounding_box_bottom_center",
            "image_point": self.image_point.as_dict(),
            "court_point": self.court_point.as_dict() if self.court_point is not None else None,
            "projection_status": self.projection_status.value,
            "court_region": self.region_state.value,
            "court_region_confidence": self.region_confidence,
            "court_region_boundary_ambiguous": self.region_boundary_ambiguous,
            "court_side": self.side.value,
            "court_side_confidence": self.side_confidence,
        }


def _outside_rectangle_distance(point: CourtPoint, court: CourtDimensions) -> float:
    dx = max(0.0, -point.x_m, point.x_m - court.width_m)
    dy = max(0.0, -point.y_m, point.y_m - court.length_m)
    return math.hypot(dx, dy)


def _court_region(
    point: CourtPoint,
    *,
    court: CourtDimensions,
    settings: PlayerIsolationSettings,
    calibration_confidence_factor: float,
) -> tuple[CourtRegionState, float, bool]:
    inside = 0 <= point.x_m <= court.width_m and 0 <= point.y_m <= court.length_m
    if inside:
        state = CourtRegionState.INSIDE
        decision_separation = min(
            point.x_m,
            court.width_m - point.x_m,
            point.y_m,
            court.length_m - point.y_m,
        )
    else:
        outside_distance = _outside_rectangle_distance(point, court)
        if outside_distance <= settings.near_court_margin_m:
            state = CourtRegionState.NEAR
            decision_separation = min(
                outside_distance,
                settings.near_court_margin_m - outside_distance,
            )
        else:
            state = CourtRegionState.OUTSIDE
            decision_separation = outside_distance - settings.near_court_margin_m

    confidence = min(1.0, decision_separation / settings.boundary_uncertainty_m)
    confidence *= calibration_confidence_factor
    boundary_ambiguous = decision_separation < settings.boundary_uncertainty_m
    return (state, confidence, boundary_ambiguous)


def _court_side(
    point: CourtPoint,
    *,
    court: CourtDimensions,
    uncertainty_m: float,
) -> tuple[CourtSide, float]:
    net_distance = point.y_m - court.net_y_m
    confidence = min(1.0, abs(net_distance) / uncertainty_m)
    if net_distance < -uncertainty_m:
        return (CourtSide.NEAR, confidence)
    if net_distance > uncertainty_m:
        return (CourtSide.FAR, confidence)
    return (CourtSide.AMBIGUOUS, confidence)


def assess_ground_contact(
    detection: PersonDetection,
    *,
    calibration: CourtCalibration,
    frame_height_px: int,
    settings: PlayerIsolationSettings,
) -> GroundContactAssessment:
    """Project a defensible bottom-center estimate and retain geometric uncertainty."""

    image_point = ground_contact_point(detection)
    if image_point.y_px >= frame_height_px - GROUND_CLIP_TOLERANCE_PX:
        return GroundContactAssessment(
            image_point=image_point,
            court_point=None,
            projection_status=GroundProjectionStatus.FRAME_EDGE_CLIPPED,
            region_state=CourtRegionState.AMBIGUOUS,
            region_confidence=0.0,
            region_boundary_ambiguous=True,
            side=CourtSide.AMBIGUOUS,
            side_confidence=0.0,
        )
    try:
        court_point = calibration.image_to_court(image_point)
    except InvalidCalibrationError:
        return GroundContactAssessment(
            image_point=image_point,
            court_point=None,
            projection_status=GroundProjectionStatus.TRANSFORM_INVALID,
            region_state=CourtRegionState.AMBIGUOUS,
            region_confidence=0.0,
            region_boundary_ambiguous=True,
            side=CourtSide.AMBIGUOUS,
            side_confidence=0.0,
        )

    calibration_confidence_factor = max(
        0.25,
        1.0 - calibration.reprojection_error.all_rmse_court_m / settings.near_court_margin_m,
    )
    region, region_confidence, boundary_ambiguous = _court_region(
        court_point,
        court=calibration.court,
        settings=settings,
        calibration_confidence_factor=calibration_confidence_factor,
    )
    side, side_confidence = _court_side(
        court_point,
        court=calibration.court,
        uncertainty_m=settings.side_uncertainty_m,
    )
    return GroundContactAssessment(
        image_point=image_point,
        court_point=court_point,
        projection_status=GroundProjectionStatus.PROJECTED,
        region_state=region,
        region_confidence=region_confidence,
        region_boundary_ambiguous=boundary_ambiguous,
        side=side,
        side_confidence=side_confidence,
    )


@dataclass(frozen=True, slots=True)
class CandidateObservation:
    """A raw-detection reference plus derived candidate-selection geometry."""

    detection_index: int
    candidate_id: str
    detection: PersonDetection
    ground_contact: GroundContactAssessment
    association_method: str
    association_confidence: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "raw_detection": {"index": self.detection_index},
            "frame_number": self.detection.frame_number,
            "timestamp_s": self.detection.timestamp_s,
            "candidate_id": self.candidate_id,
            "ground_contact": self.ground_contact.as_dict(),
            "association": {
                "method": self.association_method,
                "confidence": self.association_confidence,
            },
        }


@dataclass(frozen=True, slots=True)
class PlayerCandidate:
    """Ephemeral court-aware tracklet used only for primary-player selection."""

    candidate_id: str
    observations: tuple[CandidateObservation, ...]
    eligible: bool
    eligibility_score: float
    court_support_ratio: float
    observed_frame_ratio: float
    dominant_side: CourtSide
    region_counts: dict[str, int]
    side_counts: dict[str, int]

    @property
    def first_frame_number(self) -> int:
        return self.observations[0].detection.frame_number

    @property
    def last_frame_number(self) -> int:
        return self.observations[-1].detection.frame_number

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "identity_scope": "ephemeral_selection_tracklet_not_persistent_player_identity",
            "eligible_primary_court_candidate": self.eligible,
            "eligibility_score": self.eligibility_score,
            "observation_count": len(self.observations),
            "first_frame_number": self.first_frame_number,
            "last_frame_number": self.last_frame_number,
            "observed_frame_ratio": self.observed_frame_ratio,
            "court_support_ratio": self.court_support_ratio,
            "dominant_court_side": self.dominant_side.value,
            "court_region_counts": self.region_counts,
            "court_side_counts": self.side_counts,
            "observations": [observation.as_dict() for observation in self.observations],
        }


@dataclass(frozen=True, slots=True)
class PlayerCandidateCollection:
    """Serializable derived candidate artifact kept separate from raw detections."""

    created_at_utc: str
    source: VideoMetadata
    detections_path: str
    calibration_path: str
    configuration: dict[str, object]
    candidates: tuple[PlayerCandidate, ...]
    schema_version: int = PLAYER_CANDIDATE_SCHEMA_VERSION

    @property
    def observations(self) -> tuple[CandidateObservation, ...]:
        return tuple(
            observation for candidate in self.candidates for observation in candidate.observations
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "record_type": "primary_player_candidates",
            "source": self.source.as_dict(),
            "inputs": {
                "raw_person_detections": self.detections_path,
                "court_calibration": self.calibration_path,
            },
            "configuration": self.configuration,
            "candidate_count": len(self.candidates),
            "eligible_candidate_count": sum(candidate.eligible for candidate in self.candidates),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


@dataclass(slots=True)
class _CandidateBuilder:
    candidate_id: str
    observations: list[CandidateObservation]

    @property
    def last(self) -> CandidateObservation:
        return self.observations[-1]


def _association_cost(
    previous: CandidateObservation,
    detection: PersonDetection,
    assessment: GroundContactAssessment,
    *,
    frame_diagonal_px: float,
    settings: PlayerIsolationSettings,
) -> float | None:
    elapsed_s = detection.timestamp_s - previous.detection.timestamp_s
    if elapsed_s <= 0 or elapsed_s > settings.max_candidate_gap_s:
        return None
    previous_assessment = previous.ground_contact
    if (
        previous_assessment.side is not CourtSide.AMBIGUOUS
        and assessment.side is not CourtSide.AMBIGUOUS
        and previous_assessment.side is not assessment.side
    ):
        return None

    if previous_assessment.court_point is not None and assessment.court_point is not None:
        distance_m = math.hypot(
            assessment.court_point.x_m - previous_assessment.court_point.x_m,
            assessment.court_point.y_m - previous_assessment.court_point.y_m,
        )
        allowed_m = ASSOCIATION_BASE_DISTANCE_M + settings.max_candidate_speed_mps * elapsed_s
        if distance_m > allowed_m:
            return None
        spatial_cost = distance_m / allowed_m
    else:
        distance_px = math.hypot(
            assessment.image_point.x_px - previous_assessment.image_point.x_px,
            assessment.image_point.y_px - previous_assessment.image_point.y_px,
        )
        allowed_fraction = 0.03 + 0.08 * (elapsed_s / settings.max_candidate_gap_s)
        allowed_px = frame_diagonal_px * allowed_fraction
        if distance_px > allowed_px:
            return None
        spatial_cost = distance_px / allowed_px

    previous_height = (
        previous.detection.bounding_box.bottom_px - previous.detection.bounding_box.top_px
    )
    current_height = detection.bounding_box.bottom_px - detection.bounding_box.top_px
    scale_cost = min(1.0, abs(math.log(current_height / previous_height)))
    return min(1.0, spatial_cost * 0.85 + scale_cost * 0.15)


def _candidate_summary(
    builder: _CandidateBuilder,
    settings: PlayerIsolationSettings,
) -> PlayerCandidate:
    observations = tuple(builder.observations)
    region_counter = Counter(
        observation.ground_contact.region_state.value for observation in observations
    )
    side_counter = Counter(observation.ground_contact.side.value for observation in observations)
    court_support = (
        region_counter[CourtRegionState.INSIDE.value]
        + 0.5 * region_counter[CourtRegionState.NEAR.value]
    ) / len(observations)
    span_frames = (
        observations[-1].detection.frame_number - observations[0].detection.frame_number + 1
    )
    persistence = len(observations) / span_frames
    near_count = side_counter[CourtSide.NEAR.value]
    far_count = side_counter[CourtSide.FAR.value]
    if near_count > far_count:
        dominant_side = CourtSide.NEAR
    elif far_count > near_count:
        dominant_side = CourtSide.FAR
    else:
        dominant_side = CourtSide.AMBIGUOUS
    eligibility_score = 0.55 * court_support + 0.45 * persistence
    eligible = (
        len(observations) >= settings.min_candidate_observations
        and court_support >= settings.min_court_support_ratio
        and dominant_side is not CourtSide.AMBIGUOUS
    )
    return PlayerCandidate(
        candidate_id=builder.candidate_id,
        observations=observations,
        eligible=eligible,
        eligibility_score=eligibility_score,
        court_support_ratio=court_support,
        observed_frame_ratio=persistence,
        dominant_side=dominant_side,
        region_counts=dict(region_counter),
        side_counts=dict(side_counter),
    )


def build_player_candidates(
    detections: PersonDetectionRun,
    *,
    calibration: CourtCalibration,
    detections_path: Path,
    calibration_path: Path,
    settings: PlayerIsolationSettings,
) -> PlayerCandidateCollection:
    """Build court-aware short-gap tracklets without selecting exactly four people."""

    source_dimensions = (detections.source.width, detections.source.height)
    calibration_dimensions = (
        calibration.source.frame_width_px,
        calibration.source.frame_height_px,
    )
    if source_dimensions != calibration_dimensions:
        raise PlayerIsolationInputError(
            f"detection dimensions {source_dimensions} do not match calibration dimensions "
            f"{calibration_dimensions}"
        )

    indexed_by_frame: dict[int, list[tuple[int, PersonDetection]]] = defaultdict(list)
    for detection_index, detection in enumerate(detections.detections):
        indexed_by_frame[detection.frame_number].append((detection_index, detection))

    builders: list[_CandidateBuilder] = []
    next_candidate_number = 1
    frame_diagonal = math.hypot(detections.source.width, detections.source.height)
    for frame_number in sorted(indexed_by_frame):
        frame_items = indexed_by_frame[frame_number]
        frame_timestamp_s = frame_items[0][1].timestamp_s
        assessments = tuple(
            assess_ground_contact(
                detection,
                calibration=calibration,
                frame_height_px=detections.source.height,
                settings=settings,
            )
            for _, detection in frame_items
        )
        pair_costs: list[tuple[float, int, int]] = []
        active_builders = (
            (builder_index, builder)
            for builder_index, builder in enumerate(builders)
            if frame_timestamp_s - builder.last.detection.timestamp_s
            <= settings.max_candidate_gap_s
        )
        for builder_index, builder in active_builders:
            for item_index, ((_, detection), assessment) in enumerate(
                zip(frame_items, assessments, strict=True)
            ):
                cost = _association_cost(
                    builder.last,
                    detection,
                    assessment,
                    frame_diagonal_px=frame_diagonal,
                    settings=settings,
                )
                if cost is not None:
                    pair_costs.append((cost, builder_index, item_index))

        matched_builders: set[int] = set()
        matched_items: set[int] = set()
        for cost, builder_index, item_index in sorted(pair_costs):
            if builder_index in matched_builders or item_index in matched_items:
                continue
            detection_index, detection = frame_items[item_index]
            builders[builder_index].observations.append(
                CandidateObservation(
                    detection_index=detection_index,
                    candidate_id=builders[builder_index].candidate_id,
                    detection=detection,
                    ground_contact=assessments[item_index],
                    association_method="short_gap_ground_proximity",
                    association_confidence=1.0 - cost,
                )
            )
            matched_builders.add(builder_index)
            matched_items.add(item_index)

        for item_index, ((detection_index, detection), assessment) in enumerate(
            zip(frame_items, assessments, strict=True)
        ):
            if item_index in matched_items:
                continue
            candidate_id = f"candidate-{next_candidate_number:06d}"
            next_candidate_number += 1
            builders.append(
                _CandidateBuilder(
                    candidate_id=candidate_id,
                    observations=[
                        CandidateObservation(
                            detection_index=detection_index,
                            candidate_id=candidate_id,
                            detection=detection,
                            ground_contact=assessment,
                            association_method="new_candidate",
                            association_confidence=None,
                        )
                    ],
                )
            )

    candidates = tuple(_candidate_summary(builder, settings) for builder in builders)
    return PlayerCandidateCollection(
        created_at_utc=datetime.now(UTC).isoformat(),
        source=detections.source,
        detections_path=str(detections_path.expanduser().resolve()),
        calibration_path=str(calibration_path.expanduser().resolve()),
        configuration=settings.as_dict(),
        candidates=candidates,
    )


@dataclass(frozen=True, slots=True)
class ManualPlayerAssignment:
    """A human logical-role assertion anchored to one candidate observation."""

    logical_player: LogicalPlayerRole
    candidate_id: str
    anchor_detection_index: int
    anchor_frame_number: int
    anchor_timestamp_s: float
    observed_side: CourtSide

    def as_dict(self) -> dict[str, object]:
        return {
            "logical_player": self.logical_player.value,
            "candidate_id": self.candidate_id,
            "assignment_source": "manual",
            "anchor_raw_detection": {"index": self.anchor_detection_index},
            "anchor_frame_number": self.anchor_frame_number,
            "anchor_timestamp_s": self.anchor_timestamp_s,
            "observed_court_side": self.observed_side.value,
        }


@dataclass(frozen=True, slots=True)
class LogicalPlayerAssignments:
    """Exactly four manual logical identities, stored apart from raw detections."""

    created_at_utc: str
    candidates_path: str
    detections_path: str
    assignments: tuple[ManualPlayerAssignment, ...]
    corrected_from_path: str | None = None
    schema_version: int = PLAYER_ASSIGNMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        roles = tuple(assignment.logical_player for assignment in self.assignments)
        candidate_ids = tuple(assignment.candidate_id for assignment in self.assignments)
        if set(roles) != set(LOGICAL_PLAYER_ROLES) or len(roles) != len(LOGICAL_PLAYER_ROLES):
            raise PlayerIsolationInputError(
                "assignments must contain each logical role exactly once"
            )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise PlayerIsolationInputError("each logical role must use a distinct candidate")

    def by_role(self) -> dict[LogicalPlayerRole, ManualPlayerAssignment]:
        return {assignment.logical_player: assignment for assignment in self.assignments}

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "record_type": "logical_player_assignments",
            "identity_contract": (
                "logical_roles_are_human_owned_and_independent_of_detector_or_candidate_ids"
            ),
            "inputs": {
                "player_candidates": self.candidates_path,
                "raw_person_detections": self.detections_path,
                "corrected_from_assignments": self.corrected_from_path,
            },
            "assignments": [assignment.as_dict() for assignment in self.assignments],
        }


def make_manual_assignment(
    role: LogicalPlayerRole,
    observation: CandidateObservation,
) -> ManualPlayerAssignment:
    return ManualPlayerAssignment(
        logical_player=role,
        candidate_id=observation.candidate_id,
        anchor_detection_index=observation.detection_index,
        anchor_frame_number=observation.detection.frame_number,
        anchor_timestamp_s=observation.detection.timestamp_s,
        observed_side=observation.ground_contact.side,
    )


def build_logical_player_assignments(
    selections: dict[LogicalPlayerRole, CandidateObservation],
    *,
    candidates_path: Path,
    detections_path: Path,
    corrected_from_path: Path | None = None,
) -> LogicalPlayerAssignments:
    if set(selections) != set(LOGICAL_PLAYER_ROLES):
        raise PlayerIsolationInputError("manual selection must assign all four logical roles")
    return LogicalPlayerAssignments(
        created_at_utc=datetime.now(UTC).isoformat(),
        candidates_path=str(candidates_path.expanduser().resolve()),
        detections_path=str(detections_path.expanduser().resolve()),
        corrected_from_path=(
            str(corrected_from_path.expanduser().resolve())
            if corrected_from_path is not None
            else None
        ),
        assignments=tuple(
            make_manual_assignment(role, selections[role]) for role in LOGICAL_PLAYER_ROLES
        ),
    )


def save_player_assignments(assignments: LogicalPlayerAssignments, path: Path) -> Path:
    output_path = path.expanduser().resolve()
    if output_path.suffix.lower() != ".json":
        raise PlayerAssignmentIoError(str(output_path), reason="output must use a .json extension")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(assignments.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as error:
        raise PlayerAssignmentIoError(str(output_path), reason=str(error)) from error
    return output_path


def _assignment_json_object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _assignment_json_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be an integer")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{field} must be an integer")
    return int(numeric)


def _assignment_json_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _assignment_json_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def load_player_assignments(path: Path) -> LogicalPlayerAssignments:
    """Load an existing manual assignment file for correction."""

    input_path = path.expanduser().resolve()
    try:
        root = _assignment_json_object(
            json.loads(input_path.read_text(encoding="utf-8")), field="root"
        )
        if _assignment_json_int(root["schema_version"], field="schema_version") != 1:
            raise ValueError("unsupported assignment schema version")
        if (
            _assignment_json_string(root["record_type"], field="record_type")
            != "logical_player_assignments"
        ):
            raise ValueError("record_type must be logical_player_assignments")
        inputs = _assignment_json_object(root["inputs"], field="inputs")
        raw_assignments = root["assignments"]
        if not isinstance(raw_assignments, list):
            raise ValueError("assignments must be an array")
        assignments: list[ManualPlayerAssignment] = []
        for index, value in enumerate(raw_assignments):
            field = f"assignments[{index}]"
            raw = _assignment_json_object(value, field=field)
            anchor = _assignment_json_object(
                raw["anchor_raw_detection"], field=f"{field}.anchor_raw_detection"
            )
            assignments.append(
                ManualPlayerAssignment(
                    logical_player=LogicalPlayerRole(
                        _assignment_json_string(
                            raw["logical_player"], field=f"{field}.logical_player"
                        )
                    ),
                    candidate_id=_assignment_json_string(
                        raw["candidate_id"], field=f"{field}.candidate_id"
                    ),
                    anchor_detection_index=_assignment_json_int(
                        anchor["index"], field=f"{field}.anchor_raw_detection.index"
                    ),
                    anchor_frame_number=_assignment_json_int(
                        raw["anchor_frame_number"], field=f"{field}.anchor_frame_number"
                    ),
                    anchor_timestamp_s=_assignment_json_float(
                        raw["anchor_timestamp_s"], field=f"{field}.anchor_timestamp_s"
                    ),
                    observed_side=CourtSide(
                        _assignment_json_string(
                            raw["observed_court_side"], field=f"{field}.observed_court_side"
                        )
                    ),
                )
            )
        corrected_raw = inputs.get("corrected_from_assignments")
        return LogicalPlayerAssignments(
            created_at_utc=_assignment_json_string(root["created_at_utc"], field="created_at_utc"),
            candidates_path=_assignment_json_string(
                inputs["player_candidates"], field="inputs.player_candidates"
            ),
            detections_path=_assignment_json_string(
                inputs["raw_person_detections"], field="inputs.raw_person_detections"
            ),
            corrected_from_path=(
                _assignment_json_string(corrected_raw, field="inputs.corrected_from_assignments")
                if corrected_raw is not None
                else None
            ),
            assignments=tuple(assignments),
        )
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        PlayerIsolationInputError,
    ) as error:
        raise PlayerAssignmentIoError(str(input_path), reason=str(error)) from error


def assignment_selections_for_candidates(
    assignments: LogicalPlayerAssignments,
    candidates: PlayerCandidateCollection,
) -> dict[LogicalPlayerRole, CandidateObservation]:
    """Resolve persisted anchors against a freshly deterministic candidate collection."""

    by_candidate_and_detection = {
        (observation.candidate_id, observation.detection_index): observation
        for observation in candidates.observations
    }
    selections: dict[LogicalPlayerRole, CandidateObservation] = {}
    for assignment in assignments.assignments:
        key = (assignment.candidate_id, assignment.anchor_detection_index)
        observation = by_candidate_and_detection.get(key)
        if observation is None:
            raise PlayerIsolationInputError(
                f"existing assignment for {assignment.logical_player.value} does not match "
                "the current candidate artifact"
            )
        selections[assignment.logical_player] = observation
    return selections
