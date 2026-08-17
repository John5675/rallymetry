"""Structured shot reconstruction and interpretable initial classification rules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from pickleball_vision.config import ShotClassificationSettings
from pickleball_vision.contact_detection import ContactCandidate
from pickleball_vision.court import CourtDimensions, CourtPoint, ImagePoint
from pickleball_vision.hitter_identification import UNKNOWN_PLAYER_ID
from pickleball_vision.rally_segmentation import BallEvidenceStatus, RallyBallFrame

SHOT_RECONSTRUCTION_SCHEMA_VERSION = 1


class ShotType(StrEnum):
    """The deliberately small initial shot-class vocabulary."""

    SERVE = "SERVE"
    RETURN = "RETURN"
    DINK = "DINK"
    DROP = "DROP"
    DRIVE = "DRIVE"
    VOLLEY = "VOLLEY"
    OVERHEAD = "OVERHEAD"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


SHOT_TYPES = tuple(ShotType)


@dataclass(frozen=True, slots=True)
class ShotRally:
    """One accepted automatic rally interval used to group contacts."""

    rally_id: str
    start_frame: int
    end_frame: int
    confidence: float


@dataclass(frozen=True, slots=True)
class ShotBounce:
    """One accepted bounce candidate available for shot landing linkage."""

    bounce_id: str
    frame: int
    timestamp_seconds: float
    court_position: CourtPoint | None
    confidence: float


@dataclass(frozen=True, slots=True)
class ShotHitterDecision:
    """One immutable logical-hitter decision linked by source contact ID."""

    contact_id: str
    player_id: str
    confidence: float


@dataclass(frozen=True, slots=True)
class ShotPlayerPosition:
    """Raw bottom-center player position at a source frame."""

    player_id: str
    image_ground_point: ImagePoint
    court_point: CourtPoint | None
    confidence: float
    tracking_state: str
    court_side: str | None
    court_region: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "playerId": self.player_id,
            "imageGroundPoint": {
                **self.image_ground_point.as_dict(),
                "coordinateSystem": "source_frame_pixels_top_left",
                "method": "bounding_box_bottom_center",
            },
            "courtPoint": (
                {
                    **self.court_point.as_dict(),
                    "coordinateSystem": "canonical_pickleball_court",
                    "source": "raw_player_ground_contact",
                }
                if self.court_point is not None
                else None
            ),
            "confidence": self.confidence,
            "trackingState": self.tracking_state,
            "courtSide": self.court_side,
            "courtRegion": self.court_region,
        }


@dataclass(frozen=True, slots=True)
class ShotTrajectorySegment:
    """A bounded reference and summary over the immutable ball trajectory."""

    start_frame: int
    end_frame: int
    segment_ids: tuple[str, ...]
    observed_count: int
    interpolated_count: int
    unknown_count: int
    known_fraction: float
    initial_image_position: ImagePoint | None
    final_image_position: ImagePoint | None
    initial_speed_diagonals_per_second: float | None
    peak_speed_diagonals_per_second: float | None

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def known_count(self) -> int:
        return self.observed_count + self.interpolated_count

    def as_dict(self) -> dict[str, object]:
        return {
            "source": "inputs.ballTrajectory",
            "startFrame": self.start_frame,
            "endFrame": self.end_frame,
            "frameRangeInclusive": True,
            "frameCount": self.frame_count,
            "segmentIds": list(self.segment_ids),
            "observedPointCount": self.observed_count,
            "interpolatedPointCount": self.interpolated_count,
            "unknownFrameCount": self.unknown_count,
            "knownPointCount": self.known_count,
            "knownFraction": self.known_fraction,
            "initialImagePosition": (
                {
                    **self.initial_image_position.as_dict(),
                    "coordinateSystem": "source_frame_pixels_top_left",
                }
                if self.initial_image_position is not None
                else None
            ),
            "finalImagePosition": (
                {
                    **self.final_image_position.as_dict(),
                    "coordinateSystem": "source_frame_pixels_top_left",
                }
                if self.final_image_position is not None
                else None
            ),
            "initialSpeedDiagonalsPerSecond": self.initial_speed_diagonals_per_second,
            "peakSpeedDiagonalsPerSecond": self.peak_speed_diagonals_per_second,
            "containsOnlyObservedInterpolatedOrUnknown": True,
        }


@dataclass(frozen=True, slots=True)
class Shot:
    """One reconstructed shot and its independently inferred class."""

    shot_id: str
    rally_id: str
    shot_index: int
    hitter_id: str
    hitter_confidence: float
    contact_id: str
    contact_frame: int
    contact_timestamp_seconds: float
    trajectory_segment: ShotTrajectorySegment
    bounce_id: str | None
    landing_court_position: CourtPoint | None
    hitter_court_position: ShotPlayerPosition | None
    shot_type: ShotType
    confidence: float
    classification_evidence: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "shotId": self.shot_id,
            "rallyId": self.rally_id,
            "shotIndex": self.shot_index,
            "hitterId": self.hitter_id,
            "hitterConfidence": self.hitter_confidence,
            "contactId": self.contact_id,
            "contactFrame": self.contact_frame,
            "contactTimestamp": self.contact_timestamp_seconds,
            "trajectorySegment": self.trajectory_segment.as_dict(),
            "bounceId": self.bounce_id,
            "landingCourtPosition": (
                {
                    **self.landing_court_position.as_dict(),
                    "coordinateSystem": "canonical_pickleball_court",
                    "source": "accepted_visually_plane_gated_bounce",
                }
                if self.landing_court_position is not None
                else None
            ),
            "hitterCourtPosition": (
                self.hitter_court_position.as_dict()
                if self.hitter_court_position is not None
                else None
            ),
            "shotType": self.shot_type.value,
            "confidence": self.confidence,
            "classificationEvidence": self.classification_evidence,
        }


@dataclass(frozen=True, slots=True)
class ShotReconstructionResult:
    """Chronological reconstructed shots plus contacts excluded from rallies."""

    shots: tuple[Shot, ...]
    accepted_contact_count: int
    contact_outside_rally_count: int


def _trajectory_segment(
    frames: tuple[RallyBallFrame, ...],
    *,
    start_frame: int,
    end_frame: int,
    frame_diagonal_px: float,
) -> ShotTrajectorySegment:
    selected = frames[start_frame : end_frame + 1]
    observed = sum(item.status is BallEvidenceStatus.OBSERVED for item in selected)
    interpolated = sum(item.status is BallEvidenceStatus.INTERPOLATED for item in selected)
    unknown = sum(item.status is BallEvidenceStatus.UNKNOWN for item in selected)
    known = tuple(item for item in selected if item.point is not None)
    speeds: list[float] = []
    initial_speed: float | None = None
    previous: RallyBallFrame | None = None
    for item in known:
        if (
            previous is not None
            and previous.point is not None
            and item.point is not None
            and item.segment_id is not None
            and item.segment_id == previous.segment_id
        ):
            elapsed = item.timestamp_seconds - previous.timestamp_seconds
            if elapsed > 0:
                speed = math.hypot(
                    item.point.x_px - previous.point.x_px,
                    item.point.y_px - previous.point.y_px,
                ) / (frame_diagonal_px * elapsed)
                speeds.append(speed)
                if initial_speed is None:
                    initial_speed = speed
        previous = item
    return ShotTrajectorySegment(
        start_frame,
        end_frame,
        tuple(sorted({item.segment_id for item in known if item.segment_id is not None})),
        observed,
        interpolated,
        unknown,
        (observed + interpolated) / len(selected),
        known[0].point if known else None,
        known[-1].point if known else None,
        initial_speed,
        max(speeds, default=None),
    )


def _backcourt_distance_from_kitchen(
    position: ShotPlayerPosition | None,
    court: CourtDimensions,
) -> float | None:
    if position is None or position.court_point is None:
        return None
    if position.court_side == "near_side":
        return max(0.0, court.near_kitchen_y_m - position.court_point.y_m)
    if position.court_side == "far_side":
        return max(0.0, position.court_point.y_m - court.far_kitchen_y_m)
    return None


def _landing_in_opponent_kitchen(
    landing: CourtPoint | None,
    hitter_position: ShotPlayerPosition | None,
    court: CourtDimensions,
) -> bool | None:
    if landing is None or hitter_position is None:
        return None
    if hitter_position.court_side == "near_side":
        return court.net_y_m <= landing.y_m <= court.far_kitchen_y_m
    if hitter_position.court_side == "far_side":
        return court.near_kitchen_y_m <= landing.y_m <= court.net_y_m
    return None


def _contact_height_ratio(
    contact: ContactCandidate,
    hitter_id: str,
) -> float | None:
    player = next((item for item in contact.candidate_players if item.role == hitter_id), None)
    if player is None:
        return None
    _left, top, _right, bottom = player.bounding_box
    height = bottom - top
    if height <= 0:
        return None
    return (contact.ball_image_position.y_px - top) / height


def _opponent_position_summary(
    positions: dict[str, ShotPlayerPosition],
    hitter: ShotPlayerPosition | None,
    *,
    court: CourtDimensions,
    kitchen_proximity_m: float,
) -> dict[str, object]:
    if hitter is None or hitter.court_side not in {"near_side", "far_side"}:
        return {"available": False, "opponents": [], "atKitchenCount": 0}
    opposite = "far_side" if hitter.court_side == "near_side" else "near_side"
    values: list[dict[str, object]] = []
    for position in positions.values():
        if position.court_side != opposite:
            continue
        distance = _backcourt_distance_from_kitchen(position, court)
        values.append(
            {
                "playerId": position.player_id,
                "courtSide": position.court_side,
                "backcourtDistanceFromKitchenMeters": distance,
                "atKitchen": distance is not None and distance <= kitchen_proximity_m,
                "trackingConfidence": position.confidence,
            }
        )
    return {
        "available": bool(values),
        "opponents": values,
        "atKitchenCount": sum(bool(item["atKitchen"]) for item in values),
    }


def _unknown_confidence(
    *,
    known_hitter: bool,
    position_available: bool,
    speed_available: bool,
    hitter_confidence: float,
    trajectory_coverage: float,
    known_points: int,
    settings: ShotClassificationSettings,
) -> float:
    evidence = min(
        1.0 if known_hitter else 0.0,
        1.0 if position_available else 0.0,
        1.0 if speed_available else 0.0,
        hitter_confidence / settings.minimum_hitter_confidence,
        trajectory_coverage / settings.minimum_trajectory_coverage,
        known_points / settings.minimum_known_trajectory_points,
    )
    return 1.0 - max(0.0, min(1.0, evidence))


def classify_shot(
    *,
    shot_index: int,
    contact: ContactCandidate,
    hitter: ShotHitterDecision,
    hitter_position: ShotPlayerPosition | None,
    trajectory: ShotTrajectorySegment,
    bounce: ShotBounce | None,
    previous_shot: Shot | None,
    positions_at_contact: dict[str, ShotPlayerPosition],
    settings: ShotClassificationSettings,
    court: CourtDimensions,
) -> tuple[ShotType, float, dict[str, object]]:
    """Apply the documented ordered rule set and return inspectable evidence."""

    kitchen_distance = _backcourt_distance_from_kitchen(hitter_position, court)
    landing = bounce.court_position if bounce is not None else None
    opponent_kitchen = _landing_in_opponent_kitchen(landing, hitter_position, court)
    initial_speed = trajectory.initial_speed_diagonals_per_second
    height_ratio = _contact_height_ratio(contact, hitter.player_id)
    incoming_bounce = previous_shot is not None and previous_shot.bounce_id is not None
    opponent_positions = _opponent_position_summary(
        positions_at_contact,
        hitter_position,
        court=court,
        kitchen_proximity_m=settings.kitchen_proximity_m,
    )
    missing: list[str] = []
    if hitter.player_id == UNKNOWN_PLAYER_ID:
        missing.append("knownHitter")
    if hitter.confidence < settings.minimum_hitter_confidence:
        missing.append("minimumHitterConfidence")
    if hitter_position is None or hitter_position.court_point is None:
        missing.append("hitterCourtPosition")
    if trajectory.known_fraction < settings.minimum_trajectory_coverage:
        missing.append("trajectoryCoverage")
    if trajectory.known_count < settings.minimum_known_trajectory_points:
        missing.append("knownTrajectoryPoints")
    if initial_speed is None:
        missing.append("initialImageSpeed")
    common = {
        "shotIndex": shot_index,
        "hitterId": hitter.player_id,
        "hitterConfidence": hitter.confidence,
        "hitterBackcourtDistanceFromKitchenMeters": kitchen_distance,
        "bounceAvailable": bounce is not None,
        "landingCourtPositionAvailable": landing is not None,
        "landingInOpponentKitchen": opponent_kitchen,
        "incomingShotBouncedBeforeContact": incoming_bounce if previous_shot is not None else None,
        "initialSpeedDiagonalsPerSecond": initial_speed,
        "peakSpeedDiagonalsPerSecond": trajectory.peak_speed_diagonals_per_second,
        "trajectoryKnownFraction": trajectory.known_fraction,
        "knownTrajectoryPointCount": trajectory.known_count,
        "contactHeightWithinHitterBoxRatio": height_ratio,
        "previousShotType": previous_shot.shot_type.value if previous_shot is not None else None,
        "opponentPositioning": opponent_positions,
        "airborneBallProjectedThroughHomography": False,
        "newNeuralNetworkUsed": False,
    }
    if missing:
        return (
            ShotType.UNKNOWN,
            _unknown_confidence(
                known_hitter=hitter.player_id != UNKNOWN_PLAYER_ID,
                position_available=(
                    hitter_position is not None and hitter_position.court_point is not None
                ),
                speed_available=initial_speed is not None,
                hitter_confidence=hitter.confidence,
                trajectory_coverage=trajectory.known_fraction,
                known_points=trajectory.known_count,
                settings=settings,
            ),
            {
                **common,
                "selectedRule": "UNKNOWN_INSUFFICIENT_EVIDENCE",
                "failedEvidenceGates": missing,
                "rulePrecedence": [],
            },
        )
    if kitchen_distance is None or initial_speed is None:
        raise AssertionError("classification evidence gates must retain required features")
    base_confidence = min(
        contact.visual_confidence,
        hitter.confidence,
        max(settings.minimum_trajectory_coverage, trajectory.known_fraction),
    )
    rules: list[dict[str, object]] = []

    def assess(name: str, matched: bool, strength: float) -> bool:
        rules.append({"rule": name, "matched": matched, "strength": strength})
        return matched

    serve_strength = min(1.0, kitchen_distance / settings.serve_minimum_backcourt_distance_m)
    if assess(
        "SERVE_FIRST_SHOT_BACKCOURT_WITH_BOUNCE",
        shot_index == 1
        and bounce is not None
        and kitchen_distance >= settings.serve_minimum_backcourt_distance_m,
        serve_strength,
    ):
        selected = ShotType.SERVE
        strength = serve_strength
    elif assess(
        "RETURN_SECOND_SHOT_AFTER_SERVE",
        shot_index == 2
        and previous_shot is not None
        and previous_shot.shot_type is ShotType.SERVE
        and previous_shot.bounce_id is not None,
        1.0,
    ):
        selected = ShotType.RETURN
        strength = 1.0
    elif assess(
        "OVERHEAD_HIGH_CONTACT_FAST_NO_INCOMING_BOUNCE",
        shot_index > 2
        and not incoming_bounce
        and height_ratio is not None
        and height_ratio <= settings.overhead_maximum_contact_height_ratio
        and initial_speed >= settings.overhead_minimum_speed_diagonals_per_second,
        min(1.0, initial_speed / settings.overhead_minimum_speed_diagonals_per_second),
    ):
        selected = ShotType.OVERHEAD
        strength = min(1.0, initial_speed / settings.overhead_minimum_speed_diagonals_per_second)
    elif assess(
        "DINK_NEAR_KITCHEN_SOFT_OPPONENT_KITCHEN_LANDING",
        opponent_kitchen is True
        and kitchen_distance <= settings.kitchen_proximity_m
        and initial_speed <= settings.dink_maximum_speed_diagonals_per_second,
        1.0 - min(1.0, initial_speed / settings.dink_maximum_speed_diagonals_per_second),
    ):
        selected = ShotType.DINK
        strength = 1.0 - min(1.0, initial_speed / settings.dink_maximum_speed_diagonals_per_second)
    elif assess(
        "DROP_BACKCOURT_SOFT_OPPONENT_KITCHEN_LANDING",
        opponent_kitchen is True
        and kitchen_distance >= settings.drop_minimum_backcourt_distance_m
        and initial_speed <= settings.drop_maximum_speed_diagonals_per_second,
        min(1.0, kitchen_distance / settings.drop_minimum_backcourt_distance_m),
    ):
        selected = ShotType.DROP
        strength = min(1.0, kitchen_distance / settings.drop_minimum_backcourt_distance_m)
    elif assess(
        "VOLLEY_NEAR_KITCHEN_WITHOUT_INCOMING_BOUNCE",
        shot_index > 2 and not incoming_bounce and kitchen_distance <= settings.kitchen_proximity_m,
        1.0 - min(1.0, kitchen_distance / settings.kitchen_proximity_m),
    ):
        selected = ShotType.VOLLEY
        strength = 1.0 - min(1.0, kitchen_distance / settings.kitchen_proximity_m)
    elif assess(
        "DRIVE_HIGH_INITIAL_IMAGE_SPEED",
        initial_speed >= settings.drive_minimum_speed_diagonals_per_second,
        min(1.0, initial_speed / settings.drive_minimum_speed_diagonals_per_second),
    ):
        selected = ShotType.DRIVE
        strength = min(1.0, initial_speed / settings.drive_minimum_speed_diagonals_per_second)
    else:
        assess("OTHER_SUFFICIENT_EVIDENCE_NO_SPECIALIZED_RULE", True, 0.5)
        selected = ShotType.OTHER
        strength = 0.5
    confidence = max(0.0, min(1.0, base_confidence * (0.70 + 0.30 * strength)))
    return (
        selected,
        confidence,
        {
            **common,
            "selectedRule": next(item["rule"] for item in rules if item["matched"]),
            "failedEvidenceGates": [],
            "rulePrecedence": rules,
            "baseEvidenceConfidence": base_confidence,
            "ruleStrength": strength,
        },
    )


def reconstruct_shots(
    *,
    frames: tuple[RallyBallFrame, ...],
    rallies: tuple[ShotRally, ...],
    contacts: tuple[ContactCandidate, ...],
    bounces: tuple[ShotBounce, ...],
    hitters_by_contact: dict[str, ShotHitterDecision],
    player_positions_by_frame: tuple[dict[str, ShotPlayerPosition], ...],
    frame_width_px: int,
    frame_height_px: int,
    settings: ShotClassificationSettings,
    court: CourtDimensions | None = None,
) -> ShotReconstructionResult:
    """Connect accepted source events into rally-local shots and classify them."""

    if not frames or tuple(item.frame_number for item in frames) != tuple(range(len(frames))):
        raise ValueError("shot reconstruction requires a complete zero-based ball timeline")
    if len(player_positions_by_frame) != len(frames):
        raise ValueError("player position timeline must match the ball timeline")
    if frame_width_px < 1 or frame_height_px < 1:
        raise ValueError("shot reconstruction dimensions must be positive")
    ordered_rallies = tuple(sorted(rallies, key=lambda item: (item.start_frame, item.end_frame)))
    if len({item.rally_id for item in ordered_rallies}) != len(ordered_rallies):
        raise ValueError("shot reconstruction rally IDs must be unique")
    for index, rally in enumerate(ordered_rallies):
        if not 0 <= rally.start_frame <= rally.end_frame < len(frames):
            raise ValueError(f"rally {rally.rally_id} is outside the source timeline")
        if index > 0 and ordered_rallies[index - 1].end_frame >= rally.start_frame:
            raise ValueError("shot reconstruction rallies must not overlap")
    if len({item.contact_id for item in contacts}) != len(contacts):
        raise ValueError("shot reconstruction contact IDs must be unique")
    ordered_bounces = tuple(sorted(bounces, key=lambda item: item.frame))
    diagonal = math.hypot(frame_width_px, frame_height_px)
    court_dimensions = court or CourtDimensions()
    accepted = tuple(item for item in contacts if item.accepted_fused)
    used_contacts: set[str] = set()
    shots: list[Shot] = []
    for rally in ordered_rallies:
        rally_contacts = tuple(
            item for item in accepted if rally.start_frame <= item.frame <= rally.end_frame
        )
        previous_shot: Shot | None = None
        for index, contact in enumerate(rally_contacts, start=1):
            used_contacts.add(contact.contact_id)
            end_frame = (
                rally_contacts[index].frame if index < len(rally_contacts) else rally.end_frame
            )
            end_frame = max(contact.frame, end_frame)
            trajectory = _trajectory_segment(
                frames,
                start_frame=contact.frame,
                end_frame=end_frame,
                frame_diagonal_px=diagonal,
            )
            has_next_contact = index < len(rally_contacts)
            bounce = next(
                (
                    item
                    for item in ordered_bounces
                    if contact.frame < item.frame
                    and (item.frame < end_frame if has_next_contact else item.frame <= end_frame)
                ),
                None,
            )
            hitter = hitters_by_contact.get(
                contact.contact_id,
                ShotHitterDecision(contact.contact_id, UNKNOWN_PLAYER_ID, 1.0),
            )
            positions = player_positions_by_frame[contact.frame]
            hitter_position = positions.get(hitter.player_id)
            shot_type, confidence, evidence = classify_shot(
                shot_index=index,
                contact=contact,
                hitter=hitter,
                hitter_position=hitter_position,
                trajectory=trajectory,
                bounce=bounce,
                previous_shot=previous_shot,
                positions_at_contact=positions,
                settings=settings,
                court=court_dimensions,
            )
            shot = Shot(
                shot_id=f"shot-{len(shots) + 1:06d}",
                rally_id=rally.rally_id,
                shot_index=index,
                hitter_id=hitter.player_id,
                hitter_confidence=hitter.confidence,
                contact_id=contact.contact_id,
                contact_frame=contact.frame,
                contact_timestamp_seconds=contact.timestamp_seconds,
                trajectory_segment=trajectory,
                bounce_id=bounce.bounce_id if bounce is not None else None,
                landing_court_position=bounce.court_position if bounce is not None else None,
                hitter_court_position=hitter_position,
                shot_type=shot_type,
                confidence=confidence,
                classification_evidence={
                    **evidence,
                    "sourceContactVisualConfidence": contact.visual_confidence,
                    "sourceContactFusedConfidence": contact.fused_confidence,
                    "rallyConfidence": rally.confidence,
                    "bounceConfidence": bounce.confidence if bounce is not None else None,
                },
            )
            shots.append(shot)
            previous_shot = shot
    return ShotReconstructionResult(
        tuple(shots),
        len(accepted),
        len(accepted) - len(used_contacts),
    )
