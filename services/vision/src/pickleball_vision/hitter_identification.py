"""Conservative logical-player resolution for visual paddle-contact candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

from pickleball_vision.config import HitterIdentificationSettings
from pickleball_vision.contact_detection import ContactCandidate, ContactCandidatePlayer
from pickleball_vision.court import ImagePoint

HITTER_IDENTIFICATION_SCHEMA_VERSION = 1
UNKNOWN_PLAYER_ID = "UNKNOWN"
LOGICAL_PLAYER_IDS = ("ME", "PARTNER", "OPPONENT_1", "OPPONENT_2")


@dataclass(frozen=True, slots=True)
class HitterAlternative:
    """One ranked logical-player hypothesis retained for inspection."""

    player_id: str
    display_name: str | None
    confidence: float
    rank: int
    court_side: str | None
    distance_diagonal_fraction: float
    tracking_confidence: float

    def as_dict(self) -> dict[str, object]:
        return {
            "playerId": self.player_id,
            "displayName": self.display_name,
            "confidence": self.confidence,
            "rank": self.rank,
            "courtSide": self.court_side,
            "distanceDiagonalFraction": self.distance_diagonal_fraction,
            "trackingConfidence": self.tracking_confidence,
        }


@dataclass(frozen=True, slots=True)
class HitterIdentification:
    """One identity decision linked to an immutable source contact candidate."""

    contact_id: str
    frame: int
    timestamp_seconds: float
    media_timestamp_seconds: float
    ball_image_position: ImagePoint
    player_id: str
    display_name: str | None
    confidence: float
    selected_court_side: str | None
    source_visual_contact_confidence: float
    source_contact_eligible: bool
    candidate_players: tuple[ContactCandidatePlayer, ...]
    alternatives: tuple[HitterAlternative, ...]
    supporting_signals: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "contactId": self.contact_id,
            "frame": self.frame,
            "timestamp": self.timestamp_seconds,
            "mediaTimestamp": self.media_timestamp_seconds,
            "ballImagePosition": {
                **self.ball_image_position.as_dict(),
                "coordinateSystem": "source_frame_pixels_top_left",
            },
            "playerId": self.player_id,
            "displayName": self.display_name,
            "confidence": self.confidence,
            "alternatives": [item.as_dict() for item in self.alternatives],
            "supportingSignals": self.supporting_signals,
        }


@dataclass(frozen=True, slots=True)
class HitterIdentificationResult:
    """Chronological hitter decisions for the complete contact candidate set."""

    identifications: tuple[HitterIdentification, ...]


@dataclass(frozen=True, slots=True)
class _ScoredPlayer:
    player: ContactCandidatePlayer
    score: float
    components: dict[str, object]


@dataclass(frozen=True, slots=True)
class _PreviousHitter:
    player_id: str
    confidence: float
    court_side: str | None
    frame: int
    timestamp_seconds: float
    rally_id: str | None


def _nested_object(root: dict[str, object], key: str) -> dict[str, object]:
    value = root.get(key)
    if isinstance(value, dict) and all(isinstance(name, str) for name in value):
        return cast(dict[str, object], value)
    return {}


def _optional_number(root: dict[str, object], key: str) -> float | None:
    value = root.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _trajectory_velocities(
    contact: ContactCandidate,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    discontinuity = _nested_object(contact.supporting_signals, "trajectoryVelocityDiscontinuity")
    before = _nested_object(discontinuity, "beforeVelocityPixelsPerSecond")
    after = _nested_object(discontinuity, "afterVelocityPixelsPerSecond")
    before_x = _optional_number(before, "x")
    before_y = _optional_number(before, "y")
    after_x = _optional_number(after, "x")
    after_y = _optional_number(after, "y")
    return (
        (before_x, before_y) if before_x is not None and before_y is not None else None,
        (after_x, after_y) if after_x is not None and after_y is not None else None,
    )


def _rally_id(contact: ContactCandidate) -> str | None:
    payload = _nested_object(contact.supporting_signals, "plausibleRallySequence")
    value = payload.get("rallyId")
    return value if isinstance(value, str) and value else None


def _state_quality(state: str) -> float:
    return {
        "observed": 1.0,
        "reacquired": 0.85,
        "suspected_identity_switch": 0.30,
        "temporarily_missing": 0.0,
    }.get(state, 0.0)


def _court_quality(region: str | None) -> float:
    if region is None:
        return 0.25
    return {
        "inside": 1.0,
        "near": 0.65,
        "ambiguous": 0.35,
        "outside": 0.0,
    }.get(region, 0.25)


def _direction_quality(
    player: ContactCandidatePlayer,
    *,
    before_velocity: tuple[float, float] | None,
    after_velocity: tuple[float, float] | None,
    frame_diagonal_px: float,
    minimum_speed_fraction: float,
) -> tuple[float, dict[str, object]]:
    expected_signs = {
        "near_side": (1, -1),
        "far_side": (-1, 1),
    }
    expected = expected_signs.get(player.court_side or "")
    samples: list[tuple[str, float, int]] = []
    if expected is not None and before_velocity is not None:
        samples.append(("incoming", before_velocity[1], expected[0]))
    if expected is not None and after_velocity is not None:
        samples.append(("outgoing", after_velocity[1], expected[1]))
    evaluated: list[dict[str, object]] = []
    matches: list[float] = []
    for name, vertical_speed, expected_sign in samples:
        normalized_speed = abs(vertical_speed) / frame_diagonal_px
        available = normalized_speed >= minimum_speed_fraction
        matches_expected = available and math.copysign(1.0, vertical_speed) == expected_sign
        evaluated.append(
            {
                "phase": name,
                "verticalSpeedPixelsPerSecond": vertical_speed,
                "verticalSpeedDiagonalFractionPerSecond": normalized_speed,
                "expectedVerticalSign": expected_sign,
                "available": available,
                "matchesExpectedCourtDirection": matches_expected if available else None,
            }
        )
        if available:
            matches.append(1.0 if matches_expected else 0.0)
    quality = sum(matches) / len(matches) if matches else 0.5
    return quality, {
        "courtSide": player.court_side,
        "quality": quality,
        "availablePhaseCount": len(matches),
        "minimumSpeedDiagonalFractionPerSecond": minimum_speed_fraction,
        "phases": evaluated,
        "airborneBallProjectedThroughHomography": False,
    }


def _previous_context_applies(
    previous: _PreviousHitter | None,
    *,
    contact: ContactCandidate,
    rally_id: str | None,
    settings: HitterIdentificationSettings,
) -> bool:
    if previous is None:
        return False
    if (
        contact.timestamp_seconds - previous.timestamp_seconds
        > settings.maximum_sequence_gap_seconds
    ):
        return False
    if rally_id is not None or previous.rally_id is not None:
        return rally_id is not None and rally_id == previous.rally_id
    return True


def _sequence_quality(
    player: ContactCandidatePlayer,
    previous: _PreviousHitter | None,
    *,
    applies: bool,
) -> float:
    if not applies or previous is None:
        return 0.5
    if player.court_side not in {"near_side", "far_side"} or previous.court_side not in {
        "near_side",
        "far_side",
    }:
        return 0.5
    return 1.0 if player.court_side != previous.court_side else 0.0


def _score_player(
    player: ContactCandidatePlayer,
    *,
    contact: ContactCandidate,
    before_velocity: tuple[float, float] | None,
    after_velocity: tuple[float, float] | None,
    previous: _PreviousHitter | None,
    previous_applies: bool,
    frame_diagonal_px: float,
    settings: HitterIdentificationSettings,
) -> _ScoredPlayer:
    proximity = 1.0 if player.ball_inside_person_box else player.proximity_confidence
    tracking = player.tracking_confidence * _state_quality(player.tracking_state)
    direction, direction_details = _direction_quality(
        player,
        before_velocity=before_velocity,
        after_velocity=after_velocity,
        frame_diagonal_px=frame_diagonal_px,
        minimum_speed_fraction=(settings.minimum_direction_speed_diagonal_fraction_per_second),
    )
    court = _court_quality(player.court_region)
    sequence = _sequence_quality(player, previous, applies=previous_applies)
    components = {
        "proximity": proximity,
        "tracking": tracking,
        "trajectoryDirection": direction,
        "visualContactConfidence": contact.visual_confidence,
        "courtContext": court,
        "sequence": sequence,
    }
    weights = {
        "proximity": settings.proximity_weight,
        "tracking": settings.tracking_weight,
        "trajectoryDirection": settings.direction_weight,
        "visualContactConfidence": settings.contact_weight,
        "courtContext": settings.court_context_weight,
        "sequence": settings.sequence_weight,
    }
    total_weight = sum(weights.values())
    score = sum(float(components[name]) * weight for name, weight in weights.items()) / total_weight
    return _ScoredPlayer(
        player,
        max(0.0, min(1.0, score)),
        {
            "playerId": player.role,
            "displayName": player.display_name,
            "score": max(0.0, min(1.0, score)),
            "components": components,
            "weights": weights,
            "distancePx": player.distance_px,
            "distanceDiagonalFraction": player.distance_diagonal_fraction,
            "trackingConfidence": player.tracking_confidence,
            "trackingState": player.tracking_state,
            "courtSide": player.court_side,
            "courtRegion": player.court_region,
            "ballInsidePlayerBoundingBox": player.ball_inside_person_box,
            "directionDetails": direction_details,
        },
    )


def _unknown_confidence(gate_strengths: tuple[float, ...]) -> float:
    if not gate_strengths:
        return 1.0
    decisiveness = max(0.0, min(1.0, min(gate_strengths)))
    return 1.0 - decisiveness


def identify_hitters(
    contacts: tuple[ContactCandidate, ...],
    *,
    frame_width_px: int,
    frame_height_px: int,
    settings: HitterIdentificationSettings,
) -> HitterIdentificationResult:
    """Resolve logical hitters chronologically while retaining conservative unknowns."""

    if frame_width_px < 1 or frame_height_px < 1:
        raise ValueError("hitter-identification dimensions must be positive")
    if any(current.frame >= following.frame for current, following in pairwise(contacts)):
        raise ValueError("contact candidates must be strictly chronological")
    if len({item.contact_id for item in contacts}) != len(contacts):
        raise ValueError("contact candidate IDs must be unique")
    diagonal = math.hypot(frame_width_px, frame_height_px)
    previous: _PreviousHitter | None = None
    rally_indices: dict[str, int] = {}
    results: list[HitterIdentification] = []
    for contact in contacts:
        rally_id = _rally_id(contact)
        contact_eligible = contact.visual_confidence >= settings.minimum_contact_confidence
        rally_index: int | None = None
        if contact_eligible and rally_id is not None:
            rally_indices[rally_id] = rally_indices.get(rally_id, 0) + 1
            rally_index = rally_indices[rally_id]
        previous_applies = _previous_context_applies(
            previous,
            contact=contact,
            rally_id=rally_id,
            settings=settings,
        )
        before_velocity, after_velocity = _trajectory_velocities(contact)
        scored = sorted(
            (
                _score_player(
                    player,
                    contact=contact,
                    before_velocity=before_velocity,
                    after_velocity=after_velocity,
                    previous=previous,
                    previous_applies=previous_applies,
                    frame_diagonal_px=diagonal,
                    settings=settings,
                )
                for player in contact.candidate_players
                if player.role in LOGICAL_PLAYER_IDS
            ),
            key=lambda item: (-item.score, item.player.role),
        )
        best = scored[0] if scored else None
        runner_up = scored[1] if len(scored) > 1 else None
        margin = (
            best.score - runner_up.score
            if best is not None and runner_up is not None
            else (best.score if best is not None else 0.0)
        )
        distance_pass = bool(
            best is not None
            and (
                best.player.ball_inside_person_box
                or best.player.distance_diagonal_fraction
                <= settings.maximum_player_distance_diagonal_fraction
            )
        )
        tracking_pass = bool(
            best is not None
            and best.player.tracking_confidence >= settings.minimum_tracking_confidence
        )
        score_pass = bool(best is not None and best.score >= settings.minimum_assignment_confidence)
        margin_pass = bool(best is not None and margin >= settings.minimum_assignment_margin)
        gates = {
            "contactConfidence": contact_eligible,
            "candidateAvailable": best is not None,
            "playerDistance": distance_pass,
            "trackingConfidence": tracking_pass,
            "assignmentConfidence": score_pass,
            "assignmentMargin": margin_pass,
        }
        assigned = all(gates.values())
        reasons = [name for name, passed in gates.items() if not passed]
        if best is None:
            gate_strengths: tuple[float, ...] = ()
        else:
            gate_strengths = (
                contact.visual_confidence / settings.minimum_contact_confidence,
                best.score / settings.minimum_assignment_confidence,
                margin / settings.minimum_assignment_margin,
                best.player.tracking_confidence / settings.minimum_tracking_confidence,
                1.0 if distance_pass else 0.0,
            )
        player_id = best.player.role if assigned and best is not None else UNKNOWN_PLAYER_ID
        display_name = best.player.display_name if assigned and best is not None else None
        confidence = (
            best.score if assigned and best is not None else _unknown_confidence(gate_strengths)
        )
        alternatives = tuple(
            HitterAlternative(
                item.player.role,
                item.player.display_name,
                item.score,
                rank,
                item.player.court_side,
                item.player.distance_diagonal_fraction,
                item.player.tracking_confidence,
            )
            for rank, item in enumerate(scored, start=1)
            if not assigned or item is not best
        )
        previous_payload = {
            "applied": previous_applies,
            "playerId": previous.player_id if previous is not None else None,
            "confidence": previous.confidence if previous is not None else None,
            "courtSide": previous.court_side if previous is not None else None,
            "contactFrame": previous.frame if previous is not None else None,
            "secondsSincePrevious": (
                contact.timestamp_seconds - previous.timestamp_seconds
                if previous is not None
                else None
            ),
            "sameRally": (
                previous_applies
                and rally_id is not None
                and previous is not None
                and rally_id == previous.rally_id
            ),
            "canOverrideCurrentVisualEvidence": False,
        }
        supporting = {
            "contactEvidence": {
                "visualConfidence": contact.visual_confidence,
                "fusedConfidence": contact.fused_confidence,
                "audioConfidence": contact.audio_confidence,
                "confidenceUsedForIdentity": "visualConfidence",
                "audioIdentityContribution": 0.0,
                "acceptedFused": contact.accepted_fused,
            },
            "ballLocation": {
                **contact.ball_image_position.as_dict(),
                "coordinateSystem": "source_frame_pixels_top_left",
                "airborneBallProjectedThroughHomography": False,
            },
            "trajectoryDirection": {
                "beforeVelocityPixelsPerSecond": (
                    {"x": before_velocity[0], "y": before_velocity[1]}
                    if before_velocity is not None
                    else None
                ),
                "afterVelocityPixelsPerSecond": (
                    {"x": after_velocity[0], "y": after_velocity[1]}
                    if after_velocity is not None
                    else None
                ),
                "usedForCourtSideConsistencyOnly": True,
            },
            "previousHitter": previous_payload,
            "rallyOrdering": {
                "rallyId": rally_id,
                "sequenceIndexAmongEligibleContacts": rally_index,
                "source": "contact_supporting_signals",
            },
            "playerScores": [item.components for item in scored],
            "decision": {
                "assigned": assigned,
                "bestPlayerId": best.player.role if best is not None else None,
                "bestScore": best.score if best is not None else None,
                "runnerUpPlayerId": runner_up.player.role if runner_up is not None else None,
                "runnerUpScore": runner_up.score if runner_up is not None else None,
                "margin": margin if best is not None else None,
                "gates": gates,
                "failedGates": reasons,
                "unknownIsSupported": True,
            },
        }
        result = HitterIdentification(
            contact.contact_id,
            contact.frame,
            contact.timestamp_seconds,
            contact.media_timestamp_seconds,
            contact.ball_image_position,
            player_id,
            display_name,
            max(0.0, min(1.0, confidence)),
            best.player.court_side if assigned and best is not None else None,
            contact.visual_confidence,
            contact_eligible,
            contact.candidate_players,
            alternatives,
            supporting,
        )
        results.append(result)
        if contact_eligible:
            if (
                assigned
                and best is not None
                and result.confidence >= settings.previous_hitter_minimum_confidence
            ):
                previous = _PreviousHitter(
                    result.player_id,
                    result.confidence,
                    best.player.court_side,
                    result.frame,
                    result.timestamp_seconds,
                    rally_id,
                )
            else:
                previous = None
    return HitterIdentificationResult(tuple(results))
