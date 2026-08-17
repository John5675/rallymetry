"""Post-inference hitter accuracy evaluation against human player labels."""

from __future__ import annotations

from typing import cast

from pickleball_vision.config import HitterIdentificationSettings
from pickleball_vision.contact_evaluation import GroundTruthContact
from pickleball_vision.hitter_identification import (
    LOGICAL_PLAYER_IDS,
    UNKNOWN_PLAYER_ID,
    HitterIdentification,
)


def _ground_truth_side(
    prediction: HitterIdentification,
    annotated_player_id: str,
) -> str | None:
    values = prediction.supporting_signals.get("playerScores")
    if not isinstance(values, list):
        return None
    for value in values:
        if not isinstance(value, dict):
            continue
        item = cast(dict[object, object], value)
        if item.get("playerId") != annotated_player_id:
            continue
        side = item.get("courtSide")
        return side if side in {"near_side", "far_side"} else None
    return None


def _accuracy_payload(matches: list[dict[str, object]]) -> dict[str, object]:
    correct = sum(bool(item["correct"]) for item in matches)
    unknown = sum(item["predictedPlayerId"] == UNKNOWN_PLAYER_ID for item in matches)
    incorrect = len(matches) - correct - unknown
    assigned = correct + incorrect
    return {
        "evaluatedMatchCount": len(matches),
        "correctCount": correct,
        "incorrectCount": incorrect,
        "unknownCount": unknown,
        "accuracy": correct / len(matches) if matches else None,
        "assignmentCoverage": assigned / len(matches) if matches else None,
        "decisiveAccuracy": correct / assigned if assigned else None,
    }


def evaluate_hitters(
    identifications: tuple[HitterIdentification, ...],
    annotations: tuple[GroundTruthContact, ...],
    *,
    settings: HitterIdentificationSettings,
    evaluation_partition: str = "validation",
) -> dict[str, object]:
    """Match contacts by time only, then score the independently inferred player ID."""

    if evaluation_partition not in {"development", "validation", "test"}:
        raise ValueError("evaluation partition must be development, validation, or test")
    labeled = tuple(item for item in annotations if item.annotated_player_id in LOGICAL_PLAYER_IDS)
    unlabeled_count = len(annotations) - len(labeled)
    if not labeled:
        return unavailable_hitter_evaluation(
            evaluation_partition=evaluation_partition,
            unlabeled_contact_count=unlabeled_count,
        )
    eligible = tuple(item for item in identifications if item.source_contact_eligible)
    tolerance_seconds = settings.evaluation_tolerance_ms / 1000.0
    proposals: list[tuple[float, float, HitterIdentification, GroundTruthContact]] = []
    for predicted in eligible:
        for annotated in labeled:
            delta = predicted.timestamp_seconds - annotated.timestamp_seconds
            if abs(delta) <= tolerance_seconds:
                proposals.append(
                    (abs(delta), -predicted.source_visual_contact_confidence, predicted, annotated)
                )
    proposals.sort(key=lambda item: item[:2])
    used_predictions: set[str] = set()
    used_annotations: set[str] = set()
    matches: list[dict[str, object]] = []
    confusion = {
        player_id: {predicted: 0 for predicted in (*LOGICAL_PLAYER_IDS, UNKNOWN_PLAYER_ID)}
        for player_id in LOGICAL_PLAYER_IDS
    }
    for absolute_error, _negative_score, predicted, annotated in proposals:
        if predicted.contact_id in used_predictions or annotated.event_id in used_annotations:
            continue
        annotated_player = annotated.annotated_player_id
        if annotated_player is None:
            continue
        used_predictions.add(predicted.contact_id)
        used_annotations.add(annotated.event_id)
        correct = predicted.player_id == annotated_player
        side = _ground_truth_side(predicted, annotated_player)
        confusion[annotated_player][predicted.player_id] += 1
        matches.append(
            {
                "predictedContactId": predicted.contact_id,
                "annotatedEventId": annotated.event_id,
                "annotatedEventType": annotated.event_type,
                "annotatedPlayerId": annotated_player,
                "predictedPlayerId": predicted.player_id,
                "predictionConfidence": predicted.confidence,
                "correct": correct,
                "observedGroundTruthPlayerCourtSide": side,
                "timingErrorMs": (predicted.timestamp_seconds - annotated.timestamp_seconds) * 1000,
                "absoluteTimingErrorMs": absolute_error * 1000,
            }
        )
    by_side: dict[str, object] = {}
    for side in ("near_side", "far_side"):
        side_matches = [
            item for item in matches if item["observedGroundTruthPlayerCourtSide"] == side
        ]
        by_side[side] = _accuracy_payload(side_matches)
    unavailable_side_count = sum(
        item["observedGroundTruthPlayerCourtSide"] not in {"near_side", "far_side"}
        for item in matches
    )
    missed = [item.as_dict() for item in labeled if item.event_id not in used_annotations]
    return {
        "evaluationAvailable": True,
        "evaluationPartition": evaluation_partition,
        "thresholdTuningPerformed": False,
        "heldOutThresholdTuningAllowed": evaluation_partition == "development",
        "groundTruthEventTypes": ["SERVE_CONTACT", "PADDLE_CONTACT"],
        "groundTruthLabeledContactCount": len(labeled),
        "ignoredUnlabeledContactCount": unlabeled_count,
        "eligiblePredictedContactCount": len(eligible),
        "matchedContactCount": len(matches),
        "missedGroundTruthContactCount": len(missed),
        "contactMatchCoverage": len(matches) / len(labeled),
        "matchingConfiguration": {
            "oneToOneMatching": True,
            "timingToleranceMs": settings.evaluation_tolerance_ms,
            "playerIdentityUsedForTemporalMatching": False,
            "audioUsedForIdentity": False,
        },
        "overall": _accuracy_payload(matches),
        "byObservedPlayerCourtSide": {
            **by_side,
            "unavailableMatchCount": unavailable_side_count,
            "courtSideSource": "ground_truth_player_track_observation_at_matched_contact",
            "ballProjectedThroughHomography": False,
        },
        "confusionMatrix": confusion,
        "matches": matches,
        "missedGroundTruthContacts": missed,
    }


def unavailable_hitter_evaluation(
    *,
    evaluation_partition: str,
    unlabeled_contact_count: int = 0,
) -> dict[str, object]:
    """Return an explicit state when human player labels are unavailable."""

    return {
        "evaluationAvailable": False,
        "evaluationPartition": evaluation_partition,
        "thresholdTuningPerformed": False,
        "reason": "no human contact annotations with logical playerId labels supplied",
        "ignoredUnlabeledContactCount": unlabeled_contact_count,
        "overall": None,
        "byObservedPlayerCourtSide": None,
    }
