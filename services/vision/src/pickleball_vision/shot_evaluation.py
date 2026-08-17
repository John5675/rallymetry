"""Post-inference evaluation for the fixed initial shot-class vocabulary."""

from __future__ import annotations

from dataclasses import dataclass

from pickleball_vision.config import ShotClassificationSettings
from pickleball_vision.shot_reconstruction import SHOT_TYPES, Shot, ShotType


@dataclass(frozen=True, slots=True)
class GroundTruthShot:
    """One human-authored shot label tied to a source contact frame/time."""

    event_id: str
    frame: int
    timestamp_seconds: float
    shot_type: ShotType

    def as_dict(self) -> dict[str, object]:
        return {
            "eventId": self.event_id,
            "frame": self.frame,
            "timestamp": self.timestamp_seconds,
            "shotType": self.shot_type.value,
        }


def evaluate_shots(
    shots: tuple[Shot, ...],
    annotations: tuple[GroundTruthShot, ...],
    *,
    settings: ShotClassificationSettings,
    evaluation_partition: str = "validation",
    unsupported_human_label_count: int = 0,
) -> dict[str, object]:
    """Match by contact time only, then calculate fixed-label class metrics."""

    if evaluation_partition not in {"development", "validation", "test"}:
        raise ValueError("evaluation partition must be development, validation, or test")
    if not annotations:
        return unavailable_shot_evaluation(
            evaluation_partition=evaluation_partition,
            unsupported_human_label_count=unsupported_human_label_count,
        )
    tolerance_seconds = settings.evaluation_tolerance_ms / 1000.0
    proposals: list[tuple[float, float, Shot, GroundTruthShot]] = []
    for predicted in shots:
        for annotated in annotations:
            delta = predicted.contact_timestamp_seconds - annotated.timestamp_seconds
            if abs(delta) <= tolerance_seconds:
                proposals.append((abs(delta), -predicted.confidence, predicted, annotated))
    proposals.sort(key=lambda item: item[:2])
    used_shots: set[str] = set()
    used_annotations: set[str] = set()
    confusion = {
        actual.value: {predicted.value: 0 for predicted in SHOT_TYPES} for actual in SHOT_TYPES
    }
    matches: list[dict[str, object]] = []
    for absolute_error, _negative_confidence, predicted, annotated in proposals:
        if predicted.shot_id in used_shots or annotated.event_id in used_annotations:
            continue
        used_shots.add(predicted.shot_id)
        used_annotations.add(annotated.event_id)
        confusion[annotated.shot_type.value][predicted.shot_type.value] += 1
        matches.append(
            {
                "predictedShotId": predicted.shot_id,
                "predictedContactId": predicted.contact_id,
                "annotatedEventId": annotated.event_id,
                "actualShotType": annotated.shot_type.value,
                "predictedShotType": predicted.shot_type.value,
                "predictionConfidence": predicted.confidence,
                "correct": predicted.shot_type is annotated.shot_type,
                "timingErrorMs": (predicted.contact_timestamp_seconds - annotated.timestamp_seconds)
                * 1000,
                "absoluteTimingErrorMs": absolute_error * 1000,
            }
        )
    per_class: dict[str, object] = {}
    for shot_type in SHOT_TYPES:
        label = shot_type.value
        true_positive = confusion[label][label]
        predicted_count = sum(confusion[actual.value][label] for actual in SHOT_TYPES)
        matched_actual_count = sum(confusion[label].values())
        actual_count = sum(item.shot_type is shot_type for item in annotations)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / actual_count if actual_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": actual_count,
            "matchedSupport": matched_actual_count,
            "predictedCount": predicted_count,
            "truePositiveCount": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    missed = tuple(item for item in annotations if item.event_id not in used_annotations)
    correct = sum(bool(item["correct"]) for item in matches)
    return {
        "evaluationAvailable": True,
        "evaluationPartition": evaluation_partition,
        "thresholdTuningPerformed": False,
        "heldOutThresholdTuningAllowed": evaluation_partition == "development",
        "supportedClasses": [item.value for item in SHOT_TYPES],
        "groundTruthShotCount": len(annotations),
        "unsupportedHumanLabelCount": unsupported_human_label_count,
        "predictedShotCount": len(shots),
        "matchedShotCount": len(matches),
        "missedGroundTruthShotCount": len(missed),
        "matchCoverage": len(matches) / len(annotations),
        "accuracy": correct / len(matches) if matches else 0.0,
        "unknownRate": (
            sum(item.shot_type is ShotType.UNKNOWN for item in shots) / len(shots) if shots else 0.0
        ),
        "matchedUnknownRate": (
            sum(item["predictedShotType"] == ShotType.UNKNOWN.value for item in matches)
            / len(matches)
            if matches
            else 0.0
        ),
        "perClass": per_class,
        "confusionMatrix": {
            "rowLabel": "actualShotType",
            "columnLabel": "predictedShotType",
            "labels": [item.value for item in SHOT_TYPES],
            "counts": confusion,
        },
        "matchingConfiguration": {
            "oneToOneMatching": True,
            "timingToleranceMs": settings.evaluation_tolerance_ms,
            "shotTypeUsedForTemporalMatching": False,
            "precisionScope": "time_matched_predictions_only",
            "recallDenominator": "all_supported_ground_truth_shots",
            "unmatchedPredictionsCountedAsFalsePositives": False,
            "reason": "ground truth can cover representative rallies rather than the full video",
        },
        "matches": matches,
        "missedGroundTruthShots": [item.as_dict() for item in missed],
    }


def unavailable_shot_evaluation(
    *,
    evaluation_partition: str,
    unsupported_human_label_count: int = 0,
) -> dict[str, object]:
    """Return an explicit no-ground-truth evaluation state."""

    return {
        "evaluationAvailable": False,
        "evaluationPartition": evaluation_partition,
        "thresholdTuningPerformed": False,
        "reason": "no supported human shotType annotations supplied",
        "unsupportedHumanLabelCount": unsupported_human_label_count,
        "accuracy": None,
        "perClass": None,
        "confusionMatrix": None,
        "unknownRate": None,
    }
