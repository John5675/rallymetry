"""One-to-one evaluation of visual-only and audio-fused contact candidates."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from pickleball_vision.config import ContactDetectionSettings
from pickleball_vision.contact_detection import ContactCandidate


@dataclass(frozen=True, slots=True)
class GroundTruthContact:
    """One human-authored serve or paddle contact on the video timeline."""

    event_id: str
    event_type: str
    frame: int
    timestamp_seconds: float
    annotated_player_id: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "eventId": self.event_id,
            "eventType": self.event_type,
            "frame": self.frame,
            "timestamp": self.timestamp_seconds,
            "annotatedPlayerId": self.annotated_player_id,
        }


@dataclass(frozen=True, slots=True)
class ContactReviewedInterval:
    """Human-reviewed rally interval used to bound sparse evaluation."""

    interval_id: str
    start_frame: int
    end_frame: int


def _in_sparse_scope(
    candidate: ContactCandidate,
    annotations: tuple[GroundTruthContact, ...],
    reviewed_intervals: tuple[ContactReviewedInterval, ...],
    *,
    fps: float,
    margin_seconds: float,
) -> bool:
    margin_frames = round(margin_seconds * fps)
    if reviewed_intervals:
        return any(
            max(0, item.start_frame - margin_frames)
            <= candidate.frame
            <= item.end_frame + margin_frames
            for item in reviewed_intervals
        )
    return any(abs(candidate.frame - item.frame) <= margin_frames for item in annotations)


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _metric(payload: dict[str, object], field: str) -> float:
    value = payload[field]
    if not isinstance(value, float):
        raise AssertionError(f"{field} metric must be a float")
    return value


def _evaluate_mode(
    candidates: tuple[ContactCandidate, ...],
    annotations: tuple[GroundTruthContact, ...],
    *,
    score_field: str,
    settings: ContactDetectionSettings,
    fps: float,
    annotations_complete: bool,
    reviewed_intervals: tuple[ContactReviewedInterval, ...],
) -> dict[str, object]:
    accepted = tuple(
        candidate
        for candidate in candidates
        if (
            candidate.visual_confidence
            if score_field == "visualConfidence"
            else candidate.fused_confidence
        )
        >= settings.accepted_confidence
    )
    evaluated = (
        accepted
        if annotations_complete
        else tuple(
            item
            for item in accepted
            if _in_sparse_scope(
                item,
                annotations,
                reviewed_intervals,
                fps=fps,
                margin_seconds=settings.sparse_evaluation_margin_seconds,
            )
        )
    )
    ignored = tuple(item for item in accepted if item not in evaluated)
    tolerance_seconds = settings.evaluation_tolerance_ms / 1000.0
    proposals: list[tuple[float, float, ContactCandidate, GroundTruthContact]] = []
    for predicted in evaluated:
        score = (
            predicted.visual_confidence
            if score_field == "visualConfidence"
            else predicted.fused_confidence
        )
        for annotated in annotations:
            error = predicted.timestamp_seconds - annotated.timestamp_seconds
            if abs(error) <= tolerance_seconds:
                proposals.append((abs(error), -score, predicted, annotated))
    proposals.sort(key=lambda item: item[:2])
    used_predictions: set[str] = set()
    used_annotations: set[str] = set()
    matches: list[dict[str, object]] = []
    signed_errors_ms: list[float] = []
    for absolute_error, negative_score, predicted, annotated in proposals:
        if predicted.contact_id in used_predictions or annotated.event_id in used_annotations:
            continue
        used_predictions.add(predicted.contact_id)
        used_annotations.add(annotated.event_id)
        signed_ms = (predicted.timestamp_seconds - annotated.timestamp_seconds) * 1000
        signed_errors_ms.append(signed_ms)
        matches.append(
            {
                "predictedContactId": predicted.contact_id,
                "annotatedEventId": annotated.event_id,
                "annotatedEventType": annotated.event_type,
                "annotatedPlayerId": annotated.annotated_player_id,
                "annotatedPlayerUsedForMatching": False,
                "timingErrorMs": signed_ms,
                "absoluteTimingErrorMs": absolute_error * 1000,
                "score": -negative_score,
            }
        )
    false_candidates = tuple(item for item in evaluated if item.contact_id not in used_predictions)
    missed = tuple(item for item in annotations if item.event_id not in used_annotations)
    precision = len(matches) / len(evaluated) if evaluated else 0.0
    recall = len(matches) / len(annotations) if annotations else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    absolute_errors = [abs(value) for value in signed_errors_ms]
    return {
        "scoreField": score_field,
        "acceptanceThreshold": settings.accepted_confidence,
        "candidateCountAboveThreshold": len(accepted),
        "evaluatedCandidateCount": len(evaluated),
        "ignoredCandidateCount": len(ignored),
        "ignoredCandidateIds": [item.contact_id for item in ignored],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matchedContactCount": len(matches),
        "falseContactCount": len(false_candidates),
        "missedContactCount": len(missed),
        "matches": matches,
        "falseContacts": [item.as_dict() for item in false_candidates],
        "missedContacts": [item.as_dict() for item in missed],
        "contactTimingError": {
            "meanSignedMs": _mean_or_none(signed_errors_ms),
            "meanAbsoluteMs": _mean_or_none(absolute_errors),
            "maximumAbsoluteMs": max(absolute_errors, default=None),
        },
    }


def evaluate_contacts(
    candidates: tuple[ContactCandidate, ...],
    annotations: tuple[GroundTruthContact, ...],
    *,
    fps: float,
    settings: ContactDetectionSettings,
    annotations_complete: bool,
    reviewed_intervals: tuple[ContactReviewedInterval, ...] = (),
    evaluation_partition: str = "validation",
) -> dict[str, object]:
    """Compare identical visual candidates using visual and fused confidence."""

    if evaluation_partition not in {"development", "validation", "test"}:
        raise ValueError("evaluation partition must be development, validation, or test")
    if not annotations:
        return unavailable_contact_evaluation(evaluation_partition=evaluation_partition)
    vision = _evaluate_mode(
        candidates,
        annotations,
        score_field="visualConfidence",
        settings=settings,
        fps=fps,
        annotations_complete=annotations_complete,
        reviewed_intervals=reviewed_intervals,
    )
    fused = _evaluate_mode(
        candidates,
        annotations,
        score_field="fusedConfidence",
        settings=settings,
        fps=fps,
        annotations_complete=annotations_complete,
        reviewed_intervals=reviewed_intervals,
    )
    return {
        "evaluationAvailable": True,
        "evaluationPartition": evaluation_partition,
        "thresholdTuningPerformed": False,
        "heldOutThresholdTuningAllowed": evaluation_partition == "development",
        "annotationCoverage": {
            "mode": "complete_video" if annotations_complete else "reviewed_windows_only",
            "reviewedRallyIntervalCount": len(reviewed_intervals),
            "unannotatedTimeTreatedAsNegative": annotations_complete,
            "sparseWindowMarginSeconds": (
                None if annotations_complete else settings.sparse_evaluation_margin_seconds
            ),
        },
        "groundTruthEventTypes": ["SERVE_CONTACT", "PADDLE_CONTACT"],
        "matchingConfiguration": {
            "oneToOneMatching": True,
            "timingToleranceMs": settings.evaluation_tolerance_ms,
            "playerIdentityUsedForMatching": False,
        },
        "visionOnly": vision,
        "visionPlusAudio": fused,
        "comparison": {
            "precisionDelta": _metric(fused, "precision") - _metric(vision, "precision"),
            "recallDelta": _metric(fused, "recall") - _metric(vision, "recall"),
            "f1Delta": _metric(fused, "f1") - _metric(vision, "f1"),
            "sameVisualCandidateSet": True,
            "audioCreatedVisualCandidates": False,
        },
    }


def unavailable_contact_evaluation(*, evaluation_partition: str) -> dict[str, object]:
    """Return an explicit no-ground-truth evaluation state."""

    return {
        "evaluationAvailable": False,
        "evaluationPartition": evaluation_partition,
        "thresholdTuningPerformed": False,
        "reason": "no human SERVE_CONTACT or PADDLE_CONTACT annotations supplied",
        "visionOnly": None,
        "visionPlusAudio": None,
        "comparison": None,
    }
