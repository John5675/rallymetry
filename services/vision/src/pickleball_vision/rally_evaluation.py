"""One-to-one evaluation of predicted rallies against human rally boundaries."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from pickleball_vision.config import RallySegmentationSettings
from pickleball_vision.rally_segmentation import RallyPrediction


@dataclass(frozen=True, slots=True)
class GroundTruthRally:
    """One paired human RALLY_START/RALLY_END interval."""

    rally_id: str
    start_event_id: str
    end_event_id: str
    start_frame: int
    end_frame: int
    start_timestamp_seconds: float
    end_timestamp_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "rallyId": self.rally_id,
            "startEventId": self.start_event_id,
            "endEventId": self.end_event_id,
            "startFrame": self.start_frame,
            "endFrame": self.end_frame,
            "startTimestamp": self.start_timestamp_seconds,
            "endTimestamp": self.end_timestamp_seconds,
        }


def _interval_iou(
    predicted: RallyPrediction,
    annotated: GroundTruthRally,
) -> float:
    intersection = max(
        0,
        min(predicted.end_frame, annotated.end_frame)
        - max(predicted.start_frame, annotated.start_frame)
        + 1,
    )
    union = (
        max(predicted.end_frame, annotated.end_frame)
        - min(predicted.start_frame, annotated.start_frame)
        + 1
    )
    return intersection / union


def _overlaps_scope(
    prediction: RallyPrediction,
    annotations: tuple[GroundTruthRally, ...],
    *,
    margin_frames: int,
) -> bool:
    return any(
        prediction.end_frame >= max(0, item.start_frame - margin_frames)
        and prediction.start_frame <= item.end_frame + margin_frames
        for item in annotations
    )


def _mean_or_none(values: tuple[float, ...]) -> float | None:
    return statistics.fmean(values) if values else None


def evaluate_rallies(
    predictions: tuple[RallyPrediction, ...],
    annotations: tuple[GroundTruthRally, ...],
    *,
    fps: float,
    settings: RallySegmentationSettings,
    annotations_complete: bool,
    evaluation_partition: str,
) -> dict[str, object]:
    """Match intervals once; sparse annotations never make unreviewed time negative."""

    if evaluation_partition not in {"development", "validation", "test"}:
        raise ValueError("evaluation partition must be development, validation, or test")
    margin_frames = round(settings.sparse_evaluation_margin_seconds * fps)
    evaluated_predictions = (
        predictions
        if annotations_complete
        else tuple(
            item
            for item in predictions
            if _overlaps_scope(item, annotations, margin_frames=margin_frames)
        )
    )
    ignored_predictions = tuple(item for item in predictions if item not in evaluated_predictions)
    boundary_tolerance_frames = round(settings.evaluation_boundary_tolerance_seconds * fps)
    proposals: list[tuple[float, int, int, RallyPrediction, GroundTruthRally, int, int]] = []
    for prediction in evaluated_predictions:
        for annotated in annotations:
            iou = _interval_iou(prediction, annotated)
            start_error_frames = prediction.start_frame - annotated.start_frame
            end_error_frames = prediction.end_frame - annotated.end_frame
            within_boundary_tolerance = (
                abs(start_error_frames) <= boundary_tolerance_frames
                and abs(end_error_frames) <= boundary_tolerance_frames
            )
            if iou >= settings.evaluation_minimum_iou or within_boundary_tolerance:
                proposals.append(
                    (
                        iou,
                        -(abs(start_error_frames) + abs(end_error_frames)),
                        -abs(start_error_frames),
                        prediction,
                        annotated,
                        start_error_frames,
                        end_error_frames,
                    )
                )
    proposals.sort(key=lambda item: item[:3], reverse=True)
    used_predictions: set[str] = set()
    used_annotations: set[str] = set()
    matches: list[dict[str, object]] = []
    start_errors: list[float] = []
    end_errors: list[float] = []
    for (
        iou,
        _combined_error,
        _start_rank,
        prediction,
        annotated,
        start_error_frames,
        end_error_frames,
    ) in proposals:
        if prediction.rally_id in used_predictions or annotated.rally_id in used_annotations:
            continue
        used_predictions.add(prediction.rally_id)
        used_annotations.add(annotated.rally_id)
        start_error_seconds = start_error_frames / fps
        end_error_seconds = end_error_frames / fps
        start_errors.append(start_error_seconds)
        end_errors.append(end_error_seconds)
        matches.append(
            {
                "predictedRallyId": prediction.rally_id,
                "annotatedRallyId": annotated.rally_id,
                "intervalIoU": iou,
                "startTimingErrorSeconds": start_error_seconds,
                "endTimingErrorSeconds": end_error_seconds,
                "absoluteStartTimingErrorSeconds": abs(start_error_seconds),
                "absoluteEndTimingErrorSeconds": abs(end_error_seconds),
            }
        )
    false_rallies = tuple(
        item for item in evaluated_predictions if item.rally_id not in used_predictions
    )
    missed_rallies = tuple(item for item in annotations if item.rally_id not in used_annotations)
    precision = len(matches) / len(evaluated_predictions) if evaluated_predictions else 0.0
    recall = len(matches) / len(annotations) if annotations else 0.0
    start_error_values = tuple(start_errors)
    end_error_values = tuple(end_errors)
    return {
        "evaluationAvailable": True,
        "evaluationPartition": evaluation_partition,
        "thresholdTuningPerformed": False,
        "heldOutThresholdTuningAllowed": evaluation_partition == "development",
        "annotationCoverage": {
            "mode": "complete_video" if annotations_complete else "reviewed_rally_windows_only",
            "unannotatedTimeTreatedAsNegative": False if not annotations_complete else None,
            "sparseWindowMarginSeconds": (
                None if annotations_complete else settings.sparse_evaluation_margin_seconds
            ),
            "ignoredPredictionCount": len(ignored_predictions),
            "ignoredPredictionIds": [item.rally_id for item in ignored_predictions],
        },
        "matchingConfiguration": {
            "minimumIntervalIoU": settings.evaluation_minimum_iou,
            "boundaryToleranceSeconds": settings.evaluation_boundary_tolerance_seconds,
            "oneToOneMatching": True,
        },
        "precision": precision,
        "recall": recall,
        "matchedRallyCount": len(matches),
        "missedRallyCount": len(missed_rallies),
        "falseRallyCount": len(false_rallies),
        "evaluatedPredictionCount": len(evaluated_predictions),
        "annotatedRallyCount": len(annotations),
        "matchedRallies": matches,
        "missedRallies": [item.as_dict() for item in missed_rallies],
        "falseRallies": [item.as_dict() for item in false_rallies],
        "startTimingError": {
            "meanSignedSeconds": _mean_or_none(start_error_values),
            "meanAbsoluteSeconds": _mean_or_none(tuple(abs(item) for item in start_error_values)),
            "maximumAbsoluteSeconds": (
                max((abs(item) for item in start_error_values), default=None)
            ),
        },
        "endTimingError": {
            "meanSignedSeconds": _mean_or_none(end_error_values),
            "meanAbsoluteSeconds": _mean_or_none(tuple(abs(item) for item in end_error_values)),
            "maximumAbsoluteSeconds": (max((abs(item) for item in end_error_values), default=None)),
        },
    }


def unavailable_evaluation(*, evaluation_partition: str) -> dict[str, object]:
    """Return an explicit no-ground-truth evaluation artifact."""

    return {
        "evaluationAvailable": False,
        "evaluationPartition": evaluation_partition,
        "thresholdTuningPerformed": False,
        "reason": "no ground-truth annotation artifact was supplied",
        "precision": None,
        "recall": None,
        "matchedRallyCount": None,
        "missedRallyCount": None,
        "falseRallyCount": None,
        "startTimingError": None,
        "endTimingError": None,
    }
