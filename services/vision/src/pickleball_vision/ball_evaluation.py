"""Fixed-split pickleball detector evaluation and strategy comparison."""

from __future__ import annotations

import json
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from pickleball_vision.ball_config import (
    BallExperimentConfiguration,
    BallInferenceStrategy,
    load_ball_experiment_configuration,
)
from pickleball_vision.ball_dataset import (
    BallCourtSide,
    BallDetectorDataset,
    DetectorDatasetFrame,
    GroundTruthBall,
    load_ball_detector_dataset,
)
from pickleball_vision.ball_detection import (
    BallDetection,
    BallDetector,
    BallFrameInference,
    infer_ball_frame,
    intersection_over_union,
)
from pickleball_vision.calibration import CourtCalibration, load_calibration
from pickleball_vision.dataset import DatasetSplit
from pickleball_vision.detectors import UltralyticsBallDetector
from pickleball_vision.errors import BallEvaluationError, OutputWriteError

BALL_EVALUATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FrameMatch:
    """One frame's one-to-one detector/ground-truth assignment."""

    matched_pairs: tuple[tuple[int, int, float], ...]
    unmatched_detection_indices: tuple[int, ...]
    unmatched_ground_truth_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BallEvaluationArtifacts:
    """Persisted metrics and raw predictions for one inference strategy."""

    metrics_path: Path
    detections_path: Path
    strategy_name: str
    partition: DatasetSplit
    frame_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "metrics_path": str(self.metrics_path),
            "detections_path": str(self.detections_path),
            "strategy_name": self.strategy_name,
            "partition": self.partition.value,
            "frame_count": self.frame_count,
        }


@dataclass(frozen=True, slots=True)
class BallStrategyComparisonArtifacts:
    """One comparison table whose strategies used identical fixed frames."""

    comparison_path: Path
    strategy_artifacts: tuple[BallEvaluationArtifacts, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "comparison_path": str(self.comparison_path),
            "strategies": [item.as_dict() for item in self.strategy_artifacts],
        }


def match_frame_detections(
    detections: tuple[BallDetection, ...],
    ground_truth: tuple[GroundTruthBall, ...],
    *,
    iou_threshold: float,
) -> FrameMatch:
    """Greedily match highest-confidence detections to unused ground-truth boxes."""

    available = set(range(len(ground_truth)))
    pairs: list[tuple[int, int, float]] = []
    unmatched_detections: list[int] = []
    for detection_index in sorted(
        range(len(detections)), key=lambda index: detections[index].confidence, reverse=True
    ):
        candidates = [
            (
                ground_truth_index,
                intersection_over_union(
                    detections[detection_index].bounding_box,
                    ground_truth[ground_truth_index].bounding_box,
                ),
            )
            for ground_truth_index in available
        ]
        best = max(candidates, key=lambda item: item[1], default=None)
        if best is None or best[1] < iou_threshold:
            unmatched_detections.append(detection_index)
            continue
        available.remove(best[0])
        pairs.append((detection_index, best[0], best[1]))
    return FrameMatch(
        matched_pairs=tuple(pairs),
        unmatched_detection_indices=tuple(unmatched_detections),
        unmatched_ground_truth_indices=tuple(sorted(available)),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _union_duration_seconds(frames: tuple[DetectorDatasetFrame, ...]) -> float:
    intervals: dict[str, set[tuple[float, float]]] = defaultdict(set)
    for frame in frames:
        intervals[frame.source_id].add((frame.clip_start_time_s, frame.clip_end_time_s))
    duration = 0.0
    for source_intervals in intervals.values():
        merged: list[list[float]] = []
        for start, end in sorted(source_intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        duration += sum(end - start for start, end in merged)
    return duration


def _side_metrics(
    frames: tuple[DetectorDatasetFrame, ...],
    matches: tuple[FrameMatch, ...],
    side: BallCourtSide,
) -> dict[str, object]:
    ground_truth_count = 0
    true_positives = 0
    positive_frames = 0
    covered_frames = 0
    for frame, match in zip(frames, matches, strict=True):
        side_indices = {
            index for index, ball in enumerate(frame.objects) if ball.court_side is side
        }
        if not side_indices:
            continue
        positive_frames += 1
        ground_truth_count += len(side_indices)
        matched_indices = {pair[1] for pair in match.matched_pairs} & side_indices
        true_positives += len(matched_indices)
        if matched_indices:
            covered_frames += 1
    return {
        "ground_truth_objects": ground_truth_count,
        "true_positives": true_positives,
        "false_negatives": ground_truth_count - true_positives,
        "recall": _ratio(true_positives, ground_truth_count),
        "positive_frames": positive_frames,
        "covered_positive_frames": covered_frames,
        "detection_coverage": _ratio(covered_frames, positive_frames),
        "side_source": "human_annotation_not_homography_projection",
    }


def calculate_detection_metrics(
    frames: tuple[DetectorDatasetFrame, ...],
    inferences: tuple[BallFrameInference, ...],
    *,
    iou_threshold: float,
) -> tuple[dict[str, object], tuple[FrameMatch, ...]]:
    """Compute documented object and frame metrics for an identical fixed frame set."""

    if len(frames) != len(inferences):
        raise BallEvaluationError("frame and inference counts differ")
    matches = tuple(
        match_frame_detections(
            inference.detections,
            frame.objects,
            iou_threshold=iou_threshold,
        )
        for frame, inference in zip(frames, inferences, strict=True)
    )
    true_positives = sum(len(match.matched_pairs) for match in matches)
    false_positives = sum(len(match.unmatched_detection_indices) for match in matches)
    false_negatives = sum(len(match.unmatched_ground_truth_indices) for match in matches)
    positive_frames = sum(bool(frame.objects) for frame in frames)
    covered_positive_frames = sum(
        bool(frame.objects) and bool(match.matched_pairs)
        for frame, match in zip(frames, matches, strict=True)
    )
    duration_s = _union_duration_seconds(frames)
    metrics: dict[str, object] = {
        "matching_iou_threshold": iou_threshold,
        "evaluated_frames": len(frames),
        "evaluated_duration_seconds": duration_s,
        "evaluated_duration_method": "union_of_fixed_source_clip_intervals",
        "ground_truth_objects": true_positives + false_negatives,
        "predicted_objects": true_positives + false_positives,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": _ratio(true_positives, true_positives + false_positives),
        "recall": _ratio(true_positives, true_positives + false_negatives),
        "false_positives_per_minute": (
            false_positives / (duration_s / 60) if duration_s > 0 else None
        ),
        "positive_frames": positive_frames,
        "covered_positive_frames": covered_positive_frames,
        "detection_coverage": _ratio(covered_positive_frames, positive_frames),
        "near_side": _side_metrics(frames, matches, BallCourtSide.NEAR),
        "far_side": _side_metrics(frames, matches, BallCourtSide.FAR),
    }
    return metrics, matches


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except (OSError, TypeError, ValueError) as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise OutputWriteError(str(path), reason=str(error)) from error


def _calibration_for_frame(
    frame: DetectorDatasetFrame,
    *,
    strategy: BallInferenceStrategy,
    config: BallExperimentConfiguration,
    cache: dict[str, CourtCalibration],
) -> CourtCalibration | None:
    if not strategy.mode.uses_court_roi:
        return None
    if frame.source_id not in cache:
        path = config.evaluation.calibrations_by_source_id.get(frame.source_id)
        if path is None:
            raise BallEvaluationError(
                f"strategy {strategy.name!r} requires a calibration for source {frame.source_id!r}"
            )
        cache[frame.source_id] = load_calibration(path)
    return cache[frame.source_id]


def _evaluate_loaded_strategy(
    *,
    config: BallExperimentConfiguration,
    dataset: BallDetectorDataset,
    detector: BallDetector,
    strategy: BallInferenceStrategy,
    partition: DatasetSplit,
    output_dir: Path,
) -> BallEvaluationArtifacts:
    frames = dataset.partition(partition)
    if not frames:
        raise BallEvaluationError(f"fixed {partition.value} partition is empty")
    output = output_dir.expanduser().resolve()
    metrics_path = output / "metrics.json"
    detections_path = output / "detections.json"
    if metrics_path.exists() or detections_path.exists():
        raise OutputWriteError(str(output), reason="evaluation output already exists")
    calibrations: dict[str, CourtCalibration] = {}
    inferences: list[BallFrameInference] = []
    for frame in frames:
        image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise BallEvaluationError(f"unable to decode evaluation image {frame.image_path}")
        typed_image = np.asarray(image, dtype=np.uint8)
        inferences.append(
            infer_ball_frame(
                typed_image,
                frame_number=frame.frame_number,
                timestamp_s=frame.timestamp_s,
                strategy=strategy,
                detector=detector,
                calibration=_calibration_for_frame(
                    frame, strategy=strategy, config=config, cache=calibrations
                ),
            )
        )
    inference_tuple = tuple(inferences)
    metrics, matches = calculate_detection_metrics(
        frames,
        inference_tuple,
        iou_threshold=config.evaluation.matching_iou_threshold,
    )
    common = {
        "schema_version": BALL_EVALUATION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": dataset.as_dict(),
        "partition": partition.value,
        "fixed_frame_record_ids": [frame.record_id for frame in frames],
        "model": detector.metadata.as_dict(),
        "strategy": strategy.as_dict(),
    }
    _write_json(
        detections_path,
        {
            **common,
            "record_type": "pickleball_detector_evaluation_detections",
            "temporal_processing": {"tracking": False, "interpolation": False},
            "frames": [
                {
                    "record_id": frame.record_id,
                    "source_id": frame.source_id,
                    "frame_number": frame.frame_number,
                    "timestamp_s": frame.timestamp_s,
                    "region_predictions": [
                        prediction.as_dict() for prediction in inference.region_predictions
                    ],
                    "detections": [item.as_dict() for item in inference.detections],
                    "matching": {
                        "matched_pairs": [
                            {
                                "detection_index": detection_index,
                                "annotation_index": annotation_index,
                                "iou": iou,
                            }
                            for detection_index, annotation_index, iou in match.matched_pairs
                        ],
                        "unmatched_detection_indices": list(match.unmatched_detection_indices),
                        "unmatched_annotation_indices": list(match.unmatched_ground_truth_indices),
                    },
                }
                for frame, inference, match in zip(frames, inference_tuple, matches, strict=True)
            ],
        },
    )
    _write_json(
        metrics_path,
        {
            **common,
            "record_type": "pickleball_detector_evaluation_metrics",
            "metric_definitions_version": 1,
            "metrics": metrics,
            "detections_path": str(detections_path),
        },
    )
    return BallEvaluationArtifacts(
        metrics_path=metrics_path,
        detections_path=detections_path,
        strategy_name=strategy.name,
        partition=partition,
        frame_count=len(frames),
    )


def evaluate_ball_detector(
    config_path: Path,
    *,
    weights_path: Path,
    strategy_name: str,
    partition: DatasetSplit,
    output_dir: Path,
    device: str = "auto",
    detector: BallDetector | None = None,
) -> BallEvaluationArtifacts:
    """Evaluate one strategy against one immutable validation or test partition."""

    config = load_ball_experiment_configuration(config_path)
    dataset = load_ball_detector_dataset(
        dataset_version=config.dataset.version,
        split_manifest_path=config.dataset.split_manifest_path,
        annotations_path=config.dataset.annotations_path,
    )
    active_detector = detector or UltralyticsBallDetector(
        weights_path,
        model_version=config.model.version,
        device=device,
    )
    return _evaluate_loaded_strategy(
        config=config,
        dataset=dataset,
        detector=active_detector,
        strategy=config.strategy(strategy_name),
        partition=partition,
        output_dir=output_dir,
    )


def compare_ball_inference_strategies(
    config_path: Path,
    *,
    weights_path: Path,
    partition: DatasetSplit,
    output_dir: Path,
    strategy_names: tuple[str, ...] | None = None,
    device: str = "auto",
    detector: BallDetector | None = None,
) -> BallStrategyComparisonArtifacts:
    """Evaluate named strategies on the exact same fixed frames and write a table."""

    config = load_ball_experiment_configuration(config_path)
    dataset = load_ball_detector_dataset(
        dataset_version=config.dataset.version,
        split_manifest_path=config.dataset.split_manifest_path,
        annotations_path=config.dataset.annotations_path,
    )
    selected = (
        tuple(config.strategy(name) for name in strategy_names)
        if strategy_names
        else config.strategies
    )
    if len(selected) < 2:
        raise BallEvaluationError("strategy comparison requires at least two strategies")
    active_detector = detector or UltralyticsBallDetector(
        weights_path,
        model_version=config.model.version,
        device=device,
    )
    output = output_dir.expanduser().resolve()
    comparison_path = output / "comparison.json"
    if comparison_path.exists():
        raise OutputWriteError(str(comparison_path), reason="comparison output already exists")
    artifacts = tuple(
        _evaluate_loaded_strategy(
            config=config,
            dataset=dataset,
            detector=active_detector,
            strategy=strategy,
            partition=partition,
            output_dir=output / strategy.name,
        )
        for strategy in selected
    )
    rows: list[dict[str, object]] = []
    expected_ids: list[str] | None = None
    for artifact in artifacts:
        payload = json.loads(artifact.metrics_path.read_text(encoding="utf-8"))
        frame_ids = list(payload["fixed_frame_record_ids"])
        if expected_ids is None:
            expected_ids = frame_ids
        elif frame_ids != expected_ids:
            raise BallEvaluationError("strategies were not evaluated on identical fixed frames")
        rows.append(
            {
                "strategy": artifact.strategy_name,
                **dict(payload["metrics"]),
                "metrics_path": str(artifact.metrics_path),
            }
        )
    _write_json(
        comparison_path,
        {
            "schema_version": BALL_EVALUATION_SCHEMA_VERSION,
            "record_type": "pickleball_inference_strategy_comparison",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset": dataset.as_dict(),
            "partition": partition.value,
            "fixed_frame_record_ids": expected_ids,
            "model": active_detector.metadata.as_dict(),
            "strategies_compared": [strategy.as_dict() for strategy in selected],
            "rows": rows,
        },
    )
    return BallStrategyComparisonArtifacts(
        comparison_path=comparison_path,
        strategy_artifacts=artifacts,
    )
