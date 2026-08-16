"""Model-neutral raw ball observations and spatial inference strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pickleball_vision.ball_config import BallInferenceStrategy
from pickleball_vision.calibration import CourtCalibration
from pickleball_vision.court import CourtPoint
from pickleball_vision.errors import BallInferenceError, DetectionInputError
from pickleball_vision.person_detection import BoundingBox
from pickleball_vision.video import Image

BALL_DETECTION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BallDetectorMetadata:
    """Custom-model provenance retained with raw observations."""

    adapter: str
    framework: str
    framework_version: str | None
    model_version: str
    weights_path: Path
    weights_sha256: str
    device: str

    def as_dict(self) -> dict[str, object]:
        return {
            "adapter": self.adapter,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "model_version": self.model_version,
            "weights_path": str(self.weights_path),
            "weights_sha256": self.weights_sha256,
            "device": self.device,
            "class_name": "pickleball",
            "class_id": 0,
        }


@dataclass(frozen=True, slots=True)
class BallModelPrediction:
    """One model-returned box in the supplied crop's pixel coordinates."""

    bounding_box: BoundingBox
    confidence: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("ball prediction confidence must be finite and in [0, 1]")


class BallDetector(Protocol):
    """Model adapter boundary; spatial crops and temporal data stay outside it."""

    @property
    def metadata(self) -> BallDetectorMetadata:
        """Return model, weights, framework, and effective-device provenance."""

        ...

    def predict(
        self,
        image: Image,
        *,
        inference_size_px: int,
        minimum_confidence: float,
        nms_iou_threshold: float,
        maximum_detections: int,
    ) -> tuple[BallModelPrediction, ...]:
        """Return crop-local pickleball predictions without temporal association."""

        ...


@dataclass(frozen=True, slots=True)
class InferenceRegion:
    """An integer source-frame crop with an exclusive right/bottom boundary."""

    region_id: str
    left_px: int
    top_px: int
    right_px: int
    bottom_px: int
    kind: str

    @property
    def width(self) -> int:
        return self.right_px - self.left_px

    @property
    def height(self) -> int:
        return self.bottom_px - self.top_px

    def as_dict(self) -> dict[str, object]:
        return {
            "region_id": self.region_id,
            "kind": self.kind,
            "bounding_box": {
                "left_px": self.left_px,
                "top_px": self.top_px,
                "right_px": self.right_px,
                "bottom_px": self.bottom_px,
            },
        }


@dataclass(frozen=True, slots=True)
class BallRegionPrediction:
    """Append-only model output translated from one crop into source pixels."""

    prediction_id: str
    region_id: str
    bounding_box: BoundingBox
    confidence: float

    def as_dict(self) -> dict[str, object]:
        return {
            "prediction_id": self.prediction_id,
            "region_id": self.region_id,
            "bounding_box": self.bounding_box.as_dict(),
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class BallDetection:
    """Frame-local raw detection after overlap-only cross-crop deduplication."""

    detection_id: str
    frame_number: int
    timestamp_s: float
    bounding_box: BoundingBox
    confidence: float
    supporting_prediction_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "detection_id": self.detection_id,
            "frame_number": self.frame_number,
            "timestamp_s": self.timestamp_s,
            "bounding_box": self.bounding_box.as_dict(),
            "confidence": self.confidence,
            "supporting_prediction_ids": list(self.supporting_prediction_ids),
            "observation_status": "observed",
            "temporal_track_id": None,
        }


@dataclass(frozen=True, slots=True)
class BallFrameInference:
    """All region evidence and deduplicated detections for one source frame."""

    frame_number: int
    timestamp_s: float
    regions: tuple[InferenceRegion, ...]
    region_predictions: tuple[BallRegionPrediction, ...]
    detections: tuple[BallDetection, ...]


def intersection_over_union(left: BoundingBox, right: BoundingBox) -> float:
    """Return axis-aligned IoU in a shared image coordinate system."""

    intersection_width = max(
        0.0,
        min(left.right_px, right.right_px) - max(left.left_px, right.left_px),
    )
    intersection_height = max(
        0.0, min(left.bottom_px, right.bottom_px) - max(left.top_px, right.top_px)
    )
    intersection = intersection_width * intersection_height
    left_area = (left.right_px - left.left_px) * (left.bottom_px - left.top_px)
    right_area = (right.right_px - right.left_px) * (right.bottom_px - right.top_px)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _court_roi(
    *,
    frame_width: int,
    frame_height: int,
    calibration: CourtCalibration,
    margin_px: int,
) -> InferenceRegion:
    if (
        calibration.source.frame_width_px != frame_width
        or calibration.source.frame_height_px != frame_height
    ):
        raise DetectionInputError(
            "court calibration dimensions do not match the ball-inference frame"
        )
    corners = (
        CourtPoint(0.0, 0.0),
        CourtPoint(calibration.court.width_m, 0.0),
        CourtPoint(calibration.court.width_m, calibration.court.length_m),
        CourtPoint(0.0, calibration.court.length_m),
    )
    projected = tuple(calibration.court_to_image(point) for point in corners)
    left = max(0, math.floor(min(point.x_px for point in projected) - margin_px))
    top = max(0, math.floor(min(point.y_px for point in projected) - margin_px))
    right = min(frame_width, math.ceil(max(point.x_px for point in projected) + margin_px))
    bottom = min(frame_height, math.ceil(max(point.y_px for point in projected) + margin_px))
    if right <= left or bottom <= top:
        raise DetectionInputError("projected primary-court ROI does not intersect the frame")
    return InferenceRegion("court-roi", left, top, right, bottom, "court_roi")


def _tile_positions(start: int, stop: int, tile_size: int, overlap: float) -> tuple[int, ...]:
    extent = stop - start
    if extent <= tile_size:
        return (start,)
    stride = max(1, round(tile_size * (1 - overlap)))
    positions = list(range(start, stop - tile_size + 1, stride))
    last = stop - tile_size
    if positions[-1] != last:
        positions.append(last)
    return tuple(positions)


def build_inference_regions(
    *,
    frame_width: int,
    frame_height: int,
    strategy: BallInferenceStrategy,
    calibration: CourtCalibration | None,
) -> tuple[InferenceRegion, ...]:
    """Create deterministic full-frame, court-ROI, and overlapping tile regions."""

    if frame_width < 1 or frame_height < 1:
        raise DetectionInputError("ball-inference frame dimensions must be positive")
    if strategy.mode.uses_court_roi:
        if calibration is None:
            raise DetectionInputError(f"strategy {strategy.name!r} requires a court calibration")
        base = _court_roi(
            frame_width=frame_width,
            frame_height=frame_height,
            calibration=calibration,
            margin_px=strategy.court_roi_margin_px,
        )
    else:
        base = InferenceRegion("full-frame", 0, 0, frame_width, frame_height, "full_frame")
    if not strategy.mode.uses_tiles:
        return (base,)

    assert strategy.tile_size_px is not None
    tile_width = min(strategy.tile_size_px, base.width)
    tile_height = min(strategy.tile_size_px, base.height)
    x_positions = _tile_positions(
        base.left_px, base.right_px, tile_width, strategy.tile_overlap_fraction
    )
    y_positions = _tile_positions(
        base.top_px, base.bottom_px, tile_height, strategy.tile_overlap_fraction
    )
    return tuple(
        InferenceRegion(
            region_id=f"tile-{row:03d}-{column:03d}",
            left_px=left,
            top_px=top,
            right_px=left + tile_width,
            bottom_px=top + tile_height,
            kind="court_tile" if strategy.mode.uses_court_roi else "full_frame_tile",
        )
        for row, top in enumerate(y_positions)
        for column, left in enumerate(x_positions)
    )


def _deduplicate(
    predictions: tuple[BallRegionPrediction, ...],
    *,
    frame_number: int,
    timestamp_s: float,
    iou_threshold: float,
    maximum_detections: int,
) -> tuple[BallDetection, ...]:
    retained: list[BallRegionPrediction] = []
    support: dict[str, list[str]] = {}
    for prediction in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        duplicate = next(
            (
                candidate
                for candidate in retained
                if intersection_over_union(prediction.bounding_box, candidate.bounding_box)
                >= iou_threshold
            ),
            None,
        )
        if duplicate is None:
            if len(retained) >= maximum_detections:
                continue
            retained.append(prediction)
            support[prediction.prediction_id] = [prediction.prediction_id]
        else:
            support[duplicate.prediction_id].append(prediction.prediction_id)
    return tuple(
        BallDetection(
            detection_id=f"frame-{frame_number:09d}-ball-{index:04d}",
            frame_number=frame_number,
            timestamp_s=timestamp_s,
            bounding_box=prediction.bounding_box,
            confidence=prediction.confidence,
            supporting_prediction_ids=tuple(support[prediction.prediction_id]),
        )
        for index, prediction in enumerate(retained)
    )


def infer_ball_frame(
    image: Image,
    *,
    frame_number: int,
    timestamp_s: float,
    strategy: BallInferenceStrategy,
    detector: BallDetector,
    calibration: CourtCalibration | None = None,
) -> BallFrameInference:
    """Run one strategy on one image without any temporal interpolation or tracking."""

    frame_height, frame_width = image.shape[:2]
    regions = build_inference_regions(
        frame_width=frame_width,
        frame_height=frame_height,
        strategy=strategy,
        calibration=calibration,
    )
    translated: list[BallRegionPrediction] = []
    try:
        for region in regions:
            crop = image[region.top_px : region.bottom_px, region.left_px : region.right_px]
            predictions = detector.predict(
                crop,
                inference_size_px=strategy.inference_size_px,
                minimum_confidence=strategy.minimum_confidence,
                nms_iou_threshold=strategy.model_nms_iou_threshold,
                maximum_detections=strategy.maximum_detections,
            )
            for index, prediction in enumerate(predictions):
                local = prediction.bounding_box
                left = min(max(local.left_px, 0.0), float(region.width))
                top = min(max(local.top_px, 0.0), float(region.height))
                right = min(max(local.right_px, 0.0), float(region.width))
                bottom = min(max(local.bottom_px, 0.0), float(region.height))
                if right <= left or bottom <= top:
                    continue
                translated.append(
                    BallRegionPrediction(
                        prediction_id=(
                            f"frame-{frame_number:09d}-{region.region_id}-prediction-{index:04d}"
                        ),
                        region_id=region.region_id,
                        bounding_box=BoundingBox(
                            left_px=left + region.left_px,
                            top_px=top + region.top_px,
                            right_px=right + region.left_px,
                            bottom_px=bottom + region.top_px,
                        ),
                        confidence=prediction.confidence,
                    )
                )
    except Exception as error:
        if isinstance(error, BallInferenceError):
            raise
        raise BallInferenceError(str(error), operation="spatial_inference") from error
    translated_tuple = tuple(translated)
    return BallFrameInference(
        frame_number=frame_number,
        timestamp_s=timestamp_s,
        regions=regions,
        region_predictions=translated_tuple,
        detections=_deduplicate(
            translated_tuple,
            frame_number=frame_number,
            timestamp_s=timestamp_s,
            iou_threshold=strategy.merge_iou_threshold,
            maximum_detections=strategy.maximum_detections,
        ),
    )
