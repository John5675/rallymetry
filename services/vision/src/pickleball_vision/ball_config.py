"""Versioned external configuration for ball-detector experiments."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from pickleball_vision.config import VALID_INFERENCE_DEVICE
from pickleball_vision.errors import DatasetInputError

BALL_EXPERIMENT_CONFIG_SCHEMA_VERSION = 1
SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class BallInferenceMode(StrEnum):
    """Supported spatial inference strategies; none performs temporal tracking."""

    FULL_FRAME = "full_frame"
    COURT_ROI = "court_roi"
    TILED = "tiled"
    COURT_TILED = "court_tiled"

    @property
    def uses_court_roi(self) -> bool:
        return self in {self.COURT_ROI, self.COURT_TILED}

    @property
    def uses_tiles(self) -> bool:
        return self in {self.TILED, self.COURT_TILED}


@dataclass(frozen=True, slots=True)
class BallDatasetReference:
    """Immutable references and human-owned version for one detector dataset."""

    version: str
    split_manifest_path: Path
    annotations_path: Path

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "split_manifest": str(self.split_manifest_path),
            "annotations": str(self.annotations_path),
        }


@dataclass(frozen=True, slots=True)
class BallModelSettings:
    """Custom-model identity separated from a mutable weights filename."""

    version: str
    base_model: str

    def as_dict(self) -> dict[str, object]:
        return {"version": self.version, "base_model": self.base_model}


@dataclass(frozen=True, slots=True)
class BallTrainingSettings:
    """Reproducible Ultralytics training inputs."""

    epochs: int
    image_size_px: int
    batch_size: int
    device: str
    workers: int
    seed: int
    deterministic: bool
    patience_epochs: int

    def as_dict(self) -> dict[str, object]:
        return {
            "epochs": self.epochs,
            "image_size_px": self.image_size_px,
            "batch_size": self.batch_size,
            "device": self.device,
            "workers": self.workers,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "patience_epochs": self.patience_epochs,
        }


@dataclass(frozen=True, slots=True)
class BallInferenceStrategy:
    """One named, comparable spatial inference configuration."""

    name: str
    mode: BallInferenceMode
    inference_size_px: int
    minimum_confidence: float
    model_nms_iou_threshold: float
    maximum_detections: int
    tile_size_px: int | None
    tile_overlap_fraction: float
    court_roi_margin_px: int
    merge_iou_threshold: float

    def __post_init__(self) -> None:
        if SAFE_VERSION.fullmatch(self.name) is None:
            raise DatasetInputError(
                "inference strategy name must use letters, numbers, period, underscore, or hyphen"
            )
        if self.inference_size_px < 320:
            raise DatasetInputError("inference_size_px must be at least 320")
        if not 0 <= self.minimum_confidence <= 1 or not math.isfinite(self.minimum_confidence):
            raise DatasetInputError("minimum_confidence must be finite and between 0 and 1")
        for value, field in (
            (self.model_nms_iou_threshold, "model_nms_iou_threshold"),
            (self.merge_iou_threshold, "merge_iou_threshold"),
        ):
            if not 0 <= value <= 1 or not math.isfinite(value):
                raise DatasetInputError(f"{field} must be finite and between 0 and 1")
        if self.maximum_detections < 1:
            raise DatasetInputError("maximum_detections must be at least 1")
        if self.mode.uses_tiles and (self.tile_size_px is None or self.tile_size_px < 64):
            raise DatasetInputError("tiled strategies require tile_size_px of at least 64")
        if not self.mode.uses_tiles and self.tile_size_px is not None:
            raise DatasetInputError("tile_size_px is valid only for tiled strategies")
        if not 0 <= self.tile_overlap_fraction < 0.9 or not math.isfinite(
            self.tile_overlap_fraction
        ):
            raise DatasetInputError("tile_overlap_fraction must be in [0, 0.9)")
        if self.court_roi_margin_px < 0:
            raise DatasetInputError("court_roi_margin_px must be non-negative")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mode": self.mode.value,
            "inference_size_px": self.inference_size_px,
            "minimum_confidence": self.minimum_confidence,
            "model_nms_iou_threshold": self.model_nms_iou_threshold,
            "maximum_detections": self.maximum_detections,
            "tile_size_px": self.tile_size_px,
            "tile_overlap_fraction": self.tile_overlap_fraction,
            "court_roi_margin_px": self.court_roi_margin_px,
            "merge_iou_threshold": self.merge_iou_threshold,
        }


@dataclass(frozen=True, slots=True)
class BallEvaluationSettings:
    """Fixed annotation matching settings plus per-source ROI calibration."""

    matching_iou_threshold: float
    calibrations_by_source_id: dict[str, Path]

    def __post_init__(self) -> None:
        if not 0 < self.matching_iou_threshold <= 1 or not math.isfinite(
            self.matching_iou_threshold
        ):
            raise DatasetInputError("matching_iou_threshold must be finite and in (0, 1]")

    def as_dict(self) -> dict[str, object]:
        return {
            "matching_iou_threshold": self.matching_iou_threshold,
            "calibrations_by_source_id": {
                key: str(value) for key, value in sorted(self.calibrations_by_source_id.items())
            },
        }


@dataclass(frozen=True, slots=True)
class BallExperimentConfiguration:
    """Complete external contract shared by training and evaluation."""

    path: Path
    experiment_name: str
    dataset: BallDatasetReference
    model: BallModelSettings
    training: BallTrainingSettings
    strategies: tuple[BallInferenceStrategy, ...]
    evaluation: BallEvaluationSettings
    schema_version: int = BALL_EXPERIMENT_CONFIG_SCHEMA_VERSION

    def strategy(self, name: str) -> BallInferenceStrategy:
        for strategy in self.strategies:
            if strategy.name == name:
                return strategy
        choices = ", ".join(item.name for item in self.strategies)
        raise DatasetInputError(f"unknown inference strategy {name!r}; choose one of: {choices}")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "record_type": "pickleball_detector_experiment_configuration",
            "experiment_name": self.experiment_name,
            "dataset": self.dataset.as_dict(),
            "model": self.model.as_dict(),
            "training": self.training.as_dict(),
            "inference": {"strategies": [item.as_dict() for item in self.strategies]},
            "evaluation": self.evaluation.as_dict(),
        }


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return cast(list[object], value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _optional_integer(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _integer(value, field)


def _resolved_path(value: object, field: str, *, base_dir: Path) -> Path:
    raw = Path(_string(value, field)).expanduser()
    return (base_dir / raw).resolve() if not raw.is_absolute() else raw.resolve()


def _validated_version(value: object, field: str) -> str:
    result = _string(value, field)
    if SAFE_VERSION.fullmatch(result) is None:
        raise ValueError(
            f"{field} must start with an alphanumeric and use only letters, numbers, "
            "period, underscore, or hyphen"
        )
    return result


def _parse_strategy(value: object, index: int) -> BallInferenceStrategy:
    field = f"inference.strategies[{index}]"
    raw = _object(value, field)
    try:
        mode = BallInferenceMode(_string(raw.get("mode"), f"{field}.mode"))
    except ValueError as error:
        choices = ", ".join(item.value for item in BallInferenceMode)
        raise ValueError(f"{field}.mode must be one of {choices}") from error
    return BallInferenceStrategy(
        name=_validated_version(raw.get("name"), f"{field}.name"),
        mode=mode,
        inference_size_px=_integer(raw.get("inference_size_px"), f"{field}.inference_size_px"),
        minimum_confidence=_number(
            raw.get("minimum_confidence", 0.10), f"{field}.minimum_confidence"
        ),
        model_nms_iou_threshold=_number(
            raw.get("model_nms_iou_threshold", 0.50),
            f"{field}.model_nms_iou_threshold",
        ),
        maximum_detections=_integer(
            raw.get("maximum_detections", 100), f"{field}.maximum_detections"
        ),
        tile_size_px=_optional_integer(raw.get("tile_size_px"), f"{field}.tile_size_px"),
        tile_overlap_fraction=_number(
            raw.get("tile_overlap_fraction", 0.20), f"{field}.tile_overlap_fraction"
        ),
        court_roi_margin_px=_integer(
            raw.get("court_roi_margin_px", 96), f"{field}.court_roi_margin_px"
        ),
        merge_iou_threshold=_number(
            raw.get("merge_iou_threshold", 0.50), f"{field}.merge_iou_threshold"
        ),
    )


def load_ball_experiment_configuration(path: Path) -> BallExperimentConfiguration:
    """Load a strict JSON experiment configuration with paths resolved to the file."""

    resolved = path.expanduser().resolve()
    try:
        root = _object(json.loads(resolved.read_text(encoding="utf-8")), "root")
        if root.get("schema_version") != BALL_EXPERIMENT_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        if root.get("record_type") != "pickleball_detector_experiment_configuration":
            raise ValueError("record_type must be pickleball_detector_experiment_configuration")
        dataset = _object(root.get("dataset"), "dataset")
        model = _object(root.get("model"), "model")
        training = _object(root.get("training"), "training")
        inference = _object(root.get("inference"), "inference")
        evaluation = _object(root.get("evaluation"), "evaluation")
        base_dir = resolved.parent

        requested_device = _string(training.get("device", "auto"), "training.device").lower()
        if VALID_INFERENCE_DEVICE.fullmatch(requested_device) is None:
            raise ValueError("training.device must be auto, cpu, mps, cuda, cuda:N, or N")
        training_settings = BallTrainingSettings(
            epochs=_integer(training.get("epochs"), "training.epochs"),
            image_size_px=_integer(training.get("image_size_px"), "training.image_size_px"),
            batch_size=_integer(training.get("batch_size"), "training.batch_size"),
            device=requested_device,
            workers=_integer(training.get("workers", 4), "training.workers"),
            seed=_integer(training.get("seed"), "training.seed"),
            deterministic=_boolean(training.get("deterministic", True), "training.deterministic"),
            patience_epochs=_integer(
                training.get("patience_epochs", 25), "training.patience_epochs"
            ),
        )
        if training_settings.epochs < 1:
            raise ValueError("training.epochs must be at least 1")
        if training_settings.image_size_px < 320:
            raise ValueError("training.image_size_px must be at least 320")
        if training_settings.batch_size == 0 or training_settings.batch_size < -1:
            raise ValueError("training.batch_size must be -1 or a positive integer")
        if training_settings.workers < 0 or training_settings.patience_epochs < 0:
            raise ValueError("training.workers and patience_epochs must be non-negative")

        raw_strategies = _array(inference.get("strategies"), "inference.strategies")
        strategies = tuple(
            _parse_strategy(value, index) for index, value in enumerate(raw_strategies)
        )
        if not strategies:
            raise ValueError("inference.strategies must not be empty")
        if len({item.name for item in strategies}) != len(strategies):
            raise ValueError("inference strategy names must be unique")

        calibrations_raw = _object(
            evaluation.get("calibrations_by_source_id", {}),
            "evaluation.calibrations_by_source_id",
        )
        calibrations = {
            source_id: _resolved_path(
                value,
                f"evaluation.calibrations_by_source_id.{source_id}",
                base_dir=base_dir,
            )
            for source_id, value in calibrations_raw.items()
        }
        return BallExperimentConfiguration(
            path=resolved,
            experiment_name=_validated_version(root.get("experiment_name"), "experiment_name"),
            dataset=BallDatasetReference(
                version=_validated_version(dataset.get("version"), "dataset.version"),
                split_manifest_path=_resolved_path(
                    dataset.get("split_manifest"), "dataset.split_manifest", base_dir=base_dir
                ),
                annotations_path=_resolved_path(
                    dataset.get("annotations"), "dataset.annotations", base_dir=base_dir
                ),
            ),
            model=BallModelSettings(
                version=_validated_version(model.get("version"), "model.version"),
                base_model=_string(model.get("base_model"), "model.base_model"),
            ),
            training=training_settings,
            strategies=strategies,
            evaluation=BallEvaluationSettings(
                matching_iou_threshold=_number(
                    evaluation.get("matching_iou_threshold", 0.5),
                    "evaluation.matching_iou_threshold",
                ),
                calibrations_by_source_id=calibrations,
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError, DatasetInputError) as error:
        raise DatasetInputError(
            f"unable to load ball experiment configuration {resolved}: {error}"
        ) from error
