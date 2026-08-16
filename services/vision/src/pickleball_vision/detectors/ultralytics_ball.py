"""Ultralytics adapter for custom single-class pickleball weights."""

from __future__ import annotations

import hashlib
import importlib
import sys
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from pickleball_vision.ball_detection import (
    BallDetectorMetadata,
    BallModelPrediction,
)
from pickleball_vision.detectors.ultralytics_person import resolve_inference_device
from pickleball_vision.errors import BallModelError
from pickleball_vision.person_detection import BoundingBox
from pickleball_vision.video import Image

PICKLEBALL_CLASS_ID = 0


class _TensorLike(Protocol):
    def cpu(self) -> _TensorLike: ...

    def numpy(self) -> object: ...


class _BoxesLike(Protocol):
    @property
    def xyxy(self) -> _TensorLike: ...

    @property
    def conf(self) -> _TensorLike: ...

    @property
    def cls(self) -> _TensorLike: ...


class _ResultLike(Protocol):
    @property
    def boxes(self) -> _BoxesLike | None: ...


class _ModelLike(Protocol):
    @property
    def names(self) -> Mapping[int, str] | Sequence[str]: ...

    def predict(
        self,
        *,
        source: Image,
        conf: float,
        iou: float,
        imgsz: int,
        device: str,
        classes: list[int],
        max_det: int,
        verbose: bool,
    ) -> Sequence[_ResultLike]: ...


def _framework_version() -> str | None:
    try:
        return version("ultralytics")
    except PackageNotFoundError:
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise BallModelError(str(error), operation="weights_hash") from error
    return digest.hexdigest()


def _load_model(weights_path: Path) -> _ModelLike:
    try:
        with redirect_stdout(sys.stderr):
            module = cast(Any, importlib.import_module("ultralytics"))
            return cast(_ModelLike, module.YOLO(str(weights_path)))
    except Exception as error:
        raise BallModelError(str(error), operation="model_load") from error


def _class_names(model: _ModelLike) -> tuple[str, ...]:
    names = model.names
    if isinstance(names, Mapping):
        return tuple(str(names[key]) for key in sorted(names))
    return tuple(str(value) for value in names)


class UltralyticsBallDetector:
    """Translate a custom Ultralytics model into crop-local ball predictions."""

    def __init__(
        self,
        weights_path: Path,
        *,
        model_version: str,
        device: str = "auto",
        model: _ModelLike | None = None,
        effective_device: str | None = None,
        framework_version: str | None = None,
    ) -> None:
        resolved_weights = weights_path.expanduser().resolve()
        if not resolved_weights.is_file():
            raise BallModelError(
                f"weights file does not exist: {resolved_weights}", operation="model_load"
            )
        self._device = effective_device or resolve_inference_device(device)
        self._model = model if model is not None else _load_model(resolved_weights)
        names = _class_names(self._model)
        if names != ("pickleball",):
            raise BallModelError(
                f"expected exactly one class named 'pickleball'; model declares {names}",
                operation="class_validation",
            )
        self._metadata = BallDetectorMetadata(
            adapter="ultralytics_ball",
            framework="ultralytics",
            framework_version=(
                framework_version if framework_version is not None else _framework_version()
            ),
            model_version=model_version,
            weights_path=resolved_weights,
            weights_sha256=_file_sha256(resolved_weights),
            device=self._device,
        )

    @property
    def metadata(self) -> BallDetectorMetadata:
        return self._metadata

    def predict(
        self,
        image: Image,
        *,
        inference_size_px: int,
        minimum_confidence: float,
        nms_iou_threshold: float,
        maximum_detections: int,
    ) -> tuple[BallModelPrediction, ...]:
        try:
            results = self._model.predict(
                source=image,
                conf=minimum_confidence,
                iou=nms_iou_threshold,
                imgsz=inference_size_px,
                device=self._device,
                classes=[PICKLEBALL_CLASS_ID],
                max_det=maximum_detections,
                verbose=False,
            )
            if len(results) != 1:
                raise BallModelError(
                    f"expected one result for one crop, received {len(results)}",
                    operation="inference",
                )
            boxes = results[0].boxes
            if boxes is None:
                return ()
            coordinates = np.asarray(boxes.xyxy.cpu().numpy(), dtype=np.float64)
            confidences = np.asarray(boxes.conf.cpu().numpy(), dtype=np.float64)
            classes = np.asarray(boxes.cls.cpu().numpy(), dtype=np.float64)
            if coordinates.ndim != 2 or coordinates.shape[1:] != (4,):
                raise BallModelError(
                    f"unexpected xyxy tensor shape {coordinates.shape}",
                    operation="result_translation",
                )
            if confidences.shape != (coordinates.shape[0],) or classes.shape != (
                coordinates.shape[0],
            ):
                raise BallModelError(
                    "box, confidence, and class tensor lengths differ",
                    operation="result_translation",
                )
            predictions: list[BallModelPrediction] = []
            for xyxy, confidence, class_id in zip(coordinates, confidences, classes, strict=True):
                confidence_value = float(confidence)
                if int(class_id) != PICKLEBALL_CLASS_ID:
                    continue
                if confidence_value < minimum_confidence:
                    continue
                predictions.append(
                    BallModelPrediction(
                        bounding_box=BoundingBox(
                            left_px=float(xyxy[0]),
                            top_px=float(xyxy[1]),
                            right_px=float(xyxy[2]),
                            bottom_px=float(xyxy[3]),
                        ),
                        confidence=confidence_value,
                    )
                )
            return tuple(predictions)
        except BallModelError:
            raise
        except Exception as error:
            raise BallModelError(str(error), operation="inference") from error
