"""Ultralytics-specific adapter for broad COCO person detection."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol, cast

import numpy as np

from pickleball_vision.config import PersonDetectionSettings
from pickleball_vision.errors import DetectionModelError
from pickleball_vision.person_detection import (
    BoundingBox,
    DetectorMetadata,
    PersonDetection,
)
from pickleball_vision.video import Image

PERSON_CLASS_ID = 0


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


def _available_accelerators() -> tuple[bool, bool]:
    """Probe optional CUDA and MPS backends without requiring either one."""

    try:
        torch = importlib.import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        mps_available = bool(mps_backend is not None and mps_backend.is_available())
    except (ImportError, AttributeError, RuntimeError):
        return (False, False)
    return (cuda_available, mps_available)


def resolve_inference_device(
    requested: str,
    *,
    cuda_available: bool | None = None,
    mps_available: bool | None = None,
) -> str:
    """Resolve `auto` to CUDA, MPS, or CPU while preserving explicit choices."""

    if requested != "auto":
        return requested
    if cuda_available is None or mps_available is None:
        cuda_available, mps_available = _available_accelerators()
    if cuda_available:
        return "cuda:0"
    if mps_available:
        return "mps"
    return "cpu"


def _framework_version() -> str | None:
    try:
        return version("ultralytics")
    except PackageNotFoundError:
        return None


def _load_model(model_name: str) -> _ModelLike:
    try:
        # Keep the CLI's stdout machine-readable while preserving download/setup
        # diagnostics for a human on stderr.
        with redirect_stdout(sys.stderr):
            module = cast(Any, importlib.import_module("ultralytics"))
            factory = cast(Any, module.YOLO)
            return cast(_ModelLike, factory(model_name))
    except Exception as error:
        raise DetectionModelError(str(error), operation="model_load") from error


class UltralyticsPersonDetector:
    """Translate Ultralytics results into stable raw person observations."""

    def __init__(
        self,
        settings: PersonDetectionSettings,
        *,
        model: _ModelLike | None = None,
        effective_device: str | None = None,
        framework_version: str | None = None,
    ) -> None:
        self._settings = settings
        self._device = effective_device or resolve_inference_device(settings.device)
        self._model = model if model is not None else _load_model(settings.model)
        self._metadata = DetectorMetadata(
            adapter="ultralytics_person",
            model=settings.model,
            device=self._device,
            framework="ultralytics",
            framework_version=(
                framework_version if framework_version is not None else _framework_version()
            ),
        )

    @property
    def metadata(self) -> DetectorMetadata:
        return self._metadata

    def detect(
        self,
        frame: Image,
        *,
        frame_number: int,
        timestamp_s: float,
    ) -> tuple[PersonDetection, ...]:
        """Run class-filtered inference and retain original xyxy coordinates."""

        try:
            results = self._model.predict(
                source=frame,
                conf=self._settings.min_confidence,
                iou=self._settings.iou_threshold,
                imgsz=self._settings.image_size,
                device=self._device,
                classes=[PERSON_CLASS_ID],
                max_det=self._settings.max_detections,
                verbose=False,
            )
            if len(results) != 1:
                raise DetectionModelError(
                    f"expected one result for one frame, received {len(results)}",
                    operation="inference",
                )
            boxes = results[0].boxes
            if boxes is None:
                return ()

            coordinates = np.asarray(boxes.xyxy.cpu().numpy(), dtype=np.float64)
            confidences = np.asarray(boxes.conf.cpu().numpy(), dtype=np.float64)
            classes = np.asarray(boxes.cls.cpu().numpy(), dtype=np.float64)
            if coordinates.ndim != 2 or coordinates.shape[1:] != (4,):
                raise DetectionModelError(
                    f"unexpected xyxy tensor shape {coordinates.shape}",
                    operation="result_translation",
                )
            if confidences.shape != (coordinates.shape[0],) or classes.shape != (
                coordinates.shape[0],
            ):
                raise DetectionModelError(
                    "box, confidence, and class tensor lengths differ",
                    operation="result_translation",
                )

            detections: list[PersonDetection] = []
            for xyxy, confidence, class_id in zip(
                coordinates,
                confidences,
                classes,
                strict=True,
            ):
                confidence_value = float(confidence)
                if int(class_id) != PERSON_CLASS_ID:
                    continue
                if confidence_value < self._settings.min_confidence:
                    continue
                detections.append(
                    PersonDetection(
                        bounding_box=BoundingBox(
                            left_px=float(xyxy[0]),
                            top_px=float(xyxy[1]),
                            right_px=float(xyxy[2]),
                            bottom_px=float(xyxy[3]),
                        ),
                        confidence=confidence_value,
                        frame_number=frame_number,
                        timestamp_s=timestamp_s,
                    )
                )
            return tuple(detections)
        except DetectionModelError:
            raise
        except Exception as error:
            raise DetectionModelError(str(error), operation="inference") from error
