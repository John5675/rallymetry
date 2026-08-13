"""Model-independent person-detection observations and detector boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from pickleball_vision.video import Image, VideoMetadata

PERSON_DETECTION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An axis-aligned rectangle in original source-frame pixels."""

    left_px: float
    top_px: float
    right_px: float
    bottom_px: float

    def __post_init__(self) -> None:
        coordinates = (self.left_px, self.top_px, self.right_px, self.bottom_px)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("bounding-box coordinates must be finite")
        if self.left_px < 0 or self.top_px < 0:
            raise ValueError("bounding-box coordinates must be non-negative")
        if self.right_px <= self.left_px or self.bottom_px <= self.top_px:
            raise ValueError("bounding box must have positive width and height")

    def as_dict(self) -> dict[str, float]:
        """Serialize source-pixel xyxy coordinates without normalization."""

        return {
            "left_px": self.left_px,
            "top_px": self.top_px,
            "right_px": self.right_px,
            "bottom_px": self.bottom_px,
        }


@dataclass(frozen=True, slots=True)
class PersonDetection:
    """Raw model evidence for one visible person in one source frame."""

    bounding_box: BoundingBox
    confidence: float
    frame_number: int
    timestamp_s: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0 or not math.isfinite(self.confidence):
            raise ValueError("detection confidence must be finite and between 0 and 1")
        if self.frame_number < 0:
            raise ValueError("frame number must be non-negative")
        if self.timestamp_s < 0 or not math.isfinite(self.timestamp_s):
            raise ValueError("timestamp must be finite and non-negative")

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_number": self.frame_number,
            "timestamp_s": self.timestamp_s,
            "bounding_box": self.bounding_box.as_dict(),
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class DetectorMetadata:
    """Model adapter provenance retained with each detection run."""

    adapter: str
    model: str
    device: str
    framework: str
    framework_version: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "adapter": self.adapter,
            "model": self.model,
            "device": self.device,
            "framework": self.framework,
            "framework_version": self.framework_version,
        }


class PersonDetector(Protocol):
    """Model-neutral interface consumed by the video pipeline."""

    @property
    def metadata(self) -> DetectorMetadata:
        """Describe the loaded adapter and effective inference device."""

        ...

    def detect(
        self,
        frame: Image,
        *,
        frame_number: int,
        timestamp_s: float,
    ) -> tuple[PersonDetection, ...]:
        """Detect people in one source-resolution BGR frame."""

        ...


@dataclass(frozen=True, slots=True)
class PersonDetectionRun:
    """Serializable raw-observation document for one video run."""

    created_at_utc: str
    source: VideoMetadata
    calibration_path: str
    calibration_schema_version: int
    detector: DetectorMetadata
    configuration: dict[str, object]
    detections: tuple[PersonDetection, ...]
    schema_version: int = PERSON_DETECTION_SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "record_type": "raw_person_observations",
            "source": self.source.as_dict(),
            "calibration": {
                "path": self.calibration_path,
                "schema_version": self.calibration_schema_version,
                "usage": "validated_provenance_only_no_person_projection",
            },
            "coordinate_system": {
                "origin": "top_left",
                "x_axis": "right",
                "y_axis": "down",
                "unit": "pixels",
                "frame_numbering": "zero_based",
            },
            "detector": self.detector.as_dict(),
            "configuration": self.configuration,
            "detections": [detection.as_dict() for detection in self.detections],
        }
