"""Model-independent person-detection observations and detector boundary."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from pickleball_vision.errors import DetectionIoError
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


def _json_object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _json_array(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return cast(list[object], value)


def _json_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _json_optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _json_string(value, field=field)


def _json_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _json_int(value: object, *, field: str) -> int:
    result = _json_float(value, field=field)
    if not result.is_integer():
        raise ValueError(f"{field} must be an integer")
    return int(result)


def _parse_detection(value: object, *, index: int) -> PersonDetection:
    field = f"detections[{index}]"
    raw = _json_object(value, field=field)
    box = _json_object(raw["bounding_box"], field=f"{field}.bounding_box")
    return PersonDetection(
        bounding_box=BoundingBox(
            left_px=_json_float(box["left_px"], field=f"{field}.bounding_box.left_px"),
            top_px=_json_float(box["top_px"], field=f"{field}.bounding_box.top_px"),
            right_px=_json_float(box["right_px"], field=f"{field}.bounding_box.right_px"),
            bottom_px=_json_float(box["bottom_px"], field=f"{field}.bounding_box.bottom_px"),
        ),
        confidence=_json_float(raw["confidence"], field=f"{field}.confidence"),
        frame_number=_json_int(raw["frame_number"], field=f"{field}.frame_number"),
        timestamp_s=_json_float(raw["timestamp_s"], field=f"{field}.timestamp_s"),
    )


def load_person_detection_run(path: Path) -> PersonDetectionRun:
    """Load and validate immutable raw person observations from JSON."""

    input_path = path.expanduser().resolve()
    try:
        decoded: object = json.loads(input_path.read_text(encoding="utf-8"))
        raw = _json_object(decoded, field="root")
        schema_version = _json_int(raw["schema_version"], field="schema_version")
        if schema_version != PERSON_DETECTION_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version {schema_version}")
        if _json_string(raw["record_type"], field="record_type") != "raw_person_observations":
            raise ValueError("record_type must be raw_person_observations")

        source_raw = _json_object(raw["source"], field="source")
        detector_raw = _json_object(raw["detector"], field="detector")
        calibration_raw = _json_object(raw["calibration"], field="calibration")
        configuration = _json_object(raw["configuration"], field="configuration")
        source = VideoMetadata(
            filename=_json_string(source_raw["filename"], field="source.filename"),
            path=Path(_json_string(source_raw["path"], field="source.path")),
            width=_json_int(source_raw["width"], field="source.width"),
            height=_json_int(source_raw["height"], field="source.height"),
            fps=_json_float(source_raw["fps"], field="source.fps"),
            frame_count=_json_int(source_raw["frame_count"], field="source.frame_count"),
            duration=_json_float(source_raw["duration"], field="source.duration"),
            codec=_json_optional_string(source_raw["codec"], field="source.codec"),
        )
        if (
            source.width < 1
            or source.height < 1
            or source.frame_count < 1
            or source.fps <= 0
            or source.duration <= 0
        ):
            raise ValueError(
                "source metadata dimensions, FPS, frame count, and duration must be positive"
            )
        detector = DetectorMetadata(
            adapter=_json_string(detector_raw["adapter"], field="detector.adapter"),
            model=_json_string(detector_raw["model"], field="detector.model"),
            device=_json_string(detector_raw["device"], field="detector.device"),
            framework=_json_string(detector_raw["framework"], field="detector.framework"),
            framework_version=_json_optional_string(
                detector_raw["framework_version"], field="detector.framework_version"
            ),
        )
        detections = tuple(
            _parse_detection(item, index=index)
            for index, item in enumerate(_json_array(raw["detections"], field="detections"))
        )
        previous_frame = -1
        for index, detection in enumerate(detections):
            if detection.frame_number < previous_frame:
                raise ValueError("detections must be ordered by nondecreasing frame number")
            if detection.frame_number >= source.frame_count:
                raise ValueError(f"detections[{index}] frame number exceeds source frame count")
            if (
                detection.bounding_box.right_px > source.width + 1
                or detection.bounding_box.bottom_px > source.height + 1
            ):
                raise ValueError(f"detections[{index}] bounding box exceeds source dimensions")
            previous_frame = detection.frame_number

        return PersonDetectionRun(
            created_at_utc=_json_string(raw["created_at_utc"], field="created_at_utc"),
            source=source,
            calibration_path=_json_string(calibration_raw["path"], field="calibration.path"),
            calibration_schema_version=_json_int(
                calibration_raw["schema_version"], field="calibration.schema_version"
            ),
            detector=detector,
            configuration=dict(configuration),
            detections=detections,
            schema_version=schema_version,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise DetectionIoError(str(input_path), reason=str(error)) from error
