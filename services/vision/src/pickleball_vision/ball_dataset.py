"""Validated annotated dataset loading and YOLO training materialization."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeVar, cast

from pickleball_vision.dataset import DatasetSplit
from pickleball_vision.errors import DatasetInputError, DatasetIoError
from pickleball_vision.person_detection import BoundingBox

BALL_ANNOTATION_SCHEMA_VERSION = 1
PREPARED_DATASET_SCHEMA_VERSION = 1
EnumValue = TypeVar("EnumValue", bound=StrEnum)


class BallCourtSide(StrEnum):
    """Human-annotated side; no airborne-ball homography projection is used."""

    NEAR = "near"
    FAR = "far"
    UNKNOWN = "unknown"


class BallScope(StrEnum):
    """Human-annotated relationship to the primary match."""

    PRIMARY_MATCH = "primary_match"
    NEIGHBORING_COURT = "neighboring_court"
    UNKNOWN = "unknown"


class BallVisibility(StrEnum):
    """Supported visible-evidence states from the annotation guide."""

    CLEAR = "clear"
    PARTIAL = "partial"
    BLURRED = "blurred"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class GroundTruthBall:
    """One human-reviewed visible pickleball box."""

    annotation_id: str
    bounding_box: BoundingBox
    court_side: BallCourtSide
    scope: BallScope
    visibility: BallVisibility

    def as_dict(self) -> dict[str, object]:
        return {
            "annotation_id": self.annotation_id,
            "class_name": "pickleball",
            "bounding_box": self.bounding_box.as_dict(),
            "court_side": self.court_side.value,
            "scope": self.scope.value,
            "visibility": self.visibility.value,
        }


@dataclass(frozen=True, slots=True)
class DetectorDatasetFrame:
    """One fixed-split image joined to its reviewed annotations and provenance."""

    record_id: str
    source_id: str
    image_path: Path
    width: int
    height: int
    frame_number: int
    timestamp_s: float
    split: DatasetSplit
    split_unit_key: str
    clip_id: str | None
    group_id: str | None
    clip_start_time_s: float
    clip_end_time_s: float
    objects: tuple[GroundTruthBall, ...]


@dataclass(frozen=True, slots=True)
class BallDetectorDataset:
    """Complete fixed detector dataset with content fingerprints."""

    version: str
    split_manifest_path: Path
    annotations_path: Path
    split_manifest_sha256: str
    annotations_sha256: str
    frames: tuple[DetectorDatasetFrame, ...]

    def partition(self, split: DatasetSplit) -> tuple[DetectorDatasetFrame, ...]:
        return tuple(frame for frame in self.frames if frame.split is split)

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "split_manifest": str(self.split_manifest_path),
            "annotations": str(self.annotations_path),
            "split_manifest_sha256": self.split_manifest_sha256,
            "annotations_sha256": self.annotations_sha256,
            "fixed_partitions": True,
            "counts": {split.value: len(self.partition(split)) for split in DatasetSplit},
        }


@dataclass(frozen=True, slots=True)
class PreparedBallDataset:
    """YOLO-compatible local links and labels generated from immutable records."""

    root: Path
    dataset_yaml_path: Path
    metadata_path: Path
    frame_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "dataset_yaml": str(self.dataset_yaml_path),
            "metadata": str(self.metadata_path),
            "frame_count": self.frame_count,
        }


@dataclass(frozen=True, slots=True)
class BallAnnotationTemplateArtifact:
    """Unlabeled review template generated without inventing ground truth."""

    output_path: Path
    frame_count: int
    dataset_version: str

    def as_dict(self) -> dict[str, object]:
        return {
            "output_path": str(self.output_path),
            "frame_count": self.frame_count,
            "dataset_version": self.dataset_version,
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
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field)


def _read_json(path: Path, kind: str) -> dict[str, object]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), "root")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise DatasetIoError(str(path), reason=f"unable to load {kind}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise DatasetIoError(str(path), reason=f"unable to hash file: {error}") from error
    return digest.hexdigest()


def _enum_value(enum_type: type[EnumValue], value: object, field: str) -> EnumValue:
    try:
        return enum_type(_string(value, field))
    except ValueError as error:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{field} must be one of {choices}") from error


def _parse_box(raw: object, field: str, *, width: int, height: int) -> BoundingBox:
    box = _object(raw, field)
    result = BoundingBox(
        left_px=_number(box.get("left_px"), f"{field}.left_px"),
        top_px=_number(box.get("top_px"), f"{field}.top_px"),
        right_px=_number(box.get("right_px"), f"{field}.right_px"),
        bottom_px=_number(box.get("bottom_px"), f"{field}.bottom_px"),
    )
    if result.right_px > width or result.bottom_px > height:
        raise ValueError(f"{field} lies outside the {width}x{height} source image")
    return result


def _manifest_path(value: str, *, split_path: Path) -> Path:
    raw = Path(value).expanduser()
    return (split_path.parent / raw).resolve() if not raw.is_absolute() else raw.resolve()


def load_ball_detector_dataset(
    *,
    dataset_version: str,
    split_manifest_path: Path,
    annotations_path: Path,
) -> BallDetectorDataset:
    """Join fixed split records with complete reviewed annotation records."""

    split_path = split_manifest_path.expanduser().resolve()
    annotation_path = annotations_path.expanduser().resolve()
    split_root = _read_json(split_path, "dataset split manifest")
    annotations_root = _read_json(annotation_path, "ball annotations")
    try:
        if split_root.get("schema_version") != 1 or split_root.get("record_type") != (
            "ball_dataset_split_assignments"
        ):
            raise ValueError("split manifest has an unsupported schema or record_type")
        if annotations_root.get("schema_version") != BALL_ANNOTATION_SCHEMA_VERSION:
            raise ValueError("annotations have an unsupported schema_version")
        if annotations_root.get("record_type") != "pickleball_detection_annotations":
            raise ValueError("annotation record_type must be pickleball_detection_annotations")
        if annotations_root.get("dataset_version") != dataset_version:
            raise ValueError("annotation dataset_version does not match configured dataset version")
        if annotations_root.get("class_name") != "pickleball":
            raise ValueError("the only supported annotation class_name is pickleball")

        annotations_by_record: dict[str, dict[str, object]] = {}
        for index, value in enumerate(_array(annotations_root.get("frames"), "frames")):
            field = f"frames[{index}]"
            item = _object(value, field)
            record_id = _string(item.get("record_id"), f"{field}.record_id")
            if record_id in annotations_by_record:
                raise ValueError(f"duplicate annotation frame record_id {record_id!r}")
            annotations_by_record[record_id] = item

        manifest_cache: dict[Path, tuple[dict[str, dict[str, object]], dict[str, object]]] = {}
        frames: list[DetectorDatasetFrame] = []
        unit_splits: dict[str, DatasetSplit] = {}
        for index, value in enumerate(_array(split_root.get("frames"), "split.frames")):
            field = f"split.frames[{index}]"
            split_record = _object(value, field)
            record_id = _string(split_record.get("record_id"), f"{field}.record_id")
            try:
                partition = DatasetSplit(_string(split_record.get("split"), f"{field}.split"))
            except ValueError as error:
                raise ValueError(f"{field}.split is unsupported") from error
            unit_key = _string(split_record.get("split_unit_key"), f"{field}.split_unit_key")
            previous_partition = unit_splits.setdefault(unit_key, partition)
            if previous_partition is not partition:
                raise ValueError(f"split unit {unit_key!r} crosses dataset partitions")

            source_manifest_path = _manifest_path(
                _string(split_record.get("manifest_path"), f"{field}.manifest_path"),
                split_path=split_path,
            )
            if source_manifest_path not in manifest_cache:
                source_root = _read_json(source_manifest_path, "source dataset manifest")
                source_frames = {
                    _string(_object(item, "source frame").get("record_id"), "record_id"): _object(
                        item, "source frame"
                    )
                    for item in _array(source_root.get("frames"), "source.frames")
                }
                manifest_cache[source_manifest_path] = (source_frames, source_root)
            source_frames, source_root = manifest_cache[source_manifest_path]
            source_frame = source_frames.get(record_id)
            if source_frame is None:
                raise ValueError(f"record {record_id!r} is missing from {source_manifest_path}")
            annotation = annotations_by_record.get(record_id)
            if annotation is None:
                raise ValueError(f"record {record_id!r} has no annotation review record")
            if annotation.get("review_status") != "reviewed":
                raise ValueError(f"record {record_id!r} is not human-reviewed")

            width = _integer(source_frame.get("width"), f"{record_id}.width")
            height = _integer(source_frame.get("height"), f"{record_id}.height")
            objects: list[GroundTruthBall] = []
            seen_annotation_ids: set[str] = set()
            for object_index, object_value in enumerate(
                _array(annotation.get("objects"), f"annotation[{record_id}].objects")
            ):
                object_field = f"annotation[{record_id}].objects[{object_index}]"
                item = _object(object_value, object_field)
                if item.get("class_name") != "pickleball":
                    raise ValueError(f"{object_field}.class_name must be pickleball")
                annotation_id = _string(item.get("annotation_id"), f"{object_field}.annotation_id")
                if annotation_id in seen_annotation_ids:
                    raise ValueError(f"duplicate annotation_id {annotation_id!r}")
                seen_annotation_ids.add(annotation_id)
                visibility = _enum_value(
                    BallVisibility, item.get("visibility"), f"{object_field}.visibility"
                )
                if visibility is BallVisibility.AMBIGUOUS:
                    raise ValueError(
                        f"{object_field} is ambiguous and cannot be detector ground truth"
                    )
                objects.append(
                    GroundTruthBall(
                        annotation_id=annotation_id,
                        bounding_box=_parse_box(
                            item.get("bounding_box"),
                            f"{object_field}.bounding_box",
                            width=width,
                            height=height,
                        ),
                        court_side=_enum_value(
                            BallCourtSide,
                            item.get("court_side", "unknown"),
                            f"{object_field}.court_side",
                        ),
                        scope=_enum_value(
                            BallScope,
                            item.get("scope", "unknown"),
                            f"{object_field}.scope",
                        ),
                        visibility=visibility,
                    )
                )

            source = _object(source_root.get("source"), "source")
            source_id = _string(source.get("source_id"), "source.source_id")
            relative_image = _string(
                source_frame.get("relative_image_path"), f"{record_id}.relative_image_path"
            )
            image_path = source_manifest_path.parent / relative_image
            if not image_path.is_file():
                raise ValueError(f"image for record {record_id!r} does not exist: {image_path}")
            clip_id = _optional_string(source_frame.get("clip_id"), f"{record_id}.clip_id")
            source_media = _object(source.get("media"), "source.media")
            clip_start = 0.0
            clip_end = _number(source_media.get("duration"), "source.media.duration")
            if clip_id is not None:
                clip_matches = [
                    _object(item, "clip")
                    for item in _array(source_root.get("clips"), "source.clips")
                    if _object(item, "clip").get("clip_id") == clip_id
                ]
                if len(clip_matches) != 1:
                    raise ValueError(f"record {record_id!r} has an unresolved clip_id")
                clip_start = _number(clip_matches[0].get("start_time_s"), "clip.start_time_s")
                clip_end = _number(clip_matches[0].get("end_time_s"), "clip.end_time_s")
            frames.append(
                DetectorDatasetFrame(
                    record_id=record_id,
                    source_id=source_id,
                    image_path=image_path.resolve(),
                    width=width,
                    height=height,
                    frame_number=_integer(
                        source_frame.get("frame_number"), f"{record_id}.frame_number"
                    ),
                    timestamp_s=_number(
                        source_frame.get("timestamp_s"), f"{record_id}.timestamp_s"
                    ),
                    split=partition,
                    split_unit_key=unit_key,
                    clip_id=clip_id,
                    group_id=_optional_string(
                        source_frame.get("group_id"), f"{record_id}.group_id"
                    ),
                    clip_start_time_s=clip_start,
                    clip_end_time_s=clip_end,
                    objects=tuple(objects),
                )
            )

        if not frames:
            raise ValueError("fixed split contains no frames")
        for partition in DatasetSplit:
            if not any(frame.split is partition for frame in frames):
                raise ValueError(f"fixed {partition.value} partition must not be empty")
        if len({frame.record_id for frame in frames}) != len(frames):
            raise ValueError("fixed split contains duplicate source-frame records")
        return BallDetectorDataset(
            version=dataset_version,
            split_manifest_path=split_path,
            annotations_path=annotation_path,
            split_manifest_sha256=_sha256(split_path),
            annotations_sha256=_sha256(annotation_path),
            frames=tuple(frames),
        )
    except (KeyError, ValueError) as error:
        raise DatasetInputError(f"invalid ball detector dataset: {error}") from error


def _write_text(path: Path, value: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    except OSError as error:
        raise DatasetIoError(str(path), reason=str(error)) from error


def _link_image(source: Path, destination: Path) -> str:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(source, destination)
        return "symbolic_link"
    except OSError:
        try:
            os.link(source, destination)
            return "hard_link"
        except OSError as error:
            raise DatasetIoError(
                str(destination), reason=f"unable to link image: {error}"
            ) from error


def prepare_yolo_ball_dataset(
    dataset: BallDetectorDataset,
    *,
    output_dir: Path,
) -> PreparedBallDataset:
    """Materialize deterministic image links and YOLO labels for Ultralytics."""

    root = output_dir.expanduser().resolve()
    metadata_path = root / "dataset-metadata.json"
    if metadata_path.exists():
        raise DatasetIoError(str(metadata_path), reason="prepared dataset already exists")
    link_modes: set[str] = set()
    records: list[dict[str, object]] = []
    for frame in dataset.frames:
        artifact_id = hashlib.sha256(frame.record_id.encode("utf-8")).hexdigest()[:20]
        suffix = frame.image_path.suffix.lower() or ".jpg"
        image_path = root / "images" / frame.split.value / f"{artifact_id}{suffix}"
        label_path = root / "labels" / frame.split.value / f"{artifact_id}.txt"
        if image_path.exists() or image_path.is_symlink() or label_path.exists():
            raise DatasetIoError(str(root), reason="prepared artifact path already exists")
        link_modes.add(_link_image(frame.image_path, image_path))
        lines: list[str] = []
        for ball in frame.objects:
            box = ball.bounding_box
            center_x = (box.left_px + box.right_px) / (2 * frame.width)
            center_y = (box.top_px + box.bottom_px) / (2 * frame.height)
            width = (box.right_px - box.left_px) / frame.width
            height = (box.bottom_px - box.top_px) / frame.height
            lines.append(f"0 {center_x:.10f} {center_y:.10f} {width:.10f} {height:.10f}")
        _write_text(label_path, "\n".join(lines) + ("\n" if lines else ""))
        records.append(
            {
                "record_id": frame.record_id,
                "split": frame.split.value,
                "source_image": str(frame.image_path),
                "prepared_image": str(image_path),
                "prepared_label": str(label_path),
                "object_count": len(frame.objects),
            }
        )

    yaml_path = root / "dataset.yaml"
    yaml = (
        f"path: {json.dumps(str(root))}\n"
        "train: images/train\n"
        "val: images/validation\n"
        "test: images/test\n"
        "names:\n"
        "  0: pickleball\n"
    )
    _write_text(yaml_path, yaml)
    metadata = {
        "schema_version": PREPARED_DATASET_SCHEMA_VERSION,
        "record_type": "prepared_pickleball_detector_dataset",
        "dataset": dataset.as_dict(),
        "class_names": ["pickleball"],
        "image_materialization": sorted(link_modes),
        "records": records,
    }
    _write_text(
        metadata_path,
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return PreparedBallDataset(
        root=root,
        dataset_yaml_path=yaml_path,
        metadata_path=metadata_path,
        frame_count=len(records),
    )


def create_ball_annotation_template(
    split_manifest_path: Path,
    *,
    dataset_version: str,
    output_path: Path,
) -> BallAnnotationTemplateArtifact:
    """Create explicit unreviewed records for every fixed-split frame."""

    split_path = split_manifest_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    if output.exists():
        raise DatasetIoError(str(output), reason="annotation template already exists")
    root = _read_json(split_path, "dataset split manifest")
    try:
        if root.get("schema_version") != 1 or root.get("record_type") != (
            "ball_dataset_split_assignments"
        ):
            raise ValueError("split manifest has an unsupported schema or record_type")
        records: list[dict[str, object]] = []
        seen: set[str] = set()
        for index, value in enumerate(_array(root.get("frames"), "frames")):
            item = _object(value, f"frames[{index}]")
            record_id = _string(item.get("record_id"), f"frames[{index}].record_id")
            if record_id in seen:
                raise ValueError(f"duplicate record_id {record_id!r}")
            seen.add(record_id)
            records.append(
                {
                    "record_id": record_id,
                    "review_status": "unreviewed",
                    "objects": [],
                }
            )
        if not records:
            raise ValueError("split manifest contains no frames")
    except ValueError as error:
        raise DatasetInputError(f"unable to create annotation template: {error}") from error
    payload = {
        "schema_version": BALL_ANNOTATION_SCHEMA_VERSION,
        "record_type": "pickleball_detection_annotations",
        "dataset_version": dataset_version,
        "class_name": "pickleball",
        "source_split_manifest": str(split_path),
        "instructions": (
            "Set review_status to reviewed only after human review. Empty objects then means "
            "a reviewed negative; unreviewed never means negative."
        ),
        "frames": records,
    }
    _write_text(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return BallAnnotationTemplateArtifact(
        output_path=output,
        frame_count=len(records),
        dataset_version=dataset_version,
    )
