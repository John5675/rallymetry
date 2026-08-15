"""Local, model-free ball-dataset extraction and split workflows."""

from __future__ import annotations

import hashlib
import json
import math
import re
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import cast

import cv2

from pickleball_vision.dataset import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    DATASET_PRODUCER_VERSION,
    DATASET_SPLIT_SCHEMA_VERSION,
    ClipRange,
    DatasetClipRecord,
    DatasetFrameRecord,
    DatasetFrameReference,
    DatasetLabelGroup,
    DatasetSource,
    DatasetSplit,
    FrameSelectionSettings,
    SplitRatios,
    SplitUnit,
    assign_dataset_splits,
    new_dataset_manifest,
    select_dataset_frames,
)
from pickleball_vision.errors import DatasetInputError, DatasetIoError
from pickleball_vision.media import MediaTimeline, extract_lossless_clip, inspect_media
from pickleball_vision.video import Image, iter_selected_video_frames

DATASET_MANIFEST_NAME = "dataset-manifest.json"
CLIP_DEFINITION_SCHEMA_VERSION = 1
SOURCE_HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class DatasetExtractionArtifacts:
    """Paths and counts produced by one single-source extraction run."""

    output_dir: Path
    manifest_path: Path
    frame_count: int
    written_clip_count: int
    source_id: str
    label_group_counts: dict[DatasetLabelGroup, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "frame_count": self.frame_count,
            "written_clip_count": self.written_clip_count,
            "source_id": self.source_id,
            "label_group_counts": {
                group.value: self.label_group_counts[group] for group in DatasetLabelGroup
            },
        }


@dataclass(frozen=True, slots=True)
class DatasetSplitArtifact:
    """Leakage-safe split manifest output."""

    output_path: Path
    frame_count: int
    unit_count: int
    split_counts: dict[DatasetSplit, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "output_path": str(self.output_path),
            "frame_count": self.frame_count,
            "unit_count": self.unit_count,
            "split_counts": {split.value: self.split_counts[split] for split in DatasetSplit},
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


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field)


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _read_json(path: Path, *, kind: str) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    try:
        return _object(json.loads(resolved.read_text(encoding="utf-8")), "root")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise DatasetIoError(str(resolved), reason=f"unable to load {kind}: {error}") from error


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except (OSError, TypeError, ValueError) as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise DatasetIoError(str(path), reason=str(error)) from error


def _content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(SOURCE_HASH_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as error:
        raise DatasetIoError(str(path), reason=f"unable to hash source media: {error}") from error
    return digest.hexdigest()


def _parse_label_group(value: object, field: str) -> DatasetLabelGroup:
    try:
        return DatasetLabelGroup(_string(value, field))
    except ValueError as error:
        choices = ", ".join(group.value for group in DatasetLabelGroup)
        raise ValueError(f"{field} must be one of {choices}") from error


def load_clip_ranges(
    path: Path,
    *,
    default_label_group: DatasetLabelGroup,
    default_group_id: str | None,
) -> tuple[ClipRange, ...]:
    """Load named, half-open clip/group definitions for extraction."""

    resolved = path.expanduser().resolve()
    root = _read_json(resolved, kind="clip definitions")
    try:
        if root.get("schema_version") != CLIP_DEFINITION_SCHEMA_VERSION:
            raise ValueError("unsupported clip-definition schema_version")
        if root.get("record_type") != "ball_dataset_clips":
            raise ValueError("record_type must be ball_dataset_clips")
        clips: list[ClipRange] = []
        seen_ids: set[str] = set()
        for index, raw_value in enumerate(_array(root.get("clips"), "clips")):
            field = f"clips[{index}]"
            raw = _object(raw_value, field)
            clip_id = _string(raw.get("clip_id"), f"{field}.clip_id")
            if clip_id in seen_ids:
                raise ValueError(f"duplicate clip_id {clip_id!r}")
            seen_ids.add(clip_id)
            raw_label = raw.get("label_group")
            label_group = (
                default_label_group
                if raw_label is None
                else _parse_label_group(raw_label, f"{field}.label_group")
            )
            clips.append(
                ClipRange(
                    clip_id=clip_id,
                    start_time_s=_finite_number(raw.get("start_time_s"), f"{field}.start_time_s"),
                    end_time_s=_finite_number(raw.get("end_time_s"), f"{field}.end_time_s"),
                    group_id=(
                        default_group_id
                        if raw.get("group_id") is None
                        else _optional_string(raw.get("group_id"), f"{field}.group_id")
                    ),
                    label_group=label_group,
                )
            )
        if not clips:
            raise ValueError("clips must contain at least one definition")
        return tuple(clips)
    except (ValueError, DatasetInputError) as error:
        raise DatasetInputError(f"invalid clip definitions in {resolved}: {error}") from error


def _single_range(
    *,
    start_time_s: float | None,
    end_time_s: float | None,
    duration_s: float,
    label_group: DatasetLabelGroup,
    group_id: str | None,
) -> tuple[ClipRange, ...]:
    return (
        ClipRange(
            clip_id="time-range",
            start_time_s=0.0 if start_time_s is None else start_time_s,
            end_time_s=duration_s if end_time_s is None else end_time_s,
            group_id=group_id,
            label_group=label_group,
        ),
    )


def _prepare_output_dir(path: Path) -> tuple[Path, Path]:
    output_dir = path.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise DatasetIoError(str(output_dir), reason="path is not a directory")
    manifest_path = output_dir / DATASET_MANIFEST_NAME
    if manifest_path.exists():
        raise DatasetIoError(
            str(manifest_path),
            reason="dataset output already contains a manifest; choose a new output directory",
        )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise DatasetIoError(str(output_dir), reason=str(error)) from error
    return output_dir, manifest_path


def _safe_source_stem(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_")
    return (normalized or "source")[:64]


def _write_image(path: Path, image: Image) -> None:
    if path.exists():
        raise DatasetIoError(str(path), reason="image output already exists")
    temporary = path.with_name(f".{path.stem}.tmp.jpg")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        written = cv2.imwrite(str(temporary), image)
        if not written:
            raise DatasetIoError(str(path), reason="OpenCV did not write the image")
        temporary.replace(path)
    except (cv2.error, OSError) as error:
        temporary.unlink(missing_ok=True)
        raise DatasetIoError(str(path), reason=str(error)) from error


def extract_ball_dataset_frames(
    video_path: Path,
    *,
    output_dir: Path,
    selection: FrameSelectionSettings,
    label_group: DatasetLabelGroup = DatasetLabelGroup.UNLABELED,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
    clip_definitions_path: Path | None = None,
    group_id: str | None = None,
    write_clips: bool = False,
    timeline: MediaTimeline | None = None,
) -> DatasetExtractionArtifacts:
    """Extract source-resolution annotation frames and optional lossless clips."""

    if clip_definitions_path is not None and (start_time_s is not None or end_time_s is not None):
        raise DatasetInputError("--clips cannot be combined with --start-time or --end-time")
    selected_timeline = timeline or MediaTimeline()
    media = inspect_media(video_path, timeline=selected_timeline)
    ranges = (
        load_clip_ranges(
            clip_definitions_path,
            default_label_group=label_group,
            default_group_id=group_id,
        )
        if clip_definitions_path is not None
        else _single_range(
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            duration_s=media.video.duration,
            label_group=label_group,
            group_id=group_id,
        )
    )
    selected = select_dataset_frames(
        fps=media.video.fps,
        frame_count=media.video.frame_count,
        duration_s=media.video.duration,
        ranges=ranges,
        settings=selection,
    )
    digest = _content_sha256(media.video.path)
    source_id = f"sha256:{digest}"
    source = DatasetSource(
        source_id=source_id,
        content_sha256=digest,
        file_size_bytes=media.video.path.stat().st_size,
        media=media,
    )
    resolved_output, manifest_path = _prepare_output_dir(output_dir)
    try:
        for group in DatasetLabelGroup:
            (resolved_output / "images" / group.value).mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise DatasetIoError(str(resolved_output), reason=str(error)) from error
    source_stem = f"{_safe_source_stem(media.video.path.stem)}-{digest[:12]}"
    by_index = {item.frame_number: item for item in selected}
    frames: list[DatasetFrameRecord] = []
    for decoded in iter_selected_video_frames(media.video.path, sorted(by_index)):
        selected_frame = by_index[decoded.frame_index]
        relative_path = (
            Path("images")
            / selected_frame.label_group.value
            / (f"{source_stem}__frame_{decoded.frame_index:09d}.jpg")
        )
        image_path = resolved_output / relative_path
        _write_image(image_path, decoded.image)
        frames.append(
            DatasetFrameRecord(
                record_id=f"{source_id}:frame:{decoded.frame_index}",
                source_id=source_id,
                frame_number=decoded.frame_index,
                timestamp_s=decoded.timestamp,
                time_base_fps=media.video.fps,
                relative_image_path=relative_path.as_posix(),
                width=media.video.width,
                height=media.video.height,
                label_group=selected_frame.label_group,
                clip_id=selected_frame.clip_id,
                group_id=selected_frame.group_id,
                selection_method=selection.method,
            )
        )

    clip_records: list[DatasetClipRecord] = []
    for clip in ranges:
        relative_clip_path: Path | None = None
        artifact_payload: dict[str, object] | None = None
        if write_clips:
            relative_clip_path = Path("clips") / f"{source_stem}__{clip.clip_id}.mkv"
            artifact = extract_lossless_clip(
                media.video.path,
                output_path=resolved_output / relative_clip_path,
                start_time_s=clip.start_time_s,
                end_time_s=clip.end_time_s,
                timeline=selected_timeline,
            )
            artifact_payload = artifact.as_dict()
        clip_records.append(
            DatasetClipRecord(
                clip=clip,
                relative_media_path=(
                    relative_clip_path.as_posix() if relative_clip_path is not None else None
                ),
                media_artifact=artifact_payload,
            )
        )

    manifest = new_dataset_manifest(
        source=source,
        selection=selection,
        clip_definitions_path=(
            str(clip_definitions_path.expanduser().resolve())
            if clip_definitions_path is not None
            else None
        ),
        clips=tuple(clip_records),
        frames=tuple(frames),
    )
    _write_json(manifest_path, manifest.as_dict())
    counts = {
        group: sum(frame.label_group is group for frame in frames) for group in DatasetLabelGroup
    }
    return DatasetExtractionArtifacts(
        output_dir=resolved_output,
        manifest_path=manifest_path,
        frame_count=len(frames),
        written_clip_count=sum(record.relative_media_path is not None for record in clip_records),
        source_id=source_id,
        label_group_counts=counts,
    )


def _validated_relative_path(value: object, field: str) -> str:
    raw = _string(value, field)
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path")
    return path.as_posix()


def load_dataset_frame_references(path: Path) -> tuple[DatasetFrameReference, ...]:
    """Load only the generated manifest fields needed by split assignment."""

    resolved = path.expanduser().resolve()
    root = _read_json(resolved, kind="dataset manifest")
    try:
        if root.get("schema_version") != DATASET_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported dataset manifest schema_version")
        if root.get("record_type") != "ball_annotation_frame_dataset":
            raise ValueError("record_type must be ball_annotation_frame_dataset")
        source = _object(root.get("source"), "source")
        source_id = _string(source.get("source_id"), "source.source_id")
        records: list[DatasetFrameReference] = []
        seen_ids: set[str] = set()
        for index, raw_value in enumerate(_array(root.get("frames"), "frames")):
            field = f"frames[{index}]"
            raw = _object(raw_value, field)
            record_id = _string(raw.get("record_id"), f"{field}.record_id")
            if record_id in seen_ids:
                raise ValueError(f"duplicate record_id {record_id!r}")
            seen_ids.add(record_id)
            frame_source_id = _string(raw.get("source_id"), f"{field}.source_id")
            if frame_source_id != source_id:
                raise ValueError(f"{field}.source_id differs from source.source_id")
            frame_number = _integer(raw.get("frame_number"), f"{field}.frame_number")
            if frame_number < 0:
                raise ValueError(f"{field}.frame_number must be non-negative")
            records.append(
                DatasetFrameReference(
                    manifest_path=str(resolved),
                    record_id=record_id,
                    source_id=source_id,
                    relative_image_path=_validated_relative_path(
                        raw.get("relative_image_path"), f"{field}.relative_image_path"
                    ),
                    frame_number=frame_number,
                    label_group=_parse_label_group(raw.get("label_group"), f"{field}.label_group"),
                    clip_id=_optional_string(raw.get("clip_id"), f"{field}.clip_id"),
                    group_id=_optional_string(raw.get("group_id"), f"{field}.group_id"),
                )
            )
        return tuple(records)
    except ValueError as error:
        raise DatasetInputError(f"invalid dataset manifest {resolved}: {error}") from error


def _split_frame_payload(
    record: DatasetFrameReference,
    *,
    unit_key: str,
    split: DatasetSplit,
) -> dict[str, object]:
    return {
        "manifest_path": record.manifest_path,
        "record_id": record.record_id,
        "source_id": record.source_id,
        "relative_image_path": record.relative_image_path,
        "frame_number": record.frame_number,
        "label_group": record.label_group.value,
        "clip_id": record.clip_id,
        "group_id": record.group_id,
        "split_unit_key": unit_key,
        "split": split.value,
    }


def split_ball_dataset(
    manifest_paths: tuple[Path, ...],
    *,
    output_path: Path,
    split_by: SplitUnit,
    ratios: SplitRatios,
    random_seed: int,
) -> DatasetSplitArtifact:
    """Write split assignments without copying images or splitting provenance units."""

    if not manifest_paths:
        raise DatasetInputError("at least one dataset manifest is required")
    resolved_manifests = tuple(path.expanduser().resolve() for path in manifest_paths)
    resolved_output = output_path.expanduser().resolve()
    if resolved_output in resolved_manifests:
        raise DatasetIoError(str(resolved_output), reason="split output would overwrite an input")
    records = tuple(
        record
        for manifest_path in resolved_manifests
        for record in load_dataset_frame_references(manifest_path)
    )
    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise DatasetInputError(
            "input manifests contain duplicate source-frame records; deduplicate before splitting"
        )
    assignments = assign_dataset_splits(
        records,
        split_by=split_by,
        ratios=ratios,
        random_seed=random_seed,
    )
    frames: list[dict[str, object]] = []
    split_counts = {split: 0 for split in DatasetSplit}
    unit_counts: dict[str, int] = {}
    for record in records:
        unit_key = record.unit_key(split_by)
        split = assignments[unit_key]
        split_counts[split] += 1
        unit_counts[unit_key] = unit_counts.get(unit_key, 0) + 1
        frames.append(_split_frame_payload(record, unit_key=unit_key, split=split))
    payload = {
        "schema_version": DATASET_SPLIT_SCHEMA_VERSION,
        "record_type": "ball_dataset_split_assignments",
        "producer_version": DATASET_PRODUCER_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_manifests": [str(path) for path in resolved_manifests],
        "configuration": {
            "split_by": split_by.value,
            "ratios": ratios.as_dict(),
            "random_seed": random_seed,
        },
        "unit_assignments": [
            {
                "unit_key": key,
                "split": assignments[key].value,
                "frame_count": unit_counts[key],
            }
            for key in sorted(assignments)
        ],
        "frames": frames,
        "summary": {
            "frame_count": len(records),
            "unit_count": len(assignments),
            "split_frame_counts": {split.value: split_counts[split] for split in DatasetSplit},
        },
        "contracts": {
            "images_copied_or_moved": False,
            "individual_frames_randomly_split": False,
            "all_frames_in_a_unit_share_one_split": True,
        },
    }
    _write_json(resolved_output, payload)
    return DatasetSplitArtifact(
        output_path=resolved_output,
        frame_count=len(records),
        unit_count=len(assignments),
        split_counts=split_counts,
    )
