"""Versioned ball-dataset records, frame selection, and leakage-safe splitting."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from pickleball_vision.errors import DatasetInputError
from pickleball_vision.media import MediaMetadata

DATASET_MANIFEST_SCHEMA_VERSION = 1
DATASET_SPLIT_SCHEMA_VERSION = 1
DATASET_PRODUCER_VERSION = "ball_dataset_0.1"
SAFE_GROUP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class DatasetLabelGroup(StrEnum):
    """Human-curation bucket, not a model prediction."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNLABELED = "unlabeled"


class SamplingMethod(StrEnum):
    """Supported source-frame selection methods."""

    CADENCE = "cadence"
    RANDOM = "random"


class SplitUnit(StrEnum):
    """Indivisible provenance unit used to prevent neighboring-frame leakage."""

    VIDEO = "video"
    CLIP = "clip"
    GROUP = "group"


class DatasetSplit(StrEnum):
    """Supported model-development partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class ClipRange:
    """Named half-open source-video range used as a dataset grouping boundary."""

    clip_id: str
    start_time_s: float
    end_time_s: float
    group_id: str | None
    label_group: DatasetLabelGroup

    def __post_init__(self) -> None:
        _validate_group_id(self.clip_id, field="clip_id")
        if self.group_id is not None:
            _validate_group_id(self.group_id, field="group_id")
        if (
            not math.isfinite(self.start_time_s)
            or not math.isfinite(self.end_time_s)
            or self.start_time_s < 0
            or self.end_time_s <= self.start_time_s
        ):
            raise DatasetInputError("clip times must be finite and satisfy 0 <= start < end")

    def as_dict(self) -> dict[str, object]:
        return {
            "clip_id": self.clip_id,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "end_exclusive": True,
            "group_id": self.group_id,
            "label_group": self.label_group.value,
        }


def _validate_group_id(value: str, *, field: str) -> None:
    if SAFE_GROUP_ID.fullmatch(value) is None:
        raise DatasetInputError(
            f"{field} must start with an alphanumeric character and contain only "
            "letters, numbers, period, underscore, or hyphen"
        )


@dataclass(frozen=True, slots=True)
class SelectedFrame:
    """One selected source index plus its curation grouping."""

    frame_number: int
    clip_id: str | None
    group_id: str | None
    label_group: DatasetLabelGroup


@dataclass(frozen=True, slots=True)
class FrameSelectionSettings:
    """Mutually exclusive cadence/random settings recorded in a manifest."""

    every_frames: int | None = None
    random_count: int | None = None
    random_seed: int = 0

    def __post_init__(self) -> None:
        if (self.every_frames is None) == (self.random_count is None):
            raise DatasetInputError("select exactly one of every_frames or random_count")
        if self.every_frames is not None and self.every_frames <= 0:
            raise DatasetInputError("every_frames must be a positive integer")
        if self.random_count is not None and self.random_count <= 0:
            raise DatasetInputError("random_count must be a positive integer")

    @property
    def method(self) -> SamplingMethod:
        return SamplingMethod.CADENCE if self.every_frames is not None else SamplingMethod.RANDOM

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method.value,
            "every_frames": self.every_frames,
            "random_count": self.random_count,
            "random_seed": self.random_seed if self.method is SamplingMethod.RANDOM else None,
        }


def _frame_bounds(
    *,
    start_time_s: float,
    end_time_s: float,
    fps: float,
    frame_count: int,
) -> tuple[int, int]:
    first = max(0, math.ceil(start_time_s * fps - 1e-9))
    stop = min(frame_count, math.ceil(end_time_s * fps - 1e-9))
    if first >= stop:
        raise DatasetInputError(
            f"time range [{start_time_s}, {end_time_s}) contains no source frame timestamps"
        )
    return first, stop


def select_dataset_frames(
    *,
    fps: float,
    frame_count: int,
    duration_s: float,
    ranges: tuple[ClipRange, ...],
    settings: FrameSelectionSettings,
) -> tuple[SelectedFrame, ...]:
    """Select deterministic source indices without inventing labels or timestamps."""

    if not ranges:
        raise DatasetInputError("at least one time range is required")
    by_frame: dict[int, SelectedFrame] = {}
    candidates: list[SelectedFrame] = []
    for clip in ranges:
        if clip.end_time_s > duration_s:
            raise DatasetInputError(
                f"clip {clip.clip_id!r} ends after source duration {duration_s:.6f} seconds"
            )
        first, stop = _frame_bounds(
            start_time_s=clip.start_time_s,
            end_time_s=clip.end_time_s,
            fps=fps,
            frame_count=frame_count,
        )
        for frame_number in range(first, stop):
            selected = SelectedFrame(
                frame_number=frame_number,
                clip_id=clip.clip_id,
                group_id=clip.group_id,
                label_group=clip.label_group,
            )
            if frame_number in by_frame:
                other = by_frame[frame_number]
                raise DatasetInputError(
                    f"clips {other.clip_id!r} and {clip.clip_id!r} overlap at frame {frame_number}"
                )
            by_frame[frame_number] = selected
            candidates.append(selected)

    if settings.every_frames is not None:
        selected_frames: list[SelectedFrame] = []
        for clip in ranges:
            clip_candidates = [item for item in candidates if item.clip_id == clip.clip_id]
            selected_frames.extend(clip_candidates[:: settings.every_frames])
    else:
        assert settings.random_count is not None
        if settings.random_count > len(candidates):
            raise DatasetInputError(
                f"random_count must not exceed {len(candidates)} eligible frames"
            )
        generator = random.Random(settings.random_seed)
        selected_frames = generator.sample(candidates, settings.random_count)
    return tuple(sorted(selected_frames, key=lambda item: item.frame_number))


@dataclass(frozen=True, slots=True)
class DatasetSource:
    """Local source-media identity and stream metadata."""

    source_id: str
    content_sha256: str
    file_size_bytes: int
    media: MediaMetadata

    def as_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "content_sha256": self.content_sha256,
            "file_size_bytes": self.file_size_bytes,
            "media": self.media.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class DatasetFrameRecord:
    """Extracted image with immutable source-frame provenance."""

    record_id: str
    source_id: str
    frame_number: int
    timestamp_s: float
    time_base_fps: float
    relative_image_path: str
    width: int
    height: int
    label_group: DatasetLabelGroup
    clip_id: str | None
    group_id: str | None
    selection_method: SamplingMethod

    def as_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "source_id": self.source_id,
            "frame_number": self.frame_number,
            "timestamp_s": self.timestamp_s,
            "time_base": {"kind": "frames_per_second", "fps": self.time_base_fps},
            "relative_image_path": self.relative_image_path,
            "width": self.width,
            "height": self.height,
            "label_group": self.label_group.value,
            "annotation_status": "not_annotated",
            "clip_id": self.clip_id,
            "group_id": self.group_id,
            "selection_method": self.selection_method.value,
        }


@dataclass(frozen=True, slots=True)
class DatasetClipRecord:
    """Named source range and optional lossless review artifact."""

    clip: ClipRange
    relative_media_path: str | None
    media_artifact: dict[str, object] | None

    def as_dict(self) -> dict[str, object]:
        return {
            **self.clip.as_dict(),
            "relative_media_path": self.relative_media_path,
            "media_artifact": self.media_artifact,
        }


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Single-source extraction manifest consumed by annotation and split tooling."""

    source: DatasetSource
    selection: FrameSelectionSettings
    label_class: str
    clip_definitions_path: str | None
    clips: tuple[DatasetClipRecord, ...]
    frames: tuple[DatasetFrameRecord, ...]
    created_at_utc: str
    schema_version: int = DATASET_MANIFEST_SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        counts = {
            group.value: sum(frame.label_group is group for frame in self.frames)
            for group in DatasetLabelGroup
        }
        return {
            "schema_version": self.schema_version,
            "record_type": "ball_annotation_frame_dataset",
            "producer_version": DATASET_PRODUCER_VERSION,
            "created_at_utc": self.created_at_utc,
            "label_class": self.label_class,
            "source": self.source.as_dict(),
            "selection": self.selection.as_dict(),
            "clip_definitions_path": self.clip_definitions_path,
            "clips": [clip.as_dict() for clip in self.clips],
            "frames": [frame.as_dict() for frame in self.frames],
            "summary": {
                "frame_count": len(self.frames),
                "clip_count": len(self.clips),
                "label_group_counts": counts,
            },
            "contracts": {
                "source_media_preserved": True,
                "frame_coordinates_preserve_source_resolution": True,
                "label_groups_are_human_curation_buckets": True,
                "model_predictions_included": False,
            },
        }


def new_dataset_manifest(
    *,
    source: DatasetSource,
    selection: FrameSelectionSettings,
    clip_definitions_path: str | None,
    clips: tuple[DatasetClipRecord, ...],
    frames: tuple[DatasetFrameRecord, ...],
) -> DatasetManifest:
    return DatasetManifest(
        source=source,
        selection=selection,
        label_class="pickleball",
        clip_definitions_path=clip_definitions_path,
        clips=clips,
        frames=frames,
        created_at_utc=datetime.now(UTC).isoformat(),
    )


@dataclass(frozen=True, slots=True)
class DatasetFrameReference:
    """Minimal persisted-frame fields needed for leakage-safe splitting."""

    manifest_path: str
    record_id: str
    source_id: str
    relative_image_path: str
    frame_number: int
    label_group: DatasetLabelGroup
    clip_id: str | None
    group_id: str | None

    def unit_key(self, split_by: SplitUnit) -> str:
        if split_by is SplitUnit.VIDEO:
            return f"video:{self.source_id}"
        if split_by is SplitUnit.CLIP:
            clip = self.clip_id or "__full_source__"
            return f"clip:{self.source_id}:{clip}"
        if self.group_id is None:
            raise DatasetInputError(
                f"record {self.record_id!r} has no group_id required by --by group"
            )
        return f"group:{self.source_id}:{self.group_id}"


@dataclass(frozen=True, slots=True)
class SplitRatios:
    """Validated train/validation/test target proportions."""

    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise DatasetInputError("split ratios must be finite and non-negative")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise DatasetInputError("train, validation, and test ratios must sum to 1.0")
        if self.train <= 0:
            raise DatasetInputError("train ratio must be greater than zero")

    def as_dict(self) -> dict[str, float]:
        return {
            DatasetSplit.TRAIN.value: self.train,
            DatasetSplit.VALIDATION.value: self.validation,
            DatasetSplit.TEST.value: self.test,
        }


def assign_dataset_splits(
    records: tuple[DatasetFrameReference, ...],
    *,
    split_by: SplitUnit,
    ratios: SplitRatios,
    random_seed: int,
) -> dict[str, DatasetSplit]:
    """Assign whole provenance units while approximately balancing frame counts."""

    if not records:
        raise DatasetInputError("at least one dataset frame is required for splitting")
    unit_sizes: dict[str, int] = {}
    for record in records:
        key = record.unit_key(split_by)
        unit_sizes[key] = unit_sizes.get(key, 0) + 1

    shuffled = sorted(unit_sizes)
    random.Random(random_seed).shuffle(shuffled)
    shuffled.sort(key=lambda key: unit_sizes[key], reverse=True)
    total_frames = len(records)
    ratio_by_split = {
        DatasetSplit.TRAIN: ratios.train,
        DatasetSplit.VALIDATION: ratios.validation,
        DatasetSplit.TEST: ratios.test,
    }
    targets = {split: ratio * total_frames for split, ratio in ratio_by_split.items()}
    counts = {split: 0 for split in DatasetSplit}
    assignments: dict[str, DatasetSplit] = {}
    eligible_splits = tuple(split for split in DatasetSplit if ratio_by_split[split] > 0)
    for key in shuffled:
        size = unit_sizes[key]

        def score(candidate: DatasetSplit, unit_size: int = size) -> tuple[float, int]:
            projected_error = sum(
                abs(counts[split] + (unit_size if split is candidate else 0) - targets[split])
                for split in DatasetSplit
            )
            return projected_error, tuple(DatasetSplit).index(candidate)

        selected = min(eligible_splits, key=score)
        assignments[key] = selected
        counts[selected] += size
    return assignments
