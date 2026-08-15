import pytest

from pickleball_vision.dataset import (
    ClipRange,
    DatasetFrameReference,
    DatasetLabelGroup,
    DatasetSplit,
    FrameSelectionSettings,
    SplitRatios,
    SplitUnit,
    assign_dataset_splits,
    select_dataset_frames,
)
from pickleball_vision.errors import DatasetInputError


def _reference(
    frame_number: int,
    *,
    source_id: str = "sha256:source-a",
    clip_id: str | None = "clip-a",
    group_id: str | None = "rally-a",
) -> DatasetFrameReference:
    return DatasetFrameReference(
        manifest_path="/tmp/dataset-manifest.json",
        record_id=f"{source_id}:frame:{frame_number}",
        source_id=source_id,
        relative_image_path=f"images/unlabeled/frame-{frame_number}.jpg",
        frame_number=frame_number,
        label_group=DatasetLabelGroup.UNLABELED,
        clip_id=clip_id,
        group_id=group_id,
    )


def test_cadence_selection_resets_at_named_clip_boundaries() -> None:
    ranges = (
        ClipRange("rally-a", 0.0, 1.0, "rally-a", DatasetLabelGroup.POSITIVE),
        ClipRange("negative-a", 2.0, 2.5, "break-a", DatasetLabelGroup.NEGATIVE),
    )

    selected = select_dataset_frames(
        fps=10.0,
        frame_count=100,
        duration_s=10.0,
        ranges=ranges,
        settings=FrameSelectionSettings(every_frames=3),
    )

    assert tuple(item.frame_number for item in selected) == (0, 3, 6, 9, 20, 23)
    assert tuple(item.clip_id for item in selected) == (
        "rally-a",
        "rally-a",
        "rally-a",
        "rally-a",
        "negative-a",
        "negative-a",
    )
    assert selected[0].label_group is DatasetLabelGroup.POSITIVE
    assert selected[-1].label_group is DatasetLabelGroup.NEGATIVE


def test_random_selection_is_unique_reproducible_and_range_bounded() -> None:
    ranges = (ClipRange("rally-a", 1.0, 4.0, "rally-a", DatasetLabelGroup.UNLABELED),)
    settings = FrameSelectionSettings(random_count=8, random_seed=42)

    first = select_dataset_frames(
        fps=10.0,
        frame_count=100,
        duration_s=10.0,
        ranges=ranges,
        settings=settings,
    )
    second = select_dataset_frames(
        fps=10.0,
        frame_count=100,
        duration_s=10.0,
        ranges=ranges,
        settings=settings,
    )

    indices = tuple(item.frame_number for item in first)
    assert indices == tuple(item.frame_number for item in second)
    assert len(indices) == len(set(indices)) == 8
    assert all(10 <= index < 40 for index in indices)


def test_selection_rejects_overlap_invalid_counts_and_empty_ranges() -> None:
    overlapping = (
        ClipRange("one", 0.0, 1.0, None, DatasetLabelGroup.UNLABELED),
        ClipRange("two", 0.5, 1.5, None, DatasetLabelGroup.UNLABELED),
    )
    with pytest.raises(DatasetInputError, match="overlap"):
        select_dataset_frames(
            fps=10.0,
            frame_count=20,
            duration_s=2.0,
            ranges=overlapping,
            settings=FrameSelectionSettings(every_frames=1),
        )
    with pytest.raises(DatasetInputError, match="random_count"):
        select_dataset_frames(
            fps=10.0,
            frame_count=20,
            duration_s=2.0,
            ranges=(ClipRange("one", 0.0, 0.2, None, DatasetLabelGroup.UNLABELED),),
            settings=FrameSelectionSettings(random_count=3),
        )
    with pytest.raises(DatasetInputError, match="contains no source frame"):
        select_dataset_frames(
            fps=10.0,
            frame_count=20,
            duration_s=2.0,
            ranges=(ClipRange("tiny", 0.01, 0.02, None, DatasetLabelGroup.UNLABELED),),
            settings=FrameSelectionSettings(every_frames=1),
        )


def test_split_assigns_whole_video_clip_and_group_units_deterministically() -> None:
    records = tuple(
        [_reference(frame, clip_id="clip-a", group_id="rally-a") for frame in range(4)]
        + [_reference(frame, clip_id="clip-b", group_id="rally-b") for frame in range(4, 8)]
        + [
            _reference(
                frame,
                source_id="sha256:source-b",
                clip_id="clip-c",
                group_id="rally-c",
            )
            for frame in range(8, 12)
        ]
    )
    ratios = SplitRatios(train=0.5, validation=0.0, test=0.5)

    for split_by in SplitUnit:
        first = assign_dataset_splits(
            records,
            split_by=split_by,
            ratios=ratios,
            random_seed=7,
        )
        second = assign_dataset_splits(
            records,
            split_by=split_by,
            ratios=ratios,
            random_seed=7,
        )
        assert first == second
        for record in records:
            assert first[record.unit_key(split_by)] in {
                DatasetSplit.TRAIN,
                DatasetSplit.TEST,
            }

    by_video = assign_dataset_splits(
        records,
        split_by=SplitUnit.VIDEO,
        ratios=ratios,
        random_seed=7,
    )
    assert len(by_video) == 2


def test_group_split_requires_group_id_and_ratios_sum_to_one() -> None:
    with pytest.raises(DatasetInputError, match="has no group_id"):
        assign_dataset_splits(
            (_reference(0, group_id=None),),
            split_by=SplitUnit.GROUP,
            ratios=SplitRatios(),
            random_seed=0,
        )
    with pytest.raises(DatasetInputError, match=r"sum to 1\.0"):
        SplitRatios(train=0.8, validation=0.2, test=0.2)
