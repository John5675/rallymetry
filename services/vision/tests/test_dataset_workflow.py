import json
from pathlib import Path

import cv2
import pytest

from pickleball_vision.dataset import (
    DatasetLabelGroup,
    FrameSelectionSettings,
    SplitRatios,
    SplitUnit,
)
from pickleball_vision.dataset_workflow import (
    extract_ball_dataset_frames,
    split_ball_dataset,
)
from pickleball_vision.errors import DatasetInputError, DatasetIoError
from pickleball_vision.media import inspect_media


def test_cadence_extraction_writes_full_resolution_grouped_images_and_manifest(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "dataset"

    artifacts = extract_ball_dataset_frames(
        synthetic_video,
        output_dir=output_dir,
        selection=FrameSelectionSettings(every_frames=3),
        label_group=DatasetLabelGroup.NEGATIVE,
    )

    assert artifacts.frame_count == 4
    assert artifacts.written_clip_count == 0
    assert all((output_dir / "images" / group.value).is_dir() for group in DatasetLabelGroup)
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["record_type"] == "ball_annotation_frame_dataset"
    assert manifest["source"]["source_id"].startswith("sha256:")
    assert len(manifest["source"]["content_sha256"]) == 64
    assert manifest["source"]["media"]["frame_count"] == 12
    assert tuple(frame["frame_number"] for frame in manifest["frames"]) == (0, 3, 6, 9)
    assert all(frame["label_group"] == "negative" for frame in manifest["frames"])
    assert manifest["contracts"]["model_predictions_included"] is False
    for frame in manifest["frames"]:
        relative_path = Path(frame["relative_image_path"])
        assert relative_path.parts[:2] == ("images", "negative")
        image = cv2.imread(str(output_dir / relative_path))
        assert image is not None
        assert image.shape[:2] == (64, 96)

    with pytest.raises(DatasetIoError, match="already contains a manifest"):
        extract_ball_dataset_frames(
            synthetic_video,
            output_dir=output_dir,
            selection=FrameSelectionSettings(every_frames=3),
        )


def test_seeded_random_time_range_is_reproducible(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    manifests: list[dict[str, object]] = []
    for name in ("first", "second"):
        artifacts = extract_ball_dataset_frames(
            synthetic_video,
            output_dir=tmp_path / name,
            selection=FrameSelectionSettings(random_count=4, random_seed=19),
            start_time_s=0.4,
            end_time_s=1.4,
        )
        manifests.append(json.loads(artifacts.manifest_path.read_text(encoding="utf-8")))

    first_frames = manifests[0]["frames"]
    second_frames = manifests[1]["frames"]
    assert isinstance(first_frames, list)
    assert isinstance(second_frames, list)
    first_indices = tuple(frame["frame_number"] for frame in first_frames)
    second_indices = tuple(frame["frame_number"] for frame in second_frames)
    assert first_indices == second_indices
    assert all(3 <= index < 11 for index in first_indices)


def test_named_clips_group_frames_and_preserve_audio_in_lossless_review_clips(
    synthetic_media_with_audio: Path,
    tmp_path: Path,
) -> None:
    source_bytes = synthetic_media_with_audio.read_bytes()
    clips_path = tmp_path / "clips.json"
    clips_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "ball_dataset_clips",
                "clips": [
                    {
                        "clip_id": "rally-001",
                        "start_time_s": 0.0,
                        "end_time_s": 0.8,
                        "group_id": "rally-001",
                        "label_group": "positive",
                    },
                    {
                        "clip_id": "between-rallies-001",
                        "start_time_s": 0.8,
                        "end_time_s": 1.6,
                        "group_id": "between-rallies-001",
                        "label_group": "negative",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    artifacts = extract_ball_dataset_frames(
        synthetic_media_with_audio,
        output_dir=tmp_path / "dataset",
        selection=FrameSelectionSettings(every_frames=3),
        clip_definitions_path=clips_path,
        write_clips=True,
    )

    assert synthetic_media_with_audio.read_bytes() == source_bytes
    assert artifacts.frame_count == 4
    assert artifacts.written_clip_count == 2
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert tuple(frame["frame_number"] for frame in manifest["frames"]) == (0, 3, 6, 9)
    assert {frame["label_group"] for frame in manifest["frames"]} == {
        "positive",
        "negative",
    }
    for clip in manifest["clips"]:
        clip_path = artifacts.output_dir / clip["relative_media_path"]
        clip_media = inspect_media(clip_path)
        assert clip_media.audio is not None
        assert clip_media.video.width == 96
        assert clip_media.video.height == 64
        assert clip["media_artifact"]["source_preserved"] is True
        assert clip["media_artifact"]["conversion"]["video_lossless"] is True


def test_split_manifest_keeps_each_clip_wholly_in_one_split(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    clips_path = tmp_path / "clips.json"
    clips_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "ball_dataset_clips",
                "clips": [
                    {
                        "clip_id": "clip-a",
                        "start_time_s": 0.0,
                        "end_time_s": 0.8,
                        "group_id": "rally-a",
                    },
                    {
                        "clip_id": "clip-b",
                        "start_time_s": 0.8,
                        "end_time_s": 1.6,
                        "group_id": "rally-b",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    extracted = extract_ball_dataset_frames(
        synthetic_video,
        output_dir=tmp_path / "dataset",
        selection=FrameSelectionSettings(every_frames=2),
        clip_definitions_path=clips_path,
    )

    split = split_ball_dataset(
        (extracted.manifest_path,),
        output_path=tmp_path / "split.json",
        split_by=SplitUnit.CLIP,
        ratios=SplitRatios(train=0.5, validation=0.0, test=0.5),
        random_seed=5,
    )

    payload = json.loads(split.output_path.read_text(encoding="utf-8"))
    assert split.unit_count == 2
    assert {frame["split"] for frame in payload["frames"]} == {"train", "test"}
    by_clip: dict[str, set[str]] = {}
    for frame in payload["frames"]:
        by_clip.setdefault(frame["clip_id"], set()).add(frame["split"])
    assert all(len(splits) == 1 for splits in by_clip.values())
    assert payload["contracts"]["individual_frames_randomly_split"] is False


def test_extraction_rejects_mixed_range_modes_and_group_split_without_groups(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    clips_path = tmp_path / "clips.json"
    clips_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "ball_dataset_clips",
                "clips": [{"clip_id": "clip-a", "start_time_s": 0.0, "end_time_s": 1.0}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DatasetInputError, match="cannot be combined"):
        extract_ball_dataset_frames(
            synthetic_video,
            output_dir=tmp_path / "invalid",
            selection=FrameSelectionSettings(every_frames=2),
            start_time_s=0.1,
            clip_definitions_path=clips_path,
        )

    extracted = extract_ball_dataset_frames(
        synthetic_video,
        output_dir=tmp_path / "ungrouped",
        selection=FrameSelectionSettings(every_frames=2),
    )
    with pytest.raises(DatasetInputError, match="has no group_id"):
        split_ball_dataset(
            (extracted.manifest_path,),
            output_path=tmp_path / "split.json",
            split_by=SplitUnit.GROUP,
            ratios=SplitRatios(),
            random_seed=0,
        )
