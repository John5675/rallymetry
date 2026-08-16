import json
from pathlib import Path
from typing import Any, cast

import pytest

from pickleball_vision.ball_review import BallAnnotationReviewStore
from pickleball_vision.dataset import DatasetLabelGroup, FrameSelectionSettings
from pickleball_vision.dataset_workflow import extract_ball_dataset_frames
from pickleball_vision.errors import DatasetInputError


def _review_inputs(synthetic_video: Path, tmp_path: Path) -> tuple[Path, Path, Path]:
    extraction = extract_ball_dataset_frames(
        synthetic_video,
        output_dir=tmp_path / "extracted",
        selection=FrameSelectionSettings(every_frames=4),
        label_group=DatasetLabelGroup.UNLABELED,
    )
    manifest = json.loads(extraction.manifest_path.read_text(encoding="utf-8"))
    source_id = manifest["source"]["source_id"]
    split_frames: list[dict[str, Any]] = []
    for split, frame in zip(("train", "validation", "test"), manifest["frames"], strict=True):
        split_frames.append(
            {
                "manifest_path": str(extraction.manifest_path),
                "record_id": frame["record_id"],
                "source_id": source_id,
                "relative_image_path": frame["relative_image_path"],
                "frame_number": frame["frame_number"],
                "label_group": "unlabeled",
                "clip_id": frame["clip_id"],
                "group_id": frame["group_id"],
                "split_unit_key": f"unit-{split}",
                "split": split,
            }
        )
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "ball_dataset_split_assignments",
                "frames": split_frames,
            }
        ),
        encoding="utf-8",
    )
    predictions_path = tmp_path / "detections.json"
    predictions_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "raw_pickleball_detections",
                "source": manifest["source"]["media"],
                "detector": {
                    "model_version": "test-ball-v1",
                    "weights_sha256": "a" * 64,
                },
                "strategy": {"name": "full-640"},
                "temporal_processing": {
                    "tracking": False,
                    "interpolation": False,
                    "events": False,
                },
                "frames": [
                    {
                        "frame_number": frame["frame_number"],
                        "detections": (
                            [
                                {
                                    "detection_id": "suggestion-1",
                                    "confidence": 0.42,
                                    "bounding_box": {
                                        "left_px": 5,
                                        "top_px": 6,
                                        "right_px": 12,
                                        "bottom_px": 13,
                                    },
                                }
                            ]
                            if index == 0
                            else []
                        ),
                    }
                    for index, frame in enumerate(manifest["frames"])
                ],
            }
        ),
        encoding="utf-8",
    )
    return split_path, tmp_path / "annotations.json", predictions_path


def test_review_store_keeps_suggestions_separate_and_resumes(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    split_path, annotations_path, predictions_path = _review_inputs(synthetic_video, tmp_path)
    store = BallAnnotationReviewStore(
        split_path,
        annotations_path=annotations_path,
        dataset_version="review-v1",
        prediction_paths=(predictions_path,),
    )

    initial = store.frame_payload(0)
    annotation = initial["annotation"]
    assert isinstance(annotation, dict)
    assert annotation["review_status"] == "unreviewed"
    assert annotation["objects"] == []
    assert len(cast(list[object], initial["suggestions"])) == 1

    store.update_frame(
        0,
        {
            "review_status": "reviewed",
            "reviewer": "human-reviewer",
            "objects": [
                {
                    "bounding_box": {
                        "left_px": 5,
                        "top_px": 6,
                        "right_px": 12,
                        "bottom_px": 13,
                    },
                    "court_side": "far",
                    "scope": "primary_match",
                    "visibility": "blurred",
                    "annotation_confidence": 0.9,
                }
            ],
        },
    )
    store.update_frame(
        1,
        {"review_status": "reviewed", "reviewer": "human-reviewer", "objects": []},
    )

    summary = store.session_payload()
    counts = summary["counts"]
    assert isinstance(counts, dict)
    assert counts["reviewed_frames"] == 2
    assert counts["positive_frames"] == 1
    assert counts["negative_frames"] == 1
    assert counts["annotation_objects"] == 1
    contracts = cast(dict[str, object], summary["contracts"])
    assert contracts["model_suggestions_are_ground_truth"] is False
    saved = json.loads(annotations_path.read_text(encoding="utf-8"))
    assert saved["frames"][0]["objects"][0]["court_side"] == "far"
    assert saved["frames"][0]["review_metadata"]["reviewer"] == "human-reviewer"
    assert store.summary_path.is_file()

    resumed = BallAnnotationReviewStore(
        split_path,
        annotations_path=annotations_path,
        dataset_version="review-v1",
        prediction_paths=(predictions_path,),
    )
    resumed_counts = resumed.session_payload()["counts"]
    assert isinstance(resumed_counts, dict)
    assert resumed_counts["reviewed_frames"] == 2


def test_review_store_rejects_ambiguous_reviewed_box_and_out_of_frame_box(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    split_path, annotations_path, _ = _review_inputs(synthetic_video, tmp_path)
    store = BallAnnotationReviewStore(
        split_path,
        annotations_path=annotations_path,
        dataset_version="review-v1",
    )
    base_object = {
        "bounding_box": {"left_px": 5, "top_px": 5, "right_px": 10, "bottom_px": 10},
        "court_side": "unknown",
        "scope": "unknown",
        "visibility": "ambiguous",
    }
    with pytest.raises(DatasetInputError, match="ambiguous boxes"):
        store.update_frame(
            0,
            {"review_status": "reviewed", "reviewer": "reviewer", "objects": [base_object]},
        )

    outside = {**base_object, "visibility": "clear"}
    outside["bounding_box"] = {
        "left_px": 90,
        "top_px": 5,
        "right_px": 100,
        "bottom_px": 10,
    }
    with pytest.raises(DatasetInputError, match="outside"):
        store.update_frame(
            0,
            {"review_status": "unreviewed", "reviewer": "reviewer", "objects": [outside]},
        )


def test_review_store_requires_version_when_creating_annotations(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    split_path, annotations_path, _ = _review_inputs(synthetic_video, tmp_path)

    with pytest.raises(DatasetInputError, match="dataset-version"):
        BallAnnotationReviewStore(split_path, annotations_path=annotations_path)
