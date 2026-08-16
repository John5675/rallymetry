from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from pickleball_vision.ball_config import (
    BallInferenceMode,
    BallInferenceStrategy,
    load_ball_experiment_configuration,
)
from pickleball_vision.ball_dataset import (
    BallCourtSide,
    BallScope,
    BallVisibility,
    DetectorDatasetFrame,
    GroundTruthBall,
    create_ball_annotation_template,
    load_ball_detector_dataset,
    prepare_yolo_ball_dataset,
)
from pickleball_vision.ball_detection import (
    BallDetection,
    BallDetectorMetadata,
    BallFrameInference,
    BallModelPrediction,
    build_inference_regions,
    infer_ball_frame,
)
from pickleball_vision.ball_detection_workflow import detect_balls_in_video
from pickleball_vision.ball_evaluation import (
    calculate_detection_metrics,
    compare_ball_inference_strategies,
)
from pickleball_vision.ball_training import BallTrainingOutcome, train_ball_detector
from pickleball_vision.calibration import load_calibration
from pickleball_vision.dataset import (
    DatasetLabelGroup,
    DatasetSplit,
    FrameSelectionSettings,
)
from pickleball_vision.dataset_workflow import extract_ball_dataset_frames
from pickleball_vision.detectors.ultralytics_ball import UltralyticsBallDetector
from pickleball_vision.errors import DatasetInputError
from pickleball_vision.person_detection import BoundingBox
from pickleball_vision.trainers import UltralyticsBallTrainer
from pickleball_vision.video import Image


def _strategy(
    name: str = "full-640",
    mode: BallInferenceMode = BallInferenceMode.FULL_FRAME,
    *,
    tile_size_px: int | None = None,
) -> BallInferenceStrategy:
    return BallInferenceStrategy(
        name=name,
        mode=mode,
        inference_size_px=640,
        minimum_confidence=0.1,
        model_nms_iou_threshold=0.5,
        maximum_detections=20,
        tile_size_px=tile_size_px,
        tile_overlap_fraction=0.5,
        court_roi_margin_px=4,
        merge_iou_threshold=0.5,
    )


class StaticBallDetector:
    def __init__(self, predictions: tuple[BallModelPrediction, ...]) -> None:
        self.predictions = predictions
        self.calls: list[tuple[tuple[int, ...], int]] = []

    @property
    def metadata(self) -> BallDetectorMetadata:
        return BallDetectorMetadata(
            adapter="static_test",
            framework="test",
            framework_version="1",
            model_version="ball-test-v1",
            weights_path=Path("/tmp/test-ball.pt"),
            weights_sha256="0" * 64,
            device="cpu",
        )

    def predict(
        self,
        image: Image,
        *,
        inference_size_px: int,
        minimum_confidence: float,
        nms_iou_threshold: float,
        maximum_detections: int,
    ) -> tuple[BallModelPrediction, ...]:
        del minimum_confidence, nms_iou_threshold, maximum_detections
        self.calls.append((image.shape, inference_size_px))
        return self.predictions


class OverlappingTileDetector(StaticBallDetector):
    def predict(
        self,
        image: Image,
        *,
        inference_size_px: int,
        minimum_confidence: float,
        nms_iou_threshold: float,
        maximum_detections: int,
    ) -> tuple[BallModelPrediction, ...]:
        del minimum_confidence, nms_iou_threshold, maximum_detections
        self.calls.append((image.shape, inference_size_px))
        left = 50 if len(self.calls) == 1 else 18
        return (
            BallModelPrediction(
                BoundingBox(left, 10, left + 10, 20),
                0.9 if len(self.calls) == 1 else 0.8,
            ),
        )


def _make_fixed_dataset(
    synthetic_video: Path,
    tmp_path: Path,
    *,
    calibration_path: Path | None = None,
) -> tuple[Path, Path, Path, str]:
    extraction = extract_ball_dataset_frames(
        synthetic_video,
        output_dir=tmp_path / "extracted",
        selection=FrameSelectionSettings(every_frames=4),
        label_group=DatasetLabelGroup.UNLABELED,
    )
    source_manifest = json.loads(extraction.manifest_path.read_text(encoding="utf-8"))
    source_frames = source_manifest["frames"]
    source_id = source_manifest["source"]["source_id"]
    partitions = ("train", "validation", "test")
    split_frames = []
    for partition, frame in zip(partitions, source_frames, strict=True):
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
                "split_unit_key": f"test-unit-{partition}",
                "split": partition,
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
    boxes: tuple[list[dict[str, Any]], ...] = (
        [
            {
                "annotation_id": "train-ball",
                "class_name": "pickleball",
                "bounding_box": {
                    "left_px": 10,
                    "top_px": 10,
                    "right_px": 20,
                    "bottom_px": 20,
                },
                "court_side": "near",
                "scope": "primary_match",
                "visibility": "clear",
            }
        ],
        [
            {
                "annotation_id": "validation-ball",
                "class_name": "pickleball",
                "bounding_box": {
                    "left_px": 30,
                    "top_px": 10,
                    "right_px": 40,
                    "bottom_px": 20,
                },
                "court_side": "far",
                "scope": "primary_match",
                "visibility": "blurred",
            }
        ],
        [],
    )
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "pickleball_detection_annotations",
                "dataset_version": "synthetic-v1",
                "class_name": "pickleball",
                "frames": [
                    {
                        "record_id": frame["record_id"],
                        "review_status": "reviewed",
                        "objects": objects,
                    }
                    for frame, objects in zip(source_frames, boxes, strict=True)
                ],
            }
        ),
        encoding="utf-8",
    )
    calibration_mapping = {source_id: str(calibration_path)} if calibration_path is not None else {}
    config_path = tmp_path / "experiment-config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "pickleball_detector_experiment_configuration",
                "experiment_name": "synthetic-ball",
                "dataset": {
                    "version": "synthetic-v1",
                    "split_manifest": "split.json",
                    "annotations": "annotations.json",
                },
                "model": {"version": "ball-test-v1", "base_model": "fake-base.pt"},
                "training": {
                    "epochs": 2,
                    "image_size_px": 640,
                    "batch_size": 2,
                    "device": "cpu",
                    "workers": 0,
                    "seed": 2026,
                    "deterministic": True,
                    "patience_epochs": 1,
                },
                "inference": {
                    "strategies": [
                        _strategy().as_dict(),
                        _strategy("tiled-640", BallInferenceMode.TILED, tile_size_px=64).as_dict(),
                    ]
                },
                "evaluation": {
                    "matching_iou_threshold": 0.5,
                    "calibrations_by_source_id": calibration_mapping,
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path, split_path, annotations_path, source_id


def test_config_dataset_loading_and_yolo_preparation(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    config_path, _, _, _ = _make_fixed_dataset(synthetic_video, tmp_path)
    config = load_ball_experiment_configuration(config_path)
    assert config.dataset.version == "synthetic-v1"
    assert config.strategy("tiled-640").mode is BallInferenceMode.TILED

    dataset = load_ball_detector_dataset(
        dataset_version=config.dataset.version,
        split_manifest_path=config.dataset.split_manifest_path,
        annotations_path=config.dataset.annotations_path,
    )
    assert len(dataset.frames) == 3
    assert len(dataset.partition(DatasetSplit.TRAIN)[0].objects) == 1
    assert dataset.partition(DatasetSplit.TEST)[0].objects == ()

    prepared = prepare_yolo_ball_dataset(dataset, output_dir=tmp_path / "prepared")
    assert prepared.dataset_yaml_path.is_file()
    assert "0: pickleball" in prepared.dataset_yaml_path.read_text(encoding="utf-8")
    train_labels = tuple((prepared.root / "labels/train").glob("*.txt"))
    test_labels = tuple((prepared.root / "labels/test").glob("*.txt"))
    assert len(train_labels) == 1
    assert train_labels[0].read_text(encoding="utf-8").startswith("0 ")
    assert test_labels[0].read_text(encoding="utf-8") == ""


def test_annotation_template_is_unreviewed_and_training_rejects_it(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    _, split_path, _, _ = _make_fixed_dataset(synthetic_video, tmp_path)
    template = create_ball_annotation_template(
        split_path,
        dataset_version="synthetic-v2",
        output_path=tmp_path / "template.json",
    )
    payload = json.loads(template.output_path.read_text(encoding="utf-8"))
    assert template.frame_count == 3
    assert all(frame["review_status"] == "unreviewed" for frame in payload["frames"])
    with pytest.raises(DatasetInputError, match="not human-reviewed"):
        load_ball_detector_dataset(
            dataset_version="synthetic-v2",
            split_manifest_path=split_path,
            annotations_path=template.output_path,
        )


def test_tiled_inference_restores_source_coordinates_and_retains_proposals() -> None:
    detector = OverlappingTileDetector(())
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    result = infer_ball_frame(
        image,
        frame_number=4,
        timestamp_s=0.5,
        strategy=_strategy("tiled", BallInferenceMode.TILED, tile_size_px=64),
        detector=detector,
    )
    assert len(result.regions) == 2
    assert len(result.region_predictions) == 2
    assert result.region_predictions[0].bounding_box.left_px == 50
    assert result.region_predictions[1].bounding_box.left_px == 50
    assert len(result.detections) == 1
    assert len(result.detections[0].supporting_prediction_ids) == 2


def test_court_roi_is_an_image_crop_only(
    synthetic_calibration: Path,
) -> None:
    calibration = load_calibration(synthetic_calibration)
    regions = build_inference_regions(
        frame_width=96,
        frame_height=64,
        strategy=_strategy("court", BallInferenceMode.COURT_ROI),
        calibration=calibration,
    )
    assert len(regions) == 1
    assert regions[0].kind == "court_roi"
    assert regions[0].width < 96 or regions[0].height < 64


def test_metrics_include_false_positives_coverage_and_near_far() -> None:
    frames = (
        DetectorDatasetFrame(
            record_id="near",
            source_id="source",
            image_path=Path("near.jpg"),
            width=100,
            height=100,
            frame_number=0,
            timestamp_s=0,
            split=DatasetSplit.VALIDATION,
            split_unit_key="clip-a",
            clip_id="clip-a",
            group_id="rally-a",
            clip_start_time_s=0,
            clip_end_time_s=60,
            objects=(
                GroundTruthBall(
                    "near-ball",
                    BoundingBox(10, 10, 20, 20),
                    BallCourtSide.NEAR,
                    BallScope.PRIMARY_MATCH,
                    BallVisibility.CLEAR,
                ),
            ),
        ),
        DetectorDatasetFrame(
            record_id="far",
            source_id="source",
            image_path=Path("far.jpg"),
            width=100,
            height=100,
            frame_number=1,
            timestamp_s=1,
            split=DatasetSplit.VALIDATION,
            split_unit_key="clip-a",
            clip_id="clip-a",
            group_id="rally-a",
            clip_start_time_s=0,
            clip_end_time_s=60,
            objects=(
                GroundTruthBall(
                    "far-ball",
                    BoundingBox(50, 50, 60, 60),
                    BallCourtSide.FAR,
                    BallScope.PRIMARY_MATCH,
                    BallVisibility.CLEAR,
                ),
            ),
        ),
    )
    detection = BallDetection(
        "detection",
        0,
        0,
        BoundingBox(10, 10, 20, 20),
        0.9,
        ("proposal",),
    )
    false_positive = BallDetection(
        "false-positive",
        0,
        0,
        BoundingBox(70, 70, 80, 80),
        0.8,
        ("proposal-2",),
    )
    inferences = (
        BallFrameInference(0, 0, (), (), (detection, false_positive)),
        BallFrameInference(1, 1, (), (), ()),
    )
    metrics, _ = calculate_detection_metrics(frames, inferences, iou_threshold=0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["false_positives"] == 1
    assert metrics["false_positives_per_minute"] == pytest.approx(1.0)
    assert metrics["detection_coverage"] == pytest.approx(0.5)
    near_side = cast(dict[str, object], metrics["near_side"])
    far_side = cast(dict[str, object], metrics["far_side"])
    assert near_side["recall"] == pytest.approx(1.0)
    assert far_side["recall"] == pytest.approx(0.0)


class FakeTrainingBackend:
    def train(self, *, config: Any, prepared_dataset: Any, output_dir: Path) -> BallTrainingOutcome:
        del config, prepared_dataset
        weights = output_dir / "weights" / "best.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"fake custom weights")
        return BallTrainingOutcome(
            backend="fake",
            framework_version="1",
            effective_device="cpu",
            best_weights_path=weights,
            last_weights_path=None,
            metrics={"metrics/precision(B)": 0.75},
        )


class FakeUltralyticsTrainingResult:
    def __init__(self, save_dir: Path) -> None:
        self.save_dir = save_dir
        self.results_dict: dict[str, object] = {"metrics/precision(B)": 0.5}


class FakeUltralyticsTrainingModel:
    def __init__(self, save_dir: Path) -> None:
        self.save_dir = save_dir
        self.arguments: dict[str, object] = {}

    def train(self, **kwargs: object) -> FakeUltralyticsTrainingResult:
        self.arguments = kwargs
        weights = self.save_dir / "weights"
        weights.mkdir(parents=True)
        (weights / "best.pt").write_bytes(b"best")
        return FakeUltralyticsTrainingResult(self.save_dir)


def test_ultralytics_trainer_preserves_pickleball_class_name(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    config_path, split_path, annotations_path, _ = _make_fixed_dataset(synthetic_video, tmp_path)
    config = load_ball_experiment_configuration(config_path)
    dataset = load_ball_detector_dataset(
        dataset_version=config.dataset.version,
        split_manifest_path=split_path,
        annotations_path=annotations_path,
    )
    prepared = prepare_yolo_ball_dataset(dataset, output_dir=tmp_path / "prepared")
    model = FakeUltralyticsTrainingModel(tmp_path / "backend" / "ultralytics")

    UltralyticsBallTrainer(
        model=model,
        effective_device="cpu",
        framework_version="test",
    ).train(config=config, prepared_dataset=prepared, output_dir=tmp_path / "backend")

    assert model.arguments["single_cls"] is False
    assert "0: pickleball" in prepared.dataset_yaml_path.read_text(encoding="utf-8")


def test_training_persists_versions_hashes_and_metrics(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    config_path, _, _, _ = _make_fixed_dataset(synthetic_video, tmp_path)
    artifacts = train_ball_detector(
        config_path,
        output_dir=tmp_path / "experiment",
        backend=FakeTrainingBackend(),
    )
    experiment = json.loads(artifacts.experiment_path.read_text(encoding="utf-8"))
    metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
    assert experiment["status"] == "complete"
    assert experiment["configuration"]["dataset"]["version"] == "synthetic-v1"
    assert experiment["model_artifacts"]["model_version"] == "ball-test-v1"
    assert len(experiment["model_artifacts"]["best_weights_sha256"]) == 64
    assert metrics["fixed_validation_set"] is True
    assert metrics["fixed_test_set"] is True
    assert metrics["backend_metrics"]["metrics/precision(B)"] == pytest.approx(0.75)


def test_video_inference_and_strategy_comparison_write_raw_outputs(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    config_path, _, _, _ = _make_fixed_dataset(synthetic_video, tmp_path)
    detector = StaticBallDetector((BallModelPrediction(BoundingBox(30, 10, 40, 20), 0.9),))
    video_artifacts = detect_balls_in_video(
        synthetic_video,
        weights_path=tmp_path / "unused.pt",
        model_version="ball-test-v1",
        strategy=_strategy(),
        output_dir=tmp_path / "video-output",
        detector=detector,
    )
    payload = json.loads(video_artifacts.detections_path.read_text(encoding="utf-8"))
    assert payload["record_type"] == "raw_pickleball_detections"
    assert payload["temporal_processing"]["tracking"] is False
    assert payload["frames"][0]["detections"][0]["temporal_track_id"] is None

    comparison = compare_ball_inference_strategies(
        config_path,
        weights_path=tmp_path / "unused.pt",
        partition=DatasetSplit.VALIDATION,
        output_dir=tmp_path / "comparison",
        detector=detector,
    )
    report = json.loads(comparison.comparison_path.read_text(encoding="utf-8"))
    assert len(report["rows"]) == 2
    assert report["rows"][0]["evaluated_frames"] == 1
    assert report["fixed_frame_record_ids"]


class FakeTensor:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values, dtype=np.float32)

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> object:
        return self.values


class FakeBoxes:
    xyxy = FakeTensor([[5, 6, 10, 12], [1, 1, 3, 3]])
    conf = FakeTensor([0.8, 0.9])
    cls = FakeTensor([0, 1])


class FakeResult:
    boxes = FakeBoxes()


class FakeUltralyticsModel:
    def __init__(self) -> None:
        self.names = {0: "pickleball"}
        self.arguments: dict[str, object] = {}

    def predict(self, **kwargs: object) -> Sequence[FakeResult]:
        self.arguments = kwargs
        return [FakeResult()]


def test_ultralytics_adapter_validates_custom_class_and_forwards_resolution(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "ball.pt"
    weights.write_bytes(b"weights")
    model = FakeUltralyticsModel()
    detector = UltralyticsBallDetector(
        weights,
        model_version="ball-v1",
        device="cpu",
        model=model,
        framework_version="test",
    )
    predictions = detector.predict(
        np.zeros((64, 96, 3), dtype=np.uint8),
        inference_size_px=1280,
        minimum_confidence=0.2,
        nms_iou_threshold=0.4,
        maximum_detections=50,
    )
    assert len(predictions) == 1
    assert model.arguments["imgsz"] == 1280
    assert model.arguments["classes"] == [0]
    assert detector.metadata.model_version == "ball-v1"
    assert len(detector.metadata.weights_sha256) == 64
