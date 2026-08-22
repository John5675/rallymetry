import json
from pathlib import Path

import numpy as np
import pytest

from pickleball_vision.errors import ShotModelInputError
from pickleball_vision.racketvision import (
    FEATURE_COUNT,
    RacketVisionManifest,
    discover_racketvision_sequences,
    load_racketvision_features,
)
from pickleball_vision.shot_pretraining import (
    RepresentationTrainingOutcome,
    ShotRepresentationSettings,
    pretrain_shot_representation,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _external_root(tmp_path: Path) -> Path:
    root = tmp_path / "RacketVision"
    root.mkdir()
    (root / "README.md").write_text("---\nlicense: mit\n---\n", encoding="utf-8")
    pairs = {
        "train": ["match1", "000"],
        "validation": ["match2", "000"],
        "test": ["match3", "000"],
    }
    for sport in ("badminton", "tabletennis", "tennis"):
        for filename, partition in (
            ("train.json", "train"),
            ("val.json", "validation"),
            ("test.json", "test"),
        ):
            split = [pairs[partition]] if sport == "badminton" else []
            _write_json(root / sport / "info" / filename, split)
    for match_id in ("match1", "match2", "match3"):
        ball_path = root / "badminton" / "interp_ball" / match_id / "000" / "results.csv"
        ball_path.parent.mkdir(parents=True, exist_ok=True)
        rows = ["Frame,X,Y,Visibility,Confidence"]
        rows.extend(f"{frame},{100 + frame},{200 + frame},1,0.9" for frame in range(30))
        ball_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        racket_frame = [
            {
                "keypoints": [[100 + point, 200 + point] for point in range(5)],
                "keypoint_scores": [0.9] * 5,
            }
        ]
        _write_json(
            root / "badminton" / "merged_racket" / match_id / "000" / "result.json",
            [racket_frame for _ in range(30)],
        )
    return root


class _FakeBackend:
    @property
    def name(self) -> str:
        return "fake_temporal_backend"

    @property
    def version(self) -> str:
        return "1"

    def train(
        self,
        manifest: RacketVisionManifest,
        *,
        settings: ShotRepresentationSettings,
        weights_path: Path,
    ) -> RepresentationTrainingOutcome:
        weights_path.write_bytes(b"representation-only")
        return RepresentationTrainingOutcome(
            weights_path=weights_path,
            training_losses=(0.5, 0.25),
            validation_losses=(0.6, 0.3),
            effective_device="cpu",
            train_sequence_count=1,
            validation_sequence_count=1,
            train_window_count=6,
            validation_window_count=6,
        )


def test_safe_adapter_loads_normalized_ball_and_racket_features(tmp_path: Path) -> None:
    root = _external_root(tmp_path)
    manifest = discover_racketvision_sequences(root, upstream_revision="85157ca")

    assert len(manifest.sequences) == 3
    assert {item.partition for item in manifest.sequences} == {"train", "validation", "test"}
    features = load_racketvision_features(manifest.sequences[0])
    assert features.shape == (30, FEATURE_COUNT)
    assert features.dtype == np.float32
    assert features[0, 2] == 1.0
    assert features[0, 15] == 1.0


def test_safe_adapter_rejects_untrusted_pickle(tmp_path: Path) -> None:
    root = _external_root(tmp_path)
    (root / "data_traj").mkdir()
    (root / "data_traj" / "unsafe.pkl").write_bytes(b"not loaded")

    with pytest.raises(ShotModelInputError, match="unsafe pickle"):
        discover_racketvision_sequences(root, upstream_revision="85157ca")


def test_pretraining_records_representation_only_provenance(tmp_path: Path) -> None:
    root = _external_root(tmp_path)
    config_path = tmp_path / "config.json"
    _write_json(
        config_path,
        {
            "recordType": "shot_representation_pretraining_configuration",
            "schemaVersion": 1,
            "experimentName": "test-representation",
            "modelVersion": "temporal-v1",
            "datasetVersion": "racketvision-test",
            "datasetRoot": str(root),
            "upstreamRevision": "85157ca",
            "training": {
                "historyFrames": 20,
                "futureFrames": 5,
                "hiddenSize": 16,
                "epochs": 2,
                "stepsPerEpoch": 1,
                "validationSteps": 1,
                "batchSize": 2,
                "learningRate": 0.001,
                "seed": 7,
                "deterministic": True,
                "device": "cpu",
                "maximumSequencesPerPartition": None,
            },
        },
    )

    artifacts = pretrain_shot_representation(
        config_path,
        output_dir=tmp_path / "experiment",
        backend=_FakeBackend(),
    )

    experiment = json.loads(artifacts.experiment_path.read_text(encoding="utf-8"))
    metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
    assert experiment["status"] == "complete"
    assert experiment["semanticShotClassifierTrained"] is False
    assert experiment["pickleballSemanticLabelsConsumed"] is False
    assert experiment["externalDataset"]["pickleLoaded"] is False
    assert metrics["semanticAccuracy"] is None
    assert metrics["finalValidationLoss"] == 0.3
