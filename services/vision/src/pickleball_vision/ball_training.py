"""Reproducible custom pickleball detector training orchestration."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol

from pickleball_vision.ball_config import (
    BallExperimentConfiguration,
    load_ball_experiment_configuration,
)
from pickleball_vision.ball_dataset import (
    PreparedBallDataset,
    load_ball_detector_dataset,
    prepare_yolo_ball_dataset,
)
from pickleball_vision.errors import BallTrainingError, OutputWriteError

BALL_EXPERIMENT_SCHEMA_VERSION = 1
BALL_METRICS_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BallTrainingOutcome:
    """Model-specific results translated into stable experiment fields."""

    backend: str
    framework_version: str | None
    effective_device: str
    best_weights_path: Path
    last_weights_path: Path | None
    metrics: dict[str, float]


class BallTrainingBackend(Protocol):
    """Model-specific training boundary."""

    def train(
        self,
        *,
        config: BallExperimentConfiguration,
        prepared_dataset: PreparedBallDataset,
        output_dir: Path,
    ) -> BallTrainingOutcome:
        """Train one model and return weights plus numeric validation metrics."""

        ...


@dataclass(frozen=True, slots=True)
class BallTrainingArtifacts:
    """Persisted experiment metadata, metrics, and model paths."""

    experiment_path: Path
    metrics_path: Path
    prepared_dataset_path: Path
    best_weights_path: Path
    model_version: str
    dataset_version: str
    experiment_id: str

    def as_dict(self) -> dict[str, object]:
        return {
            "experiment_path": str(self.experiment_path),
            "metrics_path": str(self.metrics_path),
            "prepared_dataset_path": str(self.prepared_dataset_path),
            "best_weights_path": str(self.best_weights_path),
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "experiment_id": self.experiment_id,
        }


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except (OSError, TypeError, ValueError) as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise OutputWriteError(str(path), reason=str(error)) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise BallTrainingError(f"unable to hash {path}: {error}") from error
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_revision(working_directory: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(working_directory), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _experiment_id(
    config: BallExperimentConfiguration,
    split_hash: str,
    annotation_hash: str,
) -> str:
    payload = json.dumps(config.as_dict(), sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(payload + split_hash.encode() + annotation_hash.encode()).hexdigest()
    return f"{config.experiment_name}-{digest[:12]}"


def train_ball_detector(
    config_path: Path,
    *,
    output_dir: Path,
    backend: BallTrainingBackend | None = None,
) -> BallTrainingArtifacts:
    """Validate fixed splits, prepare labels, train, and persist experiment evidence."""

    config = load_ball_experiment_configuration(config_path)
    dataset = load_ball_detector_dataset(
        dataset_version=config.dataset.version,
        split_manifest_path=config.dataset.split_manifest_path,
        annotations_path=config.dataset.annotations_path,
    )
    output = output_dir.expanduser().resolve()
    experiment_path = output / "experiment.json"
    metrics_path = output / "metrics.json"
    if experiment_path.exists() or metrics_path.exists():
        raise OutputWriteError(str(output), reason="training experiment output already exists")
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OutputWriteError(str(output), reason=str(error)) from error
    prepared = prepare_yolo_ball_dataset(dataset, output_dir=output / "prepared-dataset")
    experiment_id = _experiment_id(
        config, dataset.split_manifest_sha256, dataset.annotations_sha256
    )
    started_at = datetime.now(UTC).isoformat()
    base_metadata = {
        "schema_version": BALL_EXPERIMENT_SCHEMA_VERSION,
        "record_type": "pickleball_detector_training_experiment",
        "experiment_id": experiment_id,
        "status": "running",
        "created_at_utc": started_at,
        "configuration_path": str(config.path),
        "configuration_sha256": _sha256(config.path),
        "configuration": config.as_dict(),
        "dataset": dataset.as_dict(),
        "prepared_dataset": prepared.as_dict(),
        "reproducibility": {
            "code_revision": _git_revision(config.path.parent),
            "python_version": sys.version,
            "platform": platform.platform(),
            "random_seed": config.training.seed,
            "deterministic_requested": config.training.deterministic,
            "packages": {
                "opencv-python": _package_version("opencv-python"),
                "torch": _package_version("torch"),
                "ultralytics": _package_version("ultralytics"),
            },
        },
    }
    _write_json(experiment_path, base_metadata)
    try:
        if backend is None:
            from pickleball_vision.trainers import UltralyticsBallTrainer

            active_backend: BallTrainingBackend = UltralyticsBallTrainer()
        else:
            active_backend = backend
        outcome = active_backend.train(
            config=config,
            prepared_dataset=prepared,
            output_dir=output / "backend",
        )
        best_weights = outcome.best_weights_path.expanduser().resolve()
        if not best_weights.is_file():
            raise BallTrainingError(f"training backend produced no best weights: {best_weights}")
        metrics_payload = {
            "schema_version": BALL_METRICS_SCHEMA_VERSION,
            "record_type": "pickleball_detector_training_metrics",
            "experiment_id": experiment_id,
            "dataset_version": config.dataset.version,
            "model_version": config.model.version,
            "fixed_validation_set": True,
            "fixed_test_set": True,
            "backend_metrics": outcome.metrics,
        }
        _write_json(metrics_path, metrics_payload)
        completed = {
            **base_metadata,
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "training_backend": {
                "name": outcome.backend,
                "framework_version": outcome.framework_version,
                "effective_device": outcome.effective_device,
            },
            "model_artifacts": {
                "model_version": config.model.version,
                "best_weights_path": str(best_weights),
                "best_weights_sha256": _sha256(best_weights),
                "last_weights_path": (
                    str(outcome.last_weights_path.resolve())
                    if outcome.last_weights_path is not None
                    else None
                ),
            },
            "metrics_path": str(metrics_path),
        }
        _write_json(experiment_path, completed)
    except Exception as error:
        failed = {
            **base_metadata,
            "status": "failed",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "failure": str(error),
        }
        _write_json(experiment_path, failed)
        if isinstance(error, (BallTrainingError, OutputWriteError)):
            raise
        raise BallTrainingError(str(error)) from error
    return BallTrainingArtifacts(
        experiment_path=experiment_path,
        metrics_path=metrics_path,
        prepared_dataset_path=prepared.root,
        best_weights_path=best_weights,
        model_version=config.model.version,
        dataset_version=config.dataset.version,
        experiment_id=experiment_id,
    )
