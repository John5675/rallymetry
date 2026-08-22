"""Reproducible temporal representation pretraining without semantic relabeling."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import random
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from pickleball_vision.detectors.ultralytics_person import resolve_inference_device
from pickleball_vision.errors import ShotModelInputError, ShotModelTrainingError
from pickleball_vision.racketvision import (
    FEATURE_COUNT,
    RacketVisionManifest,
    RacketVisionSequence,
    discover_racketvision_sequences,
    load_racketvision_features,
)


@dataclass(frozen=True, slots=True)
class ShotRepresentationSettings:
    """Externalized temporal-prediction hyperparameters."""

    history_frames: int
    future_frames: int
    hidden_size: int
    epochs: int
    steps_per_epoch: int
    validation_steps: int
    batch_size: int
    learning_rate: float
    seed: int
    deterministic: bool
    device: str
    maximum_sequences_per_partition: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "historyFrames": self.history_frames,
            "futureFrames": self.future_frames,
            "hiddenSize": self.hidden_size,
            "epochs": self.epochs,
            "stepsPerEpoch": self.steps_per_epoch,
            "validationSteps": self.validation_steps,
            "batchSize": self.batch_size,
            "learningRate": self.learning_rate,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "device": self.device,
            "maximumSequencesPerPartition": self.maximum_sequences_per_partition,
        }


@dataclass(frozen=True, slots=True)
class ShotRepresentationConfiguration:
    """Complete versioned pretraining input contract."""

    config_path: Path
    experiment_name: str
    model_version: str
    dataset_version: str
    dataset_root: Path
    upstream_revision: str
    settings: ShotRepresentationSettings

    def as_dict(self) -> dict[str, object]:
        return {
            "recordType": "shot_representation_pretraining_configuration",
            "schemaVersion": 1,
            "experimentName": self.experiment_name,
            "modelVersion": self.model_version,
            "datasetVersion": self.dataset_version,
            "datasetRoot": str(self.dataset_root),
            "upstreamRevision": self.upstream_revision,
            "training": self.settings.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class RepresentationTrainingOutcome:
    """Model-independent backend result persisted by the orchestrator."""

    weights_path: Path
    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    effective_device: str
    train_sequence_count: int
    validation_sequence_count: int
    train_window_count: int
    validation_window_count: int


class RepresentationPretrainingBackend(Protocol):
    """Boundary around a concrete temporal model implementation."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def train(
        self,
        manifest: RacketVisionManifest,
        *,
        settings: ShotRepresentationSettings,
        weights_path: Path,
    ) -> RepresentationTrainingOutcome: ...


@dataclass(frozen=True, slots=True)
class ShotPretrainingArtifacts:
    """Experiment records for one completed representation run."""

    experiment_path: Path
    metrics_path: Path
    weights_path: Path
    experiment_id: str

    def as_dict(self) -> dict[str, object]:
        return {
            "experimentPath": str(self.experiment_path),
            "metricsPath": str(self.metrics_path),
            "weightsPath": str(self.weights_path),
            "experimentId": self.experiment_id,
        }


def _read_object(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ShotModelInputError(f"unable to read pretraining config {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ShotModelInputError("pretraining config root must be an object")
    return cast(dict[str, object], raw)


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ShotModelInputError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ShotModelInputError(f"{field} must be a non-empty string")
    return value.strip()


def _integer(value: object, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ShotModelInputError(f"{field} must be an integer >= {minimum}")
    return value


def _float(value: object, field: str, *, minimum_exclusive: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ShotModelInputError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= minimum_exclusive:
        raise ShotModelInputError(f"{field} must be finite and > {minimum_exclusive}")
    return result


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ShotModelInputError(f"{field} must be a boolean")
    return value


def load_shot_representation_configuration(path: Path) -> ShotRepresentationConfiguration:
    """Load and validate one immutable external pretraining configuration."""

    resolved = path.expanduser().resolve()
    root = _read_object(resolved)
    if root.get("recordType") != "shot_representation_pretraining_configuration":
        raise ShotModelInputError(
            "config.recordType must be shot_representation_pretraining_configuration"
        )
    if root.get("schemaVersion") != 1:
        raise ShotModelInputError("config.schemaVersion must be 1")
    training = _object(root.get("training"), "training")
    raw_maximum = training.get("maximumSequencesPerPartition")
    maximum = (
        None
        if raw_maximum is None
        else _integer(raw_maximum, "training.maximumSequencesPerPartition", minimum=1)
    )
    device = _string(training.get("device", "auto"), "training.device").lower()
    if device != "auto" and device != "cpu" and device != "mps" and not device.startswith("cuda"):
        raise ShotModelInputError("training.device must be auto, cpu, mps, or cuda[:N]")
    settings = ShotRepresentationSettings(
        history_frames=_integer(training.get("historyFrames"), "training.historyFrames", minimum=2),
        future_frames=_integer(training.get("futureFrames"), "training.futureFrames", minimum=1),
        hidden_size=_integer(training.get("hiddenSize"), "training.hiddenSize", minimum=8),
        epochs=_integer(training.get("epochs"), "training.epochs", minimum=1),
        steps_per_epoch=_integer(
            training.get("stepsPerEpoch"), "training.stepsPerEpoch", minimum=1
        ),
        validation_steps=_integer(
            training.get("validationSteps"), "training.validationSteps", minimum=1
        ),
        batch_size=_integer(training.get("batchSize"), "training.batchSize", minimum=1),
        learning_rate=_float(training.get("learningRate"), "training.learningRate"),
        seed=_integer(training.get("seed"), "training.seed", minimum=0),
        deterministic=_boolean(training.get("deterministic", True), "training.deterministic"),
        device=device,
        maximum_sequences_per_partition=maximum,
    )
    raw_root = Path(_string(root.get("datasetRoot"), "datasetRoot")).expanduser()
    dataset_root = (
        raw_root.resolve() if raw_root.is_absolute() else (resolved.parent / raw_root).resolve()
    )
    return ShotRepresentationConfiguration(
        config_path=resolved,
        experiment_name=_string(root.get("experimentName"), "experimentName"),
        model_version=_string(root.get("modelVersion"), "modelVersion"),
        dataset_version=_string(root.get("datasetVersion"), "datasetVersion"),
        dataset_root=dataset_root,
        upstream_revision=_string(root.get("upstreamRevision"), "upstreamRevision"),
        settings=settings,
    )


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise ShotModelTrainingError(f"unable to hash {path}: {error}") from error
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except OSError as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise ShotModelTrainingError(
            f"unable to write experiment artifact {path}: {error}"
        ) from error


def _selected_sequences(
    manifest: RacketVisionManifest,
    partition: str,
    maximum: int | None,
) -> tuple[RacketVisionSequence, ...]:
    selected = tuple(item for item in manifest.sequences if item.partition == partition)
    return selected if maximum is None else selected[:maximum]


def _load_partition_features(
    manifest: RacketVisionManifest,
    *,
    partition: str,
    settings: ShotRepresentationSettings,
) -> tuple[list[NDArray[np.float32]], int]:
    minimum_length = settings.history_frames + settings.future_frames
    arrays: list[NDArray[np.float32]] = []
    window_count = 0
    for sequence in _selected_sequences(
        manifest,
        partition,
        settings.maximum_sequences_per_partition,
    ):
        features = load_racketvision_features(sequence)
        if features.shape[0] < minimum_length:
            continue
        arrays.append(features)
        window_count += features.shape[0] - minimum_length + 1
    if not arrays:
        raise ShotModelTrainingError(f"no usable {partition} RacketVision sequences")
    return arrays, window_count


def _sample_batch(
    arrays: list[NDArray[np.float32]],
    *,
    history_frames: int,
    future_frames: int,
    batch_size: int,
    generator: random.Random,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    histories: list[NDArray[np.float32]] = []
    futures: list[NDArray[np.float32]] = []
    window_size = history_frames + future_frames
    for _ in range(batch_size):
        sequence = arrays[generator.randrange(len(arrays))]
        start = generator.randrange(sequence.shape[0] - window_size + 1)
        histories.append(sequence[start : start + history_frames])
        futures.append(sequence[start + history_frames : start + window_size])
    return np.stack(histories), np.stack(futures)


class TorchTrajectoryPretrainer:
    """Small GRU trajectory/racket future-prediction backend."""

    @property
    def name(self) -> str:
        return "torch_gru_future_prediction"

    @property
    def version(self) -> str:
        return _package_version("torch") or "unknown"

    def train(
        self,
        manifest: RacketVisionManifest,
        *,
        settings: ShotRepresentationSettings,
        weights_path: Path,
    ) -> RepresentationTrainingOutcome:
        try:
            torch = cast(Any, importlib.import_module("torch"))
            nn = cast(Any, importlib.import_module("torch.nn"))
        except ImportError as error:
            raise ShotModelTrainingError(
                "PyTorch is required for representation pretraining"
            ) from error
        effective_device = resolve_inference_device(settings.device)
        random.seed(settings.seed)
        np.random.seed(settings.seed)
        torch.manual_seed(settings.seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(settings.deterministic, warn_only=True)
        train_arrays, train_window_count = _load_partition_features(
            manifest,
            partition="train",
            settings=settings,
        )
        validation_arrays, validation_window_count = _load_partition_features(
            manifest,
            partition="validation",
            settings=settings,
        )
        gru = nn.GRU(FEATURE_COUNT, settings.hidden_size, batch_first=True).to(effective_device)
        head = nn.Linear(
            settings.hidden_size,
            settings.future_frames * FEATURE_COUNT,
        ).to(effective_device)
        optimizer = torch.optim.AdamW(
            [*gru.parameters(), *head.parameters()],
            lr=settings.learning_rate,
        )
        loss_fn = nn.SmoothL1Loss()
        training_losses: list[float] = []
        validation_losses: list[float] = []
        train_generator = random.Random(settings.seed)
        validation_generator = random.Random(settings.seed + 1)
        for _epoch in range(settings.epochs):
            gru.train()
            head.train()
            epoch_loss = 0.0
            for _step in range(settings.steps_per_epoch):
                history_np, future_np = _sample_batch(
                    train_arrays,
                    history_frames=settings.history_frames,
                    future_frames=settings.future_frames,
                    batch_size=settings.batch_size,
                    generator=train_generator,
                )
                history = torch.from_numpy(history_np).to(effective_device)
                future = torch.from_numpy(future_np).to(effective_device)
                optimizer.zero_grad(set_to_none=True)
                encoded, _hidden = gru(history)
                predicted = head(encoded[:, -1, :]).reshape(
                    settings.batch_size,
                    settings.future_frames,
                    FEATURE_COUNT,
                )
                loss = loss_fn(predicted, future)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu().item())
            training_losses.append(epoch_loss / settings.steps_per_epoch)
            gru.eval()
            head.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for _step in range(settings.validation_steps):
                    history_np, future_np = _sample_batch(
                        validation_arrays,
                        history_frames=settings.history_frames,
                        future_frames=settings.future_frames,
                        batch_size=settings.batch_size,
                        generator=validation_generator,
                    )
                    history = torch.from_numpy(history_np).to(effective_device)
                    future = torch.from_numpy(future_np).to(effective_device)
                    encoded, _hidden = gru(history)
                    predicted = head(encoded[:, -1, :]).reshape(
                        settings.batch_size,
                        settings.future_frames,
                        FEATURE_COUNT,
                    )
                    validation_loss += float(loss_fn(predicted, future).cpu().item())
            validation_losses.append(validation_loss / settings.validation_steps)
        try:
            torch.save(
                {
                    "recordType": "racket_temporal_representation_weights",
                    "schemaVersion": 1,
                    "featureCount": FEATURE_COUNT,
                    "historyFrames": settings.history_frames,
                    "futureFrames": settings.future_frames,
                    "hiddenSize": settings.hidden_size,
                    "gruStateDict": gru.state_dict(),
                    "headStateDict": head.state_dict(),
                },
                weights_path,
            )
        except OSError as error:
            raise ShotModelTrainingError(
                f"unable to save representation weights: {error}"
            ) from error
        return RepresentationTrainingOutcome(
            weights_path=weights_path,
            training_losses=tuple(training_losses),
            validation_losses=tuple(validation_losses),
            effective_device=effective_device,
            train_sequence_count=len(train_arrays),
            validation_sequence_count=len(validation_arrays),
            train_window_count=train_window_count,
            validation_window_count=validation_window_count,
        )


def pretrain_shot_representation(
    config_path: Path,
    *,
    output_dir: Path,
    backend: RepresentationPretrainingBackend | None = None,
) -> ShotPretrainingArtifacts:
    """Validate external data, pretrain only representation, and persist provenance."""

    config = load_shot_representation_configuration(config_path)
    manifest = discover_racketvision_sequences(
        config.dataset_root,
        upstream_revision=config.upstream_revision,
    )
    selected_backend = backend or TorchTrajectoryPretrainer()
    config_digest = hashlib.sha256(
        json.dumps(config.as_dict(), sort_keys=True).encode("utf-8")
    ).hexdigest()
    experiment_id = f"{config.experiment_name}-{config_digest[:8]}-{manifest.content_sha256[:8]}"
    resolved_output = output_dir.expanduser().resolve()
    experiment_path = resolved_output / "experiment.json"
    metrics_path = resolved_output / "metrics.json"
    weights_path = resolved_output / "representation.pt"
    if experiment_path.exists() or metrics_path.exists() or weights_path.exists():
        raise ShotModelTrainingError(f"pretraining output already exists in {resolved_output}")
    base = {
        "recordType": "shot_representation_pretraining_experiment",
        "schemaVersion": 1,
        "experimentId": experiment_id,
        "status": "running",
        "configuration": config.as_dict(),
        "configurationSha256": config_digest,
        "externalDataset": manifest.as_dict(),
        "backend": {"name": selected_backend.name, "version": selected_backend.version},
        "environment": {
            "python": _package_version("pickleball-vision"),
            "numpy": _package_version("numpy"),
            "torch": _package_version("torch"),
            "codeRevision": _git_revision(config.config_path.parent),
        },
        "semanticShotClassifierTrained": False,
        "pickleballSemanticLabelsConsumed": False,
        "authoritativeUnknownPolicyChanged": False,
    }
    _write_json(experiment_path, base)
    try:
        outcome = selected_backend.train(
            manifest,
            settings=config.settings,
            weights_path=weights_path,
        )
        if not outcome.weights_path.is_file():
            raise ShotModelTrainingError("pretraining backend produced no weights")
        metrics = {
            "recordType": "shot_representation_pretraining_metrics",
            "schemaVersion": 1,
            "experimentId": experiment_id,
            "objective": "future_ball_and_racket_feature_prediction",
            "trainingLossByEpoch": list(outcome.training_losses),
            "validationLossByEpoch": list(outcome.validation_losses),
            "finalTrainingLoss": outcome.training_losses[-1],
            "finalValidationLoss": outcome.validation_losses[-1],
            "trainSequenceCount": outcome.train_sequence_count,
            "validationSequenceCount": outcome.validation_sequence_count,
            "trainWindowCount": outcome.train_window_count,
            "validationWindowCount": outcome.validation_window_count,
            "effectiveDevice": outcome.effective_device,
            "semanticAccuracy": None,
            "semanticAccuracyUnavailableReason": (
                "RacketVision contains no pickleball shot semantics"
            ),
        }
        _write_json(metrics_path, metrics)
        _write_json(
            experiment_path,
            {
                **base,
                "status": "complete",
                "metricsPath": str(metrics_path),
                "weights": {
                    "path": str(weights_path),
                    "sha256": _sha256(weights_path),
                    "provenance": "RACKETVISION_REPRESENTATION_PRETRAINING_ONLY",
                },
            },
        )
    except Exception as error:
        _write_json(experiment_path, {**base, "status": "failed", "error": str(error)})
        if isinstance(error, ShotModelTrainingError):
            raise
        raise ShotModelTrainingError(str(error)) from error
    return ShotPretrainingArtifacts(
        experiment_path=experiment_path,
        metrics_path=metrics_path,
        weights_path=weights_path,
        experiment_id=experiment_id,
    )
