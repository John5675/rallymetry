"""Ultralytics implementation of the custom pickleball training boundary."""

from __future__ import annotations

import importlib
import sys
from contextlib import redirect_stdout
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol, cast

from pickleball_vision.ball_config import BallExperimentConfiguration
from pickleball_vision.ball_dataset import PreparedBallDataset
from pickleball_vision.ball_training import BallTrainingOutcome
from pickleball_vision.detectors.ultralytics_person import resolve_inference_device
from pickleball_vision.errors import BallTrainingError


class _TrainingResultLike(Protocol):
    @property
    def save_dir(self) -> str | Path: ...

    @property
    def results_dict(self) -> dict[str, object]: ...


class _TrainingModelLike(Protocol):
    def train(self, **kwargs: object) -> _TrainingResultLike: ...


def _framework_version() -> str | None:
    try:
        return version("ultralytics")
    except PackageNotFoundError:
        return None


def _load_model(base_model: str) -> _TrainingModelLike:
    try:
        with redirect_stdout(sys.stderr):
            module = cast(Any, importlib.import_module("ultralytics"))
            return cast(_TrainingModelLike, module.YOLO(base_model))
    except Exception as error:
        raise BallTrainingError(f"unable to load base model {base_model!r}: {error}") from error


class UltralyticsBallTrainer:
    """Train a single-class model while keeping framework details isolated."""

    def __init__(
        self,
        *,
        model: _TrainingModelLike | None = None,
        effective_device: str | None = None,
        framework_version: str | None = None,
    ) -> None:
        self._model = model
        self._effective_device = effective_device
        self._framework_version = framework_version

    def train(
        self,
        *,
        config: BallExperimentConfiguration,
        prepared_dataset: PreparedBallDataset,
        output_dir: Path,
    ) -> BallTrainingOutcome:
        model = self._model or _load_model(config.model.base_model)
        device = self._effective_device or resolve_inference_device(config.training.device)
        try:
            result = model.train(
                data=str(prepared_dataset.dataset_yaml_path),
                epochs=config.training.epochs,
                imgsz=config.training.image_size_px,
                batch=config.training.batch_size,
                device=device,
                workers=config.training.workers,
                seed=config.training.seed,
                deterministic=config.training.deterministic,
                patience=config.training.patience_epochs,
                # The prepared dataset already contains exactly one class. Ultralytics'
                # single_cls mode rewrites that class name to the generic "item", which
                # breaks the persisted custom-model contract and inference validation.
                single_cls=False,
                project=str(output_dir),
                name="ultralytics",
                exist_ok=False,
                save=True,
                val=True,
                plots=True,
                verbose=True,
            )
            save_dir = Path(result.save_dir).expanduser().resolve()
            metrics = {
                str(key): float(value)
                for key, value in result.results_dict.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            best = save_dir / "weights" / "best.pt"
            last = save_dir / "weights" / "last.pt"
            return BallTrainingOutcome(
                backend="ultralytics_yolo",
                framework_version=(
                    self._framework_version
                    if self._framework_version is not None
                    else _framework_version()
                ),
                effective_device=device,
                best_weights_path=best,
                last_weights_path=last if last.is_file() else None,
                metrics=metrics,
            )
        except BallTrainingError:
            raise
        except Exception as error:
            raise BallTrainingError(str(error)) from error
