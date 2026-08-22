"""Trainable multi-head temporal shot model adapter and abstaining decoder."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pickleball_vision.errors import ShotModelInputError, ShotModelTrainingError
from pickleball_vision.racketvision import FEATURE_COUNT
from pickleball_vision.shot_taxonomy import (
    AxisDecision,
    ContactMode,
    LegacyShotType,
    ShotIntent,
    ShotLabelProvenance,
    ShotPhase,
    ShotSemantics,
    StrokeSide,
    axis_decision_from_probabilities,
)

AXIS_LABELS = {
    "phase": (ShotPhase.SERVE, ShotPhase.RETURN, ShotPhase.RALLY),
    "contactMode": (ContactMode.GROUNDSTROKE, ContactMode.VOLLEY, ContactMode.OVERHEAD),
    "strokeSide": (
        StrokeSide.FOREHAND,
        StrokeSide.BACKHAND,
        StrokeSide.TWO_HANDED_BACKHAND,
    ),
    "intent": (
        ShotIntent.DINK,
        ShotIntent.DROP,
        ShotIntent.DRIVE,
        ShotIntent.LOB,
        ShotIntent.RESET,
        ShotIntent.SPEEDUP,
        ShotIntent.OTHER,
    ),
}
UNKNOWN_BY_AXIS = {
    "phase": ShotPhase.UNKNOWN,
    "contactMode": ContactMode.UNKNOWN,
    "strokeSide": StrokeSide.UNKNOWN,
    "intent": ShotIntent.UNKNOWN,
}
CONTEXT_FEATURE_NAMES = (
    "shotIndexScaled",
    "contactConfidence",
    "hitterKnown",
    "hitterConfidence",
    "hitterCourtXNormalized",
    "hitterCourtYNormalized",
    "hitterCourtPositionAvailable",
    "distanceFromKitchenNormalized",
    "trajectoryKnownFraction",
    "incomingBounceKnown",
    "incomingBounceOccurred",
    "landingCourtXNormalized",
    "landingCourtYNormalized",
    "landingCourtPositionAvailable",
)


@dataclass(frozen=True, slots=True)
class MultiAxisShotPrediction:
    """Decoded model heads with both best-guess and authoritative projections."""

    phase: AxisDecision
    contact_mode: AxisDecision
    stroke_side: AxisDecision
    intent: AxisDecision
    authoritative_legacy_shot_type: LegacyShotType
    best_guess_legacy_shot_type: LegacyShotType

    def as_dict(self) -> dict[str, object]:
        return {
            "taxonomyVersion": 1,
            "phase": self.phase.as_dict(),
            "contactMode": self.contact_mode.as_dict(),
            "strokeSide": self.stroke_side.as_dict(),
            "intent": self.intent.as_dict(),
            "authoritativeLegacyShotType": self.authoritative_legacy_shot_type.value,
            "bestGuessLegacyShotType": self.best_guess_legacy_shot_type.value,
            "provenance": ShotLabelProvenance.MODEL_PREDICTION.value,
            "audioUsedForShotType": False,
        }


def _softmax(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ShotModelInputError("each shot-model axis must contain finite logits")
    offset = max(values)
    exponentials = tuple(math.exp(value - offset) for value in values)
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def _legacy_from_decisions(
    phase: AxisDecision,
    contact_mode: AxisDecision,
    stroke_side: AxisDecision,
    intent: AxisDecision,
    *,
    best_guess: bool,
) -> LegacyShotType:
    attribute = "best_guess" if best_guess else "authoritative"
    semantics = ShotSemantics(
        phase=ShotPhase(getattr(phase, attribute)),
        contact_mode=ContactMode(getattr(contact_mode, attribute)),
        stroke_side=StrokeSide(getattr(stroke_side, attribute)),
        intent=ShotIntent(getattr(intent, attribute)),
        provenance=ShotLabelProvenance.MODEL_PREDICTION,
        human_accepted=False,
    )
    return semantics.legacy_shot_type


def decode_multi_axis_logits(
    logits: Mapping[str, tuple[float, ...]],
    *,
    thresholds: Mapping[str, float],
) -> MultiAxisShotPrediction:
    """Decode independent heads without replacing low-confidence axes with guesses."""

    decisions: dict[str, AxisDecision] = {}
    for axis, labels in AXIS_LABELS.items():
        raw_logits = logits.get(axis)
        if raw_logits is None or len(raw_logits) != len(labels):
            raise ShotModelInputError(
                f"shot-model axis {axis} must contain exactly {len(labels)} logits"
            )
        threshold = thresholds.get(axis)
        if threshold is None:
            raise ShotModelInputError(f"shot-model threshold missing for axis {axis}")
        probabilities = _softmax(raw_logits)
        decisions[axis] = axis_decision_from_probabilities(
            dict(zip(labels, probabilities, strict=True)),
            unknown=UNKNOWN_BY_AXIS[axis],
            threshold=threshold,
        )
    phase = decisions["phase"]
    contact_mode = decisions["contactMode"]
    stroke_side = decisions["strokeSide"]
    intent = decisions["intent"]
    return MultiAxisShotPrediction(
        phase=phase,
        contact_mode=contact_mode,
        stroke_side=stroke_side,
        intent=intent,
        authoritative_legacy_shot_type=_legacy_from_decisions(
            phase,
            contact_mode,
            stroke_side,
            intent,
            best_guess=False,
        ),
        best_guess_legacy_shot_type=_legacy_from_decisions(
            phase,
            contact_mode,
            stroke_side,
            intent,
            best_guess=True,
        ),
    )


class TorchMultiHeadTemporalShotModel:
    """Trainable GRU encoder plus independent semantic heads behind a small adapter."""

    def __init__(
        self,
        *,
        hidden_size: int,
        context_feature_count: int = len(CONTEXT_FEATURE_NAMES),
        device: str = "cpu",
    ) -> None:
        if hidden_size < 8:
            raise ShotModelInputError("shot-model hidden_size must be at least 8")
        if context_feature_count < 1:
            raise ShotModelInputError("shot-model context_feature_count must be positive")
        try:
            self._torch = cast(Any, importlib.import_module("torch"))
            nn = cast(Any, importlib.import_module("torch.nn"))
        except ImportError as error:
            raise ShotModelTrainingError(
                "PyTorch is required for the temporal shot model"
            ) from error
        self.device = device
        self.hidden_size = hidden_size
        self.context_feature_count = context_feature_count
        self.gru = nn.GRU(FEATURE_COUNT, hidden_size, batch_first=True).to(device)
        self.heads = {
            axis: nn.Linear(hidden_size + context_feature_count, len(labels)).to(device)
            for axis, labels in AXIS_LABELS.items()
        }

    def load_representation_weights(self, path: Path, *, freeze_encoder: bool = False) -> None:
        """Load only tensor state through PyTorch's restricted weights-only mode."""

        resolved = path.expanduser().resolve()
        try:
            payload = self._torch.load(resolved, map_location=self.device, weights_only=True)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ShotModelTrainingError(
                f"unable to load representation weights: {error}"
            ) from error
        if not isinstance(payload, dict) or payload.get("recordType") != (
            "racket_temporal_representation_weights"
        ):
            raise ShotModelInputError("representation weights have an unsupported record type")
        if payload.get("featureCount") != FEATURE_COUNT:
            raise ShotModelInputError("representation feature count is incompatible")
        if payload.get("hiddenSize") != self.hidden_size:
            raise ShotModelInputError("representation hidden size is incompatible")
        try:
            self.gru.load_state_dict(payload["gruStateDict"])
        except (KeyError, RuntimeError, TypeError) as error:
            raise ShotModelTrainingError(f"invalid GRU representation state: {error}") from error
        if freeze_encoder:
            for parameter in self.gru.parameters():
                parameter.requires_grad = False

    def parameters(self) -> list[Any]:
        """Expose trainable parameters to a semantic trainer without leaking internals."""

        parameters = [item for item in self.gru.parameters() if item.requires_grad]
        for head in self.heads.values():
            parameters.extend(head.parameters())
        return parameters

    def forward(self, temporal_features: Any, context_features: Any) -> dict[str, Any]:
        """Return independent raw logits; decoding and confidence gates stay separate."""

        if temporal_features.ndim != 3 or temporal_features.shape[2] != FEATURE_COUNT:
            raise ShotModelInputError(
                f"temporal features must have shape [batch, time, {FEATURE_COUNT}]"
            )
        if context_features.ndim != 2 or context_features.shape[1] != (self.context_feature_count):
            raise ShotModelInputError(
                f"context features must have shape [batch, {self.context_feature_count}]"
            )
        if temporal_features.shape[0] != context_features.shape[0]:
            raise ShotModelInputError("temporal and context batch sizes must match")
        encoded, _hidden = self.gru(temporal_features.to(self.device))
        fused = self._torch.cat(
            (encoded[:, -1, :], context_features.to(self.device)),
            dim=1,
        )
        return {axis: head(fused) for axis, head in self.heads.items()}
