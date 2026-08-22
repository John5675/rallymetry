import pytest
import torch

from pickleball_vision.errors import ShotModelInputError
from pickleball_vision.racketvision import FEATURE_COUNT
from pickleball_vision.shot_model import (
    CONTEXT_FEATURE_NAMES,
    TorchMultiHeadTemporalShotModel,
    decode_multi_axis_logits,
)


def test_decoder_keeps_best_guess_when_authoritative_axis_abstains() -> None:
    prediction = decode_multi_axis_logits(
        {
            "phase": (0.0, -1.0, 2.0),
            "contactMode": (0.2, 0.1, 0.0),
            "strokeSide": (0.3, 0.2, 0.1),
            "intent": (1.0, 0.9, 0.8, 0.0, -0.1, -0.2, -0.3),
        },
        thresholds={
            "phase": 0.70,
            "contactMode": 0.80,
            "strokeSide": 0.80,
            "intent": 0.80,
        },
    )

    assert prediction.phase.authoritative == "RALLY"
    assert prediction.intent.authoritative == "UNKNOWN"
    assert prediction.intent.best_guess == "DINK"
    assert prediction.authoritative_legacy_shot_type.value == "UNKNOWN"
    assert prediction.best_guess_legacy_shot_type.value == "DINK"
    assert prediction.as_dict()["audioUsedForShotType"] is False


def test_decoder_rejects_missing_axis_logits() -> None:
    with pytest.raises(ShotModelInputError, match="contactMode"):
        decode_multi_axis_logits(
            {"phase": (1.0, 0.0, 0.0)},
            thresholds={"phase": 0.5},
        )


def test_torch_adapter_has_trainable_independent_heads() -> None:
    model = TorchMultiHeadTemporalShotModel(hidden_size=16)
    temporal = torch.zeros((2, 20, FEATURE_COUNT), dtype=torch.float32)
    context = torch.zeros((2, len(CONTEXT_FEATURE_NAMES)), dtype=torch.float32)

    logits = model.forward(temporal, context)

    assert logits["phase"].shape == (2, 3)
    assert logits["contactMode"].shape == (2, 3)
    assert logits["strokeSide"].shape == (2, 3)
    assert logits["intent"].shape == (2, 7)
    assert model.parameters()
