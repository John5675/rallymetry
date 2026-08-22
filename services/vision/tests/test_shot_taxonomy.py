import pytest

from pickleball_vision.shot_taxonomy import (
    ContactMode,
    LegacyShotType,
    ShotIntent,
    ShotLabelProvenance,
    ShotPhase,
    StrokeSide,
    axis_decision_from_probabilities,
    semantics_from_legacy,
)


def test_multi_axis_semantics_preserve_overlapping_meaning() -> None:
    semantics = semantics_from_legacy(
        "DRIVE",
        rally_anchored=True,
        provenance=ShotLabelProvenance.HUMAN_ACCEPTED,
        human_accepted=True,
    )

    assert semantics.phase is ShotPhase.RALLY
    assert semantics.intent is ShotIntent.DRIVE
    assert semantics.contact_mode is ContactMode.UNKNOWN
    assert semantics.stroke_side is StrokeSide.UNKNOWN
    assert semantics.legacy_shot_type is LegacyShotType.DRIVE


def test_legacy_serve_does_not_invent_groundstroke_or_forehand() -> None:
    semantics = semantics_from_legacy(
        "SERVE",
        rally_anchored=True,
        provenance=ShotLabelProvenance.AI_PSEUDO_LABEL,
        human_accepted=False,
    )

    assert semantics.phase is ShotPhase.SERVE
    assert semantics.contact_mode is ContactMode.UNKNOWN
    assert semantics.stroke_side is StrokeSide.UNKNOWN
    assert semantics.intent is ShotIntent.UNKNOWN
    assert semantics.legacy_shot_type is LegacyShotType.SERVE


def test_low_confidence_axis_keeps_best_guess_and_authoritative_unknown() -> None:
    decision = axis_decision_from_probabilities(
        {
            ShotIntent.DINK: 0.42,
            ShotIntent.DROP: 0.38,
            ShotIntent.DRIVE: 0.20,
        },
        unknown=ShotIntent.UNKNOWN,
        threshold=0.70,
    )

    assert decision.best_guess == "DINK"
    assert decision.authoritative == "UNKNOWN"
    assert decision.abstained is True
    assert decision.alternatives[0].value == "DROP"


def test_high_confidence_axis_is_authoritative() -> None:
    decision = axis_decision_from_probabilities(
        {ShotPhase.SERVE: 0.8, ShotPhase.RETURN: 0.1, ShotPhase.RALLY: 0.1},
        unknown=ShotPhase.UNKNOWN,
        threshold=0.75,
    )

    assert decision.best_guess == "SERVE"
    assert decision.authoritative == "SERVE"
    assert decision.abstained is False


def test_ai_provenance_cannot_claim_human_acceptance() -> None:
    with pytest.raises(ValueError, match="human acceptance"):
        semantics_from_legacy(
            "DINK",
            rally_anchored=True,
            provenance=ShotLabelProvenance.AI_PSEUDO_LABEL,
            human_accepted=True,
        )
