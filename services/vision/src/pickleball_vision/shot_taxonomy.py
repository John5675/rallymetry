"""Versioned multi-axis pickleball shot semantics and calibrated abstention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar


class ShotPhase(StrEnum):
    """A contact's position in rally order."""

    SERVE = "SERVE"
    RETURN = "RETURN"
    RALLY = "RALLY"
    UNKNOWN = "UNKNOWN"


class ContactMode(StrEnum):
    """How the ball was contacted relative to a bounce and body mechanics."""

    GROUNDSTROKE = "GROUNDSTROKE"
    VOLLEY = "VOLLEY"
    OVERHEAD = "OVERHEAD"
    UNKNOWN = "UNKNOWN"


class StrokeSide(StrEnum):
    """The hitter-facing stroke side."""

    FOREHAND = "FOREHAND"
    BACKHAND = "BACKHAND"
    TWO_HANDED_BACKHAND = "TWO_HANDED_BACKHAND"
    UNKNOWN = "UNKNOWN"


class ShotIntent(StrEnum):
    """Tactical intent or outcome supported by trajectory and court context."""

    DINK = "DINK"
    DROP = "DROP"
    DRIVE = "DRIVE"
    LOB = "LOB"
    RESET = "RESET"
    SPEEDUP = "SPEEDUP"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class LegacyShotType(StrEnum):
    """Backwards-compatible mutually exclusive Milestone 17 vocabulary."""

    SERVE = "SERVE"
    RETURN = "RETURN"
    DINK = "DINK"
    DROP = "DROP"
    DRIVE = "DRIVE"
    VOLLEY = "VOLLEY"
    OVERHEAD = "OVERHEAD"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class ShotLabelProvenance(StrEnum):
    """Origin of semantics without conflating predictions and ground truth."""

    HUMAN_ACCEPTED = "HUMAN_ACCEPTED"
    HUMAN_CORRECTED = "HUMAN_CORRECTED"
    AI_PSEUDO_LABEL = "AI_PSEUDO_LABEL"
    MODEL_PREDICTION = "MODEL_PREDICTION"
    LEGACY_RULE = "LEGACY_RULE"


@dataclass(frozen=True, slots=True)
class AxisAlternative:
    """One inspectable non-authoritative axis alternative."""

    value: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("axis alternative value must not be empty")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("axis alternative confidence must be finite and in [0, 1]")

    def as_dict(self) -> dict[str, object]:
        return {"value": self.value, "confidence": self.confidence}


@dataclass(frozen=True, slots=True)
class AxisDecision:
    """A best guess plus a separately thresholded authoritative axis value."""

    authoritative: str
    best_guess: str
    confidence: float
    alternatives: tuple[AxisAlternative, ...]
    abstained: bool
    threshold: float

    def __post_init__(self) -> None:
        if not self.authoritative or not self.best_guess:
            raise ValueError("axis decision values must not be empty")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("axis decision confidence must be finite and in [0, 1]")
        if not math.isfinite(self.threshold) or not 0.0 < self.threshold <= 1.0:
            raise ValueError("axis decision threshold must be finite and in (0, 1]")
        if self.abstained != (self.authoritative == "UNKNOWN"):
            raise ValueError("axis abstention must match an authoritative UNKNOWN value")

    def as_dict(self) -> dict[str, object]:
        return {
            "authoritative": self.authoritative,
            "bestGuess": self.best_guess,
            "confidence": self.confidence,
            "alternatives": [item.as_dict() for item in self.alternatives],
            "abstained": self.abstained,
            "threshold": self.threshold,
        }


@dataclass(frozen=True, slots=True)
class ShotSemantics:
    """Independent semantic axes with explicit label provenance."""

    phase: ShotPhase
    contact_mode: ContactMode
    stroke_side: StrokeSide
    intent: ShotIntent
    provenance: ShotLabelProvenance
    human_accepted: bool

    def __post_init__(self) -> None:
        if self.human_accepted and self.provenance not in {
            ShotLabelProvenance.HUMAN_ACCEPTED,
            ShotLabelProvenance.HUMAN_CORRECTED,
        }:
            raise ValueError("human acceptance requires human label provenance")

    @property
    def legacy_shot_type(self) -> LegacyShotType:
        """Project the axes into the old vocabulary without inventing evidence."""

        if self.phase is ShotPhase.SERVE:
            return LegacyShotType.SERVE
        if self.phase is ShotPhase.RETURN:
            return LegacyShotType.RETURN
        if self.contact_mode is ContactMode.OVERHEAD:
            return LegacyShotType.OVERHEAD
        if self.intent is ShotIntent.DINK:
            return LegacyShotType.DINK
        if self.intent is ShotIntent.DROP:
            return LegacyShotType.DROP
        if self.intent is ShotIntent.DRIVE:
            return LegacyShotType.DRIVE
        if self.contact_mode is ContactMode.VOLLEY:
            return LegacyShotType.VOLLEY
        if (
            self.phase is ShotPhase.RALLY
            and self.contact_mode is not ContactMode.UNKNOWN
            and self.intent is ShotIntent.OTHER
        ):
            return LegacyShotType.OTHER
        return LegacyShotType.UNKNOWN

    def as_dict(self) -> dict[str, object]:
        return {
            "taxonomyVersion": 1,
            "phase": self.phase.value,
            "contactMode": self.contact_mode.value,
            "strokeSide": self.stroke_side.value,
            "intent": self.intent.value,
            "legacyShotType": self.legacy_shot_type.value,
            "provenance": self.provenance.value,
            "humanAccepted": self.human_accepted,
        }


_EnumT = TypeVar("_EnumT", bound=StrEnum)


def axis_decision_from_probabilities(
    probabilities: dict[_EnumT, float],
    *,
    unknown: _EnumT,
    threshold: float,
    alternative_limit: int = 3,
) -> AxisDecision:
    """Keep a best guess while abstaining authoritatively below a confidence gate."""

    if not probabilities:
        raise ValueError("axis probabilities must not be empty")
    if not 0.0 < threshold <= 1.0 or not math.isfinite(threshold):
        raise ValueError("axis decision threshold must be finite and in (0, 1]")
    if alternative_limit < 0:
        raise ValueError("alternative_limit must be nonnegative")
    candidates: list[tuple[_EnumT, float]] = []
    for label, probability in probabilities.items():
        if label is unknown:
            continue
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError("axis probabilities must be finite and nonnegative")
        candidates.append((label, probability))
    if not candidates:
        raise ValueError("axis probabilities must contain a non-UNKNOWN label")
    total = sum(probability for _, probability in candidates)
    if total <= 0.0:
        raise ValueError("axis probabilities must have positive mass")
    ranked = sorted(
        ((label, probability / total) for label, probability in candidates),
        key=lambda item: (-item[1], item[0].value),
    )
    best_label, confidence = ranked[0]
    abstained = confidence < threshold
    authoritative = unknown.value if abstained else best_label.value
    alternatives = tuple(
        AxisAlternative(label.value, probability)
        for label, probability in ranked[1 : alternative_limit + 1]
    )
    return AxisDecision(
        authoritative=authoritative,
        best_guess=best_label.value,
        confidence=confidence,
        alternatives=alternatives,
        abstained=abstained,
        threshold=threshold,
    )


def semantics_from_legacy(
    shot_type: str,
    *,
    rally_anchored: bool,
    provenance: ShotLabelProvenance,
    human_accepted: bool,
) -> ShotSemantics:
    """Translate one old label without asserting unsupported mechanics or side."""

    try:
        legacy = LegacyShotType(shot_type.upper())
    except ValueError:
        legacy = LegacyShotType.UNKNOWN
    phase = ShotPhase.RALLY if rally_anchored else ShotPhase.UNKNOWN
    contact_mode = ContactMode.UNKNOWN
    intent = ShotIntent.UNKNOWN
    if legacy is LegacyShotType.SERVE:
        phase = ShotPhase.SERVE
    elif legacy is LegacyShotType.RETURN:
        phase = ShotPhase.RETURN
    elif legacy is LegacyShotType.VOLLEY:
        contact_mode = ContactMode.VOLLEY
    elif legacy is LegacyShotType.OVERHEAD:
        contact_mode = ContactMode.OVERHEAD
    elif legacy is LegacyShotType.DINK:
        intent = ShotIntent.DINK
    elif legacy is LegacyShotType.DROP:
        intent = ShotIntent.DROP
    elif legacy is LegacyShotType.DRIVE:
        intent = ShotIntent.DRIVE
    elif legacy is LegacyShotType.OTHER:
        intent = ShotIntent.OTHER
    return ShotSemantics(
        phase=phase,
        contact_mode=contact_mode,
        stroke_side=StrokeSide.UNKNOWN,
        intent=intent,
        provenance=provenance,
        human_accepted=human_accepted,
    )
