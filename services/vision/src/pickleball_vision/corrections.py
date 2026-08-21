"""Human-correction contracts and correction-aware semantic projections."""

from __future__ import annotations

from copy import deepcopy
from enum import StrEnum
from typing import cast

from pickleball_vision.errors import PersistenceValidationError
from pickleball_vision.persistence.models import Document


class CorrectionType(StrEnum):
    PLAYER_IDENTITY = "PLAYER_IDENTITY"
    RALLY_BOUNDARY = "RALLY_BOUNDARY"
    BOUNCE = "BOUNCE"
    HITTER = "HITTER"
    SHOT_TYPE = "SHOT_TYPE"


TARGET_COLLECTION: dict[CorrectionType, str] = {
    CorrectionType.PLAYER_IDENTITY: "players",
    CorrectionType.RALLY_BOUNDARY: "rallies",
    CorrectionType.BOUNCE: "bounces",
    CorrectionType.HITTER: "shots",
    CorrectionType.SHOT_TYPE: "shots",
}

SHOT_TYPES = frozenset(
    {"SERVE", "RETURN", "DINK", "DROP", "DRIVE", "VOLLEY", "OVERHEAD", "OTHER", "UNKNOWN"}
)
_BOUNDARY_FIELDS = frozenset({"startFrame", "endFrame", "startTimestamp", "endTimestamp"})


def validate_human_correction(
    correction_type: CorrectionType,
    value: dict[str, object],
) -> dict[str, object]:
    """Validate the small semantic payload without accepting model/raw artifact data."""

    if not value:
        raise PersistenceValidationError("human correction must not be empty")
    if correction_type is CorrectionType.PLAYER_IDENTITY:
        _require_only(value, {"playerId", "logicalIdentity", "displayName"})
        if not any(_nonempty_text(value.get(key)) for key in value):
            raise PersistenceValidationError(
                "player identity correction requires a non-empty value"
            )
    elif correction_type is CorrectionType.RALLY_BOUNDARY:
        _require_only(value, _BOUNDARY_FIELDS)
        for key, item in value.items():
            if key.endswith("Frame"):
                if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                    raise PersistenceValidationError(f"{key} must be a nonnegative integer")
            elif not isinstance(item, (int, float)) or isinstance(item, bool) or item < 0:
                raise PersistenceValidationError(f"{key} must be nonnegative seconds")
        _validate_order(value, "startFrame", "endFrame")
        _validate_order(value, "startTimestamp", "endTimestamp")
    elif correction_type is CorrectionType.BOUNCE:
        _require_only(value, {"isBounce", "frame", "timestampSeconds", "courtPosition"})
        if not isinstance(value.get("isBounce"), bool):
            raise PersistenceValidationError("bounce correction requires boolean isBounce")
        frame = value.get("frame")
        if frame is not None and (
            not isinstance(frame, int) or isinstance(frame, bool) or frame < 0
        ):
            raise PersistenceValidationError("bounce frame must be a nonnegative integer")
        timestamp = value.get("timestampSeconds")
        if timestamp is not None and (
            not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool) or timestamp < 0
        ):
            raise PersistenceValidationError("bounce timestampSeconds must be nonnegative")
        court_position = value.get("courtPosition")
        if court_position is not None and not isinstance(court_position, dict):
            raise PersistenceValidationError("bounce courtPosition must be an object")
    elif correction_type is CorrectionType.HITTER:
        _require_only(value, {"playerId"})
        if not _nonempty_text(value.get("playerId")):
            raise PersistenceValidationError("hitter correction requires playerId")
    else:
        _require_only(value, {"shotType"})
        shot_type = value.get("shotType")
        if not isinstance(shot_type, str) or shot_type not in SHOT_TYPES:
            raise PersistenceValidationError("shotType is not a supported initial shot class")
    return deepcopy(value)


def prediction_snapshot(
    correction_type: CorrectionType,
    document: Document,
) -> dict[str, object]:
    """Capture the machine value once; later correction updates never regenerate it."""

    payload = document.get("payload")
    domain = payload if isinstance(payload, dict) else {}
    if correction_type is CorrectionType.PLAYER_IDENTITY:
        return {
            key: deepcopy(document[key])
            for key in ("playerId", "logicalIdentity", "displayName")
            if key in document
        }
    if correction_type is CorrectionType.RALLY_BOUNDARY:
        return {key: deepcopy(domain[key]) for key in _BOUNDARY_FIELDS if key in domain}
    if correction_type is CorrectionType.BOUNCE:
        result: dict[str, object] = {"isBounce": True}
        for key in ("frame", "courtPosition"):
            if key in domain:
                result[key] = deepcopy(domain[key])
        timestamp = domain.get("timestampSeconds", document.get("timestampSeconds"))
        if timestamp is not None:
            result["timestampSeconds"] = deepcopy(timestamp)
        return result
    field = "hitterId" if correction_type is CorrectionType.HITTER else "shotType"
    return {field: deepcopy(domain.get(field))}


def prediction_version(document: Document) -> str | None:
    for key in ("modelVersion", "pipelineVersion"):
        value = document.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def apply_verified_corrections(
    document: Document,
    corrections: tuple[Document, ...] | list[Document],
) -> Document:
    """Return a correction-aware view while leaving persisted prediction fields intact."""

    projected = deepcopy(document)
    payload = document.get("payload")
    effective_payload = deepcopy(payload) if isinstance(payload, dict) else {}
    applied: list[Document] = []
    for correction in corrections:
        if not _applies(correction, document):
            continue
        value = correction.get("humanCorrection")
        if not isinstance(value, dict):
            continue
        correction_type = _parse_type(correction.get("correctionType"))
        if correction_type in {CorrectionType.RALLY_BOUNDARY, CorrectionType.BOUNCE}:
            effective_payload.update(deepcopy(value))
        elif correction_type is CorrectionType.HITTER:
            effective_payload["hitterId"] = deepcopy(value.get("playerId"))
        elif correction_type is CorrectionType.SHOT_TYPE:
            effective_payload["shotType"] = deepcopy(value.get("shotType"))
        elif correction_type is CorrectionType.PLAYER_IDENTITY:
            projected["effectivePlayer"] = deepcopy(value)
        applied.append(deepcopy(correction))
    if isinstance(payload, dict):
        projected["effectivePayload"] = effective_payload
    projected["verifiedCorrections"] = applied
    return projected


def _parse_type(value: object) -> CorrectionType | None:
    try:
        return CorrectionType(str(value))
    except ValueError:
        return None


def _applies(correction: Document, document: Document) -> bool:
    return (
        correction.get("active") is True
        and correction.get("verified") is True
        and correction.get("matchId") == document.get("matchId")
        and correction.get("targetRecordId") in {document.get("recordId"), document.get("playerId")}
    )


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require_only(value: dict[str, object], allowed: set[str] | frozenset[str]) -> None:
    unsupported = set(value).difference(allowed)
    if unsupported:
        raise PersistenceValidationError(
            "human correction contains unsupported fields: " + ", ".join(sorted(unsupported))
        )


def _validate_order(value: dict[str, object], start_key: str, end_key: str) -> None:
    start = value.get(start_key)
    end = value.get(end_key)
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end < start:
        raise PersistenceValidationError(f"{end_key} must not precede {start_key}")


def corrected_payload(document: Document) -> dict[str, object]:
    """Read an effective payload produced by :func:`apply_verified_corrections`."""

    value = document.get("effectivePayload", document.get("payload", {}))
    return cast(dict[str, object], value) if isinstance(value, dict) else {}
