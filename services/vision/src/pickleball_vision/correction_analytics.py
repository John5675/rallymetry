"""Deterministically apply verified semantic corrections to persisted analytics views."""

from __future__ import annotations

from copy import deepcopy

from pickleball_vision.corrections import SHOT_TYPES, apply_verified_corrections, corrected_payload
from pickleball_vision.persistence.models import Document

_COUNTED_TYPES = ("DINK", "DRIVE", "DROP", "VOLLEY", "OVERHEAD")


def correction_aware_analytics(
    analytics: Document,
    *,
    players: tuple[Document, ...],
    rallies: tuple[Document, ...],
    shots: tuple[Document, ...],
    corrections: tuple[Document, ...],
) -> Document:
    """Recalculate correction-sensitive metrics while retaining predicted metrics."""

    verified = tuple(
        item for item in corrections if item.get("active") is True and item.get("verified") is True
    )
    if not verified:
        return deepcopy(analytics)
    projected = deepcopy(analytics)
    original_metrics = analytics.get("metrics")
    metrics = deepcopy(original_metrics) if isinstance(original_metrics, dict) else {}
    projected_players = tuple(apply_verified_corrections(item, verified) for item in players)
    projected_rallies = tuple(apply_verified_corrections(item, verified) for item in rallies)
    projected_shots = tuple(apply_verified_corrections(item, verified) for item in shots)
    identity_map = _identity_map(projected_players)
    shot_values = [_effective_shot(item, identity_map) for item in projected_shots]
    rally_values = [corrected_payload(item) for item in projected_rallies]
    _update_match_metrics(metrics, rally_values, shot_values)
    _update_player_metrics(metrics, shot_values)
    _update_tactical_metrics(metrics, shot_values)
    projected["predictionMetrics"] = deepcopy(original_metrics)
    projected["metrics"] = metrics
    projected["appliedCorrectionIds"] = [
        str(item["correctionId"]) for item in verified if isinstance(item.get("correctionId"), str)
    ]
    return projected


def _identity_map(players: tuple[Document, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for player in players:
        player_id = player.get("playerId")
        if not isinstance(player_id, str):
            continue
        effective = player.get("effectivePlayer")
        effective_value = effective if isinstance(effective, dict) else {}
        identity = effective_value.get(
            "logicalIdentity",
            effective_value.get("playerId", player.get("logicalIdentity", player_id)),
        )
        if isinstance(identity, str) and identity:
            result[player_id] = identity
        logical = player.get("logicalIdentity")
        if isinstance(logical, str) and logical:
            result[logical] = result.get(player_id, logical)
    return result


def _effective_shot(document: Document, identities: dict[str, str]) -> dict[str, object]:
    payload = corrected_payload(document)
    hitter = payload.get("hitterId")
    shot_type = payload.get("shotType")
    return {
        "recordId": document.get("recordId"),
        "rallyId": payload.get("rallyId"),
        "shotIndex": payload.get("shotIndex"),
        "hitterId": identities.get(hitter, hitter) if isinstance(hitter, str) else None,
        "shotType": (
            shot_type if isinstance(shot_type, str) and shot_type in SHOT_TYPES else "UNKNOWN"
        ),
    }


def _metric_value(container: dict[str, object], key: str, value: object) -> None:
    existing = container.get(key)
    if isinstance(existing, dict):
        existing["value"] = value
        existing["correctionAware"] = True
    else:
        container[key] = {"value": value, "correctionAware": True}


def _update_match_metrics(
    metrics: dict[str, object],
    rallies: list[dict[str, object]],
    shots: list[dict[str, object]],
) -> None:
    match = metrics.get("match")
    if not isinstance(match, dict):
        return
    _metric_value(match, "rallyCount", len(rallies))
    _metric_value(match, "shotCount", len(shots))
    durations: list[float] = []
    for rally in rallies:
        start = rally.get("startTimestamp")
        end = rally.get("endTimestamp")
        start_number = _as_number(start)
        end_number = _as_number(end)
        if start_number is not None and end_number is not None and end_number >= start_number:
            durations.append(end_number - start_number)
    _metric_value(
        match,
        "averageRallyDuration",
        sum(durations) / len(durations) if durations else None,
    )
    _metric_value(match, "averageRallyLength", len(shots) / len(rallies) if rallies else None)


def _update_player_metrics(metrics: dict[str, object], shots: list[dict[str, object]]) -> None:
    player_metrics = metrics.get("players")
    if not isinstance(player_metrics, dict):
        return
    for player_id, raw_metrics in player_metrics.items():
        if not isinstance(player_id, str) or not isinstance(raw_metrics, dict):
            continue
        player_shots = [shot for shot in shots if shot.get("hitterId") == player_id]
        classified = [shot for shot in player_shots if shot.get("shotType") != "UNKNOWN"]
        _metric_value(raw_metrics, "totalHits", len(player_shots))
        raw_metrics["classifiedHitCount"] = len(classified)
        raw_metrics["unknownShotTypeHitCount"] = len(player_shots) - len(classified)
        type_metrics = raw_metrics.get("shotTypes")
        if not isinstance(type_metrics, dict):
            continue
        for shot_type in _COUNTED_TYPES:
            bucket = type_metrics.get(shot_type)
            if not isinstance(bucket, dict):
                bucket = {}
                type_metrics[shot_type] = bucket
            count = sum(shot.get("shotType") == shot_type for shot in classified)
            bucket.update(
                {
                    "count": count,
                    "rate": count / len(classified) if classified else None,
                    "rateDenominatorClassifiedHits": len(classified),
                    "correctionAware": True,
                }
            )


def _update_tactical_metrics(metrics: dict[str, object], shots: list[dict[str, object]]) -> None:
    tactical = metrics.get("tactical")
    if not isinstance(tactical, dict):
        return
    third = [shot for shot in shots if shot.get("shotIndex") == 3]
    classified = [shot for shot in third if shot.get("shotType") != "UNKNOWN"]
    for key, shot_type in (("thirdShotDropRate", "DROP"), ("thirdShotDriveRate", "DRIVE")):
        metric = tactical.get(key)
        count = sum(shot.get("shotType") == shot_type for shot in classified)
        if isinstance(metric, dict):
            metric["value"] = count / len(classified) if classified else None
            metric["numerator"] = count
            metric["denominator"] = len(classified)
            metric["correctionAware"] = True
    quality = tactical.get("thirdShotDataQuality")
    if isinstance(quality, dict):
        quality.update(
            {
                "thirdShotCount": len(third),
                "classifiedThirdShotCount": len(classified),
                "unknownThirdShotCount": len(third) - len(classified),
                "correctionAware": True,
            }
        )


def _as_number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
