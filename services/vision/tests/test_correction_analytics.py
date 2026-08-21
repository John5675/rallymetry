from __future__ import annotations

from datetime import UTC, datetime

from pickleball_vision.correction_analytics import correction_aware_analytics
from pickleball_vision.persistence.models import Document

NOW = datetime(2026, 8, 21, tzinfo=UTC)


def test_verified_corrections_drive_semantic_analytics_without_destroying_metrics() -> None:
    analytics: Document = {
        "_id": "analytics-1",
        "analyticsId": "analytics-1",
        "matchId": "match-1",
        "calculationVersion": "v1",
        "metrics": {
            "match": {
                "rallyCount": {"value": 1},
                "shotCount": {"value": 1},
                "averageRallyDuration": {"value": 9.0},
                "averageRallyLength": {"value": 1.0},
            },
            "players": {
                "ME": {
                    "totalHits": {"value": 1},
                    "classifiedHitCount": 0,
                    "unknownShotTypeHitCount": 1,
                    "shotTypes": {"DRIVE": {"count": 0, "rate": None}},
                }
            },
            "tactical": {},
        },
        "createdAt": NOW,
    }
    player: Document = {
        "matchId": "match-1",
        "playerId": "JOHN",
        "logicalIdentity": "ME",
    }
    rally: Document = {
        "matchId": "match-1",
        "recordId": "rally-1",
        "payload": {"startTimestamp": 1.0, "endTimestamp": 10.0},
    }
    shot: Document = {
        "matchId": "match-1",
        "recordId": "shot-1",
        "payload": {
            "rallyId": "rally-1",
            "shotIndex": 1,
            "hitterId": "JOHN",
            "shotType": "UNKNOWN",
        },
    }
    correction: Document = {
        "correctionId": "correction-1",
        "matchId": "match-1",
        "correctionType": "SHOT_TYPE",
        "targetRecordId": "shot-1",
        "humanCorrection": {"shotType": "DRIVE"},
        "active": True,
        "verified": True,
    }

    result = correction_aware_analytics(
        analytics,
        players=(player,),
        rallies=(rally,),
        shots=(shot,),
        corrections=(correction,),
    )

    assert result["predictionMetrics"] == analytics["metrics"]
    original = analytics["metrics"]
    assert isinstance(original, dict)
    original_players = original["players"]
    assert isinstance(original_players, dict)
    original_me = original_players["ME"]
    assert isinstance(original_me, dict)
    assert original_me["classifiedHitCount"] == 0
    effective = result["metrics"]
    assert isinstance(effective, dict)
    players = effective["players"]
    assert isinstance(players, dict)
    me = players["ME"]
    assert isinstance(me, dict)
    shot_types = me["shotTypes"]
    assert isinstance(shot_types, dict)
    drive = shot_types["DRIVE"]
    assert isinstance(drive, dict)
    assert me["classifiedHitCount"] == 1
    assert drive["count"] == 1
    assert result["appliedCorrectionIds"] == ["correction-1"]
