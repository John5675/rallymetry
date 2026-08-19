from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pickleball_vision.config import MatchAnalyticsSettings
from pickleball_vision.errors import MatchAnalyticsInputError
from pickleball_vision.match_analytics_workflow import analyze_match
from pickleball_vision.video import inspect_video


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _structured_artifacts(
    synthetic_video: Path,
    tmp_path: Path,
    *,
    invalid_rallies_hash: bool = False,
) -> tuple[Path, Path, Path, Path]:
    metadata = inspect_video(synthetic_video)
    source = metadata.as_dict()
    tracks = tmp_path / "tracks.json"
    tracks.write_text("{}", encoding="utf-8")
    rallies = tmp_path / "rallies.json"
    _write_json(
        rallies,
        {
            "recordType": "automatic_rally_segments",
            "source": source,
            "rallies": [
                {
                    "rallyId": "rally-1",
                    "startTimestamp": 0.0,
                    "endTimestamp": 0.8,
                    "startFrame": 0,
                    "endFrame": 6,
                    "confidence": 0.8,
                }
            ],
        },
    )
    shots = tmp_path / "shots.json"
    _write_json(
        shots,
        {
            "recordType": "reconstructed_classified_shots",
            "source": source,
            "inputs": {
                "rallies": {
                    "path": str(rallies.resolve()),
                    "sha256": "not-the-hash" if invalid_rallies_hash else _hash(rallies),
                },
                "playerTracks": {"path": str(tracks.resolve())},
            },
            "shots": [
                {
                    "shotId": "shot-1",
                    "rallyId": "rally-1",
                    "shotIndex": 1,
                    "hitterId": "ME",
                    "hitterConfidence": 0.7,
                    "shotType": "SERVE",
                    "confidence": 0.75,
                    "hitterCourtPosition": {
                        "courtPoint": {"x_m": 2.0, "y_m": 1.0},
                        "courtRegion": "inside",
                    },
                },
                {
                    "shotId": "shot-2",
                    "rallyId": "rally-1",
                    "shotIndex": 2,
                    "hitterId": "OPPONENT_1",
                    "hitterConfidence": 0.6,
                    "shotType": "RETURN",
                    "confidence": 0.65,
                    "hitterCourtPosition": None,
                },
            ],
        },
    )
    positions = tmp_path / "player_positions.json"
    player_records: dict[str, list[dict[str, object]]] = {}
    for player_index, player_id in enumerate(("ME", "PARTNER", "OPPONENT_1", "OPPONENT_2")):
        player_records[player_id] = [
            {
                "frame_number": frame,
                "timestamp_s": frame / metadata.fps,
                "confidence": 0.9,
                "raw_court_region": "inside",
                "smoothed_court_coordinate": {
                    "x_m": 1.0 + player_index,
                    "y_m": 4.6 if player_index < 2 else 8.8,
                },
            }
            for frame in range(metadata.frame_count)
        ]
    _write_json(
        positions,
        {
            "record_type": "logical_player_court_positions",
            "source": source,
            "court": {
                "unit": "meters",
                "width_m": 6.096,
                "length_m": 13.4112,
                "non_volley_zone_depth_m": 2.1336,
            },
            "inputs": {"persistent_player_tracks": str(tracks.resolve())},
            "configuration": {
                "maximum_step_gap_seconds": 0.2,
                "maximum_step_speed_mps": 8.0,
                "transition_zone_depth_m": 2.1336,
            },
            "player_names": {
                "ME": "John",
                "PARTNER": "Denny",
                "OPPONENT_1": "Oksana",
                "OPPONENT_2": "Diana",
            },
            "players": player_records,
        },
    )
    return rallies, shots, positions, tracks


def test_workflow_writes_complete_analytics_without_mutating_inputs(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    rallies, shots, positions, _ = _structured_artifacts(synthetic_video, tmp_path)
    original_hashes = tuple(_hash(path) for path in (rallies, shots, positions))
    output = tmp_path / "analytics" / "match-analytics.json"

    artifacts = analyze_match(
        synthetic_video,
        rallies_path=rallies,
        shots_path=shots,
        player_positions_path=positions,
        output_path=output,
        settings=MatchAnalyticsSettings(),
    )

    assert artifacts.analytics_path == output.resolve()
    assert artifacts.rally_count == 1
    assert artifacts.shot_count == 2
    assert original_hashes == tuple(_hash(path) for path in (rallies, shots, positions))
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["recordType"] == "deterministic_match_analytics"
    assert payload["contracts"]["structuredDomainObjectsOnly"] is True
    assert payload["contracts"]["rawYoloOutputsAccessed"] is False
    assert payload["players"]["ME"]["displayName"] == "John"
    assert payload["match"]["shotCount"]["value"] == 2
    assert payload["dataQuality"]["unknownHitterShotCount"] == 0


def test_workflow_rejects_changed_rally_provenance(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    rallies, shots, positions, _ = _structured_artifacts(
        synthetic_video,
        tmp_path,
        invalid_rallies_hash=True,
    )

    with pytest.raises(MatchAnalyticsInputError, match="changed after shots"):
        analyze_match(
            synthetic_video,
            rallies_path=rallies,
            shots_path=shots,
            player_positions_path=positions,
            output_path=tmp_path / "match-analytics.json",
            settings=MatchAnalyticsSettings(),
        )
