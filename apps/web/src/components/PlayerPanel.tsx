import { Footprints, Target, UsersRound } from "lucide-react";

import type { Analytics, Artifact, Player } from "../api/types";
import {
  analyticsNumber,
  findPublicArtifact,
  formatPercent,
  metricNumber,
  playerName,
} from "../domain";

interface PlayerPanelProps {
  player: Player;
  analytics: Analytics | null;
  artifacts: Artifact[];
}

const SHOT_TYPES = ["DINK", "DRIVE", "DROP", "VOLLEY", "OVERHEAD"] as const;

export function PlayerPanel({ player, analytics, artifacts }: PlayerPanelProps) {
  const effectiveIdentity = player.effectivePlayer?.logicalIdentity;
  const identity =
    typeof effectiveIdentity === "string"
      ? effectiveIdentity
      : player.logicalIdentity ?? player.playerId;
  const totalHits = metricNumber(analytics, ["players", identity, "totalHits"]);
  const distance = metricNumber(analytics, [
    "players",
    identity,
    "positions",
    "distanceTraveled",
  ]);
  const spacing = metricNumber(analytics, [
    "players",
    identity,
    "positions",
    "averagePartnerSpacing",
  ]);
  const heatmap = findPublicArtifact(
    artifacts,
    (artifact) =>
      artifact.contentType.startsWith("image/") &&
      artifact.artifactType.toLowerCase().includes("heatmap") &&
      (artifact.artifactType.toLowerCase().includes(identity.toLowerCase()) ||
        artifact.pathname.toLowerCase().includes(identity.toLowerCase())),
  );

  return (
    <article className="player-panel">
      <header>
        <div className="player-avatar">{playerName(player).slice(0, 1).toUpperCase()}</div>
        <div>
          <span>{player.logicalIdentity?.replaceAll("_", " ") ?? "Player"}</span>
          <h3>{playerName(player)}</h3>
          <p>{player.team ?? "Team not assigned"}</p>
          {(player.verifiedCorrections?.length ?? 0) > 0 ? (
            <small className="human-correction-note">
              Human corrected · AI: {player.displayName ?? player.logicalIdentity ?? player.playerId}
            </small>
          ) : null}
        </div>
      </header>
      <div className="player-mini-metrics">
        <div>
          <Target aria-hidden="true" />
          <span>Hits</span>
          <strong>{totalHits ?? "N/A"}</strong>
        </div>
        <div>
          <Footprints aria-hidden="true" />
          <span>Distance</span>
          <strong>{distance === null ? "N/A" : `${distance.toFixed(0)}m`}</strong>
        </div>
        <div>
          <UsersRound aria-hidden="true" />
          <span>Avg. spacing</span>
          <strong>{spacing === null ? "N/A" : `${spacing.toFixed(1)}m`}</strong>
        </div>
      </div>
      <div className="shot-distribution">
        <strong>Shot distribution</strong>
        {SHOT_TYPES.map((shotType) => {
          const rate = analyticsNumber(analytics, [
            "players",
            identity,
            "shotTypes",
            shotType,
            "rate",
          ]);
          return (
            <div key={shotType} className="distribution-row">
              <span>{shotType.toLowerCase()}</span>
              <div className="distribution-track">
                <span style={{ width: `${Math.max(0, Math.min(100, (rate ?? 0) * 100))}%` }} />
              </div>
              <strong>{formatPercent(rate)}</strong>
            </div>
          );
        })}
      </div>
      {heatmap === null || heatmap.url === null ? null : (
        <img className="player-heatmap" src={heatmap.url} alt={`${playerName(player)} heatmap`} />
      )}
    </article>
  );
}
