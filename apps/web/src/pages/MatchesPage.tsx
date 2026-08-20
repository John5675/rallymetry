import { ArrowUpRight, CalendarDays, ImageOff, Users } from "lucide-react";
import { useCallback } from "react";
import { Link } from "react-router";

import type { Artifact, Match, Player } from "../api/types";
import { useApi } from "../api/context";
import { AsyncState } from "../components/AsyncState";
import { StatusBadge } from "../components/StatusBadge";
import { findThumbnail, matchStatus, playerName } from "../domain";
import { useAsyncData } from "../hooks/useAsyncData";

interface MatchListItem {
  match: Match;
  players: Player[];
  thumbnail: Artifact | null;
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Date unavailable";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(date);
}

export function MatchesPage() {
  const api = useApi();
  const loadMatches = useCallback(
    async (signal: AbortSignal): Promise<MatchListItem[]> => {
      const matches = await api.getMatches(signal);
      return Promise.all(
        matches.map(async (match) => {
          const [players, artifacts] = await Promise.all([
            api.getPlayers(match.matchId, signal).catch(() => []),
            api.getArtifacts(match.matchId, signal).catch(() => []),
          ]);
          return { match, players, thumbnail: findThumbnail(artifacts) };
        }),
      );
    },
    [api],
  );
  const { data, loading, error } = useAsyncData(loadMatches);

  return (
    <div className="page page--matches">
      <section className="hero">
        <div>
          <span className="eyebrow">Match library</span>
          <h1>See the game between the points.</h1>
          <p>
            Review processed matches, trace structured events, and inspect player and shot
            analytics from one focused workspace.
          </p>
        </div>
        <div className="hero-stat">
          <strong>{data?.length ?? "—"}</strong>
          <span>matches available</span>
        </div>
      </section>

      <section aria-labelledby="match-list-title">
        <div className="list-heading">
          <div>
            <h2 id="match-list-title">Your matches</h2>
            <p>Most recently updated recordings appear first.</p>
          </div>
        </div>
        <AsyncState
          loading={loading}
          error={error}
          empty={data?.length === 0}
          emptyTitle="No matches yet"
          emptyMessage="Create a match through the FastAPI service, then queue it for analysis."
        >
          <div className="match-grid">
            {data?.map(({ match, players, thumbnail }) => (
              <Link key={match.matchId} to={`/matches/${match.matchId}`} className="match-card">
                <div className="match-card-media">
                  {thumbnail === null || thumbnail.url === null ? (
                    <div className="match-card-placeholder">
                      <ImageOff aria-hidden="true" />
                      <span>Awaiting thumbnail</span>
                    </div>
                  ) : (
                    <img
                      src={thumbnail.url}
                      alt={`${match.title ?? "Untitled match"} thumbnail`}
                      loading="lazy"
                    />
                  )}
                  <StatusBadge status={matchStatus(match)} />
                </div>
                <div className="match-card-body">
                  <span className="match-date">
                    <CalendarDays aria-hidden="true" /> {formatDate(match.updatedAt)}
                  </span>
                  <h3>{match.title ?? "Untitled match"}</h3>
                  <p className="match-players">
                    <Users aria-hidden="true" />
                    {players.length > 0
                      ? players.map(playerName).join(" · ")
                      : "Players pending"}
                  </p>
                  <span className="match-card-link">
                    Open match <ArrowUpRight aria-hidden="true" />
                  </span>
                </div>
              </Link>
            ))}
          </div>
        </AsyncState>
      </section>
    </div>
  );
}
