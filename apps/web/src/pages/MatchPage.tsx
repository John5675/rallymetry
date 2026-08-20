import {
  ArrowLeft,
  BarChart3,
  Clock3,
  Film,
  Map,
  Target,
  Trophy,
  UsersRound,
} from "lucide-react";
import { useCallback, useMemo, useRef } from "react";
import { Link, useParams } from "react-router";

import { useApi } from "../api/context";
import { AsyncState } from "../components/AsyncState";
import { CourtMap } from "../components/CourtMap";
import { EventTimeline } from "../components/EventTimeline";
import { MetricCard } from "../components/MetricCard";
import { PlayerPanel } from "../components/PlayerPanel";
import { RallyTable } from "../components/RallyTable";
import { SectionHeading } from "../components/SectionHeading";
import { ShotTable } from "../components/ShotTable";
import { StatusBadge } from "../components/StatusBadge";
import { VideoPanel } from "../components/VideoPanel";
import type { SeekableMediaHandle } from "../components/VideoPanel";
import {
  buildTimelineEvents,
  findPrimaryVideo,
  findPublicArtifact,
  formatDuration,
  matchStatus,
  metricNumber,
  numberValue,
  playerName,
  shotCourtPoints,
} from "../domain";
import { useAsyncData } from "../hooks/useAsyncData";

export function MatchPage() {
  const { matchId } = useParams();
  const api = useApi();
  const mediaRef = useRef<SeekableMediaHandle>(null);
  const loadMatch = useCallback(
    (signal: AbortSignal) => {
      if (matchId === undefined) return Promise.reject(new Error("A match ID is required."));
      return api.getMatchDashboard(matchId, signal);
    },
    [api, matchId],
  );
  const { data, loading, error } = useAsyncData(loadMatch);
  const events = useMemo(
    () =>
      data === null
        ? []
        : buildTimelineEvents(data.rallies, data.contacts, data.bounces, data.shots),
    [data],
  );
  const courtPoints = useMemo(() => shotCourtPoints(data?.shots ?? []), [data]);

  return (
    <div className="page page--detail">
      <Link to="/matches" className="back-link">
        <ArrowLeft aria-hidden="true" /> All matches
      </Link>
      <AsyncState loading={loading} error={error}>
        {data === null ? null : (
          <>
            <header className="match-header">
              <div>
                <span className="eyebrow">Match review</span>
                <h1>{data.match.title ?? "Untitled match"}</h1>
                <p>
                  {data.players.length > 0
                    ? data.players.map(playerName).join(" · ")
                    : "Player assignments pending"}
                </p>
              </div>
              <div className="match-header-actions">
                <StatusBadge status={matchStatus(data.match)} />
                <Link to={`/matches/${data.match.matchId}/analysis`} className="button button--primary">
                  <BarChart3 aria-hidden="true" /> Full analysis
                </Link>
              </div>
            </header>

            <nav className="section-nav" aria-label="Match sections">
              <a href="#overview"><Trophy aria-hidden="true" /> Overview</a>
              <a href="#video"><Film aria-hidden="true" /> Video</a>
              <a href="#players"><UsersRound aria-hidden="true" /> Players</a>
              <a href="#rallies"><Clock3 aria-hidden="true" /> Rallies</a>
              <a href="#shots"><Target aria-hidden="true" /> Shots</a>
              <a href="#court-maps"><Map aria-hidden="true" /> Court maps</a>
            </nav>

            <section id="overview" className="content-section">
              <SectionHeading
                eyebrow="Overview"
                title="Match at a glance"
                description="Deterministic metrics produced by the analysis pipeline. Missing values stay unknown."
              />
              <div className="metric-grid">
                <MetricCard
                  label="Rallies"
                  value={metricNumber(data.analytics, ["match", "rallyCount"]) ?? "—"}
                  detail="Structured rally records"
                />
                <MetricCard
                  label="Shots"
                  value={metricNumber(data.analytics, ["match", "shotCount"]) ?? "—"}
                  detail="Reconstructed shots"
                />
                <MetricCard
                  label="Avg. rally length"
                  value={
                    metricNumber(data.analytics, ["match", "averageRallyLength"])?.toFixed(1) ??
                    "—"
                  }
                  detail="Shots per rally"
                />
                <MetricCard
                  label="Avg. duration"
                  value={formatDuration(
                    metricNumber(data.analytics, ["match", "averageRallyDuration"]),
                  )}
                  detail="Across evaluable rallies"
                />
              </div>
            </section>

            <section id="video" className="content-section">
              <SectionHeading
                eyebrow="Video"
                title="Review the match in context"
                description="Use the structured timeline to jump to visible match events."
              />
              <VideoPanel
                ref={mediaRef}
                youtubeVideoId={data.match.youtubeVideoId}
                videoArtifact={findPrimaryVideo(data.artifacts)}
                hasPrivateVideo={data.artifacts.some(
                  (artifact) => artifact.contentType.startsWith("video/") && artifact.access === "PRIVATE",
                )}
              />
              <EventTimeline
                events={events}
                durationSeconds={numberValue(data.match.summary, "durationSeconds")}
                onSeek={(timestamp) => mediaRef.current?.seek(timestamp)}
              />
            </section>

            <section id="players" className="content-section">
              <SectionHeading
                eyebrow="Players"
                title="Movement and shot profiles"
                description="Logical identities remain separate from detector and tracker IDs."
              />
              <div className="player-grid">
                {data.players.map((player) => (
                  <PlayerPanel
                    key={player.playerId}
                    player={player}
                    analytics={data.analytics}
                    artifacts={data.artifacts}
                  />
                ))}
              </div>
              {data.players.length === 0 ? <p className="inline-empty">No players persisted yet.</p> : null}
            </section>

            <section id="rallies" className="content-section">
              <SectionHeading
                eyebrow="Rallies"
                title="Rally-by-rally review"
                description="Filter predictions by confidence and sort without changing the source analysis."
              />
              <RallyTable rallies={data.rallies} />
            </section>

            <section id="shots" className="content-section">
              <SectionHeading
                eyebrow="Shots"
                title="Reconstructed shot sequence"
                description="Unknown hitters, classes, and landing locations remain explicitly unavailable."
              />
              <ShotTable shots={data.shots} players={data.players} />
            </section>

            <section id="court-maps" className="content-section">
              <SectionHeading
                eyebrow="Court maps"
                title="Where the ball landed"
                description="Only structured, court-plane landing coordinates are shown."
              />
              <div className="court-layout">
                <CourtMap points={courtPoints} />
                <div className="artifact-gallery">
                  {data.artifacts
                    .filter(
                      (artifact) =>
                        artifact.access === "PUBLIC" &&
                        artifact.url !== null &&
                        (artifact.artifactType.toLowerCase().includes("topdown") ||
                          artifact.artifactType.toLowerCase().includes("heatmap")),
                    )
                    .map((artifact) =>
                      artifact.contentType.startsWith("video/") ? (
                        <video key={artifact.artifactId} controls preload="metadata" src={artifact.url ?? undefined} />
                      ) : (
                        <img key={artifact.artifactId} src={artifact.url ?? undefined} alt={artifact.artifactType.replaceAll("_", " ")} />
                      ),
                    )}
                  {findPublicArtifact(
                    data.artifacts,
                    (artifact) =>
                      artifact.artifactType.toLowerCase().includes("topdown") ||
                      artifact.artifactType.toLowerCase().includes("heatmap"),
                  ) === null ? (
                    <div className="artifact-empty">
                      <Map aria-hidden="true" />
                      <strong>No public court artifacts</strong>
                      <p>Top-down video and heatmaps appear here when published for browser viewing.</p>
                    </div>
                  ) : null}
                </div>
              </div>
            </section>
          </>
        )}
      </AsyncState>
    </div>
  );
}
