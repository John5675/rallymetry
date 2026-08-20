import { ArrowLeft, Footprints, MapPinned, ShieldCheck, Target, TrendingUp } from "lucide-react";
import { useCallback, useMemo } from "react";
import { Link, useParams } from "react-router";

import { useApi } from "../api/context";
import { AsyncState } from "../components/AsyncState";
import { CourtMap } from "../components/CourtMap";
import { MetricCard } from "../components/MetricCard";
import { SectionHeading } from "../components/SectionHeading";
import {
  analyticsNumber,
  formatPercent,
  metricNumber,
  nestedValue,
  playerName,
  shotCourtPoints,
} from "../domain";
import { useAsyncData } from "../hooks/useAsyncData";

const SHOT_TYPES = ["DINK", "DRIVE", "DROP", "VOLLEY", "OVERHEAD"] as const;

export function AnalysisPage() {
  const { matchId } = useParams();
  const api = useApi();
  const loadAnalysis = useCallback(
    (signal: AbortSignal) => {
      if (matchId === undefined) return Promise.reject(new Error("A match ID is required."));
      return api.getMatchDashboard(matchId, signal);
    },
    [api, matchId],
  );
  const { data, loading, error } = useAsyncData(loadAnalysis);
  const courtPoints = useMemo(() => shotCourtPoints(data?.shots ?? []), [data]);

  return (
    <div className="page page--analysis">
      <Link to={matchId === undefined ? "/matches" : `/matches/${matchId}`} className="back-link">
        <ArrowLeft aria-hidden="true" /> Back to match
      </Link>
      <AsyncState loading={loading} error={error}>
        {data === null ? null : (
          <>
            <header className="analysis-header">
              <span className="eyebrow">Deterministic analytics</span>
              <h1>{data.match.title ?? "Untitled match"}</h1>
              <p>
                This view presents pipeline-calculated statistics and their available confidence context.
                The browser does not recompute match analytics.
              </p>
            </header>

            {data.analytics === null ? (
              <div className="state-panel">
                <Target aria-hidden="true" />
                <div>
                  <strong>Analytics are not available yet</strong>
                  <p>The match can still be reviewed, but no statistics will be inferred in the browser.</p>
                </div>
              </div>
            ) : (
              <>
                <section className="analysis-band">
                  <MetricCard
                    label="Third-shot drop rate"
                    value={formatPercent(metricNumber(data.analytics, ["tactical", "thirdShotDropRate"]))}
                    detail="Classified third shots"
                  />
                  <MetricCard
                    label="Third-shot drive rate"
                    value={formatPercent(metricNumber(data.analytics, ["tactical", "thirdShotDriveRate"]))}
                    detail="Classified third shots"
                  />
                  <MetricCard
                    label="Near-team kitchen arrival"
                    value={formatPercent(metricNumber(data.analytics, ["teams", "nearTeam", "kitchenArrivalRate"]))}
                    detail="Evaluable rallies"
                  />
                  <MetricCard
                    label="Far-team kitchen arrival"
                    value={formatPercent(metricNumber(data.analytics, ["teams", "farTeam", "kitchenArrivalRate"]))}
                    detail="Evaluable rallies"
                  />
                </section>

                <section className="content-section">
                  <SectionHeading
                    eyebrow="Player comparison"
                    title="Four-player analysis"
                    description="Position and shot-selection metrics preserve unknown and excluded samples."
                  />
                  <div className="data-table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Player</th>
                          <th>Hits</th>
                          <th>Distance</th>
                          <th>Kitchen</th>
                          <th>Transition</th>
                          <th>Backcourt</th>
                          <th>Spacing</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.players.map((player) => {
                          const identity = player.logicalIdentity ?? player.playerId;
                          return (
                            <tr key={player.playerId}>
                              <td>
                                <strong>{playerName(player)}</strong>
                                <small>{identity.replaceAll("_", " ")}</small>
                              </td>
                              <td>{metricNumber(data.analytics, ["players", identity, "totalHits"]) ?? "—"}</td>
                              <td>
                                {metricNumber(data.analytics, ["players", identity, "positions", "distanceTraveled"])?.toFixed(1) ?? "—"} m
                              </td>
                              <td>{formatPercent(analyticsNumber(data.analytics, ["players", identity, "positions", "courtOccupancy", "kitchen", "shareOfInCourtFrames"]))}</td>
                              <td>{formatPercent(analyticsNumber(data.analytics, ["players", identity, "positions", "courtOccupancy", "transitionZone", "shareOfInCourtFrames"]))}</td>
                              <td>{formatPercent(analyticsNumber(data.analytics, ["players", identity, "positions", "courtOccupancy", "backcourt", "shareOfInCourtFrames"]))}</td>
                              <td>
                                {metricNumber(data.analytics, ["players", identity, "positions", "averagePartnerSpacing"])?.toFixed(1) ?? "—"} m
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </section>

                <section className="content-section analysis-split">
                  <div>
                    <SectionHeading
                      eyebrow="Shot selection"
                      title="Distribution by player"
                      description="Rates use classified hits only."
                    />
                    <div className="analysis-player-list">
                      {data.players.map((player) => {
                        const identity = player.logicalIdentity ?? player.playerId;
                        return (
                          <article key={player.playerId} className="analysis-player-row">
                            <header>
                              <strong>{playerName(player)}</strong>
                              <span>{metricNumber(data.analytics, ["players", identity, "totalHits"]) ?? "—"} hits</span>
                            </header>
                            <div>
                              {SHOT_TYPES.map((type) => (
                                <span key={type}>
                                  <small>{type.toLowerCase()}</small>
                                  <strong>{formatPercent(analyticsNumber(data.analytics, ["players", identity, "shotTypes", type, "rate"]))}</strong>
                                </span>
                              ))}
                            </div>
                          </article>
                        );
                      })}
                    </div>
                  </div>
                  <div>
                    <SectionHeading
                      eyebrow="Quality"
                      title="Analysis provenance"
                      description="Version and input context retained by the API."
                    />
                    <div className="quality-card">
                      <ShieldCheck aria-hidden="true" />
                      <dl>
                        <div><dt>Calculation</dt><dd>{data.analytics.calculationVersion}</dd></div>
                        <div><dt>Pipeline</dt><dd>{data.analytics.pipelineVersion ?? "Not recorded"}</dd></div>
                        <div><dt>Input artifacts</dt><dd>{data.analytics.inputArtifactIds.length}</dd></div>
                        <div>
                          <dt>Quality record</dt>
                          <dd>{nestedValue(data.analytics.metrics, ["dataQuality"]) === undefined ? "Unavailable" : "Available"}</dd>
                        </div>
                      </dl>
                    </div>
                  </div>
                </section>

                <section className="content-section">
                  <SectionHeading
                    eyebrow="Court view"
                    title="Shot landing map"
                    description="Landing points are accepted court-plane observations from structured shots."
                  />
                  <div className="analysis-map-layout">
                    <CourtMap points={courtPoints} />
                    <div className="map-principles">
                      <div><MapPinned aria-hidden="true" /><span><strong>Known landings only</strong>Missing coordinates remain missing.</span></div>
                      <div><TrendingUp aria-hidden="true" /><span><strong>Pipeline confidence</strong>Each point retains its source confidence.</span></div>
                      <div><Footprints aria-hidden="true" /><span><strong>Canonical court</strong>6.10 × 13.41 meters, near baseline at y = 0.</span></div>
                    </div>
                  </div>
                </section>
              </>
            )}
          </>
        )}
      </AsyncState>
    </div>
  );
}
