import {
  ArrowUpRight,
  CalendarDays,
  CheckCircle2,
  ImageOff,
  Link2,
  LoaderCircle,
  Upload,
  Users,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import type { FormEvent } from "react";
import { Link } from "react-router";

import { ApiClientError } from "../api/client";
import type {
  Artifact,
  Match,
  Player,
  ProcessingJob,
  YouTubeMatchSubmission,
} from "../api/types";
import { useApi } from "../api/context";
import { AsyncState } from "../components/AsyncState";
import { ProcessingProgress } from "../components/ProcessingProgress";
import { StatusBadge } from "../components/StatusBadge";
import { findThumbnail, matchStatus, playerName } from "../domain";
import { useAsyncData } from "../hooks/useAsyncData";

interface MatchListItem {
  match: Match;
  players: Player[];
  thumbnail: Artifact | null;
  job: ProcessingJob | null;
}

const TERMINAL_JOB_STATUSES = new Set(["COMPLETE", "FAILED", "CANCELED"]);

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
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [title, setTitle] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<Error | null>(null);
  const [submission, setSubmission] = useState<YouTubeMatchSubmission | null>(null);
  const loadMatches = useCallback(
    async (signal: AbortSignal): Promise<MatchListItem[]> => {
      const matches = await api.getMatches(signal);
      return Promise.all(
        matches.map(async (match) => {
          const [players, artifacts, job] = await Promise.all([
            api.getPlayers(match.matchId, signal).catch(() => []),
            api.getArtifacts(match.matchId, signal).catch(() => []),
            api.getLatestProcessingJob(match.matchId, signal).catch(() => null),
          ]);
          return { match, players, thumbnail: findThumbnail(artifacts), job };
        }),
      );
    },
    [api],
  );
  const { data, loading, error, reload } = useAsyncData(loadMatches);

  useEffect(() => {
    const job = submission?.job;
    if (job === undefined || TERMINAL_JOB_STATUSES.has(job.status)) return;
    const controller = new AbortController();
    const poll = window.setInterval(() => {
      void api
        .getProcessingJob(job.jobId, controller.signal)
        .then((nextJob) => {
          setSubmission((current) =>
            current === null ? null : { ...current, job: nextJob },
          );
          if (TERMINAL_JOB_STATUSES.has(nextJob.status)) {
            window.clearInterval(poll);
            reload();
          }
        })
        .catch(() => undefined);
    }, 5_000);
    return () => {
      controller.abort();
      window.clearInterval(poll);
    };
  }, [api, reload, submission?.job]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setSubmitError(null);
    setSubmission(null);
    try {
      const result = await api.submitYouTubeMatch(
        youtubeUrl,
        title.trim() === "" ? null : title.trim(),
      );
      setSubmission(result);
      setYoutubeUrl("");
      setTitle("");
      reload();
    } catch (error) {
      setSubmitError(
        error instanceof Error ? error : new Error("Unable to submit this recording."),
      );
    } finally {
      setSubmitting(false);
    }
  }

  const existingMatchId =
    submitError instanceof ApiClientError && submitError.code === "youtube_match_exists"
      ? submitError.details?.matchId
      : null;

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
          <strong className={loading ? "hero-stat-loading" : undefined}>
            {loading ? <span className="sr-only">Loading match count</span> : data?.length ?? "N/A"}
          </strong>
          <span>matches available</span>
        </div>
      </section>

      <section className="submission-panel" aria-labelledby="submit-match-title">
        <div className="submission-copy">
          <h2 id="submit-match-title">Paste a match link.</h2>
          <p>
            Add an unlisted YouTube recording and Rallymetry will queue the configured
            court, player, ball, audio, rally, shot, and analytics pipeline.
          </p>
          <small>Only submit recordings you own or have permission to analyze.</small>
        </div>
        <form className="submission-form" onSubmit={(event) => void handleSubmit(event)}>
          <label htmlFor="youtube-url">YouTube link</label>
          <div className="submission-input-wrap">
            <Link2 aria-hidden="true" />
            <input
              id="youtube-url"
              type="url"
              inputMode="url"
              autoComplete="url"
              placeholder="https://www.youtube.com/watch?v=…"
              value={youtubeUrl}
              onChange={(event) => setYoutubeUrl(event.target.value)}
              required
              disabled={submitting}
            />
          </div>
          <label htmlFor="match-title">Match title <span>optional</span></label>
          <input
            id="match-title"
            type="text"
            maxLength={512}
            placeholder="Thursday night doubles"
            value={title}
            onChange={(event) => setTitle(event.target.value)}
            disabled={submitting}
          />
          <button className="button button--primary submission-button" type="submit" disabled={submitting}>
            {submitting ? <LoaderCircle className="spin" aria-hidden="true" /> : <Upload aria-hidden="true" />}
            {submitting ? "Queueing…" : "Upload & analyze"}
          </button>
          <div className="submission-result" aria-live="polite">
            {submission === null ? null : (
              <div className="submission-result--success">
                <CheckCircle2 aria-hidden="true" />
                <div>
                  <strong>
                    {submission.job.currentStepLabel ??
                      submission.job.stage?.replaceAll("_", " ") ??
                      submission.job.status}
                  </strong>
                  <span>
                    {Math.round(submission.job.progress * 100)}% -{" "}
                    {submission.job.currentStepDescription ??
                      "Analysis runs in the background."}
                  </span>
                  <ProcessingProgress
                    value={submission.job.progress}
                    label={`${submission.job.currentStepLabel ?? submission.job.stage?.replaceAll("_", " ") ?? submission.job.status} progress`}
                  />
                  <Link to={`/matches/${submission.match.matchId}`}>Open match</Link>
                </div>
              </div>
            )}
            {submitError === null ? null : (
              <div className="submission-result--error">
                <strong>{submitError.message}</strong>
                {typeof existingMatchId === "string" && existingMatchId !== "" ? (
                  <Link to={`/matches/${existingMatchId}`}>Open existing match</Link>
                ) : null}
              </div>
            )}
          </div>
        </form>
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
          emptyMessage="Paste a YouTube recording above to create and queue your first analysis."
        >
          <div className="match-grid">
            {data?.map(({ match, players, thumbnail, job }) => (
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
                  <StatusBadge status={job?.status ?? matchStatus(match)} />
                </div>
                <div className="match-card-body">
                  <span className="match-date">
                    <CalendarDays aria-hidden="true" /> {formatDate(match.updatedAt)}
                  </span>
                  <h3>{match.title ?? "Untitled match"}</h3>
                  <p className="match-players">
                    <Users aria-hidden="true" />
                    {players.length > 0
                      ? players.map(playerName).join(", ")
                      : "Players pending"}
                  </p>
                  {job !== null && !TERMINAL_JOB_STATUSES.has(job.status) ? (
                    <p className="match-card-progress">
                      <span>{job.currentStepLabel ?? job.stage?.replaceAll("_", " ") ?? job.status}</span>
                      <strong>{Math.round(job.progress * 100)}%</strong>
                    </p>
                  ) : null}
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
