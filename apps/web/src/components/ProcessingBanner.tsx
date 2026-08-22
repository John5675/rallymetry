import type { ProcessingJob } from "../api/types";
import { ProcessingProgress } from "./ProcessingProgress";

interface ProcessingBannerProps {
  job: ProcessingJob;
}

function formatWorkerCheckIn(value: string | null): string | null {
  if (value === null) return null;
  const timestamp = new Date(value);
  if (Number.isNaN(timestamp.getTime())) return null;
  return `Worker checked in at ${timestamp.toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  })}`;
}

export function ProcessingBanner({ job }: ProcessingBannerProps) {
  const failed = job.status === "FAILED";
  const label = job.currentStepLabel ?? job.stage?.replaceAll("_", " ") ?? job.status;
  const description = failed
    ? job.errorMessage ?? "Analysis stopped before completion."
    : job.currentStepDescription ??
      "Analysis is running in the background. You can safely leave this page.";
  const workerCheckIn = formatWorkerCheckIn(job.heartbeatAt ?? null);

  return (
    <section
      className={`processing-banner processing-banner--${failed ? "failed" : "active"}`}
      aria-live="polite"
    >
      <div className="processing-banner__copy">
        <strong>{label}</strong>
        <span>{description}</span>
        {failed ? null : (
          <div className="processing-banner__meta">
            {typeof job.currentStepIndex === "number" && typeof job.totalSteps === "number" ? (
              <span>
                Analysis step {job.currentStepIndex} of {job.totalSteps}
              </span>
            ) : null}
            {workerCheckIn === null ? null : <span>{workerCheckIn}</span>}
          </div>
        )}
      </div>
      <div className="processing-banner__meter">
        <span>{Math.round(job.progress * 100)}%</span>
        <ProcessingProgress value={job.progress} label={`${label} progress`} />
      </div>
    </section>
  );
}
