import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { ProcessingJob } from "../api/types";
import { ProcessingBanner } from "./ProcessingBanner";

afterEach(cleanup);

function processingJob(overrides: Partial<ProcessingJob> = {}): ProcessingJob {
  return {
    jobId: "job-1",
    matchId: "match-1",
    jobType: "analyze_match",
    status: "PLAYER_PROCESSING",
    progress: 0.15,
    stage: "PLAYER_PROCESSING",
    currentStep: "detect-people",
    currentStepLabel: "Detecting people",
    currentStepDescription: "Scanning every video frame for visible people.",
    currentStepIndex: 1,
    totalSteps: 13,
    renderTriggeredAt: null,
    claimedAt: "2026-08-22T06:30:00Z",
    heartbeatAt: "2026-08-22T06:35:00Z",
    leaseExpiresAt: "2026-08-22T06:38:00Z",
    workerId: "johns-mac",
    startedAt: "2026-08-22T06:30:00Z",
    completedAt: null,
    failedAt: null,
    failedStage: null,
    renderTaskRunId: null,
    processingRunId: "run-1",
    attemptCount: 1,
    errorCode: null,
    errorMessage: null,
    pipelineVersion: "pipeline-1",
    sourceType: "YOUTUBE",
    sourceArtifactId: null,
    youtubeVideoId: "6f7M8b6uKi4",
    resultArtifactIds: [],
    resultSummary: {},
    createdAt: "2026-08-22T06:29:00Z",
    updatedAt: "2026-08-22T06:35:00Z",
    ...overrides,
  };
}

describe("ProcessingBanner", () => {
  it("shows the current pipeline command, step count, heartbeat, and progress", () => {
    render(<ProcessingBanner job={processingJob()} />);

    expect(screen.getByText("Detecting people")).toBeVisible();
    expect(screen.getByText("Analysis step 1 of 13")).toBeVisible();
    expect(screen.getByText(/Worker checked in at/)).toBeVisible();
    expect(screen.getByRole("progressbar", { name: "Detecting people progress" })).toHaveAttribute(
      "aria-valuenow",
      "15",
    );
  });

  it("keeps a failure message more prominent than stale step metadata", () => {
    render(
      <ProcessingBanner
        job={processingJob({
          status: "FAILED",
          errorMessage: "Unable to decode source media",
        })}
      />,
    );

    expect(screen.getByText("Unable to decode source media")).toBeVisible();
    expect(screen.queryByText("Analysis step 1 of 13")).not.toBeInTheDocument();
  });
});
