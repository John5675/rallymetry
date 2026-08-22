import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { DomainRecord } from "../api/types";
import { ShotTable } from "./ShotTable";

describe("ShotTable", () => {
  it("shows AI visual review separately without replacing the machine prediction", () => {
    const shot: DomainRecord = {
      matchId: "match-1",
      recordId: "shot-1",
      payload: {
        shotIndex: 1,
        rallyId: "rally-1",
        hitterId: "ME",
        shotType: "UNKNOWN",
        contactTimestamp: 2,
        confidence: 0.2,
        landingCourtPosition: null,
        aiVisualReview: {
          predictionPreserved: true,
          humanCorrectionCreated: false,
          review: {
            legacyBestGuess: "DINK",
            legacyBestGuessConfidence: 0.72,
            humanAccepted: false,
          },
        },
      },
      confidence: 0.2,
      timestampSeconds: 2,
      pipelineVersion: "test",
      modelVersion: null,
      createdAt: "2026-08-21T00:00:00Z",
    };

    render(<ShotTable shots={[shot]} players={[]} />);

    expect(screen.getByText("UNKNOWN")).toBeInTheDocument();
    expect(screen.getByText("AI visual review: DINK · 72%")).toBeInTheDocument();
  });
});
