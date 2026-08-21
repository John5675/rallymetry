import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ApiClient } from "../api/client";
import { ApiProvider } from "../api/context";
import type { MatchDashboardData } from "../api/types";
import { CorrectionWorkspace } from "./CorrectionWorkspace";

const DATA: MatchDashboardData = {
  match: {
    matchId: "match-1",
    title: "Review match",
    youtubeVideoId: null,
    sourceArtifactId: null,
    analysisProfileMatchId: null,
    analysisSetup: {},
    pipelineVersion: "pipeline-v1",
    modelVersions: {},
    summary: {},
    artifactIds: [],
    createdAt: "2026-08-21T00:00:00Z",
    updatedAt: "2026-08-21T00:00:00Z",
  },
  players: [],
  rallies: [],
  contacts: [],
  bounces: [],
  shots: [
    {
      matchId: "match-1",
      recordId: "shot-1",
      payload: { shotIndex: 1, shotType: "UNKNOWN", hitterId: "ME" },
      confidence: 0.42,
      timestampSeconds: 1,
      pipelineVersion: "pipeline-v1",
      modelVersion: "shot-model-v1",
      createdAt: "2026-08-21T00:00:00Z",
    },
  ],
  analytics: null,
  artifacts: [],
  corrections: [],
};

describe("CorrectionWorkspace", () => {
  it("shows AI and human values separately and saves through the API", async () => {
    const user = userEvent.setup();
    const client = new ApiClient("");
    const create = vi.spyOn(client, "createCorrection").mockResolvedValue({} as never);
    const onChanged = vi.fn();

    render(
      <ApiProvider client={client}>
        <CorrectionWorkspace data={DATA} onChanged={onChanged} />
      </ApiProvider>,
    );

    expect(screen.getByText("AI prediction")).toBeVisible();
    expect(screen.getByText("Human correction")).toBeVisible();
    expect(screen.getAllByText(/UNKNOWN/).length).toBeGreaterThan(0);
    await user.selectOptions(screen.getByLabelText("Correct shot type"), "DRIVE");
    await user.click(screen.getByRole("button", { name: "Save correction" }));

    await waitFor(() => {
      expect(create).toHaveBeenCalledWith(
        "match-1",
        expect.objectContaining({
          correctionType: "SHOT_TYPE",
          targetRecordId: "shot-1",
          humanCorrection: { shotType: "DRIVE" },
          verified: true,
        }),
      );
    });
    expect(onChanged).toHaveBeenCalledOnce();
  });
});
