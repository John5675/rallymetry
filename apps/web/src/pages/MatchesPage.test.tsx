import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiClient } from "../api/client";
import { ApiProvider } from "../api/context";
import { AppRoutes } from "../App";

function page(items: unknown[]) {
  return { items, total: items.length, limit: 100, offset: 0 };
}

describe("MatchesPage", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("renders persisted match, player, status, and public thumbnail data", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url =
        typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      let body: unknown;
      if (url.includes("/processing-job")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({ error: { code: "processing_job_not_found", message: "Not found" } }),
            { status: 404, headers: { "Content-Type": "application/json" } },
          ),
        );
      } else if (url.includes("/players")) {
        body = page([
          {
            matchId: "match-1",
            playerId: "p1",
            displayName: "John",
            logicalIdentity: "ME",
            team: "near",
            metadata: {},
            createdAt: "2026-08-20T00:00:00Z",
            updatedAt: "2026-08-20T00:00:00Z",
          },
        ]);
      } else if (url.includes("/artifacts")) {
        body = page([
          {
            artifactId: "a1",
            matchId: "match-1",
            artifactType: "thumbnail",
            category: "VIEWABLE_MEDIA",
            pathname: "random/thumb.jpg",
            provider: "VERCEL_BLOB",
            access: "PUBLIC",
            contentType: "image/jpeg",
            size: 10,
            createdAt: "2026-08-20T00:00:00Z",
            pipelineVersion: "0.1",
            url: "https://blob.example/thumb.jpg",
            checksumSha256: null,
          },
        ]);
      } else {
        body = page([
          {
            matchId: "match-1",
            title: "Friday night final",
            youtubeVideoId: null,
            sourceArtifactId: "source-1",
            analysisProfileMatchId: null,
            analysisSetup: {},
            pipelineVersion: "0.1",
            modelVersions: {},
            summary: { status: "COMPLETE" },
            artifactIds: [],
            createdAt: "2026-08-20T00:00:00Z",
            updatedAt: "2026-08-20T00:00:00Z",
          },
        ]);
      }
      return Promise.resolve(
        new Response(JSON.stringify(body), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <ApiProvider client={new ApiClient("https://api.example.test")}>
        <MemoryRouter initialEntries={["/matches"]}>
          <AppRoutes />
        </MemoryRouter>
      </ApiProvider>,
    );

    expect(await screen.findByRole("heading", { name: "Friday night final" })).toBeVisible();
    expect(screen.getByText("John")).toBeVisible();
    expect(screen.getByText("COMPLETE")).toBeVisible();
    expect(screen.getByRole("img", { name: "Friday night final thumbnail" })).toHaveAttribute(
      "src",
      "https://blob.example/thumb.jpg",
    );
  });

  it("submits a YouTube link and shows the queued analysis job", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url =
        typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (url.endsWith("/api/matches/import-youtube") && init?.method === "POST") {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              match: {
                matchId: "match-new",
                title: "Thursday night doubles",
                youtubeVideoId: "_cPF1fTnk0Y",
                sourceArtifactId: null,
                analysisProfileMatchId: "match-profile",
                analysisSetup: {},
                pipelineVersion: null,
                modelVersions: {},
                summary: { status: "CREATED" },
                artifactIds: [],
                createdAt: "2026-08-20T00:00:00Z",
                updatedAt: "2026-08-20T00:00:00Z",
              },
              job: {
                jobId: "job-new",
                matchId: "match-new",
                jobType: "analyze_match",
                status: "QUEUED",
                progress: 0,
                stage: "QUEUED",
                renderTriggeredAt: "2026-08-20T00:00:00Z",
                startedAt: null,
                completedAt: null,
                failedAt: null,
                failedStage: null,
                renderTaskRunId: "trn-new",
                processingRunId: "run-new",
                attemptCount: 0,
                errorCode: null,
                errorMessage: null,
                pipelineVersion: null,
                sourceType: "YOUTUBE",
                sourceArtifactId: null,
                youtubeVideoId: "_cPF1fTnk0Y",
                resultArtifactIds: [],
                resultSummary: {},
                createdAt: "2026-08-20T00:00:00Z",
                updatedAt: "2026-08-20T00:00:00Z",
              },
            }),
            { status: 202, headers: { "Content-Type": "application/json" } },
          ),
        );
      }
      return Promise.resolve(
        new Response(JSON.stringify(page([])), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <ApiProvider client={new ApiClient("https://api.example.test")}>
        <MemoryRouter initialEntries={["/matches"]}>
          <AppRoutes />
        </MemoryRouter>
      </ApiProvider>,
    );

    await user.type(
      screen.getByRole("textbox", { name: "YouTube link" }),
      "https://www.youtube.com/watch?v=_cPF1fTnk0Y",
    );
    await user.type(
      screen.getByRole("textbox", { name: /Match title/i }),
      "Thursday night doubles",
    );
    await user.click(screen.getByRole("button", { name: "Upload & analyze" }));

    expect(await screen.findByText(/Analysis runs in the background\./)).toBeVisible();
    expect(screen.getByRole("link", { name: "Open match" })).toHaveAttribute(
      "href",
      "/matches/match-new",
    );
    const submissionCall = fetchMock.mock.calls.find(([input]) => {
      const url =
        typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      return url.endsWith("/api/matches/import-youtube");
    });
    expect(submissionCall?.[1]).toEqual(
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          youtubeUrl: "https://www.youtube.com/watch?v=_cPF1fTnk0Y",
          title: "Thursday night doubles",
        }),
      }),
    );
  });
});
