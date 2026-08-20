import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiClient } from "../api/client";
import { ApiProvider } from "../api/context";
import { AppRoutes } from "../App";

function page(items: unknown[]) {
  return { items, total: items.length, limit: 100, offset: 0 };
}

describe("MatchesPage", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("renders persisted match, player, status, and public thumbnail data", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url =
        typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      let body: unknown;
      if (url.includes("/players")) {
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
});
