import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiClient } from "./client";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("ApiClient", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("normalizes the configured base URL and reads paginated match data", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        items: [{ matchId: "match-1" }],
        total: 1,
        limit: 100,
        offset: 0,
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const client = new ApiClient("https://api.example.test/");

    const matches = await client.getMatches();

    expect(matches).toEqual([{ matchId: "match-1" }]);
    expect(fetchMock).toHaveBeenCalledWith(
      "https://api.example.test/api/matches?limit=100&offset=0",
      expect.objectContaining({ headers: { Accept: "application/json" } }),
    );
  });

  it("turns the API error envelope into a stable typed error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(
          {
            error: {
              code: "match_not_found",
              message: "Match was not found.",
              requestId: "request-7",
            },
          },
          404,
        ),
      ),
    );
    const client = new ApiClient("");

    await expect(client.getMatch("missing")).rejects.toEqual(
      expect.objectContaining({
        status: 404,
        code: "match_not_found",
        requestId: "request-7",
      }),
    );
  });

  it("treats missing analytics as unavailable without hiding other errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ error: {} }, 404)));
    const client = new ApiClient("");

    await expect(client.getAnalytics("match-1")).resolves.toBeNull();
  });

  it("submits one YouTube link as JSON to the asynchronous import endpoint", async () => {
    const payload = { match: { matchId: "match-1" }, job: { jobId: "job-1" } };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(payload, 202));
    vi.stubGlobal("fetch", fetchMock);
    const client = new ApiClient("https://api.example.test");

    await expect(
      client.submitYouTubeMatch("https://youtu.be/_cPF1fTnk0Y", null),
    ).resolves.toEqual(payload);
    expect(fetchMock).toHaveBeenCalledWith(
      "https://api.example.test/api/matches/import-youtube",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ youtubeUrl: "https://youtu.be/_cPF1fTnk0Y" }),
        headers: { Accept: "application/json", "Content-Type": "application/json" },
      }),
    );
  });

  it("creates and removes human corrections through FastAPI only", async () => {
    const correction = {
      correctionId: "correction-1",
      correctionType: "SHOT_TYPE",
      targetRecordId: "shot-1",
    };
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(correction, 201))
      .mockResolvedValueOnce(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);
    const client = new ApiClient("https://api.example.test");

    await expect(
      client.createCorrection("match-1", {
        correctionType: "SHOT_TYPE",
        targetRecordId: "shot-1",
        humanCorrection: { shotType: "DRIVE" },
        verified: true,
      }),
    ).resolves.toEqual(correction);
    await expect(client.removeCorrection("match-1", "correction-1")).resolves.toBeUndefined();

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "https://api.example.test/api/matches/match-1/corrections",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          correctionType: "SHOT_TYPE",
          targetRecordId: "shot-1",
          humanCorrection: { shotType: "DRIVE" },
          verified: true,
        }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "https://api.example.test/api/matches/match-1/corrections/correction-1",
      expect.objectContaining({ method: "DELETE" }),
    );
  });
});
