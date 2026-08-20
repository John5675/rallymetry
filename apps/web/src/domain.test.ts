import { describe, expect, it } from "vitest";

import type { Artifact, DomainRecord } from "./api/types";
import { buildTimelineEvents, findPrimaryVideo, shotCourtPoints } from "./domain";

function record(recordId: string, payload: DomainRecord["payload"]): DomainRecord {
  return {
    matchId: "match-1",
    recordId,
    payload,
    confidence: null,
    timestampSeconds: null,
    pipelineVersion: null,
    modelVersion: null,
    createdAt: "2026-08-20T00:00:00Z",
  };
}

function artifact(overrides: Partial<Artifact>): Artifact {
  return {
    artifactId: "artifact-1",
    matchId: "match-1",
    artifactType: "annotated",
    category: "VIEWABLE_MEDIA",
    pathname: "random/video.mp4",
    provider: "VERCEL_BLOB",
    access: "PUBLIC",
    contentType: "video/mp4",
    size: 10,
    createdAt: "2026-08-20T00:00:00Z",
    pipelineVersion: null,
    url: "https://blob.example/video.mp4",
    checksumSha256: null,
    ...overrides,
  };
}

describe("dashboard domain projections", () => {
  it("creates distinct chronological rally, contact, bounce, and shot markers", () => {
    const timeline = buildTimelineEvents(
      [record("r1", { rallyId: "rally-1", startTimestamp: 2, endTimestamp: 8, confidence: 0.9 })],
      [record("c1", { contactId: "contact-1", timestamp: 3, fusedConfidence: 0.8 })],
      [record("b1", { bounceId: "bounce-1", timestamp: 4, fusedConfidence: 0.7 })],
      [record("s1", { shotId: "shot-1", contactTimestamp: 3.1, confidence: 0.75 })],
    );

    expect(timeline.map((item) => item.kind)).toEqual([
      "rally",
      "contact",
      "shot",
      "bounce",
      "rally",
    ]);
    expect(timeline[2]?.timestampSeconds).toBe(3.1);
  });

  it("plots only defensible in-court landing positions", () => {
    const points = shotCourtPoints([
      record("known", {
        shotType: "DROP",
        landingCourtPosition: { x: 2.1, y: 8.4 },
      }),
      record("missing", { shotType: "DRIVE", landingCourtPosition: null }),
      record("outside", {
        shotType: "DINK",
        landingCourtPosition: { x: 20, y: 2 },
      }),
    ]);

    expect(points).toHaveLength(1);
    expect(points[0]).toEqual(expect.objectContaining({ id: "known", x: 2.1, y: 8.4 }));
  });

  it("never selects a private artifact as browser-playable media", () => {
    expect(findPrimaryVideo([artifact({ access: "PRIVATE" })])).toBeNull();
    expect(findPrimaryVideo([artifact({ access: "PUBLIC" })])?.url).toContain("blob.example");
  });

  it("does not mistake court-map media for the primary match video", () => {
    expect(
      findPrimaryVideo([
        artifact({
          artifactType: "player_topdown_video",
          pathname: "random/topdown.mp4",
          url: "https://blob.example/topdown.mp4",
        }),
      ]),
    ).toBeNull();
  });
});
