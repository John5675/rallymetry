import type {
  Analytics,
  Artifact,
  DomainRecord,
  JsonValue,
  Match,
  Player,
} from "./api/types";

export const COURT_WIDTH_METERS = 6.096;
export const COURT_LENGTH_METERS = 13.4112;

export type EventKind = "rally" | "contact" | "bounce" | "shot";

export interface TimelineEvent {
  id: string;
  kind: EventKind;
  label: string;
  timestampSeconds: number;
  confidence: number | null;
}

export interface CourtPoint {
  id: string;
  x: number;
  y: number;
  label: string;
  shotType: string;
  confidence: number | null;
}

export function asObject(value: JsonValue | undefined): Record<string, JsonValue> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value : null;
}

export function stringValue(
  object: Record<string, JsonValue>,
  key: string,
): string | null {
  const value = object[key];
  return typeof value === "string" ? value : null;
}

export function numberValue(
  object: Record<string, JsonValue>,
  key: string,
): number | null {
  const value = object[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function nestedValue(
  root: Record<string, JsonValue>,
  path: readonly string[],
): JsonValue | undefined {
  let current: JsonValue = root;
  for (const segment of path) {
    const object = asObject(current);
    if (object === null || !(segment in object)) {
      return undefined;
    }
    current = object[segment] as JsonValue;
  }
  return current;
}

export function metricNumber(
  analytics: Analytics | null,
  path: readonly string[],
): number | null {
  if (analytics === null) {
    return null;
  }
  const metric = asObject(nestedValue(analytics.metrics, path));
  const value = metric?.value;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function analyticsNumber(
  analytics: Analytics | null,
  path: readonly string[],
): number | null {
  if (analytics === null) {
    return null;
  }
  const value = nestedValue(analytics.metrics, path);
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function playerName(player: Player): string {
  return player.displayName ?? player.logicalIdentity ?? player.playerId;
}

export function playerNameById(players: Player[], playerId: string | null): string {
  if (playerId === null) {
    return "Unknown";
  }
  const player = players.find(
    (item) => item.playerId === playerId || item.logicalIdentity === playerId,
  );
  return player === undefined ? playerId.replaceAll("_", " ") : playerName(player);
}

export function matchStatus(match: Match): string {
  for (const key of ["processingStatus", "status", "jobStatus"] as const) {
    const value = match.summary[key];
    if (typeof value === "string" && value.length > 0) {
      return value;
    }
  }
  if (match.pipelineVersion !== null) {
    return "COMPLETE";
  }
  if (match.sourceArtifactId !== null) {
    return "READY";
  }
  return "DRAFT";
}

export function isPublicViewable(artifact: Artifact): boolean {
  return artifact.access === "PUBLIC" && artifact.url !== null;
}

export function findPublicArtifact(
  artifacts: Artifact[],
  predicate: (artifact: Artifact) => boolean,
): Artifact | null {
  return artifacts.find((artifact) => isPublicViewable(artifact) && predicate(artifact)) ?? null;
}

export function findThumbnail(artifacts: Artifact[]): Artifact | null {
  return findPublicArtifact(
    artifacts,
    (artifact) =>
      artifact.contentType.startsWith("image/") &&
      artifact.artifactType.toLowerCase().includes("thumbnail"),
  );
}

export function findPrimaryVideo(artifacts: Artifact[]): Artifact | null {
  const priorities = [
    "annotated",
    "shot_debug_video",
    "ball_tracking_video",
    "player_tracking_video",
    "rally_debug_video",
  ];
  for (const name of priorities) {
    const artifact = findPublicArtifact(
      artifacts,
      (item) =>
        item.contentType.startsWith("video/") &&
        item.artifactType.toLowerCase().includes(name),
    );
    if (artifact !== null) {
      return artifact;
    }
  }
  return null;
}

function recordTimestamp(record: DomainRecord, ...payloadKeys: string[]): number | null {
  for (const key of payloadKeys) {
    const value = numberValue(record.payload, key);
    if (value !== null) {
      return value;
    }
  }
  return record.timestampSeconds;
}

function recordConfidence(record: DomainRecord): number | null {
  return (
    numberValue(record.payload, "fusedConfidence") ??
    numberValue(record.payload, "confidence") ??
    record.confidence
  );
}

export function buildTimelineEvents(
  rallies: DomainRecord[],
  contacts: DomainRecord[],
  bounces: DomainRecord[],
  shots: DomainRecord[],
): TimelineEvent[] {
  const events: TimelineEvent[] = [];
  for (const rally of rallies) {
    const rallyId = stringValue(rally.payload, "rallyId") ?? rally.recordId;
    const start = recordTimestamp(rally, "startTimestamp");
    const end = recordTimestamp(rally, "endTimestamp");
    if (start !== null) {
      events.push({
        id: `${rally.recordId}-start`,
        kind: "rally",
        label: `${rallyId} start`,
        timestampSeconds: start,
        confidence: recordConfidence(rally),
      });
    }
    if (end !== null) {
      events.push({
        id: `${rally.recordId}-end`,
        kind: "rally",
        label: `${rallyId} end`,
        timestampSeconds: end,
        confidence: recordConfidence(rally),
      });
    }
  }
  for (const [kind, records, key] of [
    ["contact", contacts, "contactId"],
    ["bounce", bounces, "bounceId"],
    ["shot", shots, "shotId"],
  ] as const) {
    for (const record of records) {
      const timestamp = recordTimestamp(record, "timestamp", "contactTimestamp", "mediaTimestamp");
      if (timestamp !== null) {
        events.push({
          id: `${kind}-${record.recordId}`,
          kind,
          label: stringValue(record.payload, key) ?? record.recordId,
          timestampSeconds: timestamp,
          confidence: recordConfidence(record),
        });
      }
    }
  }
  return events.sort((left, right) => left.timestampSeconds - right.timestampSeconds);
}

export function shotCourtPoints(shots: DomainRecord[]): CourtPoint[] {
  const points: CourtPoint[] = [];
  for (const shot of shots) {
    const landing = asObject(shot.payload.landingCourtPosition);
    const x = landing === null ? null : numberValue(landing, "x");
    const y = landing === null ? null : numberValue(landing, "y");
    if (
      x === null ||
      y === null ||
      x < 0 ||
      x > COURT_WIDTH_METERS ||
      y < 0 ||
      y > COURT_LENGTH_METERS
    ) {
      continue;
    }
    points.push({
      id: shot.recordId,
      x,
      y,
      label: stringValue(shot.payload, "shotId") ?? shot.recordId,
      shotType: stringValue(shot.payload, "shotType") ?? "UNKNOWN",
      confidence: recordConfidence(shot),
    });
  }
  return points;
}

export function formatDuration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) {
    return "N/A";
  }
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remaining = Math.round(seconds % 60);
  return `${minutes}:${remaining.toString().padStart(2, "0")}`;
}

export function formatPercent(value: number | null): string {
  return value === null ? "N/A" : `${(value * 100).toFixed(0)}%`;
}

export function formatConfidence(value: number | null): string {
  return value === null ? "N/A" : `${Math.round(value * 100)}%`;
}
