export type JsonPrimitive = boolean | number | string | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface Match {
  matchId: string;
  title: string | null;
  youtubeVideoId: string | null;
  sourceArtifactId: string | null;
  pipelineVersion: string | null;
  modelVersions: Record<string, string>;
  summary: Record<string, JsonValue>;
  artifactIds: string[];
  createdAt: string;
  updatedAt: string;
}

export interface Player {
  matchId: string;
  playerId: string;
  displayName: string | null;
  logicalIdentity: string | null;
  team: string | null;
  metadata: Record<string, JsonValue>;
  createdAt: string;
  updatedAt: string;
}

export interface DomainRecord {
  matchId: string;
  recordId: string;
  payload: Record<string, JsonValue>;
  confidence: number | null;
  timestampSeconds: number | null;
  pipelineVersion: string | null;
  modelVersion: string | null;
  createdAt: string;
}

export interface Analytics {
  matchId: string;
  analyticsId: string;
  calculationVersion: string;
  metrics: Record<string, JsonValue>;
  inputArtifactIds: string[];
  pipelineVersion: string | null;
  createdAt: string;
}

export type ArtifactAccess = "PRIVATE" | "PUBLIC";

export interface Artifact {
  artifactId: string;
  matchId: string | null;
  artifactType: string;
  category: string;
  pathname: string;
  provider: string;
  access: ArtifactAccess;
  contentType: string;
  size: number;
  createdAt: string;
  pipelineVersion: string | null;
  url: string | null;
  checksumSha256: string | null;
}

export interface ApiErrorEnvelope {
  error: {
    code: string;
    message: string;
    details?: Record<string, JsonValue> | null;
    requestId?: string | null;
  };
}

export interface MatchDashboardData {
  match: Match;
  players: Player[];
  rallies: DomainRecord[];
  shots: DomainRecord[];
  contacts: DomainRecord[];
  bounces: DomainRecord[];
  analytics: Analytics | null;
  artifacts: Artifact[];
}
