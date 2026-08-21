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
  analysisProfileMatchId: string | null;
  analysisSetup: Record<string, string>;
  pipelineVersion: string | null;
  modelVersions: Record<string, string>;
  summary: Record<string, JsonValue>;
  artifactIds: string[];
  createdAt: string;
  updatedAt: string;
}

export interface ProcessingJob {
  jobId: string;
  matchId: string;
  jobType: string;
  status: string;
  progress: number;
  stage: string | null;
  renderTriggeredAt: string | null;
  startedAt: string | null;
  completedAt: string | null;
  failedAt: string | null;
  failedStage: string | null;
  renderTaskRunId: string | null;
  processingRunId: string | null;
  attemptCount: number;
  errorCode: string | null;
  errorMessage: string | null;
  pipelineVersion: string | null;
  sourceType: string | null;
  sourceArtifactId: string | null;
  youtubeVideoId: string | null;
  resultArtifactIds: string[];
  resultSummary: Record<string, JsonValue>;
  createdAt: string;
  updatedAt: string;
}

export interface YouTubeMatchSubmission {
  match: Match;
  job: ProcessingJob;
}

export interface Player {
  matchId: string;
  playerId: string;
  displayName: string | null;
  logicalIdentity: string | null;
  team: string | null;
  metadata: Record<string, JsonValue>;
  effectivePlayer?: Record<string, JsonValue> | null;
  verifiedCorrections?: Correction[];
  createdAt: string;
  updatedAt: string;
}

export interface DomainRecord {
  matchId: string;
  recordId: string;
  payload: Record<string, JsonValue>;
  effectivePayload?: Record<string, JsonValue>;
  verifiedCorrections?: Correction[];
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
  predictionMetrics?: Record<string, JsonValue> | null;
  appliedCorrectionIds?: string[];
  inputArtifactIds: string[];
  pipelineVersion: string | null;
  createdAt: string;
}

export type CorrectionType =
  | "PLAYER_IDENTITY"
  | "RALLY_BOUNDARY"
  | "BOUNCE"
  | "HITTER"
  | "SHOT_TYPE";

export interface Correction {
  correctionId: string;
  matchId: string;
  correctionType: CorrectionType;
  targetCollection: string;
  targetRecordId: string;
  prediction: Record<string, JsonValue>;
  predictionConfidence: number | null;
  predictionVersion: string | null;
  humanCorrection: Record<string, JsonValue>;
  verified: boolean;
  active: boolean;
  revision: number;
  history: Record<string, JsonValue>[];
  reason: string | null;
  correctedBy: string | null;
  visualEvidence: Record<string, JsonValue> | null;
  audioEvidence: Record<string, JsonValue> | null;
  createdAt: string;
  correctedAt: string;
  updatedAt: string;
  deletedAt: string | null;
}

export interface CorrectionInput {
  correctionType: CorrectionType;
  targetRecordId: string;
  humanCorrection: Record<string, JsonValue>;
  verified: boolean;
  reason?: string;
  correctedBy?: string;
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
  corrections: Correction[];
}
