import type {
  Analytics,
  ApiErrorEnvelope,
  Artifact,
  DomainRecord,
  Match,
  MatchDashboardData,
  Page,
  Player,
} from "./types";

const PAGE_SIZE = 100;

export class ApiClientError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId: string | null;

  constructor(message: string, status: number, code: string, requestId: string | null = null) {
    super(message);
    this.name = "ApiClientError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

function normalizeBaseUrl(value: string | undefined): string {
  return (value ?? "").trim().replace(/\/$/, "");
}

function isApiErrorEnvelope(value: unknown): value is ApiErrorEnvelope {
  if (typeof value !== "object" || value === null || !("error" in value)) {
    return false;
  }
  const error = value.error;
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    typeof error.code === "string" &&
    "message" in error &&
    typeof error.message === "string"
  );
}

async function parseResponse<T>(response: Response): Promise<T> {
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    body = null;
  }

  if (!response.ok) {
    if (isApiErrorEnvelope(body)) {
      throw new ApiClientError(
        body.error.message,
        response.status,
        body.error.code,
        body.error.requestId ?? null,
      );
    }
    throw new ApiClientError(
      `The API returned ${response.status} ${response.statusText}.`,
      response.status,
      "http_error",
    );
  }
  return body as T;
}

export class ApiClient {
  readonly baseUrl: string;

  constructor(baseUrl: string | undefined = import.meta.env.VITE_API_BASE_URL) {
    this.baseUrl = normalizeBaseUrl(baseUrl);
  }

  private async request<T>(path: string, signal?: AbortSignal): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      headers: { Accept: "application/json" },
      signal,
    });
    return parseResponse<T>(response);
  }

  private async listAll<T>(path: string, signal?: AbortSignal): Promise<T[]> {
    const items: T[] = [];
    let offset = 0;
    let total = Number.POSITIVE_INFINITY;

    while (offset < total) {
      const separator = path.includes("?") ? "&" : "?";
      const page = await this.request<Page<T>>(
        `${path}${separator}limit=${PAGE_SIZE}&offset=${offset}`,
        signal,
      );
      items.push(...page.items);
      total = page.total;
      if (page.items.length === 0) {
        break;
      }
      offset += page.items.length;
    }
    return items;
  }

  getMatches(signal?: AbortSignal): Promise<Match[]> {
    return this.listAll<Match>("/api/matches", signal);
  }

  getMatch(matchId: string, signal?: AbortSignal): Promise<Match> {
    return this.request<Match>(`/api/matches/${encodeURIComponent(matchId)}`, signal);
  }

  getPlayers(matchId: string, signal?: AbortSignal): Promise<Player[]> {
    return this.listAll<Player>(`/api/matches/${encodeURIComponent(matchId)}/players`, signal);
  }

  getRallies(matchId: string, signal?: AbortSignal): Promise<DomainRecord[]> {
    return this.listAll<DomainRecord>(`/api/matches/${encodeURIComponent(matchId)}/rallies`, signal);
  }

  getShots(matchId: string, signal?: AbortSignal): Promise<DomainRecord[]> {
    return this.listAll<DomainRecord>(`/api/matches/${encodeURIComponent(matchId)}/shots`, signal);
  }

  getContacts(matchId: string, signal?: AbortSignal): Promise<DomainRecord[]> {
    return this.listAll<DomainRecord>(`/api/matches/${encodeURIComponent(matchId)}/contacts`, signal);
  }

  getBounces(matchId: string, signal?: AbortSignal): Promise<DomainRecord[]> {
    return this.listAll<DomainRecord>(`/api/matches/${encodeURIComponent(matchId)}/bounces`, signal);
  }

  getArtifacts(matchId: string, signal?: AbortSignal): Promise<Artifact[]> {
    return this.listAll<Artifact>(`/api/matches/${encodeURIComponent(matchId)}/artifacts`, signal);
  }

  async getAnalytics(matchId: string, signal?: AbortSignal): Promise<Analytics | null> {
    try {
      return await this.request<Analytics>(
        `/api/matches/${encodeURIComponent(matchId)}/analytics`,
        signal,
      );
    } catch (error) {
      if (error instanceof ApiClientError && error.status === 404) {
        return null;
      }
      throw error;
    }
  }

  async getMatchDashboard(matchId: string, signal?: AbortSignal): Promise<MatchDashboardData> {
    const [match, players, rallies, shots, contacts, bounces, analytics, artifacts] =
      await Promise.all([
        this.getMatch(matchId, signal),
        this.getPlayers(matchId, signal),
        this.getRallies(matchId, signal),
        this.getShots(matchId, signal),
        this.getContacts(matchId, signal),
        this.getBounces(matchId, signal),
        this.getAnalytics(matchId, signal),
        this.getArtifacts(matchId, signal),
      ]);
    return { match, players, rallies, shots, contacts, bounces, analytics, artifacts };
  }
}

export const apiClient = new ApiClient();
