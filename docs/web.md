# React/Vite match dashboard

Milestone 22 adds a strict TypeScript React application under `apps/web`. It is a
presentation client for the FastAPI application contract; it is not an analytics
engine, media downloader, worker, persistence adapter, or credential boundary.

## Routes

| Route | Purpose |
| --- | --- |
| `/` | Redirect to the match library |
| `/matches` | Match title, date, logical players, processing state, public thumbnail |
| `/matches/:matchId` | Overview, video/timeline, players, rallies, shots, and court maps |
| `/matches/:matchId/analysis` | Deterministic player, position, tactical, provenance, and landing analysis |

Vite owns the local development fallback for these client routes. Production SPA
rewrites and Vercel deployment are intentionally deferred to Milestone 23.

## API contract

The client reads the camelCase JSON schemas documented in [`api.md`](api.md). The
base origin is injected at build/development time:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

No production URL is hard-coded. The value is an HTTP origin only; do not place a
MongoDB URL, Blob token, or any other secret in a `VITE_*` variable because Vite
exposes those values to browser code.

The client loads match-scoped players, rallies, shots, contacts, bounces, analytics,
and artifact manifests. List endpoints are consumed through their documented
pagination envelope. FastAPI error envelopes become typed client errors. A missing
analytics record stays unavailable rather than being approximated in React.

## Media and timeline behavior

- When `youtubeVideoId` exists, the dashboard embeds the unlisted video. It never
  downloads or proxies it.
- A Vercel Blob artifact is rendered only when the API marks it `PUBLIC` and returns
  a URL. `PRIVATE` artifacts are described as unavailable to the browser.
- Rally starts/ends, contacts, bounces, and shots remain distinct structured event
  markers. Selecting a marker seeks native video or sends a YouTube player seek
  command when that playback surface permits it.
- Heatmaps, thumbnails, annotated videos, and top-down media are artifact references;
  the dashboard receives no Blob write token.

## Analytics and court maps

All match and player statistics are read from the persisted deterministic analytics
record. React may sort, filter, format, and visualize values, but it does not derive
rally counts, shot rates, distance, occupancy, spacing, or tactical metrics.

The landing map draws only `landingCourtPosition` values already present on
structured `Shot` records and within the documented 6.096 by 13.4112 meter court.
Missing and out-of-court points are omitted. Airborne ball observations are never
projected through the homography by the browser.

## Local development

Run FastAPI from `services/vision`:

```bash
export CORS_ORIGINS='http://localhost:5173'
uv run uvicorn pickleball_vision.api.main:app --host 127.0.0.1 --port 8000
```

Run the dashboard in another terminal:

```bash
cd apps/web
cp .env.example .env.local
npm ci
npm run dev
```

Verify the dashboard with:

```bash
npm run lint
npm run typecheck
npm test
npm run build
```

The tests use mocked HTTP responses and require no MongoDB, Vercel, private video,
or internet access.
