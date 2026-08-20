# Friend-viewable deployment

## Deployment boundary

Milestone 23 deploys only the React/Vite application to Vercel. FastAPI runs as a
small persistent Python service, and the heavy analysis worker remains a separate
outbound-only process. Neither FastAPI nor the CV/audio pipeline runs in Vercel
Functions.

```text
Friends
   |
   v
Vercel: React/Vite
   |
   v
Persistent FastAPI service
   |
   +---------------------> MongoDB Atlas

Developer PC / worker machine
   |
   v
MongoDB processing job -> CV + audio pipeline
   |                         |
   +-------------------------+
   |
   +--> MongoDB Atlas + private/public Vercel Blob
                               |
                               v
                         Website results
```

This topology is intended for approximately six users. It adds no Redis, Celery,
inbound worker listener, or browser credential. Authentication is still deferred,
so anyone who knows the deployment URL can open the dashboard and anyone who has a
public Blob URL can open that generated artifact.

## Environment contract

Copy the checked-in examples only as a list of names. Store real values in each
host's encrypted environment or secret manager; do not commit a populated file.

### Frontend (Vercel build environment)

| Variable | Example | Notes |
| --- | --- | --- |
| `VITE_API_BASE_URL` | `https://api.example.com` | Public HTTPS origin of FastAPI; exposed in the browser by design |

No MongoDB or Blob variable belongs in the frontend environment. Any `VITE_*`
value is public build output.

### FastAPI and worker

| Variable | Purpose |
| --- | --- |
| `MONGODB_URL` | Atlas connection string from the server/worker secret store |
| `MONGODB_DATABASE` | Dedicated database, normally `rallymetry` |
| `PICKLEBALL_VISION_ARTIFACT_BACKEND` | Set to `vercel_blob` for hosted artifacts |
| `BLOB_READ_WRITE_TOKEN` | Private Blob store used for source and internal artifacts |
| `PUBLIC_BLOB_READ_WRITE_TOKEN` | Separate public Blob store used only for explicit public viewable media |

FastAPI additionally reads `CORS_ORIGINS`. The worker additionally reads the
`PICKLEBALL_VISION_WORKER_*` variables listed in
[`services/vision/.env.example`](../services/vision/.env.example).

## 1. MongoDB Atlas

1. Create or select an Atlas cluster and a dedicated `rallymetry` database.
2. Create a least-privilege application database user. Do not reuse a credential
   belonging to another application.
3. Allow outbound connections from the FastAPI host and worker machine through
   Atlas Network Access. Avoid a global allow-list when fixed egress ranges are
   available.
4. Put the Atlas connection string in `MONGODB_URL` on FastAPI and the worker.
   Set `MONGODB_DATABASE=rallymetry` so unrelated databases and collections remain
   outside this application's scope.
5. Start FastAPI once; its lifespan initializes the Rallymetry collection indexes.
   A successful `/health` response with `databaseReady: true` confirms access.

MongoDB stores compact match, event, analytics, job, correction, and artifact
manifest records. It never stores MP4 bytes, raw frames, audio waveforms, model
weights, or huge frame-level arrays.

## 2. Vercel Blob stores

Blob access is a property of the store, not an individual object. Use two stores:

| Store | Application variable | Allowed content |
| --- | --- | --- |
| Private | `BLOB_READ_WRITE_TOKEN` | `SOURCE_MEDIA`, `INTERNAL_ARTIFACT`, private viewable drafts |
| Public | `PUBLIC_BLOB_READ_WRITE_TOKEN` | Only explicit `PUBLIC` + `VIEWABLE_MEDIA` outputs |

Create one private store and one public store in the Vercel dashboard under
Storage. When connecting the public store, set the advanced environment-variable
prefix to `PUBLIC_BLOB_`. The equivalent CLI connection for an existing public
store is:

```bash
npx --yes vercel@latest integration resource connect \
  rallymetry-viewable rallymetry \
  --prefix PUBLIC_BLOB_ \
  --yes
```

Copy each store's server-side read/write token into the corresponding FastAPI and
worker secret. The `PUBLIC_BLOB_` prefix creates the expected
`PUBLIC_BLOB_READ_WRITE_TOKEN`. Do not add either token to `apps/web`, any `VITE_*`
variable, browser JavaScript, or MongoDB.

The storage adapter uses a generated random path component and asks Blob for an
additional random suffix. Random names reduce accidental discovery but are not an
authorization mechanism. Anyone with a public URL can view that object. The worker
example plan deliberately marks annotated/top-down/debug videos and heatmaps
`PUBLIC`; ball tracks, audio observations, analytics artifacts, and source media
remain `PRIVATE`. Review that policy before every new artifact type.

When a match has `youtubeVideoId`, the dashboard embeds YouTube for normal source
playback. Do not upload a duplicate source copy to public Blob merely for playback.

## 3. FastAPI deployment

Use any provider that runs a persistent Docker container and provides outbound
HTTPS access to Atlas and Blob. Do not deploy this image as a Vercel Function.

Build and test the vendor-neutral image from the repository root:

```bash
docker build \
  --file services/vision/Dockerfile \
  --tag rallymetry-api:0.1 \
  services/vision

docker run --rm \
  --publish 8000:8000 \
  --env-file services/vision/.env.production \
  rallymetry-api:0.1
```

The image startup command is:

```bash
uvicorn pickleball_vision.api.main:app --host 0.0.0.0 --port "$PORT"
```

Configure the host to check `GET /health`. The container health check verifies
process liveness. The JSON response separately reports `databaseReady`; production
is ready only when it is `true`. Terminate TLS at the host so the public API origin
uses HTTPS.

Set exact frontend origins on FastAPI, for example:

```bash
export CORS_ORIGINS='http://localhost:5173,https://rallymetry.vercel.app,https://matches.example.com'
```

Replace the example production names with the real Vercel and custom domains.
Preview domains must be explicitly added if they need API access. Do not use a
credentialed wildcard.

## 4. Vercel frontend deployment

The repository-root [`vercel.json`](../vercel.json) defines the monorepo contract:

| Setting | Value |
| --- | --- |
| Framework | Vite |
| Install command | `npm --prefix apps/web ci` |
| Build command | `npm --prefix apps/web run build` |
| Output directory | `apps/web/dist` |
| SPA fallback | rewrite every application route to `/index.html` |

In the Vercel project, keep the project root at the repository root. Add
`VITE_API_BASE_URL` to Production (and Preview when desired), using the public HTTPS
FastAPI origin. Environment changes require a new deployment.

Deploy from the repository root:

```bash
npx --yes vercel@latest link
npx --yes vercel@latest env add VITE_API_BASE_URL production
npx --yes vercel@latest --prod
```

The committed SPA rewrite makes direct navigation and refreshes at
`/matches/<matchId>` and `/matches/<matchId>/analysis` resolve to React. Static
assets continue to be served from `apps/web/dist`.

## 5. Worker startup

The worker can run on the developer PC. It needs outbound access only and uses the
same MongoDB database and both Blob tokens as FastAPI:

```bash
cd services/vision
set -a
source .env.production
set +a
uv sync --locked --extra dev
uv run pickleball-vision worker \
  --pipeline-plan ../../docs/examples/worker-pipeline-plan.json
```

Use `--once` for a single queued job smoke test. Before continuous use, replace the
example plan's calibration, assignment, model configuration, and weights paths with
reviewed files available on that worker. The worker remains single-concurrency by
design.

## 6. Smoke test

Run these checks after the API and frontend are deployed:

1. Verify FastAPI and MongoDB readiness:

   ```bash
   curl --fail --silent https://api.example.com/health
   ```

   Confirm `status` is `ok` and `databaseReady` is `true`.

2. Verify production CORS using the exact frontend origin:

   ```bash
   curl --include --request OPTIONS https://api.example.com/api/matches \
     --header 'Origin: https://rallymetry.vercel.app' \
     --header 'Access-Control-Request-Method: GET'
   ```

   Confirm `access-control-allow-origin` equals the requesting frontend origin.

3. Open the Vercel deployment, then directly refresh `/matches` and one
   `/matches/<matchId>/analysis` route. Confirm neither returns a Vercel 404.
4. Inspect browser network requests. They must target `VITE_API_BASE_URL`, return
   JSON-friendly IDs, and contain no MongoDB URL or Blob write token.
5. Open a match with `youtubeVideoId`; confirm normal playback uses YouTube.
6. Open a generated annotated video or heatmap. Its manifest must be
   `VIEWABLE_MEDIA` + `PUBLIC`, and its randomized public URL must load without a
   token. Confirm a private source/internal URL is not rendered by the dashboard.
7. Queue one job through `POST /api/matches/<matchId>/process`, start the worker with
   `--once`, and confirm the job reaches `COMPLETE`, structured results appear in
   Atlas, and selected public generated artifacts appear in the dashboard.

If the frontend shows no matches, check `/health`, `VITE_API_BASE_URL`, production
CORS, the selected `MONGODB_DATABASE`, and the browser network error before changing
the analysis pipeline.
