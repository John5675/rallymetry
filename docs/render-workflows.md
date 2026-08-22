# On-demand match analysis with Render Workflows

## Why this execution model

FastAPI must return quickly and must not run a multi-stage video pipeline inside an
HTTP request. Rallymetry therefore uses Render Workflows for task queuing and
temporary compute. MongoDB records the application job and its domain stage, but no
Rallymetry process polls MongoDB for work. When there is no task run, no workflow
compute remains active.

One match follows this path:

```text
POST /api/matches/{matchId}/process
  -> MongoDB processing_jobs CREATED
  -> RenderAsync.workflows.start_task(...)
  -> MongoDB processing_jobs QUEUED + renderTaskRunId
  -> Render provisions one task instance
  -> analyze_match downloads private inputs and runs the existing CLI stages
  -> structured results to MongoDB; allow-listed media to Vercel Blob
  -> processing_jobs COMPLETE
  -> temporary files removed; Render deprovisions the instance
```

The task never commits output, modifies the React application, or triggers a Vercel
deployment. The existing website reads new MongoDB/Blob references through FastAPI.

For the friend-facing path, `POST /api/matches/import-youtube` validates one video
URL, creates a match using the configured analysis profile, creates the job, and
starts the same task in a single short request. FastAPI never downloads the media.

## Code boundaries

- `pickleball_vision.api.services.render_workflows` wraps the official async Render
  SDK. Routes never call SDK methods directly.
- `pickleball_vision.workflows.app` registers `analyze_match` and calls `app.start()`.
- `pickleball_vision.workflows.orchestration` coordinates existing pipeline stages,
  persistence, publication, errors, and cleanup.
- `pickleball_vision.analysis_runtime.pipeline` is a reusable trusted CLI-plan adapter;
  it contains no poller, queue claim, heartbeat, or lease behavior.
- `docs/examples/render-workflow-pipeline-plan.json` is the operator-controlled stage
  plan. Task inputs cannot inject commands.

The configured task identifier has Render's required form:

```text
{workflow-slug}/{task-name}
rallymetry-analysis/analyze_match
```

`RENDER_WORKFLOW_TASK` supplies that identifier to FastAPI. The task input is only:

```json
{"job_id": "job_...", "match_id": "match_..."}
```

## Match prerequisites and one-click profile

An existing match can reference a private Vercel Blob `SOURCE_MEDIA` artifact. A
one-click submission instead persists a canonical `youtubeVideoId`; Render downloads
that one accessible recording only after the asynchronous task starts. Both paths
require this `analysisSetup` object:

```json
{
  "calibrationArtifactId": "artifact_...",
  "playerAssignmentsArtifactId": "artifact_...",
  "ballExperimentArtifactId": "artifact_...",
  "ballWeightsArtifactId": "artifact_..."
}
```

Every setup reference must belong to the same match and be a private hosted
`INTERNAL_ARTIFACT`. `DEFAULT_ANALYSIS_PROFILE_MATCH_ID` selects an explicitly
configured existing match. The import endpoint creates new match-scoped artifact
references plus an explicit `analysisProfileMatchId` owner reference to the same
immutable private setup objects; it never duplicates or modifies the profile match.
This is appropriate only for recordings compatible with that court, camera,
player-role, and model profile.

The workflow downloads setup plus either the private Blob source or the YouTube
source into the current job's temporary `input` directory. It preserves audio,
limits YouTube duration/size with `YOUTUBE_MAX_DURATION_SECONDS` and
`YOUTUBE_MAX_BYTES`, never accepts playlists/channels, and removes the downloaded
copy during normal job cleanup. The original recording remains unchanged.

Generated MP4 artifacts use browser-compatible H.264 through bundled FFmpeg. A
trusted pipeline plan may declare file-only `cleanupPaths` for stage intermediates
that are neither downstream inputs nor publication results. Those files are removed
after their stage succeeds so multiple full-resolution debug videos do not exhaust
the workflow's temporary disk. Local CLI commands still produce every documented
artifact by default.

## Job state and duplicate protection

Application states are `CREATED`, `QUEUED`, `STARTING`, `DOWNLOADING_MEDIA`,
`PREPARING_MEDIA`, player/ball/audio/rally/bounce/contact/hitter/shot processing,
`ANALYTICS`, `RENDERING_ARTIFACTS`, `UPLOADING_RESULTS`, `COMPLETE`, and `FAILED`.
Cancellation-ready enum values are retained, but no cancellation endpoint exists yet.

A partial unique MongoDB index on `processing_jobs.matchId` where `active=true`
atomically prevents two active application jobs for one match. A repeated Analyze
request returns the existing active job and does not call Render again. A future
Reanalyze action must be explicit.

`processingRunId` is generated once with the application job and reused by every
Render attempt. Structured event IDs are upserted at their stable match-scoped IDs
and retain `processingRunId`. Published artifacts are looked up by match, run, and
artifact type before upload, so a retry reuses an existing object. A provider upload
that succeeds immediately before MongoDB metadata fails can leave an unreferenced
Blob; that object is never presented and can be removed by a later maintenance audit.

MongoDB is the frontend's source of truth for domain progress. `renderTaskRunId`
links to Render's infrastructure run for debugging; FastAPI does not continuously
mirror Render status.

## Compute, timeout, retry, and device

The initial task decorator selects Render `pro`: 2 CPU and 4 GB RAM. This is an
explicit starting point for 1080p CV and video rendering, not a claim that every
match is optimally sized. Benchmark representative recordings and move up or down
only from measured peak memory and elapsed time.

The timeout is 21,600 seconds (six hours). Local 1080p work has historically taken
tens of minutes, while a CPU-only hosted run can be substantially slower; six hours
allows download, all inference stages, rendering, and upload while staying bounded
well below Render's 24-hour maximum.

The SDK retry policy allows one retry after 30 seconds with 2x backoff. Persistence
and artifact-store failures are treated as transient. Missing/corrupt media, missing
setup, invalid models, and pipeline/configuration failures are recorded as permanent
application failures and returned without raising, preventing an expensive automatic
rerun. After the allowed transient retry is exhausted, the job becomes `FAILED`.

`MODEL_DEVICE=cpu` is the default and the compact result summary records `CPU`.
Change it only when the deployed environment demonstrably provides and uses another
device. The workflow never claims GPU acceleration from instance naming alone.

## Temporary and published artifacts

Each run uses only:

```text
/tmp/rallymetry/<job-id>/
  input/
  working/
  output/
```

Cleanup is in `finally` and is constrained to that child path. Persisted Blob objects
are never deleted by scratch cleanup.

Only `VIEWABLE_MEDIA` with an explicit artifact-type allow-list is published. This
includes annotated/top-down videos, thumbnails, heatmaps, court visualizations, and
shot maps. Raw detections, extracted PCM audio, large intermediate JSON, debug logs,
training data, and model internals remain temporary/private. A job becomes `COMPLETE`
only after structured writes, required uploads, and artifact metadata writes finish.

## Environment

FastAPI requires:

```text
MONGODB_URL
MONGODB_DATABASE
RENDER_API_KEY
RENDER_WORKFLOW_TASK
CORS_ORIGINS
DEFAULT_ANALYSIS_PROFILE_MATCH_ID
```

The workflow service requires:

```text
MONGODB_URL
MONGODB_DATABASE
PICKLEBALL_VISION_ARTIFACT_BACKEND=vercel_blob
BLOB_READ_WRITE_TOKEN
PUBLIC_BLOB_READ_WRITE_TOKEN
PIPELINE_CONFIG
MODEL_DEVICE=cpu
WORKFLOW_TEMP_DIR=/tmp/rallymetry
RENDER_WORKFLOW_PLAN=pro
RENDER_WORKFLOW_TIMEOUT_SECONDS=21600
YOUTUBE_MAX_DURATION_SECONDS=7200
YOUTUBE_MAX_BYTES=4000000000
YOUTUBE_POT_PROVIDER_URL=http://<private-provider-host>:4416
```

`YOUTUBE_POT_PROVIDER_URL` points to a private, pinned
`bgutil-ytdlp-pot-provider` service. The workflow supplies its generated proof-of-origin
tokens to yt-dlp so public and unlisted link submissions do not depend on an end user's
YouTube cookies. Keep the provider private: it is workflow infrastructure, not a browser
API. A provider can improve cloud-IP acceptance but cannot override a video's actual
privacy, geographic, age, or account restrictions.

`RENDER_API_KEY` is a FastAPI trigger credential. The task does not need it unless a
future task calls Render APIs itself. No Render, MongoDB, or Blob credential belongs
in `VITE_*` or browser-visible data.

## Local workflow development

From `services/vision`, install dependencies and start Render's local task server:

```bash
uv sync --locked --extra dev
export MONGODB_URL='mongodb://...'
export MONGODB_DATABASE='rallymetry'
export PICKLEBALL_VISION_ARTIFACT_BACKEND='vercel_blob'
export BLOB_READ_WRITE_TOKEN='...'
export PUBLIC_BLOB_READ_WRITE_TOKEN='...'
export PIPELINE_CONFIG="$PWD/../../docs/examples/render-workflow-pipeline-plan.json"
export MODEL_DEVICE='cpu'
export WORKFLOW_TEMP_DIR='/tmp/rallymetry'
render workflows dev -- uv run python -m pickleball_vision.workflows.app
```

In a second terminal:

```bash
render workflows tasks list --local
render workflows start analyze_match \
  --input '{"job_id":"job_...","match_id":"match_..."}' \
  --local
```

To have a local FastAPI instance target the local task server, export
`RENDER_USE_LOCAL_DEV=true` according to the Render CLI contract and configure
`RENDER_WORKFLOW_TASK` with the locally registered slug.

## Render setup and deployment

1. Push the repository revision containing the workflow task.
2. In Render Dashboard choose **New > Workflow** and connect the repository. Render
   Workflows are configured separately from the existing FastAPI web service.
3. Set root directory to `services/vision`.
4. Set build command to `uv sync --locked`.
5. Set start command to `uv run python -m pickleball_vision.workflows.app`.
6. Configure the workflow environment listed above. Set `PIPELINE_CONFIG` to
   `/opt/render/project/src/docs/examples/render-workflow-pipeline-plan.json` if the
   workflow root still exposes the repository root; otherwise copy/use the absolute
   path shown by the build logs.
7. Confirm the task page shows `analyze_match`, plan `pro`, timeout six hours, and
   at most one retry. Copy its `{workflow-slug}/analyze_match` identifier.
8. Create a least-privilege Render API key and set it as `RENDER_API_KEY` only on the
   FastAPI service. Set the copied identifier as `RENDER_WORKFLOW_TASK`.
9. Redeploy FastAPI once for those environment changes. Per-match analysis never
   redeploys FastAPI or Vercel.

Inspect infrastructure state and detailed traces in Render Dashboard's workflow
**Runs** tab using `renderTaskRunId`. Inspect Rallymetry stage/error state through
`GET /api/jobs/{jobId}`. Safe MongoDB messages omit credentials; full tracebacks stay
in Render logs.

Hosted setup files remain immutable Blob artifacts. The workflow rewrites only its
temporary calibration copy so `source.video_path` names the staged media file, while
retaining the original pathname in `runtime_binding`. Player tracking uses
`--portable-profile` to match enriched manual image anchors against the fresh immutable
person detections for that run. Width, height, FPS, court side, anchor frame, and a
bounded image-distance gate must still agree; failure remains explicit instead of
silently assigning a nearby person. A new camera angle or materially different player
setup still requires a new reviewed calibration/assignment profile.

## Smoke test

1. Confirm `/health` reports MongoDB ready.
2. Confirm the profile match and its four required private setup artifacts exist.
3. Call `POST /api/matches/import-youtube` with `youtubeUrl` (and optional `title`),
   or call `/api/matches/{matchId}/process` for a preconfigured match; verify an
   immediate `202` response.
4. Verify the response/job includes `status=QUEUED`, `processingRunId`, and
   `renderTaskRunId`.
5. Refresh `GET /api/jobs/{jobId}` and observe meaningful stage changes.
6. Confirm one Render run exists, the workspace is removed on exit, and the job does
   not become `COMPLETE` before uploads and metadata writes.
7. Fetch match rallies, shots, analytics, and artifacts through FastAPI and refresh
   the existing website. No Vercel redeployment should occur.
