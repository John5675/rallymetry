# Background analysis worker

## Boundary

Milestone 21 runs the existing local CLI pipeline in a standalone Python process.
FastAPI only creates `QUEUED` records. The worker opens outbound connections to
MongoDB Atlas and, when configured, Vercel Blob; it exposes no HTTP listener and
processes one match at a time.

MongoDB is sufficient for the expected six-user scale. One atomic
`find_one_and_update` operation both selects and claims a queue document. Every
later heartbeat, stage, completion, or failure write includes both `jobId` and
`workerId` in its filter. Reclaiming an expired lease changes `workerId`, so the
stale process immediately loses write authority.

## Job contract

`processing_jobs` uses these states:

```text
QUEUED -> CLAIMED
       -> PLAYER_PROCESSING -> BALL_PROCESSING -> AUDIO_PROCESSING
       -> RALLY_PROCESSING -> EVENT_PROCESSING -> SHOT_PROCESSING
       -> ANALYTICS -> PUBLISHING -> COMPLETE
                                   \-> FAILED
```

Optional stages can be absent from a configured plan, but progress must increase.
The record retains claim/start/heartbeat/completion times, owner, progress, attempt
count, safe error code/message, pipeline version, source reference, and result
artifact IDs. An explicit pipeline failure is terminal. A crashed worker is detected
from an expired heartbeat and may be reclaimed until `maxAttempts`; an expired job
at that limit becomes `FAILED` with `WORKER_LEASE_EXHAUSTED`.

## Source media

- `LOCAL_PATH` accepts a validated direct file path or a local `SOURCE_MEDIA`
  artifact reference. Direct files are opened in place and never modified.
- `BLOB` requires a private Vercel Blob `SOURCE_MEDIA` artifact manifest. The worker
  downloads it into a per-job temporary directory.
- YouTube URLs are not accepted or downloaded.

The API-created job derives its source type from the match's `sourceArtifactId`.
The artifact must exist and be categorized as `SOURCE_MEDIA`; otherwise process
submission returns `409`.

## Trusted pipeline plan

Commands are not read from MongoDB. The worker loads an operator-controlled JSON
file from `--pipeline-plan` or `PICKLEBALL_VISION_WORKER_PIPELINE_PLAN`. Each stage
invokes the installed `pickleball-vision` executable without a shell. Only the
existing inference/analysis commands allowed for that stage are accepted; ball
training commands are rejected.

The plan supports three substitutions in arguments:

- `{source}`: staged source-media path
- `{workspace}`: isolated temporary job directory
- `{matchId}`: persisted match ID

`structuredResults` maps existing JSON output arrays into the separate `rallies`,
`contacts`, `bounces`, and `shots` collections. Analytics remain one compact
deterministic record. `artifacts` lists selected files and their explicit
`VIEWABLE_MEDIA` or `INTERNAL_ARTIFACT` policy. Raw detections and frame-level data
remain artifacts rather than large MongoDB documents.

The checked-in friend-view publication policy marks review videos and optional
heatmaps `PUBLIC` `VIEWABLE_MEDIA`. It marks ball tracks, raw audio observations,
and analytics artifacts `PRIVATE` `INTERNAL_ARTIFACT`. Public publication requires
the separate `PUBLIC_BLOB_READ_WRITE_TOKEN`; it never falls back to weakening the
private store.

The checked-in [example plan](examples/worker-pipeline-plan.json) shows the complete
stage chain. Its `/opt/pickleball-vision/...` calibration, assignment, configuration,
and weights paths are deliberate placeholders: operators must provide reviewed
match-specific inputs rather than having the worker invent them.

## Configuration and operation

Required hosted settings:

```bash
export MONGODB_URL='mongodb+srv://<user>:<password>@<cluster>/'
export MONGODB_DATABASE='pickleball_vision'
export PICKLEBALL_VISION_ARTIFACT_BACKEND='vercel_blob'
export BLOB_READ_WRITE_TOKEN='<server-side-token>'
export PUBLIC_BLOB_READ_WRITE_TOKEN='<separate-public-store-server-token>'
export PICKLEBALL_VISION_WORKER_PIPELINE_PLAN="$PWD/../../docs/examples/worker-pipeline-plan.json"
```

Optional worker settings:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PICKLEBALL_VISION_WORKER_ID` | generated host/process ID | Lease owner identifier |
| `PICKLEBALL_VISION_WORKER_WORK_ROOT` | `output/worker` | Isolated temporary work root |
| `PICKLEBALL_VISION_WORKER_POLL_INTERVAL_SECONDS` | `5` | Empty-queue polling interval |
| `PICKLEBALL_VISION_WORKER_LEASE_TIMEOUT_SECONDS` | `180` | Stale heartbeat age |
| `PICKLEBALL_VISION_WORKER_HEARTBEAT_INTERVAL_SECONDS` | `20` | Lease heartbeat interval |
| `PICKLEBALL_VISION_WORKER_MAX_ATTEMPTS` | `3` | Bounded crash-recovery attempts |

From `services/vision`, validate one claim and exit:

```bash
uv run pickleball-vision worker \
  --pipeline-plan ../../docs/examples/worker-pipeline-plan.json \
  --once
```

Run continuous polling on the developer workstation:

```bash
uv run pickleball-vision worker \
  --pipeline-plan ../../docs/examples/worker-pipeline-plan.json
```

Only one job is executed at a time in either mode. Multiple worker processes are
safe from double-claiming, but initial deployment should run one process because
that is sufficient for the expected load.
