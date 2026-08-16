# Multimodal Match Annotation

Milestone 12 provides a local, human-operated editor for creating versioned match
ground truth. It does not infer rallies, contacts, bounces, winners, or shot types.
Every saved event is an explicit human annotation.

## Start or resume an annotation session

From `services/vision`:

```bash
uv run pickleball-vision annotate-match /absolute/path/to/match.mp4 \
  --output ../../output/match-annotations.json
```

When Prompt 10 audio-analysis artifacts are available, add their raw event file as
optional context:

```bash
uv run pickleball-vision annotate-match /absolute/path/to/match.mp4 \
  --output ../../output/match-annotations.json \
  --audio-events ../../output/audio-analysis/audio-events.json
```

The command opens a loopback-only browser editor. Pass `--no-open` to print the
local URL without opening a browser, or `--port 0` to select a free port. Stop the
server with `Ctrl+C` or the editor's **Save & close** button.

Reusing the same output path resumes a compatible annotation file. The editor
validates the source by content hash and media properties, derives timestamps from
the selected frame, and atomically replaces the JSON after each edit. It never
modifies the source recording or raw audio-analysis artifact.

## Editor controls

The editor uses the source video's synchronized audio during normal playback. Its
timeline can optionally show the waveform and generic Prompt 10 `TRANSIENT`
markers. Those markers are non-semantic context and are never copied into match
events automatically.

Keyboard shortcuts:

| Key | Action |
| --- | --- |
| `Space` | Play or pause |
| `,` / `.` | Previous / next exact frame |
| `J` / `L` | Seek backward / forward five seconds |
| `1`–`7` | Select an event type |
| `A` | Add the form as an event at the current frame |
| `E` | Update the selected event |
| `Delete` | Delete the selected event after confirmation |

Click an event row or timeline marker to select it, seek to its frame, and load it
into the form. The **Clear selection** action returns the form to add mode.

## Versioned annotation contract

The root document records `annotationVersion`, source video identity and metadata,
the canonical media-timeline mapping, optional audio-context provenance, and an
ordered `events` array. A representative event is:

```json
{
  "id": "match-event-0000001",
  "type": "PADDLE_CONTACT",
  "frame": 912,
  "videoTimestampSeconds": 30.4,
  "mediaTimestampSeconds": 30.4,
  "playerId": "ME",
  "team": "MY_TEAM",
  "shotType": null,
  "courtPosition": {
    "xMeters": 2.8,
    "yMeters": 4.1,
    "coordinateSystem": "canonical_pickleball_court",
    "source": "HUMAN_ANNOTATION"
  },
  "audioLabel": "PRIMARY_EVENT_AUDIBLE",
  "notes": "Visible paddle-ball contact; clean transient nearby.",
  "confidence": 0.95,
  "source": "HUMAN"
}
```

`playerId`, `team`, `shotType`, `courtPosition`, `audioLabel`, `notes`, and
`confidence` are optional. A missing optional value means unannotated or unknown;
it must not be interpreted as negative evidence. Court positions are human-entered
court-plane coordinates, not projections of airborne ball image points.

Event types:

- `RALLY_START`: the first frame at which the rally begins under the chosen review
  convention, normally the serve-contact frame.
- `RALLY_END`: the first frame at which the rally outcome is visually settled.
- `SERVE_CONTACT`: visible paddle-ball contact that initiates a serve.
- `PADDLE_CONTACT`: a non-serve paddle-ball contact.
- `BOUNCE`: the best-supported court-contact frame.
- `RALLY_WINNER`: the rally outcome; use optional team/player/notes to record the
  winner without guessing an unsupported identity.
- `SHOT_TYPE`: a human shot-class label. Put the class in `shotType`; unknown is
  preferable to guessing.

Optional audio labels:

- `PRIMARY_EVENT_AUDIBLE`
- `PRIMARY_EVENT_NOT_AUDIBLE`
- `OTHER_COURT_TRANSIENT`
- `AMBIGUOUS_AUDIO`

An audio label describes the annotator's assessment of audio around an already
reviewed video event. A transient cannot prove a bounce or contact, and nearby
courts can generate convincing unrelated sounds. Contradictory visual evidence
always wins. Annotation remains fully supported when audio is missing.

## Recommended 5–10-rally workflow

1. Choose rallies representing near/far action, occlusion, neighboring-court
   activity, and both clear and noisy audio. Do not select only easy rallies.
2. Make a first visual pass. Mark exact `RALLY_START` and `RALLY_END` frames, then
   add `SERVE_CONTACT`, `PADDLE_CONTACT`, and `BOUNCE` only where the frame evidence
   supports them. Frame-step around each candidate.
3. Add `RALLY_WINNER` and `SHOT_TYPE` metadata in a second pass. Leave player,
   winner, shot type, location, or confidence empty when the evidence is unclear.
4. If audio context is loaded, perform a separate audio pass. Compare the native
   playback, waveform, and transient markers, then add an optional audio label to
   human events. Never create a semantic event from a marker alone.
5. Reopen the same JSON and audit every event in timeline order. Check event pairs,
   exact frames, player/team consistency, confidence, and notes for ambiguous cases.
6. Double-review at least two difficult rallies and record disagreements in notes
   until an adjudicated event set is defensible. Preserve uncertainty rather than
   forcing complete labels.
