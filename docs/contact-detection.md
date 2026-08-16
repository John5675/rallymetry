# Multimodal Paddle-Contact Detection

Milestone 15 derives inspectable paddle-contact candidates from the structured
primary-ball trajectory and logical-player tracks. Visual evidence is mandatory.
Optional generic audio transients can increase confidence but can never create a
candidate or prove that a sound came from the primary court.

## Run the stage

From `services/vision`:

```bash
uv run pickleball-vision detect-contacts /absolute/path/to/match.mp4 \
  --ball-tracks ../../output/ball-tracking/ball_tracks.json \
  --player-tracks ../../output/player-tracking/tracks.json \
  --rallies ../../output/rally-segmentation/rallies.json \
  --bounces ../../output/bounce-detection/bounces.json \
  --audio-events ../../output/audio-analysis/audio-events.json \
  --annotations ../../output/match-annotations.json \
  --output-dir ../../output/contact-detection
```

`--ball-tracks`, `--player-tracks`, and `--output-dir` are required. Rally,
bounce, audio, and annotation artifacts are optional. Omit `--audio-events` for a
vision-only run. The command writes:

- `contacts.json`: every retained visual candidate, including low-confidence ones;
- `contact-debug.mp4`: source-space ball and candidate-player evidence; and
- `contact-evaluation.json`: visual-only versus visual-plus-audio evaluation, or an
  explicit unavailable state when no human contacts were supplied.

Existing artifacts are not overwritten. Use a different output directory for each
configuration experiment.

## Visual-first inference

For each known ball point, the detector fits image-space velocity before and after
the frame within one trajectory segment. A visual candidate requires:

- enough known trajectory points on both sides;
- a minimum abrupt velocity change;
- either a direction discontinuity or a substantial speed-ratio change; and
- sufficient same-segment continuity.

Confidence separately retains velocity change, direction/speed discontinuity,
observed-versus-interpolated support, continuity, center-point provenance,
logical-player proximity, player court context, optional rally support, and
optional previous-bounce sequence support. Nearby candidates are suppressed so one
discontinuity does not become several contacts.

This is an image-space heuristic. It does not reconstruct true 3D velocity and it
does not project the airborne ball through court homography.

## Candidate players are not hitters

Logical-player observations are linked to their raw tracker boxes. Ball proximity
uses distance to the box extent; it does not use the bounding-box center as a
physical court position. The separately retained player physical position remains
the bottom-center ground-contact estimate.

`candidatePlayers` ranks all available logical players by visual proximity and
retains tracking confidence/state, box geometry, bottom-center ground point, court
side, and court region. Ranking is evidence for the later hitter-identification
milestone. It is not a hitter decision:

- every player entry has `isAssignedHitter=false`;
- every contact has `assignedHitter=null`; and
- annotation player labels are not used during inference or temporal evaluation.

An airborne ball is never assigned a court side through homography. Court-side
support comes only from the tracked players' defensible ground-plane state.

## Rally, bounce, and prior-contact context

Optional predicted rallies can add a small confidence boost to an already visual
candidate but cannot create one. Optional accepted bounce candidates provide two
conservative signals: a coincident reversal can be excluded as a likely bounce, and
a preceding bounce can support a plausible sequence. Low-confidence bounce records
are not silently promoted.

Each retained contact also records the gap from the preceding contact candidate.
This preserves event-state context without classifying a hitter, shot, or rally
outcome.

## Audio fusion and timing

Generic audio candidates are remapped from analysis time using source stream start
times and the current run's configured offset:

```text
audioVideoTime = (audioStartTime or 0)
               + analysisTimestamp
               + audioVideoOffsetMs / 1000
               - (videoStartTime or 0)
```

At most one transient is matched to one visual candidate within
`fusionToleranceMs`. Timing proximity discounts raw transient confidence. A match
can increase `fusedConfidence` only on an existing visual candidate. Neighboring
courts, speech, footsteps, and other noise remain possible explanations.

Evidence modes are:

- `VISUAL_ONLY`: accepted without a matched transient;
- `VISUAL_PLUS_AUDIO`: accepted with synchronized audio support; and
- `LOW_CONFIDENCE`: retained visual evidence below the fused threshold.

## Evaluation

Human `SERVE_CONTACT` and `PADDLE_CONTACT` annotations are loaded only after
inference. The evaluator thresholds the identical visual candidate set using
`visualConfidence` and then `fusedConfidence`, performs one-to-one temporal
matching, and reports precision, recall, F1, and signed/absolute contact timing
error. Annotated player identity is retained for audit but is not used for matching
because hitter identification is not part of this milestone.

Normal annotation files may cover only representative rallies. Evaluation therefore
defaults to annotated rally intervals or small contact-centered windows. Use
`--annotations-complete` only after reviewing the complete recording. Record
`--evaluation-partition development|validation|test`; the command never tunes
thresholds automatically.

## Configuration

Settings use the `PICKLEBALL_VISION_` prefix:

- `CONTACT_TRAJECTORY_WINDOW_SECONDS`
- `CONTACT_MINIMUM_OBSERVATIONS_EACH_SIDE`
- `CONTACT_MINIMUM_VELOCITY_CHANGE_DIAGONALS_PER_SECOND`
- `CONTACT_MINIMUM_DIRECTION_CHANGE_DEGREES`
- `CONTACT_MINIMUM_SPEED_CHANGE_RATIO`
- `CONTACT_MINIMUM_CONTINUITY_FRACTION`
- `CONTACT_MAXIMUM_PLAYER_PROXIMITY_DIAGONAL_FRACTION`
- `CONTACT_MINIMUM_VISUAL_CANDIDATE_CONFIDENCE`
- `CONTACT_ACCEPTED_CONFIDENCE`
- `CONTACT_MINIMUM_BETWEEN_SECONDS`
- `CONTACT_BOUNCE_EXCLUSION_WINDOW_SECONDS`
- `CONTACT_MAXIMUM_PREVIOUS_BOUNCE_GAP_SECONDS`
- `CONTACT_AUDIO_CONFIDENCE_WEIGHT`
- `CONTACT_RALLY_SEQUENCE_CONFIDENCE_BOOST`
- `CONTACT_PREVIOUS_BOUNCE_CONFIDENCE_BOOST`
- `CONTACT_EVALUATION_TOLERANCE_MS`
- `CONTACT_SPARSE_EVALUATION_MARGIN_SECONDS`
- `AUDIO_VIDEO_OFFSET_MS`
- `FUSION_TOLERANCE_MS`

Every effective value is persisted. Tune on development data, freeze the values,
and compare unchanged validation/test results.
