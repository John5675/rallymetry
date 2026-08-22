# Temporal shot-model foundation

Milestone 24A adds a trainable temporal representation layer and a corrected,
auditable shot-label contract. It does not claim that the current eight-video
corpus can train or evaluate a multiclass pickleball shot classifier.

## Why the taxonomy is multi-axis

The legacy `shotType` field is retained for compatibility, but it mixes rally
phase, contact mechanics, and tactical intent. These meanings can overlap. For
example, a return can also be a forehand groundstroke and a drive. Version 1 of the
new taxonomy therefore stores:

| Axis | Values |
| --- | --- |
| `phase` | `SERVE`, `RETURN`, `RALLY`, `UNKNOWN` |
| `contactMode` | `GROUNDSTROKE`, `VOLLEY`, `OVERHEAD`, `UNKNOWN` |
| `strokeSide` | `FOREHAND`, `BACKHAND`, `TWO_HANDED_BACKHAND`, `UNKNOWN` |
| `intent` | `DINK`, `DROP`, `DRIVE`, `LOB`, `RESET`, `SPEEDUP`, `OTHER`, `UNKNOWN` |

Legacy labels are translated conservatively. A legacy `SERVE` establishes phase,
but does not fabricate forehand, groundstroke, or tactical-intent labels. A legacy
`DINK` establishes intent, but does not establish stroke side or whether the ball
was volleyed. The backwards-compatible projection follows the precedence `SERVE`,
`RETURN`, `OVERHEAD`, `DINK`, `DROP`, `DRIVE`, `VOLLEY`, `OTHER`, `UNKNOWN` and
returns `UNKNOWN` when the axes do not justify a legacy class.

The definitions are informed by USA Pickleball's
[basic terms](https://usapickleball.org/pickleball-skills/level-one/pickleball-basic-terms/),
[reset guidance](https://usapickleball.org/pickleball-skills/level-three/what-is-a-pickleball-reset-shot-and-how-to-hit-it/),
and [speed-up guidance](https://usapickleball.org/strategies/what-is-a-speedup-and-when-to-use-a-speedup/).
Definitions guide annotation; they are not training footage.

## Prediction and abstention contract

Every learned axis may expose:

- `bestGuess`: the highest-probability non-`UNKNOWN` value for review or product
  display;
- `confidence` and ranked `alternatives`;
- `authoritative`: the best guess only when it passes that axis's calibrated
  threshold, otherwise `UNKNOWN`;
- `abstained`: whether the authoritative value is `UNKNOWN`.

This distinction permits a best guess for every detected contact without claiming
zero uncertainty. Analytics must consume the authoritative value or a verified
human correction, not the best guess.

## Corrected eight-video dataset

Build a separate corrected copy and audit from `services/vision`:

```bash
uv run pickleball-vision shot-model build-dataset \
  ../../output/ai-review-8videos/multievent/dataset/multievent-ai-adjudicated-v1.json \
  --corrections ../../ml/datasets/shot-classification/eight-video-corrections.json \
  --output-dir ../../output/shot-model-eight-video-audit
```

The command never overwrites the source. A correction has a stable ID, exact target,
expected old value, replacement, reviewer, and evidence. Expected-value checks make
the correction fail if the source has silently changed. Video 8 contacts `V8-C0001`
and `V8-C0005`, plus their linked observations, correct `DENNY/PARTNER` to
`DIANA/OPPONENT_2`. This is a user-confirmed identity correction only. It does not
turn the AI-pseudo-labeled serve into human-accepted shot ground truth.

The audit enforces:

- disjoint whole-video train/validation/test splits;
- a fixed source-dataset hash for every correction layer;
- human-accepted semantic labels in validation and test;
- configurable minimum train and held-out examples for every claimed axis class;
- explicit blockers instead of a misleading accuracy number.

The current corpus correctly fails the semantic-training gate: it has 55 contacts,
zero human-accepted semantic labels, only serves in held-out data, no contact-mode or
stroke-side labels, and only four train-side dinks.

## Licensed representation pretraining

[RacketVision](https://huggingface.co/datasets/linfeng302/RacketVision) is MIT
licensed and supplies fixed-split ball trajectories and five-point racket poses for
badminton, table tennis, and tennis. It has no pickleball shot semantics. The local
adapter consumes only paired:

- `<sport>/interp_ball/<match>/<rally>/results.csv`;
- `<sport>/merged_racket/<match>/<rally>/result.json`;
- `<sport>/info/{train,val,test}.json`;
- the dataset card and immutable upstream revision.

The adapter rejects `.pkl` and `.pickle`. Pickle can execute code when loaded, and
the release's prebuilt pickle windows are unnecessary. It records the MIT license,
upstream revision, split counts, and a combined content hash.

Pretrain the temporal encoder:

```bash
uv run pickleball-vision shot-model pretrain-representation \
  --config ../../ml/training/shot-representation-pretrain.example.json \
  --output-dir ../../ml/experiments/racketvision-temporal-gru-v1
```

The initial backend is a small GRU trained to predict five future normalized
ball/racket feature frames from twenty history frames. Features retain ball
visibility/confidence and up to two rackets' normalized keypoints and scores. The
train and validation samplers honor RacketVision's fixed sequence splits; its test
split is not used during training.

`TorchMultiHeadTemporalShotModel` can load the encoder through PyTorch's restricted
`weights_only` mode and fuse its temporal state with explicit structured context:
shot index, contact/hitter confidence, hitter court position availability, distance
from the kitchen, trajectory coverage, incoming-bounce state, and landing position
availability. Independent linear heads emit phase, contact mode, stroke side, and
intent logits. The adapter is trainable, but Milestone 24A deliberately does not fit
those semantic heads while the dataset gate is false.

Outputs retain `experiment.json`, `metrics.json`, and ignored
`representation.pt`. Metrics are representation losses—not semantic accuracy. The
experiment explicitly records `semanticShotClassifierTrained=false`,
`pickleballSemanticLabelsConsumed=false`, and that the authoritative unknown policy
did not change.

Research supports the feature direction: the
[BST stroke classifier](https://arxiv.org/abs/2502.21085) combines pose, trajectory,
and player position; trajectory-only and pose-only racket-sport studies also show
that temporal evidence is material. The representation is therefore intended to be
combined later with contact-centered player pose/crop motion, primary-match ball
trajectory, bottom-center court position, bounce/landing, and rally order. Audio may
support contact timing but never determine shot type by itself.

## What is required before semantic-head training

1. Human-review contact-centered clips from all eight videos.
2. Correct hitter identity and bottom-center court position before labeling shots.
3. Mark pass-backs, dead-ball feeds, and neighboring-court activity as non-rally.
4. Label all independent axes and retain uncertainty.
5. Ensure every class being claimed has representative, human-accepted support in
   validation and test.
6. Keep Video 8 untouched while tuning; evaluate it only after the class-support
   and human-acceptance gates pass.

Until then, the Milestone 17 rules remain the production baseline/fallback and
`UNKNOWN` remains valid.

## Hosted AI-review overlay

The second-pass visual review of all 55 candidate contacts is packaged as
`pickleball_vision/resources/eight_video_ai_review_v1.json`. It contains no media,
credentials, human-ground-truth claims, or absolute source paths. Each reviewed
video is identified by its full source SHA-256 hash.

The Render pipeline runs:

```bash
pickleball-vision shot-model apply-review <video> \
  --shots shots.json \
  --output reviewed-shots.json
```

The command attaches `aiVisualReview` only when the uploaded video bytes exactly
match one of the eight reviewed sources and a reviewed contact is within the
configured timestamp tolerance. It preserves the machine `shotType`, hitter,
confidence, and classification evidence. A nonmatching source receives overlay
metadata with `matchedSource=false` and no record-level review. The dashboard may
show the AI visual-review best guess as a separate line, but it is not a human
correction and deterministic analytics continue to use the original reconstructed
shots until a verified human correction exists.
