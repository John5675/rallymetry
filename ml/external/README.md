# External representation data

Third-party datasets are local-only and must not be committed. This directory
retains only these instructions; downloaded files are ignored by Git.

Milestone 24A may use the MIT-licensed
[RacketVision](https://huggingface.co/datasets/linfeng302/RacketVision) dataset for
ball/racket temporal representation pretraining. It is **not** pickleball shot-type
ground truth. Download only safe static annotations (`.json`, `.csv`, and the
dataset card); never load the repository's pickle files.

```bash
uvx --from huggingface-hub hf download \
  linfeng302/RacketVision \
  --repo-type dataset \
  --revision 85157ca21faa2abca96d837dd2b963738029bcc8 \
  --include README.md \
  --include "annotations/dataset_info.json" \
  --include "*/info/train.json" \
  --include "*/info/val.json" \
  --include "*/info/test.json" \
  --include "*/merged_racket/**/*.json" \
  --include "*/interp_ball/**/*.csv" \
  --local-dir ml/external/RacketVision
```

The shot-model tooling validates the downloaded revision, safe extensions, license,
and checksums before consuming any examples.
