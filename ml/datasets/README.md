# Datasets

Dataset content is local and ignored. Use the vision CLI to create versioned
`dataset-manifest.json` files with source hashes, frame/time provenance, clip/rally
groups, and positive/negative/unlabeled curation buckets. Split manifests assign
whole videos, clips, or groups; never individual neighboring frames.

Do not commit extracted images, review clips, private videos or URLs, large arrays,
or model weights. The exact commands and schemas are documented in the
[ball dataset tooling guide](../../docs/ball-dataset-tooling.md).
