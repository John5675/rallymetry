# Training

Keep small, versioned experiment configurations here. `ball-detector.example.json`
describes the completed ball-detector schema, while
`shot-representation-pretrain.example.json` describes Milestone 24A's licensed
external trajectory/racket representation pretraining. Run products and checkpoints
belong in ignored experiment/artifact directories, never Git.

Training configurations must pin a human dataset version, fixed split and annotation
manifests, model version, base model, resolution, seed, and deterministic setting.

Representation pretraining is not pickleball semantic supervision. A semantic shot
head may train only after its dataset audit passes whole-video leakage, human-accepted
held-out, and per-class support gates. External racket-sport data never changes those
gates.
