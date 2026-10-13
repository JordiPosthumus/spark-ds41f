# Staged token-aware suppression trial

Ring1 uses the owner-authorized corrected temporary phrase guard described in
[the experiment](../experiments/2026-10-13-token-aware-suppression/README.md).
It fixes alternative-tokenization escapes, carries the generated prefix into the
native sampler, and expires at the precise sixty-word/control boundary.

The original image, model, GPU kernels, DSpark, cache, context, output and request
capacity remain unchanged. The existing model-runner mount is replaced by its
exact prior contents plus one request-cleanup hook. Additional sampler/parameter
sources and hash-checked tables are mounted read-only. The native patch and
before/after source hashes are recorded beside the experiment; the image build
pins stay unchanged because this trial uses explicit source overlays.

Twenty-five isolated CUDA cases and21 API CPU tests passed. These are not proof
of natural loop recovery or production stability. Ring2 remains on its previous
recipe until the deployed Ring1 passes chat/output checks and a full hour of
stable serving. Its accepted M64 changes must be retained when rolling forward.

Rollback is the exact previous Ring1 commit28cfb890e42ddb51be00d4be1e2022f12dede79c,
after drain/native idle and coordinated three-node stop/sync/start. No cache or
model deletion is part of this change.
