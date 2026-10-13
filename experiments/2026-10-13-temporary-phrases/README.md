# Temporary phrase intervention experiment

Owner authorized installation on both rings on 2026-10-12 Bermuda time.
An earlier intervention complements the unchanged severe-collapse rewind/retry.
Trigger: at least eight occurrences of Hmm variants or whole let-me-run/go
sequences in each of two 200-word windows, evaluated 32 words apart. At least
160 words must be outside fenced code. Matching is case-insensitive. This is
an experimental repetition heuristic, not a semantic error classifier.

On a trigger, close/abort the previous native request before continuing its exact
rendered token prefix with fresh seed and native bad_words. For 60 completed
Unicode words, block Hmm/Hmmm/Hmmmm with period/comma and Let me run./Let me go.,
in lowercase/titlecase/uppercase. Native tokenization adds leading-space forms.
No individual ordinary words are blocked. Detection tolerates more variants than
the finite blocklist: exotic mixed case, punctuation and spacing can escape.

Remove bans at the first native output chunk after 60 words, or a reasoning/tool
boundary. A native chunk may overshoot that boundary; a few already-computed
answer/tool tokens may have been sampled with the temporary list. No tokens or
parser markers are replayed on this continuation. Already displayed text cannot
be retracted. A phrase already partially emitted before activation can finish.
Natural EOS/finish never causes a continuation. No attempt changes temperature,
top-p, context, or output capacity. All observed generation counts against the
original token budget; metadata and multimodal objects remain intact. Max ten
interventions per request; severe rewind/retries retain their separate ten limit.
This remains limited to the existing compatible streaming reasoning path.

Cost: two native abort/continuations per full intervention, reuse of eligible
prefix cache, and native bad-phrase masking while active. Requests without a
trigger only incur the bounded CPU detector. Native abort can discard a few
in-flight tokens whose compute cannot be observed. No zero-cost or speedup claim.

Validation: targeted offline regression tests and saved trace replay, followed
by native streaming/math/tool/cancellation checks before admission. The owner
explicitly requested a quick live experiment; the inherited full benchmark is
not relabelled or rerun as a prerequisite to this trial. Preserve all failed runs.
Rollback: redeploy the base.json recipe_commit through coordinated cluster
stop/sync/start, after draining and verifying idle. No caches/images deleted.
