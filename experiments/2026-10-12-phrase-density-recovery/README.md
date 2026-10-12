# Phrase-density recovery

This owner-approved Ring 2 recipe replaces permanent phrase bans with bounded,
case-insensitive detection of sustained concentrations of “let me”, “actually”,
and “hmm” variants during reasoning. It retries from an earlier suspicious
checkpoint with a fresh seed, up to ten retries within the original observed
output-token budget. Temperature and top-p stay unchanged.

Normal output streams immediately. Only sustained suspicious output is buffered.
Already-streamed reasoning cannot be erased from clients, but is removed from
the model's retry context where the checkpoint precedes it. Answer/tool markers
disarm recovery; completed tool calls are not replayed. Image inputs and rendered
prompt metadata are retained. Client cancellation aborts the active attempt.

This is a targeted heuristic, not universal loop prevention. The latest captured
290,173-character failure was detected at character 8,783; an older differently
worded failure was missed. The replay corpus produced no alerts in 79 tool-producing
blocks; one short no-tool block remains unadjudicated. These controls informed
threshold selection and are not an independent false-positive estimate.
Unsupported sampling/output features pass through native serving with a reason
logged, including structured-output helper requests and custom bad_words.
Retry work consumes the original observed generation budget; tokens still in
flight at abort may not be observable. Retrying a detected loop changes output.

Native validation: one and two retries, preserved image identity, exact retry
prompt, distinct seeds, ordered abort/close/restart, client disconnect cleanup,
SSE usage accounting, valid tool calls, and actual Pi/Hermes requests.
A final test rewound 14 model-context tokens after 9 had already streamed:
469 visible tokens + 5 unpublished discarded tokens = 474 native generated
tokens and 474 reported completion tokens. It completed with the correct answer.
Diagnostic injection hooks are removed from this production commit.

All existing Ring 2 M64 kernels, image, weights, sampling, 524288 context,
eight sequences, 4096 batch tokens, uncapped images, caches, DSpark, graphs,
async scheduling and host guards are retained. Only the two API mounts and
the phrase-blocking/recovery feature flag change. Prior benchmark results are
not relabelled as a new performance benchmark; no new speed claim is made.
The inherited complete benchmark is not rerun for this API-only rollout.
Ring 1 remains unchanged until at least 30 minutes of healthy meaningful traffic.

Rollback: drain new arrivals, wait for native and gateway requests to finish,
then use the coordinated launcher to restore commit
cf76d70c557ba1cb865c73b21cbee774303cf2c3 and config/cluster.json on all ranks.
Timestamped full configuration/container backups remain under
/home/jordi/spark-lab/2026-10-12-ring2-baseline/phrase-density-20261012T233616Z/backup.
Do not prune caches, images or these backups.
