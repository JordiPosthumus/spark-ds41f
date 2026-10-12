# M64 rows for large TP3 MoE prefills

This is an **unpromoted Ring 2 canary**, based on the owner's accepted runtime
at recipe commit `9ff535dc68ddbf248f52c4eb41f4d6fccbbd9efa`. It is published on
top of owner-fork commit `2a1d91cdf40612704881d2a40ff7b97f63616bb4`; the 188
runtime-relevant tracked files checked between those revisions are identical.
The repository is public. No private prompts, images, generated conversations
or credentials are part of this experiment.

## Hypothesis and exact change

The native E384/K6/H5120/I768 routed-expert path may process large prefills faster
with 64-row tiles. The adapter selects M64 only at 2,048 or more rows for that
shape; all other cases keep M32. Both plans use the existing N128 weight packing,
three pipeline stages, ordered FP32 combine and BF16 output fold. The extra
workspace at 4,096 rows is about 165.4 MiB per pooled lane, not per model layer.

The candidate adapter is SHA-256
`072d2a99d1cbdb538ce5fbba6c0ca66935f234bb9f9e33e0d2e9ded860e73526`.
Its patch applies to the bundled accepted adapter, not directly to upstream
vLLM. See [source.json](source.json), [patches/series](patches/series) and the
accepted source snapshots for provenance.

`canary.json` changes one runtime bind source plus the repository and branch
needed to publish this experiment. It preserves the accepted image, effective
environment, other six overlays, native weights, 524,288-token context, eight
sequences, 4,096-token batch budget, uncapped images, sampling, DSpark, graphs,
asynchronous scheduling, memory guards and normal compiler-cache backing.
No normal launcher or default configuration is selected by this experiment.

## Evidence already collected in the isolated layout

- 1,300 isolated same-input replays and 8,100 model-shadow comparisons were
  byte-identical. These finite checks do not constitute a universal proof.
- All six capacity cases passed: paired baseline/candidate runs at the actual
  524,288 total-token boundary, eight overlapping 65K–72K histories, and seven
  generating chats alongside a 13-image prompt. The suite completed 66 inference
  requests, including its warmups. Boundary responses matched all 128 native
  output token IDs.
- Large isolated kernels improved about 7–10%; whole-prompt observations support
  a much smaller roughly 1–3% improvement. There is no demonstrated steady-decode
  speedup. Temperature-1 throughput changes also depend on output length and
  speculative acceptance and must not be called pure kernel improvements.
- Retain the unfavorable observations: the single warm boundary comparison was
  3.371 versus 3.415 seconds; the worst background SSE chunk gap in one mixed run
  was 2.569 versus 3.443 seconds. Neither single comparison establishes causality.
- Candidate minimum available memory was 4.725/6.379/6.364 GiB across the three
  ranks in these capacity runs. No completed case had an OOM, engine restart,
  request failure or KV preemption. The earlier backup-preparation interruption
  and gateway-preflight interruption are retained separately; neither is a
  completed model test.

The aggregate evidence is in [evidence/](evidence/). Raw run receipts, complete
three-rank logs and the synthetic replay corpus are retained in the local lab;
this public bundle contains the measured aggregates and source hashes. The
accepted baseline was restored and its real 13-image cold-to-warm cache replay
validated before the next service window.

## Remaining qualification and decision

The next test is the same candidate from this published commit, using the normal
compiler-cache paths rather than the isolated experiment cache. It must pass
exact runtime/source checks, native output fingerprints and the original image
cache/overlap replay before any promotion proposal. Source checks and a passing
`bin/spark doctor` alone do not qualify GPU correctness or service stability.

The historical patch header and some preparation gates still describe capacity
as pending; the completed capacity evidence above supersedes that preparation
status. It does not supersede the pending final-layout canary, complete recipe
benchmark, or exact owner acceptance. The adapter itself is intentionally kept
byte-for-byte identical to the tested artifact.

Decision: retain as a candidate. [PHASES.txt](PHASES.txt) describes the later,
separate production commit and actual immutable benchmark/baseline records.
No empty production manifests are included. Loop-recovery work is a separate
experiment and is not installed by this profile.

## Controlled execution and rollback

Only use the reserved Ring 2 lab window after the gateway is drained and both
gateway and engine report idle. Back up the exact three-rank Git metadata,
accepted profile, topology, costs, source identities, logs and cache inventory.
Use the recipe's coordinated `cluster sync`, `start` and `stop` operations with
the explicit canary profile and an exact published commit; never copy a dirty
checkout onto a serving node. This site's ignored node map remains local.

Rollback restores each rank's exact original Git configuration and origin,
accepted commit `9ff535dc68ddbf248f52c4eb41f4d6fccbbd9efa`, and original profile
SHA-256 `9309ad55b33bb884096c7df70eb6c2de450aeb500744a1bfe5c010b466aa5073`.
Verify the restored runtime and a real image-cache replay before readmission.
Do not remove existing compiler-cache entries. The normal launcher, Ring 1,
host/network configuration and model weights are outside this canary's scope.

Publication note: evidence-map keys use lab-relative paths; 451 redundant Mac
workspace prefixes were removed from the copied maps. Measurements and evidence
SHA-256 values are unchanged, and the original local receipts remain intact.
