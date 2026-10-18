# r6m: collectives that stop when they fail, Engram rows that always arrive

r6l plus five pieces, built to remove the first-request crash after boot
(experiments/2026-10-14-poisson-crash, root cause in
experiments/2026-10-15-engram-stall) and the failure class behind the startup
failures of a three-node r6h deployment reported on the NVIDIA forum (2026-10-14):

1. **No fallback in the RoCE collective policy** (vLLM 0142). In-place all-reduce,
   gather, broadcast and batched send/receive no longer reach other backends.
2. **Fail-stop that stops** (sparknet `timeout-stop`, 22c6caa, on `tilelang-only`):
   - waits time out by the GPU clock (`SPARKNET_ROCE_TIMEOUT_MS`, 10 s), not by a
     poll count of unmeasured duration;
   - a timed-out wait records the peer, its HCA lane, the time waited and the
     sequence, then traps, instead of letting the step compute on unwritten
     buffers until a NaN check fires;
   - the ring relay uses the same timeout (it had a fixed 5 s);
   - TileLang kernels only (the CuTe pair is retired).
3. **Startup that waits on the host** (vLLM 0143): ranks meet on the CPU group
   before memory profiling and after the local JIT warm-up (a monitored barrier,
   `VLLM_STARTUP_RENDEZVOUS_TIMEOUT_MS`, that names a missing rank), and a stopped
   collective's report becomes the step's error. Collectives keep one timeout from
   boot onward: a kernel compiling inside a collective phase is a defect to find
   and precompile, not a wait to lengthen (the cold-cache boot below inventories
   them).
4. **No compile on the decode path** (vLLM 0144): the draft-budget selection, a
   torch.compile function first compiled on the first step that trims a budget,
   runs eagerly with its kernels loaded at startup. Its lazy module load held the
   GIL while waiting for the GPU, which deadlocked the Engram reader: the crash's
   trigger. The JIT monitor now reports torch.compile in serving.
5. **Engram rows from a native reader** (vLLM 0145): the model's stream copies a
   step's row IDs to the host and writes the step's number into a control block in
   mapped host memory; a native worker (no GIL, no CUDA calls) reads the rows with
   io_uring and publishes the step; the forward's TileLang gather waits with a bound
   (a timeout records the step and stops the GPU) and dequantizes as B12X did, bit
   for bit. The Python reader thread, side stream, events, flags and gate are gone
   (experiments/2026-10-15-engram-io).

Source: vLLM tree `55e1367e` (patch head `a10879ff`, as the build replays it; 0142
and 0143 replay to `5e985c34`/`0285d3ff`, and the replay reproduces r6l's
`37e56d34`/`bc089a5c` two commits earlier), sparknet tree `8d69b6f4`. The arms are overlays of the recipe's `tp4-ring` shape
([r6m-release.json](r6m-release.json)); against r6l's arm they change only the
image, the trees, the lock, `SPARK3_ENGRAM_ASYNC` (retired) and `SPARKNET_ROCE_KERNELS`/`SPIN_LIMIT` ->
`SPARKNET_ROCE_TIMEOUT_MS` (tests/test_recipe.py).

## Runs

1. [image.json](image.json): build on dgx4, copy, gate (temperature-0 determinism,
   16 and 8 distinct streams).
2. [qualify.json](qualify.json): the Engram reader, gather, model, draft-budget and
   JIT-monitor unit tests from the image ([bundles/engram-tests](bundles/engram-tests),
   dgx4); the collective probe through vLLM's communicator; sparknet's fail-stop
   scenarios (eager, and a graph captured under a 60 s timeout replayed under 2 s)
   on every rank, each judging its own stop time and report (with a 1 MiB dispatch
   limit: the probe registers 1 MiB, and serving's 2 MiB limit is not under test); one boot on an empty
   TileLang cache (every rank cold), which reproduces the forum report's startup
   skew and lists what compiles inside a collective phase; six traced boots that
   each open with a lone short request (the crash's trigger), collecting every
   node's log and sparknet trace; then two boots with every Engram read delayed
   200 ms ([r6m-delay.json](r6m-delay.json)), which must wait for their rows, never
   hang (r6k hung on 2 of 2).
   [qualify-boots.json](qualify-boots.json) repeats qualify's boots, most important
   first (the delay boots, the traced boots, then the cold cache): a failed boot ends a
   run. On 2026-10-15 the cold-cache boot failed at setup on every rank: sparknet's
   TileLang trace clock (on in that arm and the traced arms) postpones its annotations,
   so TileLang cannot resolve the probe's symbolic shape (`NameError: name 'n' is not
   defined`; fixed in sparknet 3bc8d10). These boots run without the trace: the
   first-request boots on the release arm, the cold cache without `SPARKNET_ROCE_TRACE`.
3. [release.json](release.json): the release suite against r6k.

TP3 follows once these pass: see experiments/2026-10-14-one-recipe ("Releasing TP3").
