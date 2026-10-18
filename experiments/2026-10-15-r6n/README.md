# r6n: r6m with Engram rows v2

**Promoted for TP4 on 2026-10-15** as [`2026-10-15-karmic-kraken-r6n`](../../manifests/baselines/2026-10-15-karmic-kraken-r6n.json), on the release below ([suite run](../../manifests/suites/2026-10-15-karmic-kraken-r6n-tp4/)). Owner: post and promote. The TP3 shapes resolve to r6n but stay unqualified until the triangle is re-cabled and released. The r6k recipe this release ran against is frozen in [r6k/cluster.json](r6k/cluster.json) with its lock.

r6m (experiments/2026-10-14-r6m) was not promoted: against r6k it had the same bits,
decode 1.2% faster on the geometric mean of the grid and up to 4.5% at small row
counts, but every prefill size 10-13% slower and the 72- and 96-row decode steps
1.5-2.3% slower. The step back of 2026-10-15 found the cause: its Engram gather polled
the control block in host memory from every block, about 100 ms per table of an 8K
chunk ([engram-rows-v2](../2026-10-15-engram-rows-v2/README.md)).

r6n is r6m with:

1. **Engram rows v2** (vLLM 0145, rewritten): one native worker reads the tables in
   layer order; in the forward one thread waits for the table's step (bounded, and
   stopped at once by a failed read) and the TileLang gather loads only this rank's
   rows. Against B12X on the same data the gather is within about 1 us at 16-96
   tokens and faster from 512 tokens, bits equal; the two tables' reads are level at
   decode sizes and faster at prefill sizes.
2. **sparknet 3bc8d10** (`timeout-stop`): r6m's fail-stop collectives, plus the
   fail-stop probe bounding its dispatch by its own registration, and the trace
   clock's TileLang kernel fixed (every traced runtime failed setup on 22c6caa).

Unchanged from r6m: no fallback in the RoCE collective policy (0142), a stopped
collective named as the cause and host rendezvous at startup (0143), and the draft
budget run eagerly (0144), which alone removes the first-request crash's trigger.
Nothing in r6n depends on the TP size; TP3 uses the same settings.

Source: vLLM tree `0c60314d` (patch head `b1309b4a`, as the build replays it), sparknet
`3bc8d10` (tree `f72599fa`); B12X, NCCL, TileLang and TileKernels as r6m.

## Runs

1. [image.json](image.json): build on dgx4, copy, gate (temperature-0 determinism,
   16 and 8 distinct streams).
2. [qualify.json](qualify.json): the Engram, draft-budget and JIT-monitor unit tests
   from the image ([bundles/engram-tests](bundles/engram-tests), dgx4); the collective
   probe; sparknet's fail-stop scenarios on every rank; then boots, most important
   first because a failed boot ends a run: two with every Engram read delayed 200 ms
   ([r6n-delay.json](r6n-delay.json)), which must wait for their rows, never hang;
   six traced boots that each open with a lone short request (the crash's trigger);
   one boot on an empty TileLang cache.
3. [release.json](release.json): the release suite against r6k (r6n, r6k, r6n again).
   The owner decides promotion.

TP3 follows once these pass: see experiments/2026-10-14-one-recipe ("Releasing TP3").

## Qualify (2026-10-15)

- Gate: temperature-0 output identical in every scenario and byte-identical to r6k's;
  16 distinct streams 227.4 tok/s (±0.24%), 8 streams 176.6 (±0.27%).
- Collective probe and both fail-stop scenarios pass on all four ranks.
- Two boots with every Engram read delayed 200 ms: neither hung (a 64-token request in
  9.0 s: each step now waits for the two tables in turn).
- Six traced boots that open with a lone short request, then Poisson load: 10 of 10
  requests served on each, none failed.
- Empty TileLang cache on every rank: ready in 419 s.
- The Engram unit tests: the reader and gather tests passed, but the first Engram model
  test then spun for 37 minutes in a host-to-device copy while loading its checkpoint
  (GPU idle, no Engram reader yet) and was killed. A diagnostic
  ([engram-debug.json](engram-debug.json), dgx3 and dgx4) did not reproduce it: the model
  tests alone and right after the GPU-stopping tests all passed. The bundle now bounds
  every step, dumps every thread's stack after 240 s, and runs the tests that stop the
  GPU last; it reruns the whole sequence after the release.
  Root cause, found 2026-10-16: driver R580 on the 64 KiB-page kernel never completes a
  large pageable copy from untouched, unaligned file pages (safetensors' views into
  the mmapped checkpoint). vLLM 0147 copies such sources into anonymous memory first
  ([experiments/2026-10-16-r6o](../2026-10-16-r6o/README.md)).

## Release against r6k (2026-10-15, 19:25 UTC)

r6n, r6k, then r6n again, in one session on TP4. Verdicts against the run's own repeat:

| metric | r6k to r6n | ±95% | repeat (r6n to r6n again) | verdict |
|---|---|---|---|---|
| decode step time, geometric mean of 20 shapes | -2.21% | 0.15% | +0.17% | better |
| 16 streams, output tok/s | +0.85% | 0.98% | -0.09% | same |
| 8 streams, output tok/s | +0.20% | 0.71% | +0.05% | same |
| prefill 32K tok/s | +0.50% | 0.34% | -0.27% | same |
| prefill 128K | +0.48% | 1.17% | -0.23% | same |
| prefill 256K | +0.44% | 0.49% | -0.19% | same |
| prefill 512K | +0.35% | 0.37% | -0.09% | same |
| prefill 1M | +0.25% | 0.67% | -0.06% | same |

Every one of the 20 decode shapes is shorter (0.9% at 96 rows to 4.4% at one stream and
256K of context), where r6m was up to 2.3% longer at 72 and 96 rows; prefill is level
where r6m was 10-13% slower. Model behaviour is identical: 256 of 256 greedy responses,
and the teacher-forced distributions match exactly (KL 0). Nothing is slower than r6k.

## Credits

huxiaofengtiger reported r6h's TP3 startup failures on the NVIDIA Developer Forums
(the DeepSeek 4.1 Flash recipe thread, post 16, 2026-10-14). The report included the
deployment commit, image, kernel, driver and file hashes, and per-rank timings: an 83 s
cold-JIT skew between ranks that tripped the one-shot collectives' spin limit and the
Engram reader's 20 s deadline. They also tested a monitored barrier after local JIT
warmup on their own nodes. The startup rendezvous (vLLM 0143), the fail-stop timeouts
and the cold-cache qualification boot carried into r6n follow from that report.
