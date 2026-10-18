# r6n on TP3

**Promoted for TP3 on 2026-10-16** (owner: "promote tp=3"); `tp3-triangle`'s `benchmark_reference` is the release's r6n run, [manifests/suites/2026-10-15-karmic-kraken-r6n-tp3](../../manifests/suites/2026-10-15-karmic-kraken-r6n-tp3/).

r6n is TP4 production since 2026-10-15
([experiments/2026-10-15-r6n](../2026-10-15-r6n/README.md)). The owner recabled the
dgx1–dgx2–dgx3 triangle the same evening; the site shape is `tp3-triangle`, and
production serves r6h on TP3 from its archived record
(`experiments/2026-10-14-one-recipe/tp3-r6h-production.json`) until this release.

Nothing in r6n depends on the TP size. TP3 runs the recipe's `tp3-triangle` shape: r6h's
TP3 sizing (8 streams, 512K context) and NCCL profile, on r6n's image and settings. Its
DSpark cost curves are r6n's own (no pinned directory since 2026-10-15): the release's
first r6n boot profiles them and r6n again reuses them, while r6h keeps its pinned
`tp3-r6h-20261012`. The gate ran before that change, on r6h's curves; the traced and
cold-cache boots profile curves of their own, which production never uses.

## Arms

- [r6n-tp3.json](r6n-tp3.json): the recipe's TP3 shape.
- [r6n-tp3-dev.json](r6n-tp3-dev.json), [r6h-tp3-dev.json](r6h-tp3-dev.json): each with
  the dev-mode collective RPC for the release suite's engine grid; r6h with the
  engine-grid worker files as the TP4 releases' reference arms had.
- [r6n-tp3-traced.json](r6n-tp3-traced.json): sparknet's per-op trace.
- [r6n-tp3-coldcache.json](r6n-tp3-coldcache.json): an empty TileLang cache on every
  rank, the forum report's case (an 83 s cold-JIT skew between TP3 ranks).

## Runs, queued in order (production config: r6h's TP3 record)

1. [gate.json](gate.json): temperature 0 with and without logprobs (eight prompts, the
   fifteen-stream mix with a long prompt), and 8 and 4 distinct streams.
2. [qualify.json](qualify.json): the collective probe and sparknet's fail-stop scenarios
   (eager and graph) on every rank; three traced boots that open with a lone short
   request, then Poisson load; one boot on an empty TileLang cache.
3. [release.json](release.json): the `release-tp3` suite, r6n / r6h / r6n again, through
   `bin/spark bench --suite`. The owner decides promotion.

## Results (2026-10-15, 21:33 UTC to 2026-10-16 00:09 UTC)

**Gate** (on r6h's curves, before per-build curves): temperature 0 identical in all six
scenarios, the fifteen-stream mix included; 8 distinct streams 138.3 tok/s (±0.6%), 4
streams 111.3 (±0.8%).

**Qualify:** the collective probe and both fail-stop scenarios pass on all three ranks;
three traced boots that open with a lone short request, ready in 80–93 s, served every
request (6 of 6 each, none failed); the empty-TileLang-cache boot was ready in 409 s
(TP4: 419 s), with no startup failure.

**Release** (`release-tp3`, r6n / r6h / r6n again, one session, through
`bin/spark bench --suite` with the fan curve on every node and no thermal slowdown).
r6n profiled its own TP3 cost curves on its first boot
(`/cache/kkref/dspark-costs/build-77b3fa3c35f1ce6d`) and r6n again reused them; r6h ran
its pinned `tp3-r6h-20261012`. Verdicts against the run's own repeat:

| metric | r6h to r6n | ±95% | repeat (r6n to r6n again) | verdict |
|---|---|---|---|---|
| decode step time, geometric mean of 15 shapes | -5.46% | 0.14% | +0.34% | better |
| 8 streams, output tok/s (137.1 to 139.7) | +1.92% | 1.31% | -0.25% | better |
| 4 streams, output tok/s (109.3 to 112.1) | +2.50% | 2.18% | -0.22% | better |
| Poisson 0.42 req/s, output tok/s | +1.04% | 19.0% | -0.17% | same |
| prefill 32K tok/s | -0.44% | 1.52% | -0.40% | same |
| prefill 128K | -0.34% | 1.24% | -0.53% | same |
| prefill 256K | -0.53% | 1.15% | -0.52% | same |
| prefill 500K | -0.37% | 0.31% | -0.49% | same |

- Every one of the 15 decode shapes is shorter: 3.2% at 8 streams x 48 rows to 7.3% at
  one stream x 2 rows.
- Poisson at 0.42 req/s (no interval): goodput 0.280 to 0.290 req/s within a 2 s time to
  first token and 100 ms per output token; time to first token p50 -10%, p99 18.3 to
  14.5 s; time per output token p50 -4%. A p99 wait of 14–18 s says 0.42 req/s is near
  the triangle's capacity.
- Model behaviour: r6n's text differs from r6h's (0 of 256 greedy responses identical):
  scored on r6n's tokens r6h agrees on the top token at 95.04% of positions, KL 0.022
  nats, as r6h against r6k on TP4 (95.1%, 0.021): r6n carries r6k's reference-precision
  arithmetic. Acceptance length is level (2.336 to 2.331); with its own curves r6n
  verifies more drafts per step (2.66 to 2.85) for the same accepted length.
- Prefill reads 0.3–0.5% slower at every size, but r6n's own repeat moved as much, so no
  size counts as slower.
