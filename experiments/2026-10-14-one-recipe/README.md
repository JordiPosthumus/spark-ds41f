# 2026-10-14: one recipe for every shape

Until today each cluster shape had its own whole-file configuration:
`config/cluster.json` (TP3 triangle, 64 KiB), a byte-identical
`config/cluster-64k.json`, `config/cluster-4k.json` (TP3 on 4 KiB pages) and
`config/cluster-tp4.json` (TP4 ring). TP3 was frozen on r6h because benching it
needs the triangle re-cabled, so every change since r6h landed in the TP4 file
only. Of the 45 settings that differed between TP3 and TP4, these were drift,
not decisions:

- TP3 ran sparknet's CuTe collectives (no `SPARKNET_ROCE_KERNELS`), TP4 TileLang;
- the L2 prefetch kernels: CuTe on TP3, TileLang on TP4; prefetch grid 4 on TP3,
  2 on TP4 (the 2026-10-06 A/B ran on TP4 only);
- no big-core proxy (`SPARKNET_ROCE_PROXY_CPU`) and no CPU pinning
  (`--cpuset-cpus=5-9,15-19`) on TP3;
- the image, lock and baseline: r6h on TP3, r6k on TP4.

The rest belong to the shape: the node map, transport and NCCL settings (equal
to sparknet's `tp3-triangle` and `tp4-ring` profiles), the sizing that follows
from memory per rank (context, streams, batched tokens, KV cache, prefix-cache
retention), the DSpark cost curves measured per shape, and the display
carve-out (TP3 needs it for its KV cache; TP4 keeps the head in regular memory,
where it keeps its L2 reuse).

## The recipe

[config/cluster.json](../../config/cluster.json) now holds everything the
shapes share once and a `shapes` table with only the above. A shape may extend
another (`tp3-triangle-4k` changes the host kernel, context and KV only).
[scripts/recipe.py](../../scripts/recipe.py) resolves it to the concrete form
every command reads; derived values are computed, not copied (TP size from the
node count, CUDA-graph capture sizes up to streams x 6, the long-prefill
threshold from the batched tokens). The site names its cabling once in the
git-ignored `config/site.json` (`bin/spark site shape NAME --apply`, copied by
`cluster sync`); `--shape NAME` overrides it.

A shape is qualified for the recipe's image when the recipe names its release
benchmark (`benchmark_reference`). `cluster start --apply` refuses an
unqualified shape unless given `--unqualified-shape`, so a release cannot reach
a shape it was not benched on. Today only `tp4-ring` is qualified (r6k).

Receipts: `tests/test_recipe.py` resolves `tp4-ring` to exactly the archived r6k
TP4 file, and `tp3-triangle` and `tp3-triangle-4k` to exactly the archived TP3
serve arguments, fabric and per-shape environment.

## Archived records

- [tp4-r6k-production.json](tp4-r6k-production.json): TP4 production on r6k.
- [tp3-r6h-production.json](tp3-r6h-production.json): TP3 production on r6h
  (its lock is `experiments/2026-10-13-r6k/r6h/upstreams.lock.json`).
- [tp3-4k-r6h.json](tp3-4k-r6h.json): the 4 KiB TP3 variant on r6h.

## Releasing TP3

1. The next candidate passes its TP4 gate and release comparison.
2. Re-cable the triangle (dgx1-dgx2-dgx3), then `bin/spark site shape tp3-triangle
   --apply`, `cluster sync --apply` and `doctor --live` (the start guard checks
   the cable subnets against the live links).
3. In a lab window: boot the candidate on TP3 (`--unqualified-shape`), gate it and
   run the release comparison against r6h's archived record.
4. Promote: add `benchmark_reference` for `tp3-triangle` (and the cost curves if
   re-measured). Until then, a triangle restores r6h with
   `--production-config experiments/2026-10-14-one-recipe/tp3-r6h-production.json`.
