# r6h: the last 0.9% at sixteen streams, at r6g's bits

[r6g's release comparison](../2026-10-12-r6g-release/README.md) left one metric below r6d:
sixteen different prompts at once, -0.9% ±0.7%. At 96 verify rows r6g's step is about
1.6 ms longer than r6d's. Per call (fixed-row profiles, adaptive verification off):

| kernel | r6d | r6g | per step |
| --- | ---: | ---: | ---: |
| mHC `project_streams`, 96 rows (grid 40 x 6, twice a layer) | 30.6 µs | 41.5 µs | +0.9 ms |
| mHC `project_streams`, 48 rows (40 x 3) | 17.8 µs | 22.4 µs | +0.2 ms |
| MoE combine (5 x 96) | 12.0 µs | 16.7 µs | +0.18 ms |
| DSpark Markov projection (505 x 1) | 46.7 µs | 67.3 µs | +0.11 ms |

The largest single kernel is not a regression: the vocabulary head GEMM (grid 1010 x 2,
verification and drafting) takes about 2.17 ms a call on both, 4.5% of a step. Its
packed 12-bit shard (32,320 x 5,120) is about 248 MB, about 115 GB/s, under half the
GB10's DRAM bandwidth; it lives in the display carve-out.

## Window 1 ([w1.json](w1.json)): two configuration levers on the r6g image

No code change, same bits:
- **head**: `SPARK3_DISPLAY_CARVEOUT_WEIGHTS=0`, the embedding and packed head in regular
  memory (about 0.55 GB a rank; TP4's lowest MemAvailable is about 20 GiB);
- **curves**: DSpark cost curves profiled for this image (empty
  `r6g-image-20261012`, real-row profiling) instead of r6's of 2026-10-05; adaptive
  verification sizes each step from them, and the committed tokens do not depend on
  how many drafts a step verifies.

base / head / curves / base again, six samples of sixteen and eight different prompts,
then a fixed-row profile of base and head (the head GEMM a call).

## Window 2 ([w2.json](w2.json)): three kernel changes, bit-identical by construction

vLLM branch `r6h-mhc-fused-split` (on r6g's `fbaa04151`):
- `544ee7703` mHC: fn split into its TF32 parts while loading it
  ([mhc-split](../2026-10-12-mhc-split/README.md): bit-identical, 2-5 µs a call);
- `54af64774` MoE: the folded combine loads the shared row before the routes;
- `7fd70268a` vocabulary heads: token tiles vary fastest. Past 64 rows a head call has
  two token tiles; with vocabulary blocks fastest the second tile re-read the whole
  248 MB shard from DRAM (about 2.2 ms a 96-row call). At sixteen streams the drafter's
  head call always has 80 rows, so every step paid it; r6d has the same kernel.

[bundles/kernels](bundles/kernels) runs the mHC, MoE and vocabulary head tests with the
three files (dgx1); [bundles/head](bundles/head) times the head per call with both grids
and compares the outputs bit for bit (dgx2).

### Window 1 results (2026-10-12, 05:20-05:45 UTC)

| arm | sixteen prompts, tok/s | eight prompts |
| --- | ---: | ---: |
| base (r6g) | 220.1 ±1.5% | 178.4 ±0.4% |
| head out of the carve-out | 220.3 ±1.3% | 178.5 ±0.4% |
| curves profiled for the image | **223.5** ±0.9% | **180.3** ±0.4% |
| base again | 220.2 ±1.4% | 178.4 ±0.4% |

The carve-out streams at full speed: no change. Curves profiled for the image add 1.5%
and 1.1% (adaptive verification had sized r6g's steps from r6's cost shape).

### Window 2 and 3 results (05:45-05:48 UTC)

The mHC, MoE and vocabulary head tests pass with the three changes (82 passed). The
head bench, bit-identical: one tile (up to 64 rows) unchanged at 1.10-1.24 ms a call;
two tiles 2.29-2.38 ms with vocabulary blocks fastest, 1.60-1.89 ms with token tiles
fastest (80 rows: 2.34 to 1.65 ms; the drafter's call at sixteen streams).

In serving the grid change does not pay: the served head lives in the uncached display
carve-out, where the second tile cannot reuse L2, and reads 2.22 ms a call. With the head
in regular memory it reads 1.73 ms ([r6h-loss](../2026-10-12-r6h-loss/README.md)).

## r6h

r6g's series and vLLM `r6h-mhc-fused-split` (0115-0117): patch head `e39cd9da9938`,
tree `68c9abd9c1d7` (replayed as build prepare does). [r6h-tp4.json](r6h-tp4.json) is
r6g-tp4 with the `-r6h` image and its own cost curves (`ring4-r6h-20261012`, profiled
with real rows on the image's first boot). [image.json](image.json) builds and copies it
and gates it (determinism with and without logprobs); [release-tp4.json](release-tp4.json)
runs the release comparison, r6d / r6h / r6d again.

## TP3 ([release-tp3.json](release-tp3.json)), trimmed for a same-morning re-cable

r6d (`r6d-tp3.json`) against r6h ([r6h-tp3.json](r6h-tp3.json): the promoted TP3 profile
with the `-r6h` image and its own cost curves, `tp3-r6h-20261012`, profiled on its first
boot) on the triangle: quality, prefill at 32K / 256K / 500K (two repeats),
determinism `--wide`, the paired acceptance sweep, and four and eight different prompts
at once. The bench's one-prompt decode points and the latency extras are left out.

Arm order (owner, 2026-10-12): the candidate first, so heat soak does not count against
it, and again last for drift. The TP3 comparison had already started in the old order
(r6d first, r6h second), so its r6h numbers are conservative. The TP4 comparison runs
r6h / r6d / r6h again.

### TP3 results (2026-10-12, 06:40-07:09 UTC, the triangle)

`release_table.py r6h-release-tp3 r6d r6h` (r6d ran first, r6h second, under the old
order). Both benches started with every node at 47-51 °C (below the 55 °C cooling
threshold), the fan performance curve active and no thermal slowdown.

| metric | r6d | r6h | change |
| --- | ---: | ---: | --- |
| temperature 0, every scenario | mixed15 1 differing | **identical** | |
| quality | 5/5 | 5/5 | |
| paired sweep, 1 stream (16 prompts) | | | -1.0% ±2.3% |
| 4 different prompts at once, tok/s | 107.5 | 107.1 | -0.4% ±1.1% (r6f: -3.1%) |
| 8 different prompts at once | 137.2 | 134.9 | **-1.6% ±0.6%** |
| prefill 32K, tok/s | 4,304 | 4,252 | -1.2% ±1.5% |
| prefill 256K | 4,123 | 4,060 | **-1.5% ±0.4%** |
| prefill 500K | 3,797 | 3,753 | **-1.2% ±1.0%** |

r6h is identical at temperature 0 and closes r6f's four-prompt gap. It still trails r6d
on TP3 at eight prompts and on long prefill, about 1.5%. r6h is r6f plus bit-identical
work, so it is not behind r6f; the owner promotes it as a meaningful improvement and
moves the TP3 shortfall to next week's release.

Next week's TP3 leads: the precision pieces cost more per rank on three nodes (a third
of the vocabulary and experts each): the mHC projection (three TF32 GEMMs), MoE one
rounding on 768-3072-row prefill chunks, the finalize launch per mHC sublayer, and TP3's
own fixed-row profile, which this release did not take.

### TP4 results (2026-10-12, from 07:48 UTC, the ring)

After a 30-minute cool-down at the maximum fan floor; candidate first: r6h / r6d / r6h
again. `release_table.py r6h-release-tp4 r6d r6h`; every bench started below 55 °C,
under the fan performance curve, without thermal slowdown.

| metric | r6d | r6h | change |
| --- | ---: | ---: | --- |
| temperature 0, fifteen streams + long prompt | 13 of 16 differing | **0** (r6f's references) | |
| quality | 5/5 | 5/5 | |
| paired sweep, 1 stream (16 prompts) | | | **+2.6%** ±2.5% |
| 4 different prompts at once, tok/s | 134.0 | 136.5 | +1.9% ±2.1% |
| 8 different prompts at once | 178.5 | 184.7 | **+3.5%** ±1.5% |
| 16 different prompts at once | 224.1 | 226.5 | +1.1% ±1.5% |
| prefill 32K, tok/s | 5,781 | 5,870 | **+1.5%** ±0.8% |
| prefill 128K | 5,669 | 5,726 | **+1.0%** ±1.0% |
| prefill 256K | 5,429 | 5,496 | **+1.3%** ±0.4% |
| prefill 512K | 4,998 | 5,074 | **+1.5%** ±0.2% |
| prefill 1M | 4,346 | 4,391 | **+1.0%** ±0.3% |
| mixed traffic, p95 stream gap (one run, no interval) | 86.6 ms | 90.0 ms | +3.9% |

r6h is at or above r6d on every metric with an interval on four nodes; sixteen prompts
moved from r6g's -0.9% to level. The bench's one-prompt points follow the text (code c1
+21.1% with acceptance 1.75 to 2.37, prose c1 -4.7%).

**Drift arm** (r6h again, last, 09:01-09:40 UTC): identical at temperature 0 with the same
references, quality 5/5. Against the first r6h arm, decode read 0.6-1.4% lower (paired
+1.9%, 4/8/16 prompts +1.3/+2.1/+0.3% against r6d) and prefill 0.3-0.4% higher (+1.2% to
+1.9% against r6d), every difference inside its interval. The repeat still ranks level with
or ahead of r6d on every metric with an interval. Mixed traffic's one-run p95 stream gap read
93.1 ms (r6d 86.6, first r6h 90.0).

## Promotion (2026-10-12)

Owner: promote unless there is a technical regression; r6h is better than r6f and the
TP3 shortfall moves to next week's release.

- Profiles: `config/cluster-tp4.json` and the three TP3 profiles run the `-r6h` image
  with its own cost curves (`ring4-r6h-20261012`, `tp3-r6h-20261012`).
- Root lock: this series, source manifest
  [manifests/sources/2026-10-12-r6h-source.json](../../manifests/sources/2026-10-12-r6h-source.json).
- Baseline: [2026-10-12-karmic-kraken-r6h](../../manifests/baselines/2026-10-12-karmic-kraken-r6h.json).
- Benchmark references: the comparisons' r6h arms,
  [TP4](../../manifests/benchmarks/2026-10-12-karmic-kraken-r6h-tp4.json) (full) and
  [TP3](../../manifests/benchmarks/2026-10-12-karmic-kraken-r6h.json) (quality and prefill).
- Deployed through the TP4 window's close: a sync job moves the checkouts to this
  promotion, so the window's production restore boots r6h.
