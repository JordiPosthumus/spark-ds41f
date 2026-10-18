# r6h's remaining cost against r6d, per kernel, on the ring

r6h is level with or ahead of r6d on every TP4 metric, but on TP3 it trails r6d by 1.6%
±0.6 at eight prompts and by 1.2-1.5% on prefill
([r6h](../2026-10-12-r6h/README.md)). r6f's exact arithmetic (FP32 logits, the three-part
TF32 mHC projection, FP32 MoE partials) added cost that r6g and r6h won back on four
nodes, partly through the head grid fix, which never applies on three (at most 48 rows a
step).

The triangle cannot be cabled now. A TP3 step has at most 48 verify rows, and the
replicated kernels (mHC, norms, the drafter) cost the same per rank on either topology,
so a fixed-row profile on the ring at 48 rows shows most of what TP3 pays. The sharded
kernels (MoE, the heads) are bigger per rank on TP3; their TP3 shapes go to kernel
bundles on one node.

## Window 1 ([w1.json](w1.json))

A profile job, the candidate first: r6h ([r6h-fixed.json](r6h-fixed.json), the promoted
`config/cluster-tp4.json`) then r6d ([r6d-fixed.json](r6d-fixed.json), the fixed-row
profile's r6d arm), each with adaptive verification off (every step verifies the anchor
and all five drafts) and under the torch profiler:

- `code16`: sixteen streams, 96 rows a step;
- `c8`: eight streams, 48 rows a step (TP3's maximum);
- `prefill`: one 16K prefill (sequence parallelism).

The lab compares kernels per step between the arms with `analyze_costs.py`. Each
remaining loss becomes a bit-identical kernel change, measured at both topologies'
shapes, and then a TP4 screen; TP3 is confirmed at the next re-cable.

### Window 1 results (2026-10-12, 18:39-18:45 UTC)

Rank 0, per target step (TileLang kernels are all `main_kernel`, so
[group_costs.py](group_costs.py) groups them by launch signature; routing-dependent grids
are pooled). [analyze_costs.py](analyze_costs.py) now counts steps from the TileLang head
(half a step per launch: the drafter calls it too).

| | 48 rows (c8) | 96 rows (code16) |
| --- | ---: | ---: |
| kernel time per step, r6d -> r6h | 109.0 -> 108.4 ms (-0.6) | 105.7 -> 104.1 ms (-1.6) |
| FP32 logits all-gather (greedy verification removes it) | -0.41 | -0.84 |
| sparknet one-shot all-reduce (FP32 MoE partials: twice the bytes) | +0.33 | -0.22 |
| mHC `project_streams` (three TF32 parts), per call | 18.3 -> 22.4 us (+0.27 ms) | 29.7 -> 37.0 us (+0.58 ms) |
| DSpark Markov projection (FP32 output), per call | 40.3 -> 44.5 us | 44.4 -> 68.0 us (+0.12 ms) |
| vocabulary head, two tiles, per call | (one tile) 1.15 -> 1.16 ms | **2.16 -> 2.22 ms** |

**The head grid fix does nothing in serving.** r6h's head bench read 1.65-1.73 ms for
a two-tile call; in serving it is 2.22 ms, against r6d's 2.16 (BF16 output). The served
head lives in the display carve-out, which the GPU maps uncached: the second token tile
cannot find the shard in L2 and reads it from DRAM again. The bench used regular memory.
r6h window 1's "head out of the carve-out" arm ran on r6g's grid, where L2 cannot help
either way, so the working combination (token tiles fastest, head in regular memory)
has not run in serving. r6h's gain over r6g came from the image's cost curves and the
mHC split.

Prefill (one 16K prefill, whole window, rank 0): span 745.6 -> 753.0 ms. The rank-order
collectives (NCCL SendRecv in place of NCCL's ring all-reduce) cost about 10 ms net;
the FP32 MoE GEMMs and combine about 9 ms; the all-gather of FP32 unreduced partials
about 5 ms; the WO projection's slices about 11 ms of GEMM time, overlapped with its
reduce-scatter. On TP3 each weighs more.

## Window 2 ([w2.json](w2.json)): the head in regular memory

r6h with `SPARK3_DISPLAY_CARVEOUT_WEIGHTS=0` (about 0.55 GB a rank more regular memory;
TP4's lowest MemAvailable is about 20 GiB), the candidate first: head out (r6h's pinned
cost curves), r6h, head out with its own curves (real rows, empty
`ring4-r6h-headout-20261012`), head out again. Each arm: `token_determinism.py --wide`
(references must stay r6f's, `7ae87256d3ea` / `b8b35451dcc4`), six samples of sixteen and
of eight different prompts. Then a fixed-row profile of the head-out arm (code16): the
two-tile head should read near the bench's 1.65-1.73 ms. TP3 never has two tiles, so
this changes nothing there.

### Window 2 results (2026-10-12, 18:51-19:24 UTC)

Every arm identical at temperature 0 with r6f's references. Different prompts at once,
six samples, tok/s:

| arm | sixteen | eight |
| --- | ---: | ---: |
| head out (r6h's curves) | 224.1 | 181.3 |
| r6h | 223.0 | 180.9 |
| head out, its own curves | **226.3** | 180.2 |
| head out again | 224.7 | 181.1 |

The two-tile head reads 1.73 ms a call in regular memory against 2.22 in the carve-out
(every rank), as the bench predicted: 0.95 ms less head time a 96-row step. The step does
not shrink by that much (profiled span -0.2%): the one-shot all-reduce, which spins until
the slowest rank arrives, takes 0.43 ms more, so at sixteen streams the head is not what
bounds the step. With curves profiled for it, adaptive verification spends the cheaper
two-tile call on more drafts: +1.0% ±1.0 against head out on r6h's curves, about +1.5%
against r6h. Eight prompts (one tile) is level. TP3 never has two tiles.

The 128-row tile in TODO.md shares this ceiling: at sixteen streams the time a faster
head saves becomes all-reduce wait unless the step's skew is addressed too.

**Deployed (2026-10-12, owner):** `config/cluster-tp4.json` keeps the embedding and head in
regular memory with the curves window 2 profiled for it (`ring4-r6h-headout-20261012`).
Same image, same bits; TP3 profiles keep the carve-out.
