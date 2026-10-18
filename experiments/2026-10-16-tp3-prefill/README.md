# TP3 prefill: where r6n's time goes against r6h

r6n's TP3 release (experiments/2026-10-15-r6n-tp3) read prefill 0.3-0.5% slower than r6h
at every size (32K-500K). r6n's own repeat moved as much, so the release counts it as level,
but the arms ran in time order (r6n, r6h, r6n again) and the repeat drifted down 0.5%, so a
small real cost of about 0.5% is plausible. TP4 shows no such loss (r6h to r6k +0.5-0.8%
at long prefill, r6k to r6n level), so the suspects are what TP3 changes:

1. **r6k's TileLang WO** (WO-A on BF16 activations, B12X used FP8), its tiles tuned on TP4's
   per-rank shapes; a TP3 rank holds a third of WO.
2. **Engram rows v2**: a TP3 rank reads 8 of the 24 heads' rows (TP4: 6); v2 was
   benchmarked against B12X only on TP4's quarter shard. The forward waits for the rows at
   layers 1 and 14 (r6n's one-thread `engram_rows_wait` kernel shows that wait on the GPU).

Not the cost curves: they size verification only.

## Method ([profile.json](profile.json))

The lab's profile job: each arm boots with vLLM's torch profiler
([r6n-tp3-prof.json](r6n-tp3-prof.json), [r6h-tp3-prof.json](r6h-tp3-prof.json)), and
[profile_prefill.py](profile_prefill.py) profiles one cold 16K prefill of servebench's
prefill document (the release suite's text), after an unprofiled warm prefill of it. Every
rank's trace is collected; with the cluster stopped, [summarize_kernels.py](summarize_kernels.py)
and [analyze_costs.py](analyze_costs.py) (`--total`) compare kernel time by family, rank by
rank, against r6n. Arms: r6n, r6h, r6n again: the second r6n bounds a family's
boot-to-boot noise.

## Results (2026-10-16, 06:40-06:47 UTC)

The 16K prefill (16,388 tokens) takes 3.963 s on r6n, 3.901 s on r6h and 3.977 s on r6n again
(unprofiled warm passes 3.958 / 3.906 / 3.956 s): r6n is about 1.4% slower at this size, well
beyond the two r6n boots' difference.

r6h's TileLang kernels predate their names (all `main_kernel`), so
[match_grids.py](match_grids.py) pairs them with r6n's by grid, block and call count, and
[wo_totals.py](wo_totals.py) totals the paths that differ (ms per rank over the request):

| rank | build | span | WO-A and inverse RoPE | WO-B (MXFP8) | Engram wait kernel | NCCL | everything else |
|---|---|---:|---:|---:|---:|---:|---:|
| dgx1 | r6h | 3869.6 | 270.0 (B12X, FP8) | (in B12X's) | (not a kernel) | 614.0 | 2927.5 |
| dgx1 | r6n | 3931.2 | 204.5 | 124.0 | 53.0 | 642.0 | 2927.6 |
| dgx2 | r6h | 3866.5 | 269.3 | | | 569.7 | 2936.9 |
| dgx2 | r6n | 3928.2 | 211.3 | 126.4 | 86.2 | 586.6 | 2933.7 |
| dgx3 | r6h | 3866.5 | 270.5 | | | 544.2 | 2924.3 |
| dgx3 | r6n | 3925.2 | 216.3 | 124.8 | 119.2 | 553.8 | 2929.6 |

- **The WO projection is the loss.** Everything outside WO, the Engram wait and NCCL is level
  (within 5 ms of 2,930 on every rank). r6n's WO takes 329-341 ms per rank against r6h's 270
  (B12X's fused FP8 WO with its quantize and inverse-RoPE kernels): 59-71 ms more, the whole
  59-62 ms span difference. r6n against r6n again moves 4 ms.
- WO-A runs on BF16 activations (the reference's precision) where B12X quantized them to
  FP8, and r6k's TileLang tiles were swept on TP4's per-rank shapes. On TP4, r6k's prefill
  was 0.5-0.8% faster than r6h, so the TP3 shapes are the gap to work on: sweep WO-A and WO-B
  tiles for TP3's per-rank shapes (ceiling: about 60 ms of a 16K prefill, 1.5%).
- NCCL grows 10-28 ms per rank, consistent with waiting on slower peers.
- **Engram wait:** r6n's one-thread wait kernel spins 53 / 86 / 119 ms on dgx1 / dgx2 / dgx3
  (1.4-3% of the request) for the rows of the 16K prompt. r6h waits too, as a stream wait that
  is not a kernel, so the profile cannot compare them; the span difference is fully explained
  by WO. It is the larger lever: prefill read-ahead (TODO, Engram after r6n) would take
  the wait off the critical path, and dgx3, the slowest reader, paces every rank.

## The WO ceiling (dgx4, off the fabric, 2026-10-16 07:14 UTC)

[wo-ceiling/bench_wo_ceiling.py](wo-ceiling/bench_wo_ceiling.py) ([run.sh](wo-ceiling/run.sh), r6n
image, kbench timing: CUDA graphs, warm / cold) at the serving shapes: TP3 24 heads in 3 groups a
rank, TP4 16 heads in 2; rows of the sequence-parallel prefill slice (TP3 683, TP4 1024) and
twice that. Cold, us per call:

| shape, rows | B12X fused WO | TileLang WO | inverse RoPE | WO-A (BF16 MMA) | WO-B (MXFP8) | cuBLAS BF16 WO-A |
|---|---:|---:|---:|---:|---:|---:|
| TP3, 683 | 359.5 | 491.5 (1.37x) | 14.3 | 270.3 (64 TFLOP/s) | 206.8 | 295.7 (58 TFLOP/s) |
| TP3, 1366 | 760.8 | 863.2 (1.13x) | 23.6 | 489.8 (70) | 336.9 | 557.0 (62) |
| TP4, 1024 | 368.5 | 464.7 (1.26x) | 14.1 | 255.8 (67) | 187.2 | 274.3 (63) |
| TP4, 2048 | 763.9 | 824.1 (1.08x) | 24.6 | 481.3 (71) | 318.3 | 542.5 (63) |

- **WO-A is at GB10's BF16 ceiling.** It is faster than cuBLAS's BF16 bmm on weights widened
  once, so widening in the loop costs nothing measurable. Of 32 tiles (64-128 rows, 64-128
  columns, 2-3 stages, 128-256 threads), today's is within 0.4% of the best at TP3 683 rows and
  2.4% at TP4 1024; every working tile gives the same bytes.
- **WO-B runs at the FP8 rate** (about 104 TFLOP/s at TP3 683 rows), level with B12X's.
- **The gap is the reference's precision.** WO-A on BF16 activations is a BF16 MMA (GB10: about
  65-70 TFLOP/s against about 105 for FP8). B12X quantized WO-A's activations to FP8, which is
  why it is faster, and is the deviation r6k removed. The same FLOPs as FP8 would take about
  160 us at 683 rows against 270. No tile closes that.
- At decode B12X's one kernel stays ahead (1.03-1.28x cold), as r6k measured.

So the WO cost stays (it is what reference precision costs on this GPU). The prefill time to
win back on both topologies is the Engram wait (above): table 0's rows are read only after the
GPU hashes the chunk at the step's start, about 50 ms before layer 1 needs them, so every chunk
waits 4-36 ms (dgx3 119 ms per 16K prefill, about 3%), while table 1, needed at layer 14, is
always ready.
