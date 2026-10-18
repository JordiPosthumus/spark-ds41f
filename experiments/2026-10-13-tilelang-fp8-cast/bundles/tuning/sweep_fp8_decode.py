"""Decode tiles for block-32 FP8 projections that have none tuned yet (WO-B: 5120 x 2048
per TP4 rank), against the configuration production uses today.

For each tile height (16, 32, 64, and 96 / 128 as wide decode tiles), every decode
configuration (block_N, block_K, stages) is timed as a whole call (production's
activation cast + the decode GEMM over padded rows) at the rows that tile serves,
warm (same weight) and cold (distinct weights), under CUDA graphs. Today's path is the
default decode configuration up to 64 rows and the 128 x 128 prefill tile past it.
Every candidate's bits are checked against the prefill GEMM (decode tiles accumulate
K in the same order, so they must agree).

Prints the winners per height (lowest worst-of-warm/cold ratio to today's path) and a
JSON summary for gemm.py's tables. Exits 1 only if bits differ.
"""
import itertools
import json
import sys

import tile_kernels
import torch

from vllm.models.deepseek_v4_1.tilelang import gemm as g

DEV = torch.device("cuda")
GEN = torch.Generator(device=DEV).manual_seed(23)
SHAPES = {"wo_b": (5120, 2048)}
HEIGHTS = {16: (1, 8, 16), 32: (24, 32), 64: (48, 64), 96: (72, 80, 96), 128: (112, 128)}
COPIES = 8  # distinct weights for cold timing (each beyond L2 together)


def timed(fn, rounds=20):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(rounds):
        graph.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / rounds


summary, bad = {}, 0
for name, (n, k) in SHAPES.items():
    ws = [(torch.randn(n, k, device=DEV, generator=GEN) * 0.5).to(torch.float8_e4m3fn) for _ in range(COPIES)]
    wsf = [g.pack_scale_words(torch.randint(118, 130, (n // 32, k // 32), device=DEV, generator=GEN,
                                            dtype=torch.uint8), rows=n) for _ in range(COPIES)]
    rmax = 128
    x = (torch.randn(rmax, k, device=DEV, generator=GEN)).bfloat16()
    xq = torch.empty(rmax, k, dtype=torch.float8_e4m3fn, device=DEV)
    sf = torch.empty(rmax, k // 32, dtype=torch.uint8, device=DEV)
    out = torch.empty(rmax, n, dtype=torch.bfloat16, device=DEV)
    ref = torch.empty(rmax, n, dtype=torch.bfloat16, device=DEV)
    prefill = g.mxfp8_gemm(n, k, **g.fp8_prefill_config(n, k))
    default = {bm: g.mxfp8_gemm_decode(n, k, **g.fp8_decode_config(n, k, bm), padded_rows=True) for bm in (16, 32, 64)}

    def call(kernel, rows, padded, cold):
        state = {"i": 0}

        def fn():
            i = state["i"] % COPIES if cold else 0
            state["i"] += 1
            tile_kernels.quant.per_token_cast(x[:rows], "e4m3", 32, round_sf=True, use_packed_ue8m0=True,
                                              out=(xq[:rows], sf[:rows]))
            kernel(xq[:padded], ws[i], sf[:padded].view(torch.uint32), wsf[i], out[:rows])
        return fn

    def today(rows):
        if rows <= 64:
            bm = next(h for h in (16, 32, 64) if rows <= h)
            return default[bm], -(-rows // bm) * bm
        return prefill, rows

    entry = summary.setdefault(name, {})
    for height, rows_list in HEIGHTS.items():
        configs = [dict(block_N=bn, block_K=bk, num_stages=st)
                   for bn, bk, st in itertools.product((32, 64, 128), (128, 256), (2, 3, 4))
                   if st * (height + bn) * bk <= 92 * 1024]
        results = {}
        for cfg in configs:
            try:
                kernel = g.mxfp8_gemm_decode(n, k, block_M=height, **cfg, padded_rows=True)
            except Exception as error:  # noqa: BLE001
                print(f"  {name} m{height} {cfg}: {type(error).__name__}", flush=True)
                continue
            ratios, times = [], []
            for rows in rows_list:
                padded = -(-rows // height) * height
                base_kernel, base_padded = today(rows)
                for cold in (False, True):
                    t = timed(call(kernel, rows, padded, cold))
                    t0 = timed(call(base_kernel, rows, base_padded, cold))
                    ratios.append(t / t0)
                    times.append((rows, cold, round(t, 1), round(t0, 1)))
                # bits against the prefill GEMM on the same activations
                call(kernel, rows, padded, False)()
                prefill(xq[:rows], ws[0], sf[:rows].view(torch.uint32), wsf[0], ref[:rows])
                torch.cuda.synchronize()
                if not torch.equal(out[:rows].view(torch.int16), ref[:rows].view(torch.int16)):
                    bad += 1
                    print(f"  BITS DIFFER {name} m{height} {cfg} rows {rows}", flush=True)
            results[json.dumps(cfg, sort_keys=True)] = (max(ratios), times)
        best = min(results, key=lambda c: results[c][0])
        worst, times = results[best]
        print(f"{name} m{height}: best {best} worst ratio to today {worst:.3f}; "
              + ", ".join(f"{r}{'c' if c else 'w'} {t}/{t0}" for r, c, t, t0 in times), flush=True)
        entry[height] = {"config": json.loads(best), "worst_ratio": worst, "times": times}

print(json.dumps(summary))
print(f"{'PASS' if not bad else 'FAIL'}: {bad} bit mismatches")
sys.exit(1 if bad else 0)
