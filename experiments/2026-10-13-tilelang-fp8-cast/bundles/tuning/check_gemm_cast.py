"""gemm_mxfp8_cast (the activation cast inside the GEMM) against production's
per_token_cast + gemm_mxfp8, bit for bit, then time per call.

per_token_cast is byte-identical to DeepSeek's reference act_quant
(experiments/2026-10-13-tilelang-wo/check_cast_reference.py), so equal bits here mean
the fused kernel keeps the reference's quantization.

Shapes (per TP4 rank): the DSpark context-KV projection (port B6, 512 x 5120) and WO-B
(5120 x 2048). Rows 1-8192; shards 1 and, for the narrow B6 shape, 5.

usage: check_gemm_cast.py [--time]
Exits 1 if any output differs.
"""
import sys

import tile_kernels
import torch

from vllm.models.deepseek_v4_1.tilelang import gemm as g

DEV = torch.device("cuda")
GEN = torch.Generator(device=DEV).manual_seed(5)
SHAPES = {"context-kv": (512, 5120, (1, 5)), "wo-b": (5120, 2048, (1,))}
ROWS = (1, 7, 64, 65, 200, 512, 2048, 8192)
TIME = "--time" in sys.argv


def weights(n, k):
    w = (torch.randn(n, k, device=DEV, generator=GEN) * 0.5).to(torch.float8_e4m3fn)
    exps = torch.randint(118, 130, (n // 32, k // 32), device=DEV, generator=GEN, dtype=torch.uint8)
    return w, g.pack_scale_words(exps, rows=n)


def timed(fn, rounds=20):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rounds):
        graph.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000 / rounds


bad = 0
for name, (n, k, shard_counts) in SHAPES.items():
    w, wsf = weights(n, k)
    for shards in shard_counts:
        plain = g.mxfp8_gemm(n, k, shards=shards)
        fused = g.mxfp8_gemm_cast(n, k, shards=shards)
        for rows in ROWS:
            x = (torch.randn(rows, k, device=DEV, generator=GEN) * torch.exp(torch.randn(rows, 1, device=DEV, generator=GEN))).bfloat16()
            x[:, :64] *= 1e-6  # some blocks below the amax floor
            xq = torch.empty(rows, k, dtype=torch.float8_e4m3fn, device=DEV)
            sf = torch.empty(rows, k // 32, dtype=torch.uint8, device=DEV)
            ref = torch.empty(rows, n, dtype=torch.bfloat16, device=DEV)
            out = torch.empty(rows, n, dtype=torch.bfloat16, device=DEV)

            def production():
                tile_kernels.quant.per_token_cast(x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True, out=(xq, sf))
                plain(xq, w, sf.view(torch.uint32), wsf, ref)

            def one_kernel():
                fused(x, w, wsf, out)

            production()
            one_kernel()
            torch.cuda.synchronize()
            diff = int((ref.view(torch.int16) != out.view(torch.int16)).sum())
            bad += diff
            line = f"{name} shards {shards} rows {rows:5d}: " + ("identical" if not diff else f"DIFF {diff} of {ref.numel()}")
            if TIME:
                a, b = timed(production), timed(one_kernel)
                line += f"  cast+gemm {a:8.1f} us  fused {b:8.1f} us  ratio {b / a:.3f}"
            print(line, flush=True)

print(f"{'PASS' if not bad else 'FAIL'}: {bad} differing outputs")
sys.exit(1 if bad else 0)
