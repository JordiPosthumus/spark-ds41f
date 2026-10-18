"""gemm_mxfp8_cast's prefill tiles against production's path (per_token_cast, then
gemm_mxfp8 with the projection's prefill configuration), warm and cold, under CUDA
graphs, at the DSpark context-KV (512 x 5120, five shards) and WO-B (5120 x 2048)
shapes. Every configuration's bits are checked against production's.

Prints the best fused configuration per shape and row count, and its ratio to
production. Exits 1 only if bits differ.
"""
import itertools
import json
import sys

import tile_kernels
import torch

from vllm.models.deepseek_v4_1.tilelang import gemm as g

DEV = torch.device("cuda")
GEN = torch.Generator(device=DEV).manual_seed(29)
SHAPES = {"context-kv": (512, 5120, 5, dict(block_M=128, block_N=64, block_K=128, num_stages=2)),
          "wo-b": (5120, 2048, 1, None)}
ROWS = (256, 512, 2048, 8192)
COPIES = 8


def timed(fn, rounds=10):
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


bad, summary = 0, {}
for name, (n, k, shards, prod_cfg) in SHAPES.items():
    prod_cfg = prod_cfg or g.fp8_prefill_config(n, k)
    ws = [(torch.randn(n, k, device=DEV, generator=GEN) * 0.5).to(torch.float8_e4m3fn) for _ in range(COPIES)]
    wsf = [g.pack_scale_words(torch.randint(118, 130, (n // 32, k // 32), device=DEV, generator=GEN,
                                            dtype=torch.uint8), rows=n) for _ in range(COPIES)]
    rmax = max(ROWS)
    x = torch.randn(rmax, k, device=DEV, generator=GEN).bfloat16()
    xq = torch.empty(rmax, k, dtype=torch.float8_e4m3fn, device=DEV)
    sf = torch.empty(rmax, k // 32, dtype=torch.uint8, device=DEV)
    ref = torch.empty(rmax, n, dtype=torch.bfloat16, device=DEV)
    out = torch.empty(rmax, n, dtype=torch.bfloat16, device=DEV)
    production = g.mxfp8_gemm(n, k, **prod_cfg, shards=shards)
    configs = [dict(block_M=bm, block_N=bn, block_K=bk, num_stages=st, threads=th)
               for bm, bn, bk, st, th in itertools.product((64, 128), (64, 128), (128, 256), (2, 3), (128, 256))
               if (k // bk) % shards == 0 and st * (bm * bk * 3 + bn * bk) <= 96 * 1024]
    kernels = {}
    for cfg in configs:
        try:
            kernels[json.dumps(cfg, sort_keys=True)] = g.mxfp8_gemm_cast(n, k, **cfg, shards=shards)
        except Exception as error:  # noqa: BLE001
            print(f"  {name} {cfg}: {type(error).__name__}: {str(error)[:80]}", flush=True)
    for rows in ROWS:
        def prod_fn(cold, state={"i": 0}):
            def fn():
                i = state["i"] % COPIES if cold else 0
                state["i"] += 1
                tile_kernels.quant.per_token_cast(x[:rows], "e4m3", 32, round_sf=True, use_packed_ue8m0=True,
                                                  out=(xq[:rows], sf[:rows]))
                production(xq[:rows], ws[i], sf[:rows].view(torch.uint32), wsf[i], ref[:rows])
            return fn
        base = (timed(prod_fn(False)), timed(prod_fn(True)))
        prod_fn(False)()
        results = {}
        for key, kernel in kernels.items():
            def fused_fn(cold, kernel=kernel, state={"i": 0}):
                def fn():
                    i = state["i"] % COPIES if cold else 0
                    state["i"] += 1
                    kernel(x[:rows], ws[i], wsf[i], out[:rows])
                return fn
            try:
                t = (timed(fused_fn(False)), timed(fused_fn(True)))
            except Exception as error:  # noqa: BLE001
                print(f"  {name} {rows} {key}: {type(error).__name__}", flush=True)
                torch.cuda.synchronize()
                continue
            kernel(x[:rows], ws[0], wsf[0], out[:rows])
            torch.cuda.synchronize()
            if not torch.equal(out[:rows].view(torch.int16), ref[:rows].view(torch.int16)):
                bad += 1
                print(f"  BITS DIFFER {name} {rows} {key}", flush=True)
            results[key] = t
        best = min(results, key=lambda c: max(results[c][0] / base[0], results[c][1] / base[1]))
        t = results[best]
        print(f"{name} rows {rows:5d}: production {base[0]:7.1f}/{base[1]:7.1f} us  best fused {t[0]:7.1f}/{t[1]:7.1f} us "
              f"ratio {t[0] / base[0]:.3f}/{t[1] / base[1]:.3f}  {best}", flush=True)
        summary.setdefault(name, {})[rows] = {"config": json.loads(best), "fused": t, "production": base}

print(json.dumps(summary))
print(f"{'PASS' if not bad else 'FAIL'}: {bad} bit mismatches")
print(f"{0 if bad else 1} passed")
sys.exit(1 if bad else 0)
