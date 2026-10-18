"""WO-A's W8A16 GEMM at production's four shards, the tiles the first sweep did not try
(2026-10-11-wo-bf16-sweep): 96- and 128-row tiles for 72-128 rows (one tile reads the
weight once; three 32-row tiles read it three times), and wider tiles, block_K 128,
three stages and 256 threads from 256 rows. At each row count, the
split path (FP32 partials per shard CTA, then splitk_reduce) against the in-kernel path
(the shards added inside one CTA) over tiles, each config in its own process.

With one shard count both paths give a row the same bits, so a dispatch may pick
either per row bucket; the parent checks that every config of a shard count agrees at
every size, and prints the fastest per size and shard count. Exploratory: prints
"N passed" when bits agree.
"""
import json
import subprocess
import sys

import torch

sys.path.insert(0, "/b")
from kbench import DEVICE, per_call  # noqa: E402

from vllm.models.deepseek_v4_1.tilelang.gemm import (  # noqa: E402
    fp8w_bf16_gemm,
    fp8w_bf16_gemm_partials,
    splitk_reduce,
)

N, K, GROUPS = 1024, 4096, 2
SIZES = (64, 72, 96, 128, 256, 512, 2048, 8192)
SPLIT_MAX = 256  # longer steps are not split: their FP32 partials outgrow the GEMM's output
# B12X's whole WO (WO-A and WO-B), warm, from WO window 4, for context.
B12X_WO = {1: 24.1, 8: 23.8, 16: 24.9, 24: 24.5, 32: 25.8, 48: 37.1, 64: 39.2, 96: 61.5,
           512: 213.7, 2048: 807.8, 8192: 4523.6}
TALL = [dict(path=path, shards=4, block_M=bm, block_N=bn, block_K=bk, num_stages=st, threads=128)
        for path in ("split", "inkernel") for bm in (96, 128) for bn in (32, 64) for bk in (64, 128) for st in (2, 3)]
WIDE = [dict(path="inkernel", shards=4, block_M=bm, block_N=bn, block_K=bk, num_stages=st, threads=th)
        for bm in (64, 128) for bn in (64, 128) for bk in (64, 128) for st in (2, 3) for th in (128, 256)]
PRODUCTION = [dict(path="split", shards=4, block_M=64, block_N=64, block_K=64, num_stages=2, threads=128),
              dict(path="inkernel", shards=4, block_M=32, block_N=64, block_K=64, num_stages=2, threads=128),
              dict(path="inkernel", shards=4, block_M=64, block_N=64, block_K=64, num_stages=2, threads=128)]
CONFIGS = PRODUCTION + TALL + WIDE


def child(cfg):
    gen = torch.Generator(device=DEVICE).manual_seed(3)
    w = torch.randint(0, 256, (GROUPS * N, K), generator=gen, device=DEVICE, dtype=torch.int32).to(torch.uint8)
    w[(w & 0x7F) == 0x7F] = 0x3C
    weight = w.view(torch.float8_e4m3fn)
    exps = torch.randint(118, 130, (GROUPS * N // 32, K // 32), generator=gen, device=DEVICE,
                         dtype=torch.int32).to(torch.uint8)
    s = cfg["shards"]
    tile = {key: cfg[key] for key in ("block_M", "block_N", "block_K", "num_stages", "threads")}
    if cfg["path"] == "split":
        k = fp8w_bf16_gemm_partials(N, K, GROUPS, s, **tile)
        reduce = splitk_reduce(GROUPS * N, s)
    else:
        k = fp8w_bf16_gemm(N, K, GROUPS, **tile, shards=s)
    for rows in SIZES:
        if cfg["path"] == "split" and rows > SPLIT_MAX:
            continue  # partials of prefill steps are not a candidate
        a = (torch.randn((rows, GROUPS * K), generator=gen, device=DEVICE) * 0.5).bfloat16()
        out = torch.empty((rows, GROUPS * N), dtype=torch.bfloat16, device=DEVICE)
        if cfg["path"] == "split":
            p = torch.empty((s, rows, GROUPS * N), dtype=torch.float32, device=DEVICE)

            def fn():
                k(a, weight, exps, p)
                reduce(p, out)
        else:
            def fn():
                k(a, weight, exps, out)
        fn()
        torch.cuda.synchronize()
        digest = int(out.view(torch.int16).to(torch.int64).sum().item())
        warm, cold = per_call(fn, rows)
        print(json.dumps({"rows": rows, "cfg": cfg, "warm": warm, "cold": cold, "digest": digest}), flush=True)


if len(sys.argv) > 1:
    child(json.loads(sys.argv[1]))
    raise SystemExit(0)

records, problems = {}, []
for cfg in CONFIGS:
    proc = subprocess.run([sys.executable, __file__, json.dumps(cfg)], capture_output=True, text=True, timeout=900)
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            r = json.loads(line)
            records.setdefault((r["rows"], r["cfg"]["shards"]), []).append(r)
    if proc.returncode:
        print(f"config {cfg}: exit {proc.returncode}")
for shards in (4,):
    print(f"shards {shards}")
    for rows in SIZES:
        rs = records.get((rows, shards), [])
        if not rs:
            continue
        if len({r["digest"] for r in rs}) != 1:
            problems.append(f"shards {shards} rows {rows}: configs disagree in bits")
        best = sorted(rs, key=lambda r: r["warm"])[:3]
        context = f" (B12X whole WO {B12X_WO[rows]:.1f})" if rows in B12X_WO else ""
        print(f"  rows {rows:5d}{context}: " + "; ".join(
            f"{r['warm']:.1f}/{r['cold']:.1f} us {r['cfg']['path']} {r['cfg']['block_M']}x{r['cfg']['block_N']}x{r['cfg']['block_K']} s{r['cfg']['num_stages']} t{r['cfg']['threads']}"
            for r in best))
for p in problems:
    print("FAIL", p)
print(f"{0 if problems else 1} passed")
raise SystemExit(1 if problems else 0)
