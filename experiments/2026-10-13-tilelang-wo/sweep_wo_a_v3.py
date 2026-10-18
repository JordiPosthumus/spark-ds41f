"""WO-A as serving runs it since the fused cast (vLLM wo-fused-cast 51c752bbd), tiles per
row count, at production's four shards (the shard count fixes the bits; the tile and
the path do not).

The split path is the partials GEMM then ``splitk_reduce_cast`` (whose rows per CTA are
swept too); the in-kernel path is ``fp8w_bf16_gemm_cast``. Each times WO-A end to end to
WO-B's MXFP8 input, warm and cold under CUDA graphs, at the verification rows of eight
and sixteen streams (24-128) and at a sequence-parallel prefill slice (1024). The fixed-row
profile put TileLang's WO 0.17 ms a step behind B12X at 48 rows and level at 96.

Every config at a row count must give the same E4M3 bytes and scales. Prints the three
fastest per row count, today's tile first for comparison, and "N passed" when bits agree.
"""
import json
import subprocess
import sys

import torch

sys.path.insert(0, "/b")
from kbench import DEVICE, per_call  # noqa: E402

from vllm.models.deepseek_v4_1.tilelang.gemm import (  # noqa: E402
    fp8w_bf16_gemm_cast,
    fp8w_bf16_gemm_partials,
    splitk_reduce_cast,
)

N, K, GROUPS, SHARDS = 1024, 4096, 2, 4
SIZES = (24, 32, 40, 48, 56, 64, 72, 80, 96, 128, 1024)
SPLIT_MAX = 128
# Today's bucket tiles (vLLM wo.py _WO_A_TILES): split to 128 rows, in-kernel 96 past it.
TODAY = {16: ("split", 16), 32: ("split", 32), 64: ("split", 64), 96: ("split", 96), 128: ("split", 128)}
SPLIT = [dict(path="split", block_M=bm, block_N=bn, block_K=bk, num_stages=st, threads=th, reduce_rows=rr)
         for bm in (16, 32, 48, 64, 96) for bn in (64, 128) for bk in (64, 128)
         for st in (2, 3) for th in (128,) for rr in (4,)]
SPLIT += [dict(path="split", block_M=bm, block_N=64, block_K=64, num_stages=2, threads=128, reduce_rows=rr)
          for bm in (32, 48, 64) for rr in (1, 2, 8)]
INKERNEL = [dict(path="inkernel", block_M=bm, block_N=bn, block_K=bk, num_stages=st, threads=th, reduce_rows=0)
            for bm in (48, 64, 96) for bn in (64, 128) for bk in (64,) for st in (2, 3) for th in (128, 256)]
CONFIGS = SPLIT + INKERNEL


def label(cfg):
    extra = f" r{cfg['reduce_rows']}" if cfg["path"] == "split" else ""
    return (f"{cfg['path']} {cfg['block_M']}x{cfg['block_N']}x{cfg['block_K']} "
            f"s{cfg['num_stages']} t{cfg['threads']}{extra}")


def today(rows):
    bucket = next((b for b in TODAY if rows <= b), None)
    if bucket is None:
        return ("inkernel", 96, 64, 64, 2, 128)
    path, bm = TODAY[bucket]
    return (path, bm, 64, 64, 2, 128)


def child(cfg):
    gen = torch.Generator(device=DEVICE).manual_seed(3)
    w = torch.randint(0, 256, (GROUPS * N, K), generator=gen, device=DEVICE, dtype=torch.int32).to(torch.uint8)
    w[(w & 0x7F) == 0x7F] = 0x3C
    weight = w.view(torch.float8_e4m3fn)
    exps = torch.randint(118, 130, (GROUPS * N // 32, K // 32), generator=gen, device=DEVICE,
                         dtype=torch.int32).to(torch.uint8)
    tile = {key: cfg[key] for key in ("block_M", "block_N", "block_K", "num_stages", "threads")}
    if cfg["path"] == "split":
        k = fp8w_bf16_gemm_partials(N, K, GROUPS, SHARDS, **tile)
        reduce = splitk_reduce_cast(GROUPS * N, SHARDS, block_M=cfg["reduce_rows"])
    else:
        k = fp8w_bf16_gemm_cast(N, K, GROUPS, **tile, shards=SHARDS)
    for rows in SIZES:
        if cfg["path"] == "split" and rows > SPLIT_MAX:
            continue
        a = (torch.randn((rows, GROUPS * K), generator=gen, device=DEVICE) * 0.5).bfloat16()
        xq = torch.empty((rows, GROUPS * N), dtype=torch.float8_e4m3fn, device=DEVICE)
        sf = torch.empty((rows, GROUPS * N // 32), dtype=torch.uint8, device=DEVICE)
        if cfg["path"] == "split":
            p = torch.empty((SHARDS, rows, GROUPS * N), dtype=torch.float32, device=DEVICE)

            def fn():
                k(a, weight, exps, p)
                reduce(p, xq, sf)
        else:
            def fn():
                k(a, weight, exps, xq, sf)
        fn()
        torch.cuda.synchronize()
        digest = int(xq.view(torch.uint8).to(torch.int64).sum().item()) * 1000003 + int(
            sf.to(torch.int64).sum().item())
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
            records.setdefault(r["rows"], []).append(r)
    if proc.returncode:
        print(f"config {label(cfg)}: exit {proc.returncode}")
for rows in SIZES:
    rs = records.get(rows, [])
    if not rs:
        continue
    if len({r["digest"] for r in rs}) != 1:
        problems.append(f"rows {rows}: configs disagree in bits")
    now = today(rows)
    base = next((r for r in rs if (r["cfg"]["path"], r["cfg"]["block_M"], r["cfg"]["block_N"], r["cfg"]["block_K"],
                                    r["cfg"]["num_stages"], r["cfg"]["threads"]) == now
                 and r["cfg"].get("reduce_rows") in (0, 4)), None)
    best = sorted(rs, key=lambda r: max(r["warm"], 0.0) + 0.25 * r["cold"])[:3]
    head = f"today {base['warm']:.1f}/{base['cold']:.1f}" if base else "today -"
    print(f"rows {rows:5d}: {head}; best " + "; ".join(
        f"{r['warm']:.1f}/{r['cold']:.1f} us {label(r['cfg'])}" for r in best), flush=True)
for p in problems:
    print("FAIL", p)
print(f"{0 if problems else 1} passed")
raise SystemExit(1 if problems else 0)
