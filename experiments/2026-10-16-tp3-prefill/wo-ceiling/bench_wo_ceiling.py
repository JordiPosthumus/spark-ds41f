"""The WO projection at the serving shapes of TP3 and TP4 (r6n image): where TileLang's WO
loses to B12X's FP8 WO on TP3, and how much faster WO-A can get at the same bits.

Shapes (rank 1024, hidden 5120, 512-wide heads, 8 heads a group): TP3 runs 24 heads in 3
groups a rank (the model's 8 groups padded to 9), TP4 16 heads in 2. Rows: the
sequence-parallel prefill slice serving projects (TP3 683: a 4,096-token chunk over 3
owners and 2 slices; TP4 1024: 8,192 over 4 and 2), twice that, and decode rows 16/48/96.

1. B12X's fused WO (inverse RoPE, FP8 WO-A, WO-B) against TileLang's WO end to end and its
   parts: inverse RoPE, WO-A with the fused cast, WO-B.
2. WO-A tiles on the in-kernel path (four K shards fix the bits; the tile does not): every
   config must give the production tile's E4M3 bytes and scales. Each config builds in its
   own process, so one that fails to compile costs only itself.
3. Ceilings for WO-A: cuBLAS BF16 bmm on the weights widened to BF16 once (no widening in
   the loop, no cast epilogue), and the same FLOPs as FP8 (B12X's WO-A share is not separable).

Times are us per call under CUDA graphs, warm / cold (kbench). Prints TFLOP/s for WO-A.
"""
import dataclasses
import json
import subprocess
import sys

import torch
from torch import nn

sys.path.insert(0, "/b")
from kbench import DEVICE, per_call  # noqa: E402

HEAD_DIM, RANK, HIDDEN, ROPE, POSITIONS, SHARDS, N, K = 512, 1024, 5120, 64, 1 << 20, 4, 1024, 4096
SHAPES = {"TP3": dict(heads=24, groups=3, prefill=(683, 1366)), "TP4": dict(heads=16, groups=2, prefill=(1024, 2048))}
DECODE_ROWS = (16, 48, 96)
TODAY = dict(block_M=96, block_N=64, block_K=64, num_stages=2, threads=128)  # wo.py, past 128 rows
CONFIGS = [dict(block_M=bm, block_N=bn, block_K=64, num_stages=st, threads=th)
           for bm in (64, 80, 96, 128) for bn in (64, 128) for st in (2, 3) for th in (128, 256)]


def label(cfg):
    return f"{cfg['block_M']}x{cfg['block_N']}x{cfg['block_K']} s{cfg['num_stages']} t{cfg['threads']}"


def wo_a_inputs(groups, rows_max, seed=3):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    w = torch.randint(0, 256, (groups * N, K), generator=gen, device=DEVICE, dtype=torch.int32).to(torch.uint8)
    w[(w & 0x7F) == 0x7F] = 0x3C
    exps = torch.randint(118, 130, (groups * N // 32, K // 32), generator=gen, device=DEVICE,
                         dtype=torch.int32).to(torch.uint8)
    a = (torch.randn((rows_max, groups * K), generator=gen, device=DEVICE) * 0.5).bfloat16()
    return a, w.view(torch.float8_e4m3fn), exps


def child(cfg, shape):
    """One WO-A tile at one shape: JSON lines of rows, warm, cold and a digest of the bytes."""
    from vllm.models.deepseek_v4_1.tilelang.gemm import fp8w_bf16_gemm_cast
    groups, rows_all = SHAPES[shape]["groups"], SHAPES[shape]["prefill"]
    kernel = fp8w_bf16_gemm_cast(N, K, groups, **cfg, shards=SHARDS)
    a, weight, exps = wo_a_inputs(groups, max(rows_all))
    for rows in rows_all:
        xq = torch.empty((rows, groups * N), dtype=torch.float8_e4m3fn, device=DEVICE)
        sf = torch.empty((rows, groups * N // 32), dtype=torch.uint8, device=DEVICE)

        def fn():
            kernel(a[:rows], weight, exps, xq, sf)
        fn()
        torch.cuda.synchronize()
        digest = int(xq.view(torch.uint8).to(torch.int64).sum().item()) * 1000003 + int(sf.to(torch.int64).sum().item())
        warm, cold = per_call(fn, rows)
        print(json.dumps({"shape": shape, "rows": rows, "cfg": cfg, "warm": warm, "cold": cold, "digest": digest}),
              flush=True)


if len(sys.argv) > 2:
    child(json.loads(sys.argv[1]), sys.argv[2])
    raise SystemExit(0)


def tflops(groups, rows, us):
    return rows * groups * N * K * 2 / (us * 1e-6) / 1e12


def block32(gen, n, k):
    layer = nn.Module()
    layer.weight = (torch.randn((n, k), generator=gen, device=DEVICE) * 2).to(torch.float8_e4m3fn)
    exps = torch.randint(118, 123, (n // 32, k // 32), generator=gen, device=DEVICE).to(torch.uint8)
    layer.weight_scale_inv = exps.view(torch.float8_e8m0fnu)
    return layer


def end_to_end(shape):
    """1 and 3: B12X's fused WO, TileLang's WO and parts, and the cuBLAS BF16 ceiling."""
    from b12x.gemm import wo_projection
    from b12x.preparation import PreparationSession, PreparedCall
    from vllm.models.deepseek_v4_1.tilelang.linear import _activation_specs, _block32_gemm
    from vllm.models.deepseek_v4_1.tilelang.rope import rope_
    from vllm.models.deepseek_v4_1.tilelang.wo import TileLangWOProjection
    from vllm.v1.worker.workspace import use_preallocated_workspace

    heads, groups = SHAPES[shape]["heads"], SHAPES[shape]["groups"]
    width = heads // groups * HEAD_DIM
    rows_all = (*DECODE_ROWS, *SHAPES[shape]["prefill"])
    capacity = 8192
    gen = torch.Generator(device=DEVICE).manual_seed(101)
    wo_a, wo_b = block32(gen, groups * RANK, width), block32(gen, HIDDEN, groups * RANK)
    inv = 1.0 / (10000 ** (torch.arange(0, ROPE, 2, device=DEVICE).double() / ROPE))
    angles = torch.arange(POSITIONS, device=DEVICE).double()[:, None] * inv
    table = torch.cat((angles.cos(), angles.sin()), dim=-1).float()
    del angles
    o = torch.randn((capacity, heads, HEAD_DIM), generator=gen, device=DEVICE).bfloat16()
    positions = torch.randint(0, POSITIONS, (capacity,), generator=gen, device=DEVICE)
    work = o.clone()
    tl = TileLangWOProjection(wo_a, wo_b, groups=groups)
    t_out = torch.empty((capacity, HIDDEN), dtype=torch.bfloat16, device=DEVICE)
    scratch = torch.empty(256 << 20, dtype=torch.uint8, device=DEVICE)
    xq = torch.empty((capacity, groups * RANK), dtype=torch.float8_e4m3fn, device=DEVICE)
    sf = torch.empty((capacity, groups * RANK // 32), dtype=torch.uint8, device=DEVICE)
    partials = torch.empty(SHARDS * 128 * groups * RANK, dtype=torch.float32, device=DEVICE)

    def tilelang(rows):
        with use_preallocated_workspace(scratch):
            tl(work[:rows], positions[:rows], table, t_out[:rows])

    def part_rope(rows):
        rope_(work[:rows], positions[:rows], table, 1, True)

    def part_a(rows):
        split = tl.a.partial_specs(rows)
        p = partials[: SHARDS * rows * groups * RANK].view(SHARDS, rows, groups * RANK) if split else None
        tl.a(work[:rows].view(rows, -1), xq[:rows], sf[:rows], p)

    def part_b(rows):
        tile = _activation_specs(rows, groups * RANK, tl.b.tilelang_decode_rows)[0][0][0]
        _block32_gemm(tl.b, xq[:tile], sf[:tile], t_out[:rows], rows)

    # cuBLAS BF16 ceiling for WO-A: weights widened once, one bmm over the groups.
    exps = wo_a.weight_scale_inv.view(torch.uint8).float()
    w_bf16 = (wo_a.weight.float() * torch.exp2(exps - 127).repeat_interleave(32, 0).repeat_interleave(32, 1))
    w_bf16 = w_bf16.bfloat16().view(groups, RANK, width).transpose(1, 2).contiguous()  # [g, K, N]
    a_bf16 = o.view(capacity, groups, width)
    c_bf16 = torch.empty((groups, capacity, RANK), dtype=torch.bfloat16, device=DEVICE)

    def cublas(rows):
        torch.bmm(a_bf16[:rows].transpose(0, 1), w_bf16, out=c_bf16[:, :rows])

    weights = wo_projection.pack_weights(wo_a.weight, wo_a.weight_scale_inv, wo_b.weight, wo_b.weight_scale_inv,
                                         groups=groups, group_width=width, rank=RANK, hidden=HIDDEN,
                                         block_size=(32, 32))
    invocation = dict(operation="inv_rope", heads_per_group=heads // groups, nope_dim=HEAD_DIM - ROPE, rope_dim=ROPE,
                      positions_dtype="int64", cos_sin_dtype="float32")
    plans, scratches = {}, {}
    for key in (*DECODE_ROWS, "prefill"):
        rows = capacity if key == "prefill" else key
        plans[key] = wo_projection.plan(
            wo_projection.Caps(device=DEVICE, max_tokens=rows, groups=groups, group_width=width, rank=RANK,
                               hidden=HIDDEN),
            invocation=dict(invocation, dynamic_tokens=key == "prefill"))

    def prepare(state):
        rows = state.query.max_tokens
        source = torch.empty((rows, heads, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
        pos = torch.arange(rows, dtype=torch.int64, device=DEVICE)
        sc = tuple(torch.empty(s.shape, dtype=s.dtype, device=DEVICE) for s in state._scratch_state.scratch_specs())
        binding = state.bind_inv_rope(scratch=sc, o=source, positions=pos, cos_sin_cache=table, weights=weights,
                                      heads_per_group=heads // groups, nope_dim=HEAD_DIM - ROPE, rope_dim=ROPE)
        return PreparedCall(run=lambda: state.run_inv_rope(binding), produce=lambda: source.normal_(std=0.25))

    b_out = torch.empty((capacity, HIDDEN), dtype=torch.bfloat16, device=DEVICE)

    def b12x(rows):
        key = rows if rows in DECODE_ROWS else "prefill"
        plan = plans[key]
        if key not in scratches:
            scratches[key] = tuple(torch.empty(s.shape, dtype=s.dtype, device=DEVICE) for s in plan.scratch_specs())
        binding = wo_projection.bind_inv_rope(plan, scratch=scratches[key], o=o[:rows], positions=positions[:rows],
                                              cos_sin_cache=table, weights=weights, heads_per_group=heads // groups,
                                              nope_dim=HEAD_DIM - ROPE, rope_dim=ROPE)
        output = b_out[:rows].as_strided((rows, HIDDEN, 1), (HIDDEN, 1, rows * HIDDEN))
        binding = dataclasses.replace(binding, output=output)
        wo_projection.run_inv_rope(binding=binding, plan=plan, stream=torch.cuda.current_stream().cuda_stream)

    with PreparationSession(device="cuda", autotune=False, compile_workers=2) as session:
        session.prepare(tuple(plan.request(name=f"wo.{key}", prepare_call=prepare) for key, plan in plans.items()))
        print(f"== {shape}: {heads} heads, {groups} groups a rank. us per call, warm / cold", flush=True)
        for rows in rows_all:
            work[:rows].copy_(o[:rows])
            base = per_call(lambda: b12x(rows), rows)
            port = per_call(lambda: tilelang(rows), rows)
            parts = [per_call(lambda f=f: f(rows), rows) for f in (part_rope, part_a, part_b)]
            ceiling = per_call(lambda: cublas(rows), rows)
            print(f"  rows {rows:5d}  b12x {base[0]:8.1f}/{base[1]:8.1f}  tilelang {port[0]:8.1f}/{port[1]:8.1f} "
                  f"({port[1] / base[1]:.2f}x cold)  parts rope {parts[0][0]:.1f}/{parts[0][1]:.1f} "
                  f"wo-a {parts[1][0]:.1f}/{parts[1][1]:.1f} ({tflops(groups, rows, parts[1][1]):.0f} TFLOP/s cold) "
                  f"wo-b {parts[2][0]:.1f}/{parts[2][1]:.1f}  cublas-bf16 wo-a {ceiling[0]:.1f}/{ceiling[1]:.1f} "
                  f"({tflops(groups, rows, ceiling[1]):.0f} TFLOP/s cold)", flush=True)


for shape in SHAPES:
    end_to_end(shape)

print("== WO-A tiles (in-kernel path), us per call warm / cold; bytes must equal today's tile", flush=True)
records, problems = {}, []
for shape in SHAPES:
    for cfg in CONFIGS:
        proc = subprocess.run([sys.executable, __file__, json.dumps(cfg), shape], capture_output=True, text=True,
                              timeout=1200)
        for line in proc.stdout.splitlines():
            if line.startswith("{"):
                r = json.loads(line)
                records.setdefault((r["shape"], r["rows"]), []).append(r)
        if proc.returncode:
            tail = (proc.stderr.strip().splitlines() or ["?"])[-1][:160]
            print(f"  {shape} {label(cfg)}: exit {proc.returncode}: {tail}", flush=True)
for (shape, rows), rs in sorted(records.items()):
    groups = SHAPES[shape]["groups"]
    base = next((r for r in rs if r["cfg"] == TODAY), None)
    if base is None:
        problems.append(f"{shape} rows {rows}: today's tile did not run")
        continue
    bad = [label(r["cfg"]) for r in rs if r["digest"] != base["digest"]]
    if bad:
        problems.append(f"{shape} rows {rows}: bytes differ for {', '.join(bad)}")
    best = sorted(rs, key=lambda r: r["cold"])[:4]
    print(f"  {shape} rows {rows:5d}: today {base['warm']:.1f}/{base['cold']:.1f} "
          f"({tflops(groups, rows, base['cold']):.0f} TFLOP/s); best " + "; ".join(
              f"{r['warm']:.1f}/{r['cold']:.1f} {label(r['cfg'])} ({tflops(groups, rows, r['cold']):.0f})"
              for r in best), flush=True)
for p in problems:
    print("FAIL", p)
print(f"{len(CONFIGS)} configs x {len(SHAPES)} shapes; {'bytes agree' if not problems else 'PROBLEMS'}")
