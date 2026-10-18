"""Time the sparse MLA plan per route at serving shapes (CUDA graphs, warm).

TP4 (16 heads) and TP3 (24), SWA-only (128 keys per row) and indexed (128 + 512), at the
row counts each route serves under the plan's default thresholds. Run once per
VLLM_DS41_MLA_FAST_MATH setting to price IEEE math.
"""
import os

import torch

from vllm.models.deepseek_v4_1.tilelang.sparse_mla import HEAD_DIM, SparseMlaV41Plan

DEVICE = "cuda"
GEN = torch.Generator(device=DEVICE).manual_seed(7)
ROWS = (1, 16, 48, 96, 144, 263, 1024, 4096, 8192)
PAGE = 128


def timed(fn, rounds=50):
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


fast = os.environ.get("VLLM_DS41_MLA_FAST_MATH", "1")
for heads in (16, 24):
    for idx_width in (0, 512):
        plan = SparseMlaV41Plan(heads, 128, idx_width, PAGE, swa_page_size=PAGE, max_rows=max(ROWS))
        swa_pool = torch.randint(0, 120, (64, PAGE * 528), dtype=torch.uint8, device=DEVICE, generator=GEN)
        idx_pool = torch.randint(0, 120, (64, PAGE * 288), dtype=torch.uint8, device=DEVICE, generator=GEN)
        line = []
        for rows in ROWS:
            q = torch.randn((rows, heads, HEAD_DIM), device=DEVICE, generator=GEN).bfloat16()
            sinks = torch.randn((heads,), device=DEVICE, generator=GEN)
            swa_idx = torch.randint(0, 64 * PAGE, (rows, 128), dtype=torch.int32, device=DEVICE, generator=GEN)
            swa_len = torch.full((rows,), 128, dtype=torch.int32, device=DEVICE)
            out = torch.empty((rows, heads, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
            indexed = {}
            if idx_width:
                indexed = dict(
                    idx_cache=idx_pool,
                    idx_indices=torch.randint(0, 64 * PAGE, (rows, idx_width), dtype=torch.int32, device=DEVICE, generator=GEN),
                    idx_lengths=torch.full((rows,), idx_width, dtype=torch.int32, device=DEVICE),
                    idx_page_table=torch.arange(64, dtype=torch.int32, device=DEVICE).repeat(rows, 1),
                )
            kernel = plan.kernel_for(rows)
            scratch = [torch.empty(s, dtype=d, device=DEVICE) for s, d in plan.scratch_specs(rows)]
            t = timed(lambda: plan.run(q, swa_pool, swa_idx, swa_len, sinks, out, scratch=scratch, **indexed))
            line.append(f"{rows}:{kernel[0]} {t:.1f}")
        print(f"fast_math={fast} heads {heads} idx {idx_width}: " + ", ".join(line), flush=True)
