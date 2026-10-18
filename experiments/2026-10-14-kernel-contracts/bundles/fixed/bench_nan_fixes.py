"""The three NaN fixes (shared SwiGLU clamps, Engram gate floor, indexer score ReLU)
against r6k's kernels, which this bundle carries as r6k_*.py beside the fixed modules
it mounts.

1. Bits: on finite inputs each fixed kernel writes r6k's bits, signed zeros included
   (compared as integers), at every size.
2. Time per call under CUDA graphs, warm and cold, at the decode capture sizes and
   prefill rows (the indexer at decode rows over 64K and 1M positions and at prefill
   rows over shorter contexts).

Exits 1 if a fixed kernel differs in any bit or is more than 3% slower than r6k's at
any shape.
"""
import importlib.util
import sys

import torch

sys.path.insert(0, "/b")
from kbench import CAPACITY, DECODE, DEVICE, PREFILL, per_call, slower  # noqa: E402

import vllm.models.deepseek_v4_1.tilelang as package  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import activation, engram, indexer  # noqa: E402

failures = []


class _Op:
    """Stands in for ``torch.library.custom_op`` while r6k's engram.py loads, so its
    op does not register a second time under the same name."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def register_fake(self, fn):
        return fn


def r6k(name):
    spec = importlib.util.spec_from_file_location(
        f"{package.__name__}.r6k_{name}", f"/b/r6k_{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    custom_op = torch.library.custom_op
    torch.library.custom_op = lambda *args, **kwargs: _Op
    try:
        spec.loader.exec_module(module)
    finally:
        torch.library.custom_op = custom_op
    return module


def same_bits(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def compare(name, rows, old, new, outs):
    """Run both, check bits, time both; ``outs`` are the two outputs."""
    old()
    new()
    torch.cuda.synchronize()
    if not same_bits(*outs):
        failures.append(f"{name} rows {rows}: bits differ from r6k")
    base, port = per_call(old, rows), per_call(new, rows)
    flag = "  SLOWER" if slower(base, port) else ""
    if flag:
        failures.append(f"{name} rows {rows}: slower")
    print(
        f"  {name:9s} rows {rows:6d}  r6k {base[0]:9.2f} / {base[1]:9.2f}"
        f"  fixed {port[0]:9.2f} / {port[1]:9.2f}"
        f"  ratio {port[0] / base[0]:.3f} / {port[1] / base[1]:.3f}{flag}",
        flush=True,
    )


gen = torch.Generator(device=DEVICE).manual_seed(23)
SIZES = DECODE + PREFILL

# Shared expert SwiGLU at TP4's 576-wide rows, limit 10.
print("shared SwiGLU (us per call, warm / cold)", flush=True)
old_act = r6k("activation")
N, LIMIT = 576, 10.0
x = (torch.randn((CAPACITY, 2 * N), generator=gen, device=DEVICE) * 6).bfloat16()
y_old, y_new = (torch.empty((CAPACITY, N), dtype=torch.bfloat16, device=DEVICE) for _ in range(2))
k_old, k_new = old_act.swiglu_clamp_kernel(N, LIMIT), activation.swiglu_clamp_kernel(N, LIMIT)
for rows in SIZES:
    compare(
        "swiglu",
        rows,
        lambda r=rows: k_old(x[:r], y_old[:r]),
        lambda r=rows: k_new(x[:r], y_new[:r]),
        (y_old[:rows], y_new[:rows]),
    )

# Engram gate, 4 streams x 5120, a quarter of the tokens images.
print("Engram gate (us per call, warm / cold)", flush=True)
old_engram = r6k("engram")
HC, DIM, EPS = 4, 5120, 1e-6
xs = torch.randn((CAPACITY, HC * DIM), generator=gen, device=DEVICE).bfloat16()
kv = torch.randn((CAPACITY, (HC + 1) * DIM), generator=gen, device=DEVICE).bfloat16()
weight = torch.rand((HC * DIM,), generator=gen, device=DEVICE) + 0.5
image = torch.rand((CAPACITY,), generator=gen, device=DEVICE) < 0.25
g_old, g_new = torch.empty_like(xs), torch.empty_like(xs)
e_old = old_engram.engram_gate_kernel(HC, DIM, EPS, True)
e_new = engram.engram_gate_kernel(HC, DIM, EPS, True)


def gate(kernel, out, rows):
    kernel(
        xs[:rows].view(rows, HC, DIM),
        kv[:rows].view(rows, HC + 1, DIM),
        weight.view(HC, DIM),
        image[:rows],
        out[:rows].view(rows, HC, DIM),
    )


for rows in SIZES:
    compare(
        "engram",
        rows,
        lambda r=rows: gate(e_old, g_old, r),
        lambda r=rows: gate(e_new, g_new, r),
        (g_old[:rows], g_new[:rows]),
    )

# Indexer scores: MXFP4 queries and keys with finite scales, 64-position pages.
print("indexer score (us per call, warm / cold)", flush=True)
old_indexer = r6k("indexer")
PAGE = 64
HEADS = indexer.INDEX_HEADS
for rows, width in (
    (1, 65536),
    (16, 65536),
    (96, 65536),
    (1, 1 << 20),
    (16, 1 << 20),
    (512, 16384),
    (2048, 8192),
):
    pages = width // PAGE
    nbytes = indexer.page_bytes(PAGE)
    q = torch.randint(0, 256, (rows, HEADS, 64), generator=gen, device=DEVICE).to(torch.uint8)
    q_scales = torch.randint(118, 130, (rows, HEADS, 4), generator=gen, device=DEVICE).to(torch.uint8)
    weights = (torch.randn((rows, HEADS), generator=gen, device=DEVICE) / 64).bfloat16()
    pool = torch.randint(0, 256, (pages, nbytes), generator=gen, device=DEVICE).to(torch.uint8)
    pool[:, PAGE * 64 : PAGE * 68] = torch.randint(
        118, 130, (pages, PAGE * 4), generator=gen, device=DEVICE
    ).to(torch.uint8)
    table = torch.stack(
        [torch.randperm(pages, generator=gen, device=DEVICE) for _ in range(rows)]
    ).to(torch.int32)
    lengths = torch.full((rows,), width, dtype=torch.int32, device=DEVICE)
    active = torch.tensor([width], dtype=torch.int32, device=DEVICE)
    args = (
        q.view(torch.int8),
        q_scales.view(torch.uint32).reshape(rows, -1),
        weights,
        pool.view(torch.int8),
        pool.view(torch.uint32),
        table,
        lengths,
        active,
    )
    s_old, s_new = (torch.empty((rows, width), dtype=torch.bfloat16, device=DEVICE) for _ in range(2))
    score_old = old_indexer.indexer_score_v41(width, PAGE)
    score_new = indexer.indexer_score_v41(width, PAGE)
    compare(
        f"index {width >> 10}K",
        rows,
        lambda: score_old(*args, s_old),
        lambda: score_new(*args, s_new),
        (s_old, s_new),
    )
    del q, q_scales, weights, pool, table, s_old, s_new
    torch.cuda.empty_cache()

if failures:
    print("FAIL:", *failures, sep="\n  ")
    sys.exit(1)
print("PASS: r6k's bits everywhere, never more than 3% slower")
