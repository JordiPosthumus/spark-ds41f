"""TileLangWOProjection's batch invariance at every row count and offset.

mixed15 (fifteen streams and a long prompt arriving mid-decode) differs from each prompt
alone with the TileLang WO, from the first token's logprob, while bench_wo.py found no
size whose rows differ from the 8192-row batch. That bench checks 1-96, 512, 2048 and
8192 rows from offset 0; WO-A changes path at 64/65, 96/97 and 128/129 rows and prompts
prefill at any length, and sequence-parallel slices start mid-batch. This checks every
row count 1-300 and sizes across each boundary up to 4096, at offsets 0, 7, 100 and
1000, against the same rows of one 8192-row call (inverse RoPE in place, WO-A, WO-B).

Exits 1 at the first sizes whose rows differ (lists up to 20).
"""
import sys

import torch
from torch import nn

from vllm.models.deepseek_v4_1.tilelang.wo import TileLangWOProjection
from vllm.v1.worker.workspace import use_preallocated_workspace

DEVICE = torch.device("cuda")
HEADS, HEAD_DIM, GROUPS, RANK, HIDDEN, ROPE, POSITIONS = 16, 512, 2, 1024, 5120, 64, 1 << 20
WIDTH = HEADS // GROUPS * HEAD_DIM
CAPACITY = 8192
gen = torch.Generator(device=DEVICE).manual_seed(101)


def block32(n, k):
    layer = nn.Module()
    layer.weight = (torch.randn((n, k), generator=gen, device=DEVICE) * 2).to(torch.float8_e4m3fn)
    exps = torch.randint(118, 123, (n // 32, k // 32), generator=gen, device=DEVICE).to(torch.uint8)
    layer.weight_scale_inv = exps.view(torch.float8_e8m0fnu)
    return layer


wo_a, wo_b = block32(GROUPS * RANK, WIDTH), block32(HIDDEN, GROUPS * RANK)
inv = 1.0 / (10000 ** (torch.arange(0, ROPE, 2, device=DEVICE).double() / ROPE))
angles = torch.arange(POSITIONS, device=DEVICE).double()[:, None] * inv
table = torch.cat((angles.cos(), angles.sin()), dim=-1).float()
del angles
o = torch.randn((CAPACITY, HEADS, HEAD_DIM), generator=gen, device=DEVICE).bfloat16()
positions = torch.randint(0, POSITIONS, (CAPACITY,), generator=gen, device=DEVICE)
wo = TileLangWOProjection(wo_a, wo_b, groups=GROUPS)
scratch = torch.empty(256 << 20, dtype=torch.uint8, device=DEVICE)


def project(start, rows):
    work = o[start:start + rows].clone()  # the projection rotates its input in place
    out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=DEVICE)
    with use_preallocated_workspace(scratch):
        wo(work, positions[start:start + rows], table, out)
    return out


full = project(0, CAPACITY)
sizes = sorted({*range(1, 301), 383, 384, 385, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096})
bad = []
for offset in (0, 7, 100, 1000):
    for rows in sizes:
        if offset + rows > CAPACITY:
            continue
        got = project(offset, rows)
        if not torch.equal(got.view(torch.int16), full[offset:offset + rows].view(torch.int16)):
            n = int((got.view(torch.int16) != full[offset:offset + rows].view(torch.int16)).any(dim=1).sum())
            bad.append((offset, rows, n))
    print(f"offset {offset}: {sum(1 for b in bad if b[0] == offset)} differing sizes", flush=True)
for offset, rows, n in bad[:20]:
    print(f"  offset {offset} rows {rows}: {n} rows differ")
print(f"{'PASS' if not bad else 'FAIL'}: {len(bad)} differing (offset, rows)")
sys.exit(1 if bad else 0)
