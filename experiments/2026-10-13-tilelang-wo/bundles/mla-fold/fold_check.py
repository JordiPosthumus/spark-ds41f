"""Which arithmetic the extend route's segment fold performs (SWA-only, IEEE math).

Runs decode with explicit scratch, so its per-segment state (Partial, PartialStats) is
visible, and extend on the same rows (two or three keys each). Recomputes the fold from
decode's state in several arithmetic variants and counts, over the elements where the
routes differ, which variant reproduces each route's bits.
"""
import os

os.environ["VLLM_DS41_MLA_FAST_MATH"] = "0"
import sys  # noqa: E402

import torch  # noqa: E402

from vllm.models.deepseek_v4_1.tilelang.sparse_mla import HEAD_DIM, SparseMlaV41Plan  # noqa: E402

sys.path.insert(0, "/b")
import check_mla_routes as c  # noqa: E402

LOG2E = torch.tensor(1.44269504, dtype=torch.float32, device="cuda")


def f32(x):
    return x.to(torch.float32)


def fma(a, b, x):
    return f32(a.double() * b.double() + x.double())


for heads in (16, 24):
    gen = torch.Generator(device="cuda").manual_seed(heads)
    rows = 4096
    plan = SparseMlaV41Plan(heads, 128, 0, 128, swa_page_size=128, max_rows=rows, decode_max_rows=rows, split_max_rows=rows)
    q = torch.randn((rows, heads, HEAD_DIM), generator=gen, device="cuda").bfloat16()
    sinks = torch.randn((heads,), generator=gen, device="cuda")
    pool = c.paged(c.pack_swa(torch.randn((24 * 128, HEAD_DIM), generator=gen, device="cuda")), 128)
    keys = 2 + (torch.arange(rows, device="cuda") % 2).to(torch.int32)
    idx = torch.full((rows, 128), -1, dtype=torch.int32, device="cuda")
    idx[:, :3] = torch.randint(0, 24 * 128, (rows, 3), generator=gen, device="cuda", dtype=torch.int32)
    idx[keys == 2, 2] = -1
    scratch = [torch.zeros(s, dtype=d, device="cuda") for s, d in plan.scratch_specs(rows, "decode")]
    dec = torch.empty((rows, heads, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    ext = torch.empty_like(dec)
    plan.run(q, pool, idx, keys, sinks, dec, kernel="decode", scratch=scratch)
    plan.run(q, pool, idx, keys, sinks, ext, kernel="extend")
    torch.cuda.synchronize()
    partial, stats = scratch[0][:, 0], scratch[1][:, 0]  # [rows, heads, 512], [rows, heads, 2]
    m, l = stats[..., 0], stats[..., 1]
    M = f32(sinks * LOG2E)[None, :].expand(rows, heads)
    m_new = torch.maximum(M, m)
    a, b = torch.exp2(M - m_new), torch.exp2(m - m_new)
    variants = {}
    L = fma(torch.ones_like(a), a, f32(l * b))
    o = f32(partial * b[..., None])
    variants["decode: o*(1/L)"] = f32(o * f32(1.0 / L)[..., None]).bfloat16()
    variants["o/L"] = f32(o / L[..., None]).bfloat16()
    Lc = fma(l, b, a)
    variants["L=fma(l,b,a), o*(1/L)"] = f32(o * f32(1.0 / Lc)[..., None]).bfloat16()
    variants["L=fma(l,b,a), o/L"] = f32(o / Lc[..., None]).bfloat16()
    variants["L=a+l*b unfused, o*(1/L)"] = f32(o * f32(1.0 / f32(a + f32(l * b)))[..., None]).bfloat16()
    variants["o=P*b*(1/L) reassociated"] = f32(partial * f32(b * f32(1.0 / L))[..., None]).bfloat16()
    diff = dec.view(torch.int16) != ext.view(torch.int16)
    print(f"heads {heads}: {int(diff.sum())} elements differ between decode and extend")
    for name, v in variants.items():
        same_dec = int((v.view(torch.int16) == dec.view(torch.int16)).logical_not().sum())
        on_diff_ext = int((v.view(torch.int16) == ext.view(torch.int16))[diff].sum())
        print(f"  {name:32s}: differs from decode in {same_dec:6d}; matches extend on {on_diff_ext} of {int(diff.sum())} differing")
    if int(diff.sum()):
        r, h, d = [int(x) for x in diff.nonzero()[0]]
        print(f"  first: row {r} head {h} col {d}: decode {dec[r, h, d].item()!r} extend {ext[r, h, d].item()!r} "
              f"partial {partial[r, h, d].item()!r} m {m[r, h].item()!r} l {l[r, h].item()!r} M {M[r, h].item()!r}")
