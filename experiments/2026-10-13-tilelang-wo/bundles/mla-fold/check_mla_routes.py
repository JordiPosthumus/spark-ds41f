"""The sparse MLA plan's three routes, bit for bit, at serving shapes.

SparseMlaV41Plan picks ``split`` (16-head blocks), ``decode`` or ``extend`` by a
forward's row count and promises the same arithmetic on every route. The row trace
found a prompt's prefill row whose layer-0 attention output differed between its own
forward (17 rows) and mixed15's 263-row forward, with the same query, latent and
keys. For TP4 (16 heads) and TP3 (24), SWA-only and indexed layers, this runs every
route on the same 1024 rows (a third with 1-8 keys, as a prompt's first rows have)
and counts differing elements between each pair. Where routes differ it prints the
lines of each kernel's CUDA source that compute exponentials, reciprocals and fused
multiply-adds, to compare their arithmetic.

Exits 1 if any route differs.
"""
import re
import sys

import torch

from vllm.models.deepseek_v4_1.tilelang.sparse_mla import (
    HEAD_DIM,
    IDX_GROUP,
    SWA_GROUP,
    SparseMlaV41Plan,
)

DEVICE = "cuda"
PAD, LAYERS = 256, 3
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def paged(records, page_size):
    slots, record_bytes = records.shape
    pages = slots // page_size
    buffer = torch.full(
        (pages, LAYERS, page_size * record_bytes + PAD), 0xAB, dtype=torch.uint8, device=DEVICE
    )
    pool = buffer[:, 1]
    pool[:, : page_size * record_bytes] = records.view(pages, -1)
    return pool


def pack_swa(kv):
    n = kv.shape[0]
    groups = kv.float().reshape(n, HEAD_DIM // SWA_GROUP, SWA_GROUP)
    exponent = torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448.0)).clamp(-126, 127)
    payload = (groups / torch.exp2(exponent).unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    scales = (exponent + 127).to(torch.uint8)
    return torch.cat([payload.view(torch.uint8).reshape(n, HEAD_DIM), scales], -1).contiguous()


def pack_idx(kv):
    n = kv.shape[0]
    groups = kv.float().reshape(n, HEAD_DIM // IDX_GROUP, IDX_GROUP)
    scales = (groups.abs().amax(-1).clamp_min(6 * 2**-9) / 6.0).clamp(max=448).to(torch.float8_e4m3fn)
    normalized = groups / scales.float().unsqueeze(-1)
    lut = torch.tensor(E2M1, device=DEVICE)
    codes = (normalized.abs().clamp(max=6.0).unsqueeze(-1) - lut).abs().argmin(-1)
    codes = (codes.to(torch.uint8) | (torch.signbit(normalized).to(torch.uint8) << 3)).reshape(n, HEAD_DIM)
    payload = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return torch.cat([payload, scales.view(torch.uint8)], -1).contiguous()


def arithmetic_lines(kernel):
    source = kernel.get_kernel_source()
    keep = re.compile(r"exp2|ex2|__fdividef|rcp|/ |fma|__frcp|__fmul|__expf")
    return [line.strip() for line in source.splitlines() if keep.search(line)]


def check(heads, idx_width, seed, rows=1024, swa_width=128, page=128):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    swa_pages, idx_pages, idx_context = 24, 40, 4096
    q = torch.randn((rows, heads, HEAD_DIM), generator=gen, device=DEVICE).bfloat16()
    sinks = torch.randn((heads,), generator=gen, device=DEVICE)
    swa = pack_swa(torch.randn((swa_pages * page, HEAD_DIM), generator=gen, device=DEVICE))

    def lists(width, extent, min_len):
        lengths = torch.randint(min_len, width + 1, (rows,), generator=gen, device=DEVICE, dtype=torch.int32)
        short = torch.randint(1, 9, (rows,), generator=gen, device=DEVICE, dtype=torch.int32)
        lengths = torch.where(torch.arange(rows, device=DEVICE) % 3 == 0, short.clamp(max=width), lengths)
        chosen = torch.stack([torch.randperm(extent, generator=gen, device=DEVICE)[:width] for _ in range(rows)])
        keep = torch.arange(width, device=DEVICE) < lengths[:, None]
        return torch.where(keep, chosen, -1).to(torch.int32).contiguous(), lengths

    swa_idx, swa_len = lists(swa_width, swa_pages * page, 1)
    plan = SparseMlaV41Plan(
        heads, swa_width, idx_width, page, swa_page_size=page,
        max_rows=rows, decode_max_rows=rows, split_max_rows=rows,
    )  # fmt: skip
    indexed = {}
    if idx_width:
        idx = pack_idx(torch.randn((idx_pages * page, HEAD_DIM), generator=gen, device=DEVICE))
        idx_logical, idx_len = lists(idx_width, idx_context, 0)
        table = torch.stack(
            [torch.randperm(idx_pages, generator=gen, device=DEVICE)[: idx_context // page] for _ in range(rows)]
        ).to(torch.int32)
        indexed = dict(idx_cache=paged(idx, page), idx_indices=idx_logical, idx_lengths=idx_len, idx_page_table=table)
    pool = paged(swa, page)
    outs = {}
    for kernel in plan.kernels:
        out = torch.empty((rows, heads, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
        plan.run(q, pool, swa_idx, swa_len, sinks, out, kernel=kernel, **indexed)
        outs[kernel] = out
    torch.cuda.synchronize()
    bad = 0
    names = list(outs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            diff = outs[a].view(torch.int16) != outs[b].view(torch.int16)
            n = int(diff.sum())
            if n:
                bad += 1
                rows_hit = diff.any(-1).any(-1).nonzero().flatten()
                short_rows = int((swa_len[rows_hit] <= 8).sum())
                print(f"  heads {heads} idx {idx_width} seed {seed}: {a} vs {b}: {n} elements in "
                      f"{rows_hit.numel()} rows ({short_rows} with at most 8 SWA keys)", flush=True)
    if bad == 0:
        print(f"  heads {heads} idx {idx_width} seed {seed}: routes {names} identical", flush=True)
    return bad, plan


def main():
    failures, shown = 0, False
    for heads in (16, 24):
        for idx_width in (0, 512):
            for seed in (1, 2, 3):
                bad, plan = check(heads, idx_width, seed)
                failures += bad
                if bad and not shown:
                    shown = True
                    for name, kernel in plan.kernels.items():
                        print(f"--- {name}: arithmetic lines")
                        for line in arithmetic_lines(kernel):
                            print("   ", line[:200])
    print(f"{'PASS' if not failures else 'FAIL'}: {failures} differing route pairs")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
