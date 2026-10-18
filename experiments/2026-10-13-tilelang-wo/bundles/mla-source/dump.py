"""Full CUDA source of the sparse MLA's extend and decode kernels (SWA-only, 16 heads,
IEEE math), then route equality on rows with exactly one, two and three keys."""
import os
import sys

os.environ["VLLM_DS41_MLA_FAST_MATH"] = "0"
import torch  # noqa: E402

from vllm.models.deepseek_v4_1.tilelang.sparse_mla import HEAD_DIM, SparseMlaV41Plan  # noqa: E402

sys.path.insert(0, "/b")
import check_mla_routes as c  # noqa: E402  (runs its own check at import; output kept)

plan = SparseMlaV41Plan(16, 128, 0, 128, swa_page_size=128, max_rows=4096, decode_max_rows=4096, split_max_rows=4096)
for name in ("extend", "decode"):
    print(f"===== SOURCE {name}")
    print(plan.kernels[name].get_kernel_source())
    print(f"===== END {name}")

gen = torch.Generator(device="cuda").manual_seed(11)
rows, heads = 4096, 16
for keys in (1, 2, 3):
    q = torch.randn((rows, heads, HEAD_DIM), generator=gen, device="cuda").bfloat16()
    sinks = torch.randn((heads,), generator=gen, device="cuda")
    swa = c.pack_swa(torch.randn((24 * 128, HEAD_DIM), generator=gen, device="cuda"))
    pool = c.paged(swa, 128)
    idx = torch.full((rows, 128), -1, dtype=torch.int32, device="cuda")
    idx[:, :keys] = torch.randint(0, 24 * 128, (rows, keys), generator=gen, device="cuda", dtype=torch.int32)
    lengths = torch.full((rows,), keys, dtype=torch.int32, device="cuda")
    outs = {}
    for name in plan.kernels:
        out = torch.empty((rows, heads, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
        plan.run(q, pool, idx, lengths, sinks, out, kernel=name)
        outs[name] = out
    for name in ("decode", "split"):
        diff = outs["extend"].view(torch.int16) != outs[name].view(torch.int16)
        print(f"{keys} keys: extend vs {name}: {int(diff.sum())} elements in {int(diff.any(-1).any(-1).sum())} rows")
