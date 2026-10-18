"""Our MXFP8 activation cast against DeepSeek's reference act_quant, bit for bit.

Production quantizes every FP8 linear's activations with TileKernels'
per_token_cast (E4M3, 32-element blocks, power-of-two UE8M0 scales). DeepSeek's
reference (inference/kernel.py in the checkpoint) does it with act_quant(x, 32,
"ue8m0", float8_e8m0fnu): amax floored at 1e-4, s = 2^ceil(log2(amax / 448)),
clamp(x / s, -448, 448) to E4M3. Both are compared on the same BF16 inputs: random
data at several magnitudes, all-zero blocks, blocks below the 1e-4 floor, BF16
subnormals, block maxima on exact powers of two, and values near the clamp.

usage: check_cast_reference.py REFERENCE_INFERENCE_DIR
Exits 1 if any FP8 byte or scale byte differs.
"""
import sys

import torch

sys.path.insert(0, sys.argv[1])
import kernel as reference  # noqa: E402  (the checkpoint's inference/kernel.py)
import tile_kernels  # noqa: E402

DEV = torch.device("cuda")
GEN = torch.Generator(device=DEV).manual_seed(11)
K = 5120


def ours(x):
    rows, k = x.shape
    xq = torch.empty(rows, k, dtype=torch.float8_e4m3fn, device=DEV)
    sf = torch.empty(rows, k // 32, dtype=torch.uint8, device=DEV)
    tile_kernels.quant.per_token_cast(x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True, out=(xq, sf))
    return xq.view(torch.uint8), sf


def theirs(x):
    y, s = reference.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    return y.view(torch.uint8), s.view(torch.uint8)


def case(name, x):
    x = x.to(torch.bfloat16).contiguous()
    a, sa = ours(x)
    b, sb = theirs(x)
    data_bad = int((a != b).sum())
    scale_bad = int((sa != sb).sum())
    blocks = x.shape[0] * x.shape[1] // 32
    if data_bad or scale_bad:
        idx = (sa != sb).nonzero()[:3].tolist() if scale_bad else (a != b).nonzero()[:3].tolist()
        print(f"{name}: DIFF data {data_bad} of {a.numel()} bytes, scales {scale_bad} of {blocks}; first {idx}")
    else:
        print(f"{name}: identical ({a.numel()} bytes, {blocks} scales)")
    return data_bad + scale_bad


def rand(rows, scale):
    return torch.randn(rows, K, device=DEV, generator=GEN) * scale


bad = 0
for scale in (1e-3, 0.05, 1.0, 37.0, 1e3, 3e5):
    bad += case(f"normal x{scale:g}", rand(64, scale))
bad += case("heavy tails", rand(64, 1.0) * torch.exp(rand(64, 2.0)))
zeros = rand(32, 1.0)
zeros[:, :64] = 0
zeros[5] = 0
bad += case("all-zero blocks", zeros)
tiny = rand(32, 1.0)
tiny[:, 32:96] = rand(32, 1e-6)[:, 32:96]
bad += case("blocks below the 1e-4 floor", tiny)
sub = rand(16, 1e-39)
bad += case("bf16 subnormals", sub)
pow2 = rand(32, 1.0)
for i in range(32):
    pow2[i, ::32] = 448.0 * 2.0 ** (i - 16)  # block maxima where amax / 448 is an exact power of two
bad += case("block maxima on powers of two", pow2)
edge = rand(32, 1.0)
edge[:, ::32] = 448.0 * 1.00390625  # just above a power-of-two step
edge[:, 1::32] = -edge[:, ::32]
bad += case("near the clamp", edge)
for rows in (1, 7, 33, 200):
    bad += case(f"rows {rows}", rand(rows, 2.0))

print(f"{'PASS' if not bad else 'FAIL'}: {bad} differing bytes")
sys.exit(1 if bad else 0)
