"""CPU stand-ins for the six TileLang kernels of DeepSeek's V4.1 reference (inference/kernel.py).

Each function keeps the reference kernel's signature and its rounding points: block scales from
the same amax rule (power-of-two scales by the same bit arithmetic), the same FP8/FP4 casts, and
FP32 accumulation of exactly dequantized operands. Only the summation order inside a GEMM differs.

REF_ACCUM=float64 accumulates the GEMMs and attention products in FP64 instead of FP32: another valid
summation, to measure how far two correct implementations of this arithmetic drift apart.

REF_WO_A_QUANT=0 skips the quant/dequant of WO-A's input (the reference's 2026-10-12 change), the
only in-place act_quant on a 4-D tensor whose last dim is a group's 4096 head channels.
"""

import os

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
WO_A_QUANT = os.environ.get("REF_WO_A_QUANT", "1") != "0"
ACC = {"float32": torch.float32, "float64": torch.float64}[os.environ.get("REF_ACCUM", "float32")]
STATS = {"wo_a_quant_calls": 0, "wo_a_quant_skipped": 0}


def _log2_ceil_pow2(t: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(t)) for positive normal FP32 t, as fast_log2_ceil/fast_pow2 compute it:
    exponent - 127 + (mantissa != 0)."""
    bits = t.contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man = bits & ((1 << 23) - 1)
    e = exp - 127 + (man != 0).to(torch.int32)
    return ((e + 127) << 23).view(torch.float32)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    N = x.size(-1)
    assert N % block_size == 0
    if inplace and WO_A_QUANT is False and x.ndim == 4 and N == 4096:
        STATS["wo_a_quant_skipped"] += 1
        return x
    if inplace and x.ndim == 4 and N == 4096:
        STATS["wo_a_quant_calls"] += 1
    in_dtype = x.dtype
    z = x.contiguous().float().view(-1, N // block_size, block_size)
    amax = z.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    if scale_fmt is not None:
        s = _log2_ceil_pow2(amax * torch.tensor(1.0 / FP8_MAX, dtype=torch.float32))
    else:
        s = amax * torch.tensor(1.0 / FP8_MAX, dtype=torch.float32)
    q = (z / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        y = (q.float() * s).to(in_dtype).view(x.shape)
        x.copy_(y)
        return x
    return q.view(x.shape), s.view(*x.shape[:-1], N // block_size).to(scale_dtype)


# E2M1 magnitudes; round to nearest, ties to the even mantissa (CUDA's cvt.rn to e2m1).
_FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_FP4_EVEN = torch.tensor([True, False, True, False, True, False, True, False])


def _round_fp4(v: torch.Tensor) -> torch.Tensor:
    a = v.abs().clamp(max=FP4_MAX)
    hi = torch.bucketize(a, _FP4_GRID).clamp(1, 7)  # first grid point >= a
    lo = hi - 1
    g_lo, g_hi = _FP4_GRID[lo], _FP4_GRID[hi]
    d_lo, d_hi = a - g_lo, g_hi - a
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & _FP4_EVEN[hi])
    r = torch.where(pick_hi, g_hi, g_lo)
    return torch.copysign(r, v)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    N = x.size(-1)
    assert N % block_size == 0
    in_dtype = x.dtype
    z = x.contiguous().float().view(-1, N // block_size, block_size)
    amax = z.abs().amax(dim=-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6 * 2.0**-9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2.0**-126)
        s = _log2_ceil_pow2(amax * torch.tensor(1.0 / FP4_MAX, dtype=torch.float32))
    q = _round_fp4((z / s).clamp(-FP4_MAX, FP4_MAX))
    assert inplace, "only the in-place form is used by model.py"
    x.copy_((q * s).to(in_dtype).view(x.shape))
    return x


def _expand_rows(s: torch.Tensor, block: int, n: int) -> torch.Tensor:
    return s.float().repeat_interleave(block, dim=0)[:n]


def _expand_cols(s: torch.Tensor, block: int, k: int) -> torch.Tensor:
    return s.float().repeat_interleave(block, dim=-1)[..., :k]


ROWS = 4096  # weight rows dequantized at a time, to bound memory


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    assert ROWS % block_size == 0
    assert a_s.numel() == M * (K // block_size)
    assert tuple(b_s.shape) == ((N + block_size - 1) // block_size, K // block_size), (b.shape, b_s.shape)
    a_deq = a.view(M, K).float() * _expand_cols(a_s.view(M, -1), block_size, K)
    c = torch.empty(M, N)
    for r in range(0, N, ROWS):
        rows = b[r : r + ROWS]
        s = b_s[r // block_size : (r + rows.size(0) + block_size - 1) // block_size]
        w_scale = _expand_cols(_expand_rows(s, block_size, rows.size(0)), block_size, K)
        c[:, r : r + rows.size(0)] = (a_deq.to(ACC) @ (rows.float() * w_scale).to(ACC).t()).float()
    return c.to(torch.get_default_dtype()).view(*a.shape[:-1], N)


_FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)


def unpack_fp4(b: torch.Tensor) -> torch.Tensor:
    """[N, K//2] packed e2m1 pairs (low nibble first, as convert.py reads them) -> [N, K] FP32."""
    u = b.view(torch.uint8)
    low = _FP4_TABLE[(u & 0x0F).long()]
    high = _FP4_TABLE[((u >> 4) & 0x0F).long()]
    return torch.stack([low, high], dim=-1).flatten(-2)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    assert a_s.numel() == M * (K // act_block_size)
    assert tuple(b_s.shape) == (N, K // 32), (b.shape, b_s.shape)
    a_deq = a.view(M, K).float() * _expand_cols(a_s.view(M, -1), act_block_size, K)
    c = torch.empty(M, N)
    for r in range(0, N, ROWS):
        b_deq = unpack_fp4(b[r : r + ROWS]) * _expand_cols(b_s[r : r + ROWS], 32, K)
        c[:, r : r + b_deq.size(0)] = (a_deq.to(ACC) @ b_deq.to(ACC).t()).float()
    return c.to(torch.get_default_dtype()).view(*a.shape[:-1], N)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """The reference's online softmax over 64-index blocks: FP32 scores from BF16 operands, the
    probabilities cast to BF16 before P @ V (acc_s_cast), the running sum kept in FP32, and the
    sink added to the denominator only."""
    b, m, h, d = q.shape
    topk = topk_idxs.size(-1)
    block = 64
    qf = q.float()
    acc_o = torch.zeros(b, m, h, d)
    sum_exp = torch.zeros(b, m, h)
    scores_max = torch.full((b, m, h), -1e30)
    batch_ix = torch.arange(b).view(b, 1, 1)
    for t in range(0, topk, block):
        idx = topk_idxs[:, :, t : t + block].long()
        if idx.size(-1) < block:
            idx = torch.cat([idx, idx.new_full((b, m, block - idx.size(-1)), -1)], dim=-1)
        valid = idx != -1
        kvb = kv[batch_ix, idx.clamp_min(0)]  # [b, m, block, d]
        kvb = torch.where(valid.unsqueeze(-1), kvb, torch.zeros((), dtype=kvb.dtype))
        acc_s = torch.where(valid, 0.0, -torch.inf).unsqueeze(2).expand(b, m, h, block).clone()
        acc_s += torch.einsum("bmhd,bmkd->bmhk", qf.to(ACC), kvb.to(ACC)).float()
        acc_s *= softmax_scale
        prev = scores_max
        scores_max = torch.maximum(prev, acc_s.amax(dim=-1))
        scale = torch.exp(prev - scores_max)
        p = torch.exp(acc_s - scores_max.unsqueeze(-1))
        sum_exp = sum_exp * scale + p.sum(dim=-1)
        acc_o = acc_o * scale.unsqueeze(-1) + torch.einsum("bmhk,bmkd->bmhd", p.bfloat16().to(ACC), kvb.to(ACC)).float()
    sum_exp = sum_exp + torch.exp(attn_sink.float().view(1, 1, h) - scores_max)
    return (acc_o / sum_exp.unsqueeze(-1)).to(torch.bfloat16)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (m[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = torch.exp(comb - comb.amax(dim=-1, keepdim=True))
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb
