"""Checks of the CPU kernel stand-ins: FP4/FP8 rounding against ml_dtypes, scale arithmetic,
GEMMs against dense dequantized products, sparse_attn against a plain softmax with sink."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "shim"))
import ml_dtypes
import numpy as np
import torch

import kernel as k

torch.manual_seed(0)
# FP4 rounding: every grid point, every midpoint, and a dense sweep
v = torch.cat([torch.linspace(-7, 7, 20001), torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -5.0])])
ref = torch.from_numpy(v.clamp(-6, 6).numpy().astype(ml_dtypes.float4_e2m1fn).astype(np.float32))
got = k._round_fp4(v.clamp(-6, 6))
bad = (ref != got) & ~((ref == 0) & (got == 0))
assert not bad.any(), (v[bad][:10], ref[bad][:10], got[bad][:10])
# power-of-two ceiling
t = torch.rand(100000) * 10.0 ** torch.randint(-30, 30, (100000,)).float()
t = t[(t > 1e-37) & (t < 1e37)]
want = torch.pow(2.0, torch.ceil(torch.log2(t.double()))).float()
assert torch.equal(k._log2_ceil_pow2(t), want)
# act_quant: in place equals dequantized non-in-place
x = (torch.randn(5, 3, 256) * 3).bfloat16()
q, s = k.act_quant(x.clone(), 32, "ue8m0", torch.float8_e8m0fnu)
y = k.act_quant(x.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True)
deq = (q.float().view(5, 3, 8, 32) * s.float().unsqueeze(-1)).view(5, 3, 256).bfloat16()
assert torch.equal(y, deq)
assert (s.float().log2() == s.float().log2().round()).all()
# fp8_gemm / fp4_gemm vs dense
a = torch.randn(7, 512).bfloat16()
aq, as_ = k.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu)
w = (torch.randn(9000, 512) * 0.05).to(torch.float8_e4m3fn)
ws = torch.pow(2.0, torch.randint(-4, 3, (math.ceil(9000 / 32), 16)).float()).to(torch.float8_e8m0fnu)
c = k.fp8_gemm(aq, as_, w, ws, torch.float8_e8m0fnu, 32)
dense = (aq.float() * as_.float().repeat_interleave(32, -1)) @ (w.float() * ws.float().repeat_interleave(32, 0)[:9000].repeat_interleave(32, 1)).t()
assert torch.allclose(c.float(), dense.bfloat16().float(), rtol=1e-2, atol=1e-2)
packed = torch.randint(-128, 128, (5000, 256), dtype=torch.int8)
w4 = packed.view(torch.float4_e2m1fn_x2)
ws4 = torch.pow(2.0, torch.randint(-6, 0, (5000, 16)).float()).to(torch.float8_e8m0fnu)
c4 = k.fp4_gemm(aq, as_, w4, ws4, torch.float8_e8m0fnu, 32)
u = packed.view(torch.uint8)
vals = torch.stack([k._FP4_TABLE[(u & 15).long()], k._FP4_TABLE[(u >> 4).long()]], -1).flatten(-2)
dense4 = (aq.float() * as_.float().repeat_interleave(32, -1)) @ (vals * ws4.float().repeat_interleave(32, 1)).t()
assert torch.allclose(c4.float(), dense4.bfloat16().float(), rtol=1e-2, atol=1e-2)
# sparse_attn vs plain softmax with sink (differs only by the BF16 cast of P)
b, m, h, d, n = 1, 5, 4, 512, 150
qq = torch.randn(b, m, h, d).bfloat16()
kv = torch.randn(b, n, d).bfloat16()
sink = torch.randn(h)
idx = torch.full((b, m, 140), -1, dtype=torch.int32)
for i in range(m):
    sel = torch.randperm(n)[: 60 + 15 * i]
    idx[0, i, : sel.numel()] = sel.int()
o = k.sparse_attn(qq, kv, sink, idx, d ** -0.5)
for i in range(m):
    sel = idx[0, i][idx[0, i] >= 0].long()
    sc = (qq[0, i].float() @ kv[0, sel].float().t()) * d ** -0.5
    mx = sc.amax(-1, keepdim=True)
    p = torch.exp(sc - mx)
    den = p.sum(-1) + torch.exp(sink - mx.squeeze(-1))
    want = (p @ kv[0, sel].float()) / den.unsqueeze(-1)
    assert torch.allclose(o[0, i].float(), want, rtol=2e-2, atol=2e-3), (o[0, i].float() - want).abs().max()
# Sinkhorn: comb close to doubly stochastic
mixes = torch.randn(1, 3, 24)
pre, post, comb = k.hc_split_sinkhorn(mixes, torch.ones(3), torch.zeros(24))
assert torch.allclose(comb.sum(-1), torch.ones(1, 3, 4), atol=1e-3) and torch.allclose(comb.sum(-2), torch.ones(1, 3, 4), atol=1e-3)
print("shim checks passed")
