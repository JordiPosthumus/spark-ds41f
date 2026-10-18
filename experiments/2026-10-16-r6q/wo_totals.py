"""Prefill time by path from a kernel summary: the WO projection (r6h: B12X's dense GEMM and
its FP8 quantize/inverse-RoPE kernels; r6n: TileLang WO-A, WO-B and inverse RoPE), the Engram
wait, NCCL, and everything else. usage: wo_totals.py LABEL=SUMMARY.json ..."""
import json
import sys

WO_R6H = ("b12x_libdense_gemmDenseGemm", "_quantize_attention_inv_rope", "_quantize_group_major_trg")
WO_R6N = ("gemm_fp8w_bf16", "rope_kernel")
for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    d = json.load(open(path))
    groups = {"wo": 0.0, "wo-b mxfp8 (40x6, 80x2, 40x2, 80x1)": 0.0, "engram wait": 0.0, "nccl": 0.0, "other": 0.0}
    for name, grid, block, calls, ms in d["kernels"]:
        if any(s in name for s in WO_R6H + WO_R6N):
            groups["wo"] += ms
        elif name == "gemm_mxfp8_kernel" and tuple(grid) in ((40, 6, 1), (80, 2, 1), (40, 2, 1), (80, 1, 1)):
            groups["wo-b mxfp8 (40x6, 80x2, 40x2, 80x1)"] += ms
        elif "engram_rows_wait" in name:
            groups["engram wait"] += ms
        elif name.startswith("nccl"):
            groups["nccl"] += ms
        else:
            groups["other"] += ms
    print(f"{label:14s} span {d['span_ms']:8.1f}  " + "  ".join(f"{k} {v:7.1f}" for k, v in groups.items()))
