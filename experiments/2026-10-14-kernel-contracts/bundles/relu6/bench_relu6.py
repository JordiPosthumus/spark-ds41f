"""Sixth round (relu6): the NaN-keeping indexer ReLU and index quantizer at no cost.

Rounds 4 and 5 found:
- ``clamp`` (``T.clamp(dot, 0, inf)`` on the FP32 dot) keeps the NaN with r6k's bits, worst
  1.010 at an 8K x 2048 prefill chunk: FP32 ``tl::clamp`` is three NaN tests, ``fmaxf`` and
  ``fminf`` per element, against r6k's one ``fmaxf``;
- the NaN-keeping quantizer has r6k's bytes, but its time ratios ran from 0.89 to 1.08: a
  kernel of a few microseconds timed three calls to a replay.
Here:
- ``clamp16``: the same clamp on the BF16 dot, which the score already rounds to BF16 before
  its ReLU. TileLang lowers a BF16 pair to ``tl::clamp2``, a packed NaN-keeping max and min
  (``__hmax2_nan``, ``__hmin2_nan``): two instructions for two elements, as r6k's two
  ``fmaxf``. Scores keep r6k's bits: a ReLU of the rounded value is exact in BF16, and a zero's
  sign cannot reach a score (the head sum starts at +0);
- precise time: a warm replay holds enough calls to last 1.5 ms, cold is one evicted call,
  both the median over many replays; five rounds alternate which kernel runs first; every
  table has an r6k-against-r6k row, the noise floor of the comparison;
- the generated CUDA of clamp16's ReLU lines, to show the packed form.
Bits and bytes on finite inputs must equal r6k's; the bar is 1% at every shape and size.
"""
import importlib.util
import math
import re
import statistics
import sys
import traceback

import torch

sys.path.insert(0, "/b")
from kbench import _EVICT, DEVICE, graph  # noqa: E402

import vllm.models.deepseek_v4_1.tilelang as package  # noqa: E402
from tests.kernels.attention.test_deepseek_v41_tilelang import (  # noqa: E402
    _mxfp4_inputs,
    _page_buffer,
)

ROUNDS, BUDGET = 5, 1.01
WARM_US, MAX_CALLS = 1500.0, 4096
WARM_REPLAYS, COLD_REPLAYS = 31, 61
PAGE = 64
DUMP_SHAPE = (16, 65536)


class _Op:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def register_fake(self, fn):
        return fn


def load(path, name):
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    custom_op = torch.library.custom_op
    torch.library.custom_op = lambda *args, **kwargs: _Op
    try:
        spec.loader.exec_module(module)
    finally:
        torch.library.custom_op = custom_op
    return module


def _replay_us(g, cold=False):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    if cold:
        _EVICT.max()
    start.record()
    g.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000


def per_call(fn):
    """Median microseconds per call, warm (a replay of enough calls to last WARM_US) and
    cold (one call after evicting L2)."""
    one = graph(fn, 1)
    _replay_us(one)
    single = statistics.median(_replay_us(one) for _ in range(7))
    calls = max(3, min(MAX_CALLS, math.ceil(WARM_US / max(single, 1e-3))))
    many = graph(fn, calls)
    _replay_us(many)
    warm = statistics.median(_replay_us(many) / calls for _ in range(WARM_REPLAYS))
    cold = statistics.median(_replay_us(one, cold=True) for _ in range(COLD_REPLAYS))
    del many, one
    return warm, cold


def rounds(old, new):
    warm, cold = [], []
    for r in range(ROUNDS):
        if r % 2:
            b, a = per_call(new), per_call(old)
        else:
            a, b = per_call(old), per_call(new)
        warm.append(b[0] / a[0])
        cold.append(b[1] / a[1])
    return statistics.median(warm), statistics.median(cold)


results = {}


def record(name, bits, warm, cold):
    r = results.setdefault(name, {"bits": True, "worst": (0.0, 0.0), "error": None})
    r["bits"] &= bits
    r["worst"] = (max(r["worst"][0], warm), max(r["worst"][1], cold))


def line(label, size, warm, cold, bits=True, word="BITS"):
    print(f"  {label:9s} {size:>16s}  median ratio {warm:.3f} / {cold:.3f}{'' if bits else f'  {word} DIFFER'}",
          flush=True)


old_module = load("/b/r6k_indexer.py", "r6k_indexer")
modules = {}
for variant, path in (("clamp", "/b/indexer_clamp_quant.py"), ("clamp16", "/b/indexer_clamp16_quant.py")):
    try:
        modules[variant] = load(path, f"indexer_{variant}")
    except Exception:
        results[f"indexer {variant}"] = {"bits": False, "worst": (9, 9), "error": traceback.format_exc(limit=2)}
        print(f"indexer {variant}: load ERROR\n{results[f'indexer {variant}']['error']}", flush=True)

HEADS = old_module.INDEX_HEADS
RELU_LINE = re.compile(r"clamp|max2|min2|_nan")
gen = torch.Generator(device=DEVICE).manual_seed(23)

print(f"indexer score (median ratio of {ROUNDS} alternating rounds, warm / cold)", flush=True)
for rows, width in ((1, 65536), (16, 65536), (96, 65536), (1, 1 << 20), (16, 1 << 20), (512, 16384), (2048, 8192)):
    size = f"{width >> 10}K x {rows}"
    pages = width // PAGE
    nbytes = old_module.page_bytes(PAGE)
    q = torch.randint(0, 256, (rows, HEADS, 64), generator=gen, device=DEVICE).to(torch.uint8)
    q_scales = torch.randint(118, 130, (rows, HEADS, 4), generator=gen, device=DEVICE).to(torch.uint8)
    weights = (torch.randn((rows, HEADS), generator=gen, device=DEVICE) / 64).bfloat16()
    pool = torch.randint(0, 256, (pages, nbytes), generator=gen, device=DEVICE).to(torch.uint8)
    pool[:, PAGE * 64 : PAGE * 68] = torch.randint(118, 130, (pages, PAGE * 4), generator=gen, device=DEVICE).to(torch.uint8)
    table = torch.stack([torch.randperm(pages, generator=gen, device=DEVICE) for _ in range(rows)]).to(torch.int32)
    lengths = torch.full((rows,), width, dtype=torch.int32, device=DEVICE)
    active = torch.tensor([width], dtype=torch.int32, device=DEVICE)
    args = (q.view(torch.int8), q_scales.view(torch.uint32).reshape(rows, -1), weights, pool.view(torch.int8),
            pool.view(torch.uint32), table, lengths, active)
    s_old = torch.empty((rows, width), dtype=torch.bfloat16, device=DEVICE)
    old = old_module.indexer_score_v41(width, PAGE)
    old(*args, s_old)
    warm, cold = rounds(lambda: old(*args, s_old), lambda: old(*args, s_old))
    record("indexer r6k A/A", True, warm, cold)
    line("r6k A/A", size, warm, cold)
    for variant, module in modules.items():
        name = f"indexer {variant}"
        if results.get(name, {}).get("error"):
            continue
        try:
            kernel = module.indexer_score_v41(width, PAGE)
            if variant == "clamp16" and (rows, width) == DUMP_SHAPE:
                print(f"==== clamp16 CUDA ReLU lines at {size}", flush=True)
                for text in kernel.get_kernel_source().splitlines():
                    if RELU_LINE.search(text):
                        print(f"  {text.strip()}", flush=True)
            s_new = torch.empty_like(s_old)
            kernel(*args, s_new)
            torch.cuda.synchronize()
            bits = torch.equal(s_old.view(torch.int16), s_new.view(torch.int16))
            warm, cold = rounds(lambda: old(*args, s_old), lambda k=kernel, s=s_new: k(*args, s))
            record(name, bits, warm, cold)
            line(variant, size, warm, cold, bits)
        except Exception:
            results[name] = {"bits": False, "worst": (9, 9), "error": traceback.format_exc(limit=3)}
            print(f"  {variant}: ERROR\n{results[name]['error']}", flush=True)
    del q, q_scales, weights, pool, table, s_old
    torch.cuda.empty_cache()


def finite_inputs(n, seed):
    x = _mxfp4_inputs(n, seed).float()
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x[7] = 3.3895313892515355e38  # BF16 max
    x[8, ::2] = -3.3895313892515355e38
    x[9] = 2.0**-133 * torch.randint(1, 128, (128,), generator=g, device=DEVICE)  # subnormals
    x[10] = 6.0 * 2.0**-126  # the amax floor
    x[11] = torch.randn(128, generator=g, device=DEVICE) * 2.0**-120
    x[12] = torch.randn(128, generator=g, device=DEVICE) * 2.0**100
    return x.bfloat16()


quant = modules.get("clamp16")  # the NaN-keeping quantizer (the same in both variants)
print(f"contiguous quantizer, queries (median ratio of {ROUNDS} alternating rounds, warm / cold)", flush=True)
for n in (64, 1024, 6144, 131072, 524288):
    try:
        x = finite_inputs(n, 31)
        q0 = torch.empty((n, 64), dtype=torch.uint8, device=DEVICE)
        s0 = torch.empty((n, 4), dtype=torch.uint8, device=DEVICE)
        q1, s1 = torch.empty_like(q0), torch.empty_like(s0)
        old_module.quantize_mxfp4(x, q0, s0)
        quant.quantize_mxfp4(x, q1, s1)
        torch.cuda.synchronize()
        bits = torch.equal(q0, q1) and torch.equal(s0, s1)
        warm, cold = rounds(lambda: old_module.quantize_mxfp4(x, q0, s0), lambda: old_module.quantize_mxfp4(x, q0, s0))
        record("quantize r6k A/A", True, warm, cold)
        line("r6k A/A", f"{n} vectors", warm, cold)
        warm, cold = rounds(lambda: old_module.quantize_mxfp4(x, q0, s0), lambda: quant.quantize_mxfp4(x, q1, s1))
        record("quantize", bits, warm, cold)
        line("NaN-keep", f"{n} vectors", warm, cold, bits, "BYTES")
    except Exception:
        results["quantize"] = {"bits": False, "worst": (9, 9), "error": traceback.format_exc(limit=3)}
        print(f"  {n}: ERROR\n{results['quantize']['error']}", flush=True)

print(f"paged quantizer, index-cache keys (median ratio of {ROUNDS} alternating rounds, warm / cold)", flush=True)
for n in (1, 16, 96, 2048, 8192):
    try:
        pages = (n + PAGE - 1) // PAGE + 1
        x = finite_inputs(max(n, 16), 37)[:n]
        g = torch.Generator(device=DEVICE).manual_seed(n)
        slots = torch.randperm(pages * PAGE, generator=g, device=DEVICE)[:n].to(torch.int64)
        pool0 = _page_buffer(pages, old_module.page_bytes(PAGE))
        pool1 = _page_buffer(pages, old_module.page_bytes(PAGE))
        old_module.write_index_keys(x, pool0, slots, PAGE)
        quant.write_index_keys(x, pool1, slots, PAGE)
        torch.cuda.synchronize()
        bits = torch.equal(pool0._base, pool1._base)
        warm, cold = rounds(lambda: old_module.write_index_keys(x, pool0, slots, PAGE),
                            lambda: old_module.write_index_keys(x, pool0, slots, PAGE))
        record("write_index_keys r6k A/A", True, warm, cold)
        line("r6k A/A", f"{n} keys", warm, cold)
        warm, cold = rounds(lambda: old_module.write_index_keys(x, pool0, slots, PAGE),
                            lambda: quant.write_index_keys(x, pool1, slots, PAGE))
        record("write_index_keys", bits, warm, cold)
        line("NaN-keep", f"{n} keys", warm, cold, bits, "BYTES")
    except Exception:
        results["write_index_keys"] = {"bits": False, "worst": (9, 9), "error": traceback.format_exc(limit=3)}
        print(f"  {n}: ERROR\n{results['write_index_keys']['error']}", flush=True)

print("summary (bits and bytes on finite inputs; worst median ratio, warm / cold; A/A rows are the noise floor)",
      flush=True)
for name, r in results.items():
    ok = r["bits"] and not r["error"] and max(r["worst"]) <= BUDGET
    r["ok"] = ok
    print(f"  {name:25s} {'ERROR' if r['error'] else ('equal' if r['bits'] else 'DIFFER')}; "
          f"worst {r['worst'][0]:.3f} / {r['worst'][1]:.3f}{'  OK' if ok else ''}", flush=True)
judged = ("indexer clamp16", "quantize", "write_index_keys")
passed = all(results.get(name, {}).get("ok") for name in judged)
print(f"verdict: clamp16 ReLU and NaN-keeping quantizer {'within' if passed else 'NOT within'} "
      f"{BUDGET - 1:.0%} of r6k with its bits", flush=True)
sys.exit(0 if passed else 1)
