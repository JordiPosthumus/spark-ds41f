"""Match two kernel summaries (summarize_kernels.py) by launch grid, block and call count, so
r6h's unnamed TileLang kernels (main_kernel) pair with r6n's named ones.

usage: match_grids.py BASE.json CANDIDATE.json [AGAIN.json]
Prints matched pairs by delta, then kernels only one side has, with totals; AGAIN (the base
build on another boot) gives each matched kernel's boot-to-boot noise.
"""
import json
import sys
from collections import defaultdict


def load(path):
    d = json.load(open(path))
    rows = defaultdict(float)
    names = {}
    for name, grid, block, calls, ms in d["kernels"]:
        key = (tuple(grid), tuple(block), calls)
        rows[key] += ms
        names.setdefault(key, set()).add(name)
    return d, rows, names


base_d, base, base_names = load(sys.argv[1])
cand_d, cand, cand_names = load(sys.argv[2])
again = load(sys.argv[3])[1] if len(sys.argv) > 3 else None
print(f"span ms {base_d['span_ms']:.1f} -> {cand_d['span_ms']:.1f} ({cand_d['span_ms'] - base_d['span_ms']:+.1f}); "
      f"kernel ms {base_d['kernel_ms']:.1f} -> {cand_d['kernel_ms']:.1f} ({cand_d['kernel_ms'] - base_d['kernel_ms']:+.1f})")


def label(names):
    return "/".join(sorted(n[:60] for n in names))


matched = sorted(set(base) & set(cand), key=lambda k: -abs(cand[k] - base[k]))
total = sum(cand[k] - base[k] for k in matched)
print(f"\nmatched by grid/block/calls: {len(matched)} kernels, delta {total:+.1f} ms")
for k in matched[:25]:
    noise = f"  again {again.get(k, 0) - base[k]:+7.2f}" if again is not None else ""
    print(f"  {cand[k] - base[k]:+8.2f}  {base[k]:8.2f} -> {cand[k]:8.2f}{noise}  {label(base_names[k])} | "
          f"{label(cand_names[k])} grid {k[0]} calls {k[2]}")
only_base = sorted(set(base) - set(cand), key=lambda k: -base[k])
only_cand = sorted(set(cand) - set(base), key=lambda k: -cand[k])
print(f"\nonly in base: {sum(base[k] for k in only_base):.1f} ms")
for k in only_base[:15]:
    print(f"  {base[k]:8.2f}  {label(base_names[k])} grid {k[0]} block {k[1]} calls {k[2]}")
print(f"\nonly in candidate: {sum(cand[k] for k in only_cand):.1f} ms")
for k in only_cand[:15]:
    print(f"  {cand[k]:8.2f}  {label(cand_names[k])} grid {k[0]} block {k[1]} calls {k[2]}")
