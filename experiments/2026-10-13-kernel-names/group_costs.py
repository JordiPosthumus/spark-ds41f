"""Per-step kernel time grouped by launch signature, two arms (rank-0 summarize_kernels outputs)."""
import json, sys
from collections import defaultdict
sys.path.insert(0, sys.argv[3]) if len(sys.argv) > 3 else None
VOCAB = {1010, 2020, 1348, 2696}

def head(name, grid):
    return name in ("main_kernel", "packed_vocab_projection_kernel") and grid and (grid[0] in VOCAB or grid[1] in VOCAB)

def key(name, grid, block):
    if name != "main_kernel":
        short = name.replace("kernel_cutlass_kernel_", "")[:60]
        return short
    g0, g1 = grid[0], grid[1]
    if head(name, grid):
        return f"tl vocab head (tiles {g1 if g0 in VOCAB else g0})"
    if g1 > 32:  # dynamic: routing-dependent second dimension
        return f"tl grid [{g0}, *] block {block[0]}"
    return f"tl grid [{g0}, {g1}] block {block[0]}"

def load(path):
    d = json.load(open(path)); by = defaultdict(float); steps = 0.0
    for name, grid, block, launches, ms in d["kernels"]:
        by[key(name, grid, block)] += ms
        steps += launches * 0.5 if head(name, grid) else 0
    return d, by, steps

(da, a, sa), (db, b, sb) = load(sys.argv[1]), load(sys.argv[2])
print(f"steps {sa:.1f} / {sb:.1f}   kernel ms/step {da['kernel_ms']/sa:.2f} / {db['kernel_ms']/sb:.2f} ({db['kernel_ms']/sb - da['kernel_ms']/sa:+.2f})")
for k in sorted(set(a) | set(b), key=lambda k: -abs(b.get(k, 0)/sb - a.get(k, 0)/sa))[:22]:
    x, y = a.get(k, 0)/sa, b.get(k, 0)/sb
    print(f"  {y - x:+7.3f}  {x:8.3f} -> {y:8.3f}  {k}")
