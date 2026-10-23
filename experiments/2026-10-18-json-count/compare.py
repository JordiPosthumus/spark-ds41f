#!/usr/bin/env python3
"""Compare the production replay with the CPU reference runs.

    python3 compare.py runs/production-r6v.json runs/ref-*.json

Prints the number step (the reply's sixth token) for every run; each reference run against
production (the log-probability of every prompt token, and the reply's top tokens, as both score
them); then every reference run against the first one given, the same way: two valid
implementations of the same arithmetic, the noise floor the production comparison is read against.
"""
import json
import statistics as st
import sys

NUMBERS = {"9": 27, "10": 553, "11": 779, "12": 736, "13": 907}


def load(path):
    with open(path) as f:
        return json.load(f)


prod = load(sys.argv[1])
refs = [(p, load(p)) for p in sys.argv[2:]]


def prod_number_step():
    step = prod["steps"][5]
    top = dict((t, lp) for t, lp in step["top"])
    return {k: top.get(v) for k, v in NUMBERS.items()}


def ref_number_step(r):
    step = r["steps"][5]
    top = {t: lp for t, _, lp in step["top"]}
    return {k: top.get(v) for k, v in NUMBERS.items()}


def label(path, r):
    parts = ["WO-A input quant " + ("on" if r["wo_a_input_quant"] else "off")]
    parts.append(r.get("accumulation", "torch.float32").replace("torch.", "") + " accumulation")
    return f"{path.rsplit('/', 1)[-1]} ({', '.join(parts)})"


print("number step: log-probability of each answer after '{\"vowels\": '")
print(f"  {'run':<64}" + "".join(f"{k:>9}" for k in NUMBERS) + "   greedy reply")
row = prod_number_step()
print(f"  {'production r6v':<64}" + "".join(f"{row[k]:>9.3f}" if row[k] is not None else f"{'-':>9}" for k in NUMBERS)
      + f"   {prod['content']}")
for path, r in refs:
    row = ref_number_step(r)
    reply = "".join(t for t in [s["top"][0][1] for s in r["steps"][:-1]])
    print(f"  {label(path, r):<64}" + "".join(f"{row[k]:>9.3f}" if row[k] is not None else f"{'-':>9}" for k in NUMBERS)
          + f"   {reply}")


def as_prod(r):
    """A reference run in the production replay's shape, so either can be the baseline."""
    return {"prompt_positions": r["prompt_positions"],
            "steps": [{"token": s["argmax"], "top": [[t, lp] for t, _, lp in s["top"]]} for s in r["steps"]]}


def against(base_name, base, path, r):
    print(f"\n{label(path, r)} against {base_name}")
    rp = {x["position"]: x for x in r["prompt_positions"]}
    diffs = []
    for x in base["prompt_positions"]:
        y = rp[x["position"]]
        assert x["token"] == y["token"]
        diffs.append(y["logprob"] - x["logprob"])
    top1 = sum(1 for x in base["prompt_positions"] if x["top"][0][0] == rp[x["position"]]["top"][0][0])
    print(f"  prompt tokens ({len(diffs)}): |delta logprob| mean {st.mean(abs(d) for d in diffs):.3f}, "
          f"median {st.median(abs(d) for d in diffs):.3f}, max {max(abs(d) for d in diffs):.3f}; "
          f"top-1 agrees at {top1}/{len(diffs)}")
    for k, (ps, rs) in enumerate(zip(base["steps"], r["steps"])):
        pt = dict((t, lp) for t, lp in ps["top"])
        rt = {t: lp for t, _, lp in rs["top"]}
        common = [t for t, _ in ps["top"][:5] if t in rt]
        d = [rt[t] - pt[t] for t in common]
        print(f"  reply step {k}: top-1 {ps['top'][0][0]} ({ps['top'][0][1]:.3f}) / {rs['top'][0][0]} "
              f"({rs['top'][0][2]:.3f}); top-5 shared {len(common)}, |delta| max {max(abs(v) for v in d):.3f}")


scored = [(p, r) for p, r in refs if "prompt_positions" in r]
for path, r in scored:
    against("production", prod, path, r)
if len(scored) > 1:
    base_path, base = scored[0]
    for path, r in scored[1:]:
        against(base_path.rsplit("/", 1)[-1], as_prod(base), path, r)
