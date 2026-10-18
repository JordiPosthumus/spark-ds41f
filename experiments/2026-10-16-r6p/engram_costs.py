"""Engram's GPU time in torch-profiler traces, by part, per rank.

usage: engram_costs.py RUN_DIR LABEL... (RUN_DIR: results/private/determinism/<run>, holding
<label>/<workload>-trace/<node>/*.json.gz)

Parts: the one-thread wait for a table's rows, the gather, the release that frees the
buffers for the next step's early reads, the hash, the gate, and the two Engram
collectives, found by their CPU op and input shape (the rows all-reduce of width 6144;
the wkv product's all-gather or all-to-all of width 8544). Times are kernel time in ms
over the profiled request; "span" is the trace's GPU span.
"""
import collections
import glob
import gzip
import json
import os
import sys

KERNELS = {
    "wait": ("engram_rows_wait",),
    "gather": ("engram_rows_gather",),
    "release": ("engram_rows_release",),
    "hash": ("engram_hash",),
    "gate": ("engram_gate",),
}
WIDTH = {"6144": "rows all-reduce", "8544": "wkv exchange"}


def costs(path: str) -> dict:
    events = json.load(gzip.open(path))["traceEvents"]
    ops = {}
    for e in events:
        if e.get("cat") == "cpu_op" and e.get("name") in ("vllm::all_reduce", "vllm::all_gather",
                                                          "vllm::all_to_all"):
            args = e.get("args", {})
            dims = args.get("Input Dims") or [[]]
            width = str(dims[0][-1]) if dims and dims[0] else ""
            if width in WIDTH:
                ops[args.get("External id")] = WIDTH[width]
    launch = {e["args"]["correlation"]: e["args"].get("External id") for e in events
              if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})}
    totals = collections.defaultdict(float)
    first, last = None, None
    for e in events:
        if e.get("cat") != "kernel":
            continue
        start, dur = e["ts"], e.get("dur", 0)
        first = start if first is None else min(first, start)
        last = start + dur if last is None else max(last, start + dur)
        name = e["name"]
        for part, needles in KERNELS.items():
            if any(n in name for n in needles):
                totals[part] += dur / 1000
                break
        else:
            args = e.get("args", {})
            ext = args.get("External id") or launch.get(args.get("correlation"))
            if ext in ops:
                totals[ops[ext]] += dur / 1000
    totals["span"] = ((last or 0) - (first or 0)) / 1000
    return totals


def main() -> None:
    run, labels = sys.argv[1], sys.argv[2:]
    parts = ["span", "wait", "gather", "release", "hash", "gate", "rows all-reduce", "wkv exchange"]
    print(f"{'label':10s} {'workload':8s} {'node':5s} " + " ".join(f"{p:>15s}" for p in parts))
    for label in labels:
        for workload_dir in sorted(glob.glob(os.path.join(run, label, "*-trace"))):
            workload = os.path.basename(workload_dir)[: -len("-trace")]
            for node_dir in sorted(glob.glob(os.path.join(workload_dir, "*"))):
                traces = sorted(glob.glob(os.path.join(node_dir, "*.json.gz")))
                if not traces:
                    continue
                t = costs(traces[0])
                print(f"{label:10s} {workload:8s} {os.path.basename(node_dir):5s} "
                      + " ".join(f"{t.get(p, 0.0):15.1f}" for p in parts))


if __name__ == "__main__":
    main()
