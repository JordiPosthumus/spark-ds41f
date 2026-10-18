#!/usr/bin/env python3
"""Where a prompt's prefill rows first differ between its own forward and a shared one.

usage: compare_row_trace.py TRACE_DIR

Reads a VLLM_DS41_ROW_TRACE run (vLLM branch wo-row-trace, DEBUG ONLY): one directory
per TP rank (``rank<r>``, or the step files directly for a rank-0-only trace), and per
forward every attention layer's row hashes with the rows' positions, at these stages:
"in" (the layer's input; gathered in sequence-parallel steps), "qr"/"kv" (the gathered
projections, sequence-parallel layers without a compressor), "o" (attention output,
before WO), "wo" (this rank's WO partial, before the TP sum) and "ar" (after it; in a
sequence-parallel step only this rank's rows, from row ``row0``).

A prompt's prefill is a run of rows at positions 0, 1, ... in a forward; prompts are
told apart by the first target layer's "qr" over the run (the same on every rank), or
its attention output in traces without "qr". For each prompt
prefilled both alone and beside other rows, prints the first (layer, stage) whose
hashes differ on the rows both forwards recorded, in the alone forward's order.

Standard library only.
"""
import glob
import os
import pickle
import sys


def runs(positions):
    """(start, length) of every run of rows at positions 0, 1, ..."""
    found, i = [], 0
    while i < len(positions):
        if positions[i] == 0:
            n = 1
            while i + n < len(positions) and positions[i + n] == n:
                n += 1
            found.append((i, n))
            i += n
        else:
            i += 1
    return found


def first_output(records):
    """The first layer's record that keys prompts: "qr" if traced, else "o"."""
    return next((r for r in records if r[1] == "qr"), None) or next(
        r for r in records if r[1] == "o"
    )


def compare_rank(directory):
    steps = []
    for path in sorted(glob.glob(os.path.join(directory, "step-*.pkl"))):
        with open(path, "rb") as f:
            steps.append((os.path.basename(path), pickle.load(f)))
    alone = {}
    for name, records in steps:
        o = first_output(records)
        found = runs(o[2])
        if len(found) == 1 and found[0] == (0, len(o[2])):
            alone.setdefault(tuple(o[3]), (name, records))
    lines = []
    for name, records in steps:
        o = first_output(records)
        found = runs(o[2])
        if len(found) == 1 and found[0] == (0, len(o[2])):
            continue
        index = {}
        for r in records:
            index.setdefault((r[0], r[1]), []).append(r)
        for start, n in found:
            ref = alone.get(tuple(o[3][start : start + n]))
            if ref is None:
                continue
            result = first_difference(ref[1], index, start, n)
            sp = next((r[4] for r in records if r[1] == "o"), {}).get("sp")
            lines.append(f"  {name} ({len(o[2])} rows, sp={sp}) run at {start} ({n} rows) "
                         f"against {ref[0]}: {result}")
    return lines


def first_difference(ref, index, start, n):
    seen = {}
    compared = 0
    for prefix, stage, _, hashes, _ in ref:
        k = seen.get((prefix, stage), 0)
        seen[(prefix, stage)] = k + 1
        candidates = index.get((prefix, stage), [])
        if k >= len(candidates) or len(hashes) != n:
            continue
        _, _, positions, other, meta = candidates[k]
        row0 = meta.get("row0", 0)
        lo, hi = max(start, row0), min(start + n, row0 + len(other))
        if lo >= hi:
            continue
        if any(positions[row - row0] != row - start for row in range(lo, hi)):
            continue  # this record's rows are not the run's (compacted layout)
        compared += 1
        diff = [row - start for row in range(lo, hi) if hashes[row - start] != other[row - row0]]
        if diff:
            return (f"{prefix} {stage}: {len(diff)} of {hi - lo} rows differ "
                    f"(first at position {diff[0]}; {compared} records equal before)")
    return f"identical ({compared} records compared)"


def main():
    trace = sys.argv[1]
    ranks = sorted(glob.glob(os.path.join(trace, "rank*"))) or [trace]
    for directory in ranks:
        print(os.path.basename(directory.rstrip("/")))
        for line in compare_rank(directory):
            print(line)


if __name__ == "__main__":
    main()
