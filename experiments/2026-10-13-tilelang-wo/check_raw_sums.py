#!/usr/bin/env python3
"""Recompute the WO's TP sums of a raw row trace, row by row.

usage: check_raw_sums.py TRACE_DIR [LAYER ...]

TRACE_DIR holds ``rank0`` .. ``rank3`` of a VLLM_DS41_ROW_TRACE run whose
``VLLM_DS41_ROW_TRACE_RAW`` layers kept their WO stages raw (``wo:raw``, the rank's
partial; ``parts:raw``, the parts a sequence-parallel row's owner received, row-major
``(rows, ranks, hidden)``; ``ar:raw``, the sum). For the sequence-parallel forward with
several prompts and every prompt's own forward, per layer (default 0 and 1) and per row
of each prompt whose summed bits differ between the two, prints:

- whether each received part equals the partial its sender computed in that forward;
- whether each sender's partial equals its partial in the prompt's own forward;
- the FP32 rank-order sum of the received parts, of the senders' partials and of the
  prompt-alone partials (BF16, round to nearest even), against both recorded sums.

Standard library only.
"""
import array
import glob
import os
import pickle
import struct
import sys


def runs(positions):
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


def bf16_rows(meta):
    """The raw record as a list of rows of 16-bit patterns (unsigned)."""
    values = array.array("H")
    values.frombytes(meta["raw"])
    shape = meta["shape"]
    width = 1
    for d in shape[1:]:
        width *= d
    return [values[i * width : (i + 1) * width] for i in range(shape[0])], shape


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def from_bf16(bits):
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def to_bf16(x):
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    if (u & 0x7F800000) == 0x7F800000 and u & 0x7FFFFF:
        return (u >> 16) | 0x40
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF


def rank_order_sum(parts):
    """bf16(((p0 + p1) + p2) + p3) in FP32, element-wise, for 16-bit pattern rows."""
    out = []
    for values in zip(*parts):
        acc = from_bf16(values[0])
        for v in values[1:]:
            acc = f32(acc + from_bf16(v))
        out.append(to_bf16(acc))
    return out


def load_rank(directory):
    steps = {}
    for path in sorted(glob.glob(os.path.join(directory, "step-*.pkl"))):
        with open(path, "rb") as f:
            steps[os.path.basename(path)] = pickle.load(f)
    return steps


def stage(records, prefix, name):
    return next((r for r in records if r[0] == prefix and r[1] == name), None)


def main():
    trace = sys.argv[1]
    layers = [int(x) for x in sys.argv[2:]] or [0, 1]
    ranks = [load_rank(d) for d in sorted(glob.glob(os.path.join(trace, "rank*")))]
    world = len(ranks)
    names = sorted(ranks[0])
    first = {n: next(r for r in ranks[0][n] if r[1] == "o") for n in names}
    shared = [n for n in names if len(runs(first[n][2])) > 1 and first[n][4].get("sp")]
    alone = {}
    for n in names:
        found = runs(first[n][2])
        if len(found) == 1 and found[0] == (0, len(first[n][2])):
            alone.setdefault(tuple(first[n][3]), n)
    for sp_name in shared:
        positions = first[sp_name][2]
        tokens = len(positions)
        local = -(-tokens // world)
        print(f"{sp_name}: {tokens} rows, sequence-parallel, {local} rows per rank")
        for layer in layers:
            prefix = next(r[0] for r in ranks[0][sp_name] if r[0].endswith(f"layers.{layer}.attn"))
            for start, n in runs(positions):
                ref_name = alone.get(tuple(first[sp_name][3][start : start + n]))
                if ref_name is None:
                    continue
                wo_sp = [bf16_rows(stage(ranks[r][sp_name], prefix, "wo:raw")[4])[0] for r in range(world)]
                wo_alone = [bf16_rows(stage(ranks[r][ref_name], prefix, "wo:raw")[4])[0] for r in range(world)]
                for j in range(n):
                    row = start + j
                    owner = row // local
                    sp_ar_rec = stage(ranks[owner][sp_name], prefix, "ar:raw")
                    parts_rec = stage(ranks[owner][sp_name], prefix, "parts:raw")
                    alone_ar = bf16_rows(stage(ranks[owner][ref_name], prefix, "ar:raw")[4])[0][j]
                    row0 = sp_ar_rec[4]["row0"]
                    sp_ar = bf16_rows(sp_ar_rec[4])[0][row - row0]
                    if list(sp_ar) == list(alone_ar):
                        continue
                    parts_rows, shape = bf16_rows(parts_rec[4])
                    hidden = shape[2]
                    received = parts_rows[row - row0]
                    received = [received[r * hidden : (r + 1) * hidden] for r in range(world)]
                    sent = [wo_sp[r][row] for r in range(world)]
                    before = [wo_alone[r][j] for r in range(world)]
                    diffs = sum(1 for a, b in zip(sp_ar, alone_ar) if a != b)
                    print(f"  layer {layer} prompt at {start} row {j} (owner rank {owner}): "
                          f"{diffs} of {len(sp_ar)} summed elements differ")
                    for r in range(world):
                        got = sum(1 for a, b in zip(received[r], sent[r]) if a != b)
                        moved = sum(1 for a, b in zip(sent[r], before[r]) if a != b)
                        print(f"    rank {r}: received != sent in {got}; "
                              f"partial != prompt alone in {moved}")
                    for label, parts in (("received", received), ("sent", sent), ("alone", before)):
                        total = rank_order_sum(parts)
                        print(f"    rank-order sum of the {label} parts: "
                              f"!= SP sum in {sum(1 for a, b in zip(total, sp_ar) if a != b)}, "
                              f"!= alone sum in {sum(1 for a, b in zip(total, alone_ar) if a != b)}")
                    shown = 0
                    for k, (a, b) in enumerate(zip(sp_ar, alone_ar)):
                        if a != b and shown < 4:
                            shown += 1
                            print(f"    element {k}: SP {from_bf16(a)!r} alone {from_bf16(b)!r}; parts "
                                  + " ".join(repr(from_bf16(p[k])) for p in received))


if __name__ == "__main__":
    main()
