"""Engram rows v2 against B12X, on the same data, before any image is built.

v2 keeps r6m's native reader (no GIL, no CUDA calls, fail-stop) and fixes what the
2026-10-15 analysis found:
- one wait per table and step (a one-thread kernel), not a host-memory poll in every
  gather block (that poll cost about 12 us a token and table: r6m's prefill loss);
- the gather loads only this rank's rows; the reader leaves other slots untouched;
- one worker reads the tables in staging order (layer order), as B12X's reader did.

Measured here:
- the gather: v2's wait and gather together, and its gather alone, against B12X's
  Triton lookup on the same published rows, at 16 to 8192 tokens, with every row on
  this rank and with a quarter of them (TP4);
- a step's two tables: when each table's rows are ready, v2's ordered worker against
  B12X's reader reading the two tables one after the other on one thread, on the same
  random rows of a 1 GiB table, alternating which runs first, two ways:
  - GPU idle at staging (the row IDs are on the host at once): from staging;
  - GPU busy at staging, as in serving, where the runner stages a step while the GPU
    still runs the previous one: a 3 ms kernel runs first, and the time counts from
    when it ends (the IDs exist).
Verdict: the gather (wait included) no slower than B12X's lookup by more than 3 us at
any size, and table 0's rows no later than B12X's by more than 5% with the GPU busy.
"""
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "/b")
from kbench import DEVICE, per_call  # noqa: E402

from b12x.sequence._shared.disk_table import DiskRowCache  # noqa: E402
from b12x.sequence.engram._kernels import _lookup  # noqa: E402
from vllm.models.deepseek_v4_1.engram_rows import EngramDiskRows  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import engram_rows as gather_module  # noqa: E402

DIR = os.environ.get("ENGRAM_ROWS_TEST_DIR", "/c/engram-prefill")
TABLE_ROWS = 1 << 22  # 1 GiB of E4M3 rows, 32 MiB of E8M0 scales
HEADS = 24
MAX_TOKENS = 8192
GATHER_TOKENS = (16, 96, 512, 2048, 8192)
STEP_TOKENS = (96, 2048, 8192)
STEPS = 8
BUSY_MS = 3.0
GATHER_SLACK_US, READ_BUDGET = 3.0, 1.05


def write_table():
    os.makedirs(DIR, exist_ok=True)
    paths = (os.path.join(DIR, "weight.bin"), os.path.join(DIR, "scale.bin"))
    gen = torch.Generator().manual_seed(0)
    for path, size in zip(paths, (TABLE_ROWS * 256, TABLE_ROWS * 8)):
        if os.path.exists(path) and os.path.getsize(path) == size:
            continue
        with open(path, "wb") as f:
            left = size
            while left:
                n = min(left, 64 << 20)
                f.write(torch.randint(110, 140, (n,), generator=gen, dtype=torch.uint8).numpy().tobytes())
                left -= n
        print(f"wrote {path} ({size >> 20} MiB)", flush=True)
    return paths


def v2_rows(paths, start, end):
    rows = EngramDiskRows(device=DEVICE, table_rows=TABLE_ROWS, shard_start=start, shard_end=end,
                          max_lookups=MAX_TOKENS * HEADS)
    rows.add_source(paths[0], 0, scale=False)
    rows.add_source(paths[1], 0, scale=True)
    return rows


def b12x_rows(paths, start, end):
    cache = DiskRowCache(device=DEVICE, max_lookups=MAX_TOKENS * HEADS, table_rows=TABLE_ROWS,
                         shard_start=start, shard_end=end, shard_rows=TABLE_ROWS, weight_row_bytes=256,
                         scale_row_bytes=8, queue_depth=128)
    cache.add_shard(0, paths[0], 0, scale=False)
    cache.add_shard(0, paths[1], 0, scale=True)
    cache.freeze()
    return cache


def published(rows):
    return int(rows._control[rows.native.CTRL_ROWS_SEQ]) == rows._sequence


def wait_published(rows, timeout_s=60.0):
    deadline = time.monotonic() + timeout_s
    while not published(rows):
        if rows.failure():
            raise RuntimeError(rows.failure())
        if time.monotonic() > deadline:
            raise RuntimeError("the reader did not publish the step")


paths = write_table()
quarter = (TABLE_ROWS // 4, TABLE_ROWS // 2)
gen = torch.Generator(device=DEVICE).manual_seed(1)
stream = torch.cuda.current_stream()
failures = []

print("gather: us per call (warm / cold) on the same published rows", flush=True)
for label, (start, end) in (("all rows local", (0, TABLE_ROWS)), ("a quarter local (TP4)", quarter)):
    rows = v2_rows(paths, start, end)
    ids_full = torch.randint(0, TABLE_ROWS, (MAX_TOKENS, HEADS), generator=gen, device=DEVICE)
    rows.stage(stream, ids_full, MAX_TOKENS * HEADS)
    wait_published(rows)
    torch.cuda.synchronize()
    gather_only = gather_module.engram_rows_gather_kernel(MAX_TOKENS * HEADS, start, end)
    print(f"  {label}", flush=True)
    for tokens in GATHER_TOKENS:
        ids = ids_full[:tokens].contiguous()
        num = torch.tensor([tokens], dtype=torch.int32, device=DEVICE)
        out = {k: torch.empty((tokens, HEADS * 256), dtype=torch.bfloat16, device=DEVICE)
               for k in ("v2", "gather", "b12x")}
        calls = {
            "v2": lambda: gather_module.engram_rows_gather(ids, num, rows.rows_e4m3, rows.scales.device_view,
                                                           rows.control.device_view, out["v2"], start, end),
            "gather": lambda: gather_only(ids, num, rows.rows_e4m3, rows.scales.device_view, out["gather"]),
            "b12x": lambda: _lookup[(tokens, HEADS)](rows.rows_e4m3, rows.scales.device_view, ids, num,
                                                     out["b12x"], tokens, T=tokens, ROWS=TABLE_ROWS,
                                                     START=start, END=end, COMPACT=True,
                                                     RESIDENT_SCALES=False),
        }
        for fn in calls.values():
            fn()
        torch.cuda.synchronize()
        same = all(torch.equal(out[k].view(torch.int16), out["b12x"].view(torch.int16)) for k in ("v2", "gather"))
        timed = {k: per_call(fn, tokens) for k, fn in calls.items()}
        over = timed["v2"][0] - timed["b12x"][0]
        ok = same and over <= GATHER_SLACK_US
        if not ok:
            failures.append(f"gather {label} {tokens}: {'bits differ' if not same else f'+{over:.1f} us'}")
        line = "  ".join(f"{k} {w:8.1f}/{c:8.1f}" for k, (w, c) in timed.items())
        print(f"    {tokens:5d} tokens  {line}  v2 - b12x {over:+7.1f} us  "
              f"{'bits equal' if same else 'BITS DIFFER'}{'' if ok else '  FAIL'}", flush=True)
    rows.close()

def busy_cycles(ms):
    """Clock cycles that make torch.cuda._sleep run for about ``ms``."""
    probe = 2_000_000
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    begin.record()
    torch.cuda._sleep(probe)
    end.record()
    torch.cuda.synchronize()
    return int(probe * ms / begin.elapsed_time(end))


cycles = busy_cycles(BUSY_MS)
v2 = [v2_rows(paths, *quarter), v2_rows(paths, *quarter)]
old = [b12x_rows(paths, *quarter), b12x_rows(paths, *quarter)]
for busy in (False, True):
    print(f"a step's two tables, GPU {'busy' if busy else 'idle'} at staging: ms until each table's rows "
          f"are ready{', from the end of the busy kernel' if busy else ', from staging'} "
          "(median of the steps; a quarter of the rows local)", flush=True)
    for tokens in STEP_TOKENS:
        count = tokens * HEADS
        times = {"v2": [], "b12x": []}
        for step in range(STEPS):
            ids = [torch.randint(0, TABLE_ROWS, (count,), generator=gen, device=DEVICE) for _ in range(2)]
            for name in (("v2", "b12x") if step % 2 == 0 else ("b12x", "v2")):
                torch.cuda.synchronize()
                start = time.perf_counter()
                if busy:
                    torch.cuda._sleep(cycles)
                    start += BUSY_MS / 1e3
                if name == "v2":
                    for table, rows in enumerate(v2):
                        rows.stage(stream, ids[table], count)
                    ready = [None, None]
                    while None in ready:
                        for table, rows in enumerate(v2):
                            if ready[table] is None and published(rows):
                                ready[table] = time.perf_counter() - start
                            if rows.failure():
                                raise RuntimeError(rows.failure())
                else:
                    ready = []
                    for table, cache in enumerate(old):
                        with cache.transaction():
                            cache.read_rows(ids[table], count)
                        ready.append(time.perf_counter() - start)
                times[name].append(ready)
        t0 = {k: statistics.median(r[0] for r in v) * 1e3 for k, v in times.items()}
        t1 = {k: statistics.median(r[1] for r in v) * 1e3 for k, v in times.items()}
        ok = not busy or t0["v2"] <= t0["b12x"] * READ_BUDGET
        if not ok:
            failures.append(f"step {tokens}: table 0 at {t0['v2']:.2f} ms against {t0['b12x']:.2f}")
        print(f"  {tokens:5d} tokens  table 0: v2 {t0['v2']:7.2f}  b12x {t0['b12x']:7.2f}   "
              f"table 1: v2 {t1['v2']:7.2f}  b12x {t1['b12x']:7.2f}{'' if ok else '  FAIL'}", flush=True)
for rows in v2:
    print(f"  v2 reader stats: {rows.stats()}", flush=True)
    rows.close()
for cache in old:
    cache.close()

print(f"verdict: {'PASS' if not failures else 'FAIL: ' + '; '.join(failures)}", flush=True)
sys.exit(1 if failures else 0)
