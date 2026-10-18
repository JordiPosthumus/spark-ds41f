"""Why r6m prefills 10-13% slower than r6k: the Engram gather or the Engram reader.

r6m's release (2026-10-15) has r6k's bits and decode at small row counts 3-4.5% faster,
but every prefill size 10-13% slower and the 72- and 96-row decode steps 1.5-2.3%
slower: about 0.5 us per lookup that the forward does not hide. r6l (r6m without its
four pieces) prefilled level with r6k. Both paths use the same planner, io_uring
settings and mapped write-combined buffers, so this times the two parts that changed:
- the gather: r6m's TileLang gather (thread 0 of every (head, token) block polls the
  control block in host memory before the block reads its row), the same kernel without
  that poll, and B12X's Triton lookup, on the same published rows, at 16 to 8192 tokens;
- the reader: r6m's native worker (stage, then the host waits for the published step)
  and B12X's synchronous row cache, reading the same random rows of a 1 GiB table on
  disk, alternating which reads first, at decode and prefill lookup counts.
"""
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "/b")
from kbench import DEVICE, per_call  # noqa: E402

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
from b12x.sequence._shared.disk_table import DiskRowCache  # noqa: E402
from b12x.sequence.engram._kernels import _lookup  # noqa: E402
from vllm.models.deepseek_v4_1.engram_rows import EngramDiskRows  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import engram_rows as gather_module  # noqa: E402

DIR = os.environ.get("ENGRAM_ROWS_TEST_DIR", "/c/engram-prefill")
TABLE_ROWS = 1 << 22  # 1 GiB of E4M3 rows, 32 MiB of E8M0 scales
HEADS = 24
MAX_TOKENS = 8192
GATHER_TOKENS = (16, 96, 512, 2048, 8192)
READ_TOKENS = (96, 2048, 8192)
READ_STEPS = 8


def write_table():
    os.makedirs(DIR, exist_ok=True)
    paths = (os.path.join(DIR, "weight.bin"), os.path.join(DIR, "scale.bin"))
    sizes = (TABLE_ROWS * 256, TABLE_ROWS * 8)
    gen = torch.Generator().manual_seed(0)
    for path, size in zip(paths, sizes):
        if os.path.exists(path) and os.path.getsize(path) == size:
            continue
        with open(path, "wb") as f:
            left = size
            while left:
                n = min(left, 64 << 20)
                block = torch.randint(110, 140, (n,), generator=gen, dtype=torch.uint8)
                f.write(block.numpy().tobytes())
                left -= n
        print(f"wrote {path} ({size >> 20} MiB)", flush=True)
    return paths


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def nowait_kernel(max_lookups, shard_start, shard_end, threads=64, vec=4):
    """r6m's gather without the per-block wait: its cost against the real kernel is the
    poll's."""
    ROW, SCALES = gather_module.ROW, gather_module.SCALES
    tokens = T.dynamic("tokens")

    @T.prim_func
    def gather(
        ids: T.Tensor((tokens, HEADS), T.int64),
        num_tokens: T.Tensor((1,), T.int32),
        rows: T.Tensor((max_lookups, ROW), T.float8_e4m3fn),
        scales: T.Tensor((max_lookups, SCALES), T.uint8),
        out: T.Tensor((tokens, HEADS * ROW), T.bfloat16),
    ):
        with T.Kernel(HEADS, tokens, threads=threads) as (head, t):
            tx = T.get_thread_binding()
            row = ids[t, head]
            local = (t < num_tokens[0]) & (row >= shard_start) & (row < shard_end)
            lookup = t * HEADS + head
            exponent = T.cast(scales[lookup, tx * vec // 32], T.uint32)
            scale = T.reinterpret(T.shift_left(exponent, 23), T.float32)
            scale = T.if_then_else(exponent == 0, T.float32(2.0**-127), scale)
            scale = T.if_then_else(exponent == 255, T.float32(float("nan")), scale)
            for i in T.vectorized(vec):
                value = T.cast(rows[lookup, tx * vec + i], T.float32) * scale
                out[t, head * ROW + tx * vec + i] = T.if_then_else(
                    local, T.cast(value, T.bfloat16), T.bfloat16(0)
                )

    return gather


def wait_published(rows: EngramDiskRows, timeout_s: float = 30.0) -> None:
    n = rows.native
    deadline = time.monotonic() + timeout_s
    while int(rows._control[n.CTRL_ROWS_SEQ]) != rows._sequence:
        if rows.failure():
            raise RuntimeError(rows.failure())
        if time.monotonic() > deadline:
            raise RuntimeError("the reader did not publish the step")


weight_path, scale_path = write_table()
lookups = MAX_TOKENS * HEADS
new = EngramDiskRows(device=DEVICE, table_rows=TABLE_ROWS, shard_start=0, shard_end=TABLE_ROWS,
                     max_lookups=lookups)
new.add_source(weight_path, 0, scale=False)
new.add_source(scale_path, 0, scale=True)
old = DiskRowCache(device=DEVICE, max_lookups=lookups, table_rows=TABLE_ROWS, shard_start=0,
                   shard_end=TABLE_ROWS, shard_rows=TABLE_ROWS, weight_row_bytes=256,
                   scale_row_bytes=8, queue_depth=128)
old.add_shard(0, weight_path, 0, scale=False)
old.add_shard(0, scale_path, 0, scale=True)
old.freeze()
stream = torch.cuda.current_stream()
gen = torch.Generator(device=DEVICE).manual_seed(1)

print("reader: ms per step, the same random rows for both, alternating which reads first", flush=True)
read_results = {}
for tokens in READ_TOKENS:
    count = tokens * HEADS
    times = {"r6m": [], "b12x": []}
    for step in range(READ_STEPS):
        ids = torch.randint(0, TABLE_ROWS, (count,), generator=gen, device=DEVICE)
        order = ("r6m", "b12x") if step % 2 == 0 else ("b12x", "r6m")
        for name in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            if name == "r6m":
                new.stage(stream, ids, count)
                wait_published(new)
            else:
                with old.transaction():
                    old.read_rows(ids, count)
            times[name].append((time.perf_counter() - start) * 1e3)
    r6m, b12x = statistics.median(times["r6m"]), statistics.median(times["b12x"])
    read_results[tokens] = (r6m, b12x)
    print(f"  {tokens:5d} tokens ({count:6d} lookups)  r6m {r6m:8.2f} ms  b12x {b12x:8.2f} ms  "
          f"ratio {r6m / b12x:.3f}", flush=True)
print(f"  r6m reader stats: {new.stats()}", flush=True)
print(f"  b12x reader stats: {old.stats()}", flush=True)

# Publish one full step so every gather's wait returns at once.
ids_full = torch.randint(0, TABLE_ROWS, (MAX_TOKENS, HEADS), generator=gen, device=DEVICE)
new.stage(stream, ids_full, lookups)
wait_published(new)
torch.cuda.synchronize()
nowait = nowait_kernel(lookups, 0, TABLE_ROWS)

print("gather: us per call (warm / cold) on the same published rows", flush=True)
gather_results = {}
for tokens in GATHER_TOKENS:
    ids = ids_full[:tokens].contiguous()
    num = torch.tensor([tokens], dtype=torch.int32, device=DEVICE)
    out = {k: torch.empty((tokens, HEADS * 256), dtype=torch.bfloat16, device=DEVICE)
           for k in ("r6m", "nowait", "b12x")}
    calls = {
        "r6m": lambda: gather_module.engram_rows_gather(ids, num, new.rows_e4m3, new.scales.device_view,
                                                        new.control.device_view, out["r6m"], 0, TABLE_ROWS),
        "nowait": lambda: nowait(ids, num, new.rows_e4m3, new.scales.device_view, out["nowait"]),
        "b12x": lambda: _lookup[(tokens, HEADS)](new.rows_e4m3, new.scales.device_view, ids, num, out["b12x"],
                                                 tokens, T=tokens, ROWS=TABLE_ROWS, START=0, END=TABLE_ROWS,
                                                 COMPACT=True, RESIDENT_SCALES=False),
    }
    for fn in calls.values():
        fn()
    torch.cuda.synchronize()
    same = all(torch.equal(out[k].view(torch.int16), out["b12x"].view(torch.int16)) for k in ("r6m", "nowait"))
    timed = {k: per_call(fn, tokens) for k, fn in calls.items()}
    gather_results[tokens] = timed
    line = "  ".join(f"{k} {w:9.1f}/{c:9.1f}" for k, (w, c) in timed.items())
    print(f"  {tokens:5d} tokens  {line}  {'bits equal' if same else 'BITS DIFFER'}", flush=True)

print("summary", flush=True)
for tokens, (r6m, b12x) in read_results.items():
    print(f"  read   {tokens:5d} tokens: r6m - b12x = {r6m - b12x:+8.2f} ms", flush=True)
for tokens, timed in gather_results.items():
    print(f"  gather {tokens:5d} tokens: r6m - b12x = {timed['r6m'][0] - timed['b12x'][0]:+9.1f} us warm, "
          f"poll cost (r6m - nowait) = {timed['r6m'][0] - timed['nowait'][0]:+9.1f} us warm", flush=True)
new.close()
old.close()
