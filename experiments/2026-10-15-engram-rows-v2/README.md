# 2026-10-15: Engram rows v2

r6m read Engram rows with a native worker (no GIL, no CUDA calls) and gathered them
in the forward. It had r6k's bits and decode at small row counts 3-4.5% faster, but
every prefill size 10-13% slower and the 72- and 96-row decode steps 1.5-2.3% slower
(experiments/2026-10-14-r6m, release 2026-10-15). It is not promoted.

The step back (2026-10-15) compared B12X's Engram path (r6k), antirez's ds4 and r6m,
and timed r6m's parts against B12X's on the same data
([prefill bench](../2026-10-15-engram-io/README.md#prefill-2026-10-15)):
- r6m's gather polled the control block in host memory from every (head, token)
  block: about 12 us a token and table, 100 ms per table of an 8K prefill chunk. That
  is the whole regression. Without the poll the gather matches or beats B12X's lookup
  (709 against 776 us at 8192 tokens), with the same bits.
- The reader is within 6-8% of B12X's at prefill sizes.
- B12X's lookup reads only this rank's rows (about a quarter at TP4) and B12X's
  reader reads the two tables one after the other, so the first Engram layer's rows
  are never held back by the second's. r6m lost both.
- On random rows an 8K chunk reads about 146K 4 KiB blocks, 600 MB and 100 ms per
  table and rank. The release suite's prefill sends real text (the Python standard
  library's source, with a fresh cache salt), so its numbers include Engram I/O on
  real n-grams.

## v2 (vLLM branch `engram-io-v2`, on r6m's 0145)

Kept from r6m: the native reader (no GIL, so no Python thread on the GPU's critical
path), fail-stop on a stalled or failed read, the B12X-free TileLang gather, one path
(no asynchronous or synchronous mode, no resident scales).

Changed:
- **One wait per table and step.** A one-thread TileLang kernel waits for the
  table's step before the gather, bounded by the timeout. It also reads the reader's
  phase, so a failed read stops the GPU at once with its record instead of at the
  timeout. No gather block touches the control block.
- **This rank's rows only.** The gather loads a slot only when its row is in this
  rank's shard and writes zeros otherwise; the reader no longer zero-fills the other
  ranks' slots (about 75% at TP4).
- **Tables in layer order.** One worker serves every reader's steps in the order the
  model stages them, so the second table never competes with the first for the disk.

Not changed: what the reader reads (two planes, O_DIRECT, 4 KiB blocks), the TileLang
hash, the all-reduce and the projection. Nothing depends on the TP size: a rank reads
its share of each step's lookups, so TP3 and TP4 use the same settings.

## Measurement before an image ([v2.json](v2.json), [bundles/v2](bundles/v2))

On the r6m image with v2's files overlaid, dgx4:
- the reader and gather tests, including other ranks' slots filled with E4M3 NaN
  (never read), two tables published in staging order, and a reader failure that
  stops the GPU without waiting for the timeout;
- the gather (wait included) against B12X's lookup at 16-8192 tokens, every row local
  and a quarter local: at most 3 us slower at any size;
- a step's two tables from staging: table 0's rows no later than B12X's serial reader
  by more than 5%.

Only if both pass: an image, the gate, qualify's boots and the release against r6k.

### Run 1 (2026-10-15 13:40 UTC)

- Gather, wait included, against B12X's lookup: +1.0 to +2.3 us at 16-512 tokens
  (the wait's launch), 10 us faster at 2048 and 36-58 us faster at 8192, bits equal
  everywhere; with a quarter of the rows local, 540 against 576 us at 8192.
- Two tables: level with B12X's serial reader at 2048 and 8192 tokens (table 0 at
  27.3 against 28.3 ms at 8192), but table 0 0.11 ms later at 96 tokens (1.38
  against 1.27 ms): the worker polled for the row IDs with 20 us sleeps that the
  default 50 us timer slack stretched. It now spins for 200 us, then sleeps 10 us
  with a 1 ns slack.
- The two fail-stop child tests stopped the GPU as designed, but the gather's first
  launch then failed to load its module and the child exited before printing; the
  children now catch the stopped GPU at the launch or the synchronize.

### Run 2 (13:44 UTC)

All 8 tests pass. The gather is within 1.4 us of B12X's lookup at 16-96 tokens and
faster from 512 tokens. The two tables are faster than B12X's serial reader at 2048 and
8192 tokens (table 0 at 26.6 against 28.2 ms at 8192), but table 0 came 0.38 ms later
at 96 tokens (1.67 against 1.29 ms). The bench staged with the GPU idle, so the IDs
were on the host at once and the time included waking the worker from its idle wait,
while B12X's reader ran on the calling thread. In serving the runner stages a step
while the GPU still runs the previous one, so the worker is already waiting when the
IDs arrive.

### Run 3 (13:47 UTC): GPU idle and busy at staging

All 8 tests pass. Gather (wait included) against B12X's lookup, us warm:

| tokens | all rows local | a quarter local (TP4) |
|---|---|---|
| 16 | +1.1 | +1.2 |
| 96 | +0.8 | +0.7 |
| 512 | -5.0 | -2.0 |
| 2048 | -9.6 | -1.9 |
| 8192 | -56.7 | -19.7 |

Bits equal everywhere; the small-size cost is the wait's launch.

Two tables, v2 minus B12X's serial reader, ms (median of 8 steps, a quarter local):

| tokens | GPU idle at staging: table 0 / table 1 | GPU busy at staging: table 0 / table 1 |
|---|---|---|
| 96 | +0.17 / +0.21 | +0.05 / +0.03 |
| 2048 | +0.08 / +0.04 | -0.17 / -0.50 |
| 8192 | -0.87 / -1.95 | -1.16 / -2.41 |

The busy kernel's calibration ran short (its times came out below zero), so only the
differences are meaningful there, and the bench's ratio verdict on them is void. B12X's
reader here is its best case, a synchronous read on the calling thread; r6k serves it
through a Python executor thread polling the event every 50 us. No gross regression
remains, so the release against r6k decides. The reader's spin now also yields the
core (after run 3).
