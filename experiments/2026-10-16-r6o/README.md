# r6o: r6n with step-aware Engram rows

## Hypothesis

The first Engram layer's wait for its disk rows is prefill time on both topologies.
On TP3 it is 53 / 86 / 119 ms per 16K prefill on the three ranks
([experiments/2026-10-16-tp3-prefill](../2026-10-16-tp3-prefill/README.md)), because
table 0's rows are read only after the GPU hashes the step at its start, about 50 ms
before layer 1. Reading the rows of the tokens the host already knows before the step
starts should remove most of that wait. It should change no output bit and cost decode
nothing.

## Intended delta

r6o is r6n ([experiments/2026-10-15-r6n](../2026-10-15-r6n/README.md)) with four vLLM
changes on top of its series:

1. **0148, step-aware Engram rows.** The runner prepares a step a step ahead of the GPU,
   and a prefill chunk's token ids are on the host by then. The model state keeps each
   request's prefill tokens and hashes the chunk's rows on the host
   (`engram_rows/host_hash.py`). It uses the TileLang hash's arithmetic: wrapping
   multiplies, XOR, floor remainder. The step carries those IDs as its early IDs. Rows
   of tokens the GPU samples are marked unknown and still come from the GPU's hash.
   One reader serves both. It reads the early rows as soon as the GPU has gathered the
   previous step from the same buffers; the gather now ends with a one-thread release
   of a CONSUMED sequence word. It then waits for the GPU's IDs only if a lookup is
   unknown, reads those rows, and publishes the step once. So:
   - a step of prompt rows is ready before the GPU reaches it;
   - a decode step behaves exactly as before;
   - a mixed step reads its prompt rows early and its decode rows late.

   Adaptive verification gives verifying requests GPU-chosen row counts. They stay at
   the front of the batch with the host's total, so every later request starts at the
   same row on both sides. The host refuses a step where a verifying request follows a
   prefill chunk. `SPARK3_ENGRAM_CHECK_EARLY_IDS=1` (debug, off in serving) compares
   every early ID with the GPU's and names the first difference.
2. **0147, checkpoint copies through anonymous memory.** This fixes the Engram model
   test that spun in a host-to-device copy during r6n's qualification (below).
3. **0146, the Engram model test.** It compares the readers' counters once they are
   idle. This change was in r6n's test bundle but not in its series.
4. **0143's credit.** huxiaofengtiger's forum report of r6h's TP3 startup failures is
   credited with a `Reported-by:` trailer.

Source: vLLM tree `cfea587b` (patch head `a58660ee`, as the build replays it). sparknet,
B12X, NCCL, TileLang and TileKernels are as in r6n.

## The copy that never finished

r6n's qualification left one failure unexplained: the first Engram model test spun for
37 minutes in a host-to-device copy while it loaded its checkpoint. A later diagnostic
passed. On 2026-10-16 the failure reproduced every time on dgx4, with r6n's own files,
and outside pytest.

With driver R580 (580.178.04) on the 64 KiB-page kernel (7.0.0-1019-nvidia-64k), a
pageable host-to-device copy never completes when all three of these hold:

- the copy is larger than about 128 KiB (256 KiB already hangs);
- its source is file-backed pages the process has not touched yet;
- it does not start on a 64 KiB page boundary.

The host spins in `cuMemcpyHtoDAsync_v2` inside libcuda. The GPU is idle, and nothing
is logged: no Xid, no SMMU event.

GB10 reads pageable memory through the host page tables. A 40-line script reproduces
the hang without vLLM. These cases all complete: anonymous memory at any offset, pages
the process has already read, a page-aligned start, and copies of 128 KiB or less.
Dirty or written-back pages make no difference, and neither does a shared or private
mapping.

safetensors' `get_tensor` returns a view into the mmapped file at the tensor's offset.
The test's 3.9 MB weight met every condition. `copy_weight`'s plain copy now copies a
pageable CPU source into anonymous memory first, so the CPU reads the file. Every
safetensors-style reader goes through `copy_weight`. A new test copies from an
untouched mapping at offset 1504 in a child process, so a regression fails the test
instead of hanging the suite. It also checks that no pinned memory remains.

The first build pinned the source instead (`pin_memory()`), and its gate boot failed.
The startup memory guard stopped dgx1 at 4.72 GiB available during the drafter's graph
capture, below its 5 GiB floor. Torch's host allocator keeps every pinned block,
rounded up to a power of two, for the life of the process, and on GB10 that memory is
the GPU's. Production's `--load-format b12x` writer does not take every weight: the
ones it declines reach this path. The anonymous copy is freed when the copy returns.

## Quality and safety gates

- **Unit tests (dgx4, r6n's image with the changed files mounted, 2026-10-16):**
  - weight transfer: 6 of 6, including the untouched-mapping copy;
  - the Engram model tests: 25 of 25, including all six descriptor-loading variants
    that used to hang, and the new end-to-end test, in which host-hashed rows equal
    B12X's GPU hash with the check on (a first chunk, a later chunk with history, a
    mixed decode-and-prefill step with an image token in its history, and a
    decode-only step);
  - the Engram reader and hash tests: 14 of 14. These include the host hash bit for bit
    against the TileLang kernel on full-range int64 geometry; early rows written only
    after the previous step's gather; and a host-hashed step published while the GPU is
    still busy before its staging.
- **Gate ([gate.json](gate.json)):**
  - one boot with `SPARK3_ENGRAM_CHECK_EARLY_IDS=1`, running the determinism texts on
    real prompts; any host/GPU difference fails a step;
  - then the plain build: temperature-0 determinism, greedy, and 8 and 4 distinct
    streams;
  - outputs must be byte-identical to r6n's, since the rows are the same bytes, read
    earlier.
- **Never slower than r6n** in the release comparison.

## Runs

1. **Image (dgx4).** dgx4 is outside the triangle's node map, so the image is built on
   it from main with `bin/spark --cluster-config experiments/2026-10-16-r6o/r6o-tp3.json
   build prepare` and `build image --apply`, then copied to dgx1-3.
2. [gate.json](gate.json): the check boot, then the plain build.
3. [profile.json](profile.json): torch-profiled r6o, r6n and r6o again on a 16K prefill,
   a single-stream decode and an eight-stream decode window. Table 0's wait kernel
   should fall from 53-119 ms per prefill to near zero; decode should be level (its
   rows come from the GPU's hash as before), and the profiles show what Engram still
   costs in each.
4. [release.json](release.json): the release-tp3 suite on r6o, r6n and r6o again. The
   owner decides promotion.
5. TP4 at the next re-cable: the same gate and the release suite against r6n.

## Credits

huxiaofengtiger reported r6h's TP3 startup failures on the NVIDIA Developer Forums
(r6n's README has the details); vLLM 0143 now carries the credit.

## Results (TP3, 2026-10-16)

- **Gate passed.**
  - The image is the second build, with 0147 copying through anonymous memory. The
    first build's boot was stopped by the startup memory guard.
  - Temperature-0 and greedy outputs have zero token and logprob differences in every
    scenario, both on the boot with `SPARK3_ENGRAM_CHECK_EARLY_IDS=1` and on the plain
    boot.
  - The token and text reference digests equal r6n's gate digests exactly.
  - 8 distinct streams: 137.2 tok/s (±0.55%). 4 streams: 109.0 (±1.6%).
- **Prefill profile (16K, [engram_costs.py](engram_costs.py)).** The faster ranks wait
  for dgx3's rows inside the rows all-reduce, so wait plus all-reduce is the Engram time
  on each rank's critical path:

  | arm | wait + rows all-reduce, dgx1 / dgx2 / dgx3 (ms) | 16K prefill span (ms) |
  |---|---|---|
  | r6n | 151 / 151 / 152 | 3,945-3,956 |
  | r6o | 58 / 58 / 59 | 3,848-3,853 |
  | r6o again | 66 / 66 / 66 | 3,874-3,885 |

  r6o is 70-100 ms faster per 16K prefill (1.8-2.4%). The remaining 11-35 ms wait fits
  each prompt's first chunk, which has no previous step to read under.
- **Single-stream decode:** level with r6n (1,647 ms for 128 tokens on both; Engram
  wait 4-7 ms).
- **Eight streams:** 1.6-2.0% fewer tok/s than r6n, but not an Engram comparison. A
  profiling arm's serve arguments (the profiler directory) give it its own build
  fingerprint, hence freshly profiled DSpark cost curves, and adaptive verification
  then chose different draft budgets. r6o's steps ran different MoE row counts from
  r6n's (240-row grids against 192). r6p's profile pins every arm to production r6n's
  curves, and the release suite's engine grid times fixed-row steps.
