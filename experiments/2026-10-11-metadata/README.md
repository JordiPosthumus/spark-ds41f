# Port T1: the sparse-MLA metadata kernels in TileLang (r6d)

The first of the migration's Triton items (see the
[inventory](../2026-10-09-tilelang-migration/README.md)): the four DS4.1 metadata
kernels.
- `sparse_mla._requests` and `_tokens` launch eagerly in each KV-cache group's
  metadata build.
- `attention._chunk` and `_pages` write sliding-window and page metadata. Decode does
  this once per group per forward, inside the graphs; prefill does it per layer.

## The change (vLLM branch `r6d-metadata`, on r6d's tree)

- `6fdb42e28` (module): `tilelang/metadata.py`, with the same contracts element for
  element:
  - integer arithmetic with Triton's truncating division;
  - the same rows written: `_tokens` refreshes the first `ceil(nt / 128)` blocks;
  - ring rows bounded to a request's last ring-size rows.
- `21b416dc7` (switch): the builder and the attention layer call the TileLang kernels
  under the TileLang family. They compile with the builder and at the attention
  declaration, never during graph capture.

## Window 1 ([w1.json](w1.json))

1. Kernel bundle: the tests compare every output with the Triton kernels over packed
   steps (empty and negative-position requests, padding, holes, ring and paged groups,
   draft and replay windows). Then [bench_metadata.py](bench_metadata.py):
   - bits;
   - the builder's eager time (host launch included);
   - the layer metadata's graph time at decode steps and prefill chunks.
2. Lean screen bracketed by r6d with `token_determinism.py --wide`. The metadata is
   exact, so outputs and references must equal r6d's.

Acceptance: identical outputs to r6d; the bench no slower than Triton anywhere.

### Window 1 kernel results (2026-10-11, 11:45 UTC)

All 70 tests passed: every output identical to Triton's.

- **Builder metadata (eager, host launch included):** 15-21% faster than Triton at
  every step (about 17.5 against 21 us).
- **Layer metadata under graphs:** the window and page kernels were 0.2-0.4 us slower
  than Triton's pair at decode (2.3-2.5 against 1.9-2.3 us) and up to 17% slower at
  2048-row prefill.

The decode metadata writes both for the same rows back to back, so the follow-up
(vLLM, top of `r6d-metadata`) writes them in one launch. Window 2
([w2.json](w2.json)) is the kernel bundle again, kernel only.

### Window 2 (2026-10-11, 12:24 UTC, kernel only)

85 tests passed. The fused launch beat Triton's pair warm at every size (0.77-0.98x),
but cold it lost:
- decode was level at the timer's resolution;
- prefill was 14-17% slower (14.1 against 12.1 us at 2048 rows), because one block a
  row did both parts.

vLLM `f23f050fc` gives each row a block for its window metadata and one for its page
table, still one launch. Window 3 ([w3.json](w3.json)) is the kernel bundle again.

### Window 3 (2026-10-11, 12:54 UTC, kernel only)

85 tests passed. With two blocks a row:
- **Decode:** 0.68-0.72x Triton warm and cold.
- **Builder:** 0.81-0.85x.
- **Prefill:** 1.22-1.43x. Each block took the page table's width (256 threads), half
  idle in the window part, across 4096 blocks at 2048 rows.

vLLM `18ac093ba` makes blocks as wide as the window (128 threads), with the page part
looping. Window 4 ([w4.json](w4.json)) is the kernel bundle again.

### Window 4 (2026-10-11, kernel only)

85 tests passed: every output identical to Triton's. With window-wide blocks:

| Path | Triton | TileLang | Ratio |
| --- | ---: | ---: | ---: |
| Builder (eager, host launch included), every step | 20.1-21.1 us | 17.2-17.6 us | 0.83-0.86x |
| Layer metadata under graphs, decode 1-96 rows (warm / cold) | 1.96-2.30 / 6.14 us | 1.31-1.44 / 4.10-4.26 us | 0.63-0.69x |
| Layer metadata, 2048-row prefill chunk | 8.52 / 12.32 us | 7.10 / 10.53 us | 0.83 / 0.86x |
| Layer metadata, 8192-row prefill | 40.63 / 42.98 us | 41.13 / 41.02 us | 1.01 / 0.96x |

Faster everywhere except 8192-row prefill, which is level (0.5 us slower warm, 2 us
faster cold, of about 40 us). This is the version that shipped in r6f (the layer
metadata rows of `bench_metadata.py` label the Triton column `b12x`).
