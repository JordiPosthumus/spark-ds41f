# r6q: r6p with Engram's exchanged blocks read in place

## Delta

r6p ([experiments/2026-10-16-r6p](../2026-10-16-r6p/README.md)) exchanges Engram's `wkv`
product all-to-all under prefill sequence parallelism, but it pads each rank's `[T, C]`
block to the SP row count before sending, and copies the received blocks into packed
rows. Those are two copies of about 69 MB per call on TP3, roughly 10 ms of a 16K
prefill.

**0152** sends each rank only its real rows of every block, through a row-block
all-to-all (`all_to_all_rows`). The TileLang gate reads the received `[N, L, C]`
column blocks in place (`engram_gate_blocked`): the same loads in the same order, so
the same bits as the packed gate. The B12X mix still takes packed rows.

Source: vLLM tree `0643f0d2` (patch head `3cba8566`).

## Tests (dgx4)

- The blocked gate's bits equal the packed gate's at TP3's and TP4's block widths,
  masked and unmasked, with rows past the tokens set to NaN: 13 of 13 gate tests.
- Sequence-parallel: 155 of 155. Engram model: 25 of 25.

## Runs

1. [gate.json](gate.json): outputs identical to r6n's; 8 and 4 distinct streams.
2. [profile.json](profile.json): 16K prefill on r6q, r6p and r6q again, with pinned
   cost curves.
3. [release.json](release.json): release-tp3 against r6n, the release comparison for
   the Engram change set. The owner decides promotion.
