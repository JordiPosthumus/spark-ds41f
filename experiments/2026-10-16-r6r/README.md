# r6r: r6q with disk Engram shards on whole hash columns

## Delta

A layer's table is its 24 hash columns' prime-sized ranges back to back, each just
above the Engram vocabulary (about 16 million rows). Disk shards split the rows
equally, so on TP3 and TP4 a rank's shard cut a column by about a thousand rows. The
ranks' gathered rows then had to be summed: an all-reduce of `[T, 6144]` per Engram
layer, 26-30 ms of a TP3 16K prefill on dgx3 (r6q's profile).

**0153** puts a disk shard on whole columns when the TP size divides 24. Rank r then
holds rows `offsets[24r/N]` to `offsets[24(r+1)/N]`, within 0.002% of an equal split.
Its gathered rows are one contiguous band of `24/N * 256` values, and an all-gather
of the bands (half the bytes, one phase) replaces the all-reduce. Adding zero rows no
longer turns a -0.0 into +0.0, as the single-device reference never does. Resident
tables keep equal shards and the all-reduce.

Source: vLLM tree `13975149` (patch head `a4a52221`).

## Tests (dgx4)

- Column shards tile the table at column offsets, and their bands tile the row in rank
  order (TP 1, 2, 3, 4, 6; none for 5).
- Engram model: 30 of 30. Sequence-parallel: 155 of 155. TileLang gate: 13 of 13.

## Runs

1. [gate.json](gate.json): outputs identical to r6n's (or every difference
   explained by -0.0); 8 and 4 distinct streams.
2. [profile.json](profile.json): prefill, single-stream decode and eight streams on
   r6r, r6q and r6r again, with pinned cost curves. The rows collective is an
   all-gather in both prefill and decode.
3. [release.json](release.json): release-tp3 against r6n.
