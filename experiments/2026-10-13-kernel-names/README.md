# Named TileLang kernels (r6j)

TileLang names a CUDA kernel after its `@T.prim_func`, and 39 of ours were called
`main`: every profile, trace and SASS dump showed `main_kernel`, and kernels had to be
told apart by grid signature (for example `experiments/2026-10-12-r6h-loss/group_costs.py`),
which slowed debugging more than once.

vLLM patch [0118](vllm/0118-TileLang-V4.1-name-every-kernel-after-its-file-and-w.patch)
names every kernel `<file>_<what it computes>`, or the file's own name for a file's single
kernel, so a profile line names the module that holds it (TileLang appends `_kernel`):

| file | kernels |
| --- | --- |
| `activation.py` | `activation_swiglu_clamp` |
| `cache_writer.py` | `cache_writer` |
| `collectives.py` | `collectives_rank_order_sum` |
| `compress.py` | `compress`, `compress_gather_partial_state` |
| `engram.py` / `engram_hash.py` | `engram_gate` / `engram_hash` |
| `gemm.py` | `gemm_mxfp8`, `gemm_mxfp8_partials`, `gemm_bf16`, `gemm_bf16_two`, `gemm_bf16_partials`, `gemm_bf16_splitk`, `gemm_splitk_reduce`, `gemm_splitk_reduce_two` |
| `indexer.py` | `indexer_score`, `indexer_mxfp4_quantize`, `indexer_topk_segment`, `indexer_topk_merge` |
| `metadata.py` | `metadata_requests`, `metadata_tokens`, `metadata_chunk`, `metadata_pages`, `metadata_decode` |
| `mhc.py` | `mhc_project_streams`, `mhc_finalize` |
| `moe.py` | `moe_dispatch`, `moe_combine`, `moe_grouped_fp8_fp4_gemm` |
| `rope.py` / `router.py` | `rope` / `router_partials` |
| `sparse_mla.py` | `sparse_mla` |
| `vocab.py` | `vocab_packed_projection` |
| `glm5next/nvidia/tilelang/l2_prefetch.py` (moved from `glm5next/nvidia/l2_prefetch_tilelang.py`, the same family-directory layout as `deepseek_v4_1/tilelang/`) | `l2_prefetch` |

Variants one factory builds (decode or prefill, a tile, a row bucket) share its name and
differ by grid, as before, so retuning a tile never renames a kernel. Only each
prim_func's `def` and `return` and the prefetch's path change; the generated code and the
results do not. sparknet's TileLang collectives and DeepSeek's TileKernels live in their
own packages and are not renamed here.


## Inputs

- [vllm/series](vllm/series): r6h's series plus 0118. Replayed as build prepare does,
  the r6h part reproduces r6h's patch head `e39cd9da9938`; with 0118 the head is
  `1c82f93146fb`, tree `36e5acb9fa6a`.
- [upstreams.lock.json](upstreams.lock.json), [source.json](source.json): r6i's (sparknet
  main) with that series.
- [r6j-tp4.json](r6j-tp4.json): the r6i release candidate (sparknet TileLang collectives,
  big-core proxy, TileLang prefetch, cost scale 1.0, its curves) on image `-r6j`.

## Image ([image.json](image.json))

Built on dgx4 and copied like r6i's. Gate: `--wide` determinism with and without
logprobs (references must stay r6f's), sixteen and eight different prompts, and a
fixed-row profile ([r6j-fixed.json](r6j-fixed.json), code16) whose kernel summaries must
show the names.

## Release comparison ([release-tp4.json](release-tp4.json))

The promotable migration candidate ([r6j-release.json](r6j-release.json)): image `-r6j`
with sparknet's TileLang collectives, the big-core proxy and the TileLang prefetch, cost
scale 2.0 and production's cost curves (`ring4-r6h-headout-20261012`). That is r6i window
1's "full" arm (level with r6h, same bits) plus the named kernels; r6i's release showed
cost scale 1.0 losing at eight prompts, so it stays out. r6j / r6h / r6j again, queued on
dgx1 behind the image job so it runs unattended; the queue restores production (r6h) when
it is done.

### First attempt (2026-10-13, 04:19-07:24 UTC): the prefetch was off

The image gate passed on bits and names (no `main_kernel` left; the profile showed
`moe_grouped_fp8_fp4_gemm_kernel`, `gemm_mxfp8_kernel`, `sparse_mla_kernel`,
`mhc_project_streams_kernel`, `vocab_packed_projection_kernel`, ... and sparknet's
`oneshot_allreduce_kernel`). The release comparison then read r6j 5.5-7.5% slower than
r6h on every decode measure (paired sweep -7.4%, eight prompts -5.5%), with acceptance and
prefill unchanged, in both r6j arms.

Cause: the patch moved the TileLang prefetch file but its amend left out the launcher's
import change, so the launcher could not import the kernel, logged a warning and served
with no L2 prefetch at all (the profile had no prefetch kernel). Fixed in 0118, and vLLM
[0119](vllm/0119-L2-prefetch-refuse-to-run-without-the-TileLang-kerne.patch) makes an
explicitly requested TileLang prefetch that fails to compile stop the start instead of
disabling itself. Checked in a GPU container on the r6j image with the corrected file:
"TileLang kernel ready". New head `004b7cb75fdf`, tree `c4d86d34146d`; the image job and
the release comparison run again.

### Release comparison results (2026-10-13, 10:35-12:19 UTC)

r6j / r6h / r6j again, with the prefetch fixed. Both r6j arms identical at temperature 0
(r6f's references), 5/5 on quality. Against r6h:

| | r6j | r6j again |
| --- | ---: | ---: |
| paired sweep, one stream, tok/s | **-0.4% ±0.1** | **-0.3% ±0.1** |
| accepted per step | +0.0% | +0.0% |
| four / eight / sixteen different prompts | -0.2 / -0.1 / -0.0% | -0.2 / -0.2 / -0.5% |
| prefill 32K-256K | level | level |
| prefill 512K / 1M | -0.3% ±0.3 / **-0.2% ±0.1** | -0.1% / **-0.2% ±0.2** |
| mixed traffic | level | level |

Level at concurrency and in prefill up to 256K, but one stream reads 0.3-0.4% slower in
both arms (the paired sweep runs identical text, hence its narrow interval) and 1M prefill
0.2% slower. r6i's window 1 screened the switches only at four, eight and sixteen prompts;
at one stream the collectives, the proxy placement and the prefetch are latency-bound. Next:
a bisect at one stream and 1M before promotion.

The profile of the fixed image still shows no prefetch kernel under either family's name,
although throughput shows it running (the broken image lost 5-7% of decode without it):
the profile summary does not see the TileLang prefetch on its side stream. Open.
