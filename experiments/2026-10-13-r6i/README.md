# r6i: sparknet main (TileLang collectives, big-core proxy) and the TileLang L2 prefetch

The one-shot all-reduce is 15-16 ms of a 90 ms sixteen-stream TP4 step (17%, every rank
alike: [r6h-loss](../2026-10-12-r6h-loss/README.md)), and a larger share on TP3, where
r6f's FP32 MoE partials doubled its bytes. r6h serves sparknet `f73a3ce`. sparknet main
(`b61660f`, 0.3.0) adds two things measured on the probe but never in serving
(`docs/oneshot.md` in sparknet):

- **The TileLang kernel family** (`SPARKNET_ROCE_KERNELS=tilelang`): bit-identical to
  the CuTe kernels, 3-5% faster at 480 KiB on the ring and about 1% at 2 MiB on a pair,
  level after the residency bound (`2aa3de0`).
- **Big-core proxy placement** (`SPARKNET_ROCE_PROXY_CPU=big`): the proxy thread polls a
  host doorbell, and on the GB10 the scheduler may put it on a Cortex-A725. Unpinned, a
  10 KiB all-reduce spread over 16-20 us; confined to the X925 cores, 17.8-18.4 us.

The third switch is the migration's C1: the TileLang L2 prefetch
(`VLLM_L2_PREFETCH_KERNELS=tilelang`, already in the r6h image), which passed its kernel
gate and screened level on r6c. With C1 and C2 both on, no standalone CuTe DSL kernel
remains in the serving path; B12X's kernels are what remains of that engine.

All three switches keep the output bits: the collectives' families are bit-identical,
placement moves a host thread, and a prefetch changes no arithmetic.

## Inputs

- [upstreams.lock.json](upstreams.lock.json): the root (r6h) lock with sparknet at
  `b61660f` (version 0.3.0); [source.json](source.json): r6h's source manifest with
  sparknet's tree `ff98c92b8b3f`. vLLM's sparknet interface (`SparknetOneShotAllReduce`,
  `freeze_kernel_resolution`) is unchanged since `f73a3ce`.
- [r6i-tp4.json](r6i-tp4.json): the promoted `config/cluster-tp4.json` (head in regular
  memory, its curves) on image `-r6i`; [r6i-tilelang.json](r6i-tilelang.json),
  [r6i-tlbig.json](r6i-tlbig.json) and [r6i-full.json](r6i-full.json) add the switches
  one at a time; [r6h.json](r6h.json) is production.

## Image ([image.json](image.json))

Built on dgx4 and copied over the CX7 links like r6h's. Gate: `--wide` determinism with
and without logprobs on the defaults (CuTe, unpinned); the references must stay r6f's
(`7ae87256d3ea` / `b8b35451dcc4`).

## Window 1 ([w1.json](w1.json))

Candidate first: full, r6h, sparknet main alone, + TileLang, + big-core proxy, full
again. Each arm: `--wide` determinism, six samples of sixteen and eight different
prompts, four of four. All arms share the production cost curves, so the switches are
the only variables. A release comparison against r6h follows if the full candidate is
level or better everywhere.

### Image (2026-10-12, 22:33-22:49 UTC)

`vllm-ds41f-kkref:19f2c20ed4d6-r6i` (`sha256:91a9cd30ca5b`) built on dgx4 in 3.6 minutes
(vLLM layers cached) and copied to every node. Gate: identical in every scenario with and
without logprobs; references `7ae87256d3ea` and text digest `cd8f94c899d3`, as r6h.

### Window 1 results (22:49-23:36 UTC)

Every arm identical at temperature 0 with r6f's references. Different prompts at once,
tok/s:

| arm | 16 | 8 | 4 |
| --- | ---: | ---: | ---: |
| full (TileLang collectives, big-core proxy, TileLang prefetch) | 226.0 | 179.4 | 134.9 |
| r6h | 225.1 | 179.2 | 134.9 |
| sparknet main alone | 224.5 | 179.3 | 134.9 |
| + TileLang collectives | 225.8 | 178.9 | 134.7 |
| + big-core proxy | 225.0 | 179.2 | 134.8 |
| full again | 225.3 | 179.6 | 135.0 |

All level within the intervals (±0.2% at eight prompts, ±0.7-0.9% at sixteen). The
probe's 3-5% collective gain and the placement's narrower spread do not show in
serving throughput. r6i full is a migration step at r6h's speed and bits: no standalone
CuTe DSL kernel left in the serving path.

## Window 2 ([w2.json](w2.json)): adaptive verification's cost scale on real-row curves

The cost scale 2.0 dates from padded-row cost curves, which priced a verified row at
about 0.5 ms against about 6.5 ms on real rows (TODO, "Re-check verification pricing").
Since r6g the curves are profiled on real rows, so 2.0 may double-count and verify too
few drafts. The scale cannot change the output (it only sets how many drafts a step
verifies). Arms on r6i full: 1.0, 2.0, 1.5, 1.0 again, each with the paired
sixteen-prompt sweep (one stream) and sixteen, eight and four different prompts; then
determinism on 1.0.

### Window 2 results (2026-10-13, 02:02-02:31 UTC)

Against scale 2.0, same session (scale 1.0 measured twice):

| | 1.0 | 1.0 again | 1.5 |
| --- | ---: | ---: | ---: |
| accepted drafts per step, paired over sixteen prompts | **+2.2% ±1.0** | **+2.2% ±1.0** | +1.5% ±0.8 |
| one stream, sixteen prompts paired, tok/s | +0.7% ±0.6 | +0.4% ±0.7 | +0.6% ±0.5 |
| four / eight / sixteen different prompts | level | level | level |

Identical at temperature 0 with r6f's references. On real-row curves 2.0 priced
verification too high; 1.0 verifies more drafts and accepts 2.2% more per step without
slowing any concurrency.

## Release comparison ([release-tp4.json](release-tp4.json))

The candidate ([r6i-candidate.json](r6i-candidate.json)): r6i full with cost scale 1.0
and its own curves (`ring4-r6i-20261013`, real rows, profiled on first boot). r6i /
r6h / r6i again under the release profile. No separate fan pre-cool: the cluster had
been lightly loaded, and each bench brings every node below 55 °C first.

### Release comparison results (2026-10-13, 02:33-04:19 UTC)

r6i / r6h / r6i again. Both r6i arms identical at temperature 0 (r6f's references) and
5/5 on quality. Against r6h:

| | r6i | r6i again |
| --- | ---: | ---: |
| sixteen different prompts | +0.7% ±0.7 | +0.3% ±0.6 |
| eight | **-1.0% ±0.8** | **-1.2% ±1.0** |
| four | -1.0% ±1.3 | -0.9% ±1.3 |
| paired sweep, tok/s / accepted per step | +0.0% / +1.1% ±1.3 | +0.1% / +1.1% ±1.3 |
| prefill 32K-1M | level (512K -0.4% ±0.4) | level (1M -0.2% ±0.2) |
| mixed traffic p95 stream gap (one run) | +5.0% | +6.2% |

**Not promotable:** eight prompts is slower in both arms. Window 1 put the migration
switches level at eight prompts, so the loss comes with cost scale 1.0 and its curves:
verifying more drafts lengthens the eight-stream step by more than the extra accepted
drafts return (acceptance +1.1% ±1.3 here against window 2's +2.2%). The cost scale moves
to the draft-acceptance work, judged per concurrency. The migration part (TileLang
collectives, big-core proxy, TileLang prefetch) goes forward on r6j with scale 2.0 and
production's curves, and needs its own release comparison.
