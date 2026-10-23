# json_count: the receipt's extra qeval miss is the model's own answer

The external receipt [2026-10-17-site-tp4-4k-1017](../2026-10-17-site-tp4-4k-1017/README.md) (PR #346)
reported one qeval task that our recipe fails reproducibly at temperature 0: `json_count` answers
`{"vowels": 11}` where 10 is correct, in both battery runs with identical text. The thank-you on #346
promised a root cause. The owner's rule applies: a reproducible failure is a bug until shown otherwise.

The task (thinking off, temperature 0, 300 tokens):

> Count the vowels (a,e,i,o,u, case-insensitive) in the string "Encyclopaedia Britannica" and return
> ONLY {"vowels": &lt;integer&gt;}.

Encyclopaedia has six (E, o, a, e, i, a) and Britannica four (i, a, i, a).

## Result

**DeepSeek-V4.1-Flash-1017 answers 11 here, as DeepSeek's own reference implementation computes it.**
Our serving matches that reference. There is nothing to fix.

- **Reproduced on production** (r6v, tp4-ring, 2026-10-18 01:14 UTC): the same 8-token reply,
  identical in all nine calls. 11 is the argmax by 0.79 nats (p 0.61 against 0.27 for 10), so no
  sampling or near-tie is involved.
- **The request is rendered correctly.** The server's prompt is the 39 ids that the checkpoint's
  `encoding/encoding.py` gives in chat mode
  (`<｜begin▁of▁sentence｜><｜User｜>…<｜Assistant｜></think>`). The battery's `thinking: false` takes
  effect (0 reasoning tokens). The request's temperature 0 overrides the recipe's sampling defaults
  (1.0 / 1.0), and the battery's two runs repeating the same text confirm greedy decoding.
- **The reference gives the same reply.** The checkpoint's `inference/model.py` (snapshot `1bae0e10`)
  produces `{"vowels": 11}` token for token, with p(11) 0.62 and p(10) 0.26 at the number. It ran
  unchanged on CPU, with its six TileLang kernels replaced by PyTorch versions of the same arithmetic
  (below).
- **Production is within the reference's own noise.** The same reference with FP64 instead of FP32
  accumulation is an equally valid implementation, and it differs from the FP32 run by more than
  production does:

  | Against the FP32 reference | Prompt-token \|Δ logprob\|, median / mean / max |
  |---|---|
  | Reference, FP64 accumulation | 0.087 / 0.395 / 3.13 |
  | Production r6v | 0.056 / 0.364 / 2.54 |

  This model quantizes activations to FP8 with power-of-two block scales before nearly every
  linear layer. A rounding difference that moves one block's amax across a power of two changes
  that block's scale twofold, so summation order alone moves improbable tokens by nats. The largest gaps in both rows
  are at tokens the model gives −10 to −26.
- **DeepSeek's 2026-10-12 WO-A input quantization (adopted in r6u) is not what decides it.** With it
  removed, the reference still answers 11, but only by 0.08 nats (0.48 against 0.45). The
  quantization widens the margin to 0.87. At a 0.08 margin, any arithmetic that departs from the
  reference can land on either side. We did not examine why the receipt's other engine answers 10.
- **With thinking on**, production spells the letters out and answers `{"vowels": 10}`
  (222 reasoning tokens). The miss is a guess the task forces by asking for the count without any
  reasoning.

The number step of every run (log-probabilities; [runs/compare.txt](runs/compare.txt)):

| Run | WO-A input quant | Accumulation | 11 | 10 | 12 | Margin | Reply |
|---|---|---|---:|---:|---:|---:|---|
| production r6v | yes | (GPU) | −0.501 | −1.293 | −2.248 | 0.79 | `{"vowels": 11}` |
| reference | yes | FP32 | −0.471 | −1.337 | −2.299 | 0.87 | `{"vowels": 11}` |
| reference | yes | FP64 | −0.544 | −1.111 | −2.629 | 0.57 | `{"vowels": 11}` |
| reference | no | FP32 | −0.724 | −0.806 | −3.302 | 0.08 | `{"vowels": 11}` |

## Method

**Production replay** ([replay.py](replay.py), run on dgx1 against :8000;
[runs/production-r6v.json](runs/production-r6v.json)) makes five requests:

- `/tokenize` of the chat with thinking off;
- the battery's own chat body;
- the same body with logprobs and token ids;
- `/v1/completions` over the rendered ids with `prompt_logprobs`, which scores every prompt position;
- the same chat with thinking on.

**The reference on CPU** ([reference/ref_logits.py](reference/ref_logits.py)) imports the snapshot's
`inference/model.py`, `inference/engram.py` and `encoding/encoding.py` unchanged. It puts
[reference/shim/kernel.py](reference/shim/kernel.py) in place of `inference/kernel.py`. The shim
reimplements `act_quant`, `fp4_act_quant`, `fp8_gemm`, `fp4_gemm`, `sparse_attn` and
`hc_split_sinkhorn` and keeps the kernels' rounding points:

- the same amax rules;
- power-of-two scales from the same exponent bit arithmetic;
- the same FP8 casts, and FP4 casts with round-to-nearest-even (checked against `ml_dtypes`);
- attention probabilities cast to BF16 before P·V, with the softmax sum kept in FP32;
- FP32 accumulation of exactly dequantized operands. Only summation order differs from the GPU
  kernels, and the FP64 variant measures what that costs.

[reference/test_shim.py](reference/test_shim.py) checks the shim against `ml_dtypes`, against dense
dequantized products and against a plain softmax.

The driver keeps only one layer of the model in memory at a time:

- It runs at world size 1, so nothing is sharded. At `--model-parallel 1`, `convert.py`'s renames
  change nothing on the 1017 shards, which already carry the reference's names. Its two conversions
  are applied on load instead: `wo_a` is dequantized to BF16, and the experts' int8 bytes are viewed
  as packed E2M1.
- Each layer is built, loaded, run and freed in turn, and its KV caches carry across steps.
- A routed expert is read only when a token routes to it, and Engram table rows are read on lookup.
- Tensors are read from the Hugging Face shards with O_DIRECT, so nothing lands in the page cache.

As `generate.py` does, the driver prefills the 39 prompt tokens and then decodes one token per
forward. The reply is teacher-forced through production's tokens up to the number and continued
greedily, and at every reply step the reference's own argmax was production's token.

It ran on dgx4's CPU beside production without a window or any change to the service:

- cores 0-4 and 10-14 (production is pinned to 5-9 and 15-19);
- a 6 GiB container cap, no network, disk reads throttled to 800 MB/s;
- MemAvailable read 20.8-21.3 GiB at every check;
- each run took about 8 minutes and read about 152 GiB.

The two FP32 runs agree bit for bit: [runs/ref-woa1-run1.json](runs/ref-woa1-run1.json) predates
prompt scoring, and [runs/ref-woa1.json](runs/ref-woa1.json) adds it. The FP32 runs used the shim
before its `REF_ACCUM` option was added; with the default, the option leaves the FP32 code path
unchanged. The order and exact commands
are in [reference/run_pair.sh](reference/run_pair.sh) and [reference/run_fp64.sh](reference/run_fp64.sh).
Per-layer residual dumps (`*-hidden.pt`, 13 MB a run) stay in `~/spark-lab/json-count/out/` on dgx4.

```sh
python3 compare.py runs/production-r6v.json runs/ref-woa1.json runs/ref-woa1-fp64.json runs/ref-woa0.json runs/ref-woa1-run1.json
```

## Limits

- The reference ran as its model code with reimplemented kernels, not its TileLang kernels on a GPU.
  The FP64 control bounds that difference: it moves the margin at the number by 0.3 nats, and
  every valid variant's margin is 0.57-0.87.
- One prompt.
- Production was compared at TP4 on r6v. The receipt's r6u with 0180 on another ring gave the same
  text. The triangle (TP3, the same image, a different reduction split) was not replayed.

## Files

- `replay.py`, `compare.py`
- `reference/`: `ref_logits.py`, `shim/kernel.py`, `test_shim.py`, `run_pair.sh`, `run_fp64.sh`
- `runs/`:
  - `production-r6v.json`;
  - `ref-woa1.json` (the reference as shipped), `ref-woa1-fp64.json` (FP64 accumulation),
    `ref-woa0.json` (WO-A input quantization off), `ref-woa1-run1.json`;
  - their logs (`*-log.txt`);
  - `compare.txt`.
