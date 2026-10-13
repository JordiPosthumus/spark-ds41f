# Native bad_words during loop protection

Owner-approved change: replace custom whole-word phrase masking with the pinned
vLLM `BadWordsState` and `_bad_words_kernel`. The exact 16 strings are in
`tables/bad_words.json`. Only capitalized `Actually` is included.

Custom code remains explicitly limited to the detector, continuation/retry,
word-window accounting, and the optional enable flag/history boundary supplied
to vLLM. The native phrase comparison and logits write are unchanged.
`native/stock_bad_words.py` records the exact source for review.

The existing trigger and severe-loop retry policy are unchanged. Early detection
continues the accepted prefix with a new seed, unchanged temperature, and no
rewind. A severe-loop retry can rewind model context; already streamed text
cannot be retracted. All attempts share the original output-token budget.

The restriction ends after 2000 completed words (at the next token boundary),
or an earlier genuine reasoning/tool boundary. Stock client `bad_words` requests
keep their ordinary full-request lifetime and are excluded from this recovery
intervention. No prompt text or user session traces are included here.

Native bad_words prohibits encoded token sequences, not every possible spelling
or alternative tokenization of a character substring. Normal model-tokenizer
encodings of Let mego, Let meprint, Let me看, Hmmh, and OKwait are tested. Mixed
case, altered whitespace, and alternate token segmentations are not promised.
Blocking does not establish that a model will recover useful reasoning.

Validation: tests/ exercises API continuations, expiry, cancellation, boundaries,
and trigger rules without a model; native/validate_cuda.py checks actual GPU
masking, stock-matcher equivalence, derivatives, mixed requests, speculative rows,
retained-prefix boundaries, expiry, slot reuse, and CUDA graph replay. Runtime
canary receipts are retained separately. Image, kernels, context, concurrency,
multimodal settings, temperature and unrelated configuration remain unchanged.

Rollback: restore each ring's preceding clean recipe commit using bin/spark
cluster sync/start in a drained idle window. Keep the preceding image and caches.
