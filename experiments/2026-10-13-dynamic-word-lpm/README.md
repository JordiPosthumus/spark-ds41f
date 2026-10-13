# Dynamic word loop protection

Owner-authorized experimental intervention, built on the 2026-10-13 word alarm.
This is a sampling/quality intervention, not a lossless performance optimization.

* Entry stays four occurrences of ONE configured family in 50 completed words.
* Rewind stays at the native chunk before the first qualifying hit, with a fresh
  seed and unchanged temperature. No reminder or synthetic model text is inserted.
* Existing base bad_words entries remain active. During protection, completed
  prose words are counted in the last 3,000 generated characters. On occurrence
  13, a non-exempt word is added until protection ends. Counts use NFKC/casefold;
  code fences, inline code, indented code and numerical-only words supply no hits.
  Tool/answer boundaries disarm protection as before.
* Grammar exclusions and base spellings are explicit in tables/word_policy.json.
  The owner's capital-A-only Actually blocking exception is preserved.
* Dynamic additions use vLLM's native bad_words tokenization and matcher, with
  observed/lower/title/upper forms. They are not regex/substring restrictions or
  guarantees against all spellings, languages, token segmentations or synonyms.
  Counting exclusions are not a separate sampler grammar: an already blocked
  token sequence can also be unavailable inside later reasoning code.
* The existing 2,000-word native expiry remains. A severe-loop retry retains the
  dynamic vocabulary and rolling counts, so the retry cannot repeatedly erase
  evidence before the thirteenth occurrence. Native expiry restarts with the
  existing severe retry policy. Every genuine exit clears the active policy.
* Native worker RPC updates the current request, all TP ranks, on the existing
  serialized worker/CUDA stream. It does not reseed, rewind or re-prefill for
  word additions. Already generated/in-flight output cannot be retroactively
  blocked or retracted. Server notices report acknowledged policy additions.
* Request IDs, versions and native slot lookup isolate updates. Preemption keeps
  the vocabulary and rescheduling restores it; completion removes its record.
* A separate native bad_words table holds up to 2,048 sequences/65,536 token IDs
  per request (about 2.1 MiB of extra fixed table storage at eight requests per
  rank, plus native staging buffers). Original bad_words limits are unchanged.
  If update/encoding/capacity fails, Pi shows an explicit failure notice and the
  existing mask remains; the feature does not silently claim an addition.

Pi receives server-generated reasoning notices, outside parser/model/token usage:

    [!!! LOOP PROTECTION MODE ACTIVATED CURRENT: ...]
    [!!! ADDED: Hh CURRENT: Let, me, OK/Okay, Actually, Hm/Hmm (listed variants), Hh]
    [!!! LOOP PROTECTION MODE OFF: word_limit]

CURRENT uses readable labels for the explicit base spelling families and lists
all dynamic words. Cancelled/disconnected streams cannot receive an off notice;
server release accounting still records it. These notices never enter the
current generation prefix. Future client transcript resubmission is client-owned.

## Validation and limits

27 focused CPU tests cover chunking, exact thresholds, rolling expiry, grammar,
code exclusion, no-hint rewind, MM metadata, output budget, unchanged temperature,
notice order, cancellation, request isolation and preemption restoration. The
pinned native tokenizer was checked on CPU without an inference request. Source
and live effective settings must be checked on every rank after staged rollout.
These checks do not establish that the model will recover coherently. The owner
is evaluating generated behavior. No inference probe or benchmark is authorized
for this deployment. Actual latency is logged per word-update RPC; no zero-cost
or throughput improvement claim is made.

Upstream base: the fork's existing native-bad-words overlays on image
vllm-ds41f-kkref:19f2c20ed4d6-r6f. gpu_worker.py is based on installed SHA256
61659a7826b2d2223ba60d80946fa017b67e07ce6ff5e5c00faaec2375d4f2fe.
Local experimental overlay; not an upstream submission.
