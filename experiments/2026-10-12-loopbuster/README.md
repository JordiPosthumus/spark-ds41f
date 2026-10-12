# Loopbuster auto recovery — candidate

Owner-requested reliability intervention, not lossless sampling. Default module mode is off. `auto-recover` immediately streams ordinary reasoning; on sustained repetition it aborts the native request and regenerates from an earlier exact native token checkpoint, with at most ten fresh-seed retries. Reasoning already shown stays visible but is omitted from the restarted model context. No hidden instruction is injected. No answer/tool output is rewound: their markers permanently disarm recovery. MM objects and original rendered prompt are reused. Existing unsupported request options pass through unchanged with a reason, including FINAL_ONLY, greedy sampling, penalties, custom stops/processors and reasoning budgets. This is not universal coverage.

The native API parser remains downstream with one lifetime. Original request max_tokens is shared across all OBSERVED attempt output tokens; retries receive only the remainder. API usage counts emitted tokens; telemetry separately reports observed attempted output, attempted prompt work and excluded model-context tokens. Unobserved tokens in flight at abort remain unknown. There is no artificial holdback delay; CPU overhead/native parser behavior still require the canary.

Latest recorded failure confirms at 8,533 characters. Four actual-tokenizer reconstructed chunk replays produce one abort, a second engine call, and a synthetic successful retry. These are CPU transport doubles, not GPU recovery proof. Five selected productive controls remain unchanged, including the 147K-character case. The older 54,870-character failure is a known miss. Policy was selected after viewing these controls; false-positive rate is uncalibrated.

## Exact deployment delta

See delta.json and source-pins.json. Base is owner production commit c50c9cf6b505cc46d46dc02605bdf75ae6505b41. Each instrumented profile adds three read-only mounts (API serving, helper package, native bad_words expiry) and one mode env to its exact profile; all 86 explicitly configured env entries, 67 argv entries, ten original mounts, image, locks, capacity, host policy and M64 improvement remain identical. The operator selects the publication branch/ref explicitly; these profiles retain original deployment metadata to make the runtime delta obvious. This directory is the entire copy list. No private traces, tokenizer file, credentials or Mac paths are included.

## Native qualification (root operator only)

Use the established reservation, drain-to-idle, timestamped Git/profile backup and coordinated Git sync/start/stop procedure. Do not copy loose files to nodes. Verify source-pins against the actual original installed files before mounting the overlay. Restore original-m64.json if validation fails. A start alone is not qualification.

First start profiles/canary.json. It uses exactly the same auto-recovery core as production, but includes a separate TEST-ONLY callback for the three full IDs in synthetic-fixtures.json. It injects a decision after at least eight already-published actual native reasoning tokens, rewinds model context to the original rendered prompt, and audits the real native abort and fresh generate calls. This proves mechanics only, never real-loop detection. Every audit row keeps full identity/run/mode; use stage-specific logs. Reusing an ID in the same API process is refused. Unselected requests use the real detector.

From the experiment directory on the isolated head (example loopback origin):

```sh
curl --no-buffer --fail-with-body -H 'Content-Type: application/json' --data-binary @requests/mechanism-00.json http://127.0.0.1:8000/v1/chat/completions > mechanism-00.sse
curl --no-buffer --fail-with-body -H 'Content-Type: application/json' --data-binary @requests/mechanism-01.json http://127.0.0.1:8000/v1/chat/completions > mechanism-01.sse
curl --no-buffer --fail-with-body --max-time 8 -H 'Content-Type: application/json' --data-binary @requests/mechanism-02.json http://127.0.0.1:8000/v1/chat/completions > mechanism-02.sse
curl --no-buffer --fail-with-body -H 'Content-Type: application/json' --data-binary @requests/tool.json http://127.0.0.1:8000/v1/chat/completions > tool.sse
curl --no-buffer --fail-with-body -H 'Content-Type: application/json' --data-binary @requests/normal.json http://127.0.0.1:8000/v1/chat/completions > normal.sse
```

Do not execute returned tools. Case01 contains six images. For00 require one injected decision and two native attempts;01 two decisions/three attempts. Each retry must show exact_original_prompt, multimodal_identity_preserved, kwargs_preserved, budget_correct, sampler_preserved, fresh_retry_seed, previous_abort_and_close_complete all true. Native token hash/count rows are actual generated chunks, not reconstructed fixtures. Verify valid SSE and final usage/content/tool structure. Do not accept merely HTTP200.

For02 verify logs show a SECOND native request before the intentional HTTP timeout, then its native abort/iterator close. If model already completed or retry was not reached, the cancellation test is inconclusive. Python double cancellation is covered by CPU tests, distinct from this HTTP disconnect. Sample fresh scheduler running/waiting=0 and a finite stable generation-token counter after cleanup on the same clock, and check the following normal request succeeds. Do not count cached/historical idle data as evidence.

Then start profiles/auto-recover.json. Its helper directory has NO canary.py; serving imports integration.py and uses only the real detector. Verify loaded file hashes, mode, absence of canary callback, effective M64/runtime settings, a normal streamed request, tool response and health. Only then readmit. Save actual canary logs/receipts in a new operator run directory. No broad benchmark or stochastic reproduction of a 290K-character loop is necessary for this mechanics check.

Measured driver (plan-only unless --execute), from this directory:

```sh
python3 -B native_driver.py --case mechanism-00 --execute --out /tmp/loopbuster-mechanism-00.json
python3 -B native_driver.py --case mechanism-01 --execute --out /tmp/loopbuster-mechanism-01.json
python3 -B native_driver.py --case mechanism-02 --execute --out /tmp/loopbuster-mechanism-02.json
python3 -B native_driver.py --case tool --execute --out /tmp/loopbuster-tool.json
python3 -B native_driver.py --case normal --execute --out /tmp/loopbuster-normal.json
python3 -B native_analyze.py --log /tmp/loopbuster-canary-api.log --result /tmp/loopbuster-mechanism-00.json
python3 -B native_analyze.py --log /tmp/loopbuster-canary-api.log --result /tmp/loopbuster-mechanism-01.json
python3 -B native_analyze.py --log /tmp/loopbuster-canary-api.log --result /tmp/loopbuster-mechanism-02.json
```

Save timestamp-scoped actual API logs with the existing operator. Driver requires idle, records fresh post-cleanup metrics, valid SSE/usage/tool JSON and elapsed/first-output timings. Case02 closes HTTP after six seconds of output; analyzer requires the second native request to have been aborted/closed. Early completion makes this inconclusive. Use new result filenames on retries, and a new API process for reused synthetic identities.

## Temporary ordinary-word mask

On a confirmed loop, the retry temporarily suppresses up to16 unique ordinary ASCII words drawn from repeated trigrams in a bounded600-word in-memory tail. Lower/title/upper forms are tokenized by the native tokenizer; at most32 forms/256 candidate token entries. No raw words are logged. Only retries get bad_words and a48-generated-token expiry (approximately30 words, not exact). A native GPU mask expires independently per speculative/bonus row; </think> or DSML control tokens terminate masking earlier, preserving final answers/tool names/arguments. Ordinary caller bad_words remain permanent; empty slots reset expiry. allowed_token_ids requests keep their restriction and use seed-only recovery with an explicit reason. Native validation failures likewise preserve seed-only recovery; they are not success evidence for masking.

The canary hook explicitly injects ordinary-word candidates for its exact three IDs so retries exercise masking even without stochastic repetition. Every retry audit must have temporary_words_active=true, temporary_expiry=48, temporary_stop_ids_valid=true. For the six-image case require image_placeholders=6 for every attempt. Production has no injected candidates: it extracts from actual confirmed repeated reasoning only.

Native GPU unit test (root operator, inside the canary container after checking its loaded path; test does not load a model or send inference):

```sh
python3 -B /home/jordi/spark-ds41f/experiments/2026-10-12-loopbuster/native_ttl_gpu_test.py --expected-source-sha WORKER_HASH --execute --out /tmp/loopbuster-native-ttl.json
```

If the repository is not visible inside the container, execute the committed script through stdin using the established operator. Replace WORKER_HASH with worker-overlay/vllm/v1/worker/gpu/sample/bad_words.py in publication-manifest.json. The runner checks the actually imported module hash before testing native Triton expiry47/48/49, speculative/bonus positions, committed/draft control boundaries, mixed permanent+temporary requests, unchanged direct API and slot reuse. It allocates tiny test tensors, but CUDA process initialization itself has overhead; root owns when to run it. GPU compilation/execution is still required: CPU AST checks are not a substitute.
