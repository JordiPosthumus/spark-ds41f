# Token-aware temporary phrase suppression

Candidate only. No deployment or successful loop rescue is claimed by this record.

The previous temporary native `bad_words` implementation missed identical visible
phrases encoded with alternative token splits. For example, the tokenizer emits
`Hmm.` as `[42, 6002, 16]`, while `Hmm.\n\n` is `[42, 6002, 339]`.
The canonical token list did not cover the second route. Activation logs alone
therefore did not establish suppression.

This candidate retains the existing concentration trigger and severe-loop retry
policy, and replaces temporary canonical-token bans with a compiled byte matcher
inside the pinned V2 sampler. It matches ASCII casing variants of whole `hmm`
through `h` plus sixteen `m` characters, and whole `let me run` / `let me go`
with whitespace separators. It retains lexical and incomplete UTF-8 state from
trusted generated reasoning across the continuation boundary.

The sampler applies the restriction to each target verification position, using
only accepted history plus that position's transient draft prefix. Rejected drafts
never enter persistent state. Restriction ends after sixty completed Unicode
words. Genuine EOS, reasoning-end and DSML control tokens are always allowed and
disarm the restriction; a pending target directly before such a token is exempt.
This is a narrow phrase intervention, not a guarantee of semantic recovery.

The table compiler is pinned to the serving tokenizer hash and Unicode 15.0.0.
Its factored tables occupy about 6.17 MB. No ordinary component-word ban is added.
Normal requests take the inactive sampler path; actual GPU and serving overhead
must be measured before any performance claim.

CPU checks cover every vocabulary entry against native tokenizer decoding,
Unicode scalar classification, mixed casing, alternate token splits, UTF-8
fragments, request-local state, exact expiry, and API cancellation/continuation.
See `ACCEPTANCE.txt` for remaining GPU, output, and staged serving gates.

## Reproduce CPU checks

Set `SPARK_TEST_EXPERIMENT` to this directory and `SPARK_TEST_TOKENIZER` to the
pinned model's tokenizer.json. Run `python3 compiler/test_compiler.py` and
`python3 tests/test_api_semantics.py` with NumPy and tokenizers installed.
Rebuild tables with `python3 compiler/compiler.py TOKENIZER OUT
--unicode-policy compiler/target-unicode.json`.

## Rollout boundary

Ring 1 first, after drain and native idle. Preserve image, model, GPU kernels,
DSpark, context, output, concurrency, image support, cache and memory guards.
Validate actual protected output and ordinary tools/cancellation before admission.
Require one hour of stable serving before Ring 2, retaining its M64 kernel changes.
Any failed gate keeps Ring 2 unchanged and uses the recorded prior commit for
coordinated rollback. Test fixtures are synthetic; no private sessions belong here.
