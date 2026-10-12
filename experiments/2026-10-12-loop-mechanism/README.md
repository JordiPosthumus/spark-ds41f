# Ring 2 native loop-recovery mechanism canary

Unpromoted owner-fork experiment. This package tests whether the server can
abort a synthetic reasoning request and restart from an exact unpublished
token checkpoint while retaining the normal native parser and SSE path.
The detector decisions are deliberately **injected**. Passing the mechanism
test would not establish real-loop detection, answer quality, false-positive
rate, or lossless generation. No live results are included.

## One experimental variable

Compare the same accepted seven-overlay baseline with the temporary mechanism
hook disabled, then enabled for exactly three full synthetic request IDs.
Initial request bodies are identical except their fresh campaign identities.
Retry seeds deliberately change. Kernel candidates are excluded.

The three profiles are:

| Profile | Purpose | Serving intervention |
| --- | --- | --- |
| `baseline.json` | Experimental comparison control | None; seven accepted overlays at relocated source paths |
| `disabled.json` | Hook-overhead/native control | Immutable plan disabled; original native iterator |
| `enabled.json` | Mechanism canary | Exact three synthetic IDs only; one or two injected retries |

All three preserve the accepted image/tag/tree labels, 524,288 context, eight
sequences, 4,096 batched tokens, 3.5 GiB KV cache per rank, async scheduling,
DSpark, parser/tool settings, cache locations, no explicit four-image cap and
the complete host policy. Startup/steady memory guards remain 5/3 GiB and
`host.fan_control` remains false. No host, network, desktop or fan change is
part of this experiment. The 1,024-token budget is only in synthetic requests;
the normal server's output capacity is unchanged.

The accepted profile retains `promoted_baseline` of
`2026-10-05-karmic-kraken-r6` while selecting the real r6f lock at
`experiments/2026-10-11-r6f/upstreams.lock.json`. That lock's
`generated_from_baseline` matches the retained value, and its source manifest
describes the r6f source trees. Do not relabel this profile or replace its lock
with the repository's newer default. No published baseline or lock is modified.
The reference image ID captured on all three ranks is recorded in `base.json`;
it must be checked again on the live ring, independently of the tag.

## Exact source/configuration delta

`candidate.json` lists complete configuration deltas from the accepted profile.
The seven accepted overlay files are byte-identical and their targets/modes
unchanged. Their **source paths** move into this Git-tracked experiment so
coordinated Git sync is sufficient; no loose node-local file copying is needed.
Deployment metadata changes to the owner's fork, and the experiment omits the
production-only branch constraint. All other values remain unchanged.

Each instrumented arm adds exactly two read-only mounts: chat `serving.py`,
and its adjacent helper directory. Its source patch is saved as
`disabled-serving.patch` or `enabled-serving.patch`. The one non-beam native
`generate` call keeps every original positional argument and keyword. Native
rendering, parser, SSE, tool handling and HTTP cancellation code stay intact.
The actual server's engine and tokenizer are used when the hook is selected.

Runtime helper Python files match the independently reviewed offline kit.
Only local-origin path metadata was removed from the copied mechanism lock;
its functional dependency hashes are unchanged. Each immutable plan and its
embedded serving digest were recalculated after that metadata sanitization.
The package contains no private transcripts, captured native token arrays,
credentials or Mac workstation paths. It intentionally retains operational
Linux configuration, including `host.home=/home/jordi`, head address
`192.168.100.245` and the accepted original `/home/jordi/spark-lab/` mount
provenance. Site addresses/paths are configuration, not secret-free proof for
arbitrary future additions. Synthetic image data is generated, not user images.

## Selection and evidence

The disabled and enabled stages use distinct 12-hex campaign identities,
listed in `base.json`. Each plan allows exactly three full serving IDs with
suffixes `00`, `01`, `02`; matching substrings, wildcards and suffixed
subrequest IDs do not select. Header and body request IDs agree. IDs are test
routing, not authentication: run only on an isolated drained server.

Case `00` attempts one native retry from the first eligible unpublished chunk.
Case `01` attempts two retries while preserving a nonempty published prefix
and all six original image objects/placeholders. Case `02` attempts one retry
for a separate cancellation/disconnect check. Natural completion, tool/answer
boundaries, oversized coalesced chunks or excluded settings may make a case
inconclusive. Do not weaken exclusions or force outputs to get a pass.

Before each selected request, the hook checks the helper bytes and 26 native
source files, including both inherited request-ID sources and all seven
baseline overlays. A mismatch, disabled plan, unsupported request or missing
helper preserves native generation with a diagnostic reason. The default
non-canary path returns the original iterator without a wrapper task.

JSON audit lines use prefix `DS41F_MECHANISM_AUDIT `. Every selected/lifecycle
record carries the full ID, run, plan hash, process ID and local monotonic time.
`mechanism-wiring` records selection/bypass; `mechanism-recover-injected`
records the actual mechanism lifecycle. There is no raw prompt/reasoning/token
logging. The lifecycle includes original/retained/published prefix checks,
native chunk hashes, MM object identity, sampler/budget equality, abort
completion and native restarted output. An injected decision alone is not a
successful retry. Lost audit lines make the test inconclusive.

Use separate logs per campaign/stage. Feed only the mechanism-mode records
from one stage to `tools/analyze.py`, preserving all IDs within that stage so
its foreign/interleaved-lifecycle checks remain effective. Never combine short
case numbers across campaigns. No more than one lifecycle per exact ID may be
claimed; reruns need a newly built/published immutable plan with fresh IDs.

## Acceptance and limits

Required native evidence: actual tokenizer/token/chunk alignment; retained
public prefix and unchanged MM objects; real native abort/new request and
retry output; native parser/SSE/Unicode/tool behavior; API plus scheduler idle
after both repeated Python cancellation and actual HTTP disconnect; and
client TTFT/inter-chunk gaps/TPS plus observed engine work and memory.

The six-image case must show six placeholders in every native attempt.
Synthetic output is stochastic and may never produce a tool call; missing
tool output leaves that native-path check unqualified. Source-file identity
does not prove loaded runtime objects. Inspect the newly started process and
effective configuration before measuring.

Retries change randomness and add computation. Observed token work is only a
lower bound because aborted in-flight work is unknown. The small synthetic
holdbacks do not qualify the real detector's latency/coverage. This experiment
has no output-quality or speed claim and does not promote a permanent server
feature. Follow `RUNBOOK.md`; retain failures and restore the exact original
Git metadata, accepted HEAD, profile/selector and seven original mount sources
before normal gateway readmission. `baseline.json` is only the experimental
control: it retains owner-fork metadata and relocated mount sources. The runbook
requires private per-rank backups and separate original/owner controllers for
the stopped-state origin transition and coordinated sync. If a newer baseline
has been accepted before the wave, reconcile first; do not roll it back silently.
