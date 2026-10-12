# Permanent phrase blocking on Ring 1

Owner-requested policy: block the exact multiword phrases `Let me go` and `Let me write the command`, with original/lower/upper case variants, through native vLLM `bad_words`. No individual-word bans, loop detector, retries, temperature changes, output cap, or custom GPU sampler are enabled.

The rule applies to generated sampled chat text, including reasoning and answers. It cannot prevent every repetition pattern and intentionally changes which outputs are possible. Native constrained requests that cannot safely combine with `bad_words` pass through with a logged reason; caller sampling settings and native beam search remain unchanged. See the [payload and compatibility boundaries](../experiments/2026-10-12-phrase-blocking/README.md).

This branch selects Ring 1's captured `config/site-r6g.local.json` as normal `config/cluster.json`, adding two read-only API mounts and `SPARK_DS41F_PHRASE_BLOCKLIST=1`. All 113 existing effective environment entries and exact serve arguments remain unchanged; the enable switch makes 114 entries. All seven r6g/greedy overlay paths, normal caches, image tags, context, concurrency, DSpark, graphs, startup/steady memory guards and fan policy are preserved. Ring 1 does not adopt M64. Repository/branch and baseline/lock metadata are updated explicitly for this owner-fork branch.

The [immutable baseline record](../manifests/baselines/2026-10-12-ring1-phrase-only.json) retains both observed image identities. Rank 2's existing image ID differs from ranks 0/1, although image Config, RootFS layers, twelve relevant installed sources and seven overlays matched during preparation. Preserve each rank's recorded image/tag identity; do not pull, retag or normalize images. The selected copied lock changes only `generated_from_baseline`; source, patch and build identities are unchanged.

Before any coordinated deployment, drain Ring 1 and wait for native idle under the operator's hold. Keep timestamped private copies of its exact profile, ignored node map, Git config, container inspect/logs, seven overlays and selected lock. Publish and deploy one clean owner-fork commit through `bin/spark`; no live state was changed by the local preparation.

Rollback uses observed commit `9ff535dc68ddbf248f52c4eb41f4d6fccbbd9efa` plus its exact captured `config/site-r6g.local.json`, original Git origins, node map, seven overlays and per-rank images through coordinated lifecycle commands after drain and idle. The old tracked `config/cluster.json` differs from that live site profile and is not the rollback runtime. Preserve all model files and warm caches.

Live acceptance receipts belong in a separate record after deployment. This local change does not claim completed native validation, a full benchmark, improved throughput, or universal loop recovery. Ring 2's separate branch and historical M64 documents are not Ring 1 deployment instructions.
