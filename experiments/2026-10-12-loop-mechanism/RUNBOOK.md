# Coordinated canary wave plan — not executed

The runtime owner controls the wave. Ring 1 keeps serving. These are explicit
future steps, not authorization to start/stop/sync now. No upstream PR is part
of this owner-fork experiment. A future publication is a reviewed commit to
the owner's fork, followed by the repository's coordinated commands.

## Publish and record the original state

1. Review the publishable-file manifest and exact source/config deltas. Apply
   only this experiment directory to a clean owner-fork checkout. Confirm no
   production config, upstream lock, baseline manifest or unrelated file changed.
   The package base commit is in `base.json`; if the fork advanced, re-run the
   original local doctor and byte/config checks against that exact new base.
2. Publish one reviewed commit on an owner-fork branch; no upstream PR. Record
   its full SHA. Never copy a dirty worktree, use rsync or copy private staging
   receipts into Git. The node-local ignored site map stays outside the payload.
3. Before any GPU/cluster operation, coordinate the existing hold/window on
   Ring 2 and refresh its heartbeat as required. Do not take another holder's
   ring. Drain new gateway arrivals and wait for active work and queues to be
   idle. Confirm the site map contains only Ring 2's three nodes and correct head.
4. Before any mutation, make timestamped private backups for **each rank** of
   its actual Git config file(s), exact HEAD and detached/branch state, effective
   origin fetch and push URL arrays, selected ignored site map(s), original
   profile bytes/path and normal launcher/profile selector. Resolve the actual
   common Git directory; do not assume `.git` is a directory. Include worktree
   config and relevant include provenance if present. Preserve config bytes,
   hashes, ownership and modes; do not publish credentials that might be in Git
   URLs/config. Back up any controller metadata that will be changed too.
5. Bind that receipt to a fresh effective configuration inventory: all three
   image IDs and labels, full per-rank argv/environment, mounts, resources,
   original seven overlay hashes/paths, host policy, cache and cost-file backing,
   container identities and observation time. Retain original overlays, images,
   caches, cost files and working launchers throughout the experiment.

Use the existing private receipt convention `ring2-original-git-metadata-v1`:
`ranks` contains Spark/Spark2/Spark3, each with `head`, `backed_up_at_utc`,
`config_backup_path`, `config_sha256`, `origin_fetch_urls` and
`origin_push_urls`. Preserve additional worktree config and profile/map/selector
receipts alongside it. Backup paths stay in the private lab area. This package
contains no fabricated backup receipt or assertion that a future backup exists.

The accepted original checkout is `9ff535dc68ddbf248f52c4eb41f4d6fccbbd9efa`;
its original profile SHA256 is
`9309ad55b33bb884096c7df70eb6c2de450aeb500744a1bfe5c010b466aa5073`,
with origin `https://github.com/christopherowen/spark-ds41f.git` and the seven
original mount sources below
`/home/jordi/spark-lab/2026-10-12-ring2-baseline/baseline-overlay/overlay/greedy/vllm/`.
Use the **actual freshly recorded** original profile path and selector, not an
assumed default. Fresh state must match these accepted identities. If another
wave has promoted a new baseline before this wave, stop and reconcile the
package with the runtime owner; do not overwrite that newer accepted state
with this historical baseline.

## Prepare separate controllers, then stop and change origins

Prepare a clean **isolated head-side owner-fork controller checkout** at the
published full canary commit, with owner origin verified as
`https://github.com/JordiPosthumus/spark-ds41f.git`. It must already contain all
tracked experiment profiles and be reachable from a published origin branch.
Populate only its required ignored site map(s) from the exact saved Ring 2 map.
Also retain a separate clean **original rollback controller** at the accepted
original commit, original origin, original profile and saved ignored map(s).
Keep this rollback controller untouched by the experiment sync. Check both
controllers' profile/map paths and deployment targets before proceeding.

This separation is required: `cluster sync` loads its profile before checkout,
and rejects any node whose current origin differs from `deployment.repository`.
The old head checkout cannot resolve a new tracked experiment profile. Merely
passing an owner-fork profile does not switch origins. Do not bootstrap by
copying tracked payload files into the old checkout or by bypassing the guard.

Stop the original service through the **original controller and original
profile**, while idle, before changing any rank's origin. Inspect the plan,
then let only the runtime owner apply it:

```sh
bin/spark --cluster-config "$ORIGINAL_PROFILE" cluster stop
bin/spark --cluster-config "$ORIGINAL_PROFILE" cluster stop --apply
```

Verify all three ranks stopped. Recheck clean checkouts, HEADs, saved Git config
hashes and effective URL arrays before making the narrowly scoped origin change.
While stopped, change only repository-local `remote.origin.url` and
`remote.origin.pushurl` to the owner URL on all three ranks. Replace all existing
values for those two keys rather than leaving an extra fetch/push URL behind;
preserve every unrelated Git config key. Verify the resulting effective fetch
and push arrays each resolve to exactly the owner URL, including Git URL rewrite
rules, and record that the config delta is limited to those two keys. If shared
Git config/includes or URL rewrites prevent this exact scope, stop and reconcile;
do not change global Git settings or relax the deployment guard.

## Sync from the published owner controller

Run the following from the isolated owner-fork controller, **not** the old head
checkout or rollback controller. `CANARY_COMMIT` is the published full SHA;
`EXP` is a repository path, not an environment control for serving. Inspect all
selected nodes/paths in these plans before applying:

```sh
EXP=experiments/2026-10-12-loop-mechanism
bin/spark --cluster-config "$EXP/baseline.json" cluster sync --revision "$CANARY_COMMIT"
bin/spark --cluster-config "$EXP/disabled.json" cluster start --replace
```

After the stopped-state and origin checks above:

```sh
bin/spark --cluster-config "$EXP/baseline.json" cluster sync --revision "$CANARY_COMMIT" --apply
bin/spark --cluster-config "$EXP/disabled.json" cluster start --replace --apply
```

Require all ranks clean and detached at that exact published commit before
startup, with the shared accepted image ID/tree labels and saved site map(s).
Sync does not make untracked host files authoritative. All nine Python/helper
mount sources now reside at the published experiment paths. `start` must keep
the native guards and all-rank preflight. If any sync/start step only partially
succeeds, keep the ring drained and restore consistently; do not start mixed
ranks. If a live doctor/host mismatch blocks the operation, preserve it as a
failed/prevented run and resolve it with the owner. Do not use a blanket mismatch
override or change desktop, fan, network, kernel or memory policy to pass.

## Disabled then enabled

Inside each newly started intended container, run the read-only source check
using the installed root and the matching stage manifest:

```sh
python3 -B "$EXP/tools/preflight.py" --installed-root /opt/spark3/candidate/vllm/vllm --manifest "$EXP/disabled-manifest.json" --mounted
```

The command assumes the Git checkout is accessible from the chosen execution
context. If it is not mounted in the container, feed this exact published
stdlib verifier and manifest through the existing controlled exec mechanism;
do not add a writable runtime mount or broaden the helper's allowlist.
Verify all three nodes, effective settings, loaded module paths and the normal
native parser/tokenizer objects. A file hash alone does not prove loaded code.

Send only the three bodies from `fixtures/disabled.json`, sequentially, with
their matching `X-Request-Id` headers through the ordinary chat SSE API. Record
client timing/usage/finish/tool data and server counters, preserving failed
runs. Do not change model settings to make the fixtures faster. No request
sender or endpoint is automatically invoked by this package.

At idle, switch all ranks together and validate the enabled stage:

```sh
bin/spark --cluster-config "$EXP/disabled.json" cluster stop --apply
bin/spark --cluster-config "$EXP/enabled.json" cluster start --replace --apply
python3 -B "$EXP/tools/preflight.py" --installed-root /opt/spark3/candidate/vllm/vllm --manifest "$EXP/enabled-manifest.json" --mounted
```

Use `fixtures/enabled.json` and its fresh identities. Require exact native
attempt, prefix, MM, sampler and budget evidence. One retry in `00` and two in
`01` must actually start and emit native token output; a forced decision alone
is insufficient. Collect memory/swap, queue and error evidence throughout.

## Two different cancellation checks

An actual HTTP disconnect and repeated Python task cancellation are separate
stages. Do not claim one proves the other or reuse one ID in a process.

For HTTP, keep case `02`'s owned socket open until the server audit contains
attempt-2 `native_chunk` with a nonempty token range, then close that socket.
An elapsed delay, initial decision or first public SSE byte is not the trigger.
Require native abort completion, zero API unfinished requests and fresh
scheduler metrics: at least six increasing observations spanning ten seconds,
zero running/waiting work and a stable finite generation counter.

For Python double-cancel, use a separately built/published immutable campaign
with fresh full IDs. In the actual server process, obtain its installed
`_ds41f_mechanism_generate` gate and the rendered synthetic input/parameters.
Call the gate with the original kwargs, obtain `gate.evidence_for(server, id)`,
and explicitly run `tools/cancel_after_restart.exercise` on that iterator and
real engine. A missing evidence object means bypass/inconclusive. This helper
waits for restarted native output, cancels its owned consumer twice and retains
unfinished cleanup. It adds no public control endpoint and is not run by normal
requests. Its fresh idle probe must use the same process's monotonic clock;
cached/historical samples, infinite values and missing counters are not proof.

This second in-process campaign is still a qualification step, not included
as a hidden mode of the present three-ID enabled plan. Do not fake a server,
tokenizer or scheduler and label the result native.

## Experimental control is not original restoration

An optional post-canary comparison using `baseline.json` removes the two
mechanism mounts but still uses owner-fork Git state and the **relocated** seven
mount source paths. It is an experimental control, not the original launcher
state, and does not permit gateway readmission. If that comparison is needed,
stop the active arm at idle and start `baseline.json` through the owner
controller; perform the same current-window validation and keep Ring 2 drained.

## Restore the exact original state, then resume use

After synthetic work drains and cleanup is proven, stop whichever experimental
profile is actually active through the owner controller. `ACTIVE_PROFILE` below
must identify that arm; it is not assumed to still be `enabled.json`:

```sh
bin/spark --cluster-config "$ACTIVE_PROFILE" cluster stop --apply
```

Verify all ranks stopped. Restore each rank's exact saved Git config bytes,
permissions and effective origin fetch/push arrays **before** synchronizing from
the original authority. Check against that rank's private backup receipt; do not
substitute a guessed URL or discard unrelated config. Restore/verify the exact
saved ignored site map(s), original profile bytes and normal selector. Keep the
original seven source overlays at their original paths with unchanged hashes.

From the retained **original rollback controller**, use its original profile
and map(s), after checking its original origin and accepted HEAD. Inspect plans,
then apply the coordinated restoration:

```sh
bin/spark --cluster-config "$ORIGINAL_PROFILE" cluster sync --revision "$ORIGINAL_COMMIT"
bin/spark --cluster-config "$ORIGINAL_PROFILE" cluster start --replace
bin/spark --cluster-config "$ORIGINAL_PROFILE" cluster sync --revision "$ORIGINAL_COMMIT" --apply
bin/spark --cluster-config "$ORIGINAL_PROFILE" cluster start --replace --apply
```

`ORIGINAL_COMMIT` comes from the verified receipt and must be the accepted
`9ff535dc68ddbf248f52c4eb41f4d6fccbbd9efa` for this package. The coordinated sync
detaches all ranks. Its expected original state here is already detached; if a
fresh backup instead shows a branch or different HEAD, reconcile that mismatch
before the wave, not by silently rewriting refs during restoration.

Require fresh restoration evidence bound to this service window and the newly
started containers: original HEAD and detached state, exact Git config hashes
and effective URL arrays, saved profile/map/selector bytes, image IDs/labels,
full per-rank argv/environment, resources, cache/cost backing and host policy.
Compare all seven original mount **sources, targets, modes and hashes**; the
two mechanism mounts must be absent and native chat source restored. Checking
only overlay bytes at the relocated experiment paths is insufficient.

The experiment files may no longer exist in the restored node checkout. Use
the retained owner controller's exact published stdlib preflight and manifest
through the existing read-only controlled exec mechanism if needed; do not
assume `$EXP` still exists there or add a mount. Omit `--mounted` for this native
source check. Verify loaded runtime paths/settings as well as file hashes.

Validate current health, a fresh native inference, a real cold-to-warm cache
hit and fresh API/scheduler idle evidence. Historical canary/capacity receipts
cannot qualify this restoration, and a parsed profile or started process alone
is not enough. Only then readmit Ring 2 to the gateway and begin the agreed forty-minute
live-use window. Keep the runtime owner's reservation and refresh its heartbeat
through that window and the following experiment cycle; the existing monitor
requires the same holder. Release the hold only when the runtime owner explicitly
ends its reservation after verified restoration. Preserve failed/inconclusive
results. If restoration cannot be verified, keep the ring drained, retain the
hold and report the specific blocker; do not erase caches or reduce capacity
as a fallback. Production recovery remains a separate owner decision and
requires real-detector/quality qualification.
