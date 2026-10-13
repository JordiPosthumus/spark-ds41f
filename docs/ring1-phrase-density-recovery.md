# Ring 1 phrase-density recovery rollout

Apply the exact API recovery package qualified on Ring 2. The production change
replaces permanent phrase blocking with sustained case-insensitive phrase-density
detection and up to ten fresh-seed retries within the observed output budget.
No diagnostic injection hook is present.

Ring 1 keeps its own topology, seven existing GPU overlays, image identities,
524288 context, eight sequences, 4096 batch tokens, uncapped images, caches,
DSpark, sampling, precision, graphs, async scheduling and memory guards.
Only the two API mounts and the phrase-blocking/recovery flag change.
Ring 2 keeps its accepted M64 kernel. This API rollout does not add unqualified
cold-cache M64 compilation to Ring 1; the separate speed rollout remains tracked.

Gate: at least 30 minutes of healthy meaningful real Ring 2 traffic after
2026-10-13T00:00:04.872Z. Require completed real requests, generated tokens,
no new gateway inference/transport failures, no engine restart, and unchanged
serving capability and healthy native runtime. Time alone is insufficient.
Drain Ring 1 new arrivals and wait for both gateway and native engine idle.
Take fresh timestamped backups before coordinated all-rank sync/start.

After boot, verify exact per-node image, serving settings, API source hashes,
memory guards and clean published commits. Exercise native streaming/tools,
13-image acceptance, actual prefix-cache reuse and overlapping requests before
readmission. Retain all failure receipts. No new throughput percentage is claimed.
The complete inherited performance benchmark is not relabelled or claimed rerun.

Rollback uses the coordinated launcher to restore
df21e337199541bf37993f31ecc15cf51bcf6518 and its config/cluster.json.
Each rank's current image is retained, including Spark5's previously qualified
image ID variation. Never retag, rebuild or prune working images/caches.
Recovery limitations are in docs/phrase-density-recovery.md.
