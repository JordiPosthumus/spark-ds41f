# Ring 2 M64 production trial

The owner accepted the tested M64 routed-expert prefill change on Ring 2. Ring 1 stays on its current recipe for a real-traffic comparison. The normal Ring 2 launcher is bin/spark cluster start --apply, using config/cluster.json on the published ring2-m64-production branch.

This preserves the existing Ring 2 image, seven overlays (only the routed-expert adapter source changes), 524288 context, eight sequences, 4096 batch tokens, uncapped images, sampling, precision, DSpark, graphs, async scheduling, all 113 effective environment values, ordinary caches and memory guards. No loop-recovery code is enabled. Six output fingerprints matched all 128 native output IDs each. Four image/text requests passed, including 13 images and measured cold-to-warm cache reuse.

The qualified warm-cache startup completed in 106 seconds, with no logged TileLang compilations. An earlier cold compilation crossed the existing startup memory guard; cold-cache rebuild reliability is still unqualified. Preserve the current caches and the original rollback assets. Prior experiments showed a modest prefill improvement; sustained production throughput and stability remain a live comparison, not a promised percentage.

The immutable baseline record is manifests/baselines/2026-10-12-ring2-m64-warm.json. The inherited benchmark suite has not been relabelled as a new complete benchmark. The source lock copy changes only generated_from_baseline.

Rollback: drain Ring 2 and wait for active requests to finish; stop the three ranks with the coordinated launcher. Restore the timestamped original Git metadata and accepted 9ff535dc68ddbf248f52c4eb41f4d6fccbbd9efa checkout, then start config/site-ring2-baseline.local.json. Verified per-node backups: /home/jordi/spark-lab/2026-10-12-ring2-baseline/capacity-backup-canary-a7-20261012T213908000000Z. Retain the original seven overlay files and image. No Ring 1 action is required.
