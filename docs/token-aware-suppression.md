# Staged token-aware suppression trial

This Ring2 candidate carries the exact token-aware guard tested on Ring1.
Deployment remains conditional on a complete healthy hour of meaningful Ring1
serving and final review. The candidate retains Ring2's accepted M64 prefill
improvement and every unrelated setting, including native model precision,
DSpark, asynchronous scheduling, caches, uncapped images and request capacity.

The guard fixes tokenization escapes in temporary phrase blocking and applies
only to the request that triggered it. It expires at 60 completed words or a
reasoning/tool boundary. The native source patch, source hashes and isolated
CUDA results are in [the experiment](../experiments/2026-10-13-token-aware-suppression/README.md).
Ring1 live checks additionally verified actual blocked output, exact expiry,
concurrent request isolation, valid tool calls and cold-to-warm prefix reuse.
These checks do not establish natural loop rescue or an end-to-end speed gain.

All source/table mounts are read-only. Only the table-path environment variable
is added. The model-runner overlay preserves its existing greedy verification
with one request-cleanup hook; M64 and all other kernel overlays stay intact.

Rollback is coordinated deployment of previous Ring2 commit
3d50f9499a28f118cf0486c82fe2bd58a108401b, after drain and native idle.
No model or cache deletion is part of this change.
