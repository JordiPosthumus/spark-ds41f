# Prefix loop recovery correction

The previous temporary mask allowed "Let me write" and "Let me goo", and a severe-loop rewind cleared the mask. The retry detector also missed repeated "OK".

This correction counts every whitespace-separated "Let me" prefix for temporary intervention. During an intervention, the native sampler blocks whole case-insensitive "Let me", "OK", and the existing Hmm variants for **2,000 completed words**, or until a genuine reasoning/tool boundary. It remains request-local and inactive otherwise. Severe-loop retries start with this mask active and a fresh seed. The existing ten-retry and original output-budget limits remain unchanged.

An additional bounded detector recognizes eight consecutive copies of a one-to-eight-word prose sequence, independently of these keywords. Fenced code is excluded. Other repetitive phrases trigger a rewind, but the fixed temporary mask is not an arbitrary-phrase blocker. Neither heuristic establishes semantic failure; false positives and renewed loops remain possible.

The owner authorized Ring 1 first, then Ring 2 only after an hour of stable serving. Original images, kernels, context, concurrency, cache, sampling parameters, and memory guards are retained. A pause/drain must reach native idle before a coordinated restart. Rollback commits are recorded in each baseline.

Portable tests exercise prefix coverage, mixed case, token segmentation, whole-word negatives, exact 2,000-word expiry, short-cycle detection, fenced-code exclusions, cancellation, request isolation, and the actual API rewind/mask/seed sequence. These are separate from model recovery; live output validation is required before admission. Private session data is not committed.

The owner subsequently authorized the 2,000-word duration on both rings immediately. The earlier one-hour staging gate is superseded for this rollout. Detection thresholds and reasoning-boundary release are unchanged.
