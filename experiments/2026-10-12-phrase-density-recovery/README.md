# Phrase-density recovery canary

Base: cf76d70c557ba1cb865c73b21cbee774303cf2c3.

Replace the recipe phrase bans with case-insensitive sustained phrase-density
detection and at most ten fresh-seed retries within the original output budget.
Normal launcher is unchanged. This profile is an isolated Ring 2 canary.
All unrelated serving arguments, host guards, image, M64, context, concurrency,
image handling and kernel settings are inherited unchanged.

The canary API seam injects a detection only for five exact synthetic request
IDs. It tests abort/restart plumbing, not detector accuracy. The first four native
checkpoints are held; the fifth explicitly validates rewinding model context
after reasoning has already streamed. Displayed reasoning cannot be retracted. Production recovery does not
hold normal output; only suspected loops are buffered.

Acceptance: real native retries, fresh seeds, preserved sampler and multimodal
identity, ordered abort/close/restart, correct usage, valid tools, and cancellation
cleanup. Remove injection before promotion. Restore the previous committed
normal profile if any native invariant fails. No successful recovery claimed
from CPU tests alone.
