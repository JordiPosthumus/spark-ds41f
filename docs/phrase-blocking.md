# Permanent phrase blocking

Owner-requested policy: block the exact multiword phrases `Let me go` and `Let me write the command`, with original/lower/upper case variants, through native vLLM bad_words. No individual-word bans, loop detector, retries, temperature changes, output cap, or custom GPU sampler are enabled.

The rule applies to generated chat text, including reasoning and answers. It cannot prevent every possible repetition pattern. Native constrained requests that cannot safely combine with bad_words pass through with a logged reason; caller sampling settings are preserved.

Ring 2 retains its accepted M64 setup. The two added read-only API mounts are the only serving delta. Roll back by using the previous accepted M64 commit c50c9cf6b505cc46d46dc02605bdf75ae6505b41 through the coordinated launcher after draining and waiting for idle. Preserve all model files and warm caches.

Live acceptance receipts are recorded separately after boot; this document does not claim a completed benchmark or universal loop recovery.
