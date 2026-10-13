# Loop protection removed

The owner requested complete removal of the experimental loop protection.
The normal launcher no longer mounts its API wrapper, detector, retry machinery, notices, word tables, or native sampling changes. The pre-protection model runner overlay is restored. Native vLLM bad_words remains available only when a client explicitly requests it; the recipe supplies no automatic blocklist.

The model image and weights, image support, capacity, sampling defaults, DSpark, cache settings, memory guards and unrelated fixes remain unchanged. Ring 2 retains its M64 prefill optimization.

Historical experiments are retained for audit and are not loaded by the normal launcher. No claim is made that the underlying model will stop looping.
