"""Offline, default-off server-side recovery prototype for a vLLM engine client.

Wraps the pre-parser RequestOutput iterator; never edits live KV or worker state.
Only vanilla single-choice, unpenalized reasoning is initially eligible. Every
unsupported option falls through unchanged. No live integration is installed.
"""
import asyncio
from collections import deque
import copy
from dataclasses import dataclass
import secrets
import sys

try:
    from .collapse import CollapseDetector
    from .suspicion import HoldbackPolicy, WeakSuspicion
except ImportError:  # Standalone offline test invocation.
    from collapse import CollapseDetector
    from suspicion import HoldbackPolicy, WeakSuspicion


@dataclass(frozen=True)
class RecoveryConfig:
    mode: str = "off"  # off / observe / recover; no instruction injection
    max_retries: int = 2
    holdback_tokens: int = 2048
    holdback_bytes: int = 32768
    holdback_chunks: int = 4096
    adaptive: bool = False
    normal_holdback_tokens: int = 128
    release_misses: int = 2

    def __post_init__(self):
        if self.mode not in {"off", "observe", "recover"}:
            raise ValueError("mode")
        if type(self.adaptive) is not bool:
            raise ValueError("adaptive")
        for field, low, high in (("max_retries", 0, 2), ("holdback_tokens", 1, 8192),
                                 ("holdback_bytes", 256, 262144),
                                 ("holdback_chunks", 1, 16384)):
            if type(getattr(self, field)) is not int or not low <= getattr(self, field) <= high:
                raise ValueError(field)
        if type(self.normal_holdback_tokens) is not int or not 1 <= self.normal_holdback_tokens <= 8192:
            raise ValueError("normal_holdback_tokens")
        if self.adaptive and self.normal_holdback_tokens > self.holdback_tokens:
            raise ValueError("adaptive normal tail exceeds maximum")
        if type(self.release_misses) is not int or not 2 <= self.release_misses <= 8:
            raise ValueError("release_misses")


def eligibility(prompt, params, kwargs):
    """Reasons are aggregate labels; never print prompts, tools, or tokens."""
    if not isinstance(prompt, dict) or prompt.get("type") not in {"token", "multimodal"}:
        return "unsupported_input_type"
    if "encoder_prompt_token_ids" in prompt:
        return "encoder_decoder_input"
    if not isinstance(prompt.get("prompt_token_ids"), list):
        return "missing_rendered_token_ids"
    if kwargs.get("reasoning_ended") is not False:
        return "not_confirmed_reasoning"
    if kwargs.get("session_id") is not None:
        return "stateful_session"
    if getattr(params, "n", 1) != 1:
        return "multiple_choices"
    if not isinstance(params.max_tokens, int) or params.max_tokens < 1:
        return "unresolved_output_budget"
    # A resumed output becomes prompt. These options have per-generation state
    # whose semantics cannot be preserved by this wrapper alone.
    for name in ("presence_penalty", "frequency_penalty", "min_tokens"):
        if getattr(params, name, 0) != 0:
            return "stateful_" + name
    if getattr(params, "repetition_penalty", 1.) != 1.:
        return "stateful_repetition_penalty"
    for name in ("stop", "bad_words", "structured_outputs",
                 "thinking_token_budget", "repetition_detection", "trace_decode_token_ids",
                 "logprobs", "prompt_logprobs", "logprob_token_ids", "extra_args", "logits_processors",
                 "routed_experts_prompt_start"):
        # Zero logprobs still requests token probability metadata.
        value = getattr(params, name, None)
        if (name in {"logprobs", "prompt_logprobs"} and value is not None) or (
                name not in {"logprobs", "prompt_logprobs"} and value):
            return "stateful_or_unmapped_" + name
    if not getattr(params, "detokenize", True):
        return "no_detokenization"
    if getattr(params, "temperature", 1.) <= 0:
        return "greedy_would_repeat"
    # Changing FINAL_ONLY into DELTA also changes the native collector's merge
    # path, which currently drops some final metadata when outputs coalesce.
    # Until that independent seam is qualified, non-streaming is passthrough.
    if getattr(params.output_kind, "name", str(params.output_kind)) != "DELTA":
        return "unsupported_output_kind"
    return None


def retry_prompt(original, retained):
    """Copy metadata, share immutable MM data, replace only token-prefix arrays.

    Original MM placeholder positions remain valid because original tokens are
    an exact prefix. Do not re-render chat or re-encode images. We intentionally
    omit stale text/offsets on the retry engine input; public output restores the
    original request's prompt metadata.
    """
    result = original.copy()
    result["prompt_token_ids"] = original["prompt_token_ids"] + list(retained)
    result.pop("prompt", None)
    result.pop("prompt_token_offsets", None)
    if original.get("assistant_tokens_mask") is not None:
        result["assistant_tokens_mask"] = list(original["assistant_tokens_mask"]) + [1]*len(retained)
    return result


class PhaseGuard:
    """Conservative one-way disarm before any answer/tool bytes can be retried."""
    MARKERS = ("</think>", "<｜DSML｜ calls>", "<｜DSML｜calls>",
               '<｜DSML｜ invoke', '<｜DSML｜invoke')

    def __init__(self, boundary_token_sequences=()):
        self.sequences = tuple(tuple(s) for s in boundary_token_sequences if s)
        self.text_tail = ""
        self.token_tail = ()
        self.ended = False
        self.partial_boundary = False

    def feed(self, text, token_ids):
        joined = self.text_tail + text
        ids = self.token_tail + tuple(token_ids)
        self.ended |= any(m in joined for m in self.MARKERS)
        self.partial_boundary = any(joined.endswith(m[:n])
                                    for m in self.MARKERS for n in range(1, len(m)))
        for marker in self.sequences:
            self.ended |= any(ids[i:i+len(marker)] == marker for i in range(len(ids)-len(marker)+1))
            self.partial_boundary |= any(ids[-n:] == marker[:n] for n in range(1, len(marker)))
        self.text_tail = joined[-max(map(len, self.MARKERS)):]
        max_ids = max((len(s) for s in self.sequences), default=1)
        self.token_tail = ids[-max_ids:]
        return self.ended


@dataclass
class Pending:
    output: object
    start: int
    end: int
    byte_count: int
    fence: object


async def generate_with_recovery(engine, prompt, params, request_id, *,
                                 config=None, detector_factory=CollapseDetector,
                                 weak_detector_factory=WeakSuspicion,
                                 boundary_token_sequences=(), on_event=lambda event: None,
                                 **kwargs):
    """Yield native RequestOutputs, preserving the outer parser's single lifetime.

    on_event accepts aggregate dictionaries only. It must be nonblocking and
    exception-free; observer failure does not kill a request. Aborted attempts'
    observed tokens are a LOWER BOUND on compute (in-flight tokens may be lost).
    No fabricated claim of complete GPU billing is made.
    """
    config = config or RecoveryConfig()

    def event(event_name, **fields):
        try:
            on_event({"event": event_name, **fields})
        except Exception:
            pass

    reason = None if config.mode == "off" else eligibility(prompt, params, kwargs)
    if config.mode == "off" or reason:
        if reason:
            event("recovery_skipped", reason=reason)
        iterator = engine.generate(prompt, params, request_id, **kwargs)
        try:
            async for result in iterator:
                yield result
        finally:
            # Closing the native generator triggers its own request cancellation.
            await iterator.aclose()
        return

    # DELTA exposes each generated token once; do not mutate the caller's object,
    # even when SamplingParams.skip_clone would make .clone() return itself.
    work_params = copy.deepcopy(params)
    work_params.output_kind = type(params.output_kind).DELTA
    detector = detector_factory()
    holdback = HoldbackPolicy(config.normal_holdback_tokens if config.adaptive else config.holdback_tokens,
                             config.holdback_tokens, config.release_misses, event)
    weak_detector = weak_detector_factory(holdback.update) if config.adaptive else None
    phase = PhaseGuard(boundary_token_sequences)
    accepted_ids = []
    published = 0
    pending = deque()
    pending_bytes = 0
    retry_number = 0
    observed_generated = 0
    discarded_generated = 0
    attempted_prompt_tokens = 0
    original_metadata = None
    current_id = request_id
    current_prompt = prompt
    live_iterator = None
    active = False
    recovery_armed = True
    terminal = False
    final_published = False
    first_seed = getattr(params, "seed", None)
    seeds = {first_seed}
    aborts = 0
    close_task = None
    read_task = None
    maximum_retained_tokens = maximum_retained_bytes = maximum_retained_chunks = 0

    def public(raw):
        # The chat API derives its usage from the outputs it receives. It must
        # never count the retained completion as a new user prompt.
        result = copy.copy(raw)
        result.request_id = request_id
        result.outputs = [copy.copy(raw.outputs[0])]
        if retry_number:
            for name, value in original_metadata.items():
                setattr(result, name, value)
            # Last-attempt timings/spec acceptance do not describe the complete
            # logical request. Keep this uncertainty explicit in event telemetry.
            result.metrics = None
            result.outputs[0].spec_decode_metrics = None
        return result

    def accept_public(raw):
        nonlocal published, final_published
        result = public(raw)
        published += len(result.outputs[0].token_ids)
        final_published |= result.finished
        return result

    async def protected_outputs(iterator):
        nonlocal read_task
        while True:
            read_task = asyncio.create_task(anext(iterator))
            try:
                result = await asyncio.shield(read_task)
            except StopAsyncIteration:
                return
            finally:
                if read_task is not None and read_task.done():
                    read_task = None
            yield result

    async def perform_close(abort, closing_id, closing_iterator):
        nonlocal active, live_iterator, aborts, read_task
        # Native AsyncLLM.abort removes the local id mapping BEFORE awaiting its
        # engine-core ABORT send. Cancelling that await would make a later abort
        # find no ids, potentially orphaning the actual engine request. This
        # entire coroutine therefore has an owned, never-caller-cancelled task.
        try:
            if abort:
                aborts += 1
                await engine.abort(closing_id)
        finally:
            try:
                # A consumer cancellation must not enter native generate()'s
                # own unshielded abort before our protected core send. Its
                # __anext__ has a separate owned task; cancel/await it only now,
                # then aclose cannot race an iterator that is still running.
                if read_task is not None:
                    if not read_task.done():
                        read_task.cancel()
                    try:
                        await read_task
                    except (asyncio.CancelledError, StopAsyncIteration):
                        pass
                    except Exception as read_error:
                        event("recovery_read_cleanup_failed", exception_type=type(read_error).__name__)
                    read_task = None
                if closing_iterator is not None:
                    await closing_iterator.aclose()
            finally:
                active = False
                live_iterator = None

    async def close_attempt(abort):
        nonlocal close_task
        if close_task is None:
            if not active:
                return
            close_task = asyncio.create_task(perform_close(abort, current_id, live_iterator))
        owned = close_task
        cancellation = None
        try:
            # Retain the task and await its completion even through repeated
            # client cancellation. Only then re-propagate cancellation; no
            # retry or synthetic terminal response can race the old ABORT send.
            while not owned.done():
                try:
                    await asyncio.shield(owned)
                except asyncio.CancelledError as exc:
                    if owned.cancelled():
                        raise
                    cancellation = exc
                except BaseException:
                    break  # Retrieve the owned task's original exception below.
            owned.result()
            if cancellation is not None:
                raise cancellation
        finally:
            # Never drop the only strong reference to unfinished cleanup.
            if owned.done() and close_task is owned:
                close_task = None

    try:
        while not terminal:
            attempted_prompt_tokens += len(current_prompt["prompt_token_ids"])
            live_iterator = engine.generate(current_prompt, work_params, current_id, **kwargs)
            active = True
            retry_checkpoint = None
            async for result in protected_outputs(live_iterator):
                if retry_number == 0:
                    original_metadata = {name: getattr(result, name, None) for name in (
                        "prompt", "prompt_token_ids", "prompt_logprobs", "encoder_prompt",
                        "encoder_prompt_token_ids", "num_cached_tokens", "num_cache_creation_tokens")}
                if len(result.outputs) != 1 or result.outputs[0].index != 0:
                    raise RuntimeError("eligible request returned unexpected choices")
                output = result.outputs[0]
                ids = list(output.token_ids)
                start = len(accepted_ids)
                observed_generated += len(ids)
                accepted_ids.extend(ids)
                fence = detector.fence_state() if hasattr(detector, "fence_state") else None
                pending.append(Pending(result, start, len(accepted_ids), len(output.text.encode()), fence))
                pending_bytes += pending[-1].byte_count
                crossed_boundary = phase.feed(output.text, ids)
                # Never turn an already finished response into another generation.
                naturally_finished = result.finished or output.finish_reason is not None
                if crossed_boundary or naturally_finished:
                    recovery_armed = False
                if naturally_finished:
                    # Mark before yielding its final chunk: consumers commonly
                    # close immediately after that yield, without requesting one
                    # more item from this generator.
                    terminal = True
                detected = None
                if recovery_armed:
                    if weak_detector is not None:
                        weak_detector.feed(output.text, start, len(accepted_ids))
                    detected = detector.feed(output.text, start, len(accepted_ids))
                    # Give a partial control marker its next chunk before acting
                    # on an incidental detection immediately before a tool call.
                    if phase.partial_boundary:
                        detected = None
                if detected and config.mode == "observe":
                    event("loop_observed", **detected.evidence)
                    recovery_armed = False
                elif detected:
                    event("loop_detected", attempt=retry_number, **detected.evidence)
                    matching = next((p for p in pending if p.start == detected.start_token), None)
                    unavailable = matching is None or detected.start_token < published
                    exhausted = retry_number >= config.max_retries
                    if unavailable or exhausted:
                        # Stop visibly, retaining already accepted reasoning and
                        # emitting an explicit stop_reason. No synthetic answer,
                        # hidden tool call, or claim of successful recovery.
                        event("recovery_stopped", reason="checkpoint_published" if unavailable else "attempts_exhausted")
                        result = copy.copy(result)
                        result.outputs = [copy.copy(output)]
                        result.outputs[0].finish_reason = "stop"
                        result.outputs[0].stop_reason = "ds41f_reasoning_loop"
                        result.finished = True
                        pending[-1].output = result
                        terminal = True
                        await close_attempt(True)
                    else:
                        retry_checkpoint = matching
                        break
                flush_all = (not recovery_armed or config.mode == "observe" or terminal or naturally_finished)
                while pending and (flush_all or
                        len(accepted_ids)-pending[0].start > holdback.limit_tokens or
                        pending_bytes > config.holdback_bytes or len(pending) > config.holdback_chunks):
                    p = pending.popleft()
                    pending_bytes -= p.byte_count
                    emitted = accept_public(p.output)
                    if emitted is not None:
                        yield emitted
                # Retained state AFTER processing one indivisible native delta.
                # Arrival can transiently include one larger upstream delta;
                # no token/text slicing is used to manufacture a smaller chunk.
                maximum_retained_tokens = max(maximum_retained_tokens, len(accepted_ids)-published)
                maximum_retained_bytes = max(maximum_retained_bytes, pending_bytes)
                maximum_retained_chunks = max(maximum_retained_chunks, len(pending))
                if naturally_finished:
                    terminal = True
                    break
                if terminal:
                    break
            if retry_checkpoint is not None:
                await close_attempt(True)
                boundary = retry_checkpoint.start
                discarded_generated += len(accepted_ids)-boundary
                accepted_ids = accepted_ids[:boundary]
                pending = deque(p for p in pending if p.end <= boundary and p is not retry_checkpoint)
                pending_bytes = sum(p.byte_count for p in pending)
                retry_number += 1
                current_id = request_id + ".loop-retry-" + str(retry_number) + "-" + secrets.token_hex(6)
                current_prompt = retry_prompt(prompt, accepted_ids)
                work_params = copy.deepcopy(params)
                work_params.output_kind = type(params.output_kind).DELTA
                work_params.max_tokens = params.max_tokens - len(accepted_ids)
                seed = secrets.randbits(63)
                while seed in seeds:
                    seed = secrets.randbits(63)
                seeds.add(seed)
                work_params.seed = seed
                # SamplingParams.sampling_type is a cached_property in this
                # pinned build. A former RANDOM cache must not defeat new seed.
                getattr(work_params, "__dict__", {}).pop("sampling_type", None)
                detector = detector_factory()
                holdback.reset()
                weak_detector = weak_detector_factory(holdback.update) if config.adaptive else None
                if retry_checkpoint.fence is not None and hasattr(detector, "restore_fence"):
                    detector.restore_fence(retry_checkpoint.fence)
                    if weak_detector is not None:
                        weak_detector.restore_fence(retry_checkpoint.fence)
                phase = PhaseGuard(boundary_token_sequences)
                event("recovery_restarted", attempt=retry_number,
                      retained_tokens=len(accepted_ids), discarded_tokens=discarded_generated,
                      retry_prompt_tokens=len(current_prompt["prompt_token_ids"]))
                continue
            if not terminal:
                raise RuntimeError("engine iterator ended without final output")
            await close_attempt(False)
    finally:
        try:
            if active:
                # close_attempt owns/shields the whole cleanup on every path.
                original_error = sys.exc_info()[1]
                try:
                    await close_attempt(not terminal)
                except Exception as cleanup_error:
                    event("recovery_cleanup_failed", exception_type=type(cleanup_error).__name__)
                    if original_error is None:
                        raise
        finally:
            event("recovery_accounting", retries=retry_number,
                  visible_completion_tokens=published,
                  observed_generated_tokens=observed_generated,
                  discarded_generated_tokens=discarded_generated,
                  attempted_prompt_tokens=attempted_prompt_tokens,
                  aborted_inflight_tokens_unknown=aborts > 0,
                  buffer_expansions=holdback.expansions,
                  buffer_releases=holdback.releases,
                  maximum_retained_tokens=maximum_retained_tokens,
                  maximum_retained_bytes=maximum_retained_bytes,
                  maximum_retained_chunks=maximum_retained_chunks,
                  completed=terminal and final_published)
