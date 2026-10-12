"""Source-verified opt-in chat-serving seam. Real-detector proposal. No code here is installed live.

The package produced by prepare_overlay.py is mounted only for a separately
approved Ring 2 canary. Normal environment default is off, using the original
engine iterator directly, with zero tokenizer/detector work.
"""
import importlib.metadata
import json
import logging
import os

try:
    from .recovery import PhaseGuard, RecoveryConfig, generate_with_recovery
    from .policy import SustainedCollapse
    from .retry_words import apply_retry_words
except ImportError:
    from recovery import PhaseGuard, RecoveryConfig, generate_with_recovery
    from policy import SustainedCollapse
    from retry_words import apply_retry_words


LOG = logging.getLogger(__name__)


def _emit(event):
    # Fixed metric keys/numbers only: no request body, reasoning, token values,
    # seed, image content, tool arguments, or client-identifying request id.
    LOG.info("ds41f_loop_recovery %s", json.dumps(event, sort_keys=True))


def generate_for_ds41f(server, prompt, params, request_id, **kwargs):
    mode = os.environ.get("SPARK_DS41F_LOOP_RECOVERY_MODE", "off")
    if mode == "off":
        return server.engine_client.generate(prompt, params, request_id, **kwargs)
    reason = None
    if mode not in {"observe", "recover", "adaptive", "adaptive-observe", "guard", "auto-recover"}:
        reason = "invalid_mode"
    elif getattr(server.parser_cls, "__name__", None) != "DeepSeekV41Parser":
        reason = "unverified_model_parser"
    else:
        config = getattr(server.engine_client, "vllm_config", None)
        if config is None:
            reason = "unknown_engine_config"
        elif getattr(config, "watermark_config", None) is not None and getattr(params, "watermarking", True):
            reason = "stateful_engine_watermark"
        elif getattr(config.model_config, "enable_return_routed_experts", False):
            reason = "unmapped_routed_expert_metadata"
        elif getattr(config.model_config, "logits_processors", None) or importlib.metadata.entry_points(group="vllm.logits_processors"):
            reason = "unverified_logits_processor"
    if reason:
        _emit({"event": "recovery_skipped", "reason": reason})
        return server.engine_client.generate(prompt, params, request_id, **kwargs)
    try:
        tokenizer = server.renderer.tokenizer
        boundaries = [tuple(tokenizer.encode(marker, add_special_tokens=False))
                      for marker in PhaseGuard.MARKERS]
        if any(not sequence for sequence in boundaries):
            raise ValueError("missing boundary tokens")
    except Exception:
        _emit({"event": "recovery_skipped", "reason": "unverified_boundary_tokens"})
        return server.engine_client.generate(prompt, params, request_id, **kwargs)
    adaptive = mode in {"adaptive", "adaptive-observe"}
    recovery_mode = "recover" if mode == "adaptive" else "observe" if mode == "adaptive-observe" else mode
    return generate_with_recovery(server.engine_client, prompt, params, request_id,
                                  config=RecoveryConfig(mode=recovery_mode, adaptive=adaptive, max_retries=10 if mode == "auto-recover" else 2),
                                  boundary_token_sequences=boundaries, detector_factory=SustainedCollapse,
                                  on_event=_emit,
                                  retry_word_applier=lambda p, words: apply_retry_words(p, words, tokenizer), **kwargs)
