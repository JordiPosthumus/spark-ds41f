"""Temporary lifetime gating for vLLM bad_words; never matches phrases or writes logits."""
import os
import numpy as np
import torch
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

WORD_LIMIT = 2000
_STATE_WIDTH = 9
# word flag, UTF8 state, count, scan position, active, seed word flag,
# seed UTF8 state, seed prompt length, protected request (including expired).

@triton.jit
def _advance_token(token, lex, utf, words, active,
    token_prefix_ptr, token_body_ptr, bridge_ptr, lex_ptr, body_ptr,
    body_nonempty_ptr, body_tail_ptr, PREFIXES: tl.constexpr, BODIES: tl.constexpr):
    old_lex, old_utf, old_words = lex, utf, words
    use_token = active & (token != 1) & (token != 128822) & (token != 128825)
    prefix = tl.load(token_prefix_ptr + token)
    body_id = tl.load(token_body_ptr + token)
    bridge = tl.load(bridge_ptr + utf * PREFIXES + prefix)
    new_utf = bridge & 511
    count = (bridge >> 9) & 3
    for event in tl.static_range(3):
        category = (bridge >> (11 + event * 4)) & 15
        category = tl.where(event < count, category, 0)
        step = tl.load(lex_ptr + lex * 3 + category)
        lex = tl.where(event < count, step & 31, lex)
        words += tl.where(event < count, (step >> 5) & 255, 0)
    body_exists = tl.load(body_nonempty_ptr + body_id) != 0
    step = tl.load(body_ptr + lex * BODIES + body_id)
    lex = tl.where(body_exists, step & 31, lex)
    words += tl.where(body_exists, (step >> 5) & 255, 0)
    new_utf = tl.where(body_exists, tl.load(body_tail_ptr + body_id), new_utf)
    return (tl.where(use_token, lex, old_lex),
            tl.where(use_token, new_utf, old_utf),
            tl.where(use_token, tl.minimum(words, 2000), old_words),
            use_token & (words < 2000))

@triton.jit
def _update_committed(
    idx_mapping_ptr, states_ptr, all_ids_ptr, all_ids_stride,
    total_len_ptr, token_prefix_ptr, token_body_ptr, bridge_ptr, lex_ptr,
    body_ptr, body_nonempty_ptr, body_tail_ptr,
    PREFIXES: tl.constexpr, BODIES: tl.constexpr,
):
    batch_row = tl.program_id(0)
    req = tl.load(idx_mapping_ptr + batch_row)
    if req < 0:
        return
    base = states_ptr + req * 9
    active = tl.load(base + 4) != 0
    if not active:
        return
    lex = tl.load(base)
    utf = tl.load(base + 1)
    words = tl.load(base + 2)
    scan = tl.load(base + 3)
    end = tl.load(total_len_ptr + req)
    # Normally monotonic. Recompute from the immutable trusted seed if native
    # accepted history contracts, instead of retaining rejected suffix state.
    if scan > end:
        lex = tl.load(base + 5)
        utf = tl.load(base + 6)
        scan = tl.load(base + 7)
        words = 0
    if end < scan:
        return
    while scan < end and active:
        token = tl.load(all_ids_ptr + req * all_ids_stride + scan)
        lex, utf, words, active = _advance_token(
            token, lex, utf, words, active,
            token_prefix_ptr, token_body_ptr, bridge_ptr, lex_ptr, body_ptr,
            body_nonempty_ptr, body_tail_ptr, PREFIXES, BODIES,
        )
        scan += 1
    tl.store(base, lex)
    tl.store(base + 1, utf)
    tl.store(base + 2, words)
    tl.store(base + 3, end)
    tl.store(base + 4, active.to(tl.int32))


@triton.jit
def _protection_rows(rows_ptr, expanded_mapping_ptr, local_pos_ptr,
    input_ids_ptr, states_ptr, token_prefix_ptr, token_body_ptr, bridge_ptr,
    lex_ptr, body_ptr, body_nonempty_ptr, body_tail_ptr,
    PREFIXES: tl.constexpr, BODIES: tl.constexpr, VOCAB: tl.constexpr):
    row = tl.program_id(0)
    req = tl.load(expanded_mapping_ptr + row)
    if req < 0:
        tl.store(rows_ptr + row, False)
        return
    base = states_ptr + req * 9
    if tl.load(base + 8) == 0:
        # Ordinary client bad_words must keep their standard lifetime.
        tl.store(rows_ptr + row, True)
        return
    active = tl.load(base + 4) != 0
    lex = tl.load(base)
    utf = tl.load(base + 1)
    words = tl.load(base + 2)
    pos = tl.load(local_pos_ptr + row)
    first = row - pos
    for i in range(pos):
        token = tl.load(input_ids_ptr + first + i + 1)
        valid = (token >= 0) & (token < VOCAB)
        active = active & valid
        token = tl.where(valid, token, 1)
        lex, utf, words, active = _advance_token(
            token, lex, utf, words, active, token_prefix_ptr, token_body_ptr,
            bridge_ptr, lex_ptr, body_ptr, body_nonempty_ptr, body_tail_ptr,
            PREFIXES, BODIES)
    tl.store(rows_ptr + row, active)

class LoopProtectionState:
    """No device synchronization and no extra launches for untracked batches."""

    def __init__(self, req_states, max_rows):
        self.req_states = req_states
        self.use_protection = np.zeros(req_states.max_num_reqs, dtype=bool)
        table_dir = os.environ.get("SPARK_DS41F_PHRASE_TABLES")
        self.enabled = os.environ.get("SPARK_DS41F_TEMPORARY_PHRASES") == "1" and bool(table_dir)
        self.tables = None
        self._pending_resets = {}
        if not self.enabled:
            return
        # Pure-CPU shared loader checks hashes, dimensions, native model IDs and
        # fixed policy before any GPU allocation. Its final location is supplied
        # by the compiler prototype; this is intentionally not a public setting.
        from .loop_protection_tables import get_tables
        if not table_dir:
            raise ValueError("Loop protection requires validated tables")
        arrays = get_tables()
        expected = {
            "token_prefix_id", "token_body_id", "bridge_table", "lex_table",
            "body_table", "body_nonempty", "body_tail",
        }
        if not expected.issubset(arrays):
            raise ValueError("Incomplete native phrase suppression tables")
        self.vocab = len(arrays["token_prefix_id"])
        self.n_utf, self.n_prefixes = arrays["bridge_table"].shape
        self.n_lex, self.n_bodies = arrays["body_table"].shape
        if self.vocab != req_states.vocab_size or self.n_lex != 2:
            raise ValueError("Loop protection table/model mismatch")
        if sum(arrays[name].nbytes for name in expected) > 64 * 1024 * 1024:
            raise ValueError("Loop protection tables exceed memory bound")
        self.tables = tuple(torch.from_numpy(np.array(arrays[name], copy=True)).to(
            device=req_states.device
        ) for name in (
            "token_prefix_id", "token_body_id", "bridge_table", "lex_table",
            "body_table", "body_nonempty", "body_tail",
        ))
        self.state = StagedWriteTensor(
            (req_states.max_num_reqs, _STATE_WIDTH), dtype=torch.int32,
            device=req_states.device,
        )

        self.row_flags = torch.empty(max_rows, dtype=torch.bool, device=req_states.device)
        self.history_floor = StagedWriteTensor(
            (req_states.max_num_reqs, 1), dtype=torch.int32, device=req_states.device)

    def add_request(self, req_idx, prompt_len, sampling_params):
        metadata = getattr(sampling_params, "_ds41f_loop_protection", None)
        self.use_protection[req_idx] = False
        if metadata is not None and not self.enabled:
            raise ValueError("Loop protection request on disabled worker")
        if not self.enabled:
            return
        seed = [0, 0, 0, int(prompt_len), 0, 0, 0, int(prompt_len), 0]
        floor = int(prompt_len)
        if metadata is not None:
            from .loop_protection_tables import get_policy
            phrases, ids = get_policy()
            if (not isinstance(metadata, (tuple, list)) or len(metadata) != 4
                    or any(type(x) is not int for x in metadata)):
                raise ValueError("Invalid internal loop protection state")
            version, lex, utf, tail = metadata
            if (version != 2 or not 0 <= lex < self.n_lex or not 0 <= utf < self.n_utf
                    or not 0 <= tail <= min(prompt_len, max(map(len, ids))-1)):
                raise ValueError("Invalid loop protection seed")
            if sampling_params.bad_words != phrases or sampling_params.bad_words_token_ids != ids:
                raise ValueError("Native bad_words policy mismatch")
            seed = [lex, utf, 0, prompt_len, 1, lex, utf, prompt_len, 1]
            floor -= tail
            self.use_protection[req_idx] = True
        self._pending_resets[req_idx] = (seed, floor)

    def remove_request(self, req_idx):
        self.use_protection[req_idx] = False
        if self.enabled:
            self._pending_resets[req_idx] = ([0] * _STATE_WIDTH, 0)

    def apply_staged_writes(self):
        if self.enabled:
            for req_idx, (seed, floor) in self._pending_resets.items():
                self.state.stage_write(req_idx, 0, seed)
                self.history_floor.stage_write(req_idx, 0, [floor])
            self._pending_resets.clear()
            self.state.apply_write()
            self.history_floor.apply_write()

    def rows(self, num_rows, expanded_idx_mapping, idx_mapping, idx_mapping_np,
             input_ids, expanded_local_pos):
        if not self.enabled or not np.any(self.use_protection[idx_mapping_np]):
            return None, None
        if num_rows > self.row_flags.numel():
            raise ValueError("Loop protection expanded-row capacity exceeded")
        _update_committed[(idx_mapping.shape[0],)](
            idx_mapping, self.state.gpu, self.req_states.all_token_ids.gpu,
            self.req_states.all_token_ids.gpu.stride(0), self.req_states.total_len.gpu,
            *self.tables, self.n_prefixes, self.n_bodies, num_warps=1)
        _protection_rows[(num_rows,)](
            self.row_flags, expanded_idx_mapping, expanded_local_pos,
            input_ids, self.state.gpu, *self.tables,
            self.n_prefixes, self.n_bodies, self.vocab, num_warps=1)
        return self.row_flags[:num_rows], self.history_floor.gpu
