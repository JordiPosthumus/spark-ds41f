# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

import vllm.envs as envs
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
from vllm.v1.worker.gpu.states import RequestState


class BadWordsState:
    def __init__(self, req_states: RequestState):
        self.req_states = req_states
        self.max_num_reqs = req_states.max_num_reqs
        self.device = req_states.device

        max_total_tokens = envs.VLLM_MAX_BAD_WORDS_TOTAL_TOKENS
        max_num_bad_words = envs.VLLM_MAX_NUM_BAD_WORDS
        # flattened bad word tokens: [max_num_reqs, VLLM_MAX_BAD_WORDS_TOTAL_TOKENS]
        self.bad_word_token_ids = StagedWriteTensor(
            (self.max_num_reqs, max_total_tokens),
            dtype=torch.int32,
            device=self.device,
        )
        # cumulative offsets of bad words: [max_num_reqs, VLLM_MAX_NUM_BAD_WORDS + 1]
        self.bad_word_offsets = StagedWriteTensor(
            (self.max_num_reqs, max_num_bad_words + 1),
            dtype=torch.int32,
            device=self.device,
        )
        # number of bad words per request
        self.num_bad_words = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        # -1 preserves ordinary permanent bad_words. Reset on every slot reuse.
        self.bad_words_until = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.bad_words_stop0 = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.bad_words_stop1 = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        until = (sampling_params.extra_args or {}).get("ds41f_loopbuster_bad_words_until", -1)
        if type(until) is not int or (until != -1 and not 1 <= until <= 256):
            raise ValueError("invalid ds41f_loopbuster_bad_words_until")
        self.bad_words_until.np[req_idx] = until
        stop_ids = (sampling_params.extra_args or {}).get("ds41f_loopbuster_stop_ids", [-1, -1])
        if not isinstance(stop_ids, list) or len(stop_ids) != 2 or any(type(t) is not int or t < -1 for t in stop_ids):
            raise ValueError("invalid ds41f_loopbuster_stop_ids")
        self.bad_words_stop0.np[req_idx], self.bad_words_stop1.np[req_idx] = stop_ids
        bad_words_token_ids = sampling_params.bad_words_token_ids
        if not bad_words_token_ids:
            self.num_bad_words.np[req_idx] = 0
            return

        num_bad_words = len(bad_words_token_ids)
        max_num_bad_words = envs.VLLM_MAX_NUM_BAD_WORDS
        if num_bad_words > max_num_bad_words:
            raise ValueError(
                f"Too many bad words: {num_bad_words}. "
                f"The max number is {max_num_bad_words}."
            )

        # Flatten bad words and compute offsets
        flattened_tokens: list[int] = []
        offsets: list[int] = [0]
        for bad_word in bad_words_token_ids:
            flattened_tokens.extend(bad_word)
            offsets.append(len(flattened_tokens))

        max_total_tokens = envs.VLLM_MAX_BAD_WORDS_TOTAL_TOKENS
        if len(flattened_tokens) > max_total_tokens:
            raise ValueError(
                f"Too many total bad word tokens: {len(flattened_tokens)}. "
                f"The max is {max_total_tokens}."
            )

        # Stage writes
        self.bad_word_token_ids.stage_write(req_idx, 0, flattened_tokens)
        self.bad_word_offsets.stage_write(req_idx, 0, offsets)
        self.num_bad_words.np[req_idx] = num_bad_words

    def apply_staged_writes(self) -> None:
        self.num_bad_words.copy_to_uva()
        self.bad_words_until.copy_to_uva()
        self.bad_words_stop0.copy_to_uva()
        self.bad_words_stop1.copy_to_uva()
        self.bad_word_token_ids.apply_write()
        self.bad_word_offsets.apply_write()

    def apply_bad_words(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> None:
        max_num_bad_words = int(self.num_bad_words.np[idx_mapping_np].max())
        if max_num_bad_words == 0:
            # No request uses bad words. Skip the kernel launch.
            return

        apply_bad_words(
            logits,
            expanded_idx_mapping,
            self.bad_word_token_ids.gpu,
            self.bad_word_offsets.gpu,
            self.num_bad_words.gpu,
            self.req_states.all_token_ids.gpu,
            self.req_states.prompt_len.gpu,
            self.req_states.total_len.gpu,
            input_ids,
            expanded_local_pos,
            max_num_bad_words,
            bad_words_until=self.bad_words_until.gpu,
            bad_words_stop0=self.bad_words_stop0.gpu,
            bad_words_stop1=self.bad_words_stop1.gpu,
        )


@triton.jit
def _bad_words_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    bad_word_token_ids_ptr,
    bad_word_token_ids_stride,
    bad_word_offsets_ptr,
    bad_word_offsets_stride,
    num_bad_words_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    total_len_ptr,
    input_ids_ptr,
    expanded_local_pos_ptr,
    bad_words_until_ptr,
    bad_words_stop0_ptr,
    bad_words_stop1_ptr,
    HAS_EXPIRY: tl.constexpr,
    HAS_BOUNDARY: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    bw_idx = tl.program_id(1)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    num_bad_words = tl.load(num_bad_words_ptr + req_state_idx)

    if bw_idx >= num_bad_words:
        return

    pos = tl.load(expanded_local_pos_ptr + token_idx)
    cur_req_first_pos = token_idx - pos

    prompt_len = tl.load(prompt_len_ptr + req_state_idx)
    total_len = tl.load(total_len_ptr + req_state_idx)
    output_len = total_len - prompt_len
    effective_len = output_len + pos
    if HAS_EXPIRY:
        until = tl.load(bad_words_until_ptr + req_state_idx)
        # Apply to generated indices0..until-1, independently for every
        # speculative/bonus row. No extra abort or host sampling roundtrip.
        if until >= 0 and effective_len >= until:
            return
        if until >= 0 and HAS_BOUNDARY:
            stop0 = tl.load(bad_words_stop0_ptr + req_state_idx)
            stop1 = tl.load(bad_words_stop1_ptr + req_state_idx)
            history = tl.arange(0, 256)
            valid = (history < effective_len) & (history < until)
            from_draft = history >= output_len
            committed = tl.load(all_token_ids_ptr + req_state_idx * all_token_ids_stride + prompt_len + history, mask=valid & ~from_draft, other=-2)
            drafted = tl.load(input_ids_ptr + cur_req_first_pos + history - output_len + 1, mask=valid & from_draft, other=-2)
            preceding = tl.where(from_draft, drafted, committed)
            has_boundary = tl.sum((valid & ((preceding == stop0) | (preceding == stop1))).to(tl.int32), 0) > 0
            if has_boundary:
                return

    bd_offsets_base = bad_word_offsets_ptr + req_state_idx * bad_word_offsets_stride
    bd_tokens_base = bad_word_token_ids_ptr + req_state_idx * bad_word_token_ids_stride
    output_base = all_token_ids_ptr + req_state_idx * all_token_ids_stride + prompt_len

    start = tl.load(bd_offsets_base + bw_idx)
    end = tl.load(bd_offsets_base + bw_idx + 1)
    bad_word_len = end - start
    prefix_len = bad_word_len - 1

    if prefix_len > effective_len:
        return

    last_token = tl.load(bd_tokens_base + end - 1)
    match = 1
    for i in range(prefix_len):
        expected = tl.load(bd_tokens_base + start + i)
        actual_pos = effective_len - prefix_len + i

        from_spec_input = actual_pos >= output_len
        if from_spec_input:
            # input_ids at local position 0 is the last committed token;
            # draft tokens start at local position 1.
            spec_offset = actual_pos - output_len
            actual = tl.load(input_ids_ptr + cur_req_first_pos + spec_offset + 1)
        else:
            actual = tl.load(output_base + actual_pos)

        match = match & (expected == actual)

    if match:
        tl.store(logits_ptr + token_idx * logits_stride + last_token, -float("inf"))


def apply_bad_words(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    bad_word_token_ids: torch.Tensor,
    bad_word_offsets: torch.Tensor,
    num_bad_words: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    total_len: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    max_num_bad_words: int,
    bad_words_until: torch.Tensor | None = None,
    bad_words_stop0: torch.Tensor | None = None,
    bad_words_stop1: torch.Tensor | None = None,
) -> None:
    num_tokens = logits.shape[0]
    _bad_words_kernel[(num_tokens, max_num_bad_words)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        bad_word_token_ids,
        bad_word_token_ids.stride(0),
        bad_word_offsets,
        bad_word_offsets.stride(0),
        num_bad_words,
        all_token_ids,
        all_token_ids.stride(0),
        prompt_len,
        total_len,
        input_ids,
        expanded_local_pos,
        bad_words_until if bad_words_until is not None else num_bad_words,
        bad_words_stop0 if bad_words_stop0 is not None else num_bad_words,
        bad_words_stop1 if bad_words_stop1 is not None else num_bad_words,
        HAS_EXPIRY=bad_words_until is not None,
        HAS_BOUNDARY=bad_words_stop0 is not None and bad_words_stop1 is not None,
    )
