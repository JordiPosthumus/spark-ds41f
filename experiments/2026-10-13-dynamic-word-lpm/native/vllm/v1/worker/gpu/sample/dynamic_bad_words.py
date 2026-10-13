"""Request-keyed dynamic additions, using upstream BadWordsState unchanged matching.

Worker RPCs and model execution are serialized by the native worker queue.
Writes use the same CUDA stream and native staged/UVA buffer pools. No GPU
synchronize, reseeding, retry, or KV reset is performed for a word addition.
"""
from types import SimpleNamespace

MAX_WORDS = 2048
MAX_TOKENS = 65536


def validate(ids, vocab_size):
    if not isinstance(ids, list) or len(ids) > MAX_WORDS:
        raise ValueError("dynamic bad_words sequence capacity exceeded")
    if sum(map(len, ids)) > MAX_TOKENS:
        raise ValueError("dynamic bad_words token capacity exceeded")
    if any(not s or any(type(t) is not int or not 0 <= t < vocab_size for t in s) for s in ids):
        raise ValueError("invalid dynamic bad_words token sequence")


def update(runner, req_id, ids, version):
    sampler = runner.sampler
    if sampler is None:
        return {"status": "finished"}
    validate(ids, sampler.sampling_states.vocab_size)
    idx = runner.req_states.req_id_to_index.get(req_id)
    records = runner.ds41f_dynamic_records
    prior = records.get(req_id)
    if idx is None and prior is None:
        return {"status": "finished"}
    if idx is not None and not sampler.loop_protection_state.use_protection[idx]:
        raise ValueError("dynamic update outside a protected request")
    if prior and version <= prior[0]:
        return {"status": "applied" if idx is not None else "deferred", "version": prior[0]}
    records[req_id] = (version, ids)
    if idx is None:
        return {"status": "deferred", "version": version}
    sampler.dynamic_bad_words_state.add_request(idx, SimpleNamespace(bad_words_token_ids=ids))
    sampler.dynamic_bad_words_state.apply_staged_writes()
    return {"status": "applied", "version": version}


def restore(runner, req_id, idx):
    record = runner.ds41f_dynamic_records.get(req_id)
    if record:
        runner.sampler.dynamic_bad_words_state.add_request(
            idx, SimpleNamespace(bad_words_token_ids=record[1]))
