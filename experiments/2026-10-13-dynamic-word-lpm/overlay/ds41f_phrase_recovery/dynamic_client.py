"""Rare per-addition worker RPC; no inference requests or prompt re-prefill."""
import time
from .dynamic_words import encode_words


class DynamicClient:
    def __init__(self, engine, tokenizer):
        self.engine, self.tokenizer = engine, tokenizer
        self.version = 0
        self.encoded = []

    async def update(self, request_id, words):
        ids = encode_words(self.tokenizer, words)
        mapping = self.engine.output_processor.external_req_ids
        internal = tuple(mapping.get(request_id, ()))
        if not internal:
            return {"status": "finished", "sequences": len(ids)}
        if len(internal) != 1:
            raise RuntimeError("dynamic mask requires one internal request")
        self.encoded = ids
        self.version += 1
        start = time.monotonic()
        results = await self.engine.collective_rpc(
            "ds41f_update_dynamic_bad_words", args=(internal[0], ids, self.version))
        if not results or any(r.get("status") not in {"applied", "deferred", "finished"} for r in results):
            raise RuntimeError("dynamic mask update was not acknowledged by all ranks")
        return {"status": "finished" if all(r["status"] == "finished" for r in results) else "applied",
                "sequences": len(ids), "ranks": len(results),
                "update_ms": round((time.monotonic() - start) * 1000, 3)}
