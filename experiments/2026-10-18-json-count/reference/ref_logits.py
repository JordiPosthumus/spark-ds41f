"""DeepSeek's V4.1 reference forward (inference/model.py of the checkpoint), run on CPU one layer at a
time, for the qeval json_count prompt. Reports the reference's next-token distribution at every step
of the greedy reply, teacher-forced through the production engine's tokens up to the number.

Usage: python3 ref_logits.py <snapshot-dir> <out.json>
Env:   REF_WO_A_QUANT=0 drops the reference's quant/dequant of WO-A's input (see shim/kernel.py).

Model code, encoding and arithmetic outside the six kernels are the reference's own, imported
unchanged. What this driver adds: tensors are read straight from the Hugging Face shards with
O_DIRECT (no page cache), only the routed experts and Engram rows a step touches are read, and each
layer is built, run and freed in turn, carrying its KV caches across steps. The reference's
generate() prefills the prompt and then decodes one token per forward; so does this.
"""

import json
import mmap
import os
import sys
import time

SNAP = os.path.abspath(sys.argv[1])
OUT = sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SNAP, "encoding"))
sys.path.insert(0, os.path.join(SNAP, "inference"))
sys.path.insert(0, os.path.join(HERE, "shim"))  # first, so model.py's `from kernel import` gets the shim

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

import kernel as shim  # noqa: E402
import model as M  # noqa: E402
from encoding import encode_messages  # noqa: E402
from engram import EngramLayout, NgramHashState  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

assert M.act_quant is shim.act_quant, "model.py did not pick up the CPU kernels"

PROMPT = (
    'Count the vowels (a,e,i,o,u, case-insensitive) in the string "Encyclopaedia Britannica" and '
    'return ONLY {"vowels": <integer>}.'
)
# The 39 prompt ids the production server renders (chat_template_kwargs thinking=false).
SERVER_PROMPT_IDS = [0, 128803, 8420, 270, 60183, 343, 67, 54334, 27993, 14, 81, 70845, 14, 2352, 42803, 28118, 11,
                     295, 270, 3418, 582, 4378, 19336, 122652, 72925, 4, 305, 1354, 68861, 28612, 88, 119906, 3362,
                     818, 39648, 32, 18615, 128804, 128822]
# The production engine's greedy reply up to the number: '{"' 'v' 'owels' '":' ' '
FORCED = [24313, 88, 119906, 3362, 223]
WATCH = {"10": 553, "11": 779, "12": 736, "9": 27, "{\"": 24313, "Let": 5718}
EXTRA_GREEDY = 3
ALIGN = 4096
DT = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I32": torch.int32,
    "I64": torch.int64,
}


class Shards:
    """Reads tensors and tensor rows from the snapshot's safetensors files with O_DIRECT."""

    def __init__(self, snap):
        self.snap = snap
        self.where = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
        self.headers, self.fds = {}, {}
        self.bytes_read = 0

    def meta(self, name):
        fname = self.where[name]
        if fname not in self.headers:
            path = os.path.realpath(os.path.join(self.snap, fname))
            with open(path, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                header = json.loads(f.read(n))
            self.headers[fname] = (8 + n, header)
            self.fds[fname] = os.open(path, os.O_RDONLY | os.O_DIRECT)
        start, header = self.headers[fname]
        m = header[name]
        b, e = m["data_offsets"]
        return fname, start + b, e - b, DT[m["dtype"]], m["shape"]

    def _read(self, fname, off, length):
        a0 = off // ALIGN * ALIGN
        a1 = (off + length + ALIGN - 1) // ALIGN * ALIGN
        buf = mmap.mmap(-1, a1 - a0)
        done = 0
        while done < a1 - a0:
            chunk = min(a1 - a0 - done, 1 << 30)
            got = os.preadv(self.fds[fname], [memoryview(buf)[done : done + chunk]], a0 + done)
            if got <= 0:
                break
            done += got
        assert done >= off + length - a0, (fname, off, length, done)
        view = torch.frombuffer(buf, dtype=torch.uint8, count=length, offset=off - a0)
        out = view.clone()
        del view
        buf.close()
        self.bytes_read += a1 - a0
        return out

    def get(self, name):
        fname, off, length, dt, shape = self.meta(name)
        return self._read(fname, off, length).view(dt).reshape(shape)

    def get_rows(self, name, rows):
        fname, off, length, dt, shape = self.meta(name)
        row_bytes = length // shape[0]
        out = torch.empty(len(rows), row_bytes, dtype=torch.uint8)
        for k, r in enumerate(rows):
            assert 0 <= r < shape[0]
            out[k] = self._read(fname, off + r * row_bytes, row_bytes)
        return out.view(dt).reshape(len(rows), *shape[1:])


shards = Shards(SNAP)
CONVERSIONS = {}


def assign(p: nn.Parameter, t: torch.Tensor, name: str):
    if p.dtype == torch.float4_e2m1fn_x2:
        assert t.dtype == torch.int8, (name, t.dtype)
        t = t.view(torch.float4_e2m1fn_x2)
    elif t.dtype != p.dtype:
        key = f"{t.dtype}->{p.dtype}"
        widening = (t.dtype, p.dtype) in {(torch.bfloat16, torch.float32), (torch.float16, torch.float32)}
        assert widening, f"narrowing load {name}: {key}"
        CONVERSIONS.setdefault(key, set()).add(name.split(".", 2)[-1] if name.startswith("layers.") else name)
        t = t.to(p.dtype)
    assert tuple(t.shape) == tuple(p.shape), (name, tuple(t.shape), tuple(p.shape))
    p.data = t


def dequant_wo_a(prefix):
    """convert.py's treatment of wo_a: FP8 blocks times their scales, stored as BF16."""
    weight = shards.get(prefix + "weight")
    scale = shards.get(prefix + "scale")
    ob, ib = weight.size(0) // scale.size(0), weight.size(1) // scale.size(1)
    assert (ob, ib) in ((32, 32), (128, 128))
    w = weight.unflatten(0, (-1, ob)).unflatten(-1, (-1, ib)).float() * scale[:, None, :, None].float()
    return w.flatten(2, 3).flatten(0, 1).bfloat16()


def load_module(mod: nn.Module, prefix: str, skip=()):
    for name, p in mod.named_parameters():
        if any(name.startswith(s) for s in skip):
            continue
        full = prefix + name
        t = dequant_wo_a(full[: -len("weight")]) if full.endswith("attn.wo_a.weight") else shards.get(full)
        assign(p, t, full)


# Routed experts and the Engram tables are never allocated whole: experts are built on demand,
# table rows are read on lookup.
_expert_init = M.Expert.__init__


def _lazy_expert_init(self, dim, inter_dim, dtype=None, swiglu_limit=0.0):
    if dtype == torch.float4_e2m1fn_x2:
        with torch.device("meta"):
            _expert_init(self, dim, inter_dim, dtype, swiglu_limit)
    else:
        _expert_init(self, dim, inter_dim, dtype, swiglu_limit)


M.Expert.__init__ = _lazy_expert_init


class RowReadEngramEmbedding(nn.Module):
    """ParallelEngramEmbedding.forward at world size 1, reading only the looked-up rows."""

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.block_size = M.fp8_block_size
        self.prefix = None

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        flat = indices.flatten().tolist()
        rows = sorted(set(flat))
        _, _, _, _, shape = shards.meta(self.prefix + "weight")
        assert shape == [self.num_embeddings, self.dim], (shape, self.num_embeddings)
        w = shards.get_rows(self.prefix + "weight", rows)
        s = shards.get_rows(self.prefix + "scale", rows)
        pos = {r: k for k, r in enumerate(rows)}
        sel = torch.tensor([pos[r] for r in flat])
        values = w[sel].view(*indices.shape, self.dim)
        scales = s[sel].view(*indices.shape, -1)
        values = values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
        return values.flatten(-2).to(torch.bfloat16)


M.ParallelEngramEmbedding = RowReadEngramEmbedding

ROUTES = {}
HIDDEN = {}  # start_pos -> per-layer residual stream at the last position, [hc_mult, dim]


def moe_forward(self, x, image_mask=None):
    """MoE.forward at world size 1, with each routed expert built and loaded when first used."""
    shape = x.size()
    x = x.view(-1, self.dim)
    weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
    ROUTES[self.layer_id] = indices.tolist()
    y = torch.zeros_like(x, dtype=torch.float32)
    counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
    for i in range(self.experts_start_idx, self.experts_end_idx):
        if counts[i] == 0:
            continue
        expert = M.Expert.__new__(M.Expert)
        _expert_init(expert, self.dim, ARGS.moe_inter_dim, torch.float4_e2m1fn_x2, ARGS.swiglu_limit)
        load_module(expert, f"layers.{self.layer_id}.ffn.experts.{i}.")
        idx, top = torch.where(indices == i)
        y[idx] += expert(x[idx], weights[idx, top, None])
        del expert
    y += self.shared_experts(x)
    return y.type_as(x).view(shape)


M.MoE.forward = moe_forward

torch.set_default_dtype(torch.bfloat16)
torch.set_num_threads(int(os.environ.get("REF_THREADS", "8")))
torch.manual_seed(33377335)
with open(os.path.join(SNAP, "inference", "config.json")) as f:
    ARGS = M.ModelArgs(**json.load(f))
ARGS.max_batch_size = 1
ARGS.max_seq_len = 256
ARGS.temperature = 0.0

tokenizer = AutoTokenizer.from_pretrained(SNAP)
prompt = encode_messages([{"role": "user", "content": PROMPT}], thinking_mode="chat")
prompt_ids = tokenizer.encode(prompt)  # as image_processor.prepare_vl_inputs does
assert prompt_ids == SERVER_PROMPT_IDS, prompt_ids
layout = EngramLayout.from_args(ARGS)
engram_hash = NgramHashState(ARGS, layout, tokenizer)

final_norm = M.RMSNorm(ARGS.dim, ARGS.norm_eps)
load_module(final_norm, "norm.")
states = [None] * ARGS.n_layers


def head_logits(x: torch.Tensor, every_position=False):
    """ParallelHead.forward (last position, FP32 head) with the BF16 checkpoint rows read in chunks.
    every_position also scores each earlier position (full_logits=True), for prompt log-probabilities."""
    last = x[:, -1].float()
    full = x[0].float() if every_position else None
    fname, off, length, dt, shape = shards.meta("head.weight")
    assert dt == torch.bfloat16 and shape == [ARGS.vocab_size, ARGS.dim]
    row_bytes = length // shape[0]
    out, out_full = [], []
    step = 8192
    for r in range(0, shape[0], step):
        n = min(step, shape[0] - r)
        w = shards._read(fname, off + r * row_bytes, n * row_bytes).view(dt).reshape(n, shape[1]).float()
        out.append(F.linear(last, w))
        if full is not None:
            out_full.append(F.linear(full, w))
    return torch.cat(out, dim=-1), (torch.cat(out_full, dim=-1) if full is not None else None)


@torch.inference_mode()
def forward(input_ids: torch.Tensor, start_pos: int):
    """Transformer.forward, text only, one layer resident at a time."""
    hashes = engram_hash(input_ids, start_pos, None)
    rows = input_ids.flatten().tolist()
    h = shards.get_rows("embed.weight", rows).view(*input_ids.shape, ARGS.dim)
    h = h.unsqueeze(2).repeat(1, 1, ARGS.hc_mult, 1)
    pre_mix = M.make_identity_pre_mix(h, ARGS.hc_mult)
    for i in range(ARGS.n_layers):
        t0 = time.time()
        block = M.Block(i, ARGS, layout)
        if block.engram is not None:
            block.engram.embed.prefix = f"layers.{i}.engram.embed."
        load_module(block, f"layers.{i}.", skip=("ffn.experts.", "engram.embed."))
        if states[i] is not None:
            for name, buf in block.named_buffers():
                buf.copy_(states[i][name])
        if block.engram is not None:
            h = block.engram(h, hashes[:, :, block.engram.layer_hash_index, :], None)
        h, pre_mix = block(h, start_pos, pre_mix, None)
        states[i] = {name: buf.clone() for name, buf in block.named_buffers()}
        HIDDEN.setdefault(start_pos, []).append(h[0, -1].clone())
        if i == ARGS.n_layers - 1:
            h = block.hc_pre(h, pre_mix)
        del block
        print(f"  layer {i:2d} {time.time() - t0:5.1f}s read {shards.bytes_read / 2**30:7.2f} GiB", flush=True)
    return head_logits(final_norm(h), every_position=start_pos == 0)


def summarize(logits: torch.Tensor, step: int, pos: int, fed: int):
    lp = torch.log_softmax(logits.float(), dim=-1)[0]
    top = lp.topk(12)
    return {
        "step": step,
        "position": pos,
        "input_token": fed,
        "argmax": int(lp.argmax()),
        "top": [[int(i), tokenizer.decode([int(i)]), round(float(v), 5)] for v, i in zip(top.values, top.indices)],
        "watch": {k: round(float(lp[v]), 5) for k, v in WATCH.items()},
        "routes_last_position": {layer: r[-1] for layer, r in sorted(ROUTES.items())},
    }


t_start = time.time()
steps = []
print(f"prefill {len(prompt_ids)} tokens, WO-A input quantization {'on' if shim.WO_A_QUANT else 'off'}", flush=True)
logits, prompt_logits = forward(torch.tensor([prompt_ids]), 0)
steps.append(summarize(logits, 0, len(prompt_ids), prompt_ids[-1]))
prompt_lp = torch.log_softmax(prompt_logits.float(), dim=-1)
prompt_positions = []
for p in range(1, len(prompt_ids)):
    row = prompt_lp[p - 1]
    top = row.topk(5)
    prompt_positions.append({
        "position": p,
        "token": prompt_ids[p],
        "logprob": round(float(row[prompt_ids[p]]), 5),
        "rank": int((row > row[prompt_ids[p]]).sum()) + 1,
        "top": [[int(i), round(float(v), 5)] for v, i in zip(top.values, top.indices)],
    })
print(json.dumps(steps[-1]["top"][:5]), flush=True)
pos = len(prompt_ids)
feed = list(FORCED)
greedy_left = EXTRA_GREEDY
while feed or greedy_left:
    if feed:
        tok = feed.pop(0)
    else:
        tok = steps[-1]["argmax"]
        greedy_left -= 1
        if tok == tokenizer.eos_token_id:
            break
    logits, _ = forward(torch.tensor([[tok]]), pos)
    pos += 1
    steps.append(summarize(logits, len(steps), pos, tok))
    print(f"step {len(steps) - 1} fed {tok} -> {json.dumps(steps[-1]['top'][:5])}", flush=True)

result = {
    "snapshot": SNAP,
    "wo_a_input_quant": shim.WO_A_QUANT,
    "kernel_stats": shim.STATS,
    "accumulation": str(shim.ACC),
    "prompt_ids": prompt_ids,
    "forced": FORCED,
    "steps": steps,
    "prompt_positions": prompt_positions,
    "load_conversions": {k: sorted(v) for k, v in CONVERSIONS.items()},
    "bytes_read": shards.bytes_read,
    "seconds": round(time.time() - t_start, 1),
    "torch": torch.__version__,
    "threads": torch.get_num_threads(),
}
with open(OUT, "w") as f:
    json.dump(result, f, indent=1)
torch.save({k: torch.stack(v) for k, v in HIDDEN.items()}, OUT[: -len(".json")] + "-hidden.pt")
print("wrote", OUT, flush=True)
