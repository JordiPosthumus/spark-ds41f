#!/usr/bin/env python3
"""Replay the qeval json_count request against a running server and record its rendered prompt ids,
reply and per-token log-probabilities.

    python3 replay.py http://127.0.0.1:8000 > runs/production-r6v.json

The first request is the battery's own body (temperature 0, thinking off, 300 tokens); the second
adds logprobs with token ids; the third scores every prompt position (prompt_logprobs) from the same
ids; the fourth asks the same question with thinking on. Run on the head node.
"""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = "deepseek-v4.1-flash"
PROMPT = (
    'Count the vowels (a,e,i,o,u, case-insensitive) in the string "Encyclopaedia Britannica" and '
    'return ONLY {"vowels": <integer>}.'
)
MESSAGES = [{"role": "user", "content": PROMPT}]


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


def chat(extra, thinking=False, max_tokens=300):
    body = {"model": MODEL, "temperature": 0, "max_tokens": max_tokens, "stream": False,
            "chat_template_kwargs": {"thinking": thinking}, "messages": MESSAGES}
    body.update(extra)
    return post("/v1/chat/completions", body)


tokenized = post("/tokenize", {"model": MODEL, "messages": MESSAGES, "add_generation_prompt": True,
                               "chat_template_kwargs": {"thinking": False}, "return_token_strs": True})
plain = chat({})
with_lp = chat({"logprobs": True, "top_logprobs": 8, "return_tokens_as_token_ids": True})
steps = []
for t in with_lp["choices"][0]["logprobs"]["content"]:
    steps.append({"token": int(t["token"].split(":")[1]), "logprob": t["logprob"],
                  "top": [[int(x["token"].split(":")[1]), x["logprob"]] for x in t["top_logprobs"]]})
scored = post("/v1/completions", {"model": MODEL, "prompt": tokenized["tokens"], "max_tokens": 1, "temperature": 0,
                                  "prompt_logprobs": 5})
prompt_positions = []
for p, entry in enumerate(scored["choices"][0]["prompt_logprobs"]):
    if entry is None:
        continue
    tok = tokenized["tokens"][p]
    top = sorted(((int(k), v["logprob"]) for k, v in entry.items()), key=lambda kv: -kv[1])
    prompt_positions.append({"position": p, "token": tok, "logprob": entry[str(tok)]["logprob"],
                             "rank": entry[str(tok)]["rank"],
                             "top": [[k, lp] for k, lp in top if entry[str(k)]["rank"] <= 5][:5]})
thought = chat({}, thinking=True, max_tokens=4000)
thought_msg = thought["choices"][0]["message"]
json.dump({
    "prompt_ids": tokenized["tokens"],
    "prompt_token_strs": tokenized["token_strs"],
    "content": plain["choices"][0]["message"]["content"],
    "content_with_logprobs": with_lp["choices"][0]["message"]["content"],
    "usage": plain["usage"],
    "steps": steps,
    "prompt_positions": prompt_positions,
    "thinking_on": {"content": thought_msg.get("content"),
                    "reasoning": thought_msg.get("reasoning") or thought_msg.get("reasoning_content"),
                    "usage": thought["usage"]},
}, sys.stdout, indent=1, ensure_ascii=False)
print()
