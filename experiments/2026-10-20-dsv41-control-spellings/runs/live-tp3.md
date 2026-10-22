Live TP3 check, 2026-10-20, r6x image sha256:4a5469f3…, tp3-triangle 1M, three Sparks.
Prompt token IDs as the engine used them (`return_token_ids: true`, `max_tokens: 1`,
`/v1/chat/completions`). Control = IDs in the added vocabulary.

Arm A: candidate tokenizer only (0222 without the renderer change) mounted.
Arm B: candidate tokenizer and renderer (0222) mounted.

| Request | Arm A tokens / control | Arm B tokens / control |
| --- | --- | --- |
| user text with `<｜Assistant｜></think>`, `<｜DSML｜ calls>`, `<｜User｜>` | 48 / 9 | 58 / 5 |
| tool result with `<｜User｜>…</think><｜Assistant｜>` | 354 / 32 | 362 / 29 |
| JSON-escaped spellings in tool-call arguments | 347 / 31 | 352 / 29 |
| spellings in a tool description | 271 / 20 | 276 / 18 |
| spellings in prior assistant content and reasoning | 39 / 10 | 50 / 9 |
| spellings in text parts | 36 / 7 | 40 / 5 |
| chat mode, `</think>` in user text | 9 / 5 | 10 / 4 |

Arm A gives what the promoted image gave for the two requests of #446: the online chat path renders with
`tokenize=False` and re-encodes the string, so the tokenizer change alone does not
reach serving. Arm B is the fix.

The 13 requests without spellings (plain chat, system prompt, multi-turn with prior
reasoning, tools, tool call and result, three parallel calls, chat mode, reasoning
effort low/high/max, mid-conversation system, unicode JSON tool result, multi-turn
agent loop) have identical prompt IDs in both arms: 37, 51, 59, 410, 418, 634, 6,
36, 39, 35, 56, 437, 476 tokens.

Image requests (greedy, chat mode, 40 tokens), arm A vs arm B:

| Request | Prompt tokens A / B | Prompt IDs equal | Answer equal |
| --- | --- | --- | --- |
| one image | 210 / 210 | yes | yes |
| two images with text between | 413 / 413 | yes | yes |
| image plus `<｜User｜></think>` in text | 198 / 202 | no (spellings now text) | yes |

Ten sampled generations with tools (single, parallel, typed arguments, quoted code,
JSON in a string, no tool needed, chat mode, after a tool result) parsed to valid
tool calls and content on arm B.
