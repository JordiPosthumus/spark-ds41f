# Shared word-family alarm

Owner-approved replacement of the early phrase trigger: four hits from any one
family in the last 50 completed words enable the entire temporary native blocklist.
Families: let, me, ok/okay, actually, hm. Detection ignores case and fenced code.
Let me contributes one hit to let and one to me, not two to one family.
Spaced H m, H mm and H m m are recognized, including listed casing forms;
overlapping H m and H m m count once. Window expiry includes code/unrelated words.

`tables/word_policy.json` is the shared policy. Its explicit block entries generate
`bad_words.json` and the native token encodings. Existing Latin/Cyrillic entries
are retained. Only Actually with capital A is blocked; detection is case-insensitive.
Legacy recognition of joined H + 1–16 m letters is retained; the concrete native
blocklist is finite. It does not guarantee arbitrary spacing, Unicode lookalikes,
case permutations, or alternative token segmentation. The native vLLM matcher
itself is unchanged. No custom string/regex logits matcher is substituted.

The existing 2000-word timer, reasoning/tool boundary exit, temperature, severe
loop retry mechanics, kernels, image, capacity, and memory guards are unchanged.
Focused CPU checks cover detection and policy wiring. The owner evaluates model
behavior; passing these checks does not prove semantic loop recovery.
