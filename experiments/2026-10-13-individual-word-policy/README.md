# Individual words during loop protection

Owner-authorized policy-only change. Replace the three compound Let-me entries
with individual Let/let/LET and Me/me/ME, observed Cyrillic меня/Меня/МЕНЯ,
and the explicitly enumerated Latin/Cyrillic variants in tables/bad_words.json.
policy-codepoints.json identifies every character to remove visual ambiguity.
All prior Hmm, Actually (only that case), and OK entries remain unchanged.

No Greek, full-width, mathematical-letter, zero-width, or other speculative
variants are added. This is a finite token-sequence list, not a universal
substring or Unicode normalization guarantee. Individual tokens may also be
parts of longer words. The owner will assess actual behavior.

The detector, built-in vLLM bad_words matcher, 2000-word lifetime, retries,
temperature, images, kernels, capacities, and all unrelated settings are unchanged.
No additional blocking evaluation is run, per owner direction. Tokenization and
serving availability are checked for deployment correctness only.
